#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Lightweight convolutional layers

 * CANONICAL POLYADIC FORMAT
 * TUCKER FORMAT
 * TENSOR TRAIN FORMAT
"""

from pprint import PrettyPrinter

import math
import numpy as np
import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from sympy import Symbol, solve
from torch import Tensor
from torch.nn.functional import conv3d
from models.baseline_unet import DoubleConv
from utils.utils import get_tuning_par, get_network_size
from utils.helper import log, hi, TENSOR_NET_TYPES

pp = PrettyPrinter(4)


####################
# Helper functions #
####################

def get_patches(input: torch.Tensor, kernel_size: int = 3, stride: int = 1, padding: int = 1, value: int = 0):
    """
    Take input tensor and return unfolded patch Tensor
    """

    # First add padding if requested
    if padding:
        input = F.pad(input=input, pad=(padding,) * 6, mode='constant', value=value)

    # Unfold across three dimensions to get patches
    patches = input.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)

    return patches


def count_lr_conv3d(module, _, y):
    """
    Flop count hook for ptflops, to be used with LowRankConv3D
    """

    # All output elements
    output_voxels = y.nelement()

    # Output voxels (per channel)
    output_voxels_per_channel = np.prod(y.shape[2:])

    # Kernel flops, given by path_info (see LowRankConv3D)
    kernel_flops = module.kernel_flops

    # We add bias to each voxel in all output channels
    bias_flops = output_voxels if module.bias is not None else 0

    # We're calculating macs, not flops (hence the /2). Output channels
    # are included in the path_cost, hence we multiply by output_voxels
    # rather than output_elements
    total_ops = output_voxels_per_channel * kernel_flops / 2 + bias_flops

    # print(f'{module.__name__}: kernel ops {kernel_flops} - bias ops {bias_flops} \t TOTAL {total_ops}')

    module.__flops__ += int(total_ops)


###################################
# Low-rank 3D convolutional layer #
###################################

class AirConv3D(nn.Module):

    def __init__(self,
                 compression: int,
                 tensor_net_type: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = True,
                 comp_friendly: bool = False
                 ):

        super().__init__()

        # Initializing attributes
        self.compression = compression
        self.tensor_net_type = tensor_net_type

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.comp_friendly = comp_friendly

        assert self.tensor_net_type in TENSOR_NET_TYPES, f"Choose a valid tensor network [{TENSOR_NET_TYPES}]"

        self.__name__ = f'{self.tensor_net_type.lower()}_low_rank_conv'

        ###########################
        # Building tensor network #
        ###########################

        # Create tensor network
        self.nodes, self.output_edges = self._make_tensor_network()

        # Register and initialize parameters
        self._register_and_init()

        # Add bias
        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_channels))
        else:
            self.register_parameter("bias", None)

        # Get contraction with optimal path
        self.einsum_expression, self.path_info = self._get_contraction()

        # Save optimized flops for tensor network
        self.kernel_flops = self.path_info.opt_cost

        # Calculate actual compression rate
        self.max_params = (self.kernel_size ** 3 * self.in_channels * self.out_channels)
        self.kernel_params = self._get_tensor_network_size()
        self.actual_compression = self.max_params / self.kernel_params

    def _make_tensor_network(self):
        """
        Create a tensor network as requested, given the type:
        * Canonical polyadic decomposition format
        * Tensor train format
        * Tucker format
        """

        # Store tensor network nodes here
        nodes = {}

        # We need unfolded input (but not when we go for the computationally friendly case)
        if not self.comp_friendly:
            nodes['input'] = {
                "tensor": None,
                "shape": (1, self.in_channels,  # <-- (batch size, input channels,
                          1, 1, 1,  # <--  image height, width, depth,
                          self.kernel_size, self.kernel_size, self.kernel_size),  # <--  kernel height, width, depth)
                "legs": ['-b',
                         'c_in',
                         '-h', '-w', '-d',
                         'k_h', 'k_w', 'k_d']
            }

        ##################################################
        # CANONICAL POLYADIC TENSOR DECOMPOSITION FORMAT #
        ##################################################

        if self.tensor_net_type in ['cp', 'canonical', 'cpd']:

            log(f'Creating CPD layer [compression rate = {self.compression}].', verbosity=3, color='magenta')

            # First, obtain rank based on compression rate
            self.rank = self._get_tuning_par()

            """
            For the CPD format, we need 5 factor matrices: 
             * 1 for each of the kernel dimensions: U_k_h, U_k_w, U_kd,
             * 1 for the input channels, 1 for the output channels: U_c_in and U_c_out
             
             c_in - O - 
                        \
              k_h - O - -\
                           - O - c_out
              k_w - O -- /
                        /
              k_d - O -
            """

            # First kernel factor matrices (U_kh, U_kd, U_kw)
            kernel_dimensions = ['k_h', 'k_w', 'k_d'] if not self.comp_friendly else ['-k_h', '-k_w', '-k_d']
            for name in kernel_dimensions:
                nodes[f'U_{name}'] = {
                    "tensor": Tensor(self.kernel_size, self.rank),
                    "shape": (self.kernel_size, self.rank),
                    "legs": [name, 'r']
                }

            # Now factor matrices for input and output channels
            nodes['U_c_in'] = {
                "tensor": Tensor(self.in_channels, self.rank),
                "shape": (self.in_channels, self.rank),
                "legs": ['c_in', 'r'] if not self.comp_friendly else ['-c_in','r']
            }
            nodes['U_c_out'] = {
                "tensor": Tensor(self.rank, self.out_channels),
                "shape": (self.rank, self.out_channels),
                "legs": ['r', '-c_out']  # <-- output channels becomes dangling edge after contraction
            }

        #################
        # TUCKER FORMAT #
        #################

        elif self.tensor_net_type in ['tucker']:

            log(f'Creating Tucker layer [compression rate = {self.compression}].', verbosity=3,
                color='magenta')

            """
            For the Tucker format, we need 3 nodes: 2 factor matrices and a core tensor: 
             * 1 factor matrix for the input channels, 1 for the output channels: U_c_in and U_c_out
             * 1 core tensor: G


                        k_h   k_d
                          \   /
              c_in - O ---  O --- O - c_out
                            |            
                           k_w   
            """

            # Get tuning parameter S
            self.S = self._get_tuning_par()

            # Get r1 and r5
            self.r1 = round(self.in_channels / self.S) if round(self.in_channels / self.S) > 0 else 1
            self.r2 = round(self.out_channels / self.S) if round(self.out_channels / self.S) > 0 else 1

            # Let's start with the core tensor
            nodes['G'] = { # TODO FIX THIS MESS
                'tensor': Tensor(self.r1, self.kernel_size, self.kernel_size, self.kernel_size, self.r2),
                'shape': (self.r1, self.kernel_size, self.kernel_size, self.kernel_size, self.r2),
                'legs': ['r1', 'k_h','k_w','k_d','r2'] if not self.comp_friendly else ['r1', '-k_h','-k_w','-k_d','r2']
            }

            # Finally, add factor matrices for input and output channels
            nodes['U_c_in'] = {
                "tensor": Tensor(self.in_channels, self.r1),
                "shape": (self.in_channels, self.r1),
                "legs": ['c_in', 'r1'] if not self.comp_friendly else ['-c_in','r1']
            }
            nodes['U_c_out'] = {
                "tensor": Tensor(self.r2, self.out_channels),
                "shape": (self.r2, self.out_channels),
                "legs": ['r2', '-c_out']  # <-- output channels becomes dangling edge after contraction
            }

        #######################
        # TENSOR TRAIN FORMAT #
        #######################

        elif self.tensor_net_type in ['train', 'tensor-train', 'tt']:

            log(f'Creating Tensor Train layer {("(alt version)" if self.tensor_net_type.endswith("2") else "")}'
                f' [compression rate = {self.compression}].', verbosity=3, color='magenta')

            """
            For the TT format, we need 5 nodes:
             * 1 3rd-order tensor node for each of the kernel dimensions: U_k_h, U_k_w, U_kd,
             * 1 factor matrix for the input channels, 1 for the output channels: U_c_in and U_c_out

              O - r1 - O - r2 - O - r3 - O - r4 - O
              |        |        |        |        |
             c_in     k_h      k_w      k_d      c_out
             """

            # We tune the bond dimensions (assumed equal across the 'train')
            self.r = round(self._get_tuning_par()) if round(self._get_tuning_par()) > 0 else 1

            # Set bond dimensions, depending on type
            if self.tensor_net_type.endswith("2"):
                self.r1 = self.r4 = self.r
                self.r2 = self.r3 = 3
            else:
                self.r1 = self.r2 = self.r3 = self.r4 = self.r

            # First kernel factor matrices (U_kh, U_kd, U_kw)
            nodes['U_k_h'] = {
                'tensor': Tensor(self.r1, self.kernel_size, self.r2),
                'shape': (self.r1, self.kernel_size, self.r2),
                'legs': ['r1', 'k_h', 'r2'] if not self.comp_friendly else ['r1', '-k_h', 'r2']
            }
            nodes['U_k_w'] = {
                'tensor': Tensor(self.r2, self.kernel_size, self.r3),
                'shape': (self.r2, self.kernel_size, self.r3),
                'legs': ['r2', 'k_w', 'r3']if not self.comp_friendly else ['r2', '-k_w', 'r3']
            }
            nodes['U_k_d'] = {
                'tensor': Tensor(self.r3, self.kernel_size, self.r4),
                'shape': (self.r3, self.kernel_size, self.r4),
                'legs': ['r3', 'k_d', 'r4'] if not self.comp_friendly else ['r3', '-k_d', 'r4']
            }

            # Now factor matrices for input and output channels
            nodes['U_c_in'] = {
                "tensor": Tensor(self.in_channels, self.r1),
                "shape": (self.in_channels, self.r1),
                "legs": ['c_in', 'r1'] if not self.comp_friendly else ['-c_in','r1']
            }
            nodes['U_c_out'] = {
                "tensor": Tensor(self.r4, self.out_channels),
                "shape": (self.r4, self.out_channels),
                "legs": ['r4', '-c_out']  # <-- output channels becomes dangling edge after contraction
            }

        # Add output edges
        if self.comp_friendly:
            output_edges = ['-c_out', '-c_in', '-k_h', '-k_w', '-k_d']
        else:
            output_edges = ['-b', '-c_out', '-h', '-w', '-d']

        return nodes, output_edges

    def _get_contraction(self):
        """ Get optimal contraction expression for our tensor network"""

        if not self.comp_friendly:
            args = [self.nodes['input']['shape'], self.nodes['input']['legs']]
        else:
            args = []
        # Go over input node, factor matrices
        for node_name, node_params in self.nodes.items():
            if node_name != 'input':
                args.append(node_params['shape'])
                args.append(node_params['legs'])

        # Add dangling output legs
        args.append(self.output_edges)

        # Get contraction expression (returns einsum path and printable object containing info about path found)
        _, path_info = oe.contract_path(*args, shapes=True, optimize='optimal')

        '''print('path_info_eq', path_info.eq)
        print('path_info_shapes', *path_info.shapes)
        print('path_info_path', path_info.path)'''

        # Generate reusable expression for a given contraction with specific shapes
        expr = oe.contract_expression(path_info.eq, *path_info.shapes, optimize=path_info.path)

        return expr, path_info

    def _register_and_init(self):
        """ Register and initialize parameters"""

        # Go over all weight nodes
        for k, v in self.nodes.items():
            if k != "input":
                # Register the weight (all factor matrices) as parameters that need to be included
                # in the backpropagation
                self.register_parameter(
                    k, nn.Parameter(v['tensor'], requires_grad=True)
                )
                self.nodes[k]["tensor"] = getattr(self, k)

                # Initialize values
                nn.init.kaiming_uniform_(v['tensor'], a=math.sqrt(5))

    def _get_tuning_par(self) -> int:
        """
        Given a compression rate, return the appropriate tuning parameter,
        depending on the tensor network that is used for the low-rank convolution
        """

        return get_tuning_par(
            compression=self.compression,
            tensor_net_type=self.tensor_net_type,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size
        )


    def _get_tensor_network_size(self) -> float:
        """
        Get the number of parameters involved in the kernel decomposition/tensor network,
        so we can calculate the actual compression rate
        """

        return get_network_size(
            tuning_param=self._get_tuning_par(),
            tensor_net_type=self.tensor_net_type,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size
        )

    def forward(self, input: Tensor):
        """
        Forward function has two modes:
        * MODE 1: Computationally friendly: use conv3D and don't include input in tensor network
        * MODE 2: Computationally harsh: contract full tensor network with unfolded input included
        """

        # MODE 1
        if self.comp_friendly:

            # Get weights, which should be attributes of the layer (if registered correctly)
            weights = [getattr(self, k) for k in self.nodes.keys()]

            # Contract weight tensor
            kernel_tensor = self.einsum_expression(*weights)

            # Use pytorch's optimized conv3D function (includes bias!)
            output = conv3d(input=input, weight=kernel_tensor, bias=self.bias, stride=self.stride, padding=self.padding)

        # MODE 2
        else:

            # First, get patches (unfold the input tensor)
            patches = get_patches(input=input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

            # Now get weights, which should be attributes of the layer (if registered correctly)
            # Obviously don't count the input tensor this time!
            weights = [getattr(self, k) for k in self.nodes.keys() if k != 'input']

            # Contract
            output = self.einsum_expression(patches, *weights)

            # Add bias manually
            if self.bias is not None:
                output += self.bias[None, :, None, None, None]  # <-- cast across remaining dimensions

            output = output.contiguous()

        return output


# Double convolution block for low rank layers
class AirDoubleConv(nn.Module):

    def __init__(
            self,
            compression: int,
            tensor_net_type: str,
            in_channels: int,
            out_channels: int,
            num_groups=8,
            strides=(2, 1),
            activation=nn.LeakyReLU(inplace=True),
            double_conv_par=None,
            comp_friendly:bool = True,
            __name__='low_rank_double_conv',
    ):
        super().__init__()
        self.__name__ = __name__
        self.comp_friendly = comp_friendly

        # Initialize convolution parameters
        double_conv_par = double_conv_par if double_conv_par else {}

        # Set parameters (if not given!)
        double_conv_par.setdefault('kernel_size', 3)
        double_conv_par.setdefault('padding', 1)
        double_conv_par.setdefault('comp_friendly', self.comp_friendly)

        # Define inner block architecture
        self.block = nn.Sequential(

            # Lightweight convolutional layer
            AirConv3D(
                compression=compression,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=strides[0],
                tensor_net_type=tensor_net_type,
                #comp_friendly=comp_friendly,
                **double_conv_par
            ),

            # Normalization layer (default minibatch of 8 instances)
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),

            # Activation layer
            activation,

            # Lightweight convolutional layer
            AirConv3D(
                compression=compression,
                in_channels=out_channels,
                out_channels=out_channels,
                stride=strides[1],
                tensor_net_type=tensor_net_type,
                #comp_friendly=comp_friendly,
                **double_conv_par
            ),

            # Normalization layer (default minibatch of 8 instances)
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),

            # Activation layer
            activation
        )

    # Forward function (backward propagation is added automatically)
    def forward(self, input):
        return self.block(input)


if __name__ == '__main__':
    # Console parameters
    # set_params(verbosity=3, timestamped=False)
    hi('Lightweight layer sanity check', verbosity=3, timestamped=True)

    # Quick test (currently no cuda support on my end)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Sanity check: do dimensions make sense? Let's 'benchmark' a classic Conv3D layer
    batch_size = 1
    in_channels = 4
    out_channels = 32
    kernel_dim = 3
    stride = 1
    dim = 128

    # Test image
    image = torch.rand(1, in_channels, dim, dim, dim)
    image = image.to(device)

    # Classic convolutional layer
    layer_classic = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim, padding=1)
    layer_classic.to(device)

    # Low-rank layers
    compression = 150

    # Computational demand
    comp_friendly=False

    # Canonical layer
    layer_canon = AirConv3D(in_channels=in_channels, out_channels=out_channels,
                            compression=compression,
                            kernel_size=kernel_dim, padding=1, tensor_net_type='cpd',
                            comp_friendly=comp_friendly)
    layer_canon.to(device)

    # Tucker layer
    layer_tucker = AirConv3D(in_channels=in_channels, out_channels=out_channels,
                              compression=compression,
                              kernel_size=kernel_dim, padding=1, tensor_net_type='tucker',
                             comp_friendly=comp_friendly)
    layer_tucker.to(device)

    # TT layer
    layer_tt = AirConv3D(in_channels=in_channels, out_channels=out_channels,
                         compression=compression,
                         kernel_size=kernel_dim, padding=1, tensor_net_type='train',
                         comp_friendly=comp_friendly)
    layer_tt.to(device)

    # Sample output
    classic_output = layer_classic(image)
    canon_output = layer_canon(image)
    tucker_output = layer_tucker(image)
    tt_output = layer_tt(image)
    assert canon_output.size() == classic_output.size(), "Something went wrong with CPD format, output shapes don't match!"
    assert tucker_output.size() == classic_output.size(), "Something went wrong with Tucker format, output shapes don't match!"
    assert tt_output.size() == classic_output.size(), "Something went wrong with TT format, output shapes don't match!"

    # Double conv test
    double_conv_classic = DoubleConv(in_channels=in_channels, out_channels=out_channels)
    double_conv_classic_output = double_conv_classic(image)
    double_conv_cpd = AirDoubleConv(compression=compression, tensor_net_type='cpd', in_channels=in_channels,
                                    out_channels=out_channels, num_groups=8)
    double_conv_cpd_output = double_conv_cpd(image)
    """
    assert double_conv_cpd_output.size() == double_conv_classic_output.size(), "Something went wrong with double conv CPD, output shapes don't match!"
    """



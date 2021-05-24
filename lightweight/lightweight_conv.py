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

TODO: Add compression rate! But first, figure out how to handle the different ranks in TT and Tucker --> all the same value?

"""

import math
from pprint import PrettyPrinter

import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.helper import set_params
from utils.helper import log, hi

pp = PrettyPrinter(4)

from models.unet import DoubleConv


####################
# Helper functions #
####################

def get_patches(input: torch.Tensor, kernel_dim: int = 3, stride: int = 1, padding: int = 1, value: int = 0):
    """
    Take input tensor and return unfolded patch Tensor
    """

    # First add padding if requested
    if padding:
        input = F.pad(input=input, pad=(padding,) * 6, mode='constant', value=value)

    # Unfold across three dimensions to get patches
    patches = input.unfold(2, kernel_dim, stride).unfold(3, kernel_dim, stride).unfold(4, kernel_dim, stride)

    return patches


##########################################
# Canonical polyadic decomposition layer #
##########################################

class LowRankConv3D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 tensor_net_type: str,
                 tensor_net_args: dict,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = True,
                 ):

        super().__init__()

        # Initializing attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.tensor_net_type = tensor_net_type
        self.tensor_net_args = tensor_net_args

        types = ['cpd', 'canonical', 'tucker', 'train', 'tensor-train', 'tt']
        assert self.tensor_net_type in types, f"Choose a valid tensor network {types}"

        self.__name__ = f'{self.tensor_net_type.upper()}_low_rank_conv'

        assert 'rank' in self.tensor_net_args, f"Please pass the `rank`."
        self.rank = self.tensor_net_args['rank']

        ###########################
        # Building tensor network #
        ###########################

        # Create tensor network
        self.nodes, self.output_edges = self.make_tensor_network()

        # Register and initialize parameters
        self.register_and_init()

        # Add bias
        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_channels))
        else:
            self.register_parameter("bias", None)

        # Get contraction with optimal path
        self.einsum_expression, self.path_info = self.get_contraction()

        # Save of flops (but this can also be done using ptflops, so watch out here to make a fair comparison)
        self.flops = self.path_info.opt_cost

    def make_tensor_network(self):
        """
        Create a tensor network as requested, given the type:
        * Canonical polyadic decomposition format
        * Tensor train format
        * Tucker format
        """

        # Store tensor network nodes here
        nodes = {}

        # We always need unfolded input
        nodes['input'] = {
            "tensor": None,
            "shape": (1, self.in_channels,  # <-- (batch size, input channels,
                      1, 1, 1,  # <--  image height, width, depth,
                      self.kernel_size, self.kernel_size, self.kernel_size),  # <--  kernel height, widht, depth)
            "legs": ['-b',
                     'c_in',
                     '-h', '-w', '-d',
                     'k_h', 'k_w', 'k_d']
        }

        ##################################################
        # CANONICAL POLYADIC TENSOR DECOMPOSITION FORMAT #
        ##################################################

        if self.tensor_net_type in ['canonical', 'cpd']:

            log(f'Creating CPD tensor network [rank = {self.rank}].', verbosity=1, color='magenta')

            assert isinstance(self.rank, int), 'For CPD, pass an integer for the rank.'

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
            for name in ['k_h', 'k_w', 'k_d']:
                nodes[f'U_{name}'] = {
                    "tensor": Tensor(self.kernel_size, self.rank),
                    "shape": (self.kernel_size, self.rank),
                    "legs": [name, 'r']
                }

            # Now factor matrices for input and output channels
            nodes['U_c_in'] = {
                "tensor": Tensor(self.in_channels, self.rank),
                "shape": (self.in_channels, self.rank),
                "legs": ['c_in', 'r']
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

            log(f'Creating Tucker tensor network [rank(s) = {self.rank}].', verbosity=1, color='magenta')

            assert len(self.rank) == 5, 'For Tucker format, pass 5 rank integers.'

            self.r1, self.r2, self.r3, self.r4, self.r5 = self.tensor_net_args['rank']

            """
            For the CPD format, we need 6 nodes: 5 factor matrices and a core tensor: 
             * 1 factor matrix for each of the kernel dimensions: U_k_h, U_k_w, U_kd,
             * 1 factor matrix for the input channels, 1 for the output channels: U_c_in and U_c_out
             * 1 core tensor: G
             
             
             c_in - O ---
                          \
              k_h - O ---  \
                          \ |
                            O - O - c_out
                          / |           
              k_w - O ---  /
                          /       (middle node = core tensor G)
              k_d - O ---
            """

            # Let's start with the core tensor
            nodes['G'] = {
                'tensor': Tensor(self.r1, self.r2, self.r3, self.r4, self.r5),
                'shape': (self.r1, self.r2, self.r3, self.r4, self.r5),
                'legs': ['r1', 'r2', 'r3', 'r4', 'r5']
            }

            # Now add kernel factor matrices (U_kh, U_kd, U_kw)
            for kernel_dim, rank_leg, rank in [('k_h', 'r2', self.r2),
                                               ('k_w', 'r3', self.r3),
                                               ('k_d', 'r4', self.r4)]:
                nodes[f'U_{kernel_dim}'] = {
                    "tensor": Tensor(self.kernel_size, rank),
                    "shape": (self.kernel_size, rank),
                    "legs": [kernel_dim, rank_leg]
                }

            # Finally, add factor matrices for input and output channels
            nodes['U_c_in'] = {
                "tensor": Tensor(self.in_channels, self.r1),
                "shape": (self.in_channels, self.r1),
                "legs": ['c_in', 'r1']
            }
            nodes['U_c_out'] = {
                "tensor": Tensor(self.r5, self.out_channels),
                "shape": (self.r5, self.out_channels),
                "legs": ['r5', '-c_out']  # <-- output channels becomes dangling edge after contraction
            }

        #######################
        # TENSOR TRAIN FORMAT #
        #######################

        elif self.tensor_net_type in ['train', 'tensor-train', 'tt']:

            log(f'Creating Tensor Train network [rank(s) = {self.rank}].', verbosity=1, color='magenta')

            assert len(self.rank) == 4, 'For tensor train format, pass 5 rank integers.'

            self.r1, self.r2, self.r3, self.r4 = self.tensor_net_args['rank']

            """
            For the TT format, we need 5 nodes:
             * 1 3rd-order tensor node for each of the kernel dimensions: U_k_h, U_k_w, U_kd,
             * 1 factor matrix for the input channels, 1 for the output channels: U_c_in and U_c_out
             
              O - r1 - O - r2 - O - r3 - O - r4 - O
              |        |        |        |        |
             c_in     k_h      k_w      k_d      c_out
            """

            # First kernel factor matrices (U_kh, U_kd, U_kw)
            nodes['U_k_h'] = {
                'tensor': Tensor(self.r1, self.kernel_size, self.r2),
                'shape': (self.r1, self.kernel_size, self.r2),
                'legs': ['r1', 'k_h', 'r2']
            }
            nodes['U_k_w'] = {
                'tensor': Tensor(self.r2, self.kernel_size, self.r3),
                'shape': (self.r2, self.kernel_size, self.r3),
                'legs': ['r2', 'k_w', 'r3']
            }
            nodes['U_k_d'] = {
                'tensor': Tensor(self.r3, self.kernel_size, self.r4),
                'shape': (self.r3, self.kernel_size, self.r4),
                'legs': ['r3', 'k_d', 'r4']
            }

            # Now factor matrices for input and output channels
            nodes['U_c_in'] = {
                "tensor": Tensor(self.in_channels, self.r1),
                "shape": (self.in_channels, self.r1),
                "legs": ['c_in', 'r1']
            }
            nodes['U_c_out'] = {
                "tensor": Tensor(self.r4, self.out_channels),
                "shape": (self.r4, self.out_channels),
                "legs": ['r4', '-c_out']  # <-- output channels becomes dangling edge after contraction
            }

        # Add output edges
        output_edges = ['-b', '-c_out', '-h', '-w', '-d']
        # output_edges = [leg for node in nodes for legs in node['legs'] for leg in legs if leg.startswith('-')]

        '''print('NODES')
        pp.pprint(nodes)'''

        return nodes, output_edges

    def get_contraction(self):
        """ Get optimal contraction expression for our tensor network"""

        args = [self.nodes['input']['shape'], self.nodes['input']['legs']]
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

    def register_and_init(self):
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

    def forward(self, input: Tensor):

        # First, get patches
        patches = get_patches(input=input, kernel_dim=self.kernel_size, stride=self.stride, padding=self.padding)

        # Now get weights, which should be attributes of the layer (if registered correctly)
        # Obviously don't count the input tensor
        weights = [getattr(self, k) for k in self.nodes.keys() if k != 'input']

        # Contract
        output = self.einsum_expression(patches, *weights)

        # Add bias
        if self.bias is not None:
            output += self.bias[None, :, None, None, None]  # <-- cast across remaining dimensions

        return output


# Double convolution block for low rank layers
class LowRankDoubleConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_groups=8,
            strides=(2, 1),
            activation=nn.LeakyReLU(inplace=True),
            tensor_net_type='cpd',
            tensor_net_args={'rank': 23},
            conv_par=None,
            __name__='low_rank_double_conv',
    ):
        super().__init__()
        self.__name__ = __name__

        # Initialize convolution parameters
        conv_par = conv_par if conv_par else {}

        # Set parameters (if not given!)
        conv_par.setdefault('kernel_size', 3)
        conv_par.setdefault('padding', 1)

        # Define inner block architecture
        self.block = nn.Sequential(

            # Lightweight convolutional layer
            LowRankConv3D(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=strides[0],
                tensor_net_type=tensor_net_type,
                tensor_net_args=tensor_net_args,
                **conv_par
            ),

            # Normalization layer (default minibatch of 8 instances)
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),

            # Activation layer
            activation,

            # Lightweight convolutional layer
            LowRankConv3D(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=strides[1],
                tensor_net_type=tensor_net_type,
                tensor_net_args=tensor_net_args,
                **conv_par
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
    dim = 16

    # Test image
    image = torch.rand(1, in_channels, dim, dim, dim)
    image = image.to(device)

    # Classic convolutional layer
    layer_classic = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim, padding=1)
    layer_classic.to(device)

    # Canonical layer
    layer_canon = LowRankConv3D(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_dim, padding=1, tensor_net_type='cpd', tensor_net_args={'rank': 23})
    layer_classic.to(device)

    # Tucker layer
    layer_tucker = LowRankConv3D(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_dim, padding=1, tensor_net_type='tucker',
                                 tensor_net_args={'rank': (23,) * 5})
    layer_tucker.to(device)

    # TT layer
    layer_tt = LowRankConv3D(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_dim, padding=1, tensor_net_type='train',
                             tensor_net_args={'rank': (23,) * 4})
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
    double_conv_classic = DoubleConv(in_channels=in_channels,out_channels=out_channels)
    double_conv_classic_output = double_conv_classic(image)
    double_conv_cpd = LowRankDoubleConv(in_channels=in_channels, out_channels=out_channels, num_groups=8, )
    double_conv_cpd_output = double_conv_cpd(image)

    assert double_conv_cpd_output.size() == double_conv_classic_output.size(), "Something went wrong with double conv CPD, output shapes don't match!"

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

import math
from pprint import PrettyPrinter

import numpy as np
import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from sympy import Symbol, solve
from torch import Tensor

from models.baseline_unet import DoubleConv
from utils.helper import log, hi, TENSOR_NET_TYPES

pp = PrettyPrinter(4)


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

        if self.tensor_net_type in ['cp', 'canonical', 'cpd']:

            log(f'Creating CPD tensor network [compression rate = {self.compression}].', verbosity=3, color='magenta')

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

            log(f'Creating Tucker tensor network [compression rate = {self.compression}].', verbosity=3,
                color='magenta')

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

            # We set shape of G (core tensor) to (r1, r2, r3, r4, r5) = (in_channels/S, 3, 3, 3, out_channels/s)
            self.r2 = self.r3 = self.r4 = 3

            # Get tuning parameter S
            self.S = self._get_tuning_par()

            # Get r1 and r5
            self.r1 = round(self.in_channels / self.S) if round(self.in_channels / self.S) > 0 else 1
            self.r5 = round(self.out_channels / self.S) if round(self.out_channels / self.S) > 0 else 1

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

        ###################
        # TUCKER 2 FORMAT #
        ###################

        elif self.tensor_net_type in ['tucker2']:

            log(f'Creating Tucker (version 2) tensor network [compression rate = {self.compression}].', verbosity=3,
                color='magenta')

            """
            For the CPD format, we need 6 nodes: 5 factor matrices and a core tensor: 
             * 1 factor matrix for each of the kernel dimensions: U_k_h, U_k_w, U_kd,
             * 1 factor matrix for the input channels, 1 for the output channels: U_c_in and U_c_out
             * 1 core tensor: G



             c_in - O - O - O - c_out
                     (middle node = core tensor G)
            """

            # We set shape of G (core tensor) to (r1, r2, r3, r4, r5) = (in_channels/S, 3, 3, 3, out_channels/s)
            self.r2 = self.r3 = self.r4 = 3

            # Get tuning parameter S
            self.S = self._get_tuning_par()

            # Get r1 and r5
            self.r1 = round(self.in_channels / self.S) if round(self.in_channels / self.S) > 0 else 1
            self.r5 = round(self.out_channels / self.S) if round(self.out_channels / self.S) > 0 else 1

            # Let's start with the core tensor
            nodes['G'] = {
                'tensor': Tensor(self.r1, self.r2, self.r3, self.r4, self.r5),
                'shape': (self.r1, self.r2, self.r3, self.r4, self.r5),
                'legs': ['r1', 'k_h', 'k_w', 'k_d', 'r5']
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

        elif self.tensor_net_type in ['train', 'tensor-train', 'tt',
                                      'train2', 'tensor-train2', 'tt2']:

            log(f'Creating Tensor Train network {("(alt version)" if self.tensor_net_type.endswith("2") else "")}'
                f' [compression rate = {self.compression}].', verbosity=3, color='magenta')

            """
            For the TT format, we need 5 nodes:
             * 1 3rd-order tensor node for each of the kernel dimensions: U_k_h, U_k_w, U_kd,
             * 1 factor matrix for the input channels, 1 for the output channels: U_c_in and U_c_out

              O - r1 - O - r2 - O - r3 - O - r4 - O
              |        |        |        |        |
             c_in     k_h      k_w      k_d      c_out
             
            Special case (tt2): middle bond dimensions (r2 and r3 set to 3)
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

        '''
        #########################
        # TENSOR TRAIN FORMAT 2 #
        #########################

        elif self.tensor_net_type in ['train2', 'tensor-train2', 'tt2']:

            log(f'Creating Tensor Train network [compression rate = {self.compression}].', verbosity=3, color='magenta')

            """
            The TT2 format carbon copies the TT format, with one difference: the middle two bond dimensions are set to 3.
            """

            # We tune the bond dimensions (assumed equal across the 'train')
            self.r = self.r1 = self.r4 = round(self._get_tuning_par()) if round(self._get_tuning_par()) > 0 else 1

            # First kernel factor matrices (U_kh, U_kd, U_kw)
            nodes['U_k_h'] = {
                'tensor': Tensor(self.r1, self.kernel_size, 3),
                'shape': (self.r1, self.kernel_size, 3),
                'legs': ['r1', 'k_h', 'r2']
            }
            nodes['U_k_w'] = {
                'tensor': Tensor(3, self.kernel_size, 3),
                'shape': (3, self.kernel_size, 3),
                'legs': ['r2', 'k_w', 'r3']
            }
            nodes['U_k_d'] = {
                'tensor': Tensor(3, self.kernel_size, self.r4),
                'shape': (3, self.kernel_size, self.r4),
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
            }'''

        # Add output edges
        output_edges = ['-b', '-c_out', '-h', '-w', '-d']

        return nodes, output_edges

    def _get_contraction(self):
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

        # Number of parameters for a full-rank convolution
        max_params = self.in_channels * self.out_channels * self.kernel_size ** 3

        #######
        # CPD #
        #######

        if self.tensor_net_type in ['cp', 'cpd', 'canonical']:
            ''''
            cp_params:
            r * (in_channels + out_channels + 3 * kernel_size)

            compression = max_params / cpd_params
            '''

            rank = max_params / (self.compression * (self.in_channels + self.out_channels + 3 * self.kernel_size))
            return round(rank) if round(rank) > 0 else 1

        ##########
        # TUCKER #
        ##########

        elif self.tensor_net_type in ['tucker']:

            '''
            tucker params:
            ((C_in/S) * 3 * 3 * 3 * (C_out/S)) + 3 * (3*3)             + (input_channels**2/S) + (output_channels**2/S)
            = core tensor shape                + 3 * kernel node shape + input node shape      + output node shape 

            compression = max_params / tucker_params        
            '''

            S = Symbol('S', real=True)
            solutions = solve((max_params / (
                    (3 ** 3 * (self.in_channels / S) * (self.out_channels / S)) + 3 ** 3 + (
                    (self.in_channels ** 2) / S) + (
                            (self.out_channels ** 2) / S))) - self.compression, S)

            # Check for appropriate S values
            evaluated_solutions = []
            for s in solutions:
                evaluated = s.evalf()
                if evaluated > 0:
                    evaluated_solutions.append(evaluated)

            # Check if unique positive, real solution
            assert len(evaluated_solutions) == 1, 'Too many solutions!'

            return evaluated_solutions[0]

        ##########
        # TUCKER2 #
        ##########

        elif self.tensor_net_type in ['tucker2']:

            '''
            tucker params:
            ((C_in/S) * 3 * 3 * 3 * (C_out/S)) + (input_channels**2/S) + (output_channels**2/S)
            = core tensor shape                + input node shape      + output node shape 

            compression = max_params / tucker_params        
            '''

            S = Symbol('S', real=True)
            solutions = solve((max_params / (
                    (3 ** 3 * (self.in_channels / S) * (self.out_channels / S)) + (
                    (self.in_channels ** 2) / S) + (
                            (self.out_channels ** 2) / S))) - self.compression, S)

            # Check for appropriate S values
            evaluated_solutions = []
            for s in solutions:
                evaluated = s.evalf()
                if evaluated > 0:
                    evaluated_solutions.append(evaluated)

            # Check if unique positive, real solution
            assert len(evaluated_solutions) == 1, 'Too many solutions!'

            return evaluated_solutions[0]

        ################
        # TENSOR TRAIN #
        ################

        elif self.tensor_net_type in ['tt', 'tensor-train', 'train']:

            '''
            tensor train params:
            r * (in_channels + r*(3+3+3) + out_channels)

            compression = max_params / tt_params     
            '''

            r = Symbol('r', real=True)
            solutions = solve(max_params /
                              (r * (self.in_channels + 3 * (3 * r) + self.out_channels)) - self.compression, r)

            # Check for appropriate S values
            evaluated_solutions = []
            for s in solutions:
                evaluated = s.evalf()
                if evaluated > 0:
                    evaluated_solutions.append(evaluated)

            # Check if unique positive, real solution
            assert len(evaluated_solutions) == 1, 'Too many solutions!'

            return evaluated_solutions[0]

        ##################
        # TENSOR TRAIN 2 #
        ##################
        elif self.tensor_net_type in ['tt2', 'tensor-train2', 'train2']:
            '''
            tensor train params:
            r * in_channels + r*3*3 + 3*3*3 + r*3*3 + r * out_channels

            compression = max_params / tt_params     
            '''
            r = Symbol('r', real=True, positive=True)
            expr =  r * self.in_channels + r * 3 * 3 + 3 * 3 * 3 + r * 3 * 3 + r * self.out_channels
            solutions = solve(max_params /expr - self.compression,r)

            # Check for appropriate S values
            evaluated_solutions = []
            for s in solutions:
                evaluated_solutions.append(s.evalf())

            # Check if unique positive, real solution
            assert len(evaluated_solutions) == 1, 'Too many solutions!'

            return evaluated_solutions[0]

    def _get_tensor_network_size(self) -> float:
        """
        Get the number of parameters involved in the kernel decomposition/tensor network,
        so we can calculate the actual compression rate
        """
        #######
        # CPD #
        #######

        if self.tensor_net_type in ['cp', 'cpd', 'canonical']:
            ''''
            cpd_params:
            r * (in_channels + out_channels + 3 * kernel_size)

            compression = max_params / cpd_params
            '''

            return self.rank * (self.in_channels + self.out_channels + 3 * self.kernel_size)

        ##########
        # TUCKER #
        ##########

        if self.tensor_net_type in ['tucker']:
            '''
            tucker params:
            ((C_in/S) * 3 * 3 * 3 * (C_out/S)) + 3 * (3*3)             + (input_channels**2/S) + (output_channels**2/S)
            = core tensor shape                + 3 * kernel node shape + input node shape      + output node shape 

            '''
            in_param = round(self.in_channels / self.S)
            out_param = round(self.out_channels / self.S)
            return (in_param * self.kernel_size ** 3 * out_param) + \
                   3 * self.kernel_size ** 2 + \
                   (self.in_channels * in_param) + \
                   (self.out_channels * out_param)

        ############
        # TUCKER 2 #
        ############

        if self.tensor_net_type in ['tucker2']:
            '''
            tucker params:
            ((C_in/S) * 3 * 3 * 3 * (C_out/S)) + (input_channels**2/S) + (output_channels**2/S)
            = core tensor shape                + input node shape      + output node shape 

            '''
            in_param = round(self.in_channels / self.S)
            out_param = round(self.out_channels / self.S)
            return (in_param * self.kernel_size ** 3 * out_param) + \
                   (self.in_channels * in_param) + \
                   (self.out_channels * out_param)

        ################
        # TENSOR TRAIN #
        ################

        if self.tensor_net_type in ['tt', 'tensor-train', 'train']:
            '''
            tensor train params:
            r * (in_channels + r*(3+3+3) + out_channels)
            '''
            return self.r * (self.in_channels + self.r * 9 + self.out_channels)


        ##################
        # TENSOR TRAIN 2 #
        ##################

        if self.tensor_net_type in ['tt2', 'tensor-train2', 'train2']:
            '''
            tensor train 2 params:
            r * in_channels + r*3*3 + 3*3*3 + r*3*3 + r * out_channels
            '''
            return self.r*self.in_channels + self.r*3*3 + 3*3*3 + self.r*3*3 + self.r*self.out_channels

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
            AirConv3D(
                compression=compression,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=strides[0],
                tensor_net_type=tensor_net_type,
                **conv_par
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
    dim = 128

    # Test image
    image = torch.rand(1, in_channels, dim, dim, dim)
    image = image.to(device)

    # Classic convolutional layer
    layer_classic = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim, padding=1)
    layer_classic.to(device)

    # Low-rank layers
    compression = 100

    # Canonical layer
    layer_canon = AirConv3D(in_channels=in_channels, out_channels=out_channels,
                            compression=compression,
                            kernel_size=kernel_dim, padding=1, tensor_net_type='cpd')
    layer_canon.to(device)

    # Tucker layer
    layer_tucker2 = AirConv3D(in_channels=in_channels, out_channels=out_channels,
                             compression=compression,
                             kernel_size=kernel_dim, padding=1, tensor_net_type='tucker2')
    layer_tucker2.to(device)

    # TT layer
    layer_tt = AirConv3D(in_channels=in_channels, out_channels=out_channels,
                         compression=compression,
                         kernel_size=kernel_dim, padding=1, tensor_net_type='train')
    layer_tt.to(device)

    # Sample output
    classic_output = layer_classic(image)
    canon_output = layer_canon(image)
    tucker2_output = layer_tucker2(image)
    tt_output = layer_tt(image)

    assert canon_output.size() == classic_output.size(), "Something went wrong with CPD format, output shapes don't match!"
    assert tucker2_output.size() == classic_output.size(), "Something went wrong with Tucker format, output shapes don't match!"
    assert tt_output.size() == classic_output.size(), "Something went wrong with TT format, output shapes don't match!"

    # Double conv test
    """double_conv_classic = DoubleConv(in_channels=in_channels, out_channels=out_channels)
    double_conv_classic_output = double_conv_classic(image)
    double_conv_cpd = AirDoubleConv(compression=compression, tensor_net_type='cpd', in_channels=in_channels,
                                    out_channels=out_channels, num_groups=8)
    double_conv_cpd_output = double_conv_cpd(image)

    assert double_conv_cpd_output.size() == double_conv_classic_output.size(), "Something went wrong with double conv CPD, output shapes don't match!"
    """

    # Attempt to get flop count --> failed for tensor network versions!
    for name, model in zip(('regular', 'cpd', 'tucker2', 'tt'), (layer_classic, layer_canon, layer_tucker2, layer_tt)):
        macs, params = get_model_complexity_info(model=model, input_res=(4, 128, 128, 128), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False,
                                                 custom_modules_hooks={AirConv3D: count_lr_conv3d})
        # flops = count_ops(model=model, input=image,verbose=False)
        # print(name, '-->\tmacs: ', macs,'\tparams: ', params,'\tflops', flops)
        print(name, 'macs', macs, 'params', params)

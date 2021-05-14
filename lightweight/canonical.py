#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Lightweight convolutional layers

CANONICAL POLYADIC FORM
"""

import math

import tensornetwork as tn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out, uniform_

from utils.helper import set_params

tn.set_default_backend('pytorch')


##########################################
# Canonical polyadic decomposition layer #
##########################################

class CanonicalLayer(nn.Module):

    def __init__(self,
                 rank: int,
                 batch_size:int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = True):

        super().__init__()

        # Initializing attributes
        self.rank = rank
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        ###########################
        # Building tensor network #
        ###########################

        # 5 cores
        self.U_c_in = torch.empty(self.rank, self.in_channels, dim, dim, dim, requires_grad=True)
        self.U_kernel_h = torch.empty(self.rank, self.kernel_size, requires_grad=True)
        self.U_kernel_w = torch.empty(self.rank, self.kernel_size, requires_grad=True)
        self.U_kernel_d = torch.empty(self.rank, self.kernel_size, requires_grad=True)
        self.U_c_out = torch.empty(self.rank, self.out_channels, requires_grad=True)

        # Add bias
        if self.bias:
            self.bias = torch.empty(self.batch_size, self.out_channels, dim, dim, dim)

        # Initializing the weights (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L40)
        kaiming_uniform_(self.U_c_in, a=math.sqrt(5))
        kaiming_uniform_(self.U_kernel_h, a=math.sqrt(5))
        kaiming_uniform_(self.U_kernel_w, a=math.sqrt(5))
        kaiming_uniform_(self.U_kernel_d, a=math.sqrt(5))
        kaiming_uniform_(self.U_c_out, a=math.sqrt(5))



    def call(self, inputs):
        # https://blog.tensorflow.org/2020/02/speeding-up-neural-networks-using-tensornetwork-in-keras.html
        # Define the contraction.
        # We break it out so we can parallelize a batch using
        # tf.vectorized_map (see below).

        def f(input, U_c_in, U_k_h, U_k_w, U_k_d, U_c_out, bias=None):
            # Add padding to input tensor
            if self.padding:
                input = F.pad(input=inputs,
                              pad=(self.padding, self.padding, self.padding, self.padding, self.padding, self.padding),
                              mode='constant', value=0)
            # Initializing bias (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L40)
            if self.bias:
                fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                uniform_(self.bias, -bound, bound)

            # Manual approach
            # --> (batch_size, channels, h_windows, w_windows, d_windows, kernel_dim, kernel_dim, kernel_dim)
            input = input.unfold(2, kernel_dim, stride).unfold(3, kernel_dim, stride).unfold(4, kernel_dim, stride)

            # Reshape
            # --> (batch_size, channels, windows, kernel_dim, kernel_dim, kernel_dim)
            input = input.contiguous().view(batch_size, in_channels, -1, kernel_dim, kernel_dim, kernel_dim)

            # Create the tensor network
            # https://colab.research.google.com/drive/1Fp9DolkPT-P_Dkg_s9PLbTOKSq64EVSu
            u_c_in = tn.Node(U_c_in, backend='pytorch')
            u_k_h = tn.Node(U_k_h, backend='pytorch')
            u_k_w = tn.Node(U_k_w, backend='pytorch')
            u_k_d = tn.Node(U_k_d, backend='pytorch')
            u_c_out = tn.Node(U_c_out, backend='pytorch')
            x = tn.Node(input, backend='pytorch')

            output = tn.ncon([x, u_c_in, u_k_h, u_k_w, u_k_d, u_c_out],
                    [
                        [-1, -2, 1, 2, 3, 4],   # Input tensor: 2 dangling edges, 4 contractions
                        [1,5],                  # In_channel core: contracted with input and other cores
                        [2,5],                  # Kernel_h core: contracted with input and other cores
                        [3,5],                  # Kernel_w core: contracted with input and other cores
                        [4,5],                  # Kernel_d core: contracted with input and other cores
                        [-3,5],                 # Out_channel core: contracted with input and other cores
                    ])

            # Add bias
            if bias: output += bias

            return output


        # Vectorize
        tensor_conv = torch.vmap(f)
        result = tensor_conv(input=inputs,U_c_in=self.U_c_in,
                             U_k_h=self.U_kernel_h, U_k_w=self.U_kernel_w, U_k_d=self.U_kernel_d,
                             U_c_out=self.U_c_out, bias=self.bias)

        return result


if __name__ == '__main__':
    # Console parameters
    set_params(verbosity=3, timestamped=False)

    # Quick test (currently no cuda support on my end)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Sanity check: do dimensions make sense? Let's 'benchmark' a classic Conv3D layer
    batch_size = 1
    in_channels = 4
    out_channels = 5
    kernel_dim = 3
    stride = 1
    dim = 128

    layer_classic = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim, padding=1)
    layer_classic.to(device)

    # Create appropriately dimensioned tensor with random values
    image = torch.rand(1, 4, dim, dim, dim)
    image = image.to(device)

    canon = CanonicalLayer(5,batch_size,in_channels,out_channels,kernel_dim)

    # We'll need padding
    '''image_new = F.pad(input=image, pad=(1, 1, 1, 1, 1, 1), mode='constant', value=0)
    dim_pad = image_new.size(-1)
    print('padding image:', image.size(), '-->', image_new.size())

    # Output
    # y = layer_classic(x)

    # Get weight tensor
    filt = layer_classic.weight

    # Manual approach (batch_size, channels, h_windows, w_windows, d_windows, kernel_dim, kernel_dim, kernel_dim)
    patches = image_new.unfold(2, kernel_dim, stride).unfold(3, kernel_dim, stride).unfold(4, kernel_dim, stride)
    print('patches:', patches.size())

    # (batch_size, channels, windows, kernel_dim, kernel_dim, kernel_dim)
    patches = patches.contiguous().view(batch_size, in_channels, -1, kernel_dim, kernel_dim, kernel_dim)
    print('patches reshaped:', patches.size())

    # Get number of windows in single spatial dimension
    n_windows = round(math.pow(patches.size(2), 1 / 3))

    # Windows to batch dim (why?)
    patches_new = patches.permute(0, 2, 1, 3, 4, 5)

    # Calculate the convolutions manually
    result = (patches_new.unsqueeze(2) * filt.unsqueeze(0).unsqueeze(1)).sum([3, 4, 5, 6])
    print('result:', result.size())
    manual_result = result.permute(0, 2, 1).view(batch_size, -1, dim, dim, dim)

    model_result = layer_classic(image)

    print((manual_result - model_result).abs().max())'''

    '''print(res.shape)  # batch_size, output_pixels, out_channels
    res = res.permute(0, 2, 1)  # batch_size, out_channels, output_pixels
    # assuming h = w
    h = w = int(res.size(2) ** 0.5)
    res = res.view(batch_size, -1, h, w)

    # Module approach
    out = conv(image)
    # Can we fold it back?
    '''

    '''neat = patches_new.permute(0,2,1,3,4,5)
    neat = neat.contiguous().view(batch_size, in_channels, n_windows, n_windows, n_windows, kernel_dim, kernel_dim, kernel_dim)

    neat_new = F.fold(neat,image_new.size()[2:],kernel_size=kernel_dim,stride=stride)'''

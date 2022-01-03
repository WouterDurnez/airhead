#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Custom base layers
"""

import torch
import torch.nn as nn
from utils.helper import hi, log


##########
# Layers #
##########

# Downsample block
class DownsampleMax(nn.Module):

    def __init__(self,
                 down_par=None):
        super().__init__()
        self.__name__ = 'max_pool_3d'

        # Initialize parameters if not given
        down_par = down_par if down_par else {}
        down_par.setdefault('kernel_size', 2)
        down_par.setdefault('stride', 2)
        down_par.setdefault('padding', 0)

        self.block = nn.Sequential(
            nn.MaxPool3d(**down_par)
        )

    # Forward propagation
    def forward(self, input):
        return (self.block(input))


# Upsample block (currently transposed convolution)
class Upsample(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 up_par=None
                 ):
        super().__init__()
        self.__name__ = 'trans_conv'

        # Initialize convolution parameters
        up_par = up_par if up_par else {}

        # Set parameters (if not given!)
        up_par.setdefault('kernel_size', 3)
        up_par.setdefault('padding', 1)
        up_par.setdefault('stride', 2)
        up_par.setdefault('output_padding', 1)

        # Transposed convolution (can be turned into sequential model including normalization and activation?)
        self.block = nn.ConvTranspose3d(in_channels, out_channels, **up_par)

    # Forward propagation
    def forward(self, input):
        return self.block(input)


##########
# Blocks #
##########

# ResNet block
class ResBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 2,
            num_groups: int = 8,
            activation:nn.Module = nn.LeakyReLU,
            conv: nn.Module = nn.Conv3d,
            conv_params: dict = None,
            name:str ='res_block',
    ):
        super().__init__()
        self.__name__ = name

        # Initialize convolution parameters, set defaults
        conv_params = {} if conv_params is None else conv_params
        conv_params.setdefault("kernel_size", 3)
        conv_params.setdefault("padding", 1)

        self.conv1 = conv(
            in_channels, out_channels, stride=stride, **conv_params
        )
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = conv(out_channels, out_channels, **conv_params)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act = activation(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += shortcut

        out = self.act(out)

        return out


# Double convolution block
class DoubleConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 2,
            num_groups: int = 8,
            activation:nn.Module = nn.LeakyReLU,
            conv: nn.Module = nn.Conv3d,
            conv_params: dict = None,
            name:str='double_conv_block',
    ):
        super().__init__()
        self.__name__ = name

        # Initialize convolution parameters
        conv_params = conv_params if conv_params else {}

        # Set parameters (if not given!)
        conv_params.setdefault('kernel_size', 3)
        conv_params.setdefault('padding', 1)

        # Define inner block architecture
        self.block = nn.Sequential(

            # Convolution layer
            conv(in_channels=in_channels,
                 out_channels=out_channels,
                 stride=stride,
                 **conv_params),

            # Normalization layer (default minibatch of 8 instances)
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),

            # Activation layer
            activation(inplace=True),

            # Convolution layer
            conv(in_channels=out_channels,
                 out_channels=out_channels,
                 **conv_params),

            # Normalization layer (default minibatch of 8 instances)
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),

            # Activation layer
            activation(inplace=True)
        )

    # Forward function (backward propagation is added automatically)
    def forward(self, input):
        return self.block(input)


if __name__ == '__main__':
    # Console parameters
    hi(verbosity=3, timestamped=False)

    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create appropriately dimensioned tensor with random values
    dim = 128
    x = torch.rand(1, 4, dim, dim, dim)
    x.to(device)
    log(f'Input size (single image): {x.size()}')

    test_params = {
        'in_channels': 4,
        'out_channels': 32,
        'conv_params': {
            'kernel_size': 5,
            'padding': 2,
        },
        'stride': 1
    }

    # Init blocks
    res_block = ResBlock(**test_params)
    doub_block = DoubleConvBlock(**test_params)

    # Process dummy input
    out_res = res_block(x)
    out_doub = doub_block(x)

    # Show resulting shape
    log(f'Output size ({res_block.__name__}): {out_res.size()}')
    log(f'Output size (DoubleConv): {out_doub.size()}')

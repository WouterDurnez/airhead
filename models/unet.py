#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
3D U-Net implementation as described in
Isensee, F., Jaeger, P. F., Full, P. M., Vollmuth, P., & Maier-Hein, K. H. (2020).
 nnU-Net for Brain Tumor Segmentation. 1–15. http://arxiv.org/abs/2011.00848

-- Coded by Wouter Durnez
"""

import torch
import torch.nn as nn
from torch import cat
from helper import *


# Downsample block (currently strided convolutions)
class Downsample(nn.Module):

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


# Double convolution block
class DoubleConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_groups=8,
            strides=(2,1),
            activation=nn.LeakyReLU(inplace=True),
            conv_par=None,
            __name__='double_conv',
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

            # Convolution layer
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      stride=strides[0],
                      **conv_par),

            # Normalization layer (default minibatch of 8 instances)
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),

            # Activation layer
            activation,

            # Convolution layer
            nn.Conv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      stride=strides[1],
                      **conv_par),

            # Normalization layer (default minibatch of 8 instances)
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),

            # Activation layer
            activation
        )

    # Forward function (backward propagation is added automatically)
    def forward(self, input):
        return self.block(input)


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


# Full 3D-UNet architecture
class UNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            widths=(32, 64, 128, 256, 320),
            activation=nn.LeakyReLU(inplace=True),
            conv_par=None,
            down_par=None,
            up_par=None
    ):
        super().__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.widths = widths

        ##############
        # Parameters #
        ##############

        # Set default parameters for double convolution
        conv_par = conv_par if conv_par else {}
        conv_par.setdefault('kernel_size', 3)
        conv_par.setdefault('padding', 1)

        # Set default parameters for downsampling (max pooling)
        down_par = down_par if down_par else {}
        down_par.setdefault('kernel_size', 2)
        down_par.setdefault('stride', 2)
        down_par.setdefault('padding', 0)

        # Set default parameters for upsampling (transposed convolution)
        up_par = up_par if up_par else {}
        up_par.setdefault('kernel_size', 3)
        up_par.setdefault('padding', 1)
        up_par.setdefault('output_padding', 1)

        ################
        # Architecture #
        ################

        # ENCODER
        """
        5 segments composed of double convolution blocks, followed by pooling (downsampling)
        """
        self.enc_1 = DoubleConv(self.in_channels, self.widths[0], strides=(1,1), activation=activation, conv_par=conv_par)
        self.down_1 = Downsample(down_par=down_par)
        self.enc_2 = DoubleConv(self.widths[0], self.widths[1], activation=activation, conv_par=conv_par)
        self.down_2 = Downsample(down_par=down_par)
        self.enc_3 = DoubleConv(self.widths[1], self.widths[2], activation=activation, conv_par=conv_par)
        self.down_3 = Downsample(down_par=down_par)
        self.enc_4 = DoubleConv(self.widths[2], self.widths[3], activation=activation, conv_par=conv_par)
        self.down_4 = Downsample(down_par=down_par)
        self.enc_5 = DoubleConv(self.widths[3], self.widths[4], activation=activation, conv_par=conv_par)
        self.down_5 = Downsample(down_par=down_par)

        # BRIDGE
        self.bridge = DoubleConv(self.widths[4], self.widths[4])

        # DECODER
        """
        5 segments composed of transposed convolutions (upsampling) and double convolution blocks
        """
        self.up_1 = Upsample(self.widths[4], self.widths[4], up_par=up_par)
        self.dec_1 = DoubleConv(2 * self.widths[4], self.widths[3], strides=(1,1), activation=activation, conv_par=conv_par)  # double the filters due to concatenation
        self.up_2 = Upsample(self.widths[3], self.widths[3], up_par=up_par)
        self.dec_2 = DoubleConv(2 * self.widths[3], self.widths[2], strides=(1,1), activation=activation, conv_par=conv_par)
        self.up_3 = Upsample(self.widths[2], self.widths[2], up_par=up_par)
        self.dec_3 = DoubleConv(2 * self.widths[2], self.widths[1], strides=(1,1), activation=activation, conv_par=conv_par)
        self.up_4 = Upsample(self.widths[1], self.widths[1], up_par=up_par)
        self.dec_4 = DoubleConv(2 * self.widths[1], self.widths[0], strides=(1,1), activation=activation, conv_par=conv_par)
        self.up_5 = Upsample(self.widths[0], self.widths[0], up_par=up_par)
        self.dec_5 = DoubleConv(2 * self.widths[0], self.widths[0], strides=(1,1), activation=activation, conv_par=conv_par)

        # Output
        self.final_conv = nn.Conv3d(in_channels=self.widths[0], out_channels=out_channels, kernel_size=1)
        self.final_act = nn.Softmax(dim=1)

    # Forward propagation
    def forward(self, input):
        """
        Combine layers into encoder-decoder structure, adding skip connections
        by concatenating layer output from encoder to input for decoder.
        """
        # Encoding
        log('ENCODER')
        enc_1 = whatsgoingon(self.enc_1, input)
        #down_1 = whatsgoingon(self.down_1, enc_1)

        enc_2 = whatsgoingon(self.enc_2, enc_1)
        #down_2 = whatsgoingon(self.down_2, enc_2)

        enc_3 = whatsgoingon(self.enc_3, enc_2)
        #down_3 = whatsgoingon(self.down_3, enc_3)

        enc_4 = whatsgoingon(self.enc_4, enc_3)
        #down_4 = whatsgoingon(self.down_4, enc_4)

        enc_5 = whatsgoingon(self.enc_5, enc_4)
        #down_5 = whatsgoingon(self.down_5, enc_5)

        encoded = whatsgoingon(self.bridge, enc_5)

        # Decoding
        log('DECODER')
        up_1 = whatsgoingon(self.up_1, encoded)
        cat_1 = cat([up_1, enc_5], dim=1)
        dec_1 = whatsgoingon(self.dec_1, cat_1)  # dec_1 has widths[3] (256)

        up_2 = whatsgoingon(self.up_2, dec_1)
        cat_2 = cat([up_2, enc_4], dim=1)  # up_2 has widths[3] (256) + enc_5 has widths[4] (320) = 640
        dec_2 = whatsgoingon(self.dec_2, cat_2)

        up_3 = whatsgoingon(self.up_3, dec_2)
        cat_3 = cat([up_3, enc_3], dim=1)
        dec_3 = whatsgoingon(self.dec_3, cat_3)

        up_4 = whatsgoingon(self.up_4, dec_3)
        cat_4 = cat([up_4, enc_2], dim=1)
        dec_4 = whatsgoingon(self.dec_4, cat_4)

        up_5 = whatsgoingon(self.up_5, dec_4)
        cat_5 = cat([up_5, enc_1], dim=1)
        dec_5 = whatsgoingon(self.dec_5, cat_5)

        # Final
        log('HEAD')
        final_conv = self.final_conv(dec_5)
        output = self.final_act(final_conv)

        return output


if __name__ == '__main__':
    # Console parameters
    set_params(verbosity=3, timestamped=False)

    # Quick test (currently no cuda support on my end)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create appropriately dimensioned tensor with random values
    dim = 128
    x = torch.rand(1, 4, dim, dim, dim)
    x.to(device)
    log(f'Input size (single image): {x.size()}')

    # Initialize model
    model = UNet(in_channels=4, out_channels=1)

    # Process example input
    out = model(x)
    log(f'Output size (single image): {out.size()}')


"""
3D U-Net implementation

-- Coded by Wouter Durnez
"""

import torch
import torch.nn as nn
from torch import cat


# Double convolution
class DoubleConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_groups=8,
            conv_params=None,
    ):
        super().__init__()

        # Initialize convolution parameters
        conv_params = {} if None else conv_params

        # Set parameters (if not given!)
        conv_params.setdefault('kernel_size', 3)
        conv_params.setdefault('padding', 1)
        conv_params.setdefault('activation', nn.LeakyReLU(inplace=True))

        # Define inner block architecture
        self.block = nn.Sequential(

            # Convolution layer
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, **conv_params),

            # Normalization layer (default minibatch of 8 instances)
            nn.GroupNorm(num_groups=num_groups),

            # Activation layer
            conv_params['activation'],

            # Convolution layer
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, **conv_params),

            # Normalization layer (default minibatch of 8 instances)
            nn.GroupNorm(num_groups=num_groups),

            # Activation layer
            conv_params['activation']
        )

    # Forward function (backward propagation is added automatically)
    def forward(self, input):
        return self.block(input)


# Transposed convolution
class TransConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_params=None
                 ):
        super().__init__()

        # Initialize convolution parameters
        conv_params = {} if None else conv_params

        # Set parameters (if not given!)
        conv_params.setdefault('kernel_size', 3)
        conv_params.setdefault('padding', 1)
        conv_params.setdefault('stride', 2)
        conv_params.setdefault('output_padding', 1)

        # Transposed convolution (can be turned into sequential model including normalization and activation?)
        self.block = nn.ConvTranspose3d(in_channels, out_channels, **conv_params)

    # Forward propagation
    def forward(self, input):
        return self.block(input)


# Full 3D-UNet architecture
class UNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_filters,
            conv_par=None,
            down_par=None,
            up_par=None
    ):
        super().__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters

        # Set default parameters for downsampling (max pooling)
        down_par = {} if None else down_par
        down_par.setdefault('kernel_size', 2)
        down_par.setdefault('stride', 2)
        down_par.setdefault('padding', 0)

        # Set default parameters for upsampling (transposed convolution)
        up_par = {} if None else up_par
        up_par.setdefault('kernel_size', 3)
        up_par.setdefault('padding', 1)
        up_par.setdefault('output_padding', 1)

        # ENCODER
        """
        5 segments composed of double convolution blocks, followed by pooling (downsampling)
        """
        self.enc_1 = DoubleConv(self.in_channels, self.num_filters, **conv_par)
        self.down_1 = nn.MaxPool3d(**down_par)
        self.enc_2 = DoubleConv(self.num_filters, self.num_filters * 2, **conv_par)
        self.down_2 = nn.MaxPool3d(**down_par)
        self.enc_3 = DoubleConv(self.num_filters * 2, self.num_filters * 4, **conv_par)
        self.down_3 = nn.MaxPool3d(**down_par)
        self.enc_4 = DoubleConv(self.num_filters * 4, self.num_filters * 8, **conv_par)
        self.down_4 = nn.MaxPool3d(**down_par)
        self.enc_5 = DoubleConv(self.num_filters * 8, self.num_filters * 16, **conv_par)
        self.down_5 = nn.MaxPool3d(**down_par)

        # BRIDGE
        self.bridge = DoubleConv(self.num_filters * 16, self.num_filters * 32)

        # DECODER
        """
        5 segments composed of transposed convolutions (upsampling) and double convolution blocks
        """
        self.up_1 = TransConv(self.num_filters * 32, self.num_filters * 32)
        self.dec_1 = DoubleConv(self.num_filters * 48, self.num_filters * 16)  # 48 filters due to concatenation
        self.up_2 = nn.ConvTranspose3d(self.num_filters * 16, self.num_filters * 16, **up_par)
        self.dec_2 = DoubleConv(self.num_filters * 24, self.num_filters * 8)
        self.up_3 = nn.ConvTranspose3d(self.num_filters * 8, self.num_filters * 8, **up_par)
        self.dec_3 = DoubleConv(self.num_filters * 12, self.num_filters * 4)
        self.up_4 = nn.ConvTranspose3d(self.num_filters * 4, self.num_filters * 4, **up_par)
        self.dec_4 = DoubleConv(self.num_filters * 6, self.num_filters * 2)
        self.up_5 = nn.ConvTranspose3d(self.num_filters * 2, self.num_filters * 2, **up_par)
        self.dec_5 = DoubleConv(self.num_filters * 3, self.num_filters)

        # Output
        self.final_conv = nn.Conv3d(in_channels=self.num_filters, out_channels=out_channels, kernel_size=1)
        self.final_act = nn.Softmax()

    # Forward propagation
    def forward(self, input):
        """
        Combine layers into encoder-decoder structure, adding skip connections by concatenating
        layer output from encoder to input from decoder.
        """
        # Encoding
        enc_1 = self.enc_1(input)
        down_1 = self.down_1(enc_1)
        enc_2 = self.enc_2(down_1)
        down_2 = self.down_2(enc_2)
        enc_3 = self.enc_3(down_2)
        down_3 = self.down_3(enc_3)
        enc_4 = self.enc_4(down_3)
        down_4 = self.down_4(enc_4)
        enc_5 = self.enc_5(down_4)
        down_5 = self.down_5(enc_5)
        encoded = self.bridge(down_5)

        # Decoding
        up_1 = self.up_1(encoded)
        dec_1 = self.dec_1(cat(up_1, enc_5))
        up_2 = self.up_1(dec_1)
        dec_2 = self.dec_1(cat(up_2, enc_4))
        up_3 = self.up_1(dec_2)
        dec_3 = self.dec_1(cat(up_3, enc_3))
        up_4 = self.up_1(dec_3)
        dec_4 = self.dec_1(cat(up_4, enc_2))
        up_5 = self.up_1(dec_4)
        dec_5 = self.dec_1(cat(up_5, enc_1))

        # Final
        final_conv = self.final_conv(dec_5)
        output = self.final_conv(final_conv)

        return output


if __name__ == '__main__':
    # Quick test (currently no cuda support on my end)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create appropriately dimensioned tensor
    dim = 128
    x = torch.rand(1, 3, dim, dim, dim)
    x.to(device)
    print(f'x size: {x.size()}')

    # Initialize model
    model = UNet(in_channels=4, out_channels=4, num_filters=32)

    # Process example input
    out = model(x)

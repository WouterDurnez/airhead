#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
3D U-Net implementation as described in
Isensee, F., Jaeger, P. F., Full, P. M., Vollmuth, P., & Maier-Hein, K. H. (2020).
 nnU-Net for Brain Tumor Segmentation. 1–15. http://arxiv.org/abs/2011.00848
"""

import torch
import torch.nn as nn
from torch import cat
from torch.nn.modules import Conv3d

from src.layers.base_layers import DoubleConvBlock, Upsample, ResBlock
from src.utils.helper import *


# Full 3D-UNet architecture
class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        widths=(32, 64, 128, 256, 320),
        activation: nn.Module = nn.LeakyReLU,
        core_block: nn.Module = DoubleConvBlock,
        core_block_conv: nn.Module = Conv3d,
        core_block_conv_params: dict = None,
        downsample='strided_convolution',
        up_par=None,
        head=True,
    ):
        super().__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.widths = widths
        self.head = head

        if not downsample == 'strided_convolution':
            raise NotImplementedError(
                'Only <strided convolution> downsample method implemented!'
            )

        ##############
        # Parameters #
        ##############

        # Set default parameters for double convolution
        core_block_conv_params = (
            core_block_conv_params if core_block_conv_params else {}
        )
        core_block_conv_params.setdefault('kernel_size', 3)
        core_block_conv_params.setdefault('padding', 1)

        # Set default parameters for upsampling (transposed convolution)
        up_par = up_par if up_par else {}
        up_par.setdefault('kernel_size', 3)
        up_par.setdefault('padding', 1)
        up_par.setdefault('stride', 2)
        up_par.setdefault('output_padding', 1)

        ################
        # Architecture #
        ################

        # ENCODER
        """
        5 segments composed of double convolution blocks, followed by strided convolutoin (downsampling)
        """
        self.enc_1 = core_block(
            in_channels=self.in_channels,
            out_channels=self.widths[0],
            stride=1,
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )
        self.enc_2 = core_block(
            in_channels=self.widths[0],
            out_channels=self.widths[1],
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )
        self.enc_3 = core_block(
            in_channels=self.widths[1],
            out_channels=self.widths[2],
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )
        self.enc_4 = core_block(
            in_channels=self.widths[2],
            out_channels=self.widths[3],
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )
        self.enc_5 = core_block(
            in_channels=self.widths[3],
            out_channels=self.widths[4],
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )

        # BRIDGE
        self.bridge = core_block(
            self.widths[4],
            self.widths[4],
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )

        # DECODER
        """
        5 segments composed of transposed convolutions (upsampling) and double convolution blocks
        """
        self.up_1 = Upsample(self.widths[4], self.widths[4], up_par=up_par)
        self.dec_1 = core_block(
            2 * self.widths[4],
            self.widths[3],
            stride=1,
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )  # double the filters due to concatenation
        self.up_2 = Upsample(self.widths[3], self.widths[3], up_par=up_par)
        self.dec_2 = core_block(
            2 * self.widths[3],
            self.widths[2],
            stride=1,
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )
        self.up_3 = Upsample(self.widths[2], self.widths[2], up_par=up_par)
        self.dec_3 = core_block(
            2 * self.widths[2],
            self.widths[1],
            stride=1,
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )
        self.up_4 = Upsample(self.widths[1], self.widths[1], up_par=up_par)
        self.dec_4 = core_block(
            2 * self.widths[1],
            self.widths[0],
            stride=1,
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )
        self.up_5 = Upsample(self.widths[0], self.widths[0], up_par=up_par)
        self.dec_5 = core_block(
            2 * self.widths[0],
            self.widths[0],
            stride=1,
            activation=activation,
            conv=core_block_conv,
            conv_params=core_block_conv_params,
        )

        # Output
        self.final_conv = nn.Conv3d(
            in_channels=self.widths[0],
            out_channels=out_channels,
            kernel_size=1,
        )
        if self.head:
            # self.final_act = nn.Softmax(dim=1)
            self.final_act = nn.Sigmoid()

    # Forward propagation
    def forward(self, input):
        """
        Combine layers into encoder-decoder structure, adding skip connections
        by concatenating layer output from encoder to input for decoder.
        """
        # Encoding
        """Downsampling = strided convolutions"""
        enc_1 = self.enc_1(input)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)
        enc_4 = self.enc_4(enc_3)
        enc_5 = self.enc_5(enc_4)
        encoded = self.bridge(enc_5)

        # Decoding
        up_1 = self.up_1(encoded)
        cat_1 = cat([up_1, enc_5], dim=1)
        dec_1 = self.dec_1(cat_1)  # dec_1 has widths[3] (256)

        up_2 = self.up_2(dec_1)
        cat_2 = cat(
            [up_2, enc_4], dim=1
        )  # up_2 has widths[3] (256) + enc_5 has widths[4] (320) = 640
        dec_2 = self.dec_2(cat_2)

        up_3 = self.up_3(dec_2)
        cat_3 = cat([up_3, enc_3], dim=1)
        dec_3 = self.dec_3(cat_3)

        up_4 = self.up_4(dec_3)
        cat_4 = cat([up_4, enc_2], dim=1)
        dec_4 = self.dec_4(cat_4)

        up_5 = self.up_5(dec_4)
        cat_5 = cat([up_5, enc_1], dim=1)
        dec_5 = self.dec_5(cat_5)

        # Final
        final_conv = self.final_conv(dec_5)
        if self.head:
            output = self.final_act(final_conv)
        else:
            output = final_conv

        return output


if __name__ == '__main__':
    # Console parameters
    set_params(verbosity=3, timestamped=False)

    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create appropriately dimensioned tensor with random values
    dim = 128
    x = torch.rand(1, 4, dim, dim, dim)
    x.to(device)
    log(f'Input size (single image): {x.size()}')

    # Initialize model
    base_model = UNet(in_channels=4, out_channels=3, head=False)
    res_model = UNet(
        in_channels=4,
        out_channels=3,
        head=False,
        core_block=ResBlock,
        core_block_conv_params={'kernel_size': 5, 'padding': 2},
    )

    # Process example input
    out_base = base_model(x)
    out_res = res_model(x)
    log(f'Output size (base): {out_base.size()}')
    log(f'Output size (res): {out_res.size()}')

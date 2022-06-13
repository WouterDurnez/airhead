#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Performance measurements tools (mac/flop counter)
"""
import os

import numpy as np
import pandas as pd

from src.layers.air_layers import AirConv3D, AirResBlock
from src.layers.base_layers import ResBlock
import ptflops
import torch
from torch.nn.modules import Conv3d
from src.models.air_unet import AirUNet
from src.models.base_unet import UNet
from src.utils.helper import log, TENSOR_NET_TYPES
import src.utils.helper as hlp
from itertools import product
from tqdm import tqdm


def air_conv_counter(module, _, y):
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


def generate_test_params(base_only=False) -> list:
    """
    Generate test parameter combinations for compression rates, network types, widths and kernel sizes
    """

    # BASE
    test_params = [
        {
            'comp': 1,
            'type': 'base',
            'widths': w,
            'kernel_size': k,
        }
        for w, k in ((0, 3), (0, 5), (1, 3))
    ]
    if base_only:
        return test_params

    # TENSORIZED
    compression_rates = (2, 4, 8, 16, 32, 64, 128, 256)
    tensor_net_types = ('tucker', 'tt', 'cp')
    for c, t in product(compression_rates, tensor_net_types):
        test_params.append(
            {
                'comp': c,
                'type': t,
                'widths': 0,
                'kernel_size': 3,
            }
        )
    # ABLATION
    for c, (w, k) in product(compression_rates, ((0, 5), (1, 3))):
        test_params.append(
            {
                'comp': c,
                'type': 'tucker',
                'widths': w,
                'kernel_size': k,
            }
        )
    return test_params


class DummyAirConv3D(AirConv3D):
    """
    Dummy subclass of AirConv3D, replacing forward with random output generator
    """

    def forward(self, x):

        # Unpack input shape parameters
        b, c_in, *spatial_shape = x.shape

        # Stride affects spatial shape
        if self.stride != 1:
            spatial_shape = [
                round(
                    ((x - self.kernel_size + 2 * self.padding) / self.stride)
                    + 1
                )
                for x in spatial_shape
            ]

        # We return *out_channels* channels
        c_out = self.out_channels
        return torch.rand(b, c_out, *spatial_shape)


if __name__ == '__main__':

    # Let's go
    hlp.hi('MAC & param counter', log_dir='../../../logs/model_params')

    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get test parameters
    log('Get test parameters.')
    test_params = generate_test_params(base_only=False)

    # Log results here
    results = []

    # Dummy input
    dim = 128
    x = torch.rand(1, 4, dim, dim, dim)
    x.to(device)

    # Start counting
    log('Begin MAC & parameter count.')
    for params in tqdm(test_params, desc='Counting...'):

        # Set some necessary extra parameters
        padding = int((params['kernel_size'] - 1) / 2)
        widths = (
            (32, 64, 128, 256, 512)
            if params['widths'] == 0
            else (48, 96, 192, 384, 768)
        )

        # Create model
        if params['type'] == 'base':
            dummy_model = UNet(
                widths=widths,
                in_channels=4,
                out_channels=3,
                core_block=ResBlock,
                core_block_conv=Conv3d,
                core_block_conv_params={
                    'kernel_size': params['kernel_size'],
                    'padding': padding,
                },
            )
        else:
            dummy_model = AirUNet(
                compression=params['comp'],
                tensor_net_type=params['type'],
                widths=widths,
                in_channels=4,
                out_channels=3,
                core_block=AirResBlock,
                core_block_conv=DummyAirConv3D,
                core_block_conv_params={
                    'kernel_size': params['kernel_size'],
                    'padding': padding,
                },
                comp_friendly=False,
            )

        # Count macs
        macs, n_param = ptflops.get_model_complexity_info(
            model=dummy_model,
            input_res=(4, 128, 128, 128),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            custom_modules_hooks={DummyAirConv3D: air_conv_counter},
        )

        # Store results
        result = params.copy()
        result['macs'] = macs
        result['params'] = n_param
        results.append(result)

    # Store all info
    log('Saving results.')
    df = pd.DataFrame(results)
    df.to_csv(
        os.path.join(hlp.LOG_DIR, 'df_param_counts.csv'),
        sep=',',
        decimal='.',
        index=False,
    )

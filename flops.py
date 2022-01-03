#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Get parameters (flops/macs) for all networks
"""
import argparse
import os
from os.path import join
from pprint import PrettyPrinter

import numpy as np
from ptflops import get_model_complexity_info

import utils.helper as hlp
from layers.air_conv import count_lr_conv3d, AirConv3D, AirResBlock
from layers.base_layers import ResBlock
from models.air_unet import AirUNet
from models.base_unet import UNet
from utils.helper import log, hi, TENSOR_NET_TYPES

pp = PrettyPrinter()

if __name__ == '__main__':

    hi('Flop/mac counter', verbosity=3,
       data_dir='../data',
       log_dir='../logs/flops')

    #result_dir = join(hlp.LOG_DIR, 'results')

    # Get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', help='Network type (base, cp, tt, tucker)', default='base', type=str)
    parser.add_argument('--comp', help='Compression rate (int)', default=1, type=int)
    parser.add_argument('--kernel', help='Kernel size for core block convolutions (default: 3)', default=3, type=int)
    parser.add_argument('--widths', help='Width configuration to use', default=0, type=int)

    args = parser.parse_args()

    # Parameters
    net_type = args.type
    kernel_size = args.kernel
    widths = args.widths
    compression = args.comp

    assert net_type in TENSOR_NET_TYPES or net_type == 'base', \
        f"Choose a valid network type ('base','cp', 'tucker', or 'tt')!"
    assert kernel_size in (3, 5, 7), \
        f"Choose a valid kernel size (3, 5, 7), not <{kernel_size}>"
    assert widths in (0, 1), \
        f"Choose a valid widths setting (0, 1), not <{widths}>"

    widths = (32, 64, 128, 256, 512) if widths == 0 else (48, 96, 192, 384, 768)

    # Store all info here
    info = {}

    log('Parameters', title=True)
    log('Type:          ', net_type)
    log('Compression:   ', compression)
    log('Widths:        ', widths)
    log('Kernel size:   ', kernel_size)

    ################
    # Start counts #
    ################

    if net_type == 'base':
        log(f'Counting baseline network [w = {widths}; k = {kernel_size}]', color='blue', verbosity=1)

        # Build model
        baseline_model = UNet(in_channels=4, out_channels=3,
                              widths=widths,
                              core_block=ResBlock,
                              core_block_conv_params={
                                  'kernel_size': kernel_size,
                                  'padding': int((kernel_size - 1) / 2),
                              }
                              )

        # Get macs and parameters
        macs, params = get_model_complexity_info(model=baseline_model, input_res=(4, 128, 128, 128),
                                                 as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)

    # TENSORIZED NETWORKS #
    #######################

    else:

        log(f'Counting tensorized network [w = {widths}; k = {kernel_size}; '
            f'type = {net_type}; comp = {compression}]', color='blue', verbosity=1)

        # Build tensorized U-Net
        model = AirUNet(in_channels=4, out_channels=3,
                        compression=compression,
                        tensor_net_type=net_type,
                        comp_friendly=False,
                        widths=widths,
                        core_block=AirResBlock,
                        core_block_conv_params={
                            'kernel_size': kernel_size,
                            'padding': int((kernel_size - 1) / 2),
                        }
                        )

        # Get macs and parameters (using custom hook)
        macs, params = get_model_complexity_info(model=model, input_res=(4, 128, 128, 128),
                                                 as_strings=False,
                                                 print_per_layer_stat=False, verbose=False,
                                                 custom_modules_hooks={
                                                     AirConv3D: count_lr_conv3d})

    # Store info
    info[net_type] = {
            compression: {
                widths: {
                    kernel_size: {
                        'macs': macs,
                        'params': params
                    }
                }
            }
        }

    # Store all info
    np.save(os.path.join(hlp.LOG_DIR, f'model_flops_{net_type}_c{compression}_'
                                          f'w{widths}_k{kernel_size}.npy'), info)



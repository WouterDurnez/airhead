#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Get parameters (flops/macs) for all networks
"""

from os.path import join
from pprint import PrettyPrinter

import numpy as np
from ptflops import get_model_complexity_info

from layers.air_conv import count_lr_conv3d, AirConv3D, AirDoubleConv
from models.air_unet import AirUNet
from models.baseline_unet import UNet
from utils.helper import log, hi, LOG_DIR

pp = PrettyPrinter()

if __name__ == '__main__':

    hi('Flop/mac counter', verbosity=1, log_dir='../../logs_cv')

    result_dir = join(LOG_DIR, 'results')

    # Store all info here
    info = {}

    ############
    # Baseline #
    ############
    log('Counting baseline network.', color='blue', verbosity=1)

    # Build baseline U-Net
    baseline_model = UNet(in_channels=4, out_channels=3)

    # Get macs and parameters
    macs, params = get_model_complexity_info(model=baseline_model, input_res=(4, 128, 128, 128), as_strings=False,
                                             print_per_layer_stat=False, verbose=False)

    # Store info
    info['baseline'] = {
        'macs': macs,
        'params': params
    }

    ############
    # Low rank #
    ############

    # Loop over types and compression rates
    for tensor_net_type in ('cp', 'tt', 'tucker'):

        info[tensor_net_type] = {}

        for compression in (2, 5, 10, 20, 35, 50, 75, 100):
            log(f'Counting tensorized network [{tensor_net_type}-{compression}].', verbosity=1, color='blue')

            # Build tensorized U-Net
            model = AirUNet(compression=compression,
                            tensor_net_type=tensor_net_type,
                            double_conv=AirDoubleConv,
                            in_channels=4,
                            out_channels=3,
                            comp_friendly=False)

            # Get macs and parameters (using custom hook)
            macs, params = get_model_complexity_info(model=model, input_res=(4, 128, 128, 128), as_strings=False,
                                                     print_per_layer_stat=False, verbose=False,
                                                     custom_modules_hooks={
                                                         AirConv3D: count_lr_conv3d})

            # Store info
            info[tensor_net_type][compression] = {
                'macs': macs,
                'params': params
            }

    # Store all info
    np.save('model_flops.npy', info)

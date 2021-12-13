#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Main training script for lightweight U-Net
"""

import argparse
from os.path import join
from pprint import PrettyPrinter

import numpy as np
from medset.brats import BraTSDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from configs.live_config import CONFIG_AIR, CONFIG_BASE
from configs.debug_config import CONFIG_AIR as DEBUG_CONFIG_AIR
from configs.debug_config import CONFIG_BASE as DEBUG_CONFIG_BASE
from training.lightning import UNetLightning
from utils import helper as hlp
from utils.helper import log, set_dir
from utils.helper import TENSOR_NET_TYPES
pp = PrettyPrinter()

if __name__ == '__main__':

    # Get arguments
    debug_mode = True
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', help='Network type (base, cp, tt, tucker)')
    parser.add_argument('--comp', help='Compression rate (int)')
    parser.add_argument('--fold', help='Fold index (0-4)')

    args = parser.parse_args()

    # Parameters
    net_type = args.type if args.type else 'base'
    assert net_type in TENSOR_NET_TYPES or net_type == 'base', \
        f"Choose a valid network type ('base','cp', 'tucker', or 'tt')!"

    # Baseline or tensorized?
    if net_type == 'base':

        # Use base configuration
        config = CONFIG_BASE if not debug_mode else DEBUG_CONFIG_BASE
        compression = version = 1

    else:

        # If we use a tensorized model, compression rate must be given
        assert args.comp is not None, "Please enter a valid compression rate! (--COMP)"
        config = CONFIG_AIR if not debug_mode else DEBUG_CONFIG_AIR

        # Extra parameters for AirUNet
        config['model']['network_params']['compression'] = version = int(args.comp)
        config['model']['network_params']['tensor_net_type'] = net_type

    # Let's go
    hlp.hi("Training lightweight U-Net", log_dir=config['logs']['log_dir'])

    fold_index = int(args.fold) if args.fold else 0
    model_name = f'unet_{net_type}_f{fold_index}'

    log(f'Network type: {args.type}', color='green')
    log(f'Compression rate: {args.comp}', color='green')
    log(f'Fold index: {args.fold}', color='green')

    # Set data directory
    data_dir = config['data']['root_dir']
    tb_dir = join(hlp.LOG_DIR, 'tb_logs')
    snap_dir = join(hlp.LOG_DIR, 'snapshots', model_name)
    result_dir = join(hlp.LOG_DIR, 'results', model_name)
    set_dir(data_dir, tb_dir, snap_dir, result_dir)

    # Initialize model
    log(f"Initializing <{model_name}> model.")
    model = UNetLightning(**config['model'])

    # Initialize data module
    log("Initializing data module.")
    brats = BraTSDataModule(**config['data'])

    # Initialize logger
    tb_logger = TensorBoardLogger(save_dir=tb_dir, name=model_name, default_hp_metric=False, version=version)

    # Initialize trainer
    log("Initializing trainer.")
    trainer = Trainer(**config['training'], logger=tb_logger)

    # Train
    log("Commencing training.")
    trainer.fit(model=model,
                datamodule=brats)

    # Additional checkpoint (just in case)
    trainer.save_checkpoint(join(snap_dir, f'final_{model_name}_v{version}_fold{fold_index}.ckpt'))

    # Test
    log("Evaluating model.")
    trainer.test(model=model,
                 datamodule=brats)
    results = model.test_results

    # Save test results
    log("Saving results.")
    np.save(file=join(result_dir, f'{model_name}_v{version}_fold{fold_index}.npy'), arr=results)

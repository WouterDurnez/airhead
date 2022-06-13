#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Main training script for lightweight U-Net
"""

import argparse
import os
from os.path import join
from pprint import PrettyPrinter

from medset.brats import BraTSDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from configs.debug_config import CONFIG_AIR as DEBUG_CONFIG_AIR
from configs.debug_config import CONFIG_BASE as DEBUG_CONFIG_BASE
from configs.live_config import CONFIG_AIR, CONFIG_BASE
from src.training.lightning import UNetLightning
from src.utils import helper as hlp
from src.utils.helper import TENSOR_NET_TYPES
from src.utils.helper import log, set_dir

pp = PrettyPrinter()

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--type',
        help='Network type (base, cp, tt, tucker)',
        default='base',
        type=str,
    )
    parser.add_argument(
        '-c', '--comp', help='Compression rate (int)', default=1, type=int
    )
    parser.add_argument(
        '-f', '--fold', help='Fold index (0-4)', default=0, type=int
    )
    parser.add_argument(
        '-k',
        '--kernel',
        help='Kernel size for core block convolutions (default: 3)',
        default=3,
        type=int,
    )
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help='Use debug config',
        default=False,
    )
    parser.add_argument(
        '-w',
        '--widths',
        help='Width configuration to use',
        default=0,
        type=int,
    )
    parser.add_argument(
        '-ckpt',
        '--checkpoint',
        action='store_true',
        help='Start from checkpoint',
    )
    args = parser.parse_args()

    # Parameters
    net_type = args.type
    kernel_size = args.kernel
    widths = args.widths
    debug = args.debug
    checkpoint = args.checkpoint

    assert (
        net_type in TENSOR_NET_TYPES or net_type == 'base'
    ), f"Choose a valid network type ('base','cp', 'tucker', or 'tt')!"
    assert kernel_size in (
        3,
        5,
        7,
    ), f'Choose a valid kernel size (3, 5, 7), not <{kernel_size}>'
    assert widths in (
        0,
        1,
    ), f'Choose a valid widths setting (0, 1), not <{widths}>'

    widths = (
        (32, 64, 128, 256, 512) if widths == 0 else (48, 96, 192, 384, 768)
    )
    if debug:
        log('DEBUG MODE', title=True, color='red')

    # Baseline or tensorized?
    if net_type == 'base':

        # Use base configuration
        config = CONFIG_BASE if not debug else DEBUG_CONFIG_BASE
        compression = 1
        config['model']['network_params']['widths'] = widths

    else:

        # If we use a tensorized model, compression rate must be given
        assert (
            args.comp is not None
        ), 'Please enter a valid compression rate! (--COMP)'
        config = CONFIG_AIR if not debug else DEBUG_CONFIG_AIR

        # Extra parameters for AirUNet
        config['model']['network_params']['compression'] = compression = int(
            args.comp
        )
        config['model']['network_params']['tensor_net_type'] = net_type
        config['model']['network_params']['widths'] = widths

    # Change kernel size if supplied
    if kernel_size:

        # Adjust padding if necessary
        padding = int((kernel_size - 1) / 2)
        if (
            'core_block_conv_params'
            in config['model']['network_params'].keys()
        ):
            config['model']['network_params']['core_block_conv_params'][
                'kernel_size'
            ] = kernel_size
            config['model']['network_params']['core_block_conv_params'][
                'padding'
            ] = padding
        else:
            config['model']['network_params']['core_block_conv_params'] = {
                'kernel_size': kernel_size,
                'padding': padding,
            }

    # Log config
    log('CONFIG', color='red', timestamped=False, title=True)
    pp.pprint(config)

    # Let's go
    hlp.hi(
        'Training lightweight U-Net',
        data_dir=config['data']['root_dir'],
        log_dir=config['logs']['log_dir'],
    )

    version = f'{net_type}_comp{compression}_k{kernel_size}_w{widths}'

    fold_index = int(args.fold) if args.fold else 0
    model_name = f'unet_{net_type}_f{fold_index}'

    log(f'Network type: {args.type}', color='green')
    log(f'Compression rate: {args.comp}', color='green')
    log(f'Fold index: {args.fold}', color='green')

    # Set data directory
    data_dir = config['data']['root_dir']
    tb_dir = join(hlp.LOG_DIR, 'tb_logs')
    # snap_dir = join(hlp.LOG_DIR, 'snapshots', model_name)
    ckpt_dir = join(hlp.LOG_DIR, 'checkpoints', model_name)
    result_dir = join(hlp.LOG_DIR, 'results', model_name)
    set_dir(data_dir, tb_dir, ckpt_dir, result_dir)

    # Initialize model
    log(f'Initializing <{model_name}> model.')
    model = UNetLightning(**config['model'])

    # Initialize data module
    log('Initializing data module.')
    brats = BraTSDataModule(**config['data'])

    # Initialize logger
    config['training']['logger'] = TensorBoardLogger(
        save_dir=tb_dir,
        name=model_name,
        default_hp_metric=False,
        version=version,
    )

    # Prep checkpointing
    """config['training']['callbacks'].append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f'{model_name}_v{version}_fold{fold_index}',
            save_on_train_epoch_end=True,
            every_n_epochs=10,
        )
    )"""

    # Initialize trainer
    log('Initializing trainer.')
    trainer = Trainer(**config['training'], num_sanity_val_steps=0)

    # Set up fitting
    fit_params = {'model': model, 'datamodule': brats}

    # If we resume from checkpoint, add argument to fit method
    if checkpoint:
        checkpoint_path = join(
            ckpt_dir,
            f'{model_name}_v{version}_fold{fold_index}.ckpt',
        )
        log(
            f'Resuming from checkpoint (path={checkpoint_path}.',
            timestamped=False,
            color='blue',
        )
        fit_params['ckpt_path'] = checkpoint_path

    # Train!
    log('Commencing training.')
    trainer.fit(**fit_params)

    # Additional checkpoint (just in case)
    """trainer.save_checkpoint(
        join(snap_dir, f'{model_name}_v{version}_fold{fold_index}.ckpt')
    )"""

    # Test
    log('Evaluating model.')
    trainer.test(model=model, dataloaders=brats.val_dataloader())
    results = model.test_results

    # Save test results
    log('Saving results.')
    # np.save(file=join(result_dir, f'{model_name}_v{version}_fold{fold_index}.npy'), arr=results)
    torch.save(
        results,
        join(result_dir, f'{model_name}_v{version}_fold{fold_index}.pth'),
    )

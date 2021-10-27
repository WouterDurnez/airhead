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
import torch.cuda
from pl_bolts.callbacks import PrintTableMetricsCallback
from ptflops import get_model_complexity_info
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from layers.air_conv import AirDoubleConv
from models.air_unet import AirUNet
from training.data_module import BraTSDataModule
from training.inference import val_inference, test_inference
from training.lightning import UNetLightning
from training.losses import dice_loss, dice_metric, dice_et, dice_tc, dice_wt, hd_metric, hd_et, hd_tc, hd_wt
from utils import helper as hlp
from utils.helper import log
from utils.helper import set_dir

pp = PrettyPrinter()

if __name__ == '__main__':

    # Let's go
    hlp.hi("Training lightweight U-Net", log_dir='../logs_cv')

    # Get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', help='Tensor network type')
    parser.add_argument('--comp', help='Compression rate')
    parser.add_argument('--fold', help='Fold index')

    args = parser.parse_args()

    log(f'Tensor network type: {args.type}', color='green')
    log(f'Compression rate:    {args.comp}', color='green')
    log(f'Fold index:          {args.fold}', color='green')

    # Parameters
    tensor_net_type = args.type
    compression = version = int(args.comp)
    fold_index = int(args.fold)
    model_name = f'unet_{tensor_net_type}_f{fold_index}'

    # Set data directory
    data_dir = hlp.DATA_DIR
    tb_dir = join(hlp.LOG_DIR, 'tb_logs')
    snap_dir = join(hlp.LOG_DIR, 'snapshots', model_name)
    result_dir = join(hlp.LOG_DIR, 'results', model_name)
    set_dir(data_dir, tb_dir, snap_dir, result_dir)

    # Initialize model
    log(f"Initializing <{model_name}> model")
    model = UNetLightning(

        # Architecture settings
        network=AirUNet,
        network_params={
            'compression': compression,
            'tensor_net_type': tensor_net_type,
            'double_conv': AirDoubleConv,
            'double_conv_params': {'comp_friendly': True},
            'in_channels': 4,
            'out_channels': 3,
            'widths': (32, 64, 128, 256, 320),
            'head': False},

        # Loss and metrics
        loss=dice_loss,
        metrics=[dice_metric, dice_et, dice_tc, dice_wt,
                 hd_metric, hd_et, hd_tc, hd_wt],

        # Optimizer
        optimizer=optim.AdamW,
        optimizer_params={'lr': 1e-4, 'weight_decay': 1e-5},

        # Learning rate scheduler
        scheduler=CosineAnnealingWarmRestarts,
        scheduler_config={'interval': 'epoch'},
        scheduler_params={'T_0': 50, 'eta_min':3e-5},

        # Inference method
        inference=val_inference,
        inference_params=None,

        # Test inference method
        test_inference=test_inference,
        test_inference_params=None,
    )

    # Load checkpoint
    #model.load_from_checkpoint(checkpoint_path=join(snap_dir, f'final_{model_name}_v{version}_fold{fold_index}.ckpt'))

    # Initialize data module
    log("Initializing data module")
    brats = BraTSDataModule(data_dir=join(data_dir,"MICCAI_BraTS2020_TrainingData"),
                            num_workers=8,
                            batch_size=1,
                            fold_index=fold_index)
    brats.setup()

    # Initialize logger
    tb_logger = TensorBoardLogger(save_dir=tb_dir, name=model_name, default_hp_metric=False, version=version)

    # Initialize trainer
    log("Initializing trainer")
    trainer = Trainer(
        max_epochs=500,
        logger=tb_logger,
        gpus=-1,
        #num_nodes=1,
        deterministic=True,
        #distributed_backend='ddp',
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            PrintTableMetricsCallback(),
        ],
    )

    # Test
    log("Evaluating model")
    trainer.test(model=model,
                 ckpt_path=join(snap_dir, f'final_{model_name}_v{version}_fold{fold_index}.ckpt'),
                 datamodule=brats)
    results = model.test_results

    # Adding model parameters
    '''with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model=model, input_res=(4, 128, 128, 128), as_strings=True,
                                                 print_per_layer_stat=True, verbose=False)'''

    eval = {'model_name': model_name,
            'version': version,
            'results': results,
            'n_param': model.get_n_parameters()}

    # Save test results
    np.save(file=join(result_dir, f'{model_name}_v{version}_fold{fold_index}.npy'), arr=results)

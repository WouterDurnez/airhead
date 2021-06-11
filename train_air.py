#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Main training script for lightweight U-Net
"""

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
from models.air_unet import AirUNet
from training.lightning import UNetLightning
from training.data_module import BraTSDataModule
from training.inference import val_inference, test_inference
from training.losses import dice_loss, dice_metric, dice_et, dice_tc, dice_wt, hd_metric, hd_et, hd_tc, hd_wt
from utils import helper as hlp
from utils.helper import log
from utils.helper import set_dir
import argparse, sys
from layers.air_conv_alt import AirDoubleConvAlt

pp = PrettyPrinter()

if __name__ == '__main__':

    # Let's go
    hlp.hi("Training lightweight U-Net")

    # Get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', help='Tensor network type')
    parser.add_argument('--comp', help='Compression rate')

    args = parser.parse_args()

    log(f'Tensor network type: {args.type}', color='green')
    log(f'Compression rate:    {args.comp}', color='green')

    # Parameters
    tensor_net_type = args.type
    compression = int(args.comp)
    model_name = f'unet_{tensor_net_type}'
    version = compression

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
            'double_conv': AirDoubleConvAlt,
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

    # Initialize data module
    log("Initializing data module")
    brats = BraTSDataModule(data_dir=join(data_dir,"MICCAI_BraTS2020_TrainingData"),
                            num_workers=0,
                            batch_size=1,
                            validation_size=.2)
    brats.setup()

    # Initialize logger
    tb_logger = TensorBoardLogger(save_dir=tb_dir, name=model_name, default_hp_metric=False, version=version)

    # Initialize trainer
    log("Initializing trainer")
    trainer = Trainer(
        max_steps=100000,
        max_epochs=200,
        logger=tb_logger,
        gpus=2,
        #num_nodes=1,
        deterministic=True,
        distributed_backend='ddp',
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            PrintTableMetricsCallback(),
        ],
    )

    # Train
    log("Commencing training")
    trainer.fit(model=model,
                datamodule=brats)

    # Additional checkpoint (just in case)
    trainer.save_checkpoint(join(snap_dir, f'final_{model_name}_v{version}.ckpt'))

    # Test
    log("Evaluating model")
    trainer.test()
    results = model.test_results

    # Adding model parameters
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model=model, input_res=(4, 128, 128, 128), as_strings=True,
                                                 print_per_layer_stat=True, verbose=False)

    eval = {'model_name': model_name,
            'version': version,
            'results': results,
            'n_param': model.get_n_parameters(),
            'flops_count': macs,
            'params': params}

    # Save test results
    np.save(file=join(result_dir, f'{model_name}_v{version}.npy'), arr=results)

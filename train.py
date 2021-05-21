#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Main training script
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

from models.unet import UNet
from models.unet_lightning import UNetLightning
from training.data_module import BraTSDataModule
from training.inference import val_inference, test_inference
from training.losses import dice_loss, dice_metric, dice_et, dice_tc, dice_wt, hd_metric, hd_et, hd_tc, hd_wt
from utils import helper as hlp
from utils.helper import log
from utils.helper import set_dir
from utils.utils import WarmupCosineSchedule, TakeSnapshot

pp = PrettyPrinter()

if __name__ == '__main__':
    # Let's go
    hlp.hi("Training baseline U-Net")

    # Name
    model_name = 'unet_baseline'
    version = 1

    # Set data directory
    data_dir = hlp.DATA_DIR
    tb_dir = join(hlp.LOG_DIR, 'tb_logs')
    snap_dir = join(hlp.LOG_DIR, 'snapshots', model_name)
    result_dir = join(hlp.LOG_DIR, 'results', model_name)
    model_dir = join(hlp.LOG_DIR, 'models', model_name)
    set_dir(data_dir, tb_dir, snap_dir, result_dir, model_dir)

    # Initialize model
    log(f"Initializing <{model_name}> model")
    model = UNetLightning(

        # Architecture settings
        network=UNet,
        network_params={
            'in_channels': 4,
            'out_channels': 3,
            'widths': (32, 64, 128, 256, 320),
            'head': False},

        # Loss and metrics
        loss=dice_loss,
        metrics=[dice_metric, dice_et, dice_tc, dice_wt,
                 hd_metric, hd_et, hd_tc, hd_wt],

        # Optimizer
        optimizer=optim.AdamW,  # TODO: Why AdamW?
        optimizer_params={'lr': 1e-4, 'weight_decay': 1e-2},

        # Learning rate scheduler
        scheduler=WarmupCosineSchedule,
        scheduler_config={'interval': 'step'},
        #scheduler_params={'warmup_steps': 3*3e2, 'total_steps': 1e5},
        scheduler_params={'warmup_steps': 0, 'total_steps': 1e5},

        # Inference method
        inference=val_inference,
        inference_params=None,

        # Test inference method
        test_inference=test_inference,
        test_inference_params=None,
    )

    # Load checkpoint
    # model.load_from_checkpoint(checkpoint_path=join(snap_dir,'epoch=1.ckpt'))

    # Initialize data module
    log("Initializing data module")
    brats = BraTSDataModule(data_dir=join(data_dir,"MICCAI_BraTS2020_TrainingData"),
                            num_workers=8,
                            batch_size=1,
                            validation_size=.2)
    brats.setup()

    # Initialize logger
    tb_logger = TensorBoardLogger(save_dir=tb_dir, name=model_name, default_hp_metric=False, version=version)

    # Initialize trainer
    log("Initializing trainer")
    trainer = Trainer(
        max_steps=100000,
        max_epochs=100,
        logger=tb_logger,
        gpus=1,
        num_nodes=1,
        # distributed_backend='ddp',
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            PrintTableMetricsCallback(),
            #TakeSnapshot(epochs=(1, 24, 49), save_dir=snap_dir)
        ],
    )

    # Train
    log("Commencing training")
    trainer.fit(model=model,
                datamodule=brats)

    # Additional checkpoint (just in case)
    trainer.save_checkpoint(join(snap_dir, f'final_{model_name}_v{version}.ckpt'))

    # Test
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log("Evaluating model")
    trainer.test()
    results = model.test_results

    # Adding model parameters
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model=model, input_res=(4, 128, 128, 128), as_strings=True,
                                                 print_per_layer_stat=True, verbose=False)

    eval = {'results': results,
            'n_param': model.get_n_parameters(),
            'flops_count': macs,
            'params': params}

    # Save test results
    np.save(file=join(result_dir, f'{model_name}_v{version}.npy'), arr=results)

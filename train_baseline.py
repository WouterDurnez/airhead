#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Main training script
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
from medset.brats import BraTSDataModule as PooyaModule

from models.baseline_unet import UNet
from training.data_module import BraTSDataModule
from training.inference import val_inference, test_inference
from training.lightning import UNetLightning
from training.losses import dice_loss #dice_metric, dice_et, dice_tc, dice_wt, hd_metric, hd_et, hd_tc, hd_wt
from training.metrics import dice_metric, dice_et, dice_tc, dice_wt, hd_metric, hd_et, hd_tc, hd_wt
from utils import helper as hlp
from utils.helper import log
from utils.helper import set_dir
from utils.utils import WarmupCosineSchedule

pp = PrettyPrinter()

if __name__ == '__main__':
    # Let's go
    hlp.hi("Training baseline U-Net",
           log_dir='../logs_cv',
           data_dir='../data/')

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', help='Fold index')
    args = parser.parse_args()

    # Parameters
    version = 0
    fold_index = int(args.fold) if args.fold else 0
    model_name = f'unet_baseline_f{fold_index}'

    log(f'Model name:    {model_name}', color='green')
    log(f'Fold index:    {fold_index}', color='green')

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
        network=UNet,
        network_params={
            'in_channels': 4,
            'out_channels': 3,
            'widths': (32, 64, 128, 256, 320),
            'head': False},

        # Loss and metrics
        loss=dice_loss,
        metrics=[dice_metric, dice_et, dice_tc, dice_wt,
                 hd_et, hd_tc, hd_wt],

        # Optimizer
        optimizer=optim.AdamW,
        optimizer_params={'lr': 1e-4, 'weight_decay': 1e-2},

        # Learning rate scheduler
        scheduler=WarmupCosineSchedule,
        scheduler_config={'interval': 'step'},
        scheduler_params={"warmup_steps": 0, "total_steps": 100000},

        # Inference method
        inference=test_inference,
        inference_params={'overlap':.5},

        # Test inference method
        test_inference=test_inference,
        test_inference_params={'overlap':.5},
    )

    # Load checkpoint
    #model.load_from_checkpoint(checkpoint_path=join(snap_dir,f'final_{model_name}_v0.ckpt'))

    # Initialize data module
    log("Initializing data module")
    '''brats = PooyaModule(root_dir=join(hlp.DATA_DIR,'Task01_BrainTumour'),
                            num_workers=0,
                            batch_size=1,
                            split=fold_index,
                            cache_rate=0)'''
    brats = BraTSDataModule(data_dir=join(hlp.DATA_DIR, 'MICCAI_BraTS2020_TrainingData'))
    brats.setup(stage='fit')

    # Initialize logger
    tb_logger = TensorBoardLogger(save_dir=tb_dir, name=model_name, default_hp_metric=False, version=version)

    # Initialize trainer
    log("Initializing trainer")
    trainer = Trainer(
        max_epochs=500,
        logger=tb_logger,
        gpus=0,
        #num_nodes=8,
        deterministic=True,
        #distributed_backend='ddp',
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
    trainer.save_checkpoint(join(snap_dir, f'final_{model_name}_v{version}_fold{fold_index}.ckpt'))

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
    np.save(file=join(result_dir, f'{model_name}_v{version}_fold{fold_index}.npy'), arr=results)

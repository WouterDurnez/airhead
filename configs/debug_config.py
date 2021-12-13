#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Configuration for HPC training
"""

from copy import deepcopy

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch.optim import AdamW

from layers.air_conv import AirResBlock
from layers.base_layers import ResBlock
from models.air_unet import AirUNet
from models.base_unet import UNet
from training.inference import test_inference
from training.losses import dice_loss
from training.metrics import *
from utils.utils import WarmupCosineSchedule

GENERAL = {
    "model": {
        "loss": dice_loss,
        "metrics": [
            dice_metric,
            dice_et,
            dice_tc,
            dice_wt,
            hd_et,
            hd_tc,
            hd_wt,
        ],
        "optimizer": AdamW,
        "optimizer_params": {
            "lr": 1e-4,
            "weight_decay": 1e-2},
        "scheduler": WarmupCosineSchedule,
        "scheduler_config": {
            "interval": "step"},
        "scheduler_params": {
            "warmup_steps": 20,
            "total_steps": 180,
        },
        "inference": test_inference,
        "inference_params": {
            "overlap": 0.5},
        "test_inference": test_inference,
        "test_inference_params": {
            "overlap": 0.5},
    },
    "data": {
        "root_dir": "../data/Task01_BrainTumour",
        "spatial_size": (128, 128, 128),
        "num_splits": 5,
        # "split": 0,
        "batch_size": 1,
        "num_workers": 0,
        "cache_num": 4,
        "cache_rate": 1.0,
    },
    "training": {
        "max_epochs": 1,
        "gpus": 0,
        "num_nodes": 1,
        "strategy": DDPPlugin(find_unused_parameters=False),
        "deterministic": True,
        "callbacks": [
            LearningRateMonitor(logging_interval="step"),
            # ModelCheckpoint(every_n_train_steps=25),
            # PrintTableMetricsCallback(),
        ],
        "check_val_every_n_epoch": 25,
        "limit_train_batches":1,
        "limit_val_batches":2,
        "limit_test_batches":1,
    },
    "logs": {
        "log_dir": "../logs"},
}

BASE_MODEL = {
    "network": UNet,
    "network_params": {
        "core_block": ResBlock,
        "in_channels": 4,
        "out_channels": 3,
        "widths": (32, 64, 128, 256, 512),
        "head": False,
    },
}
AIR_MODEL = {
    "network": AirUNet,
    "network_params": {
        "core_block": AirResBlock,
        "core_block_conv_params": {
            "comp_friendly": True},
        "in_channels": 4,
        "out_channels": 3,
        "widths": (32, 64, 128, 256, 512),
        "head": False,
    },
}

CONFIG_BASE = deepcopy(GENERAL)
CONFIG_BASE["model"].update(BASE_MODEL)
CONFIG_AIR = deepcopy(GENERAL)
CONFIG_AIR["model"].update(AIR_MODEL)

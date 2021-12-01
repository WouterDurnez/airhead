from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim import AdamW
from copy import deepcopy
from layers.air_conv import AirDoubleConv
from models.air_unet import AirUNet
from models.baseline_unet import UNet
from training.inference import test_inference
from training.losses import dice_loss
from training.metrics import *
from utils.utils import WarmupCosineSchedule

GENERAL = {
    'model': {
        'loss': dice_loss,
        'metrics': [
            dice_metric,
            dice_et,
            dice_tc,
            dice_wt,
            hd_et,
            hd_tc,
            hd_wt
        ],
        'optimizer': AdamW,
        'optimizer_params': {
            'lr': 1e-4,
            'weight_decay': 1e-2
        },
        'scheduler': WarmupCosineSchedule,
        'scheduler_config': {
            'interval': 'step'
        },
        'scheduler_params': {
            'warmup_steps': 0,
            'total_steps': 100000
        },
        'inference': test_inference,
        'inference_params': {
            'overlap': .5
        },
        'test_inference': test_inference,
        'test_inference_params': {
            'overlap': .5
        }
    },
    'data': {
        'root_dir': '../data/Task01_BrainTumour',
        'num_workers': 0,
        'batch_size': 1,
        'cache_rate': .1
    },
    'training': {
        'max_epochs': 500,
        'gpus': 0,
        'deterministic': True,
        'callbacks': [
            LearningRateMonitor(logging_interval="step"),
            PrintTableMetricsCallback(),
        ],
    },
    'logs': {
        'log_dir': '../logs'
    }
}

BASE_MODEL = {
    'network': UNet,
    'network_params': {
        'in_channels': 4,
        'out_channels': 3,
        'widths': (32, 64, 128, 256, 320),
        'head': False},
}
AIR_MODEL = {
    'network': AirUNet,
    'network_params': {
        'double_conv': AirDoubleConv,
        'double_conv_params': {
            'comp_friendly': True
        },
        'in_channels': 4,
        'out_channels': 3,
        'widths': (32, 64, 128, 256, 320),
        'head': False},
}

CONFIG_BASE = deepcopy(GENERAL)
CONFIG_BASE['model'].update(BASE_MODEL)
CONFIG_AIR = deepcopy(GENERAL)
CONFIG_AIR['model'].update(AIR_MODEL)

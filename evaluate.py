#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Main evaluation script
"""
import os
from os import pardir
from os.path import join

import SimpleITK as sitk
import torch
from torch import optim

from utils import helper as hlp
from utils.helper import log
from utils.helper import set_dir
from models.unet import UNet
from models.unet_lightning import UNetLightning
from training.data_module import BraTSDataModule
from training.inference import val_inference, test_inference
from training.losses import dice_loss, dice_metric, dice_et, dice_tc, dice_wt
from utils.utils import WarmupCosineSchedule


def inference_write(
        model,
        datamodule,
        checkpoint_path,
        test_dir,
        write_dir=os.getcwd(),
        device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set datamodule
    datamodule.setup("test")
    test_dl = datamodule.test_dataloader

    # load model
    model = model.load_from_checkpoint(checkpoint_path)
    model.to(device)

    # predict and write
    model.eval()
    with torch.no_grad():
        for sample in test_dl:
            predict_single(model, sample, write_dir, device)


def predict_single(model, sample, write_dir, device):
    """Predict and write mask for a single image."""

    # get sample
    id_ = sample["id"]
    x = sample["input"].to(device)

    # get path
    path = os.path.join(write_dir, f"{id_}.nii.gz")

    if not os.path.isfile(path):
        # predict
        pred = test_inference(x, model)

        # write image as nii.gz
        pred = pred[0].astype("uint8").numpy()
        pred = sitk.GetImageFromArray(pred)
        sitk.WriteImage(pred, path)

        print(f"Subject {id_} done.")
    return None


if __name__ == '__main__':
    # Let's go
    hlp.set_params(data_dir='../data')
    hlp.hi("Training baseline UNet")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set data directory
    train_dir = join(hlp.DATA_DIR, 'MICCAI_BraTS2020_TrainingData')
    test_dir = join(hlp.DATA_DIR, 'MICCAI_BraTS2020_ValidationData')
    log_dir = set_dir(join(pardir, 'logs'))

    # Initialize model
    unet = UNetLightning(

        # Architecture settings
        network=UNet,
        network_params={
            'in_channels': 4,
            'out_channels': 3,
            'widths': (32, 64, 128, 256, 320),
            'head': False},

        # Loss and metrics
        loss=dice_loss,
        metrics=[dice_metric, dice_et, dice_tc, dice_wt],

        # Optimizer
        optimizer=optim.AdamW,  # TODO: Why AdamW?
        optimizer_params={'lr': 1e-4, 'weight_decay': 1e-2},

        # Learning rate scheduler
        scheduler=WarmupCosineSchedule,
        scheduler_config={'interval': 'step'},
        scheduler_params={'warmup_steps': 0, 'total_steps': 1e5},

        # Inference method
        inference=val_inference,
        inference_params=None
    )
    check_path = '/home/wouter/Documents/MAI/airhead/airhead/airhead/logs/unet_baseline/version_0/checkpoints/epoch=199-step=58999.ckpt'
    unet.load_from_checkpoint(checkpoint_path=check_path)

    unet = unet.to(device)

    # Initialize data module
    log("Initializing data module")
    brats = BraTSDataModule(data_dir=train_dir,
                            test_dir=test_dir,
                            num_workers=4,
                            batch_size=1,
                            validation_size=.2)
    brats.setup('test')
    test_dl = brats.test_dataloader()

    # Singles
    for idx, sample in enumerate(test_dl):

        # Predict one
        predict_single(model=unet, sample=sample, write_dir=join(log_dir, 'test'), device=device)

        if idx == 0: break

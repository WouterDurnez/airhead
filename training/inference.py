#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Inference functions
i.e. pass (unseen) input through the model and see what it says
"""
from itertools import product
from os.path import join

import SimpleITK as sitk
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from monai.transforms import Compose
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

import utils.helper as hlp
from models.baseline_unet import UNet
from training.data_module import BraTSDataModule
from training.lightning import UNetLightning
from training.losses import *
from training.metrics import dice_metric, dice_et, dice_tc, dice_wt, hd_et, hd_tc, hd_wt


####################################################
# Inference functions for validation and test data #
####################################################


def val_inference(input: torch.Tensor, model: nn.Module):
    """
    Inference function for validation data

    :param input: validation data tensor
    :param model: model through which input is passed
    :return: model output (segmentation)
    """

    # Generate output
    output = model(input)

    # Post transforms
    # * Sigmoid activation layer (if needed)
    # * Threshold the values to 0 or 1

    post_trans_list = [AsDiscrete(threshold=.5)]

    # If it's the right type of model, containing our 'head' parameter,
    # check if the head is there, otherwise add sigmoid as posttransform
    if hasattr(model, 'head'):
        if model.head is None:
            post_trans_list.insert(0, Activations(sigmoid=True))

    # Compose and apply
    post_trans = Compose(post_trans_list)
    output = post_trans(output)

    return output


def test_inference(input: torch.Tensor, model: nn.Module, roi_dim: int = 128, **kwargs):
    """
    Inference function for test data, using sliding window

    :param input: test data tensor
    :param model: model through which input is passed
    :return: model output (segmentation)
    """

    # Generate output using sliding window (sized 128^3)
    output = sliding_window_inference(
        inputs=input, roi_size=(roi_dim, roi_dim, roi_dim),
        sw_batch_size=1, predictor=model, **kwargs
    )

    # Post transforms
    # * Sigmoid activation layer (if needed))
    # * Threshold the values to 0 or 1

    post_trans_list = [AsDiscrete(threshold_values=True)]

    # If it's the right type of model, containing our 'head' parameter,
    # check if the head is there, otherwise add sigmoid as posttransform
    if hasattr(model, 'head'):
        if model.head is None:
            post_trans_list.insert(0, Activations(sigmoid=True))

    # Compose and apply
    post_trans = Compose(post_trans_list)
    output = post_trans(output)

    return output


##############
# Prediction #
##############

def predict(model: torch.nn.Module, sample: dict, device: torch.device, model_name: str, write_dir: str = None):
    if write_dir is None:
        write_dir = join(hlp.LOG_DIR, 'images', model_name)
        hlp.set_dir(write_dir)

    # Unpack
    # subject = sample['id']
    image = sample['input'].to(device)
    image = image.unsqueeze(0)

    # Make prediction
    prediction = test_inference(input=image, model=model).squeeze(0)

    # Channel order: ET, TC, WT

    return prediction


if __name__ == '__main__':
    hlp.hi('Prediction test', log_dir='../../logs_cv')

    # Set parameters
    fold_index = 0
    models = [('baseline', 0)]
    models += list(product(
        ('cpd', 'tt', 'tucker'),
        (2, 5, 10, 20, 35, 50, 75, 100)
    ))

    for type, version in tqdm(models, desc='Predicting segmentation'):
        model_name = f'unet_{type}_f{fold_index}'
        write_dir = join(hlp.DATA_DIR, 'predictions')
        hlp.set_dir(write_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
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
            optimizer_params={'lr': 1e-4, 'weight_decay': 1e-5},

            # Learning rate scheduler
            scheduler=CosineAnnealingWarmRestarts,
            scheduler_config={'interval': 'epoch'},
            scheduler_params={'T_0': 50, 'eta_min': 3e-5},

            # Inference method
            inference=val_inference,
            inference_params=None,

            # Test inference method
            test_inference=test_inference,
            test_inference_params={'overlap': .5},
        )

        # Load from checkpoint
        checkpoint_path = join(hlp.LOG_DIR, 'snapshots', model_name,
                               f'final_{model_name}_v{version}_fold{fold_index}.ckpt')

        model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)
        model = model.to(device)

        # Predict a sample
        brats = BraTSDataModule(data_dir=join(hlp.DATA_DIR, "MICCAI_BraTS2020_TrainingData"), num_workers=2)
        brats.setup('test')

        # Predict all samples
        '''for index, sample in tqdm(enumerate(brats.test_set), f"Predicting samples using {model_name} model"):
            print(index, sample['id'])
            if sample['id'] != 'BraTS20_Training_102':
                print('skip')
                continue'''

        sample = brats.test_set[16]

        with torch.no_grad():
            test = predict(model=model, sample=sample, model_name=model_name, device=device,
                           write_dir=write_dir).detach()
        test = sitk.GetImageFromArray(test.cpu().numpy())
        sitk.WriteImage(image=test, fileName=join(write_dir, f'{model_name}_v{version}_2.nii.gz'))

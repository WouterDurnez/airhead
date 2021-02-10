#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Inference functions
i.e. pass (unseen) input through the model and see what it says

-- Based on code by Pooya Ashtari
-- Adapted by Wouter Durnez
"""
from monai.transforms import Compose
from monai.transforms import Activations, AsDiscrete
from monai.inferers import sliding_window_inference
import torch
import os
import SimpleITK as sitk
import torch.nn as nn


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
    # * Sigmoid activation layer TODO: check if this is necessary (switched off for now)
    # * Threshold the values to 0 or 1
    post_trans = Compose(
        [
            #Activations(sigmoid=True),
            AsDiscrete(threshold_values=True)
        ]
    )

    # Apply post transforms
    output = post_trans(output)

    return output


def test_inference(input: torch.Tensor, model: nn.Module):
    """
    Inference function for test data, using sliding window

    :param input: test data tensor
    :param model: model through which input is passed
    :return: model output (segmentation)
    """

    # Generate output using sliding window (sized 128^3) TODO: why is this necessary? test data are resized by transforms?
    output = sliding_window_inference(
        inputs=input, roi_size=(128, 128, 128), sw_batch_size=4, predictor=model,
    )

    # Post transforms
    # * Sigmoid activation layer TODO: check if this is necessary (switched off for now)
    # * Threshold the values to 0 or 1
    post_trans = Compose(
        [
            Activations(sigmoid=True),
            AsDiscrete(threshold_values=True)
        ]
    )

    # Apply post transforms
    output = post_trans(output)
    return output

# TODO: process (understand, adapt) functions below

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

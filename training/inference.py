#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Inference functions
i.e. pass (unseen) input through the model and see what it says
"""
from monai.transforms import Compose
from monai.transforms import Activations, AsDiscrete
from monai.inferers import sliding_window_inference
import torch
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
    # * Sigmoid activation layer (if needed)
    # * Threshold the values to 0 or 1

    post_trans_list = [AsDiscrete(threshold_values=True)]

    # If it's the right type of model, containing our 'head' parameter,
    # check if the head is there, otherwise add sigmoid as posttransform
    if hasattr(model,'head'):
        if not model.head:
            post_trans_list.insert(0, Activations(sigmoid=True))

    # Compose and apply
    post_trans = Compose(post_trans_list)
    output = post_trans(output)

    return output


def test_inference(input: torch.Tensor, model: nn.Module):
    """
    Inference function for test data, using sliding window

    :param input: test data tensor
    :param model: model through which input is passed
    :return: model output (segmentation)
    """

    # Generate output using sliding window (sized 128^3)
    output = sliding_window_inference(
        inputs=input, roi_size=(128, 128, 128), sw_batch_size=2, predictor=model,
    )

    # Post transforms
    # * Sigmoid activation layer (if needed))
    # * Threshold the values to 0 or 1

    post_trans_list = [AsDiscrete(threshold_values=True)]

    # If it's the right type of model, containing our 'head' parameter,
    # check if the head is there, otherwise add sigmoid as posttransform
    if hasattr(model,'head'):
        if not model.head:
            post_trans_list.insert(0, Activations(sigmoid=True))

    # Compose and apply
    post_trans = Compose(post_trans_list)
    output = post_trans(output)

    return output



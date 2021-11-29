#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Metrics
-- Coded by Pooya Ashtari
"""

import warnings

import torch
from monai.metrics import HausdorffDistanceMetric
from torch import nn


###########
# Metrics #
###########

class DiceMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn("y_pred is not a binarized tensor here!")

        if not torch.all(y.byte() == y):
            raise ValueError("y should be a binarized tensor.")

        if y_pred.ndim < 3:
            raise ValueError("y_pred should have at least three dimensions.")

        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError("y_pred and y should have same shapes.")

        # Compute dice (BxC) for each channel for each batch
        # Reduce only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, y_pred.ndim))
        intersection = torch.sum(y * y_pred, dim=reduce_axis)
        ap = torch.sum(y, reduce_axis)
        pp = torch.sum(y_pred, dim=reduce_axis)
        denominator = ap + pp
        output = torch.where(
            denominator > 0,
            (2.0 * intersection) / denominator,
            torch.ones_like(intersection),
        )

        # Reduce batch and channel dimensions
        output = output.mean(dim=[0, 1])
        return output


class SensitivityMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn("y_pred is not a binarized tensor here!")

        if not torch.all(y.byte() == y):
            raise ValueError("y should be a binarized tensor.")

        if y_pred.ndim < 3:
            raise ValueError("y_pred should have at least three dimensions.")

        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError("y_pred and y should have same shapes.")

        # Compute dice (BxC) for each channel for each batch
        # Reduce only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, y_pred.ndim))
        tp = torch.sum(y * y_pred, dim=reduce_axis)
        ap = torch.sum(y, reduce_axis)
        output = torch.where(ap > 0, tp / ap, torch.ones_like(tp))

        # Reduce batch and channel dimensions
        output = output.mean(dim=[0, 1])
        return output


class SpecificityMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn("y_pred is not a binarized tensor here!")

        if not torch.all(y.byte() == y):
            raise ValueError("y should be a binarized tensor.")

        if y_pred.ndim < 3:
            raise ValueError("y_pred should have at least three dimensions.")

        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError("y_pred and y should have same shapes.")

        # Compute dice (BxC) for each channel for each batch
        # Reduce only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, y_pred.ndim))
        tn = torch.sum((1 - y) * (1 - y_pred), dim=reduce_axis)
        an = torch.sum((1 - y), reduce_axis)
        output = torch.where(an > 0, tn / an, torch.ones_like(tn))

        # Reduce batch and channel dimensions
        output = output.mean(dim=[0, 1])
        return output


def dice_metric(pred, true):
    dice = DiceMetric()
    return dice(pred, true)


def sens_metric(pred, true):
    sens = SensitivityMetric()
    return sens(pred, true)


def spec_metric(pred, true):
    spec = SpecificityMetric()
    return spec(pred, true)


def hd_metric(pred, true):
    hd = HausdorffDistanceMetric(include_background=True, percentile=95)
    return hd(pred, true)[0].squeeze().to(pred.dtype)


def dice_et(pred, true):
    """dice metric for enhancing tumor."""
    pred_et = pred[:, 0:1]
    true_et = true[:, 0:1]
    out = dice_metric(pred_et, true_et)
    return out


def dice_tc(pred, true):
    """dice metric for tumor core."""
    pred_tc = pred[:, 1:2]
    true_tc = true[:, 1:2]
    out = dice_metric(pred_tc, true_tc)
    return out


def dice_wt(pred, true):
    """dice metric for whole tumor."""
    pred_wt = pred[:, 2:3]
    true_wt = true[:, 2:3]
    out = dice_metric(pred_wt, true_wt)
    return out


def hd_et(pred, true):
    """Hausdorff distance for enhancing tumor."""
    pred_et = pred[:, 0:1]
    true_et = true[:, 0:1]
    out = hd_metric(pred_et, true_et)
    return out


def hd_tc(pred, true):
    """Hausdorff distance for tumor core."""
    pred_tc = pred[:, 1:2]
    true_tc = true[:, 1:2]
    out = hd_metric(pred_tc, true_tc)
    return out


def hd_wt(pred, true):
    """Hausdorff distance for whole tumor."""
    pred_wt = pred[:, 2:3]
    true_wt = true[:, 2:3]
    out = hd_metric(pred_wt, true_wt)
    return out


if __name__ == '__main__':
    # Quick test (currently no cuda support on my end)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create appropriately dimensioned tensor with random values
    dim = 128
    x = torch.randint(0, 2, (1, 4, dim, dim, dim))
    x.to(device)
    y = torch.randn(1, 4, dim, dim, dim)
    y.to(device)

    # Test metrics
    dm = dice_metric(y, x)
    # hdm = hd_metric(x,y)

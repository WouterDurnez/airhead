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
from typing import Optional, Union

import numpy as np
import torch
from monai.metrics import CumulativeIterationMetric
from monai.metrics.utils import (
    do_metric_reduction,
    ignore_background,
    get_mask_edges,
    get_surface_distance,
)
from monai.utils import MetricReduction
from torch import nn


###########
# Metrics #
###########


class DiceMetric(CumulativeIterationMetric):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
    in ``monai.transforms.post`` first to achieve binarized values.
    The `include_background` parameter can be set to ``False`` for an instance of DiceLoss to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background so excluding it in such cases helps convergence.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Args:
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to ``True``.
        include_zero_masks: whether to compute Dice for zero channels in the ground truth. Defaults to ``True``.
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result. Defaults to ``"mean"``.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(
        self,
        include_background: bool = True,
        include_zero_masks: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.include_zero_masks = include_zero_masks
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        if not isinstance(y_pred, torch.Tensor) or not isinstance(
            y, torch.Tensor
        ):
            raise ValueError('y_pred and y must be PyTorch Tensor.')

        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn('y_pred should be a binarized tensor.')

        if not torch.all(y.byte() == y):
            raise ValueError('y should be a binarized tensor.')

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError('y_pred should have at least three dimensions.')

        # compute dice (BxC) for each channel for each batch
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError('y_pred and y should have same shapes.')

        # reduce only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, y_pred.ndim))
        intersection = torch.sum(y * y_pred, dim=reduce_axis)
        ap = torch.sum(y, reduce_axis)
        pp = torch.sum(y_pred, dim=reduce_axis)
        denominator = ap + pp

        if self.include_zero_masks:
            output = torch.where(
                denominator > 0,
                (2.0 * intersection) / denominator,
                torch.ones_like(intersection),
            )
        else:
            output = torch.where(
                ap > 0,
                (2.0 * intersection) / denominator,
                torch.tensor(float('nan'), device=ap.device),
            )

        return output

    def aggregate(self):  # type: ignore
        """Execute reduction logic for the dices (BxC)."""

        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError('the data to aggregate must be PyTorch Tensor.')

        # do metric reduction
        f, not_nans = do_metric_reduction(data, self.reduction)
        return (f, not_nans) if self.get_not_nans else f


class HausdorffDistanceMetric(CumulativeIterationMetric):
    """
    Compute Hausdorff Distance between two tensors. It can support both multi-classes and multi-labels tasks.
    It supports both directed and non-directed Hausdorff distance calculation. In addition, specify the `percentile`
    parameter can get the percentile of the distance. Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
    The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

    Args:
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result. Defaults to ``"mean"``.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(
        self,
        include_background: bool = False,
        include_zero_masks: bool = True,
        distance_metric: str = 'euclidean',
        percentile: Optional[float] = None,
        directed: bool = False,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.include_zero_masks = include_zero_masks
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.directed = directed
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        if not isinstance(y_pred, torch.Tensor) or not isinstance(
            y, torch.Tensor
        ):
            raise ValueError('y_pred and y must be PyTorch Tensor.')

        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn('y_pred should be a binarized tensor.')

        if not torch.all(y.byte() == y):
            raise ValueError('y should be a binarized tensor.')

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError('y_pred should have at least three dimensions.')

        # compute (BxC) for each channel for each batch
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        if isinstance(y, torch.Tensor):
            y = y.float()

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError('y_pred and y should have same shapes.')

        batch_size, n_class = y_pred.shape[:2]
        hd = np.empty((batch_size, n_class))
        for b, c in np.ndindex(batch_size, n_class):
            (edges_pred, edges_gt) = get_mask_edges(y_pred[b, c], y[b, c])
            if not np.any(edges_gt):
                warnings.warn(
                    f'the ground truth of class {c} is all 0, this may result in nan/inf distance.'
                )
            if not np.any(edges_pred):
                warnings.warn(
                    f'the prediction of class {c} is all 0, this may result in nan/inf distance.'
                )

            distance_1 = self.compute_percent_hausdorff_distance(
                edges_pred, edges_gt, self.distance_metric, self.percentile
            )
            if self.directed:
                hd[b, c] = distance_1
            else:
                distance_2 = self.compute_percent_hausdorff_distance(
                    edges_gt, edges_pred, self.distance_metric, self.percentile
                )
                hd[b, c] = max(distance_1, distance_2)
        return torch.from_numpy(hd)

    def aggregate(self):  # type: ignore
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError('the data to aggregate must be PyTorch Tensor.')

        # do metric reduction
        f, not_nans = do_metric_reduction(data, self.reduction)
        return (f, not_nans) if self.get_not_nans else f

    def compute_percent_hausdorff_distance(
        self,
        edges_pred: np.ndarray,
        edges_gt: np.ndarray,
        distance_metric: str = 'euclidean',
        percentile: Optional[float] = None,
    ):
        """
        This function is used to compute the directed Hausdorff distance.
        """

        surface_distance = get_surface_distance(
            edges_pred, edges_gt, distance_metric=distance_metric
        )

        # for both pred and gt do not have foreground
        if surface_distance.shape == (0,):
            out = 0.0 if self.include_zero_masks else np.nan
            return out

        if not percentile:
            return surface_distance.max()

        if 0 <= percentile <= 100:
            return np.percentile(surface_distance, percentile)

        raise ValueError(
            f'percentile should be a value between 0 and 100, get {percentile}.'
        )


class SensitivityMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn('y_pred is not a binarized tensor here!')

        if not torch.all(y.byte() == y):
            raise ValueError('y should be a binarized tensor.')

        if y_pred.ndim < 3:
            raise ValueError('y_pred should have at least three dimensions.')

        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError('y_pred and y should have same shapes.')

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
            warnings.warn('y_pred is not a binarized tensor here!')

        if not torch.all(y.byte() == y):
            raise ValueError('y should be a binarized tensor.')

        if y_pred.ndim < 3:
            raise ValueError('y_pred should have at least three dimensions.')

        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError('y_pred and y should have same shapes.')

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
    return torch.mean(dice(pred, true))


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
    y = torch.zeros(1, 4, dim, dim, dim)
    y.to(device)

    # Test metrics
    # dm = dice_metric(y, x)
    hdm = hd_et(y, x)

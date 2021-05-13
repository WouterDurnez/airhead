#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Loss functions
"""

from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric

##########
# Losses #
##########

# Regular dice loss
loss = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

def dice_loss(pred, true):
    return loss(pred, true)

# Soft dice loss: add 1 to numerator and denominator to avoid 0
soft_loss = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_dr=1, smooth_nr=1)

def soft_dice_loss(pred,true):
    return soft_loss(pred,true)


###########
# Metrics #
###########


# Dice #
########

metric = DiceMetric(include_background=True, reduction='mean')

def dice_metric(pred, true):
    return metric(pred, true)[0]

def dice_et(pred, true):
    """dice metric for enhancing tumor."""
    pred_et = pred[:, 0]
    true_et = true[:, 0]
    dice_et = dice_metric(pred_et, true_et)
    return dice_et

def dice_tc(pred, true):
    """dice metric for tumor core."""
    pred_tc = pred[:, 1]
    true_tc = true[:, 1]
    dice_tc = dice_metric(pred_tc, true_tc)
    return dice_tc

def dice_wt(pred, true):
    """dice metric for whole tumor."""
    pred_wt = pred[:, 2]
    true_wt = true[:, 2]
    dice_wt = dice_metric(pred_wt, true_wt)
    return dice_wt


# Hausdorff #
#############

metric_hd = HausdorffDistanceMetric(include_background=True, reduction='mean')

def hd_metric (pred, true):
    return metric_hd(pred,true)[0]

def hd_et(pred, true):
    """Hausdorff metric for enhancing tumor."""
    pred_et = pred[:, 0]
    true_et = true[:, 0]
    hd_et = hd_metric(pred_et, true_et)
    return hd_et

def hd_tc(pred, true):
    """Hausdorff metric for tumor core."""
    pred_tc = pred[:, 1]
    true_tc = true[:, 1]
    hd_tc = hd_metric(pred_tc, true_tc)
    return hd_tc

def hd_wt(pred, true):
    """Hausdorff metric for whole tumor."""
    pred_wt = pred[:, 2]
    true_wt = true[:, 2]
    hd_wt = hd_metric(pred_wt, true_wt)
    return hd_wt
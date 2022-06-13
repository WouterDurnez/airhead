#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Loss functions
"""

import torch
from monai.losses import DiceLoss

##########
# Losses #
##########

# Regular dice loss
loss = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)


def dice_loss(pred, true):
    return loss(pred, true)


# Soft dice loss: add 1 to numerator and denominator to avoid 0
soft_loss = DiceLoss(
    to_onehot_y=False,
    sigmoid=True,
    squared_pred=True,
    smooth_dr=1,
    smooth_nr=1,
)


def soft_dice_loss(pred, true):
    return soft_loss(pred, true)


if __name__ == '__main__':
    # Quick test (currently no cuda support on my end)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create appropriately dimensioned tensor with random values
    dim = 128
    x = torch.randint(0, 2, (1, 4, dim, dim, dim))
    x.to(device)
    y = torch.randint(0, 2, (1, 4, dim, dim, dim))
    y.to(device)

    # Test metrics
    dm = dice_loss(x, y)
    hdm = soft_dice_loss(x, y)

#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Transforms (common image transformations) for use in training, validation, testing and visualisation
"""

import numpy as np
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Resized,
    CropForegroundd,
    RandFlipd,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ToTensord,
    EnsureChannelFirstD, EnsureChannelFirstd, ConvertToMultiChannelBasedOnBratsClassesd, Spacingd, Orientationd
)


##########
# Custom #
##########

class OneHotEncoder(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    (see https://www.med.upenn.edu/cbica/brats2020/data.html)

    class 1: ET (label 4) -- enhancing tumor
    class 2: NCR+NET (label 1) --  necrotic and non-enhancing tumor core (label 3 eliminated since 2017)
    class 3: ED (label 2) -- edema

    If nested (DEFAULT!):

    class 1: ET (label 4) -- enhancing tumor
    class 2: TC (label 4 + 1) -- tumor core = enhancing + non-enhancing + necrotic
    class 3: WT: (label 4 + 1 + 2) -- whole tumor = tumor core + edema
    """

    def __init__(self, nested=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nested = nested

    def __call__(self, data):

        # Nested structure
        if self.nested:

            # Make sure the data has the dict structure
            d = dict(data)

            # Loop over keys in data (which are also attributes to the super class)
            for key in self.keys:

                # Store results in list
                result = [

                    # ET (label 4)
                    d[key] == 4,

                    # TC: ET (label 4) + NCR&NET (label 1)
                    np.logical_or(d[key] == 1, d[key] == 4),

                    # WT: ET (label 4) + NCR&NET (label 1) + ED (label 2)
                    np.logical_or(
                        np.logical_or(d[key] == 1, d[key] == 2), d[key] == 4
                    )
                ]

                d[key] = np.stack(result, axis=0).astype(np.float32)

        # Disjunct zones
        else:
            d = dict(data)
            for key in self.keys:
                result = [
                    d[key] == 1,  # NCR&NET (label 1)
                    d[key] == 4,  # ET (label 4)
                    d[key] == 2,  # ED (label 2)
                ]

                d[key] = np.stack(result, axis=0).astype(np.float32)

        return d


################
# Compositions #
################

# Training transforms (including augmentation)
class EnsureTyped:
    pass


def get_train_transform(patch_dim:int = 128):
    """
    DATA PREPARATION
    * read data
    * apply one-hot-encoding
    * crop the foreground (i.e. crop brain out of volume)
    DATA AUGMENTATION
    * randomly crop a 128x128x128 volume out of image
    * random affine transformation (geometric transformation that preserves lines and parallelism
    * randomly scale intensity
    * randomly shift intensity
    DATA PREPARATION (CTD)
    * normalize intensity ((X - mu)/sigma)
        -- zero remains zero
        -- calculate mus and sigmas for channels separately (T1, T2, etc)
    * convert to tensor
    """

    transforms = [
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        OneHotEncoder(keys="label"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[128, 128, 128],
            random_size=False,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "label"],
            spatial_size=[patch_dim, patch_dim, patch_dim],
            prob=0.5,
            rotate_range=10,
            mode=("bilinear", "nearest"),
            as_tensor_output=False,
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ]
    training_transform = Compose(transforms)

    return training_transform


# Validation transforms
def get_val_transform(patch_dim:int = 128):
    """
    DATA PREPARATION
    * read data
    * apply one-hot-encoding
    * crop the foreground (i.e. crop brain out of volume)
    * resize image to 128x128x128 volume
    * normalize intensity ((X - mu)/sigma)
        -- zero remains zero
        -- calculate mus and sigmas for channels separately (T1, T2, etc)
    * convert to tensor
    """

    transforms = [
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        OneHotEncoder(keys="label"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(
            keys=["image", "label"],
            spatial_size=[patch_dim, patch_dim, patch_dim],
            mode=("trilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ]
    validation_transform = Compose(transforms)
    return validation_transform


# Test transforms
# (leave image mostly intact! no cropping or resizing since we don't have label data)
def get_test_transform():
    """
    DATA PREPARATION
    * read data
    * normalize intensity ((X - mu)/sigma)
        -- zero remains zero
        -- calculate mus and sigmas for channels separately (T1, T2, etc)
    * convert to tensor
    """

    transforms = [
        LoadImaged(keys=["image","label"], reader="ITKReader"),
        OneHotEncoder(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ]
    test_transform = Compose(transforms)
    return test_transform


# Visualization transforms
def get_vis_transform():
    """
    DATA PREPARATION
    * read data
    * add label channel
    * crop the foreground (i.e. crop brain out of volume)
    * convert to tensor
    """
    transforms = [
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        AddChanneld(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
    visualisation_transforms = Compose(transforms)
    return visualisation_transforms

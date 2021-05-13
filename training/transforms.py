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

        # Differential zones
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
def get_train_transform():
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
        LoadImaged(keys=["input", "target"], reader="ITKReader"),
        OneHotEncoder(keys="target"),
        CropForegroundd(keys=["input", "target"], source_key="input"),
        RandSpatialCropd(
            keys=["input", "target"],
            roi_size=[128, 128, 128],
            random_size=False,
        ),
        RandFlipd(keys=["input", "target"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["input", "target"],
            spatial_size=[128, 128, 128],
            prob=0.5,
            rotate_range=10,
            mode=("bilinear", "nearest"),
            as_tensor_output=False,
        ),
        RandScaleIntensityd(keys="input", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="input", offsets=0.1, prob=0.5),
        NormalizeIntensityd(keys="input", nonzero=True, channel_wise=True),
        ToTensord(keys=["input", "target"]),
    ]
    training_transform = Compose(transforms)
    return training_transform


# Validation transforms
def get_val_transform():
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
        LoadImaged(keys=["input", "target"], reader="ITKReader"),
        OneHotEncoder(keys="target"),
        CropForegroundd(keys=["input", "target"], source_key="input"),
        Resized(
            keys=["input", "target"],
            spatial_size=[128, 128, 128],
            mode=("trilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="input", nonzero=True, channel_wise=True),
        ToTensord(keys=["input", "target"]),
    ]
    validation_transform = Compose(transforms)
    return validation_transform


# Test transforms
# (leave image mostly intact! no cropping or resizing since we don't have target data)
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
        LoadImaged(keys="input", reader="ITKReader"),
        NormalizeIntensityd(keys="input", nonzero=True, channel_wise=True),
        ToTensord(keys="input"),
    ]
    validation_transform = Compose(transforms)
    return validation_transform


# Visualization transforms
def get_vis_transform():
    """
    DATA PREPARATION
    * read data
    * add target channel
    * crop the foreground (i.e. crop brain out of volume)
    * convert to tensor
    """
    transforms = [
        LoadImaged(keys=["input", "target"], reader="ITKReader"),
        AddChanneld(keys=["target"]),
        CropForegroundd(keys=["input", "target"], source_key="input"),
        ToTensord(keys=["input", "target"]),
    ]
    validation_transform = Compose(transforms)
    return validation_transform

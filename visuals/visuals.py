#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Visualization functions
"""
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import utils.helper as hlp
from utils.helper import set_dir, log
from training.data_module import BraTSDataModule
import numpy as np
import seaborn as sns
import SimpleITK as sitk
from os.path import join

def show_subject(sample:dict, axis:int = 0, slice: int = 100):

    # Do we have predictions?
    p = 'prediction' in sample

    img = np.array(sample['input']).take(indices=slice, axis=axis+1)
    msk = np.array(sample['target'][0]).take(indices=slice, axis=axis)
    if p:
        pre = np.array(sample['prediction']).take(indices=slice, axis=axis)

    # Viz parameters
    channels = ['T1', 'T1 post-contrast', 'T2', 'FLAIR']

    # Set some aesthetic parameters
    sns.set_theme(font_scale=1.3, font='serif')

    # Make plot
    rows = 3 if p else 2
    fig, ax = plt.subplots(rows, 4, dpi=300, figsize=(20, 10))

    # Define color map
    cmap_mask = ListedColormap(['none', 'red', 'green', 'yellow'])

    # Add labels
    ax[0,0].set_ylabel('Image')
    ax[1,0].set_ylabel('Target')
    if p:
        ax[2,0].set_ylabel('Prediction')

    # Plot all images and masks
    for index, channel in enumerate(channels):
        # Top row without masks
        ax[0, index].imshow(img[index, ...], cmap='gray')
        ax[0, index].set_title(channel, fontweight='bold')
        # Second row with target masks
        ax[1, index].imshow(img[index, ...], cmap='gray')
        ax[1, index].imshow(msk, alpha=.7, cmap=cmap_mask)
        # Bottom row with predicted masks
        if p:
            #ax[2, index].imshow(img[index, ...], cmap='gray')
            ax[2, index].imshow(pre, alpha=.7, cmap=cmap_mask)

    plt.setp(ax, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    plt.suptitle(f'BraTS dataset - subject {sample["id"]}', fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Let's go
    hlp.hi("Visualizing BraTS")

    model_name = 'unet_baseline'
    version = 0

    # Set data directory
    vis_dir = join(hlp.LOG_DIR, 'images', model_name)
    pred_dir = join(hlp.DATA_DIR, 'predictions')
    set_dir(vis_dir)

    # Initialize data module
    log("Initializing data module")
    brats = BraTSDataModule(data_dir=join(hlp.DATA_DIR, "MICCAI_BraTS2020_TrainingData"),
                            num_workers=8,
                            batch_size=1,
                            validation_size=.2)
    brats.setup(stage='visualize')

    # Get an image
    idx = 3  # High grade ganglioma 3, Low-grade ganglioma 319
    sample = brats.visualization_set[idx]

    # Get prediction
    file_name = f'pred_{sample["id"]}.nii.gz'
    prediction = sitk.ReadImage(join(pred_dir, model_name, file_name))
    prediction = sitk.GetArrayFromImage(prediction)
    prediction = np.sum(prediction, axis=0)
    sample['prediction'] = prediction

    # Plot
    show_subject(sample=sample, slice=30)


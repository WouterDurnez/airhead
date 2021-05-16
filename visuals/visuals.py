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

from os.path import join

def show_subject(image: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor, subject:int = 3, axis:int = 0, slice: int = 100):

    img = np.array(image).take(indices=slice, axis=axis+1)
    msk = np.array(target).take(indices=slice, axis=axis)
    pre = np.array(prediction).take(indices=slice, axis=axis)

    # Viz parameters
    channels = ['T1', 'T1 post-contrast', 'T2', 'FLAIR']

    # Set some aesthetic parameters
    sns.set_theme(font_scale=1.3, font='serif')

    # Make plot
    fig, ax = plt.subplots(2, 4, dpi=300, figsize=(20, 10))

    # Define color map
    cmap_mask = ListedColormap(['black', 'red', 'green', 'yellow'])

    # Add labels
    ax[0,0].set_ylabel('Image')
    ax[1,0].set_ylabel('Target')

    # Plot all images and masks
    for index, channel in enumerate(channels):
        # Top row without masks
        ax[0, index].imshow(img[index, ...], cmap='gray')
        ax[0, index].set_title(channel, fontweight='bold')
        # Bottom row with masks
        ax[1, index].imshow(img[index, ...], cmap='gray')
        ax[1, index].imshow(msk, alpha=.7, cmap=cmap_mask)

    plt.setp(ax, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    plt.suptitle(f'BraTS dataset - subject {subject}', fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Let's go
    hlp.hi("Visualizing BraTS")

    model_name = 'unet_baseline'
    version = 0

    # Set data directory
    # hlp.set_params(data_dir='../../../data/MICCAI_BraTS2020_TrainingData')
    hlp.set_params(data_dir=join(hlp.DATA_DIR, 'MICCAI_BraTS2020_TrainingData'))
    vis_dir = join(hlp.LOG_DIR, '', model_name)
    set_dir(vis_dir)

    # Initialize data module
    log("Initializing data module")
    brats = BraTSDataModule(data_dir=hlp.DATA_DIR,
                            num_workers=8,
                            batch_size=1,
                            validation_size=.2)
    brats.setup(stage='visualize')

    # Get an image
    subject = 3  # High grade ganglioma 3, Low-grade ganglioma 319
    sample = brats.visualization_set[subject]

    # Permute?

    # Get data
    image = sample['input']  # .permute(0,3,2,1)
    mask = sample['target'][0]  # .permute(2,1,0) # only one channel
    #mask = np.ma.masked_where(mask > 0, mask)

    show_subject(image,mask,mask)

    # Viz parameters
    '''channels = ['T1', 'T1 post-contrast', 'T2', 'FLAIR']
    slice = 100

    # Plot
    fig, ax = plt.subplots(2, 4, dpi=300, figsize=(20, 10))
    cmap_mask = ListedColormap(['black', 'red', 'green', 'yellow'])
    for index, channel in enumerate(channels):
        # Top row without masks
        ax[0, index].imshow(image[index, slice, :, :], cmap='gray')
        ax[0, index].set_title(channel, fontweight='bold')
        ax[0, index].get_xaxis().set_visible(False)
        ax[0, index].get_yaxis().set_visible(False)
        # Bottom row with masks
        ax[1, index].imshow(image[index, slice, :, :], cmap='gray')
        ax[1, index].imshow(mask[slice, :, :], alpha=.7, cmap=cmap_mask)
        ax[1, index].get_xaxis().set_visible(False)
        ax[1, index].get_yaxis().set_visible(False)

    plt.suptitle(f'BraTS dataset - subject {subject}', fontweight='bold')
    plt.tight_layout()
    plt.show()'''
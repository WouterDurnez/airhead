#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Visualization functions
"""
from itertools import product
from os.path import join

import SimpleITK as sitk
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap

import src.utils.helper as hlp
from medset.brats import BraTSDataModule
from src.utils.helper import set_dir, log

#colors = sns.color_palette('Reds', n_colors=3)
#colors = ('red','green','yellow')
colors = [ "#FF7251", "#52BEEC", "#FFCE51",]

if __name__ == '__main__':
    # Let's go
    hlp.hi("Visualizing BraTS")

    # Define all model combinations
    models = [('baseline',0)]
    types = ('cpd','tt','tt2','tucker')
    compressions = (2,5,10,20,50,100)
    models += list(product(types,compressions))

    # Set data directory
    pred_dir = join(hlp.DATA_DIR, 'predictions')
    vis_dir = join(hlp.DATA_DIR, 'visuals')
    set_dir(vis_dir)

    # Initialize data module
    log("Initializing data module")
    brats = BraTSDataModule(data_dir=join(hlp.DATA_DIR, "MICCAI_BraTS2020_TrainingData"),
                            num_workers=8,
                            batch_size=1)
    brats.setup()

    # Get an image
    idx = 16      # Sample id: 'BraTS20_Training_102'
    sample = brats.visualization_set[idx]

    # Make sure we'll store the types
    for t in types:
        sample[t] = {}

    # Get predictions
    for type, comp in models:
        file_name = f'unet_{type}_f0_v{comp}_2.nii.gz'
        prediction = sitk.ReadImage(join(pred_dir, file_name))
        prediction = sitk.GetArrayFromImage(prediction)
        prediction_new = np.zeros(shape=prediction.shape[1:])

        '''
        Remember we used this scheme:
        class 1: ET (label 4) -- enhancing tumor
        class 2: TC (label 4 + 1) -- tumor core = enhancing + non-enhancing + necrotic
        class 3: WT: (label 4 + 1 + 2) -- whole tumor = tumor core + edema
        '''

        # Nested, so first fill in WT, then TC, then ET
        prediction_new[prediction[2] == 1] = 2
        prediction_new[prediction[1] == 1] = 1
        prediction_new[prediction[0] == 1] = 4

        # Crop foreground (as we did in visual transforms)
        start = sample['foreground_start_coord']
        end = sample['foreground_end_coord']
        prediction_new = prediction_new[
                         start[0]:end[0],
                         start[1]:end[1],
                         start[2]:end[2]]

        # Add cleaned up prediction to sample dict
        if type=='baseline':
            sample['baseline'] = torch.tensor(prediction_new)
        else:
            sample[type][comp] = torch.tensor(prediction_new)

    # Plot #
    ########

    # Viz parameters
    channels = ['T1', 'T1 post-contrast', 'T2', 'FLAIR']
    alpha = 1
    axis = 0
    slice = 85

    # Prep raw input and target
    img = np.array(sample['input']).take(indices=slice, axis=axis + 1)
    tgt = np.array(sample['target'][0]).take(indices=slice, axis=axis)

    '''tgt[tgt == 1.] = 0
    tgt[tgt == 2.] = 0
    tgt[tgt == 1.] = 0'''

    # Prep model predictions
    predictions = {
        type: {} for type in types
    }
    predictions['baseline'] = np.array(sample['baseline']).take(indices=slice, axis=axis)
    for type, comp in models:
        predictions[type][comp] = np.array(sample[type][comp]).take(indices=slice, axis=axis)

    # Set some aesthetic parameters
    sns.set_theme(font_scale=1, font='Utopia')

    # Make plot 1 #
    ##############

    fig, ax = plt.subplots(5, 6, dpi=300, figsize=(12, 12))

    # Define color map
    cmap_mask = ListedColormap(['none', colors[0], colors[1], colors[2]])

    # Add labels
    ax[0, 0].set_ylabel('Images', fontdict={'weight': 'bold'})
    ax[1, 0].set_ylabel('Target', fontdict={'weight': 'bold'})
    #ax[2, 0].set_ylabel('Prediction', fontdict={'weight': 'bold'})

    # Plot all images and masks
    for index, channel in enumerate(channels):
        # Top row, first 4 columns, without masks
        ax[0, index].imshow(img[index, ...], cmap='gray')
        ax[0, index].set_title(channel, fontweight='bold')

    # Top row, column 5, ground truth
    ax[0, 4].imshow(img[0, ...], alpha=.5, cmap='gray')
    ax[0, 4].imshow(tgt, alpha=alpha, cmap=cmap_mask)
    ax[0, 4].set_ylabel('Target', fontweight='bold')

    # Top row, column 6, baseline prediction
    ax[0, 5].imshow(img[0, ...], alpha=.5, cmap='gray')
    ax[0, 5].imshow(predictions['baseline'], alpha=alpha, cmap=cmap_mask)
    ax[0, 5].set_ylabel('Baseline', fontweight='bold')

    # Show tensor model predictions for all compression rates
    for row_index, type in enumerate(types, 1):
        for col_index, comp in enumerate(compressions):
            pretty_type = {
                'tt': 'Tensor train (v1)',
                'tt2': 'Tensor train (v2)',
                'cpd': 'Canonical polyadic',
                'tucker': 'Tucker'
            }

            ax[row_index, 0].set_ylabel(pretty_type[type], fontdict={'weight': 'bold'})
            ax[row_index, col_index].imshow(img[0, ...], alpha=.5, cmap='gray')
            ax[row_index, col_index].imshow(predictions[type][comp], alpha=alpha, cmap=cmap_mask)

            if row_index == 4:
                ax[row_index,col_index].set_xlabel(comp, fontdict={'weight': 'bold'})

    plt.subplots_adjust(hspace=0)
    plt.setp(ax, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    #plt.suptitle('test', fontweight='bold')

    # plt.legend(handles=[mpatches.Patch(color=col, label=lab) for col, lab in
    #                    zip((colors[0], colors[2], colors[1]), ('ET', 'TC', 'WT'))])

    #plt.tight_layout()

    plt.savefig(join(vis_dir, 'prediction_overview.pdf'),bbox_inches='tight', pad_inches=0)
    plt.show()

    # Make plot 2 #
    ##############

    fig, ax = plt.subplots(1, 5, dpi=300, figsize=(16, 4))
    sns.set_theme(context='paper',font_scale=2, font='Utopia')

    # Define color map

    # Add labels
    ax[0].set_ylabel('Images', fontdict={'weight': 'bold'})

    # Plot all images and masks
    for index, channel in enumerate(channels):
        # Top row, first 4 columns, without masks
        ax[index].imshow(img[index, ...], cmap='gray')
        ax[index].set_title(channel, fontweight='bold')

    # Top row, column 5, ground truth
    ax[4].imshow(img[0, ...], alpha=.5, cmap='gray')
    ax[4].imshow(tgt, alpha=alpha, cmap=cmap_mask)
    ax[4].set_ylabel('Target', fontweight='bold')

    plt.subplots_adjust(hspace=0)
    plt.setp(ax, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    #plt.suptitle('test', fontweight='bold')

    plt.legend(handles=[mpatches.Patch(color=col, label=lab) for col, lab in
                        zip((colors[0], colors[2], colors[1]), ('ET', 'TC', 'WT'))],
               bbox_to_anchor=(1.05, 1), loc='upper left',)

    plt.tight_layout()

    plt.savefig(join(vis_dir, 'dataset_overview.pdf'),bbox_inches='tight', pad_inches=0)
    plt.show()

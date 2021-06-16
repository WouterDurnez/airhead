import numpy as np

from os import walk
from os.path import join

import numpy as np
from glob import glob
import utils.helper as hlp
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process_results(results: dict, type:str, compression:int, macs:int, params:int):
    """Calculate mean of main metrics"""
    dice = []
    dice_et = []
    dice_tc = []
    dice_wt = []
    hd = []
    hd_et = []
    hd_tc = []
    hd_wt = []

    for res in results:
        dice.append(res['test_dice_metric'])
        dice_et.append(res['test_dice_et'])
        dice_tc.append(res['test_dice_tc'])
        dice_wt.append(res['test_dice_wt'])
        hd.append(res['test_hd_metric'])
        hd_et.append(res['test_hd_et'])
        hd_tc.append(res['test_hd_tc'])
        hd_wt.append(res['test_hd_wt'])

    return {

        'type': type,
        'compression': compression,

        'macs':macs,
        'params':params,

        'dice': np.mean(dice),
        'dice_et': np.mean(dice_et),
        'dice_tc': np.mean(dice_tc),
        'dice_wt': np.mean(dice_wt),

        'hd': np.mean(hd),
        'hd_et': np.mean(hd_et),
        'hd_tc': np.mean(hd_tc),
        'hd_wt': np.mean(hd_wt),
    }


if __name__ == '__main__':
    hlp.hi('Analysis', log_dir='../../logs_hpc')

    # Get all results files
    files = glob(join(hlp.LOG_DIR, 'results','*','*'))
    files = [f for f in files if f.endswith('.npy')]

    # Get flops and params
    flops_params = np.load(join(hlp.LOG_DIR, 'model_flops.npy'), allow_pickle=True).item()

    # Process all results
    processed_results = {}
    for f in files:
        if f.__contains__('baseline'):
            name = type = 'baseline'
            compression=1
            macs = flops_params['baseline']['macs']
            params = flops_params['baseline']['params']
        else:
            name = f.split('/')[-1][:-4]
            compression = int(name.split('_')[-1][1:])
            type = name.split('_')[-2]
            macs = flops_params[type][compression]['macs']
            params = flops_params[type][compression]['params']

        results = np.load(f, allow_pickle=True)
        processed_results[name] = process_results(results, type=type, compression=compression, macs=macs, params=params)

    # Create df
    metrics = pd.DataFrame.from_dict(processed_results,orient='index').\
        reset_index().\
        rename(columns={'index':'model'})

    # Add some ratio variables
    metrics['dice_macs'] = metrics.dice/metrics.macs
    metrics['dice_params'] = metrics.dice/metrics.params

    # Visualize
    metrics_plot = metrics.loc[metrics.type != 'baseline']
    sns.set_theme(context='talk', style='whitegrid', palette='pastel',rc={'lines.markersize':10})

    which_metric='dice'

    ############################
    # Dice vs macs/compression #
    ############################

    fig, ax = plt.subplots(1,2,figsize=(15,8), sharey=True)
    ax[0].axhline(float(metrics.loc[metrics.type == 'baseline'][which_metric]), color='gray', ls = '--', label='baseline')
    sns.lineplot(data=metrics_plot, y=which_metric, x='macs', hue='type', hue_order=['cp','tucker','tt'],
                 style='type', markers=True, dashes=True, ax=ax[0], legend='brief')
    #ax[0].axvline(float(metrics.loc[metrics.type == 'baseline'].macs), color='gray', ls='--')
    plt.ylim(.3,.8)
    ax[0].set_title('Dice vs. macs', fontdict={'weight':'bold'})

    ax[1].axhline(float(metrics.loc[metrics.type == 'baseline'][which_metric]), color='gray', ls='--', label='baseline')
    sns.lineplot(data=metrics_plot, y=which_metric, x='compression', hue='type', hue_order=['cp', 'tucker', 'tt'],
                 style='type', markers=True, dashes=True, ax=ax[1], legend='brief')
    #ax[0].axvline(float(metrics.loc[metrics.type == 'baseline'].macs), color='gray', ls='--')
    ax[1].set_title('Dice vs. compression', fontdict={'weight':'bold'})
    plt.ylim(.3, .8)

    plt.tight_layout
    plt.show()

    ######################
    # Dice/mac trade-off #
    ######################

    fig, ax = plt.subplots(figsize=(8,6))
    sns.lineplot(data=metrics_plot, y='dice_macs', x='compression', hue='type', hue_order=['cp', 'tucker', 'tt'],
                 style='type', markers=True, dashes=True, ax=ax)
    plt.title('Dice/macs trade-off x compression', fontdict={'weight':'bold'})
    plt.axhline(float(metrics.loc[metrics.type == 'baseline'].dice_macs), label='baseline', c='gray', ls='--')
    plt.legend()
    sns.despine(top=True, right=True, left=True, bottom=True)
    plt.tight_layout()
    plt.show()

    ####################
    # Baseline metrics #
    ####################

    fig, ax = plt.subplots(1,2, figsize=(10,6))
    base_data = metrics[metrics.type == 'baseline'].T.reset_index().rename(columns={'index':'metric', 18:'value'})
    dice_data = base_data.loc[base_data.metric.str.startswith('dice')].iloc[:-2,:]
    hd_data = base_data.loc[base_data.metric.str.startswith('hd')]
    sns.barplot(data=dice_data, x='metric', y='value', ax=ax[0])
    sns.barplot(data=hd_data, x='metric', y='value', ax=ax[1])
    ax[0].set_ylim(.5,.8)
    plt.tight_layout()
    plt.show()
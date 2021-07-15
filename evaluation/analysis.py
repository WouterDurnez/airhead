#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Initial processing of metrics, production of metrics dataframe some visuals
"""
import math
from glob import glob
from os.path import join

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.helper as hlp
from utils.helper import KUL_PAL

kul_pal = sns.color_palette(KUL_PAL)


def process_results(results: dict, type: str, fold: int, compression: int, macs: int, params: int):
    """Calculate mean of main metrics"""

    all_data = []

    for res in results:
        subject_data = {
            'Format': type,
            'Compression': compression,
            'fold': fold,
            'Macs': macs,
            'Parameters': params,
            'id': res['id'],
            'Dice': res['test_dice_metric'],
            'Dice ET': res['test_dice_et'],
            'Dice TC': res['test_dice_tc'],
            'Dice WT': res['test_dice_wt'],
            'Hausdorff': res['test_hd_metric'],
            'Hausdorff ET': res['test_hd_et'],
            'Hausdorff TC': res['test_hd_tc'],
            'Hausdorff WT': res['test_hd_wt'],
        }

        subject_df = pd.Series(subject_data)
        all_data.append(subject_df)

    all_df = pd.DataFrame(all_data)

    return all_df


if __name__ == '__main__':
    hlp.hi('Analysis', log_dir='../../logs_cv')
    vis_dir = join(hlp.DATA_DIR, 'visuals')
    rebuild = False
    # Get all results files
    files = glob(join(hlp.LOG_DIR, 'results', '*', '*'))
    files = [f for f in files if f.endswith('.npy')]

    # Get flops and params
    #flops_params = np.load(join(hlp.LOG_DIR, 'model_flops.npy'), allow_pickle=True).item()
    flops_params = np.load('model_flops.npy', allow_pickle=True).item()

    if rebuild:
        # Process all results
        processed_results = []
        for f in files:
            if f.__contains__('baseline'):
                model_name = type = 'baseline'
                compression = 1
                fold = int(f[-5])
                macs = flops_params['baseline']['macs']
                params = flops_params['baseline']['params']
            else:
                model_name = f.split('/')[-1]
                compression = int(model_name.split('_')[-2][1:])
                fold = int(model_name.split('_')[-1][4])
                type = model_name.split('_')[1]
                if type == 'cpd':
                    type = 'cp'
                macs = flops_params[type][compression]['macs']
                params = flops_params[type][compression]['params']

            results = np.load(f, allow_pickle=True)
            results_df = process_results(results, type=type, fold=fold, compression=compression, macs=macs, params=params)

            processed_results.append(results_df)

        # Create df
        metrics = pd.concat(processed_results)

        # Remap type names
        metrics.Format.replace({'cp': 'Canonical polyadic',
                                'tt': 'Tensor train (v1)',
                                'tt2': 'Tensor train (v2)',
                                'tucker': 'Tucker'}, inplace=True)

        # Save df
        metrics.to_pickle('metrics.pkl')
    else:
        metrics = pd.read_pickle('metrics.pkl')

    print(metrics.groupby('Format').Dice.agg(['mean', 'std']))

    #########
    # Plots #
    #########

    # PLOT 1 - DICE AND HAUSDORFF
    #############################

    sns.set_theme(context='paper', font_scale=1.2, style='white', palette=kul_pal)
    fig, ax = plt.subplots(4,2, figsize=(12,12), dpi=300,sharex=True)
    x_var = 'Macs'

    for idx, measure in enumerate(('Dice','Dice ET','Dice TC','Dice WT', 'Hausdorff', 'Hausdorff ET','Hausdorff TC','Hausdorff WT')):
        row_idx = idx%4
        col_idx = idx//4

        baseline_mean = metrics.loc[metrics.Format == 'baseline'][measure].mean()
        baseline_sd = metrics.loc[metrics.Format == 'baseline'][measure].std()
        n = len(metrics.loc[metrics.Format == 'baseline'])
        low, up = baseline_mean - 1.96 * baseline_sd / math.sqrt(n), baseline_mean + 1.96 * baseline_sd / math.sqrt(n)

        sns.lineplot(data=metrics.loc[metrics.Format != 'baseline'], x=x_var, y=measure, err_style='bars',
                     style='Format', markers=True, markersize=10, dashes=True, ci=95,
                     hue_order=['Canonical polyadic', 'Tensor train (v1)', 'Tensor train (v2)', 'Tucker'], hue='Format',
                     ax=ax[row_idx,col_idx])

        ax[row_idx,col_idx].axhline(baseline_mean, xmin=.02, xmax=1.02, linestyle='--', color='#DD8A2E', label='Baseline (mean)')
        ax[row_idx,col_idx].axhspan(ymin=low, ymax=up, xmin=.02, xmax=1.02, alpha=.1, color='#DD8A2E', label='Baseline (95% CI)')

        ax[row_idx,col_idx].set_xlabel(x_var, fontdict={'weight': 'bold'})
        ax[row_idx,col_idx].set_ylabel(measure, fontdict={'weight': 'bold'})
        if row_idx == 0 and col_idx==1:
            legend = plt.legend(title='Format', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(ax[row_idx,col_idx].get_legend().get_title(), fontweight='bold')
        else:
            ax[row_idx,col_idx].get_legend().remove()
        sns.despine(left=True, bottom=True)
    plt.tight_layout()

    plt.savefig(join(vis_dir, 'tensor_metrics.pdf'),bbox_inches='tight', pad_inches=0)

    plt.show()

    # PLOT 2 - BASELINE
    ###################

    base_data = metrics[metrics.Format == 'baseline']
    dice_data = base_data[[col for col in base_data if col.startswith('Dice')]]
    hd_data = base_data[[col for col in base_data if col.startswith('Haus')]]

    dice_plot = dice_data.melt(value_vars=['Dice', 'Dice ET', 'Dice WT', 'Dice TC'], var_name='Metric',
                               value_name='Score')
    hd_plot = hd_data.melt(value_vars=['Hausdorff', 'Hausdorff ET', 'Hausdorff WT', 'Hausdorff TC'], var_name='Metric',
                           value_name='Score')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    sns.set_theme(context='paper', font_scale=1.2, style='white', palette=kul_pal)

    sns.boxplot(data=dice_plot, x='Metric', y='Score', ax=ax[0], order=['Dice', 'Dice ET', 'Dice TC', 'Dice WT'])
    sns.boxplot(data=hd_plot, x='Metric', y='Score', ax=ax[1],
                order=['Hausdorff', 'Hausdorff ET', 'Hausdorff TC', 'Hausdorff WT'])
    ax[0].set_xlabel('Dice score', fontdict={'weight': 'bold', 'size':14})
    ax[1].set_xlabel('Hausdorff distance', fontdict={'weight': 'bold','size':14})
    ax[0].set_ylabel(None)
    ax[1].set_ylabel(None)

    plt.legend(title='Region',handles=[mpatches.Patch(color=col, label=lab) for col, lab in
                        zip(kul_pal, ('average', 'ET', 'TC', 'WT'))],
               bbox_to_anchor=(1.05, 1), loc='upper left', )
    plt.setp(ax[1].get_legend().get_title(), fontweight='bold')
    ax[0].axes.xaxis.set_ticks([])
    ax[1].axes.xaxis.set_ticks([])

    # ax[0].set_ylim(.5,1)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(vis_dir, 'baseline_metrics.pdf'),bbox_inches='tight', pad_inches=0)
    plt.show()

    # PLOT 3 - MAC AND PARAMS COMPARISON
    ####################################

    metrics_plot = metrics.loc[metrics.Format!='baseline']
    #metrics_plot = metrics
    sns.set_theme(context='paper', font_scale=1.2, style='white', palette=kul_pal)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    sns.lineplot(data=metrics_plot, x='Compression', y='Parameters', ax=ax[0], palette=kul_pal)
    sns.lineplot(data=metrics_plot, x='Compression', y='Macs', ax=ax[1], palette=kul_pal)
    #plt.legend(title='Region',handles=[mpatches.Patch(color=col, label=lab) for col, lab in
    #                    zip(kul_pal, ('average', 'ET', 'TC', 'WT'))],
    #           bbox_to_anchor=(1.05, 1), loc='upper left', )
    #plt.setp(ax[1].get_legend().get_title(), fontweight='bold')
    #ax[0].axes.xaxis.set_ticks([])
    #ax[1].axes.xaxis.set_ticks([])
    plt.show()
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
from utils.helper import KUL_PAL, KUL_PAL2
import torch

kul_pal = sns.color_palette(KUL_PAL, as_cmap=True)
kul_pal2 = sns.color_palette(KUL_PAL2, as_cmap=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def add_macs(row: pd.Series, mac_dict: dict) -> int:
    type = row['type']
    if type == 'cpd':
        type = 'cp'
    compression = row['compression']
    return mac_dict[type][compression]['macs'] if type != 'base' else mac_dict[type]['macs']


def add_params(row: pd.Series, mac_dict:dict) -> int:

    type = row['type']
    if type == 'cpd':
        type='cp'
    compression = row['compression']
    return mac_dict[type][compression]['params'] if type != 'base' else mac_dict[type]['params']


if __name__ == '__main__':

    hlp.hi('Analysis', log_dir='../../logs/')
    vis_dir = join(hlp.DATA_DIR, 'visuals')
    rebuild = True

    # Get all results files
    files = glob(join(hlp.LOG_DIR, 'results', '*', '*'))
    files = [f for f in files if f.endswith('.pth')]

    # Get flops and params
    flops_params = np.load(join(hlp.LOG_DIR,'flops', 'model_flops.npy'), allow_pickle=True).item()
    flops_params['base'] = flops_params.pop('baseline')

    # Get results from test output
    base_files = [f for f in files if f.__contains__('base')]
    tensor_files = [f for f in files if f not in base_files]

    # Gather all data frames here
    dfs = []

    # First gather base files
    for file in files:

        # Build dict
        result_dict = {}

        # Parse parameters from file name
        params = file.split('/')[-1].split('.')[-2].split('_')
        result_dict['fold'] = int(params[-1][-1])
        result_dict['kernel_size'] = int(params[-3][-1])
        result_dict['type'] = params[1]
        result_dict['compression'] = int(params[4][4:])
        result_dict['widths']= params[6][1:]

        # Get key-tensor pairs, flatten, and convert to data frame
        values = torch.load(file, map_location='cpu')
        for key in values.keys():
            result_dict[key] = values[key].flatten().numpy()
        dfs.append(pd.DataFrame(result_dict))

    # Merge data frames
    metrics = pd.concat(dfs).reset_index(drop=True)

    # Add macs and parameter counts
    metrics['macs'] = metrics.apply(lambda row: add_macs(row, mac_dict=flops_params), axis=1)
    metrics['params'] = metrics.apply(lambda row: add_params(row, mac_dict=flops_params), axis=1)

    metrics.rename(columns={'type':'Format','compression':'Compression',
                            'params':'Parameters', 'macs': 'MACs'}, inplace=True)
    metrics.replace({'cpd': 'Canonical polyadic', 'tt': 'Tensor train', 'tucker':'Tucker', 'base': 'Baseline'}, inplace=True)
    metrics.rename(columns={'test_dice_et': 'Dice ET',
                     'test_dice_tc': 'Dice TC',
                     'test_dice_wt': 'Dice WT',
                     'test_hd_et': '95% Hausdorff ET',
                     'test_hd_tc': '95% Hausdorff TC',
                     'test_hd_wt': '95% Hausdorff WT'}, inplace=True)

    # PLOT 1 - MAC AND PARAMS COMPARISON
    ####################################

    metrics_plot = metrics.loc[metrics.Format != 'Baseline']
    sns.set_theme(context='paper', font_scale=1.3, style='whitegrid', palette=kul_pal)
    base_vals = metrics.loc[metrics.Format == 'Baseline'].iloc[0]
    base_macs = base_vals['MACs']
    base_params = base_vals['Parameters']

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    ax[0].axhline(base_params, linestyle='--', linewidth=2, color='#DD8A2E', label='Baseline')
    ax[1].axhline(base_macs, linestyle='--', linewidth=2, color='#DD8A2E', label='Baseline')

    sns.barplot(data=metrics_plot, x='Compression', hue='Format', y='Parameters',
                hue_order=['Canonical polyadic', 'Tensor train', 'Tucker'],
                ax=ax[0], palette=kul_pal, ci=None)
    sns.barplot(data=metrics_plot, x='Compression', hue='Format', y='MACs', ax=ax[1],
                hue_order=['Canonical polyadic', 'Tensor train', 'Tucker'],
                palette=kul_pal, ci=None)

    # plt.legend(title='Region',handles=[mpatches.Patch(color=col, label=lab) for col, lab in
    #                    zip(kul_pal, ('average', 'ET', 'TC', 'WT'))],
    #           bbox_to_anchor=(1.05, 1), loc='upper left', )
    # plt.setp(ax[1].get_legend().get_title(), fontweight='bold')
    ax[0].axes.xaxis.set_ticklabels([2, 5, 10, 20])
    ax[1].axes.xaxis.set_ticklabels([2, 5, 10, 20])
    ax[0].set_xlabel('Layer compression', fontdict={
        'weight': 'bold',
        'size': 14})
    ax[1].set_xlabel('Layer compression', fontdict={
        'weight': 'bold',
        'size': 14})
    ax[0].set_ylabel('Network parameters', fontdict={
        'weight': 'bold',
        'size': 14})
    ax[1].set_ylabel('MACs', fontdict={
        'weight': 'bold',
        'size': 14})

    ax[0].get_legend().remove()
    # plt.legend(title='Format', handles=[mpatches.Patch(color=col, label=lab) for col, lab in
    #                                    zip(['#DD8A2E'] + kul_pal, ['Baseline','Canonical polyadic', 'Tensor train (v1)', 'Tensor train (v2)', 'Tucker'])],)
    # bbox_to_anchor=(1.05, 1), loc='upper left', )
    plt.setp(ax[1].get_legend().get_title(), fontweight='bold')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(vis_dir, 'macs_params.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()


    # PLOT 2 - DICE AND HAUSDORFF
    #############################

    metrics['Network compression'] = base_params/metrics['Parameters']
    metrics['MAC improvement'] = metrics['MACs']/base_macs

    '''metrics.groupby(['Format', 'Compression']).agg(['mean', 'std'])[
        ['Dice ET', 'Dice TC', 'Dice WT', '95% Hausdorff ET', '95% Hausdorff TC', '95% Hausdorff WT',
         'Network compression','MAC improvement']].to_csv('dice_hausdorff.csv')'''

    for x_var in ('Compression',): #, 'Network compression','MACs'):

        sns.set_theme(context='paper', font_scale=1.4, style='ticks', palette=kul_pal)
        fig, ax = plt.subplots(3, 2, figsize=(14, 14), dpi=300, sharex=False)
        z_val = 1.96 #1.37  # 1.96

        for idx, measure in enumerate(('Dice ET','Dice TC','Dice WT', '95% Hausdorff ET','95% Hausdorff TC','95% Hausdorff WT')):
            row_idx = idx%3
            col_idx = idx//3

            baseline_mean = metrics.loc[metrics.Format == 'Baseline'][measure].mean()
            baseline_sd = metrics.loc[metrics.Format == 'Baseline'][measure].std()
            n = len(metrics.loc[metrics.Format == 'Baseline'])
            low, up = baseline_mean - z_val * baseline_sd / math.sqrt(n), baseline_mean + z_val * baseline_sd / math.sqrt(n)

            ax[row_idx,col_idx].axhline(baseline_mean, xmin=.02, xmax=1.02, linestyle='--', color='#FB7E3A', label='Baseline')
            ax[row_idx,col_idx].axhspan(ymin=low, ymax=up, xmin=.02, xmax=1.02, alpha=.1, color='#FB7E3A')

            sns.lineplot(data=metrics.loc[metrics.Format != 'Baseline'], x=x_var, y=measure, err_style='bars',
                         style='Format', markers=True, markersize=10, dashes=True, ci=95,
                         hue_order=['Canonical polyadic', 'Tensor train', 'Tucker',], hue='Format',
                         ax=ax[row_idx,col_idx])
            if x_var == 'MACs':
                ax[row_idx,col_idx].set_xscale('log')
                ax[row_idx,col_idx].axvline(base_macs,0,linestyle='--', color='#FB7E3A')
                #ax[row_idx,col_idx].invert_xaxis()

            ax[row_idx,col_idx].set_ylabel(measure, fontdict={'weight': 'bold'})
            if x_var == 'Compression': ax[row_idx,col_idx].set_xticks([2,5,10,20])
            if row_idx<2:
                ax[row_idx, col_idx].set_xlabel('')
            else:
                ax[row_idx, col_idx].set_xlabel('Layer compression' if x_var == 'Compression' else x_var, fontdict={'weight': 'bold'})
            '''if row_idx == 0 and col_idx==1:
                legend = plt.legend(title='Format', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.setp(ax[row_idx,col_idx].get_legend().get_title(), fontweight='bold')
            else:
                pass #ax[row_idx,col_idx].get_legend().remove()
            sns.despine(left=True, bottom=True)'''
        plt.tight_layout()

        plt.savefig(join(vis_dir, f'tensor_metrics_{x_var.lower()}.pdf'),bbox_inches='tight', pad_inches=0)

        plt.show()

    # PLOT 2 - BASELINE
    ###################

    """base_data = metrics[metrics.Format == 'baseline']
    dice_data = base_data[[col for col in base_data if col.startswith('Dice')]]
    hd_data = base_data[[col for col in base_data if col.startswith('95% Haus')]]

    base_aggs = base_data.agg(['mean', 'std','median'])

    dice_plot = dice_data.melt(value_vars=['Dice ET', 'Dice WT', 'Dice TC'], var_name='Metric',
                               value_name='Score')
    hd_plot = hd_data.melt(value_vars=['95% Hausdorff ET', '95% Hausdorff WT', '95% Hausdorff TC'], var_name='Metric',
                           value_name='Score')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    sns.set_theme(context='paper', font_scale=1.3, style='whitegrid', palette=['#E7580C','#FB7E3A','#FBAF3A'])

    sns.boxplot(data=dice_plot, x='Metric', y='Score', ax=ax[0], order=['Dice ET', 'Dice TC', 'Dice WT'])
    sns.boxplot(data=hd_plot, x='Metric', y='Score', ax=ax[1],
                order=['95% Hausdorff ET', '95% Hausdorff TC', '95% Hausdorff WT'])
    ax[0].set_xlabel('Dice score', fontdict={'weight': 'bold', 'size':14})
    ax[1].set_xlabel('95% Hausdorff distance', fontdict={'weight': 'bold','size':14})
    ax[0].set_ylabel(None)
    ax[1].set_ylabel(None)

    plt.legend(title='Region',handles=[mpatches.Patch(color=col, label=lab) for col, lab in
                        zip(['#E7580C','#FB7E3A','#FBAF3A'], ('ET', 'TC', 'WT'))],
               bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10 )
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

    #metrics.groupby(['Format', 'Compression']).mean()[['Parameters', 'Macs']]
    metrics.groupby(['Format', 'Compression']).mean()[['Parameters', 'MACs','MAC improvement']].pivot_table(index=['Compression'],columns='Format').to_csv('macs_params_table.csv', decimal=',')

    metrics_plot = metrics.loc[metrics.Format!='baseline']
    #metrics_plot = metrics
    sns.set_theme(context='paper', font_scale=1.3, style='whitegrid', palette=kul_pal)
    base_vals = metrics.loc[metrics.Format=='baseline'].iloc[0]
    base_macs = base_vals['MACs']
    base_params = base_vals['Parameters']

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    ax[0].axhline(base_params, linestyle='--', linewidth=2, color='#DD8A2E', label='Baseline')
    ax[1].axhline(base_macs, linestyle='--', linewidth=2, color='#DD8A2E', label='Baseline')

    sns.barplot(data=metrics_plot, x='Compression', hue='Format', y='Parameters',
                hue_order=['Canonical polyadic', 'Tensor train (v1)', 'Tensor train (v2)', 'Tucker'],
                ax=ax[0], palette=kul_pal, ci=None)
    sns.barplot(data=metrics_plot, x='Compression', hue='Format', y='MACs', ax=ax[1],
                hue_order=['Canonical polyadic', 'Tensor train (v1)', 'Tensor train (v2)', 'Tucker'],
                palette=kul_pal, ci=None)


    #plt.legend(title='Region',handles=[mpatches.Patch(color=col, label=lab) for col, lab in
    #                    zip(kul_pal, ('average', 'ET', 'TC', 'WT'))],
    #           bbox_to_anchor=(1.05, 1), loc='upper left', )
    #plt.setp(ax[1].get_legend().get_title(), fontweight='bold')
    ax[0].axes.xaxis.set_ticklabels([2,5,10,20,35,50,75,100])
    ax[1].axes.xaxis.set_ticklabels([2,5,10,20,35,50,75,100])
    ax[0].set_xlabel('Layer compression', fontdict={'weight': 'bold', 'size':14})
    ax[1].set_xlabel('Layer compression', fontdict={'weight': 'bold','size':14})
    ax[0].set_ylabel('Network parameters', fontdict={'weight': 'bold', 'size':14})
    ax[1].set_ylabel('MACs', fontdict={'weight': 'bold','size':14})

    ax[0].get_legend().remove()
    #plt.legend(title='Format', handles=[mpatches.Patch(color=col, label=lab) for col, lab in
    #                                    zip(['#DD8A2E'] + kul_pal, ['Baseline','Canonical polyadic', 'Tensor train (v1)', 'Tensor train (v2)', 'Tucker'])],)
               #bbox_to_anchor=(1.05, 1), loc='upper left', )
    plt.setp(ax[1].get_legend().get_title(), fontweight='bold')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(vis_dir, 'macs_params.pdf'),bbox_inches='tight', pad_inches=0)
    plt.show()"""
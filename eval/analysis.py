from os.path import join

import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import torch
from setuptools.glob import glob
from tqdm import tqdm

import src.utils.helper as hlp
from src.utils.helper import KUL_PAL, KUL_PAL2

kul_pal = sns.color_palette(KUL_PAL, as_cmap=True)
kul_pal2 = sns.color_palette(KUL_PAL2, as_cmap=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
tqdm.pandas()


####################
# Helper functions #
####################


def gather_results(dir: str):
    files = glob(join(dir, '*', '*'))
    files = [f for f in files if f.endswith('.pth')]

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
        result_dict['comp'] = int(params[4][4:])
        result_dict['widths'] = (
            0 if params[6][1:] == '(32, 64, 128, 256, 512)' else 1
        )

        # Get key-tensor pairs, flatten, and convert to data frame
        values = torch.load(file, map_location='cpu')
        for key in values.keys():
            result_dict[key] = values[key].flatten().numpy()
        dfs.append(pd.DataFrame(result_dict))

    # Merge data frames
    metrics = pd.concat(dfs).reset_index(drop=True)
    metrics.replace({'cpd': 'cp'}, inplace=True)

    """metrics.rename(
        columns={
            'test_dice_et': 'Dice ET',
            'test_dice_tc': 'Dice TC',
            'test_dice_wt': 'Dice WT',
            'test_hd_et': '95% Hausdorff ET',
            'test_hd_tc': '95% Hausdorff TC',
            'test_hd_wt': '95% Hausdorff WT',
        },
        inplace=True,
    )"""

    metrics.to_csv('results/df_results.csv', index=False)

    return metrics


if __name__ == '__main__':
    # Let's go
    hlp.hi('Analysis', log_dir='../../logs')

    # Get mac/param counts
    macs_params = pd.read_csv(
        join(hlp.LOG_DIR, 'model_params', 'df_param_counts.csv')
    )

    # Merge results
    results = gather_results(join(hlp.LOG_DIR, 'results'))

    # Build metrics data frame
    metrics = pd.merge(
        results, macs_params, on=['comp', 'type', 'widths', 'kernel_size']
    )

    # PLOT 1 - MAC AND PARAMS COMPARISON
    ####################################

    metrics_plot = metrics.loc[metrics.type != 'base']
    sns.set_theme(
        context='paper', font_scale=1.3, style='whitegrid', palette=kul_pal
    )
    base_vals = metrics.loc[metrics.type == 'base'].iloc[0]
    base_macs = base_vals['macs']
    base_params = base_vals['params']

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    ax[0].axhline(
        base_params,
        linestyle='--',
        linewidth=2,
        color='#DD8A2E',
        label='Baseline',
    )
    ax[1].axhline(
        base_macs,
        linestyle='--',
        linewidth=2,
        color='#DD8A2E',
        label='Baseline',
    )

    sns.barplot(
        data=metrics_plot,
        x='comp',
        hue='type',
        y='params',
        hue_order=['cp', 'tt', 'tucker'],
        ax=ax[0],
        palette=kul_pal,
        ci=None,
    )
    sns.barplot(
        data=metrics_plot,
        x='comp',
        hue='type',
        y='macs',
        ax=ax[1],
        hue_order=['cp', 'tt', 'tucker'],
        palette=kul_pal,
        ci=None,
    )

    # Axes
    ax[0].set_xlabel(
        'Layer compression', fontdict={'weight': 'bold', 'size': 14}
    )
    ax[1].set_xlabel(
        'Layer compression', fontdict={'weight': 'bold', 'size': 14}
    )
    ax[0].set_ylabel(
        'Network parameters', fontdict={'weight': 'bold', 'size': 14}
    )
    ax[1].set_ylabel('MACs', fontdict={'weight': 'bold', 'size': 14})
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    # Legend
    ax[0].get_legend().remove()
    plt.legend(title='Type')
    ax[1].legend(
        title='Type',
        labels=['Baseline', 'Canonical Polyadic', 'Tensor Train', 'Tucker'],
    )
    plt.setp(ax[1].get_legend().get_title(), fontweight='bold')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    plt.show()

    # PLOT 2 - BASELINE
    ###################

    base_data = metrics[metrics.type == 'base']
    dice_data = base_data[
        [col for col in base_data if col.startswith('test_dice')]
    ]
    hd_data = base_data[
        [col for col in base_data if col.startswith('test_hd')]
    ]

    base_aggs = base_data.agg(['mean', 'std', 'median'])

    dice_plot = dice_data.melt(
        value_vars=['test_dice_et', 'test_dice_tc', 'test_dice_wt'],
        var_name='metric',
        value_name='score',
    )
    hd_plot = hd_data.melt(
        value_vars=['test_hd_et', 'test_hd_tc', 'test_hd_wt'],
        var_name='metric',
        value_name='score',
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    sns.set_theme(
        context='paper',
        font_scale=1.3,
        style='whitegrid',
        palette=['#E7580C', '#FB7E3A', '#FBAF3A'],
    )

    sns.barplot(
        data=dice_plot,
        x='metric',
        y='score',
        ax=ax[0],
        order=['test_dice_et', 'test_dice_tc', 'test_dice_wt'],
    )
    sns.barplot(
        data=hd_plot,
        x='metric',
        y='score',
        ax=ax[1],
        order=['test_hd_et', 'test_hd_tc', 'test_hd_wt'],
    )
    ax[0].set_xlabel('Dice score', fontdict={'weight': 'bold', 'size': 14})
    ax[1].set_xlabel(
        '95% Hausdorff distance', fontdict={'weight': 'bold', 'size': 14}
    )
    ax[0].set_ylabel(None)
    ax[1].set_ylabel(None)

    plt.legend(
        title='Region',
        handles=[
            mpatches.Patch(color=col, label=lab)
            for col, lab in zip(
                ['#E7580C', '#FB7E3A', '#FBAF3A'], ('ET', 'TC', 'WT')
            )
        ],
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10,
    )
    plt.setp(ax[1].get_legend().get_title(), fontweight='bold')
    ax[0].axes.xaxis.set_ticks([])
    ax[1].axes.xaxis.set_ticks([])

    ax[0].set_ylim(0.5, 1)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    # plt.savefig(join(vis_dir, 'baseline_metrics.pdf'),bbox_inches='tight', pad_inches=0)
    plt.show()

    # PLOT 3 - DICE AND HAUSDORFF
    #############################

    metrics['net_comp'] = base_params / metrics['params']
    metrics['mac_delta'] = metrics['macs'] / base_macs

    """metrics.groupby(['Format', 'Compression']).agg(['mean', 'std'])[
        ['Dice ET', 'Dice TC', 'Dice WT', '95% Hausdorff ET', '95% Hausdorff TC', '95% Hausdorff WT',
         'Network compression','MAC improvement']].to_csv('dice_hausdorff.csv')"""

    for x_var in ('comp',):  # , 'Network compression','MACs'):

        sns.set_theme(
            context='paper', font_scale=1.4, style='ticks', palette=kul_pal
        )
        fig, ax = plt.subplots(3, 2, figsize=(14, 14), dpi=300, sharex=False)
        z_val = 1.96  # 1.37  # 1.96

        for idx, measure in enumerate(
            (
                'test_dice_et',
                'test_dice_tc',
                'test_dice_wt',
                'test_hd_et',
                'test_hd_tc',
                'test_hd_wt',
            )
        ):
            row_idx = idx % 3
            col_idx = idx // 3

            baseline_mean = metrics.loc[metrics.type == 'base'][measure].mean()
            baseline_sd = metrics.loc[metrics.type == 'base'][measure].std()
            n = len(metrics.loc[metrics.type == 'base'])
            low, up = baseline_mean - z_val * baseline_sd / math.sqrt(
                n
            ), baseline_mean + z_val * baseline_sd / math.sqrt(n)

            ax[row_idx, col_idx].axhline(
                baseline_mean,
                xmin=0.02,
                xmax=1.02,
                linestyle='--',
                color='#FB7E3A',
                label='Baseline',
            )
            ax[row_idx, col_idx].axhspan(
                ymin=low,
                ymax=up,
                xmin=0.02,
                xmax=1.02,
                alpha=0.1,
                color='#FB7E3A',
            )

            sns.lineplot(
                data=metrics.loc[metrics.type != 'base'],
                x=x_var,
                y=measure,
                err_style='bars',
                err_kws={'capsize': 5},
                style='type',
                markers=True,
                markersize=10,
                dashes=True,
                ci=95,
                hue_order=[
                    'cp',
                    'tt',
                    'tucker',
                ],
                hue='type',
                ax=ax[row_idx, col_idx],
            )
            if x_var == 'macs':
                ax[row_idx, col_idx].axvline(
                    base_macs, 0, linestyle='--', color='#FB7E3A'
                )
            ax[row_idx, col_idx].set_ylabel(
                measure, fontdict={'weight': 'bold'}
            )
            ax[row_idx, col_idx].set_xscale('log')
            ax[row_idx, col_idx].set_xticks([2**i for i in range(1,9)])
            ax[row_idx, col_idx].get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            ax[row_idx, col_idx].get_xaxis().set_tick_params(which='minor', size=0,width=0)
            #ax[row_idx, col_idx].get_xaxis().set_tick_params(which='minor', width=0)
            if row_idx < 2:
                ax[row_idx, col_idx].set_xlabel('')
            else:
                ax[row_idx, col_idx].set_xlabel(
                    'Layer compression' if x_var == 'Compression' else x_var,
                    fontdict={'weight': 'bold'},
                )
            if row_idx == 1 and col_idx == 1:
                ax[row_idx, col_idx].legend(
                    title='Type',
                    bbox_to_anchor=(1.05, 1),
                    labels=[
                        'Baseline',
                        'Canonical Polyadic',
                        'Tensor Train',
                        'Tucker',
                    ],
                    loc='upper right',
                )

                plt.setp(
                    ax[row_idx, col_idx].get_legend().get_title(),
                    fontweight='bold',
                )
            else:
                ax[row_idx, col_idx].get_legend().remove()
            sns.despine(left=True, bottom=True)
        plt.tight_layout()

        """plt.savefig(
            join(vis_dir, f'tensor_metrics_{x_var.lower()}.pdf'),
            bbox_inches='tight',
            pad_inches=0,
        )"""

        plt.show()

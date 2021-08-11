#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Statistical tests
"""

import pandas as pd
import numpy as np
import utils.helper as hlp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
from scipy.stats import mannwhitneyu,wilcoxon, ttest_ind
import scikit_posthocs as sp

def look_one_way(data:pd.DataFrame, fixed_var:str, fixed_level:str, metric:str='Dice'):

    variable = 'Compression' if fixed_var == 'Format' else 'Format'

    # Restrict data
    data = data.loc[data[fixed_var] == fixed_level]

    # Calculate anova
    aov = pg.anova(data=data, dv=metric, between=variable, detailed=True)

    # Calculate pairwise comparisons
    pair = pg.pairwise_tukey(data=data, dv=metric, between=variable)

    if variable == 'Compression':
        pair = pair.iloc[[0,5,9,12,14],:]

    return aov, pair


if __name__ == '__main__':

    hlp.hi('Statistics')

    metrics = pd.read_pickle('metrics.pkl')

    metrics_baseline = metrics.loc[metrics.Format == 'baseline']
    metrics_tensor = metrics.loc[metrics.Format != 'baseline']
    metrics_tt = metrics_tensor.loc[metrics_tensor.Format == 'Tensor train (v1)']
    metrics_tt2 = metrics_tensor.loc[metrics_tensor.Format == 'Tensor train (v2)']
    metrics_cp = metrics_tensor.loc[metrics_tensor.Format == 'Canonical polyadic']
    metrics_tucker = metrics_tensor.loc[metrics_tensor.Format == 'Tucker']

    '''for format in ('Tensor train','Tensor train 2','Canonical polyadic','Tucker'):
        hlp.log(format,title=True)
        aov, pair = look_one_way(data=metrics,fixed_var='Format',fixed_level=format,metric='Dice')
        print('ANOVA')
        print(aov)
        print('TUKEY')
        print(pair)'''

    # Build metric table
    data = []
    for format in ('Canonical polyadic', 'Tensor train (v1)', 'Tensor train (v2)', 'Tucker'):

        for compression in (2,5,10,20,35,50,75,100):
            print('COMPRESSION ', compression)
            for measure in ('Dice ET','Dice TC','Dice WT',
                            '95% Hausdorff ET','95% Hausdorff TC','95% Hausdorff WT'):

                # Get vectors
                a = metrics_baseline[measure]
                b = metrics_tensor.loc[(metrics_tensor.Format == format) & (metrics_tensor.Compression == compression)][measure]

                # Test
                t = mannwhitneyu(a,b)
                print(measure, '----', t)

                entry = {
                    'Format': format,
                    'Layer compression': compression,
                    'Measure': measure,
                    'U': t[0],
                    'p': t[1]
                }
                data.append(entry)

    table_df = pd.DataFrame(data)
    table_df = table_df.pivot_table(index=['Format','Layer compression'],columns='Measure', values=['U','p'])

    table_df.replace({'Canonical polyadic':'CP',
                      'Tensor train (v1)': 'TT (v1)',
                      'Tensor train (v2)': 'TT (v2)'
    },inplace=True)
    table_df.to_csv('u-tests.csv',decimal=',')
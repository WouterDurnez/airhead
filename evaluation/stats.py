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

    metrics_tensor = metrics.loc[metrics.Format != 'baseline']
    metrics_tt = metrics_tensor.loc[metrics_tensor.Format == 'Tensor train']
    metrics_tt2 = metrics_tensor.loc[metrics_tensor.Format == 'Tensor train 2']
    metrics_cp = metrics_tensor.loc[metrics_tensor.Format == 'Canonical polyadic']
    metrics_tucker = metrics_tensor.loc[metrics_tensor.Format == 'Tucker']

    for format in ('Tensor train','Tensor train 2','Canonical polyadic','Tucker'):
        hlp.log(format,title=True)
        aov, pair = look_one_way(data=metrics,fixed_var='Format',fixed_level=format,metric='Dice')
        print('ANOVA')
        print(aov)
        print('TUKEY')
        print(pair)



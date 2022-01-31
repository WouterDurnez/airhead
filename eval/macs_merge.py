#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Merge flop metrics
"""
import os
from os.path import join
import ast
import numpy as np
import pandas as pd
import seaborn as sns

import src.utils.helper as hlp
from src.utils.helper import KUL_PAL, KUL_PAL2

kul_pal = sns.color_palette(KUL_PAL, as_cmap=True)
kul_pal2 = sns.color_palette(KUL_PAL2, as_cmap=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)




if __name__ == '__main__':

    hlp.hi('MACs merge', log_dir='../../logs/flops')

    files = os.listdir(hlp.LOG_DIR)

    rows = []

    for f in files:

        if f == 'model_flops.npy':
            continue

        data = np.load(join(hlp.LOG_DIR,f), allow_pickle=True).item()

        params = f.split('.')[0].split('_')[2:]

        type = params[0]
        compression = int(params[1][1:])
        widths = ast.literal_eval(params[2][1:])
        kernel_size = int(params[3][1:])

        s = pd.Series({
            'type': type,
            'compression': compression,
            'widths': widths,
            'kernel_size': kernel_size,
            'macs': data[type][compression][widths][kernel_size]['macs'],
            'params': data[type][compression][widths][kernel_size]['params'],

        })

        rows.append(s)

    df = pd.DataFrame(rows)
    df.sort_values(by=['type', 'kernel_size', 'widths', 'compression'], inplace=True)
    df.to_csv("df_macs_params.csv", index=False)
import utils.helper as h
import os
import torch
from utils.helper import log
from layers import air_conv

if __name__ == '__main__':

    h.hi('Jobscript generation')

    """
    Create jobscripts per fold, each of which submits
    training of all tensorized CNNs (each tensor net type
    x 6 compression rates), amounting to a total of 24 jobs.
    """

    '''for fold in range(5):

        with open(f'jobscripts_batch_fold{fold}.sh', 'w') as f:

            f.write('#!/bin/bash\n')

            f.write(f'\n#FOLD {fold}\n')

            for comp in (2, 5, 10, 20, 50, 100):

                f.write(f'#COMPRESSION {comp}\n')

                for type in ('cpd','tucker','tt','tt2'):
                    command = f'qsub jobscript_air.pbs -v TYPE={type},FOLD={fold},COMP={comp}\n'
                    f.write(command)'''
    fold = 0
    with open(f'jobscripts_batch_test_fold{fold}.sh', 'w') as f:
        f.write('#!/bin/bash\n')

        f.write(f'\n#FOLD {fold}\n')

        for comp in (2, 5, 10, 20, 50, 100):

            f.write(f'#COMPRESSION {comp}\n')

            for type in ('cpd', 'tucker', 'tt', 'tt2'):
                command = f'qsub jobscript_test_air.pbs -v TYPE={type},FOLD={fold},COMP={comp}\n'
                f.write(command)

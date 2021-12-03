import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import utils.helper as h
#from ..utils import helper as h

if __name__ == '__main__':

    h.hi('Jobscript generation')

    """
    Create jobscripts per fold, both for tensorized and baseline networks.
    """

    h.log('Creating baseline jobscripts...')
    with open(f'../jobscripts/jobs_base_batch.sh', 'w') as f:

        f.write('#!/bin/bash\n'
                '#BASELINE\n')

        for fold in range(5):
            command = f'qsub jobscript_air.pbs -v TYPE=base,COMP=1,FOLD={fold}\n'
            f.write(command)


    h.log('Creating tensorized jobscripts...')
    for fold in range(5):

        with open(f'../jobscripts/jobs_air_batch_fold{fold}.sh', 'w') as f:

            f.write(f'#!/bin/bash\n'
                    f'\n#FOLD {fold}\n')

            for comp in (2, 5, 10, 20, 35, 50, 75, 100):

                f.write(f'#COMPRESSION {comp}\n')

                for type in ('cpd', 'tucker', 'tt'):
                    command = f'qsub jobscript_air.pbs -v TYPE={type},FOLD={fold},COMP={comp}\n'
                    f.write(command)

    h.log('Complete!', title=True, color='green')
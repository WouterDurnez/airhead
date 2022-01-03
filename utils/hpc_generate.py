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

    COMPRESSION = (35,50,75,100)
    TENSOR_NET_TYPES = ('cp','tt','tucker')
    KERNEL_SIZES = (3,5,7)
    WIDTHS = (0,1)

    h.log('Creating baseline jobscripts...')
    with open(f'../jobscripts/jobs_base_batch.sh', 'w') as f:

        f.write('#!/bin/bash\n'
                '#BASELINE\n')

        for fold in range(5):
            command = f'qsub jobscript_air.pbs -v TYPE=base,COMP=1,FOLD={fold},KERNEL={KERNEL_SIZES[0]},WIDTHS={WIDTHS[1]}\n'
            f.write(command)

    with open(f'../jobscripts/jobs_flops_base_batch.sh', 'w') as f:

        f.write('#!/bin/bash\n'
                    '#BASELINE\n')

        for kernel in KERNEL_SIZES:
            for widths in WIDTHS:
                command = f'qsub jobscript_flops.pbs -v TYPE=base,COMP=1,KERNEL={kernel},WIDTHS={widths}\n'
                f.write(command)


    h.log('Creating tensorized jobscripts...')
    for fold in range(5):

        with open(f'../jobscripts/jobs_temp_air_batch_fold{fold}.sh', 'w') as f:

            f.write(f'#!/bin/bash\n'
                    f'\n#FOLD {fold}\n')

            for comp in COMPRESSION:

                f.write(f'#COMPRESSION {comp}\n')

                for type in TENSOR_NET_TYPES:
                    command = f'qsub jobscript_air.pbs -v TYPE={type},FOLD={fold},COMP={comp},KERNEL={KERNEL_SIZES[0]},WIDTHS={WIDTHS[0]}\n'
                    f.write(command)

    with open(f'../jobscripts/jobs_flops_tensorized_batch.sh', 'w') as f:

            f.write('#!/bin/bash\n'
                    '#TENSORIZED\n')

            for kernel in KERNEL_SIZES:
                for widths in WIDTHS:
                    for comp in COMPRESSION:
                        for type in TENSOR_NET_TYPES:
                            command = f'qsub jobscript_flops.pbs -v TYPE={type},COMP={comp},KERNEL={kernel},WIDTHS={widths}\n'
                            f.write(command)

    h.log('Complete!', title=True, color='green')
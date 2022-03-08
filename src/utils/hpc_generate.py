import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import src.utils.helper as h
#from ..utils import helper as h

if __name__ == '__main__':

    h.hi('Jobscript generation')

    """
    Create jobscripts per fold, both for tensorized and baseline networks.
    """

    #COMPRESSION = (2,5,10,20,35,50,75,100)
    COMPRESSION = (2,4,8,16,32,64,128,256)
    TENSOR_NET_TYPES = ('cp','tt','tucker')
    KERNEL_SIZES = (3,5,7)
    WIDTHS = (0,1)

    h.log('Creating baseline jobscripts...')
    with open(f'../../jobscripts/jobs_full_batch.sh', 'w') as f:

        f.write('#!/bin/bash\n'
                '#BASELINE\n')

        for fold in range(5):
            command = f'qsub jobscript_air.pbs -v TYPE=base,COMP=1,FOLD={fold},KERNEL={KERNEL_SIZES[0]},WIDTHS={WIDTHS[0]}\n'
            f.write(command)


        h.log('Creating tensorized jobscripts...')
        for fold in range(5):

            f.write(f'\n#FOLD {fold}\n')

            for comp in COMPRESSION:

                f.write(f'#COMPRESSION {comp}\n')

                for type in TENSOR_NET_TYPES:
                    command = f'qsub jobscript_air.pbs -v TYPE={type},FOLD={fold},COMP={comp},KERNEL={KERNEL_SIZES[0]},WIDTHS={WIDTHS[0]}\n'
                    f.write(command)

    """with open(f'../../jobscripts/quick_del.sh', 'w') as f:
        f.write('#!/bin/bash\n'
                '#QUICKDEL\n')

        for i in range(100):
            f.write(f'qdel {40183604+i}\n')"""




    h.log('Complete!', title=True, color='green')
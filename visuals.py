#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
Visualization functions
"""
import utils.helper as hlp
from utils.helper import set_dir, log
from training.data_module import BraTSDataModule

from os.path import join

if __name__ == '__main__':

    # Let's go
    hlp.hi("Visualizing BraTS")

    model_name = 'unet_baseline'
    version = 0

    # Set data directory
    hlp.set_params(data_dir='../../../data/MICCAI_BraTS2020_TrainingData')
    vis_dir = join(hlp.LOG_DIR, 'visuals', model_name)
    set_dir(vis_dir)


    # Initialize data module
    log("Initializing data module")
    brats = BraTSDataModule(data_dir='../../../data',
                            num_workers=8,
                            batch_size=1,
                            validation_size=.2)
    brats.setup(stage='visualize')
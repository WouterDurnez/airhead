#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
BraTS data module for use in training functions
"""
from glob import glob
from os import listdir
from os.path import join

from monai.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from utils.helper import *
from training.transforms import *


# Get dict mapping subjects with data paths
def get_data(root_dir, labeled=True):
    """
    Given a directory, return a dictionary containing the file paths to all image data
    :param root_dir: data directory
    :return: path dict
    """
    assert root_dir.split('/')[-1].startswith('MICCAI_BraTS2020'), 'Invalid directory!'

    # Store data
    patients = []

    # Get patient folders
    ids = [id for id in sorted(listdir(root_dir)) if id.startswith('BraTS20')]

    # Loop over patient folders
    for id in ids:

        # Avoid system files
        if id.startswith('.'):
            continue

        # Build path dict
        path_dict = {
            'id': id,
            'input': [
                glob(join(root_dir, id, "*t1*"))[0],
                glob(join(root_dir, id, "*t1ce*"))[0],
                glob(join(root_dir, id, "*t2*"))[0],
                glob(join(root_dir, id, "*flair*"))[0],
            ]
        }
        # Add path to segmentation target (only for training data)
        if labeled:
            path_dict['target'] = glob(join(root_dir, id, "*seg*"))[0]
        patients.append(path_dict)

    return patients


# Data module
class BraTSDataModule(LightningDataModule):
    """
    BraTS-specific Lightning data module, designed to encapsulate
    all data processing steps and data loader functions
    """

    def __init__(
            self,
            data_dir=None,
            test_dir=None,
            num_workers=0,
            batch_size=2,
            validation_size=0.2,
    ):
        super().__init__()

        # Directories (TrainingData & ValidationData)
        self.data_dir = data_dir
        self.test_dir = test_dir

        # Number of workers (for leveraging multi-core CPU/GPU)
        self.num_workers = num_workers

        # Batch size during training (usually minibatch of 1 or 2)
        self.batch_size = batch_size

        # Proportion (!) of dataset to reserve for validation
        self.validation_size = validation_size

        # Get all transforms
        self.training_transform = get_train_transform()
        self.validation_transform = get_val_transform()
        self.test_transform = get_test_transform()
        self.visualization_transform = get_vis_transform()

        # To be initialized by setup (can be removed, but added for personal clarity)
        self.data = None
        self.training_data, self.validation_data, self.test_data, self.visualization_data = None, None, None, None
        self.training_set, self.validation_set, self.test_set, self.visualization_set = None, None, None, None

    ################################################################
    # SET UP MODULE for fitting, testing or visualization (or all) #
    ################################################################

    def setup(self, stage=None):

        # Assign training datasets (train & val) for use in dataloaders
        if stage == "fit" or stage is None:
            # Get train/validation data (paths)
            self.data = get_data(self.data_dir)

            # Split data into training and validation
            (
                self.training_data,
                self.validation_data,
            ) = train_test_split(
                self.data,
                test_size=self.validation_size,
                random_state=616,
            )

            # Build training and validation sets (monai Dataset subclass)
            self.training_set = Dataset(
                data=self.training_data,
                transform=self.training_transform
            )
            self.validation_set = Dataset(
                data=self.validation_data,
                transform=self.validation_transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            # Get test data (paths)
            self.test_data = get_data(self.test_dir, labeled=False)

            # Build test set
            self.test_set = Dataset(
                data=self.test_data,
                transform=self.test_transform
            )

        # Assign test dataset for visualization (inspect the results of our training)
        if stage == "visualize" or stage is None:
            # Get test subjects
            self.visualization_data = get_data(self.test_dir, labeled=False)

            # Build visualization set
            self.visualization_set = Dataset(
                data=self.visualization_data,
                transform=self.visualization_transform,
            )

    ############################################
    # DATA LOADERS (monai Dataloader subclass) #
    ############################################

    def train_dataloader(self):
        training_loader = DataLoader(
            dataset=self.training_set,
            batch_size=self.batch_size,
            shuffle=True,  # Reshuffle dataset at every epoch
            num_workers=self.num_workers,
            pin_memory=True,  # Copy Tensors in CUDA pinned memory before returning them
        )
        return training_loader

    def val_dataloader(self):
        validation_loader = DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return validation_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_loader


if __name__ == '__main__':

    # Set path to directories
    # Use helper.set_params
    root_dir = DATA_DIR
    train_dir = join(DATA_DIR, 'MICCAI_BraTS2020_TrainingData')
    test_dir = join(DATA_DIR, 'MICCAI_BraTS2020_ValidationData')

    brats = BraTSDataModule(data_dir=train_dir, test_dir=test_dir)
    brats.setup()

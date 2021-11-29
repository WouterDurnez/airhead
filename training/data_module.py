#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#

"""
BraTS data module for use in training functions
"""
import json
from glob import glob
from os import listdir
from os.path import join

from monai.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split, KFold

import utils.helper as hlp
from training.transforms import *
from utils.helper import log
from medset.brats import BraTSDataModule as PooyaModule

# Get BraTS2020 data paths
def get_data_2020(root_dir:str, labeled:bool=True) -> dict:
    """
    Given a directory, return a dictionary containing the file paths to all image data for BraTS 2020 set
    :param labeled: add path to segmentation maps in case of training data
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
                glob(join(root_dir, id, "*t1.*"))[0],
                glob(join(root_dir, id, "*t1ce.*"))[0],
                glob(join(root_dir, id, "*t2.*"))[0],
                glob(join(root_dir, id, "*flair.*"))[0],
            ]
        }
        # Add path to segmentation target (only for training data)
        if labeled:
            path_dict['target'] = glob(join(root_dir, id, "*seg*"))[0]
        patients.append(path_dict)

    return patients


# Get Decathlon data paths
def get_data_decathlon(root_dir:str, train:bool=True) -> list:
    """
    Given a directory, return a dictionary containing the file paths to all image data for Decathlon set
    :param labeled: add path to segmentation maps in case of training data
    :param root_dir: data directory
    :return: path dict
    """
    assert root_dir.split('/')[-1].startswith('Task01_BrainTumour'), 'Invalid directory!'

    with open(join(root_dir, 'dataset.json')) as fp:
        meta_data = json.load(fp)

    if train:
        return meta_data['training']
    else:
        return meta_data['test']


# Data module
class BraTSDataModule(LightningDataModule):
    """
    BraTS-specific Lightning data module, designed to encapsulate
    all data processing steps and data loader functions
    """

    def __init__(
            self,
            data_dir=None,
            patch_dim: int = 128,
            num_workers: int=0,
            batch_size: int =1,
            n_folds=5,
            fold_index:int=0
    ):
        super().__init__()

        # Directories (TrainingData only)
        self.data_dir = data_dir

        # Dimension of input (used for rescaling)
        self.patch_dim = patch_dim

        # Number of workers (for leveraging multi-core CPU/GPU)
        self.num_workers = num_workers

        # Batch size during training (usually minibatch of 1 or 2)
        self.batch_size = batch_size

        # Number of folds to use in crossvalidation scheme
        self.n_folds = n_folds

        # Index of split (fold)
        self.fold_index = fold_index

        # Get all transforms
        self.training_transform = get_train_transform(patch_dim=self.patch_dim)
        self.validation_transform = get_val_transform(patch_dim=self.patch_dim)
        self.test_transform = get_test_transform()
        self.visualization_transform = get_vis_transform()

        # To be initialized by setup (can be removed, but added for personal clarity)
        self.data = None
        self.training_data, self.validation_data, self.test_data, self.visualization_data = None, None, None, None
        self.training_set, self.validation_set, self.test_set, self.visualization_set = None, None, None, None

        """We won't be using the test set, because it involves the use of the online BraTS platform
        to calculate performance metrics. Hence, we will calculate the split here, rather than at setup.
        We must be careful, however, to not use the validation set for any purpose during training (including
        early stopping or learning rate scheduling)."""

        # Get data paths, depending on dataset
        data_folder_name = self.data_dir.split('/')[-1]
        if data_folder_name.lower().__contains__('miccai_brats2020'):
            self.dataset = 'brats2020'
            self.data = get_data_2020(self.data_dir)
        elif data_folder_name.lower().__contains__('task01_braintumour'):
            self.dataset = 'decathlon'
            self.data = get_data_decathlon(self.data_dir)
        log(f'Based on the directory, assuming <{self.dataset}> dataset.', verbosity=1)

        # 5-fold validation
        splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=616)
        splits = splitter.split(self.data)

        # Store all folds (loop over generator)
        self.folds = [next(splits) for _ in range(5)]

        # Get training and validation data, depending on fold index
        self.training_data = [self.data[idx] for idx in self.folds[self.fold_index][0]]
        self.validation_data = [self.data[idx] for idx in self.folds[self.fold_index][1]]

        # Performance metrics will be calculated on the validation dataset as well
        self.test_data = self.validation_data

        # Visualization of results will also use the validation dataset
        self.visualization_data = self.validation_data

    ################################################################
    # SET UP MODULE for fitting, testing or visualization (or all) #
    ################################################################

    def setup(self, stage=None):

        # Assign training datasets (train & val) for use in dataloaders
        if stage == "fit" or stage is None:

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

            # Build test set
            self.test_set = Dataset(
                data=self.test_data,
                transform=self.test_transform
            )

        # Assign test dataset for visualization (inspect the results of our training)
        if stage == "visualize" or stage is None:

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
            pin_memory=False,  # Copy Tensors in CUDA pinned memory before returning them
        )
        return training_loader

    def val_dataloader(self):
        validation_loader = DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        return validation_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        return test_loader


if __name__ == '__main__':

    # Set path to directories
    # Use helper.set_params
    hlp.hi('Dry run for BraTS data module')
    data_dir = join(hlp.DATA_DIR,'Task01_BrainTumour')

    brats = PooyaModule(root_dir=data_dir,
                        batch_size=1,
                        seed=616,
                        cache_num=1)
    brats.setup('fit')

    brats[0]

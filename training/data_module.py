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

from monai.apps import DecathlonDataset, CrossValidation
from monai.data import DataLoader
from pytorch_lightning import LightningDataModule

from training.transforms import *
from utils.helper import *

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
            num_workers: int = 0,
            batch_size: int = 1,
            n_folds: int = 5,
            fold_index: int = 0,
            seed: int = 616,
            **ds_params
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

        # Seed for randomization
        self.seed = seed

        # Get all transforms
        self.training_transform = get_train_transform(patch_dim=self.patch_dim)
        self.validation_transform = get_val_transform(patch_dim=self.patch_dim)
        self.test_transform = get_test_transform()
        self.visualization_transform = get_vis_transform()

        # To be initialized by setup (can be removed, but added for personal clarity)
        self.data = None
        self.training_set, self.validation_set, self.test_set, self.visualization_set = None, None, None, None

        # Set up crossvalidation data sets
        self.cross_val_ds = CrossValidation(
            dataset_cls=DecathlonDataset,
            nfolds=self.n_folds,
            seed=self.seed,
            root_dir=self.data_dir,
            task='Task01_BrainTumour',
            section='training',
            transform=self.training_transform,
            num_workers=self.num_workers,
            **ds_params
        )

    ################################################################
    # SET UP MODULE for fitting, testing or visualization (or all) #
    ################################################################

    def setup(self, stage=None):

        # Assign training datasets (train & val) for use in dataloaders
        if stage == "fit" or stage is None:
            # Build training and validation sets (monai Dataset subclass)
            self.training_set = self.cross_val_ds.get_dataset(folds=[i for i in range(5) if i != self.fold_index])
            self.validation_set = self.cross_val_ds.get_dataset(folds=self.fold_index,
                                                                transform=self.validation_transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            # Build test set
            self.test_set = self.cross_val_ds.get_dataset(folds=self.fold_index,
                                                          transform=self.test_transform)

        # Assign test dataset for visualization (inspect the results of our training)
        if stage == "visualize" or stage is None:
            # Build visualization set
            self.visualization_set = self.cross_val_ds.get_dataset(folds=self.fold_index,
                                                    transform=self.visualization_transform)

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
            dataset=self.validation_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        return validation_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        return test_loader


if __name__ == '__main__':
    # Set path to directories
    # Use helper.set_params
    hi('Dry run for BraTS data module')

    brats = BraTSDataModule(data_dir=DATA_DIR, fold_index=4, cache_rate=0.01)
    brats.setup()

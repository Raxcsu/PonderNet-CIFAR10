# //////////////////////////////////////////////
# ////////////// DATASET: CIFAR10 //////////////
# //////////////////////////////////////////////

# ==============================================
# SETUP AND IMPORTS
# ==============================================

# import Libraries
import os

# torch imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import torchmetrics

# pl imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# remaining imports
import wandb
from math import floor

# ==============================================
# CIFAR10_DATAMODULE
# ==============================================

class CIFAR10_DataModule(pl.LightningDataModule):
    '''
        DataModule to hold the CIFAR10 dataset. Accepts different transforms for train and test to
        allow for extrapolation experiments.

        Parameters
        ----------
        data_dir : str
            Directory where CIFAR10 will be downloaded or taken from.

        train_transform : [transform] 
            List of transformations for the training dataset. The same
            transformations are also applied to the validation dataset.

        test_transform : [transform] or [[transform]]
            List of transformations for the test dataset. Also accepts a list of
            lists to validate on multiple datasets with different transforms.

        batch_size : int
            Batch size for both all dataloaders.
    '''

    def __init__(self, data_dir='./', train_transform=None, test_transform=None, batch_size=128):
        
        super().__init__()
        
        self.data_dir        = data_dir
        self.batch_size      = batch_size
        self.train_transform = train_transform
        self.test_transform  = test_transform

        self.default_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data (train/val and test sets)
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''
            Called on each GPU separately - stage defines if we are
            at fit, validate, test or predict step.
        '''
        # we set up only relevant datasets when stage is specified
        if stage in [None, 'fit', 'validate']:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=(self.train_transform or self.default_transform))
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        if stage == 'test' or stage is None:
            if self.test_transform is None or isinstance(self.test_transform, transforms.Compose):
                self.cifar_test = CIFAR10(self.data_dir,
                                        train=False,
                                        transform=(self.test_transform or self.default_transform))
            else:
                self.cifar_test = [CIFAR10(self.data_dir,
                                        train=False,
                                        transform=test_transform)
                                    for test_transform in self.test_transform]

    def train_dataloader(self):
        '''returns training dataloader'''
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)
        return cifar_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        cifar_val = DataLoader(self.cifar_val, batch_size=self.batch_size)
        return cifar_val

    def test_dataloader(self):
        '''returns test dataloader(s)'''
        if isinstance(self.cifar_test, CIFAR10):
            return DataLoader(self.cifar_test, batch_size=self.batch_size)

        cifar_test = [DataLoader(test_dataset, batch_size=self.batch_size)
                      for test_dataset in self.cifar_test]
        return cifar_test
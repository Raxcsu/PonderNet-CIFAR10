# //////////////////////////////////////////////
# ////////////// DATASET: CIFAR10 //////////////
# //////////////////////////////////////////////

# ==============================================
# SETUP AND IMPORTS
# ==============================================

# import Libraries
import os
import numpy as np

# torch imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
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

# ==============================================
# CIFAR100_DATAMODULE
# ==============================================

class CIFAR100_DataModule(pl.LightningDataModule):
    '''
        This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each.
        There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100
        are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs)
        and a "coarse" label (the superclass to which it belongs).

        Here is the list of classes in the CIFAR-100:

        Superclass                          Classes
        aquatic mammals                     beaver, dolphin, otter, seal, whale
        fish                                aquarium fish, flatfish, ray, shark, trout
        flowers                             orchids, poppies, roses, sunflowers, tulips
        food containers                     bottles, bowls, cans, cups, plates
        fruit and vegetables                apples, mushrooms, oranges, pears, sweet peppers
        household electrical devices        clock, computer keyboard, lamp, telephone, television
        household furniture                 bed, chair, couch, table, wardrobe
        insects                             bee, beetle, butterfly, caterpillar, cockroach
        large carnivores                    bear, leopard, lion, tiger, wolf
        large man-made outdoor things       bridge, castle, house, road, skyscraper
        large natural outdoor scenes        cloud, forest, mountain, plain, sea
        large omnivores and herbivores      camel, cattle, chimpanzee, elephant, kangaroo
        medium-sized mammals                fox, porcupine, possum, raccoon, skunk
        non-insect invertebrates            crab, lobster, snail, spider, worm
        people                              baby, boy, girl, man, woman
        reptiles                            crocodile, dinosaur, lizard, snake, turtle
        small mammals                       hamster, mouse, rabbit, shrew, squirrel
        trees                               maple, oak, palm, pine, willow
        vehicles 1                          bicycle, bus, motorcycle, pickup truck, train
        vehicles 2                          lawn-mower, rocket, streetcar, tank, tractor

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

    def __init__(self, data_dir='./data', train_transform=None, test_transform=None, batch_size=128):
        
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
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''
            Called on each GPU separately - stage defines if we are
            at fit, validate, test or predict step.
        '''
        # we set up only relevant datasets when stage is specified

        if stage in [None, 'fit', 'validate']:
            cifar_full = CIFAR100(self.data_dir, train=True, transform=(self.train_transform or self.default_transform))
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])
            print("cifar_train: " + str(len(self.cifar_train)))
            print("cifar_val: " + str(len(self.cifar_val)))

        if stage == 'test' or stage is None:
            if self.test_transform is None or isinstance(self.test_transform, transforms.Compose):
                self.cifar_test = CIFAR100(self.data_dir,
                                        train=False,
                                        transform=(self.test_transform or self.default_transform))
            else:
                self.cifar_test = [CIFAR100(self.data_dir,
                                        train=False,
                                        transform=test_transform)
                                    for test_transform in self.test_transform]
            print("cifar_test: " + str(len(self.cifar_test)))

    def train_dataloader(self):
        '''returns training dataloader'''
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return cifar_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        cifar_val = DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=2)
        return cifar_val

    def test_dataloader(self):
        '''returns test dataloader(s)'''
        if isinstance(self.cifar_test, CIFAR100):
            return DataLoader(self.cifar_test, batch_size=100, num_workers=2)

        cifar_test = [DataLoader(test_dataset, batch_size=100, num_workers=2)
                      for test_dataset in self.cifar_test]
        return cifar_test

# ==============================================
# CIFAR100C_DATAMODULE
# ==============================================

class CIFAR100C_DataModule(pl.LightningDataModule):
    '''
        https://zenodo.org/record/3555552#.YiS6w3qZOCo

        Parameters
        ----------
        data_dir : str
            Directory where CIFAR10 will be downloaded or taken from.

        test_transform : [transform] or [[transform]]
            List of transformations for the test dataset. Also accepts a list of
            lists to validate on multiple datasets with different transforms.

        batch_size : int
            Batch size for both all dataloaders.
    '''

    def __init__(self, corruption, data_dir='data/', test_transform=None, batch_size=100, base_path='data/CIFAR100-C/CIFAR-100-C/'):
        
        super().__init__()
        
        self.data_dir        = data_dir
        self.batch_size      = batch_size
        self.test_transform  = test_transform
        self.corruption      = corruption
        self.base_path       = base_path

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data (train/val and test sets)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''
            Called on each GPU separately - stage defines if we are
            at fit, validate, test or predict step.
        '''
        # we set up only relevant datasets when stage is specified

        if stage == 'test' or stage is None:
            if self.test_transform is None or isinstance(self.test_transform, transforms.Compose):
                self.cifar_test = CIFAR100(self.data_dir,
                                        train=False,
                                        transform=(self.test_transform or self.default_transform))
            else:
                self.cifar_test = [CIFAR100(self.data_dir,
                                        train=False,
                                        transform=test_transform)
                                    for test_transform in self.test_transform]
            print("cifar_test: " + str(len(self.cifar_test)))

    def test_dataloader(self):
        '''returns test dataloader(s)'''

#        if isinstance(self.cifar_test, CIFAR100):
#            return DataLoader(self.cifar_test, batch_size=100, num_workers=2, shuffle=False, pin_memory=True)
        
        '''
        cifar_test = []

        for corruption in self.corruption:
            self.cifar_test.data = np.load(self.base_path + corruption + '.npy')
            self.cifar_test.targets = torch.LongTensor(np.load(self.base_path + 'labels.npy'))

            test = [DataLoader(self.cifar_test, batch_size=100, num_workers=2, shuffle=False, pin_memory=True)]
            cifar_test.append(test)

            print(corruption + " --- " + str(len(test)))
        '''

        self.cifar_test.data = np.load(self.base_path + self.corruption + '.npy')
        self.cifar_test.targets = torch.LongTensor(np.load(self.base_path + 'labels.npy'))

        cifar_test = [DataLoader(self.cifar_test, batch_size=100, num_workers=2)]

        print("cifar_test_" + self.corruption + "_batch: " + str(len(cifar_test)))

        return cifar_test

# ==============================================
# CIFAR100C_SV_DATAMODULE
# ==============================================

class CIFAR100C_SV_DataModule(pl.LightningDataModule):
    '''
        https://zenodo.org/record/3555552#.YiS6w3qZOCo

        Parameters
        ----------
        data_dir : str
            Directory where CIFAR10 will be downloaded or taken from.

        test_transform : [transform] or [[transform]]
            List of transformations for the test dataset. Also accepts a list of
            lists to validate on multiple datasets with different transforms.

        batch_size : int
            Batch size for both all dataloaders.
    '''

    def __init__(self, corruption, severity=1, data_dir='data/', test_transform=None, batch_size=100, base_path='data/CIFAR100-C/CIFAR-100-C/'):
        
        super().__init__()
        
        self.data_dir        = data_dir
        self.batch_size      = batch_size
        self.test_transform  = test_transform
        self.corruption      = corruption
        self.base_path       = base_path
        self.severity        = severity

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data (train/val and test sets)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''
            Called on each GPU separately - stage defines if we are
            at fit, validate, test or predict step.
        '''
        # we set up only relevant datasets when stage is specified

        if stage == 'test' or stage is None:
            if self.test_transform is None or isinstance(self.test_transform, transforms.Compose):
                self.cifar_test = CIFAR100(self.data_dir,
                                        train=False,
                                        transform=(self.test_transform or self.default_transform))
            else:
                self.cifar_test = [CIFAR100(self.data_dir,
                                        train=False,
                                        transform=test_transform)
                                    for test_transform in self.test_transform]

    def test_dataloader(self):
        '''returns test dataloader(s)'''

#        if isinstance(self.cifar_test, CIFAR100):
#            return DataLoader(self.cifar_test, batch_size=100, num_workers=2, shuffle=False, pin_memory=True)

        self.cifar_test.data = np.load(self.base_path + self.corruption + '.npy')
        self.cifar_test.targets = torch.LongTensor(np.load(self.base_path + 'labels.npy'))

        print("=======================================")
        print("cifar_test_" + self.corruption + ": " + str(len(self.cifar_test)))

        tamaño = 10000
        inicio = (self.severity-1)*tamaño
        final = (self.severity*tamaño)-1
        tmp = list(range(inicio, final, 1))
        cifar_sv = Subset(self.cifar_test, tmp)

        cifar_test = [DataLoader(cifar_sv, batch_size=50, num_workers=2, shuffle=False, pin_memory=True)]

        print("=======================================")
        print("cifar_test_" + self.corruption + "_sv" + str(self.severity) + ": " + str(len(cifar_test)))
        print("inicio: " + str(inicio) + " --- final: " + str(final))
        print("=======================================")

        return cifar_test

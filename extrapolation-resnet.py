# -*- coding: utf-8 -*-

# //////////////////////////////////////////////
# ///////////// EXTRAPOLATION TASK /////////////
# //////////////////////////////////////////////

# ==============================================
# SETUP AND IMPORTS
# ==============================================

# We use `PyTorch Lightning` (wrapping `PyTorch`) as our main framework and `wandb`
# to track and log the experiments. We set all seeds through `PyTorch Lightning`'s dedicated function.

# import Libraries
import os
import numpy as np

# torch imports
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import torch.nn.functional as F
import torchmetrics

# pl imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pondernet import *
from resnet import *
from cifardata import *

# remaining imports
import wandb
from math import floor

# set seeds
seed_everything(1234)

# log in to wandb
wandb.login()

# ==============================================
# CONSTANTS AND HYPERPARAMETERS
# ==============================================

DATA_DIR = './data'
BASE_PATH = 'data/CIFAR100-C/CIFAR-100-C/'

# Trainer settings
BATCH_SIZE = 100 #128
EPOCHS = 100

# Optimizer settings
LR = 0.1        # 0.001ADAM
GRAD_NORM_CLIP = 0.5
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Model hparams
N_ELEMS   = 512
N_HIDDEN  = 100
MAX_STEPS = 20
LAMBDA_P  = 0.1     # 0.2 - 0.4
BETA      = 1    # 1 see what happen
NUM_CLASSES = 100


# ==============================================
# CIFAR100-C SETUP
# ==============================================

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

test_transform = transforms.Compose([transforms.ToTensor(),])

def get_transforms():
    # define transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_22 = transforms.Compose([
        transforms.RandomRotation(degrees=22.5),
        transforms.ToTensor(),
    ])
    transform_45 = transforms.Compose([
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
    ])
    transform_67 = transforms.Compose([
        transforms.RandomRotation(degrees=67.5),
        transforms.ToTensor(),
    ])
    transform_90 = transforms.Compose([
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
    ])

    train_transform = train_transform
    test_transform = [transform_22, transform_45, transform_67, transform_90]

    return train_transform, test_transform

train_transform, test_transform = get_transforms()

# ==============================================
# RUN EXTRAPOLATION
# ==============================================

# Load the CIFAR10 dataset with no rotations and train PonderNet on it.
# Make sure to edit the `WandbLogger` call so that you log the experiment
# on your account's desired project.

model = ResnetCIFAR(
        num_classes=NUM_CLASSES,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY)

# training model with resnet
path = "CIFAR100_checkpoint/resnet-20220303-175848-epoch=60.ckpt"
model = ResnetCIFAR.load_from_checkpoint(path)
print(model.hparams)

def main():
    '''
    for corruption in CORRUPTIONS:

        # initialize datamodule and model
        cifar100_dm = CIFAR100C_DataModule(
            corruption=corruption,
            data_dir=DATA_DIR,
            test_transform=test_transform,
            batch_size=BATCH_SIZE,
            base_path=BASE_PATH)
        
        NAME = 'E-ResNet-ep100-' + corruption

        print(NAME)

        # setup logger
        logger = WandbLogger(project='CIFAR100C', name=NAME, offline=False)
        logger.watch(model)

        trainer = Trainer(
            logger=logger,                      # W&B integration
            gpus=-1,                            # use all available GPU's
            max_epochs=EPOCHS,                  # maximum number of epochs
            gradient_clip_val=GRAD_NORM_CLIP,   # gradient clipping
            val_check_interval=0.25,            # validate 4 times per epoch
            precision=16,                       # train in half precision
            deterministic=True)                 # for reproducibility

        # fit the model
        # trainer.fit(model, datamodule=cifar100_dm)

        # evaluate on the test set
        trainer.test(model, datamodule=cifar100_dm)

    # initialize datamodule and model
    cifar100_dm = CIFAR100C_DataModule(
        corruption=CORRUPTIONS,
        data_dir=DATA_DIR,
        test_transform=test_transform,
        batch_size=BATCH_SIZE,
        base_path=BASE_PATH)
    '''
    cifar100_dm = CIFAR100_DataModule(
        data_dir=DATA_DIR,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE)    
    NAME = 'E-ResNet-ep100-'# + corruption

    print(NAME)

    # setup logger
    logger = WandbLogger(project='CIFAR100C', name=NAME, offline=False)
    logger.watch(model)

    trainer = Trainer(
        logger=logger,                      # W&B integration
        gpus=-1,                            # use all available GPU's
        max_epochs=EPOCHS,                  # maximum number of epochs
        gradient_clip_val=GRAD_NORM_CLIP,   # gradient clipping
        val_check_interval=0.25,            # validate 4 times per epoch
        precision=16,                       # train in half precision
        deterministic=True)                 # for reproducibility

    # fit the model
    # trainer.fit(model, datamodule=cifar100_dm)

    # evaluate on the test set
    trainer.test(model, datamodule=cifar100_dm)
    wandb.finish()

if __name__ == '__main__':
    main()
    

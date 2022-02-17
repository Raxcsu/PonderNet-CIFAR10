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

# Trainer settings
BATCH_SIZE = 128
EPOCHS = 50

# Optimizer settings
LR = 0.1        # 0.001ADAM
GRAD_NORM_CLIP = 0.5
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Model hparams
N_ELEMS   = 512
N_HIDDEN  = 100
MAX_STEPS = 20
LAMBDA_P  = 0.1
BETA      = 0.01

# ==============================================
# CIFAR10 SETUP
# ==============================================

def get_transforms():
    # define transformations
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

    train_transform = transform_22
    test_transform = [transform_22, transform_45, transform_67, transform_90]

    return train_transform, test_transform

train_transform, test_transform = get_transforms()    

# ==============================================
# RUN EXTRAPOLATION
# ==============================================

# Load the CIFAR10 dataset with no rotations and train PonderNet on it.
# Make sure to edit the `WandbLogger` call so that you log the experiment
# on your account's desired project.

# initialize datamodule and model
cifar100_dm = CIFAR100_DataModule(
    data_dir=DATA_DIR,
    train_transform=train_transform,
    test_transform=test_transform,
    batch_size=BATCH_SIZE)

'''
model = PonderCIFAR(
    n_elems=N_ELEMS,
    n_hidden=N_HIDDEN,
    max_steps=MAX_STEPS,
    lambda_p=LAMBDA_P,
    beta=BETA,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY)
'''
model = ResnetCIFAR(
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY)

# setup logger
logger = WandbLogger(project='PonderNet - CIFAR100', name='extrapolation-ResnetCIFAR', offline=False)
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
trainer.fit(model, datamodule=cifar100_dm)

# evaluate on the test set
trainer.test(model, datamodule=cifar100_dm)

wandb.finish()

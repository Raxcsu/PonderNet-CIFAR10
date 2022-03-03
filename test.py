# -*- coding: utf-8 -*-

# //////////////////////////////////////////////
# ///////////// INTERPOLATION TASK /////////////
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
#wandb.login()

# ==============================================
# CONSTANTS AND HYPERPARAMETERS
# ==============================================

DATA_DIR = './data'

# Trainer settings
BATCH_SIZE = 128
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
BETA      = 0.1    # 1 see what happen
N_CLASSES = 100

# ==============================================
# CIFAR100 SETUP
# ==============================================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# ==============================================
# RUN EXTRAPOLATION
# ==============================================

# Load the CIFAR100 dataset with no rotations and train PonderNet on it.
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

model = ResnetCIFAR(
    num_classes=N_CLASSES,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY)
'''

chk_path = "/CIFAR100_checkpoint/epoch=60-step=21471-v1.ckpt"
model = PonderCIFAR.load_from_checkpoint(chk_path,
   n_elems=N_ELEMS,
   n_hidden=N_HIDDEN,
   max_steps=MAX_STEPS,
   lambda_p=LAMBDA_P,
   beta=BETA,
   lr=LR,
   momentum=MOMENTUM,
   weight_decay=WEIGHT_DECAY)
print(model["hyper_parameters"])
# prints the learning_rate you used in this checkpoint

# fit the model
#trainer.fit(model, datamodule=cifar100_dm)

# evaluate on the test set
#trainer.test(model, datamodule=cifar100_dm)

#wandb.finish()


#/home/oscar/PonderNet-CIFAR10/CIFAR100_checkpoint
#'epoch=60-step=21471-v1.ckpt'
#'epoch=60-step=21471.ckpt'
#'epoch=74-step=26135-v1.ckpt'
#'epoch=74-step=26135.ckpt'
#'epoch=83-step=29303-v1.ckpt'
#'epoch=83-step=29303.ckpt'

#/home/oscar/PonderNet-CIFAR10/wandb
#run-20220227_160047-x37jl1s7
#run-20220227_160222-14dideej
#run-20220227_160312-akoc8wb9
#run-20220227_160349-2ilz9hei
#run-20220227_160427-moxd5suq
#run-20220227_160509-2yyg0cc7
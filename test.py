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
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS)

model = ResnetCIFAR(
    num_classes=N_CLASSES,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS)
'''
print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep74-lp01-b1-20220303-094605.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         1
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep74-lp01-b1-20220303-094605.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep83-lp01-b01-20220303-094437.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         0.1
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep83-lp01-b01-20220303-094437.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep79-lp01-b001-20220311-013555.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         0.01
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep79-lp01-b001-20220311-013555.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep92-lp01-b002-20220311-013623.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         0.02
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep92-lp01-b002-20220311-013623.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep84-lp01-b003-20220311-013733.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         0.03
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep84-lp01-b003-20220311-013733.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep72-lp01-b004-20220311-013746.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         0.04
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep72-lp01-b004-20220311-013746.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep75-lp01-b005-20220311-013804.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         0.05
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep75-lp01-b005-20220311-013804.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep85-lp01-b006-20220311-013845.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         0.06
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep85-lp01-b006-20220311-013845.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep91-lp01-b007-20220311-013902.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         0.07
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep91-lp01-b007-20220311-013902.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/pondernet-ep94-lp01-b008-20220311-013913.ckpt'
model = PonderCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)
'''
"beta":         0.08
"lambda_p":     0.1
"lr":           0.1
"max_steps":    20
"momentum":     0.9
"n_elems":      512
"n_hidden":     100
"weight_decay": 0.0005
'''
# 'pondernet-ep94-lp01-b008-20220311-013913.ckpt'

print("-------------------------------------")
path = 'CIFAR100_checkpoint/resnet-20220303-175848-epoch=60.ckpt'
model = ResnetCIFAR.load_from_checkpoint(path)
print(path)
print(model.hparams)

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

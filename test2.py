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
from argparse import ArgumentParser

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
NUM_CLASSES = 100


# ==============================================
# CIFAR10 SETUP
# ==============================================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([transforms.ToTensor(),])

# ==============================================
# RUN EXTRAPOLATION
# ==============================================

# Load the CIFAR10 dataset with no rotations and train PonderNet on it.
# Make sure to edit the `WandbLogger` call so that you log the experiment
# on your account's desired project.

model = PonderCIFAR(
    n_elems=N_ELEMS,
    n_hidden=N_HIDDEN,
    max_steps=MAX_STEPS,
    lambda_p=LAMBDA_P,
    beta=BETA,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY)

# training model with beta = 0.1
path = "CIFAR100_checkpoint/pondernet-epoch=83-20220303-094437.ckpt"
model = PonderCIFAR.load_from_checkpoint(path)
print(model.hparams)

# initialize datamodule and model
cifar100_dm = CIFAR100C_DataModule(
    corruption='gaussian_noise',
    data_dir=DATA_DIR,
    test_transform=test_transform,
    batch_size=BATCH_SIZE,
    base_path=BASE_PATH)

NAME = 'E-PonderNet-b0.1-ep100-'
print(NAME)

# setup logger
logger = WandbLogger(project='test', name=NAME, offline=False)
logger.watch(model)

trainer = Trainer(
    logger=logger,                      # W&B integration
    gpus=-1,                            # use all available GPU's
    max_epochs=EPOCHS,                  # maximum number of epochs
    gradient_clip_val=GRAD_NORM_CLIP,   # gradient clipping
    val_check_interval=0.25,            # validate 4 times per epoch
    precision=16,                       # train in half precision
    deterministic=True)                 # for reproducibility

# evaluate on the test set
trainer.test(model, datamodule=cifar100_dm)

wandb.finish()

'''
    def test_dataloader(self):
        #returns test dataloader(s)

#        if isinstance(self.cifar_test, CIFAR100):
#            return DataLoader(self.cifar_test, batch_size=100, num_workers=2, shuffle=False, pin_memory=True)

        cifar_test = []

        for corruption in self.corruption:
            self.cifar_test.data = np.load(self.base_path + corruption + '.npy')
            self.cifar_test.targets = torch.LongTensor(np.load(self.base_path + 'labels.npy'))

            test = [DataLoader(self.cifar_test, batch_size=100, num_workers=2, shuffle=False, pin_memory=True)]
            cifar_test.append(test)

            print(corruption + " --- " + str(len(test)))
        return cifar_test

'''    


'''
def main():

    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]
    BASE_PATH = 'data/CIFAR100-C/CIFAR-100-C/'
    for corruption in CORRUPTIONS:
        print(BASE_PATH + corruption + '.npy')

if __name__ == '__main__':
    main()

'''

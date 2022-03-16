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
# RUN EXTRAPOLATION
# ==============================================

# Load the CIFAR10 dataset with no rotations and train PonderNet on it.
# Make sure to edit the `WandbLogger` call so that you log the experiment
# on your account's desired project.

def main(argv=None):

    parser = ArgumentParser(description='PyTorch CIFAR100C Testing')

    parser.add_argument(
        "--model",
        type=str,
        default='pondernet',
        help="Choose 'pondernet' or 'resnet' model")

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs has a default value = 100")

    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Beta has a default value = 0.1")

    parser.add_argument(
        "--lambda_p",
        type=float,
        default=0.1,
        help="Lambda_p has a default value = 0.1")

    parser.add_argument(
        "--corruption",
        type=str,
        default='gaussian_noise',
        help="Choose one of these options. CORRUPTIONS: gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression")

    parser.add_argument(
        "--severity",
        type=int,
        default=1,
        help="Severity has a value between 1 to 5")

    # Parameters
    args = parser.parse_args(argv)
    print(args)

    # ==============================================
    # CONSTANTS AND HYPERPARAMETERS
    # ==============================================

    DATA_DIR = './data'
    BASE_PATH = 'data/CIFAR100-C/CIFAR-100-C/'

    # Trainer settings
    BATCH_SIZE = 128
    EPOCHS = args.epochs

    # Optimizer settings
    LR = 0.1        # 0.001ADAM
    GRAD_NORM_CLIP = 0.5
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    # Model hparams
    N_ELEMS   = 512
    N_HIDDEN  = 100
    MAX_STEPS = 20
    LAMBDA_P  = args.lambda_p       # default lambda_p = 0.1
    BETA      = args.beta           # default beta = 0.1
    NUM_CLASSES = 100

    if args.model == 'pondernet':
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
        if BETA == 0.1:
            tmp = 'pondernet-ep83-lp01-b01-20220303-094437.ckpt'
        elif BETA == 1:
            tmp = 'pondernet-ep74-lp01-b1-20220303-094605.ckpt'
        elif BETA == 0.01:
            tmp = 'pondernet-ep79-lp01-b001-20220311-013555.ckpt'
        elif BETA == 0.02:
            tmp = 'pondernet-ep92-lp01-b002-20220311-013623.ckpt'
        elif BETA == 0.03:
            tmp = 'pondernet-ep84-lp01-b003-20220311-013733.ckpt'
        elif BETA == 0.04:
            tmp = 'pondernet-ep72-lp01-b004-20220311-013746.ckpt'
        elif BETA == 0.05:
            tmp = 'pondernet-ep75-lp01-b005-20220311-013804.ckpt'
        elif BETA == 0.06:
            tmp = 'pondernet-ep85-lp01-b006-20220311-013845.ckpt'
        elif BETA == 0.07:
            tmp = 'pondernet-ep91-lp01-b007-20220311-013902.ckpt'
        elif BETA == 0.08:
            tmp = 'pondernet-ep94-lp01-b008-20220311-013913.ckpt'
        else:
            print("Model not found")

        path = 'CIFAR100_checkpoint/' + tmp
        model = PonderCIFAR.load_from_checkpoint(path)
    elif args.model == 'resnet':
        model = ResnetCIFAR(
            num_classes=N_CLASSES,
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY)
        path = 'CIFAR100_checkpoint/resnet-ep60-20220303-175848.ckpt'
        model = ResnetCIFAR.load_from_checkpoint(path)
    else:
        print("Model not found")
    
    print(model.hparams)


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

    # ==============================================
    # MODEL
    # ==============================================

    NAME = 'E-PonderNet-b0.1-ep100-' + args.corruption + '_sv' + str(args.severity)
    print("=======================================")
    print(NAME)
    print("=======================================")

    # setup logger
    logger = WandbLogger(project='CIFARC-SEVERITY', name=NAME, offline=False)
    logger.watch(model)

    trainer = Trainer(
        logger=logger,                      # W&B integration
        gpus=-1,                            # use all available GPU's
        max_epochs=EPOCHS,                  # maximum number of epochs
        gradient_clip_val=GRAD_NORM_CLIP,   # gradient clipping
        val_check_interval=0.25,            # validate 4 times per epoch
        precision=16,                       # train in half precision
        deterministic=True,                 # for reproducibility
        progress_bar_refresh_rate=0)

    # initialize datamodule and model
    cifar100_dm = CIFAR100C_SV_DataModule(
        corruption=args.corruption,
        severity=args.severity,
        data_dir=DATA_DIR,
        test_transform=test_transform,
        batch_size=BATCH_SIZE,
        base_path=BASE_PATH)

    # evaluate on the test set
    trainer.test(model, datamodule=cifar100_dm, verbose=True)

    wandb.finish()

if __name__ == '__main__':
    main()


    #scp -r oscar@10.28.3.155:/home/oscar/PonderNet-CIFAR10/wandb/ home/o
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/1859
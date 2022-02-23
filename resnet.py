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

# remaining imports
import wandb
from math import floor

# ==============================================
# RESNET IN PYTORCH
# ==============================================

# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
# Deep Residual Learning for Image Recognition. arXiv:1512.03385
# ----------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=100):

        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers  = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)             # output of embedding
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def test():
    net = ResNet18()

    out = net(torch.randn(1, 3, 32, 32))
    print(out.size())


# ==============================================
# RESNET WITH CIFAR
# ==============================================

# Finally, we have PonderNet. We use a `PyTorch Lichtning` module,
# which allows us to control all the aspects of training, validation
# and testing in the same class. Of special importance is the forward
# pass; for the sake of simplicity, we decided to implement a hardcoded
# maximum number of steps approach instead of a threshold on the
# cumulative probability of halting.

class ResnetCIFAR(pl.LightningModule):
    '''
        PonderNet variant to perform image classification on MNIST. It is capable of
        adaptively choosing the number of steps for which to process an input.

        Parameters
        ----------
        n_hidden : int
            Hidden layer size of the propagated hidden state.

        lr : float
            Learning rate.

        Modules
        -------
        ouptut_layer : nn.Linear
            Linear module that serves as a multi-class classifier.

        lambda_layer : nn.Linear
            Linear module that generates the halting probability at each step.
    '''

    def __init__(self, num_classes, lr, momentum, weight_decay):

        super().__init__()

        # attributes
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # modules
        self.core = ResNet18()

        # metrics
        self.accuracy = torchmetrics.Accuracy()

        # save hparams on W&B
        self.save_hyperparameters()

    def forward(self, img):
        # resnet
        x = self.core(img)
        out = F.log_softmax(x, dim=1)

        return out

    def training_step(self, batch, batch_idx):
        '''
            Perform the training step.

            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Current training batch to train on.

            Returns
            -------
            loss : torch.Tensor
                Loss value of the current batch.
        '''
        #loss, _, acc = self._get_loss_and_metrics(batch)

        # extract the batch
        data, target = batch

        # calculate the loss
        logits = F.log_softmax(self.core(data), dim=1)
        loss =  F.nll_loss(logits, target)

        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, target)

        # logging
        self.log('train/accuracy', acc)
        self.log('train/total_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        '''
            Perform the validation step. Logs relevant metrics and returns
            the predictions to be used in a custom callback.

            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Current validation batch to evaluate.

            Returns
            -------
            preds : torch.Tensor
                Predictions for the current batch.
        '''
        loss, preds, acc = self._get_loss_and_metrics(batch)

        # logging
        self.log('val/accuracy', acc)
        self.log('val/total_loss', loss)

        # for custom callback
        return preds

    def test_step(self, batch, batch_idx, dataset_idx=0):
        '''
            Perform the test step. Returns relevant metrics.

            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Current teest batch to evaluate.

            Returns
            -------
            acc : torch.Tensor
                Accuracy for the current batch.
        '''
        _, _, acc = self._get_loss_and_metrics(batch)

        # logging
        self.log(f'test_{dataset_idx}/accuracy', acc)

    def configure_optimizers(self):
        '''
            Configure the optimizers and learning rate schedulers.

            Returns
            -------
            config : dict
                Dictionary with `optimizer` and `lr_scheduler` keys, with an
                optimizer and a learning scheduler respectively.
        '''
        #optimizer = Adam(self.parameters(), lr=self.lr)
        optimizer = SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode='max', verbose=True),
                "monitor": 'val/accuracy',
                "interval": 'epoch',
                "frequency": 1
            }
        }

    def configure_callbacks(self):
        '''returns a list of callbacks'''
        # we choose high patience sine we validate 4 times per epoch to have nice graphs
        early_stopping = EarlyStopping(monitor='val/accuracy',
                                       mode='max',
                                       patience=100)

        if not os.path.isdir('./model_checkpoint'):
            os.mkdir('./model_checkpoint')

        model_checkpoint = ModelCheckpoint(dirpath ='./model_checkpoint',
                                           monitor ="val/accuracy",
                                           mode    ='max')
        # pondernet-{epoch:02d}-{val/loss:.2f}
        return [early_stopping, model_checkpoint]


    def _get_loss_and_metrics(self, batch):
        '''
            Returns the losses, the predictions, the accuracy and the number of steps.

            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Batch to process.

            Returns
            -------
            loss : Loss
                Loss object from which all three losses can be retrieved.

            preds : torch.Tensor
                Predictions for the current batch.

            acc : torch.Tensor
                Accuracy obtained with the current batch.
        '''

        # extract the batch
        data, target = batch

        # calculate the loss
        logits = self(data)
        loss =  F.nll_loss(logits, target)

        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, target)

        return loss, preds, acc


# //////////////////////////////////////////////
# ////////// PONDERNET IMPLEMENTATION //////////
# //////////////////////////////////////////////


# ==============================================
# SETUP AND IMPORTS
# ==============================================

# import Libraries
import os
import time

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
# AUXILIARY NETWORKS
# ==============================================

# ----------------------------------------------
# ------------- ResNet in PyTorch --------------

# For Pre-activation ResNet, see 'preact_resnet.py'.

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
    def __init__(self, block, num_blocks, num_classes=10):
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
        # out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# ----------------------------------------------
# ----------------- Total loss -----------------

# Here we define the two terms in the loss, namely the reconstruction
# term and the regularization term. We create a class to wrap them.
# ----------------------------------------------

class ReconstructionLoss(nn.Module):
    '''
        Computes the weighted average of the given loss across steps according to
        the probability of stopping at each step.

        Parameters
        ----------
        loss_func : callable
            Loss function accepting true and predicted labels. It should output
            a loss item for each element in the input batch.
    '''

    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, p: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor):
        '''
            Compute the loss.

            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, of shape `(max_steps, batch_size)`.

            y_pred : torch.Tensor
                Predicted outputs, of shape `(max_steps, batch_size)`.

            y : torch.Tensor
                True targets, of shape `(batch_size)`.

            Returns
            -------
            total_loss : torch.Tensor
                Scalar representing the reconstruction loss.
        '''
        total_loss = p.new_tensor(0.)

        for n in range(p.shape[0]):
            loss = (p[n] * self.loss_func(y_pred[n], y)).mean()
            total_loss = total_loss + loss

        return total_loss

class RegularizationLoss(nn.Module):
    '''
        Computes the KL-divergence between the halting distribution generated
        by the network and a geometric distribution with parameter `lambda_p`.

        Parameters
        ----------
        lambda_p : float
            Parameter determining our prior geometric distribution.

        max_steps : int
            Maximum number of allowed pondering steps.
    '''

    def __init__(self, lambda_p: float, max_steps: int = 1_000, device=None):
        super().__init__()

        p_g = torch.zeros((max_steps,), device=device)
        not_halted = 1.

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p: torch.Tensor):
        '''
            Compute the loss.

            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, representing our
                halting distribution.

            Returns
            -------
            loss : torch.Tensor
                Scalar representing the regularization loss.
        '''
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        return self.kl_div(p.log(), p_g)

class Loss:
    '''
        Class to group the losses together and calculate the total loss.

        Parameters
        ----------
        rec_loss : torch.Tensor
            Reconstruction loss obtained from running the network.

        reg_loss : torch.Tensor
            Regularization loss obtained from running the network.

        beta : float
            Hyperparameter to calculate the total loss.
    '''

    def __init__(self, rec_loss, reg_loss, beta):
        self.rec_loss = rec_loss
        self.reg_loss = reg_loss
        self.beta = beta

    def get_rec_loss(self):
        '''returns the reconstruciton loss'''
        return self.rec_loss

    def get_reg_loss(self):
        '''returns the regularization loss'''
        return self.reg_loss

    def get_total_loss(self):
        '''returns the total loss'''
        return self.rec_loss + self.beta * self.reg_loss


# ==============================================
# PONDERNET
# ==============================================

# Finally, we have PonderNet. We use a `PyTorch Lichtning` module,
# which allows us to control all the aspects of training, validation
# and testing in the same class. Of special importance is the forward
# pass; for the sake of simplicity, we decided to implement a hardcoded
# maximum number of steps approach instead of a threshold on the
# cumulative probability of halting.

class PonderCIFAR(pl.LightningModule):
    '''
        PonderNet variant to perform image classification on MNIST. It is capable of
        adaptively choosing the number of steps for which to process an input.

        Parameters
        ----------
        n_hidden : int
            Hidden layer size of the propagated hidden state.

        max_steps : int
            Maximum number of steps the network is allowed to "ponder" for.

        lambda_p : float 
            Parameter of the geometric prior. Must be between 0 and 1.

        beta : float
            Hyperparameter to calculate the total loss.

        lr : float
            Learning rate.

        Modules
        -------
        ouptut_layer : nn.Linear
            Linear module that serves as a multi-class classifier.

        lambda_layer : nn.Linear
            Linear module that generates the halting probability at each step.
    '''

    def __init__(self, n_elems, n_hidden, max_steps, lambda_p, beta, lr, momentum, weight_decay):
        
        super().__init__()

        # attributes
        self.n_classes = 100
        self.n_elems = n_elems
        self.max_steps = max_steps
        self.lambda_p = lambda_p
        self.beta = beta
        self.n_hidden = n_hidden
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # modules
        # self.cnn = CNN(n_input=28, kernel_size=kernel_size, n_output=n_hidden_cnn)
        # self.mlp = MLP(n_input=n_hidden_cnn + n_hidden, n_hidden=n_hidden_lin, n_output=n_hidden)
        self.gru = nn.GRUCell(self.n_elems, self.n_hidden)
        self.core = ResNet18()
        self.outpt_layer = nn.Linear(self.n_hidden, self.n_classes)
        self.lambda_layer = nn.Linear(self.n_hidden, 1)

        # losses
        self.loss_rec = ReconstructionLoss(nn.CrossEntropyLoss())
        self.loss_reg = RegularizationLoss(self.lambda_p, max_steps=self.max_steps, device=self.device)

        # metrics
        self.accuracy = torchmetrics.Accuracy()

        # save hparams on W&B
        self.save_hyperparameters()
        

    def forward(self, img):
        '''
            Run the forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Batch of input features of shape `(batch_size, n_elems)`.

            Returns
            -------
            y : torch.Tensor
                Tensor of shape `(max_steps, batch_size)` representing
                the predictions for each step and each sample. In case
                `allow_halting=True` then the shape is
                `(steps, batch_size)` where `1 <= steps <= max_steps`.

            p : torch.Tensor
                Tensor of shape `(max_steps, batch_size)` representing
                the halting probabilities. Sums over rows (fixing a sample)
                are 1. In case `allow_halting=True` then the shape is
                `(steps, batch_size)` where `1 <= steps <= max_steps`.

            halting_step : torch.Tensor
                An integer for each sample in the batch that corresponds to
                the step when it was halted. The shape is `(batch_size,)`. The
                minimal value is 1 because we always run at least one step.
        '''
        # extract batch size for QoL
        batch_size = img.shape[0]

        # propagate to get h_1
        x = self.core(img)
        h = x.new_zeros((x.shape[0], self.n_hidden))
        h = self.gru(x, h)

        # lists to save p_n, y_n
        p = []
        y = []

        # vectors to save intermediate values
        un_halted_prob = h.new_ones((batch_size,))  # unhalted probability till step n
        halting_step = h.new_zeros((batch_size,), dtype=torch.long)  # stopping step

        # main loop
        for n in range(1, self.max_steps + 1):
            # obtain lambda_n
            if n == self.max_steps:
                lambda_n = h.new_ones(batch_size) # torch | is not related to 'h'
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(h)).squeeze() 

            # obtain output and p_n
            y_n = self.outpt_layer(h)
            p_n = un_halted_prob * lambda_n

            # append p_n, y_n
            p.append(p_n)
            y.append(y_n)

            # calculate halting step
            halting_step = torch.maximum(
                n
                * (halting_step == 0)
                * torch.bernoulli(lambda_n).to(torch.long),
                halting_step)

            # track unhalted probability and flip coin to halt
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # propagate to obtain h_n
            x = self.core(img)
            h = self.gru(x, h)

            # break if we are in inference and all elements have halting_step
            if not self.training and (halting_step > 0).sum() == batch_size:
                break

        return torch.stack(y), torch.stack(p), halting_step

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
        loss, _, acc, steps = self._get_loss_and_metrics(batch)

        # logging
        self.log('train/steps', steps)
        self.log('train/accuracy', acc)
        self.log('train/total_loss', loss.get_total_loss())
        self.log('train/reconstruction_loss', loss.get_rec_loss())
        self.log('train/regularization_loss', loss.get_reg_loss())

        return loss.get_total_loss()

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
        loss, preds, acc, steps = self._get_loss_and_metrics(batch)

        # logging
        self.log('val/steps', steps)
        self.log('val/accuracy', acc)
        self.log('val/total_loss', loss.get_total_loss())
        self.log('val/reconstruction_loss', loss.get_rec_loss())
        self.log('val/regularization_loss', loss.get_reg_loss())

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

            steps : torch.Tensor
                Average number of steps for the current batch.
        '''
        _, _, acc, steps = self._get_loss_and_metrics(batch)

        # logging
        self.log(f'test_{dataset_idx}/steps', steps)
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
                "scheduler": ReduceLROnPlateau(optimizer, mode='max', verbose=True, threshold=1e-3),
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

        timestr = time.strftime("%Y%m%d-%H%M%S")
        model_checkpoint = ModelCheckpoint(dirpath ='CIFAR100_checkpoint/',
                                           monitor ="val/accuracy",
                                           filename="pondernet-" + timestr + "-{epoch:02d}",
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

            steps : torch.Tensor
                Average number of steps in the current batch.
        '''
        # extract the batch
        data, target = batch

        # forward pass
        y, p, halted_step = self(data)

        # remove elements with infinities (after taking the log)
        if torch.any(p == 0) and self.training:
            valid_indices = torch.all(p != 0, dim=0)
            p = p[:, valid_indices]
            y = y[:, valid_indices]
            halted_step = halted_step[valid_indices]
            target = target[valid_indices]

        # calculate the loss
        loss_rec_ = self.loss_rec(p, y, target)
        loss_reg_ = self.loss_reg(p)
        loss = Loss(loss_rec_, loss_reg_, self.beta)

        halted_index = (halted_step - 1).unsqueeze(0).unsqueeze(2).repeat(1, 1, self.n_classes)

        # calculate the accuracy
        logits = y.gather(dim=0, index=halted_index).squeeze()
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)

        # calculate the average number of steps
        steps = (halted_step * 1.0).mean()

        return loss, preds, acc, steps

from __future__ import print_function

import argparse
import os
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from pondernet import *
from resnet import *
from cifardata import *
from utils import progress_bar



parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default='PonderCIFAR', choices=['PonderCIFAR', 'ResnetCIFAR'])
parser.add_argument('--robust', type=str, default='augmix')

def test(net, test_loader):
    """Evaluate network on given dataset."""
    '''
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            if args.model == 'PonderCIFAR':
                (outputs_p, outputs_y, p_m, logits) = net(images)
            else:
                (outputs_p, outputs_y, p_m, logits, prediction, actual) = net(images)
            #logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)
    '''

    net.eval()
    total_loss = 0.
    total_correct = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            if args.model == 'PonderCIFAR':
                (y, p, halted_step) = net(data)
                halted_index = (halted_step - 1).unsqueeze(0).unsqueeze(2).repeat(1, 1, n_classes=100)
                logits = y.gather(dim=0, index=halted_index).squeeze()
            else:
                (outputs_p, outputs_y, p_m, logits, prediction, actual) = net(images)
            #logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)


def test_c(net, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=100,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

        test_loss, test_acc = test(net, test_loader)
        corruption_accs.append(test_acc)
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))
    return np.mean(corruption_accs)

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

def main():
    torch.manual_seed(1)
    np.random.seed(1)


    # Load datasets
    preprocess = transforms.Compose([transforms.ToTensor(),])


    test_transform = preprocess

    if args.model == 'PonderCIFAR':
        model = PonderCIFAR(n_elems=512, n_hidden=100, max_steps=10).to('cuda')
        checkpoint = torch.load('./checkpoint/ckpt_colorjitter_PonderGRU_10step_0.3.pth')
        model.load_state_dict(checkpoint['net'])
    elif args.model == 'ResnetCIFAR':
        model = ResnetCIFAR(n_elems=512, n_hidden=512, max_steps=10).to('cuda')
        checkpoint = torch.load('./checkpoint/ckpt_colorjitter_PonderGRUAction_10step_0.3.pth')
        model.load_state_dict(checkpoint['net'])



    test_data = torchvision.datasets.CIFAR100('./data/cifar', train=False, transform=test_transform, download=True)
    base_c_path = '../CIFAR-100-C/'
    print('Model: {}, Robustness Method: {}'.format(args.model, args.robust))
    test_c_acc = test_c(model, test_data, base_c_path)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))


if __name__ == '__main__':
    main()

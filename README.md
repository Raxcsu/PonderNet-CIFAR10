# RESNET18 + PonderNet

This code is adopted from one of the [PyTorch Lightning examples](https://colab.research.google.com/drive/1Tr9dYlwBKk6-LgLKGO8KYZULnguVA992?usp=sharing#scrollTo=CxXtBfFrKYgA) and [this PonderNet implementation](https://nn.labml.ai/adaptive_computation/ponder_net/index.html).

## Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

### Prerequisites
- Python 3.6+
- PyTorch 1.0+

### Training
```
# Start training with: 
python main.py
```

### Accuracy
INTERPOLATION TASK
| Model             | Acc.        |
| ----------------- | ----------- |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| ResNet18 + PonderNet (epochs=10)          | 83.24%      |
| ResNet18 + PonderNet (epochs=50)          | 90.96%      |
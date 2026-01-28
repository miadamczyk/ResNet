# ResNet CIFAR Implementation

This repository provides a PyTorch implementation of the ResNet architecture based on the original paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).

The code supports training and evaluation on the CIFAR-10 and CIFAR-100 datasets, with configurable network depth (e.g., ResNet-34, ResNet-56) and hyperparameters such as learning rate, batch size, and number of epochs.

### Features
- Optional WandB integration for experiment tracking
- Saving checkpoints based on validation accuracy
- Configurable through command-line arguments for flexibility 

This repository is intended for experimentation and ResNet training on CIFAR datasets.

## Description of Files

### `dataset.py`

Provides a utility function `get_cifar_loaders` to load CIFAR-10 or CIFAR-100 datasets.  

Steps:  
- Splits training data into train and validation sets  
- Applies standard data augmentations during training (random crop + horizontal flip + normalization)  
- Normalizes test/validation data using dataset-specific mean and standard deviation  
- Returns PyTorch DataLoaders for training, validation, and testing

### `model.py`

Contains the implementation of a simple ResNet architecture for CIFAR datasets.

Components:
- `BasicBlock`: Standard residual block with two 3x3 convolutions and optional downsampling
- `ResNet`: Flexible ResNet class that can build networks of arbitrary depth by specifying the number of blocks per stage
- Supports small input sizes (CIFAR-style) with adjustable initial channels
- Includes weight initialization using Kaiming normalization for convolution layers and setting batch norm layers to identity
- Provides helper functions `resnet34` and `resnet56` to quickly instantiate common ResNet variants

### `main.py`

Handles the training and evaluation loop.

Features:
- Parses command-line arguments to configure model, dataset, optimizer, scheduler, and WandB logging
- Implements `train_one_epoch` and `validate` functions for training and evaluation
- Supports learning rate scheduling with `StepLR`
- Saves the best model checkpoint automatically
- Optionally logs metrics and models to WandB
- Loads the best model at the end to compute final test accuracy

### Example Run

To train a ResNet-34 on CIFAR100 with WandB logging:

```bash
python main.py --arch resnet34 --dataset CIFAR100 --wandb
```

## Dependencies and Installation

This project requires Python 3.10+ and the following libraries:

- `torch>=2.0.0` — PyTorch deep learning framework  
- `torchvision>=0.15.0` — Computer vision utilities for PyTorch  
- `wandb>=0.15.0` — Weights & Biases for experiment tracking  
- `numpy>=1.25.0` — Numerical computations

### Installing via `requirements.txt`

1. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

## Installation via Conda

Create a Conda environment from the `environment.yaml` file and activate it:

```bash
conda env create -f environment.yaml
conda activate resnet
```

## Results

The training and evaluation results for ResNet-34 and ResNet-56 implemented in this repository, run on CIFAR-100 with the default training arguments, can be viewed on Weights & Biases:

https://api.wandb.ai/links/milosz-adamczyk2002/ujekbmck

The runs include logged training and validation loss/accuracy curves, learning rate schedules, and final test performance for both architectures.


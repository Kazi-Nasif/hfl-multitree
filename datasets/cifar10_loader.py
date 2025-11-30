"""
CIFAR-10 Dataset Loader
Standard FL benchmark with 50,000 training images
"""
import torch
from torchvision import datasets, transforms
import numpy as np


def get_cifar10(data_dir='./data'):
    """
    Load CIFAR-10 dataset
    
    Returns:
        train_dataset, test_dataset
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    
    return train_dataset, test_dataset


def get_dataset_info():
    """Return CIFAR-10 dataset information"""
    return {
        'name': 'CIFAR-10',
        'num_classes': 10,
        'input_shape': (3, 32, 32),
        'train_size': 50000,
        'test_size': 10000,
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    }

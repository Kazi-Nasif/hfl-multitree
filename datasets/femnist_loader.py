"""
FEMNIST Dataset Loader
Federated Extended MNIST - 62 classes (digits + upper + lower case letters)
Naturally non-IID as each writer has different handwriting
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from pathlib import Path


class FEMNISTDataset(Dataset):
    """FEMNIST dataset"""
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx].reshape(28, 28).astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            img = self.transform(img)
        
        target = self.targets[idx]
        return img, target


def get_femnist(data_dir='./data/femnist'):
    """
    Load FEMNIST dataset
    Note: This creates a synthetic version for testing
    For real FEMNIST, download from: https://leaf.cmu.edu/
    
    Returns:
        train_dataset, test_dataset
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # For now, create synthetic FEMNIST-like data
    # In production, replace with actual FEMNIST download
    print("Creating synthetic FEMNIST data...")
    print("For real FEMNIST, download from https://leaf.cmu.edu/")
    
    # Synthetic data: 62 classes, 28x28 images
    num_train = 50000
    num_test = 10000
    
    train_data = np.random.randn(num_train, 28, 28).astype(np.float32)
    train_targets = np.random.randint(0, 62, num_train)
    
    test_data = np.random.randn(num_test, 28, 28).astype(np.float32)
    test_targets = np.random.randint(0, 62, num_test)
    
    train_dataset = FEMNISTDataset(train_data, train_targets)
    test_dataset = FEMNISTDataset(test_data, test_targets)
    
    return train_dataset, test_dataset


def get_dataset_info():
    """Return FEMNIST dataset information"""
    return {
        'name': 'FEMNIST',
        'num_classes': 62,
        'input_shape': (1, 28, 28),
        'train_size': 50000,  # Approximate
        'test_size': 10000,
        'classes': 'digits + letters (62 total)'
    }

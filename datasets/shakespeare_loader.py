"""
Shakespeare Dataset Loader
Character-level language modeling task
Naturally non-IID as each speaking role has different language patterns
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import string


class ShakespeareDataset(Dataset):
    """Shakespeare character-level dataset"""
    
    def __init__(self, data, seq_length=80):
        self.data = data
        self.seq_length = seq_length
        
        # Character vocabulary (80 printable characters)
        self.chars = list(string.printable)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        
        # Convert to indices
        indices = [self.char_to_idx.get(c, 0) for c in chunk]
        
        x = torch.tensor(indices[:-1], dtype=torch.long)
        y = torch.tensor(indices[1:], dtype=torch.long)
        
        return x, y


def get_shakespeare(data_dir='./data/shakespeare', seq_length=80):
    """
    Load Shakespeare dataset
    Note: This creates a synthetic version for testing
    For real Shakespeare, download from: https://leaf.cmu.edu/
    
    Returns:
        train_dataset, test_dataset
    """
    print("Creating synthetic Shakespeare data...")
    print("For real Shakespeare, download from https://leaf.cmu.edu/")
    
    # Synthetic Shakespeare-like text
    sample_text = """
    To be or not to be, that is the question.
    Whether tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles.
    """ * 1000  # Repeat to get more data
    
    # Split train/test
    split = int(len(sample_text) * 0.8)
    train_text = sample_text[:split]
    test_text = sample_text[split:]
    
    train_dataset = ShakespeareDataset(train_text, seq_length)
    test_dataset = ShakespeareDataset(test_text, seq_length)
    
    return train_dataset, test_dataset


def get_dataset_info():
    """Return Shakespeare dataset information"""
    return {
        'name': 'Shakespeare',
        'num_classes': 80,  # Vocabulary size
        'task': 'next character prediction',
        'seq_length': 80,
        'vocab_size': 80
    }

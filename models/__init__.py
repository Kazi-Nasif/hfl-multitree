"""
Neural network models for federated learning
"""
from .cnn_cifar import CNNCifar
from .cnn_femnist import CNNFemnist
from .lstm_shakespeare import LSTMShakespeare

__all__ = ['CNNCifar', 'CNNFemnist', 'LSTMShakespeare']

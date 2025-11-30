"""
Dataset loaders for federated learning benchmarks
"""
from .cifar10_loader import get_cifar10
from .femnist_loader import get_femnist
from .shakespeare_loader import get_shakespeare
from .partitioner import partition_data

__all__ = ['get_cifar10', 'get_femnist', 'get_shakespeare', 'partition_data']

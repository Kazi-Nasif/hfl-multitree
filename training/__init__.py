"""
Federated learning training components
"""
from .fl_trainer import FederatedTrainer
from .local_trainer import LocalTrainer

__all__ = ['FederatedTrainer', 'LocalTrainer']

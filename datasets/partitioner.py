"""
Data partitioning for IID and Non-IID scenarios
"""
import numpy as np
import torch
from torch.utils.data import Subset
from collections import defaultdict


class DataPartitioner:
    """Partition data among federated clients"""
    
    def __init__(self, dataset, num_clients, partition_type='iid', 
                 alpha=0.5, classes_per_client=2):
        """
        Args:
            dataset: PyTorch dataset
            num_clients: Number of federated clients
            partition_type: 'iid', 'niid_label', 'niid_dirichlet'
            alpha: Dirichlet concentration parameter (for dirichlet)
            classes_per_client: Number of classes per client (for label skew)
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.alpha = alpha
        self.classes_per_client = classes_per_client
        
    def partition(self):
        """Partition dataset"""
        if self.partition_type == 'iid':
            return self._partition_iid()
        elif self.partition_type == 'niid_label':
            return self._partition_niid_label_skew()
        elif self.partition_type == 'niid_dirichlet':
            return self._partition_niid_dirichlet()
        else:
            raise ValueError(f"Unknown partition type: {self.partition_type}")
    
    def _partition_iid(self):
        """IID partitioning - random uniform distribution"""
        print(f"Creating IID partition for {self.num_clients} clients...")
        
        num_samples = len(self.dataset)
        indices = np.random.permutation(num_samples)
        
        # Split evenly among clients
        samples_per_client = num_samples // self.num_clients
        client_indices = {}
        
        for i in range(self.num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < self.num_clients - 1 else num_samples
            client_indices[i] = indices[start:end].tolist()
        
        print(f"  Samples per client: ~{samples_per_client}")
        return client_indices
    
    def _partition_niid_label_skew(self):
        """Non-IID partitioning - label skew (each client has limited classes)"""
        print(f"Creating Non-IID (label skew) partition...")
        print(f"  Classes per client: {self.classes_per_client}")
        
        # Group indices by label
        label_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset):
            label_indices[label].append(idx)
        
        num_classes = len(label_indices)
        client_indices = {i: [] for i in range(self.num_clients)}
        
        # Assign classes to clients
        for client_id in range(self.num_clients):
            # Random select classes for this client
            selected_classes = np.random.choice(
                num_classes, 
                self.classes_per_client, 
                replace=False
            )
            
            # Distribute samples from selected classes
            for class_id in selected_classes:
                class_samples = label_indices[class_id]
                # Give portion of this class to current client
                samples_to_take = len(class_samples) // (num_classes // self.classes_per_client)
                start = (client_id % (num_classes // self.classes_per_client)) * samples_to_take
                end = start + samples_to_take
                client_indices[client_id].extend(class_samples[start:end])
        
        return client_indices
    
    def _partition_niid_dirichlet(self):
        """Non-IID partitioning - Dirichlet distribution"""
        print(f"Creating Non-IID (Dirichlet α={self.alpha}) partition...")
        
        # Group indices by label
        label_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset):
            label_indices[label].append(idx)
        
        num_classes = len(label_indices)
        client_indices = {i: [] for i in range(self.num_clients)}
        
        # For each class, sample from Dirichlet and distribute
        for class_id in range(num_classes):
            class_samples = np.array(label_indices[class_id])
            np.random.shuffle(class_samples)
            
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet(
                [self.alpha] * self.num_clients
            )
            
            # Distribute samples according to proportions
            splits = (np.cumsum(proportions) * len(class_samples)).astype(int)
            splits = np.concatenate([[0], splits])
            
            for client_id in range(self.num_clients):
                start, end = splits[client_id], splits[client_id + 1]
                client_indices[client_id].extend(class_samples[start:end].tolist())
        
        return client_indices


def partition_data(dataset, num_clients, partition_type='iid', **kwargs):
    """
    Convenience function to partition data
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        partition_type: 'iid', 'niid_label', 'niid_dirichlet'
        **kwargs: Additional arguments for partitioner
    
    Returns:
        Dictionary mapping client_id to list of sample indices
    """
    partitioner = DataPartitioner(dataset, num_clients, partition_type, **kwargs)
    return partitioner.partition()

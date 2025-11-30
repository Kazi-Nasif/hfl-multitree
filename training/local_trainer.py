"""
Local training for each client
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import copy
import time


class LocalTrainer:
    """Trains model locally on client data"""
    
    def __init__(self, model, device='cuda', lr=0.01, batch_size=32):
        """
        Args:
            model: PyTorch model
            device: 'cuda' or 'cpu'
            lr: Learning rate
            batch_size: Batch size
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        
        self.model.to(device)
        
    def train(self, dataset, client_indices, num_epochs=1):
        """
        Train on local data
        
        Args:
            dataset: Full dataset
            client_indices: Indices for this client
            num_epochs: Number of local epochs
            
        Returns:
            Updated model, metrics dict
        """
        # Create local dataset
        local_dataset = Subset(dataset, client_indices)
        train_loader = DataLoader(
            local_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Optimizer and loss
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        train_time = time.time() - start_time
        
        metrics = {
            'loss': total_loss / len(train_loader) if len(train_loader) > 0 else 0.0,
            'accuracy': 100. * correct / total if total > 0 else 0.0,
            'train_time': train_time,
            'samples': len(client_indices)
        }
        
        return self.model, metrics
    
    def evaluate(self, dataset, indices=None):
        """
        Evaluate model
        
        Args:
            dataset: Dataset to evaluate on
            indices: Specific indices (None = use all)
            
        Returns:
            Metrics dict
        """
        if indices is not None:
            eval_dataset = Subset(dataset, indices)
        else:
            eval_dataset = dataset
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        metrics = {
            'loss': total_loss / len(eval_loader) if len(eval_loader) > 0 else 0.0,
            'accuracy': 100. * correct / total if total > 0 else 0.0
        }
        
        return metrics

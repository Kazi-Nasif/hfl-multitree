"""
LSTM for Shakespeare character-level language modeling
"""
import torch
import torch.nn as nn


class LSTMShakespeare(nn.Module):
    """LSTM for next character prediction"""
    
    def __init__(self, vocab_size=80, embedding_dim=8, hidden_dim=256, num_layers=2):
        super(LSTMShakespeare, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        # Embedding
        x = self.embedding(x)
        
        # LSTM
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        
        # Output layer
        out = self.fc(out)
        
        return out, hidden
    
    def get_model_size(self):
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb

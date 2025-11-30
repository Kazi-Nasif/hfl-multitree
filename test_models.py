"""
Test neural network models
"""
import torch
from models import CNNCifar, CNNFemnist, LSTMShakespeare


def test_cnn_cifar():
    """Test CIFAR-10 CNN"""
    print("="*60)
    print("Testing CNN for CIFAR-10")
    print("="*60)
    
    model = CNNCifar(num_classes=10)
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model size: {model.get_model_size():.2f} MB")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ CIFAR-10 model working")


def test_cnn_femnist():
    """Test FEMNIST CNN"""
    print("\n" + "="*60)
    print("Testing CNN for FEMNIST")
    print("="*60)
    
    model = CNNFemnist(num_classes=62)
    
    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model size: {model.get_model_size():.2f} MB")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ FEMNIST model working")


def test_lstm_shakespeare():
    """Test Shakespeare LSTM"""
    print("\n" + "="*60)
    print("Testing LSTM for Shakespeare")
    print("="*60)
    
    model = LSTMShakespeare(vocab_size=80, embedding_dim=8, hidden_dim=256)
    
    # Test forward pass
    x = torch.randint(0, 80, (4, 80))  # batch=4, seq_len=80
    y, hidden = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model size: {model.get_model_size():.2f} MB")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Shakespeare model working")


def main():
    print("="*60)
    print("Model Architecture Tests")
    print("="*60)
    
    test_cnn_cifar()
    test_cnn_femnist()
    test_lstm_shakespeare()
    
    print("\n" + "="*60)
    print("✓ All model tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()

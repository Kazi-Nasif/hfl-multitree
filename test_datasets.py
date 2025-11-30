"""
Test dataset loaders and partitioning
"""
import torch
from datasets import get_cifar10, get_femnist, get_shakespeare, partition_data
from datasets.cifar10_loader import get_dataset_info as cifar_info
from datasets.femnist_loader import get_dataset_info as femnist_info
from datasets.shakespeare_loader import get_dataset_info as shakespeare_info


def test_cifar10():
    """Test CIFAR-10 loading"""
    print("="*60)
    print("Testing CIFAR-10 Dataset")
    print("="*60)
    
    train_dataset, test_dataset = get_cifar10()
    info = cifar_info()
    
    print(f"Dataset: {info['name']}")
    print(f"Classes: {info['num_classes']}")
    print(f"Input shape: {info['input_shape']}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Test sample
    img, label = train_dataset[0]
    print(f"Sample shape: {img.shape}")
    print(f"Sample label: {label} ({info['classes'][label]})")
    
    return train_dataset, test_dataset


def test_partitioning(dataset, num_clients=50):
    """Test data partitioning"""
    print("\n" + "="*60)
    print(f"Testing Data Partitioning ({num_clients} clients)")
    print("="*60)
    
    # IID partitioning
    print("\n1. IID Partitioning:")
    iid_indices = partition_data(dataset, num_clients, 'iid')
    print(f"   Client 0: {len(iid_indices[0])} samples")
    print(f"   Client 1: {len(iid_indices[1])} samples")
    
    # Non-IID label skew
    print("\n2. Non-IID (Label Skew, 2 classes/client):")
    niid_label = partition_data(dataset, num_clients, 'niid_label', classes_per_client=2)
    print(f"   Client 0: {len(niid_label[0])} samples")
    print(f"   Client 1: {len(niid_label[1])} samples")
    
    # Non-IID Dirichlet
    print("\n3. Non-IID (Dirichlet α=0.5):")
    niid_dir = partition_data(dataset, num_clients, 'niid_dirichlet', alpha=0.5)
    print(f"   Client 0: {len(niid_dir[0])} samples")
    print(f"   Client 1: {len(niid_dir[1])} samples")
    
    # Show distribution
    print("\n   Distribution across clients (IID):")
    sizes = [len(iid_indices[i]) for i in range(min(5, num_clients))]
    print(f"   First 5 clients: {sizes}")


def main():
    print("="*60)
    print("Dataset Integration Test")
    print("="*60)
    
    # Test CIFAR-10
    train_dataset, test_dataset = test_cifar10()
    
    # Test partitioning
    test_partitioning(train_dataset, num_clients=50)
    
    print("\n" + "="*60)
    print("✓ All dataset tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()

"""
Run comprehensive federated learning experiments
"""
import os
import sys
import argparse

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

from datasets import get_cifar10, partition_data
from models import CNNCifar
from training.fl_trainer import FederatedTrainer
from utils.config_loader import Config
from utils.topology import TopologyGenerator
from multitree.scheduler import MultiTreeScheduler
from simulation.communication import CommunicationSimulator
from simulation.baselines import RingSimulator


class ExperimentRunner:
    """Run and track FL experiments"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(parent_dir, 'config', 'system_config.yaml')
        self.config = Config(config_path)
        self.results_dir = Path(parent_dir) / 'results' / 'experiments'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_single_experiment(self, dataset_name='cifar10', topology_type='2D_Torus',
                             partition_type='iid', algorithm='multitree',
                             num_rounds=100, num_clients=50):
        """Run a single experiment"""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT CONFIGURATION")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Topology: {topology_type}")
        print(f"Data Distribution: {partition_type}")
        print(f"Algorithm: {algorithm}")
        print(f"Rounds: {num_rounds}")
        print(f"Clients: {num_clients}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # Load dataset
        print("Loading dataset...")
        if dataset_name == 'cifar10':
            train_dataset, test_dataset = get_cifar10()
            model = CNNCifar(num_classes=10)
        else:
            raise ValueError(f"Dataset {dataset_name} not yet implemented")
        
        # Partition data
        print(f"Partitioning data ({partition_type})...")
        client_indices = partition_data(
            train_dataset, 
            num_clients, 
            partition_type,
            alpha=0.5,
            classes_per_client=2
        )
        
        # Generate topology
        print(f"Generating {topology_type} topology...")
        self.config.config['topology']['type'] = topology_type
        topo_gen = TopologyGenerator(self.config.config)
        G = topo_gen.generate()
        
        # Create communication simulator
        if algorithm == 'multitree':
            print("Building MultiTree scheduler...")
            scheduler = MultiTreeScheduler(G, k_ary=2)
            scheduler.build_trees()
            simulator = CommunicationSimulator(G, self.config.config)
        else:
            print("Using Ring all-reduce...")
            simulator = RingSimulator(G, self.config.config)
        
        # Create federated trainer
        print("Initializing federated trainer...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        fl_config = {
            'local_epochs': 1,
            'client_lr': 0.01,
            'batch_size': 32,
            'clients_per_round': num_clients
        }
        
        trainer = FederatedTrainer(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            client_indices=client_indices,
            multitree_simulator=simulator if algorithm == 'multitree' else None,
            config=fl_config,
            device=device
        )
        
        # Run training
        print("Starting training...")
        start_time = time.time()
        history = trainer.train(num_rounds=num_rounds)
        total_time = time.time() - start_time
        
        # Save results
        results = {
            'config': {
                'dataset': dataset_name,
                'topology': topology_type,
                'partition': partition_type,
                'algorithm': algorithm,
                'num_rounds': num_rounds,
                'num_clients': num_clients,
            },
            'history': {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for v in vals] for k, vals in history.items()},
            'final_metrics': {
                'test_accuracy': float(history['test_accuracy'][-1]),
                'test_loss': float(history['test_loss'][-1]),
                'total_time': float(total_time),
            }
        }
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{dataset_name}_{topology_type}_{partition_type}_{algorithm}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run FL experiments')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset name')
    parser.add_argument('--topology', type=str, default='2D_Torus',
                       help='Network topology')
    parser.add_argument('--partition', type=str, default='iid',
                       help='Data partition type')
    parser.add_argument('--algorithm', type=str, default='multitree',
                       help='Communication algorithm')
    parser.add_argument('--rounds', type=int, default=20,
                       help='Number of training rounds')
    parser.add_argument('--clients', type=int, default=50,
                       help='Number of clients')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    results = runner.run_single_experiment(
        dataset_name=args.dataset,
        topology_type=args.topology,
        partition_type=args.partition,
        algorithm=args.algorithm,
        num_rounds=args.rounds,
        num_clients=args.clients
    )
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Final Test Accuracy: {results['final_metrics']['test_accuracy']:.2f}%")
    print(f"Final Test Loss: {results['final_metrics']['test_loss']:.4f}")
    print(f"Total Time: {results['final_metrics']['total_time']:.2f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

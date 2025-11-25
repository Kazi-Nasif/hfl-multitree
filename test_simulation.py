"""
Test Communication Simulation
"""
from utils.config_loader import Config
from utils.topology import TopologyGenerator
from multitree.scheduler import MultiTreeScheduler
from simulation.communication import CommunicationSimulator


def main():
    print("="*60)
    print("Communication Simulation Test")
    print("="*60)
    
    # Setup
    config = Config()
    topo_gen = TopologyGenerator(config.config)
    G = topo_gen.generate()
    
    # Build MultiTree
    scheduler = MultiTreeScheduler(G, k_ary=2)
    scheduler.build_trees()
    
    # Create simulator
    simulator = CommunicationSimulator(G, config.config)
    
    # Run simulation with typical ResNet-50 size (~100 MB)
    model_size = 0.1  # GB (100 MB)
    results = simulator.run_simulation(scheduler, model_size)
    
    print("\n✓ Simulation test completed!")


if __name__ == "__main__":
    main()

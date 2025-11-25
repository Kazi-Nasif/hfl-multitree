"""
Test AHFLP Optimization
"""
from utils.config_loader import Config
from utils.topology import TopologyGenerator
from multitree.scheduler import MultiTreeScheduler
from simulation.communication import CommunicationSimulator
from ahflp.optimizer import AHFLPOptimizer


def main():
    print("="*70)
    print("AHFLP Optimization Test")
    print("="*70)
    
    # Setup
    config = Config()
    topo_gen = TopologyGenerator(config.config)
    G = topo_gen.generate()
    
    # Run MultiTree simulation first
    print("\nStep 1: Running MultiTree baseline...")
    scheduler = MultiTreeScheduler(G, k_ary=2)
    scheduler.build_trees()
    
    simulator = CommunicationSimulator(G, config.config)
    mt_results = simulator.run_simulation(scheduler, model_size=0.1)
    
    # Run AHFLP optimization
    print("\nStep 2: Running AHFLP optimization...")
    optimizer = AHFLPOptimizer(config.config)
    ahflp_solution = optimizer.optimize_aggregation_schedule(
        multitree_time=mt_results['total_time'],
        multitree_energy=mt_results['avg_energy']
    )
    
    # Compare results
    print("\n" + "="*70)
    print("PERFORMANCE IMPROVEMENT")
    print("="*70)
    print(f"MultiTree alone:")
    print(f"  Time: {mt_results['total_time']*1000:.2f} ms")
    print(f"  Energy: {mt_results['avg_energy']:.2f} J")
    print(f"\nMultiTree + AHFLP:")
    print(f"  Time: {ahflp_solution['total_time']:.2f} s")
    print(f"  Energy: {ahflp_solution['total_energy']:.2f} J")
    print(f"  l1={ahflp_solution['l1']}, l2={ahflp_solution['l2']}")
    
    print("\n✓ AHFLP test completed!")


if __name__ == "__main__":
    main()

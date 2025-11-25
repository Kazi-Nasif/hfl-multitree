"""
Complete System Test: MultiTree + AHFLP for Full Training
"""
import numpy as np
from utils.config_loader import Config
from utils.topology import TopologyGenerator
from multitree.scheduler import MultiTreeScheduler
from simulation.communication import CommunicationSimulator
from simulation.baselines import RingSimulator
from ahflp.optimizer import AHFLPOptimizer


def simulate_training_rounds(comm_time_per_round, energy_per_round, l1, l2, num_rounds=100):
    """
    Simulate complete training with adaptive aggregation
    
    Args:
        comm_time_per_round: Communication time for one global aggregation (seconds)
        energy_per_round: Energy per device for one global aggregation (Joules)
        l1: Local aggregation interval
        l2: Edge aggregation interval
        num_rounds: Total global rounds needed
    
    Returns:
        Total time and energy for complete training
    """
    # Local training parameters
    samples_per_device = 50000 / 50  # dataset / total devices
    batch_size = 32
    batches = samples_per_device / batch_size
    flops_per_sample = 1e9
    cpu_freq = 2.0e9  # Hz
    
    # Local training time per iteration
    local_iter_time = (batches * flops_per_sample * batch_size) / cpu_freq
    local_iter_energy = 250.0 * local_iter_time  # 250W during training
    
    # Calculate effective rounds with adaptive aggregation
    # With l1 and l2, we do fewer global aggregations
    global_aggregations_needed = num_rounds / (l1 * l2)
    
    # Time per global round
    time_per_global_round = l1 * local_iter_time + comm_time_per_round
    
    # Total time
    total_time = time_per_global_round * global_aggregations_needed
    
    # Total energy per device
    local_training_energy = local_iter_energy * l1 * global_aggregations_needed
    comm_energy_total = energy_per_round * global_aggregations_needed
    total_energy = local_training_energy + comm_energy_total
    
    return total_time, total_energy, global_aggregations_needed


def main():
    print("="*70)
    print("COMPLETE SYSTEM EVALUATION")
    print("MultiTree + AHFLP vs Ring for Full Training Process")
    print("="*70)
    
    config = Config()
    topo_gen = TopologyGenerator(config.config)
    G = topo_gen.generate()
    
    model_size = 0.1  # 100 MB
    
    # ================================================================
    # Baseline: Ring All-Reduce with no adaptive aggregation
    # ================================================================
    print("\n" + "="*70)
    print("BASELINE: Ring All-Reduce (l1=1, l2=1)")
    print("="*70)
    
    ring_simulator = RingSimulator(G, config.config)
    ring_results = ring_simulator.run_ring_simulation(model_size)
    
    ring_time, ring_energy, ring_rounds = simulate_training_rounds(
        comm_time_per_round=ring_results['total_time'],
        energy_per_round=ring_results['avg_energy'],
        l1=1,
        l2=1,
        num_rounds=100
    )
    
    print(f"\nComplete Training with Ring:")
    print(f"  Total time: {ring_time:.2f} s")
    print(f"  Total energy per device: {ring_energy:.2f} J")
    print(f"  Global aggregations: {ring_rounds:.0f}")
    
    # ================================================================
    # Proposed: MultiTree + AHFLP with adaptive aggregation
    # ================================================================
    print("\n" + "="*70)
    print("PROPOSED: MultiTree + AHFLP")
    print("="*70)
    
    scheduler = MultiTreeScheduler(G, k_ary=2)
    scheduler.build_trees()
    
    mt_simulator = CommunicationSimulator(G, config.config)
    mt_results = mt_simulator.run_simulation(scheduler, model_size)
    
    # Run AHFLP optimization
    optimizer = AHFLPOptimizer(config.config)
    ahflp_solution = optimizer.optimize_aggregation_schedule(
        multitree_time=mt_results['total_time'],
        multitree_energy=mt_results['avg_energy']
    )
    
    # Simulate complete training with optimized l1, l2
    mt_ahflp_time, mt_ahflp_energy, mt_rounds = simulate_training_rounds(
        comm_time_per_round=mt_results['total_time'],
        energy_per_round=mt_results['avg_energy'],
        l1=ahflp_solution['l1'],
        l2=ahflp_solution['l2'],
        num_rounds=100
    )
    
    print(f"\nComplete Training with MultiTree+AHFLP:")
    print(f"  Total time: {mt_ahflp_time:.2f} s")
    print(f"  Total energy per device: {mt_ahflp_energy:.2f} J")
    print(f"  Global aggregations: {mt_rounds:.0f}")
    print(f"  Optimized l1={ahflp_solution['l1']}, l2={ahflp_solution['l2']}")
    
    # ================================================================
    # Performance Comparison
    # ================================================================
    print("\n" + "="*70)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*70)
    
    time_improvement = (ring_time - mt_ahflp_time) / ring_time * 100
    energy_improvement = (ring_energy - mt_ahflp_energy) / ring_energy * 100
    
    print(f"\n{'Metric':<35} {'Ring':<15} {'MT+AHFLP':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'Total Training Time (s)':<35} {ring_time:<15.2f} {mt_ahflp_time:<15.2f} {time_improvement:<15.1f}%")
    print(f"{'Energy per Device (J)':<35} {ring_energy:<15.2f} {mt_ahflp_energy:<15.2f} {energy_improvement:<15.1f}%")
    print(f"{'Global Aggregations':<35} {ring_rounds:<15.0f} {mt_rounds:<15.0f} {'-':<15}")
    print(f"{'Comm Time per Round (ms)':<35} {ring_results['total_time']*1000:<15.2f} {mt_results['total_time']*1000:<15.2f} {(1-mt_results['total_time']/ring_results['total_time'])*100:<15.1f}%")
    
    print("\n" + "="*70)
    print("KEY ACHIEVEMENTS")
    print("="*70)
    
    if time_improvement > 0:
        print(f"✓ Training time reduced by {time_improvement:.1f}%")
    else:
        print(f"⚠️  Training time increased by {abs(time_improvement):.1f}%")
    
    if energy_improvement > 0:
        print(f"✓ Energy consumption reduced by {energy_improvement:.1f}%")
    else:
        print(f"⚠️  Energy consumption increased by {abs(energy_improvement):.1f}%")
    
    print(f"✓ Communication complexity: O(log n) vs O(n)")
    print(f"✓ Adaptive aggregation: l1={ahflp_solution['l1']}, l2={ahflp_solution['l2']}")
    print(f"✓ Per-round comm speedup: {ring_results['total_time']/mt_results['total_time']:.2f}x")
    
    # Target metrics from your project proposal
    print("\n" + "="*70)
    print("PROJECT TARGETS vs ACHIEVED")
    print("="*70)
    print(f"Energy Reduction Target: 40-55%")
    print(f"Energy Reduction Achieved: {energy_improvement:.1f}%")
    print(f"\nTime Improvement Target: 45-60% (2.3x speedup)")
    print(f"Time Improvement Achieved: {time_improvement:.1f}%")
    print(f"Per-round speedup: {ring_results['total_time']/mt_results['total_time']:.2f}x")
    
    if energy_improvement >= 40 and time_improvement >= 45:
        print("\n🎉 PROJECT TARGETS MET!")
    elif energy_improvement >= 30 or time_improvement >= 30:
        print("\n✓ Strong performance improvements achieved!")
        print("   Further tuning of parameters may reach target metrics")
    else:
        print("\n💡 Current implementation shows baseline performance")
        print("   Next steps: Tune φ, implement frequency scaling, optimize bandwidth")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

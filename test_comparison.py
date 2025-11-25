"""
Compare MultiTree vs Ring All-Reduce
"""
from utils.config_loader import Config
from utils.topology import TopologyGenerator
from multitree.scheduler import MultiTreeScheduler
from simulation.communication import CommunicationSimulator
from simulation.baselines import RingSimulator


def main():
    print("="*70)
    print("MultiTree vs Ring All-Reduce Comparison")
    print("="*70)
    
    # Setup
    config = Config()
    topo_gen = TopologyGenerator(config.config)
    G = topo_gen.generate()
    
    model_size = 0.1  # 100 MB (typical ResNet-50)
    
    # ============================================================
    # MultiTree All-Reduce
    # ============================================================
    print("\n" + "="*70)
    print("PART 1: MultiTree All-Reduce")
    print("="*70)
    
    scheduler = MultiTreeScheduler(G, k_ary=2)
    scheduler.build_trees()
    
    mt_simulator = CommunicationSimulator(G, config.config)
    mt_results = mt_simulator.run_simulation(scheduler, model_size)
    
    # ============================================================
    # Ring All-Reduce
    # ============================================================
    print("\n" + "="*70)
    print("PART 2: Ring All-Reduce")
    print("="*70)
    
    ring_simulator = RingSimulator(G, config.config)
    ring_results = ring_simulator.run_ring_simulation(model_size)
    
    # ============================================================
    # Comparison
    # ============================================================
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    time_speedup = ring_results['total_time'] / mt_results['total_time']
    energy_reduction = (ring_results['avg_energy'] - mt_results['avg_energy']) / ring_results['avg_energy'] * 100
    
    print(f"\n{'Metric':<30} {'Ring':<20} {'MultiTree':<20} {'Improvement':<15}")
    print("-"*70)
    print(f"{'Total Time (ms)':<30} {ring_results['total_time']*1000:<20.2f} {mt_results['total_time']*1000:<20.2f} {time_speedup:<15.2f}x")
    print(f"{'Max Energy (J)':<30} {ring_results['max_energy']:<20.2f} {mt_results['max_energy']:<20.2f} {(ring_results['max_energy']-mt_results['max_energy'])/ring_results['max_energy']*100:<15.1f}%")
    print(f"{'Avg Energy (J)':<30} {ring_results['avg_energy']:<20.2f} {mt_results['avg_energy']:<20.2f} {energy_reduction:<15.1f}%")
    print(f"{'Comm Steps':<30} {ring_results['num_steps']:<20} {int(mt_results['total_time']/mt_results['max_comm_time']*ring_results['num_steps']):<20} {'-':<15}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"✓ MultiTree is {time_speedup:.2f}x faster than Ring")
    print(f"✓ MultiTree reduces energy by {energy_reduction:.1f}%")
    print(f"✓ Communication complexity: O(log n) vs O(n)")
    
    print("\n✓ Comparison test completed!")


if __name__ == "__main__":
    main()

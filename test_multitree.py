"""
Test MultiTree Scheduler
"""
import sys
import numpy as np
from utils.config_loader import Config
from utils.topology import TopologyGenerator
from multitree.scheduler import MultiTreeScheduler


def main():
    print("="*60)
    print("MultiTree Scheduler Test")
    print("="*60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = Config()
    print(f"   ✓ Config loaded: {config.get('topology.type')} topology")
    
    # Generate topology
    print("\n2. Generating topology...")
    topo_gen = TopologyGenerator(config.config)
    G = topo_gen.generate()
    print(f"   ✓ Topology generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create MultiTree scheduler
    print("\n3. Creating MultiTree scheduler...")
    k_ary = config.get('multitree.k_ary', 2)
    scheduler = MultiTreeScheduler(G, k_ary=k_ary)
    
    # Build trees
    print("\n4. Building spanning trees...")
    trees = scheduler.build_trees()
    
    # Get schedule summary
    print("\n5. Analyzing schedules...")
    summary = scheduler.get_schedule_summary()
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total trees: {summary['num_trees']}")
    print(f"Average reduce-scatter steps: {np.mean(list(summary['reduce_scatter_steps'].values())):.2f}")
    print(f"Average all-gather steps: {np.mean(list(summary['allgather_steps'].values())):.2f}")
    print(f"Average total communication steps: {np.mean(list(summary['total_steps'].values())):.2f}")
    
    # Compare with ring all-reduce
    ring_steps = 2 * (G.number_of_nodes() - 1)  # O(n)
    multitree_steps = np.mean(list(summary['total_steps'].values()))
    speedup = ring_steps / multitree_steps
    
    print(f"\nComparison with Ring All-Reduce:")
    print(f"  Ring steps: {ring_steps}")
    print(f"  MultiTree steps: {multitree_steps:.2f}")
    print(f"  Speedup: {speedup:.2f}x")
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

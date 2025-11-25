"""
Verify MultiTree construction validity
"""
import networkx as nx
from utils.config_loader import Config
from utils.topology import TopologyGenerator
from multitree.scheduler import MultiTreeScheduler


def verify_spanning_tree(G, tree_edges, num_nodes):
    """Verify that tree_edges forms a valid spanning tree"""
    # Create tree graph
    T = nx.Graph()
    T.add_nodes_from(range(num_nodes))
    T.add_edges_from(tree_edges)
    
    # Check 1: Correct number of edges (n-1)
    if len(tree_edges) != num_nodes - 1:
        return False, f"Wrong edge count: {len(tree_edges)} (expected {num_nodes-1})"
    
    # Check 2: Connected
    if not nx.is_connected(T):
        return False, "Tree is not connected"
    
    # Check 3: No cycles (is a tree)
    if not nx.is_tree(T):
        return False, "Graph has cycles"
    
    # Check 4: All edges exist in original topology
    for edge in tree_edges:
        if not G.has_edge(*edge):
            return False, f"Edge {edge} not in original topology"
    
    return True, "Valid spanning tree"


def main():
    print("="*60)
    print("MultiTree Validity Test")
    print("="*60)
    
    # Setup
    config = Config()
    topo_gen = TopologyGenerator(config.config)
    G = topo_gen.generate()
    
    scheduler = MultiTreeScheduler(G, k_ary=2)
    trees = scheduler.build_trees()
    
    print(f"\nVerifying {len(trees)} spanning trees...")
    print("-"*60)
    
    # Verify each tree
    valid_count = 0
    for tree_id, tree in trees.items():
        is_valid, message = verify_spanning_tree(G, tree['edges'], G.number_of_nodes())
        
        if is_valid:
            valid_count += 1
        else:
            print(f"Tree {tree_id}: ❌ {message}")
    
    print(f"\n✓ Valid trees: {valid_count}/{len(trees)}")
    
    if valid_count == len(trees):
        print("\n🎉 All trees are valid spanning trees!")
    else:
        print(f"\n⚠️  {len(trees) - valid_count} invalid trees found")
    
    print("="*60)


if __name__ == "__main__":
    main()

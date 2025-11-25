"""
MultiTree All-Reduce Scheduler (Optimized)
Implementation of Algorithm 1 from ISCA 2021 paper
"""
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import deque


class MultiTreeScheduler:
    """MultiTree all-reduce scheduler with topology awareness"""
    
    def __init__(self, topology_graph: nx.Graph, k_ary: int = 2):
        """Initialize MultiTree scheduler"""
        self.G = topology_graph.copy()
        self.k_ary = k_ary
        self.num_nodes = self.G.number_of_nodes()
        self.nodes = list(self.G.nodes())
        
        # Trees: one tree per node
        self.trees = {}
        self.reduce_scatter_schedule = {}
        self.allgather_schedule = {}
        
        print(f"Initialized MultiTree scheduler for {self.num_nodes} nodes")
        print(f"Using {k_ary}-ary trees")
    
    def build_trees(self):
        """Build spanning trees using BFS approach"""
        print("\n" + "="*60)
        print("Building MultiTree spanning trees...")
        print("="*60)
        
        # Initialize trees
        for node_id in self.nodes:
            self.trees[node_id] = {
                'root': node_id,
                'nodes': {node_id},
                'edges': [],
                'parent': {node_id: None},
                'children': {node_id: []},
                'level': {node_id: 0}
            }
        
        # Build each tree independently using BFS
        for tree_id in self.nodes:
            self._build_single_tree_bfs(tree_id)
            
            if (tree_id + 1) % 10 == 0:
                print(f"  Built {tree_id + 1}/{self.num_nodes} trees...")
        
        print(f"✓ All {self.num_nodes} trees constructed")
        self._print_tree_stats()
        
        # Generate schedules
        self._generate_schedules()
        
        return self.trees
    
    def _build_single_tree_bfs(self, root: int):
        """Build a single spanning tree using BFS from root"""
        tree = self.trees[root]
        visited = {root}
        queue = deque([root])
        level = 0
        
        while queue and len(visited) < self.num_nodes:
            level_size = len(queue)
            level += 1
            
            for _ in range(level_size):
                if len(visited) >= self.num_nodes:
                    break
                    
                parent = queue.popleft()
                
                # Get neighbors of parent that haven't been visited
                neighbors = [n for n in self.G.neighbors(parent) if n not in visited]
                
                # Add up to k_ary children
                for child in neighbors[:self.k_ary]:
                    if len(visited) >= self.num_nodes:
                        break
                    
                    # Add child to tree
                    tree['nodes'].add(child)
                    tree['edges'].append((parent, child))
                    tree['parent'][child] = parent
                    tree['level'][child] = level
                    
                    if parent not in tree['children']:
                        tree['children'][parent] = []
                    tree['children'][parent].append(child)
                    
                    visited.add(child)
                    queue.append(child)
        
        # If tree is not complete, connect remaining nodes
        if len(visited) < self.num_nodes:
            remaining = set(self.nodes) - visited
            for node in remaining:
                # Find closest node in tree
                min_dist = float('inf')
                closest_parent = None
                
                for tree_node in visited:
                    try:
                        dist = nx.shortest_path_length(self.G, tree_node, node)
                        if dist < min_dist:
                            min_dist = dist
                            closest_parent = tree_node
                    except nx.NetworkXNoPath:
                        continue
                
                if closest_parent is not None:
                    tree['nodes'].add(node)
                    tree['edges'].append((closest_parent, node))
                    tree['parent'][node] = closest_parent
                    tree['level'][node] = tree['level'][closest_parent] + 1
                    
                    if closest_parent not in tree['children']:
                        tree['children'][closest_parent] = []
                    tree['children'][closest_parent].append(node)
                    
                    visited.add(node)
    
    def _generate_schedules(self):
        """Generate reduce-scatter and all-gather schedules"""
        print("\nGenerating communication schedules...")
        
        for tree_id, tree in self.trees.items():
            # Get tree height
            max_level = max(tree['level'].values())
            
            # Reduce-scatter: leaf to root
            self.reduce_scatter_schedule[tree_id] = []
            for level in range(max_level, 0, -1):
                time_step = max_level - level + 1
                nodes_at_level = [n for n, l in tree['level'].items() if l == level]
                
                for child in nodes_at_level:
                    parent = tree['parent'][child]
                    if parent is not None:
                        self.reduce_scatter_schedule[tree_id].append(
                            (child, parent, time_step)
                        )
            
            # All-gather: root to leaf
            self.allgather_schedule[tree_id] = []
            for level in range(1, max_level + 1):
                time_step = level + max_level
                nodes_at_level = [n for n, l in tree['level'].items() if l == level]
                
                for child in nodes_at_level:
                    parent = tree['parent'][child]
                    if parent is not None:
                        self.allgather_schedule[tree_id].append(
                            (parent, child, time_step)
                        )
        
        print("✓ Reduce-scatter schedules generated")
        print("✓ All-gather schedules generated")
    
    def _print_tree_stats(self):
        """Print statistics about constructed trees"""
        print("\nTree Statistics:")
        print("-" * 60)
        
        heights = [max(tree['level'].values()) for tree in self.trees.values()]
        num_edges = [len(tree['edges']) for tree in self.trees.values()]
        
        print(f"Average tree height: {np.mean(heights):.2f}")
        print(f"Min tree height: {np.min(heights)}")
        print(f"Max tree height: {np.max(heights)}")
        print(f"Average edges per tree: {np.mean(num_edges):.2f}")
        
        # Sample tree info
        sample_tree_id = 0
        sample_tree = self.trees[sample_tree_id]
        print(f"\nSample Tree {sample_tree_id}:")
        print(f"  Root: {sample_tree['root']}")
        print(f"  Height: {max(sample_tree['level'].values())}")
        print(f"  Number of edges: {len(sample_tree['edges'])}")
        print(f"  Number of nodes: {len(sample_tree['nodes'])}")
        
    def get_schedule_summary(self) -> Dict:
        """Get summary of communication schedules"""
        summary = {
            'num_trees': len(self.trees),
            'reduce_scatter_steps': {},
            'allgather_steps': {},
            'total_steps': {}
        }
        
        for tree_id in self.trees.keys():
            rs_steps = max([t for _, _, t in self.reduce_scatter_schedule[tree_id]], default=0)
            ag_schedule = self.allgather_schedule[tree_id]
            ag_steps = max([t for _, _, t in ag_schedule], default=rs_steps) - rs_steps if ag_schedule else 0
            
            summary['reduce_scatter_steps'][tree_id] = rs_steps
            summary['allgather_steps'][tree_id] = ag_steps
            summary['total_steps'][tree_id] = rs_steps + ag_steps
        
        return summary


# Test function
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    from utils.config_loader import Config
    from utils.topology import TopologyGenerator
    
    # Load config and generate topology
    config = Config()
    topo_gen = TopologyGenerator(config.config)
    G = topo_gen.generate()
    
    # Create MultiTree scheduler
    k_ary = config.get('multitree.k_ary', 2)
    scheduler = MultiTreeScheduler(G, k_ary=k_ary)
    
    # Build trees
    trees = scheduler.build_trees()
    
    # Get schedule summary
    summary = scheduler.get_schedule_summary()
    
    print("\n" + "="*60)
    print("Schedule Summary:")
    print("="*60)
    print(f"Total trees: {summary['num_trees']}")
    print(f"Avg reduce-scatter steps: {np.mean(list(summary['reduce_scatter_steps'].values())):.2f}")
    print(f"Avg all-gather steps: {np.mean(list(summary['allgather_steps'].values())):.2f}")
    print(f"Avg total steps: {np.mean(list(summary['total_steps'].values())):.2f}")
    
    print("\n✓ MultiTree scheduler test completed successfully!")

"""
Network topology generator for HFL MultiTree project
"""
import networkx as nx
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path
import matplotlib.pyplot as plt


class TopologyGenerator:
    """Generate different network topologies"""
    
    def __init__(self, config: dict):
        """
        Initialize topology generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.topology_type = config.get('topology', {}).get('type', '2D_Torus')
        self.dimensions = config.get('topology', {}).get('dimensions', [8, 8])
        self.num_nodes = config.get('topology', {}).get('num_nodes', 64)
        
    def generate(self) -> nx.Graph:
        """
        Generate network topology graph
        
        Returns:
            NetworkX graph representing the topology
        """
        if self.topology_type == "2D_Torus":
            return self._generate_2d_torus()
        elif self.topology_type == "Mesh":
            return self._generate_mesh()
        elif self.topology_type == "Fat_Tree":
            return self._generate_fat_tree()
        elif self.topology_type == "BiGraph":
            return self._generate_bigraph()
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
    
    def _generate_2d_torus(self) -> nx.Graph:
        """Generate 2D Torus topology"""
        rows, cols = self.dimensions
        G = nx.grid_2d_graph(rows, cols, periodic=True)
        
        # Relabel nodes to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        # Add position attributes for visualization
        pos = {}
        for i, (r, c) in enumerate([(i // cols, i % cols) for i in range(rows * cols)]):
            pos[i] = (c, rows - r - 1)
        nx.set_node_attributes(G, pos, 'pos')
        
        return G
    
    def _generate_mesh(self) -> nx.Graph:
        """Generate 2D Mesh topology (no wraparound)"""
        rows, cols = self.dimensions
        G = nx.grid_2d_graph(rows, cols, periodic=False)
        
        # Relabel nodes to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        # Add position attributes
        pos = {}
        for i, (r, c) in enumerate([(i // cols, i % cols) for i in range(rows * cols)]):
            pos[i] = (c, rows - r - 1)
        nx.set_node_attributes(G, pos, 'pos')
        
        return G
    
    def _generate_fat_tree(self) -> nx.Graph:
        """Generate Fat-Tree topology (k-ary tree)"""
        # For simplicity, we'll create a 2-level fat tree
        k = 4  # Number of ports per switch
        
        G = nx.Graph()
        
        # Add edge switches (bottom level)
        num_edge_switches = self.num_nodes // k
        edge_switches = list(range(num_edge_switches))
        
        # Add core switches (top level)
        num_core_switches = max(2, num_edge_switches // 2)
        core_switches = list(range(num_edge_switches, num_edge_switches + num_core_switches))
        
        # Connect edge switches to nodes
        node_id = num_edge_switches + num_core_switches
        for edge_sw in edge_switches:
            for _ in range(k):
                if node_id < self.num_nodes + num_edge_switches + num_core_switches:
                    G.add_edge(edge_sw, node_id)
                    node_id += 1
        
        # Connect edge switches to core switches
        for i, edge_sw in enumerate(edge_switches):
            core_sw = core_switches[i % num_core_switches]
            G.add_edge(edge_sw, core_sw)
        
        return G
    
    def _generate_bigraph(self) -> nx.Graph:
        """Generate BiGraph topology"""
        # BiGraph: two-stage fully connected graph
        n = int(np.sqrt(self.num_nodes))
        
        G = nx.Graph()
        
        # Bottom layer nodes
        bottom_nodes = list(range(n))
        # Top layer switches
        top_switches = list(range(n, 2 * n))
        
        # Fully connect bottom to top
        for b in bottom_nodes:
            for t in top_switches:
                G.add_edge(b, t)
        
        return G
    
    def get_neighbors(self, G: nx.Graph, node: int) -> List[int]:
        """Get neighbors of a node"""
        return list(G.neighbors(node))
    
    def get_distance(self, G: nx.Graph, source: int, target: int) -> int:
        """Get shortest path distance between two nodes"""
        try:
            return nx.shortest_path_length(G, source, target)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def visualize(self, G: nx.Graph, save_path: str = None):
        """
        Visualize the topology
        
        Args:
            G: NetworkX graph
            save_path: Path to save figure
        """
        plt.figure(figsize=(12, 10))
        
        if 'pos' in nx.get_node_attributes(G, 'pos'):
            pos = nx.get_node_attributes(G, 'pos')
        else:
            pos = nx.spring_layout(G, seed=42)
        
        nx.draw(G, pos, 
                node_color='lightblue',
                node_size=300,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                width=1.5)
        
        plt.title(f"{self.topology_type} Network Topology ({self.num_nodes} nodes)")
        
        if save_path:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Topology visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


# Test function
if __name__ == "__main__":
    from config_loader import Config
    
    # Load configuration
    config = Config()
    
    # Generate topology
    topo_gen = TopologyGenerator(config.config)
    G = topo_gen.generate()
    
    print("=" * 50)
    print(f"Generated {topo_gen.topology_type} Topology")
    print("=" * 50)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # Test neighbor finding
    print(f"\nNeighbors of node 0: {topo_gen.get_neighbors(G, 0)}")
    
    # Test distance calculation
    print(f"Distance from node 0 to node 10: {topo_gen.get_distance(G, 0, 10)}")
    
    # Visualize (fixed path)
    project_root = Path(__file__).parent.parent
    save_path = project_root / "results" / "plots" / "topology_test.png"
    topo_gen.visualize(G, save_path=str(save_path))
    
    print("\nTopology test completed successfully!")

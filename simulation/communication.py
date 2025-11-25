"""
Communication Simulator using SimPy
Discrete-event simulation for MultiTree all-reduce
"""
import simpy
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx


class Link:
    """Bidirectional network link with bandwidth"""
    
    def __init__(self, env, bandwidth: float, latency: float):
        """
        Args:
            env: SimPy environment
            bandwidth: Link bandwidth in GB/s
            latency: Link latency in seconds
        """
        self.env = env
        self.bandwidth = bandwidth  # GB/s
        self.latency = latency  # seconds
        self.resource = simpy.Resource(env, capacity=1)  # Single link
        
    def transmit(self, data_size: float):
        """
        Transmit data across link
        Args:
            data_size: Data size in GB
        Returns:
            Transmission time in seconds
        """
        transmission_time = data_size / self.bandwidth + self.latency
        return transmission_time


class Device:
    """Computing device (GPU) with compute and energy model"""
    
    def __init__(self, device_id: int, config: Dict):
        """
        Args:
            device_id: Device identifier
            config: Configuration dict with compute_power, energy specs
        """
        self.device_id = device_id
        self.compute_power = config.get('compute_power', 312e12)  # FLOPS
        self.idle_power = config.get('idle_power', 100.0)  # Watts
        self.compute_power_watts = config.get('compute_power_watts', 400.0)  # Watts
        self.comm_power = config.get('comm_power', 150.0)  # Watts during communication
        
        # Tracking
        self.total_energy = 0.0  # Joules
        self.compute_time = 0.0  # seconds
        self.comm_time = 0.0  # seconds
        self.idle_time = 0.0  # seconds
        
    def compute(self, flops: float) -> float:
        """
        Compute time for given FLOPs
        Returns: computation time in seconds
        """
        comp_time = flops / self.compute_power
        self.compute_time += comp_time
        self.total_energy += comp_time * self.compute_power_watts
        return comp_time
    
    def communicate(self, duration: float):
        """Record communication time and energy"""
        self.comm_time += duration
        self.total_energy += duration * self.comm_power
    
    def idle(self, duration: float):
        """Record idle time and energy"""
        self.idle_time += duration
        self.total_energy += duration * self.idle_power


class CommunicationSimulator:
    """Discrete-event simulator for distributed communication"""
    
    def __init__(self, topology: nx.Graph, config: Dict):
        """
        Args:
            topology: NetworkX graph representing network topology
            config: System configuration
        """
        self.topology = topology
        self.config = config
        self.env = simpy.Environment()
        
        # Network parameters
        self.bandwidth = config.get('hardware.link_bandwidth', 16.0)  # GB/s
        self.latency = config.get('hardware.link_latency', 150e-9)  # seconds
        
        # Create links for all edges in topology
        self.links = {}
        for edge in topology.edges():
            link_id = tuple(sorted(edge))
            self.links[link_id] = Link(self.env, self.bandwidth, self.latency)
        
        # Create devices
        self.devices = {}
        device_config = {
            'compute_power': config.get('hardware.gpu_flops', 312e12),
            'idle_power': 100.0,
            'compute_power_watts': 400.0,
            'comm_power': 150.0
        }
        for node in topology.nodes():
            self.devices[node] = Device(node, device_config)
        
        # Results
        self.total_time = 0.0
        self.max_energy = 0.0
        self.avg_energy = 0.0
        
    def simulate_multitree_allreduce(self, scheduler, model_size: float):
        """
        Simulate MultiTree all-reduce
        Args:
            scheduler: MultiTreeScheduler with built trees
            model_size: Total model size in GB
        Returns:
            Dict with timing and energy metrics
        """
        # Data chunk size per tree
        chunk_size = model_size / len(scheduler.trees)
        
        # Run reduce-scatter phase for all trees in parallel
        rs_procs = []
        for tree_id in scheduler.trees.keys():
            proc = self.env.process(
                self._simulate_reduce_scatter(tree_id, scheduler, chunk_size)
            )
            rs_procs.append(proc)
        
        # Wait for all reduce-scatter to complete
        yield simpy.AllOf(self.env, rs_procs)
        
        rs_time = self.env.now
        print(f"  Reduce-scatter completed at t={rs_time:.6f}s")
        
        # Run all-gather phase for all trees in parallel
        ag_procs = []
        for tree_id in scheduler.trees.keys():
            proc = self.env.process(
                self._simulate_allgather(tree_id, scheduler, chunk_size)
            )
            ag_procs.append(proc)
        
        # Wait for all all-gather to complete
        yield simpy.AllOf(self.env, ag_procs)
        
        self.total_time = self.env.now
        print(f"  All-gather completed at t={self.total_time:.6f}s")
        
        # Collect energy statistics
        energies = [dev.total_energy for dev in self.devices.values()]
        self.max_energy = max(energies)
        self.avg_energy = np.mean(energies)
        
    def _simulate_reduce_scatter(self, tree_id: int, scheduler, chunk_size: float):
        """Simulate reduce-scatter for one tree"""
        schedule = scheduler.reduce_scatter_schedule[tree_id]
        
        # Group by time step
        steps = {}
        for child, parent, time_step in schedule:
            if time_step not in steps:
                steps[time_step] = []
            steps[time_step].append((child, parent))
        
        # Execute each time step
        for time_step in sorted(steps.keys()):
            transfers = steps[time_step]
            transfer_procs = []
            
            for child, parent in transfers:
                proc = self.env.process(
                    self._transfer_data(child, parent, chunk_size)
                )
                transfer_procs.append(proc)
            
            # Wait for all transfers in this step
            yield simpy.AllOf(self.env, transfer_procs)
    
    def _simulate_allgather(self, tree_id: int, scheduler, chunk_size: float):
        """Simulate all-gather for one tree"""
        schedule = scheduler.allgather_schedule[tree_id]
        
        # Group by time step
        steps = {}
        for parent, child, time_step in schedule:
            if time_step not in steps:
                steps[time_step] = []
            steps[time_step].append((parent, child))
        
        # Execute each time step
        for time_step in sorted(steps.keys()):
            transfers = steps[time_step]
            transfer_procs = []
            
            for parent, child in transfers:
                proc = self.env.process(
                    self._transfer_data(parent, child, chunk_size)
                )
                transfer_procs.append(proc)
            
            # Wait for all transfers in this step
            yield simpy.AllOf(self.env, transfer_procs)
    
    def _transfer_data(self, src: int, dst: int, data_size: float):
        """Transfer data between two nodes"""
        # Get link
        link_id = tuple(sorted([src, dst]))
        link = self.links[link_id]
        
        # Request link resource
        with link.resource.request() as req:
            yield req
            
            # Simulate transmission
            trans_time = link.transmit(data_size)
            yield self.env.timeout(trans_time)
            
            # Record communication for both devices
            self.devices[src].communicate(trans_time)
            self.devices[dst].communicate(trans_time)
    
    def run_simulation(self, scheduler, model_size: float):
        """
        Run complete simulation
        Args:
            scheduler: MultiTreeScheduler
            model_size: Model size in GB
        Returns:
            Results dictionary
        """
        # Reset environment
        self.env = simpy.Environment()
        
        # Recreate links and devices for new environment
        self.links = {}
        for edge in self.topology.edges():
            link_id = tuple(sorted(edge))
            self.links[link_id] = Link(self.env, self.bandwidth, self.latency)
        
        device_config = {
            'compute_power': self.config.get('hardware.gpu_flops', 312e12),
            'idle_power': 100.0,
            'compute_power_watts': 400.0,
            'comm_power': 150.0
        }
        self.devices = {}
        for node in self.topology.nodes():
            self.devices[node] = Device(node, device_config)
        
        print("\n" + "="*60)
        print("Running MultiTree All-Reduce Simulation")
        print("="*60)
        print(f"Model size: {model_size*1000:.2f} MB")
        print(f"Number of trees: {len(scheduler.trees)}")
        print(f"Chunk size per tree: {(model_size/len(scheduler.trees))*1000:.2f} MB")
        
        # Run simulation
        self.env.process(
            self.simulate_multitree_allreduce(scheduler, model_size)
        )
        self.env.run()
        
        # Compile results
        results = {
            'total_time': self.total_time,
            'max_energy': self.max_energy,
            'avg_energy': self.avg_energy,
            'max_comm_time': max(d.comm_time for d in self.devices.values()),
            'avg_comm_time': np.mean([d.comm_time for d in self.devices.values()]),
        }
        
        print("\n" + "="*60)
        print("Simulation Results")
        print("="*60)
        print(f"Total time: {results['total_time']*1000:.2f} ms")
        print(f"Max communication time: {results['max_comm_time']*1000:.2f} ms")
        print(f"Avg communication time: {results['avg_comm_time']*1000:.2f} ms")
        print(f"Max device energy: {results['max_energy']:.2f} J")
        print(f"Avg device energy: {results['avg_energy']:.2f} J")
        
        return results

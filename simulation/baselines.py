"""
Baseline All-Reduce Algorithms
Ring, 2D-Ring, and Direct-Binary-Tree implementations
"""
import simpy
import numpy as np
import networkx as nx
from typing import Dict, List
from simulation.communication import CommunicationSimulator, Device, Link


class RingAllReduce:
    """Ring all-reduce baseline"""
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.schedule = self._build_ring_schedule()
    
    def _build_ring_schedule(self):
        """Build ring communication schedule"""
        # Reduce-scatter: N-1 steps
        reduce_scatter = []
        for step in range(self.num_nodes - 1):
            for i in range(self.num_nodes):
                sender = i
                receiver = (i + 1) % self.num_nodes
                reduce_scatter.append((sender, receiver, step + 1))
        
        # All-gather: N-1 steps
        allgather = []
        for step in range(self.num_nodes - 1):
            for i in range(self.num_nodes):
                sender = i
                receiver = (i + 1) % self.num_nodes
                allgather.append((sender, receiver, self.num_nodes + step))
        
        return {
            'reduce_scatter': reduce_scatter,
            'allgather': allgather
        }
    
    def get_num_steps(self):
        """Total communication steps"""
        return 2 * (self.num_nodes - 1)


class RingSimulator(CommunicationSimulator):
    """Simulator for Ring all-reduce"""
    
    def simulate_ring_allreduce(self, ring: RingAllReduce, model_size: float):
        """Simulate Ring all-reduce"""
        chunk_size = model_size / self.topology.number_of_nodes()
        
        # Reduce-scatter phase
        rs_schedule = ring.schedule['reduce_scatter']
        yield self.env.process(self._execute_ring_phase(rs_schedule, chunk_size))
        rs_time = self.env.now
        print(f"  Reduce-scatter completed at t={rs_time:.6f}s")
        
        # All-gather phase
        ag_schedule = ring.schedule['allgather']
        yield self.env.process(self._execute_ring_phase(ag_schedule, chunk_size))
        
        self.total_time = self.env.now
        print(f"  All-gather completed at t={self.total_time:.6f}s")
        
        # Collect energy statistics
        energies = [dev.total_energy for dev in self.devices.values()]
        self.max_energy = max(energies)
        self.avg_energy = np.mean(energies)
    
    def _execute_ring_phase(self, schedule, chunk_size):
        """Execute one phase of ring all-reduce"""
        # Group by time step
        steps = {}
        for sender, receiver, time_step in schedule:
            if time_step not in steps:
                steps[time_step] = []
            steps[time_step].append((sender, receiver))
        
        # Execute each time step sequentially
        for time_step in sorted(steps.keys()):
            transfers = steps[time_step]
            transfer_procs = []
            
            for sender, receiver in transfers:
                # Find path between nodes
                try:
                    path = nx.shortest_path(self.topology, sender, receiver)
                    if len(path) == 2:  # Direct connection
                        proc = self.env.process(
                            self._transfer_data(sender, receiver, chunk_size)
                        )
                        transfer_procs.append(proc)
                except nx.NetworkXNoPath:
                    print(f"Warning: No path from {sender} to {receiver}")
            
            # Wait for all transfers in this step
            if transfer_procs:
                yield simpy.AllOf(self.env, transfer_procs)
    
    def run_ring_simulation(self, model_size: float):
        """Run ring all-reduce simulation"""
        # Reset environment
        self.env = simpy.Environment()
        
        # Recreate links and devices
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
        print("Running Ring All-Reduce Simulation")
        print("="*60)
        print(f"Model size: {model_size*1000:.2f} MB")
        print(f"Number of nodes: {self.topology.number_of_nodes()}")
        
        # Create ring and run simulation
        ring = RingAllReduce(self.topology.number_of_nodes())
        print(f"Communication steps: {ring.get_num_steps()}")
        
        self.env.process(self.simulate_ring_allreduce(ring, model_size))
        self.env.run()
        
        # Compile results
        results = {
            'total_time': self.total_time,
            'max_energy': self.max_energy,
            'avg_energy': self.avg_energy,
            'num_steps': ring.get_num_steps()
        }
        
        print("\n" + "="*60)
        print("Ring All-Reduce Results")
        print("="*60)
        print(f"Total time: {results['total_time']*1000:.2f} ms")
        print(f"Max device energy: {results['max_energy']:.2f} J")
        print(f"Avg device energy: {results['avg_energy']:.2f} J")
        print(f"Communication steps: {results['num_steps']}")
        
        return results

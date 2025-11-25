"""
AHFLP: Adaptive Hierarchical Federated Learning Process
Optimizes aggregation timing and resource allocation
Based on IEEE Trans. Cloud Computing 2025 paper
"""
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


class AHFLPOptimizer:
    """
    Adaptive Hierarchical FL optimizer
    Solves the joint optimization problem:
    - Minimize: weighted sum of time and energy
    - Subject to: resource constraints, convergence requirements
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: System configuration with AHFLP parameters
        """
        self.config = config
        
        # System parameters
        self.num_edges = config.get('hierarchy.num_edges', 5)
        self.devices_per_edge = config.get('hierarchy.devices_per_edge', 10)
        self.total_devices = self.num_edges * self.devices_per_edge
        
        # AHFLP parameters
        self.l1_max = config.get('ahflp.l1_max', 10)  # Max local iterations
        self.l2_max = config.get('ahflp.l2_max', 10)  # Max edge iterations
        self.phi = config.get('ahflp.phi', 10.0)      # Time-energy trade-off weight
        self.learning_rate = config.get('ahflp.learning_rate', 0.001)
        
        # Resource constraints
        self.T_max = config.get('resources.time_budget', 600.0)  # seconds
        self.E_min = config.get('resources.energy_budget_min', 200.0)  # Joules per device
        self.E_max = config.get('resources.energy_budget_max', 400.0)  # Joules per device
        
        # Hardware parameters
        self.f_min = config.get('resources.cpu_freq_min', 1.0e9)  # Min CPU freq (Hz)
        self.f_max = config.get('resources.cpu_freq_max', 3.0e9)  # Max CPU freq (Hz)
        self.B_link = config.get('hardware.link_bandwidth', 16.0) * 1e9  # bits/s
        
        # Model parameters
        self.model_size = config.get('training.model_size', 100e6)  # bits
        self.dataset_size = config.get('training.dataset_size', 50000)
        self.batch_size = config.get('training.batch_size', 32)
        self.flops_per_sample = config.get('training.flops_per_sample', 1e9)
        
        # Power model parameters (from A100 specs)
        self.P_idle = 100.0  # Watts
        self.P_max = 400.0   # Watts at max frequency
        self.P_comm = 150.0  # Watts during communication
        
        print(f"Initialized AHFLP optimizer")
        print(f"  Hierarchy: {self.num_edges} edges, {self.devices_per_edge} devices/edge")
        print(f"  Resource budget: T_max={self.T_max}s, E=[{self.E_min},{self.E_max}]J")
        print(f"  Trade-off weight: φ={self.phi}")
    
    def optimize_aggregation_schedule(self, multitree_time: float, multitree_energy: float):
        """
        Optimize local and edge aggregation intervals
        
        Returns:
            dict with optimized l1, l2, cpu_freqs, bandwidth allocations
        """
        print("\n" + "="*60)
        print("AHFLP Optimization")
        print("="*60)
        print(f"Input MultiTree metrics:")
        print(f"  Communication time: {multitree_time*1000:.2f} ms")
        print(f"  Average energy: {multitree_energy:.2f} J")
        
        # Decision variables: [l1, l2, f_local, f_edge, b_local, b_edge]
        # l1: local aggregation interval (1 to l1_max)
        # l2: edge aggregation interval (1 to l2_max)  
        # f_local: CPU frequency for local training
        # f_edge: CPU frequency for edge aggregation
        # b_local: bandwidth allocation for local-to-edge
        # b_edge: bandwidth allocation for edge-to-cloud
        
        # Initial guess: moderate values
        x0 = np.array([
            5.0,  # l1
            5.0,  # l2
            2.0e9,  # f_local
            2.5e9,  # f_edge
            self.B_link * 0.5,  # b_local
            self.B_link * 0.5   # b_edge
        ])
        
        # Bounds
        bounds = [
            (1, self.l1_max),           # l1
            (1, self.l2_max),           # l2
            (self.f_min, self.f_max),   # f_local
            (self.f_min, self.f_max),   # f_edge
            (self.B_link*0.1, self.B_link),  # b_local
            (self.B_link*0.1, self.B_link)   # b_edge
        ]
        
        # Optimize
        result = minimize(
            fun=self._objective_function,
            x0=x0,
            args=(multitree_time, multitree_energy),
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': self._time_constraint, 'args': (multitree_time,)},
                {'type': 'ineq', 'fun': self._energy_constraint, 'args': (multitree_energy,)}
            ],
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if not result.success:
            print(f"⚠️  Optimization warning: {result.message}")
            print("   Using heuristic values instead")
            return self._get_heuristic_solution(multitree_time, multitree_energy)
        
        # Extract optimized values
        l1_opt = int(round(result.x[0]))
        l2_opt = int(round(result.x[1]))
        f_local_opt = result.x[2]
        f_edge_opt = result.x[3]
        b_local_opt = result.x[4]
        b_edge_opt = result.x[5]
        
        # Calculate final metrics
        T_total = self._calculate_total_time(result.x, multitree_time)
        E_total = self._calculate_total_energy(result.x, multitree_energy)
        
        solution = {
            'l1': l1_opt,
            'l2': l2_opt,
            'f_local': f_local_opt,
            'f_edge': f_edge_opt,
            'b_local': b_local_opt,
            'b_edge': b_edge_opt,
            'total_time': T_total,
            'total_energy': E_total,
            'objective_value': result.fun
        }
        
        print("\n" + "-"*60)
        print("Optimized Solution:")
        print("-"*60)
        print(f"  Local aggregation interval (l1): {l1_opt}")
        print(f"  Edge aggregation interval (l2): {l2_opt}")
        print(f"  Local CPU frequency: {f_local_opt/1e9:.2f} GHz")
        print(f"  Edge CPU frequency: {f_edge_opt/1e9:.2f} GHz")
        print(f"  Local bandwidth: {b_local_opt/1e9:.2f} Gbps")
        print(f"  Edge bandwidth: {b_edge_opt/1e9:.2f} Gbps")
        print(f"\n  Total training time: {T_total:.2f} s")
        print(f"  Total energy per device: {E_total:.2f} J")
        print(f"  Objective value: {result.fun:.4f}")
        
        return solution
    
    def _objective_function(self, x, T_comm, E_comm):
        """
        Objective: minimize φ*T + E
        where T is total time and E is total energy
        """
        T_total = self._calculate_total_time(x, T_comm)
        E_total = self._calculate_total_energy(x, E_comm)
        
        return self.phi * T_total + E_total
    
    def _calculate_total_time(self, x, T_comm):
        """Calculate total training time"""
        l1, l2, f_local, f_edge, b_local, b_edge = x
        
        # Local training time per iteration
        samples_per_device = self.dataset_size / self.total_devices
        batches_per_device = samples_per_device / self.batch_size
        T_local_iter = (batches_per_device * self.flops_per_sample * self.batch_size) / f_local
        
        # Edge aggregation time
        T_edge_agg = (self.devices_per_edge * self.model_size) / b_local
        
        # Cloud aggregation time (using MultiTree)
        T_cloud_agg = T_comm
        
        # Total time for one round
        T_round = l1 * T_local_iter + T_edge_agg + (T_cloud_agg / l2)
        
        # Total rounds needed (assume 100 rounds for convergence)
        num_rounds = 100 / (l1 * l2)
        
        T_total = T_round * num_rounds
        
        return T_total
    
    def _calculate_total_energy(self, x, E_comm):
        """Calculate total energy per device"""
        l1, l2, f_local, f_edge, b_local, b_edge = x
        
        # Local computation energy (dynamic power proportional to f^3)
        T_local_comp = self._calculate_total_time(x, E_comm) * 0.7  # 70% time in local training
        P_local = self.P_idle + (self.P_max - self.P_idle) * (f_local / self.f_max) ** 2
        E_local = P_local * T_local_comp
        
        # Communication energy
        T_comm_total = self._calculate_total_time(x, E_comm) * 0.3  # 30% time in communication
        E_comm_total = self.P_comm * T_comm_total
        
        E_total = E_local + E_comm_total
        
        return E_total
    
    def _time_constraint(self, x, T_comm):
        """Time budget constraint: T_total <= T_max"""
        T_total = self._calculate_total_time(x, T_comm)
        return self.T_max - T_total
    
    def _energy_constraint(self, x, E_comm):
        """Energy budget constraint: E_min <= E_total <= E_max"""
        E_total = self._calculate_total_energy(x, E_comm)
        return min(E_total - self.E_min, self.E_max - E_total)
    
    def _get_heuristic_solution(self, T_comm, E_comm):
        """Fallback heuristic solution if optimization fails"""
        return {
            'l1': 5,
            'l2': 5,
            'f_local': 2.0e9,
            'f_edge': 2.5e9,
            'b_local': self.B_link * 0.6,
            'b_edge': self.B_link * 0.7,
            'total_time': self.T_max * 0.8,
            'total_energy': (self.E_min + self.E_max) / 2,
            'objective_value': 0.0
        }

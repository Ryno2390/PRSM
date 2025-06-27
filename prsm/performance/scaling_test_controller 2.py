#!/usr/bin/env python3
"""
PRSM Scaling Test Controller
Comprehensive environmental controls for large-scale performance testing

Features:
- Node scaling from 10 to 1000+ nodes
- Multiple network environment simulation
- Resource usage monitoring
- Automated test orchestration
- Performance degradation analysis
- Byzantine fault injection at scale
"""

import asyncio
import time
import json
import csv
import statistics
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys
import threading
import queue
import psutil
import tracemalloc
import gc

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PRSM_ROOT))

# Import PRSM components
try:
    from prsm.performance.benchmark_collector import get_global_collector, reset_global_collector, time_async_operation
    from comprehensive_performance_benchmark import BenchmarkConfig, BenchmarkType, NetworkCondition, BenchmarkResults
except ImportError:
    print("⚠️ Could not import PRSM performance modules - using mock implementations")


class ScalingEnvironment(str, Enum):
    """Different scaling test environments"""
    LOCAL_SIMULATION = "local_simulation"      # Single machine simulation
    DOCKER_CLUSTER = "docker_cluster"          # Multi-container local cluster
    MULTI_REGION_CLOUD = "multi_region_cloud"  # Distributed cloud deployment
    HYBRID_EDGE = "hybrid_edge"               # Mix of cloud and edge nodes


class ResourceProfile(str, Enum):
    """Resource allocation profiles for nodes"""
    MINIMAL = "minimal"       # 1 CPU, 512MB RAM
    LIGHT = "light"          # 2 CPU, 1GB RAM  
    STANDARD = "standard"    # 4 CPU, 2GB RAM
    HEAVY = "heavy"          # 8 CPU, 4GB RAM
    EXTREME = "extreme"      # 16 CPU, 8GB RAM


class NetworkTopology(str, Enum):
    """Network topology patterns"""
    FULLY_CONNECTED = "fully_connected"    # All nodes connected to all
    RING = "ring"                         # Circular connection pattern
    STAR = "star"                         # Hub and spoke pattern
    MESH = "mesh"                         # Partial mesh connectivity
    HIERARCHICAL = "hierarchical"        # Tree-like structure
    RANDOM = "random"                     # Random connection pattern


@dataclass
class ScalingTestConfig:
    """Configuration for a scaling test scenario"""
    name: str
    environment: ScalingEnvironment
    node_counts: List[int]
    resource_profile: ResourceProfile
    network_topology: NetworkTopology
    network_conditions: List[NetworkCondition]
    test_duration_per_scale: int = 60
    warmup_duration: int = 30
    cooldown_duration: int = 10
    byzantine_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    target_operations_per_second: float = 10.0
    enable_resource_monitoring: bool = True
    enable_fault_injection: bool = False
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    save_detailed_logs: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingTestResult:
    """Results from a scaling test run"""
    config: ScalingTestConfig
    start_time: datetime
    end_time: datetime
    
    # Per-scale results
    node_performance: Dict[int, Dict[str, float]]  # node_count -> metrics
    resource_usage: Dict[int, Dict[str, float]]    # node_count -> resource metrics
    network_metrics: Dict[int, Dict[str, float]]   # node_count -> network metrics
    
    # Scaling analysis
    scaling_efficiency: Dict[int, float]           # node_count -> efficiency ratio
    throughput_saturation_point: Optional[int]    # Node count where throughput plateaus
    latency_degradation_point: Optional[int]      # Node count where latency degrades significantly
    
    # Resource analysis
    memory_usage_trend: List[Tuple[int, float]]    # [(node_count, memory_mb)]
    cpu_usage_trend: List[Tuple[int, float]]       # [(node_count, cpu_percent)]
    
    # Network analysis
    bandwidth_requirements: Dict[int, float]       # node_count -> estimated bandwidth
    message_overhead: Dict[int, int]               # node_count -> total messages
    
    # Fault tolerance analysis
    byzantine_resilience: Dict[float, Dict[int, float]]  # ratio -> {node_count: success_rate}
    
    # Overall assessment
    recommended_max_nodes: int
    performance_bottlenecks: List[str]
    scaling_recommendations: List[str]
    
    # Raw data
    detailed_results: List[BenchmarkResults] = field(default_factory=list)
    resource_logs: List[Dict[str, Any]] = field(default_factory=list)


class ResourceMonitor:
    """Monitor system resource usage during scaling tests"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024  # MB
                process_cpu = process.cpu_percent()
                
                # Network I/O
                net_io = psutil.net_io_counters()
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                
                metrics = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_mb': memory.used / 1024 / 1024,
                    'memory_available_mb': memory.available / 1024 / 1024,
                    'process_memory_mb': process_memory,
                    'process_cpu_percent': process_cpu,
                    'network_bytes_sent': net_io.bytes_sent if net_io else 0,
                    'network_bytes_recv': net_io.bytes_recv if net_io else 0,
                    'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                    'disk_write_bytes': disk_io.write_bytes if disk_io else 0
                }
                
                self.metrics_queue.put(metrics)
                
            except Exception as e:
                print(f"⚠️ Resource monitoring error: {e}")
            
            time.sleep(interval)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get collected metrics"""
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return metrics


class ScalingTestController:
    """Controller for comprehensive scaling tests"""
    
    def __init__(self, output_directory: str = "scaling_test_results"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        self.resource_monitor = ResourceMonitor()
        self.current_test = None
        self.results_history = []
        
        # Performance tracking
        tracemalloc.start()
    
    def create_comprehensive_scaling_configs(self) -> List[ScalingTestConfig]:
        """Create comprehensive scaling test configurations"""
        configs = []
        
        # 1. Basic scaling progression (geometric progression)
        basic_node_counts = [10, 20, 40, 80, 160, 320, 640, 1000]
        
        configs.append(ScalingTestConfig(
            name="comprehensive_scaling_progression",
            environment=ScalingEnvironment.LOCAL_SIMULATION,
            node_counts=basic_node_counts,
            resource_profile=ResourceProfile.STANDARD,
            network_topology=NetworkTopology.FULLY_CONNECTED,
            network_conditions=[NetworkCondition.WAN],
            test_duration_per_scale=45,
            warmup_duration=15,
            byzantine_ratios=[0.0, 0.1, 0.2],
            target_operations_per_second=10.0,
            enable_resource_monitoring=True,
            metadata={"test_type": "comprehensive_scaling"}
        ))
        
        # 2. Network condition impact across scales
        network_test_nodes = [25, 100, 400]
        for condition in [NetworkCondition.LAN, NetworkCondition.WAN, NetworkCondition.INTERCONTINENTAL]:
            configs.append(ScalingTestConfig(
                name=f"network_scaling_{condition.value}",
                environment=ScalingEnvironment.LOCAL_SIMULATION,
                node_counts=network_test_nodes,
                resource_profile=ResourceProfile.STANDARD,
                network_topology=NetworkTopology.MESH,
                network_conditions=[condition],
                test_duration_per_scale=30,
                target_operations_per_second=15.0,
                metadata={"test_type": "network_impact", "primary_condition": condition.value}
            ))
        
        # 3. Resource constraint tests
        resource_test_nodes = [50, 200, 500]
        for profile in [ResourceProfile.MINIMAL, ResourceProfile.LIGHT, ResourceProfile.HEAVY]:
            configs.append(ScalingTestConfig(
                name=f"resource_constrained_{profile.value}",
                environment=ScalingEnvironment.LOCAL_SIMULATION,
                node_counts=resource_test_nodes,
                resource_profile=profile,
                network_topology=NetworkTopology.RANDOM,
                network_conditions=[NetworkCondition.WAN],
                test_duration_per_scale=40,
                max_memory_mb=8192 if profile == ResourceProfile.MINIMAL else None,
                max_cpu_percent=80.0 if profile == ResourceProfile.MINIMAL else None,
                metadata={"test_type": "resource_constraint", "primary_profile": profile.value}
            ))
        
        # 4. Network topology impact
        topology_test_nodes = [30, 120, 480]
        for topology in [NetworkTopology.STAR, NetworkTopology.RING, NetworkTopology.HIERARCHICAL]:
            configs.append(ScalingTestConfig(
                name=f"topology_scaling_{topology.value}",
                environment=ScalingEnvironment.LOCAL_SIMULATION,
                node_counts=topology_test_nodes,
                resource_profile=ResourceProfile.STANDARD,
                network_topology=topology,
                network_conditions=[NetworkCondition.WAN],
                test_duration_per_scale=35,
                metadata={"test_type": "topology_impact", "primary_topology": topology.value}
            ))
        
        # 5. Byzantine fault tolerance at scale
        configs.append(ScalingTestConfig(
            name="byzantine_fault_scaling",
            environment=ScalingEnvironment.LOCAL_SIMULATION,
            node_counts=[50, 150, 300, 600],
            resource_profile=ResourceProfile.STANDARD,
            network_topology=NetworkTopology.MESH,
            network_conditions=[NetworkCondition.WAN],
            test_duration_per_scale=60,
            byzantine_ratios=[0.0, 0.1, 0.2, 0.33],
            enable_fault_injection=True,
            target_operations_per_second=8.0,
            metadata={"test_type": "byzantine_scaling"}
        ))
        
        # 6. Extreme scale test (stress test)
        configs.append(ScalingTestConfig(
            name="extreme_scale_stress_test",
            environment=ScalingEnvironment.LOCAL_SIMULATION,
            node_counts=[100, 500, 1000],
            resource_profile=ResourceProfile.HEAVY,
            network_topology=NetworkTopology.HIERARCHICAL,
            network_conditions=[NetworkCondition.WAN],
            test_duration_per_scale=90,
            warmup_duration=30,
            cooldown_duration=20,
            target_operations_per_second=20.0,
            enable_resource_monitoring=True,
            max_memory_mb=16384,  # 16GB limit
            save_detailed_logs=True,
            metadata={"test_type": "extreme_scale"}
        ))
        
        return configs
    
    async def simulate_scaled_consensus_operation(self, 
                                                config: ScalingTestConfig,
                                                node_count: int, 
                                                byzantine_ratio: float = 0.0) -> Tuple[bool, float, Dict[str, float]]:
        """Simulate a consensus operation at scale"""
        
        # Calculate base operation time based on scaling factors
        base_time_ms = 10.0  # Base consensus time
        
        # Scaling factors
        # Node scaling factor (logarithmic degradation)
        node_factor = 1.0 + (node_count / 100.0) * 0.3  # 30% increase per 100 nodes
        
        # Network topology factor
        topology_factors = {
            NetworkTopology.FULLY_CONNECTED: 1.0 + (node_count / 50.0) * 0.1,  # Worst scaling
            NetworkTopology.STAR: 1.0 + (node_count / 200.0) * 0.1,           # Better scaling
            NetworkTopology.HIERARCHICAL: 1.0 + (node_count / 150.0) * 0.1,   # Good scaling
            NetworkTopology.MESH: 1.0 + (node_count / 100.0) * 0.1,           # Moderate scaling
            NetworkTopology.RING: 1.0 + (node_count / 80.0) * 0.1,            # Poor scaling
            NetworkTopology.RANDOM: 1.0 + (node_count / 120.0) * 0.1          # Variable scaling
        }
        topology_factor = topology_factors.get(config.network_topology, 1.0)
        
        # Resource constraint factor
        resource_factors = {
            ResourceProfile.MINIMAL: 1.5,    # 50% slower
            ResourceProfile.LIGHT: 1.2,      # 20% slower
            ResourceProfile.STANDARD: 1.0,   # Baseline
            ResourceProfile.HEAVY: 0.8,      # 20% faster
            ResourceProfile.EXTREME: 0.6     # 40% faster
        }
        resource_factor = resource_factors.get(config.resource_profile, 1.0)
        
        # Network condition factor
        network_factors = {
            NetworkCondition.IDEAL: 0.1,
            NetworkCondition.LAN: 0.5,
            NetworkCondition.WAN: 1.0,
            NetworkCondition.INTERCONTINENTAL: 2.0,
            NetworkCondition.POOR: 3.0
        }
        
        # Calculate total latency
        network_condition = config.network_conditions[0] if config.network_conditions else NetworkCondition.WAN
        network_factor = network_factors.get(network_condition, 1.0)
        
        # Byzantine impact (increases latency and failure rate)
        byzantine_factor = 1.0 + byzantine_ratio * 2.0  # Up to 200% increase
        
        # Total operation time
        total_time_ms = base_time_ms * node_factor * topology_factor * resource_factor * network_factor * byzantine_factor
        
        # Add realistic variance
        variance = total_time_ms * 0.2  # 20% variance
        actual_time_ms = total_time_ms + random.uniform(-variance, variance)
        actual_time_ms = max(1.0, actual_time_ms)  # Minimum 1ms
        
        # Simulate the operation delay
        await asyncio.sleep(actual_time_ms / 1000.0)
        
        # Calculate success probability
        base_success_rate = 0.99
        
        # Failure factors
        node_failure_factor = max(0.0, 1.0 - (node_count / 2000.0))  # More nodes = more failures
        byzantine_failure_factor = max(0.0, 1.0 - byzantine_ratio * 1.5)  # Byzantine nodes cause failures
        resource_failure_factor = resource_factors.get(config.resource_profile, 1.0)
        
        success_rate = base_success_rate * node_failure_factor * byzantine_failure_factor * min(1.0, resource_failure_factor)
        success = random.random() < success_rate
        
        # Additional metrics
        metrics = {
            'estimated_bandwidth_kb': node_count * 0.5 * network_factor,  # KB per operation
            'message_count': max(3, int(node_count * 0.1)),  # Messages per operation
            'cpu_usage_estimate': min(100.0, 10.0 + (node_count / 50.0)),  # CPU % estimate
            'memory_usage_mb': max(10.0, node_count * 0.1),  # Memory per operation
            'network_efficiency': max(0.1, 1.0 - (node_count / 1000.0))  # Network efficiency
        }
        
        return success, actual_time_ms, metrics
    
    async def run_scaling_test_at_node_count(self, 
                                           config: ScalingTestConfig, 
                                           node_count: int,
                                           byzantine_ratio: float = 0.0) -> Dict[str, Any]:
        """Run a scaling test at a specific node count"""
        
        print(f"🎯 Testing {node_count} nodes (Byzantine: {byzantine_ratio:.1%})")
        
        # Warmup phase
        if config.warmup_duration > 0:
            print(f"   🔥 Warming up for {config.warmup_duration}s...")
            warmup_start = time.time()
            while time.time() - warmup_start < config.warmup_duration:
                await self.simulate_scaled_consensus_operation(config, node_count, byzantine_ratio)
                await asyncio.sleep(0.1)
        
        # Main test phase
        print(f"   ⏱️ Running test for {config.test_duration_per_scale}s...")
        test_start = time.time()
        operations = []
        total_operations = 0
        successful_operations = 0
        total_metrics = {'estimated_bandwidth_kb': 0, 'message_count': 0, 'cpu_usage_estimate': 0, 'memory_usage_mb': 0}
        
        target_interval = 1.0 / config.target_operations_per_second
        
        while time.time() - test_start < config.test_duration_per_scale:
            operation_start = time.time()
            
            try:
                success, latency_ms, metrics = await self.simulate_scaled_consensus_operation(config, node_count, byzantine_ratio)
                
                operations.append(latency_ms)
                total_operations += 1
                if success:
                    successful_operations += 1
                
                # Aggregate metrics
                for key, value in metrics.items():
                    if key in total_metrics:
                        total_metrics[key] += value
                    else:
                        total_metrics[key] = value
                
                # Rate limiting
                elapsed = time.time() - operation_start
                remaining = target_interval - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                    
            except Exception as e:
                print(f"   ⚠️ Operation failed: {e}")
        
        test_duration = time.time() - test_start
        
        # Calculate statistics
        if operations:
            mean_latency = statistics.mean(operations)
            median_latency = statistics.median(operations)
            std_dev = statistics.stdev(operations) if len(operations) > 1 else 0
            min_latency = min(operations)
            max_latency = max(operations)
            
            sorted_ops = sorted(operations)
            p95_latency = sorted_ops[int(0.95 * len(sorted_ops))] if sorted_ops else 0
            p99_latency = sorted_ops[int(0.99 * len(sorted_ops))] if sorted_ops else 0
        else:
            mean_latency = median_latency = std_dev = min_latency = max_latency = p95_latency = p99_latency = 0
        
        # Calculate average metrics
        avg_metrics = {key: value / max(total_operations, 1) for key, value in total_metrics.items()}
        
        # Cooldown phase
        if config.cooldown_duration > 0:
            print(f"   🧊 Cooling down for {config.cooldown_duration}s...")
            await asyncio.sleep(config.cooldown_duration)
        
        result = {
            'node_count': node_count,
            'byzantine_ratio': byzantine_ratio,
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'operations_per_second': total_operations / test_duration if test_duration > 0 else 0,
            'success_rate': successful_operations / max(total_operations, 1),
            'mean_latency_ms': mean_latency,
            'median_latency_ms': median_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'std_dev_ms': std_dev,
            'test_duration_seconds': test_duration,
            'operations_per_node_per_second': (total_operations / test_duration) / node_count if test_duration > 0 and node_count > 0 else 0,
            **avg_metrics
        }
        
        print(f"   ✅ Results: {result['operations_per_second']:.2f} ops/s, {result['mean_latency_ms']:.2f}ms avg, {result['success_rate']:.1%} success")
        
        return result
    
    async def run_comprehensive_scaling_test(self, config: ScalingTestConfig) -> ScalingTestResult:
        """Run a comprehensive scaling test across all configured node counts"""
        
        print(f"\n🚀 Running scaling test: {config.name}")
        print(f"   Environment: {config.environment.value}")
        print(f"   Node counts: {config.node_counts}")
        print(f"   Resource profile: {config.resource_profile.value}")
        print(f"   Network topology: {config.network_topology.value}")
        print(f"   Network conditions: {[c.value for c in config.network_conditions]}")
        print(f"   Byzantine ratios: {config.byzantine_ratios}")
        
        start_time = datetime.now(timezone.utc)
        self.current_test = config
        
        # Start resource monitoring
        if config.enable_resource_monitoring:
            self.resource_monitor.start_monitoring()
        
        # Initialize result tracking
        node_performance = {}
        resource_usage = {}
        network_metrics = {}
        byzantine_resilience = {}
        resource_logs = []
        
        try:
            # Test each node count
            for node_count in config.node_counts:
                print(f"\n📊 Testing scale: {node_count} nodes")
                
                # Test each Byzantine ratio
                for byzantine_ratio in config.byzantine_ratios:
                    result = await self.run_scaling_test_at_node_count(config, node_count, byzantine_ratio)
                    
                    # Store results
                    if node_count not in node_performance:
                        node_performance[node_count] = {}
                        resource_usage[node_count] = {}
                        network_metrics[node_count] = {}
                    
                    # Performance metrics
                    if byzantine_ratio == 0.0:  # Base performance without Byzantine nodes
                        node_performance[node_count] = {
                            'operations_per_second': result['operations_per_second'],
                            'mean_latency_ms': result['mean_latency_ms'],
                            'p95_latency_ms': result['p95_latency_ms'],
                            'success_rate': result['success_rate'],
                            'operations_per_node_per_second': result['operations_per_node_per_second']
                        }
                        
                        # Resource metrics
                        resource_usage[node_count] = {
                            'cpu_usage_estimate': result['cpu_usage_estimate'],
                            'memory_usage_mb': result['memory_usage_mb']
                        }
                        
                        # Network metrics
                        network_metrics[node_count] = {
                            'estimated_bandwidth_kb': result['estimated_bandwidth_kb'],
                            'message_count': result['message_count'],
                            'network_efficiency': result['network_efficiency']
                        }
                    
                    # Byzantine resilience
                    if byzantine_ratio not in byzantine_resilience:
                        byzantine_resilience[byzantine_ratio] = {}
                    byzantine_resilience[byzantine_ratio][node_count] = result['success_rate']
                
                # Collect resource metrics
                if config.enable_resource_monitoring:
                    current_metrics = self.resource_monitor.get_metrics()
                    for metric in current_metrics:
                        metric['node_count'] = node_count
                        resource_logs.append(metric)
                
                # Memory management for large tests
                if node_count >= 500:
                    gc.collect()  # Force garbage collection for large tests
        
        finally:
            # Stop resource monitoring
            if config.enable_resource_monitoring:
                self.resource_monitor.stop_monitoring()
                # Get final metrics
                final_metrics = self.resource_monitor.get_metrics()
                resource_logs.extend(final_metrics)
        
        end_time = datetime.now(timezone.utc)
        
        # Calculate scaling analysis
        scaling_efficiency = self._calculate_scaling_efficiency(node_performance)
        throughput_saturation_point = self._find_throughput_saturation(node_performance)
        latency_degradation_point = self._find_latency_degradation(node_performance)
        
        # Analyze resource trends
        memory_usage_trend = [(nc, metrics.get('memory_usage_mb', 0)) for nc, metrics in resource_usage.items()]
        cpu_usage_trend = [(nc, metrics.get('cpu_usage_estimate', 0)) for nc, metrics in resource_usage.items()]
        
        # Calculate bandwidth requirements
        bandwidth_requirements = {nc: metrics.get('estimated_bandwidth_kb', 0) for nc, metrics in network_metrics.items()}
        message_overhead = {nc: metrics.get('message_count', 0) for nc, metrics in network_metrics.items()}
        
        # Performance assessment
        recommended_max_nodes = self._calculate_recommended_max_nodes(node_performance, resource_usage)
        performance_bottlenecks = self._identify_bottlenecks(node_performance, resource_usage, network_metrics)
        scaling_recommendations = self._generate_scaling_recommendations(scaling_efficiency, throughput_saturation_point, latency_degradation_point)
        
        # Create result object
        result = ScalingTestResult(
            config=config,
            start_time=start_time,
            end_time=end_time,
            node_performance=node_performance,
            resource_usage=resource_usage,
            network_metrics=network_metrics,
            scaling_efficiency=scaling_efficiency,
            throughput_saturation_point=throughput_saturation_point,
            latency_degradation_point=latency_degradation_point,
            memory_usage_trend=memory_usage_trend,
            cpu_usage_trend=cpu_usage_trend,
            bandwidth_requirements=bandwidth_requirements,
            message_overhead=message_overhead,
            byzantine_resilience=byzantine_resilience,
            recommended_max_nodes=recommended_max_nodes,
            performance_bottlenecks=performance_bottlenecks,
            scaling_recommendations=scaling_recommendations,
            resource_logs=resource_logs
        )
        
        self.results_history.append(result)
        
        # Print summary
        self._print_scaling_summary(result)
        
        return result
    
    def _calculate_scaling_efficiency(self, performance_data: Dict[int, Dict[str, float]]) -> Dict[int, float]:
        """Calculate scaling efficiency relative to smallest scale"""
        if not performance_data:
            return {}
        
        sorted_nodes = sorted(performance_data.keys())
        baseline_nodes = sorted_nodes[0]
        baseline_throughput = performance_data[baseline_nodes]['operations_per_second']
        
        efficiency = {}
        for node_count in sorted_nodes:
            current_throughput = performance_data[node_count]['operations_per_second']
            scaling_factor = node_count / baseline_nodes
            throughput_ratio = current_throughput / baseline_throughput if baseline_throughput > 0 else 0
            efficiency[node_count] = throughput_ratio / scaling_factor if scaling_factor > 0 else 0
        
        return efficiency
    
    def _find_throughput_saturation(self, performance_data: Dict[int, Dict[str, float]]) -> Optional[int]:
        """Find the node count where throughput saturates"""
        if len(performance_data) < 3:
            return None
        
        sorted_nodes = sorted(performance_data.keys())
        throughputs = [performance_data[nc]['operations_per_second'] for nc in sorted_nodes]
        
        # Look for saturation (throughput increase < 10% despite 2x nodes)
        for i in range(1, len(sorted_nodes)):
            if i < len(sorted_nodes) - 1:
                prev_throughput = throughputs[i-1]
                curr_throughput = throughputs[i]
                
                if prev_throughput > 0:
                    improvement = (curr_throughput - prev_throughput) / prev_throughput
                    if improvement < 0.1:  # Less than 10% improvement
                        return sorted_nodes[i]
        
        return None
    
    def _find_latency_degradation(self, performance_data: Dict[int, Dict[str, float]]) -> Optional[int]:
        """Find the node count where latency starts degrading significantly"""
        if len(performance_data) < 3:
            return None
        
        sorted_nodes = sorted(performance_data.keys())
        latencies = [performance_data[nc]['mean_latency_ms'] for nc in sorted_nodes]
        
        # Look for significant latency increase (>50% increase)
        baseline_latency = latencies[0]
        
        for i, latency in enumerate(latencies[1:], 1):
            if baseline_latency > 0:
                increase = (latency - baseline_latency) / baseline_latency
                if increase > 0.5:  # 50% increase
                    return sorted_nodes[i]
        
        return None
    
    def _calculate_recommended_max_nodes(self, performance_data: Dict[int, Dict[str, float]], resource_data: Dict[int, Dict[str, float]]) -> int:
        """Calculate recommended maximum nodes based on performance and resource constraints"""
        if not performance_data:
            return 100  # Default
        
        sorted_nodes = sorted(performance_data.keys())
        
        # Find the last node count with acceptable performance
        for node_count in reversed(sorted_nodes):
            perf = performance_data[node_count]
            
            # Criteria: success rate > 95%, operations_per_node > 0.01
            if perf['success_rate'] > 0.95 and perf['operations_per_node_per_second'] > 0.01:
                return node_count
        
        return sorted_nodes[0] if sorted_nodes else 100
    
    def _identify_bottlenecks(self, performance_data: Dict[int, Dict[str, float]], 
                            resource_data: Dict[int, Dict[str, float]], 
                            network_data: Dict[int, Dict[str, float]]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if not performance_data:
            return bottlenecks
        
        sorted_nodes = sorted(performance_data.keys())
        
        # Check for throughput degradation
        if len(sorted_nodes) >= 2:
            first_throughput = performance_data[sorted_nodes[0]]['operations_per_second']
            last_throughput = performance_data[sorted_nodes[-1]]['operations_per_second']
            
            if last_throughput < first_throughput * 0.5:
                bottlenecks.append("Severe throughput degradation at scale")
        
        # Check for latency explosion
        if len(sorted_nodes) >= 2:
            first_latency = performance_data[sorted_nodes[0]]['mean_latency_ms']
            last_latency = performance_data[sorted_nodes[-1]]['mean_latency_ms']
            
            if last_latency > first_latency * 3.0:
                bottlenecks.append("Latency explosion at high node counts")
        
        # Check resource constraints
        for node_count in sorted_nodes[-3:]:  # Check last 3 scales
            if node_count in resource_data:
                resource = resource_data[node_count]
                if resource.get('cpu_usage_estimate', 0) > 80:
                    bottlenecks.append("CPU usage bottleneck")
                if resource.get('memory_usage_mb', 0) > 8192:  # 8GB
                    bottlenecks.append("Memory usage bottleneck")
        
        # Check network efficiency
        for node_count in sorted_nodes[-2:]:  # Check last 2 scales
            if node_count in network_data:
                network = network_data[node_count]
                if network.get('network_efficiency', 1.0) < 0.3:
                    bottlenecks.append("Network efficiency degradation")
        
        return bottlenecks
    
    def _generate_scaling_recommendations(self, scaling_efficiency: Dict[int, float], 
                                        saturation_point: Optional[int], 
                                        degradation_point: Optional[int]) -> List[str]:
        """Generate scaling recommendations"""
        recommendations = []
        
        # Efficiency-based recommendations
        if scaling_efficiency:
            avg_efficiency = statistics.mean(scaling_efficiency.values())
            if avg_efficiency > 0.8:
                recommendations.append("Excellent scaling characteristics - suitable for large deployments")
            elif avg_efficiency > 0.6:
                recommendations.append("Good scaling characteristics - optimize for better efficiency")
            elif avg_efficiency > 0.4:
                recommendations.append("Moderate scaling - consider architectural improvements")
            else:
                recommendations.append("Poor scaling - significant optimization needed")
        
        # Saturation point recommendations
        if saturation_point:
            recommendations.append(f"Throughput saturates around {saturation_point} nodes - consider sharding or hierarchical consensus")
        
        # Degradation point recommendations
        if degradation_point:
            recommendations.append(f"Latency degrades significantly beyond {degradation_point} nodes - implement latency optimizations")
        
        # General recommendations
        recommendations.extend([
            "Consider implementing adaptive consensus based on network size",
            "Monitor resource usage closely in production deployments",
            "Implement gradual scaling with performance validation at each step"
        ])
        
        return recommendations
    
    def _print_scaling_summary(self, result: ScalingTestResult):
        """Print comprehensive scaling test summary"""
        print(f"\n📈 SCALING TEST SUMMARY: {result.config.name}")
        print("=" * 80)
        
        # Performance summary
        print("🚀 PERFORMANCE ANALYSIS:")
        for node_count in sorted(result.node_performance.keys()):
            perf = result.node_performance[node_count]
            efficiency = result.scaling_efficiency.get(node_count, 0)
            print(f"   {node_count:4d} nodes: {perf['operations_per_second']:6.2f} ops/s, "
                  f"{perf['mean_latency_ms']:6.1f}ms avg, {efficiency:.3f} efficiency")
        
        # Key findings
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   Recommended max nodes: {result.recommended_max_nodes}")
        if result.throughput_saturation_point:
            print(f"   Throughput saturation: {result.throughput_saturation_point} nodes")
        if result.latency_degradation_point:
            print(f"   Latency degradation: {result.latency_degradation_point} nodes")
        
        # Bottlenecks
        if result.performance_bottlenecks:
            print(f"\n⚠️ PERFORMANCE BOTTLENECKS:")
            for bottleneck in result.performance_bottlenecks:
                print(f"   - {bottleneck}")
        
        # Recommendations
        print(f"\n💡 SCALING RECOMMENDATIONS:")
        for rec in result.scaling_recommendations[:3]:  # Top 3 recommendations
            print(f"   - {rec}")
        
        print("=" * 80)
    
    def save_scaling_results(self, result: ScalingTestResult, filename: Optional[str] = None) -> str:
        """Save scaling test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scaling_test_{result.config.name}_{timestamp}.json"
        
        filepath = self.output_directory / filename
        
        # Convert result to JSON-serializable format
        result_data = {
            "config": {
                "name": result.config.name,
                "environment": result.config.environment.value,
                "node_counts": result.config.node_counts,
                "resource_profile": result.config.resource_profile.value,
                "network_topology": result.config.network_topology.value,
                "network_conditions": [c.value for c in result.config.network_conditions],
                "test_duration_per_scale": result.config.test_duration_per_scale,
                "byzantine_ratios": result.config.byzantine_ratios,
                "target_operations_per_second": result.config.target_operations_per_second,
                "metadata": result.config.metadata
            },
            "timing": {
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "total_duration_seconds": (result.end_time - result.start_time).total_seconds()
            },
            "performance": {
                "node_performance": result.node_performance,
                "scaling_efficiency": result.scaling_efficiency,
                "throughput_saturation_point": result.throughput_saturation_point,
                "latency_degradation_point": result.latency_degradation_point,
                "recommended_max_nodes": result.recommended_max_nodes
            },
            "resources": {
                "resource_usage": result.resource_usage,
                "memory_usage_trend": result.memory_usage_trend,
                "cpu_usage_trend": result.cpu_usage_trend
            },
            "network": {
                "network_metrics": result.network_metrics,
                "bandwidth_requirements": result.bandwidth_requirements,
                "message_overhead": result.message_overhead
            },
            "byzantine_analysis": {
                "byzantine_resilience": result.byzantine_resilience
            },
            "analysis": {
                "performance_bottlenecks": result.performance_bottlenecks,
                "scaling_recommendations": result.scaling_recommendations
            },
            "raw_data": {
                "resource_logs_count": len(result.resource_logs),
                "detailed_results_count": len(result.detailed_results)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # Also save a CSV summary
        csv_filepath = filepath.with_suffix('.csv')
        with open(csv_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'node_count', 'operations_per_second', 'mean_latency_ms', 'p95_latency_ms', 
                'success_rate', 'scaling_efficiency', 'cpu_usage_estimate', 'memory_usage_mb'
            ])
            
            for node_count in sorted(result.node_performance.keys()):
                perf = result.node_performance[node_count]
                resource = result.resource_usage.get(node_count, {})
                efficiency = result.scaling_efficiency.get(node_count, 0)
                
                writer.writerow([
                    node_count,
                    perf['operations_per_second'],
                    perf['mean_latency_ms'],
                    perf['p95_latency_ms'],
                    perf['success_rate'],
                    efficiency,
                    resource.get('cpu_usage_estimate', 0),
                    resource.get('memory_usage_mb', 0)
                ])
        
        print(f"\n💾 Results saved:")
        print(f"   📄 JSON: {filepath}")
        print(f"   📄 CSV: {csv_filepath}")
        
        return str(filepath)


# Demo and test functions
async def run_scaling_demo():
    """Run a quick scaling test demo"""
    print("🎯 PRSM Scaling Test Controller - Demo")
    print("=" * 60)
    
    controller = ScalingTestController()
    
    # Create a lightweight demo config
    demo_config = ScalingTestConfig(
        name="demo_scaling_test",
        environment=ScalingEnvironment.LOCAL_SIMULATION,
        node_counts=[10, 25, 50],
        resource_profile=ResourceProfile.STANDARD,
        network_topology=NetworkTopology.MESH,
        network_conditions=[NetworkCondition.WAN],
        test_duration_per_scale=8,  # Short for demo
        warmup_duration=2,
        byzantine_ratios=[0.0, 0.1],
        target_operations_per_second=12.0,
        enable_resource_monitoring=True,
        metadata={"demo": True}
    )
    
    # Run the test
    result = await controller.run_comprehensive_scaling_test(demo_config)
    
    # Save results
    controller.save_scaling_results(result)
    
    print(f"\n✅ Demo completed! Tested {len(demo_config.node_counts)} node scales.")
    return result


async def run_comprehensive_scaling_suite():
    """Run the full comprehensive scaling test suite"""
    print("🌟 PRSM COMPREHENSIVE SCALING TEST SUITE")
    print("=" * 80)
    
    controller = ScalingTestController()
    configs = controller.create_comprehensive_scaling_configs()
    
    print(f"🎯 Running {len(configs)} comprehensive scaling test configurations...")
    
    # Run a subset for demonstration (full suite would take hours)
    demo_configs = [
        config for config in configs 
        if config.name in [
            "comprehensive_scaling_progression",
            "network_scaling_wan",
            "resource_constrained_standard"
        ]
    ]
    
    results = []
    for i, config in enumerate(demo_configs, 1):
        print(f"\n[{i}/{len(demo_configs)}] Starting: {config.name}")
        
        try:
            result = await controller.run_comprehensive_scaling_test(config)
            results.append(result)
            controller.save_scaling_results(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Comprehensive scaling suite completed!")
    print(f"📊 Results: {len(results)} successful tests")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "comprehensive":
        # Run comprehensive suite
        asyncio.run(run_comprehensive_scaling_suite())
    else:
        # Run demo
        asyncio.run(run_scaling_demo())
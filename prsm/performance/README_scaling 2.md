# PRSM Scaling Test Controller

## 📊 Overview

The PRSM Scaling Test Controller provides comprehensive environmental controls for large-scale performance testing, supporting node counts from 10 to 1000+ nodes across various network conditions, resource profiles, and network topologies.

## 🎯 Features

### 📈 Comprehensive Scaling Analysis
- **Node Scaling**: Test performance from 10 to 1000+ nodes
- **Scaling Efficiency**: Calculate performance efficiency as nodes increase
- **Throughput Saturation**: Identify where performance plateaus
- **Latency Degradation**: Find where latency becomes unacceptable
- **Resource Analysis**: Track CPU, memory, and bandwidth usage

### 🌐 Environmental Controls
- **Multiple Environments**: Local simulation, Docker clusters, multi-region cloud, hybrid edge
- **Resource Profiles**: From minimal (512MB) to extreme (8GB+) resource allocations
- **Network Topologies**: Fully connected, ring, star, mesh, hierarchical, random patterns
- **Network Conditions**: Ideal, LAN, WAN, intercontinental, poor connectivity simulation
- **Byzantine Fault Testing**: Configurable Byzantine node ratios (0-33%)

### 📊 Performance Metrics
- **Throughput**: Operations per second, per-node efficiency
- **Latency**: Mean, median, P95, P99 percentiles
- **Success Rates**: Consensus success, message delivery success
- **Resource Usage**: CPU, memory, bandwidth estimates
- **Network Efficiency**: Message overhead, bandwidth requirements

### 🔍 Analysis & Recommendations
- **Bottleneck Identification**: Automatic detection of performance constraints
- **Scaling Recommendations**: AI-generated optimization suggestions
- **Performance Trends**: Historical analysis and regression detection
- **Resource Planning**: Capacity planning recommendations

## 🚀 Quick Start

### 1. Basic Usage
```python
from prsm.performance.scaling_test_controller import ScalingTestController

# Create controller
controller = ScalingTestController()

# Run quick demo
result = await controller.run_scaling_demo()
```

### 2. Command Line Usage
```bash
# Quick demo (3 node scales, ~1 minute)
python prsm/performance/scaling_test_controller.py

# Comprehensive suite (12 configurations, ~30 minutes)
python prsm/performance/scaling_test_controller.py comprehensive

# Quick test for validation
python test_scaling_demo.py
```

## 📋 Scaling Test Configurations

### 1. Comprehensive Scaling Progression
- **Node Counts**: 10, 20, 40, 80, 160, 320, 640, 1000
- **Purpose**: Understand overall scaling characteristics
- **Network**: WAN conditions with mesh topology
- **Byzantine**: 0%, 10%, 20% Byzantine nodes
- **Duration**: 45s per scale + 15s warmup

### 2. Network Impact Analysis
- **Node Counts**: 25, 100, 400 nodes
- **Conditions**: LAN, WAN, Intercontinental
- **Purpose**: Understand network condition impact on scaling
- **Topology**: Mesh network for realistic P2P simulation

### 3. Resource Constraint Testing
- **Profiles**: Minimal (512MB), Light (1GB), Heavy (4GB)
- **Node Counts**: 50, 200, 500 nodes
- **Purpose**: Understand resource bottlenecks
- **Constraints**: Memory and CPU limits enforced

### 4. Network Topology Comparison
- **Topologies**: Star, Ring, Hierarchical patterns
- **Node Counts**: 30, 120, 480 nodes
- **Purpose**: Optimize network architecture
- **Analysis**: Message overhead and efficiency comparison

### 5. Byzantine Fault Tolerance
- **Node Counts**: 50, 150, 300, 600 nodes
- **Byzantine Ratios**: 0%, 10%, 20%, 33%
- **Purpose**: Validate consensus resilience at scale
- **Fault Injection**: Simulated Byzantine behavior

### 6. Extreme Scale Stress Test
- **Node Counts**: 100, 500, 1000 nodes
- **Resources**: Heavy profile (8 CPU, 4GB RAM)
- **Purpose**: Test absolute limits
- **Duration**: 90s per scale with monitoring

## 🔧 Configuration Options

### Scaling Environments
```python
class ScalingEnvironment(Enum):
    LOCAL_SIMULATION = "local_simulation"      # Single machine
    DOCKER_CLUSTER = "docker_cluster"          # Multi-container
    MULTI_REGION_CLOUD = "multi_region_cloud"  # Distributed cloud
    HYBRID_EDGE = "hybrid_edge"               # Cloud + edge mix
```

### Resource Profiles
```python
class ResourceProfile(Enum):
    MINIMAL = "minimal"     # 1 CPU, 512MB RAM
    LIGHT = "light"        # 2 CPU, 1GB RAM  
    STANDARD = "standard"  # 4 CPU, 2GB RAM
    HEAVY = "heavy"        # 8 CPU, 4GB RAM
    EXTREME = "extreme"    # 16 CPU, 8GB RAM
```

### Network Topologies
```python
class NetworkTopology(Enum):
    FULLY_CONNECTED = "fully_connected"  # All-to-all (worst scaling)
    RING = "ring"                       # Circular (poor scaling)
    STAR = "star"                       # Hub-spoke (better scaling)
    MESH = "mesh"                       # Partial mesh (moderate)
    HIERARCHICAL = "hierarchical"       # Tree structure (good scaling)
    RANDOM = "random"                   # Random connections (variable)
```

### Network Conditions
- **Ideal**: 0ms latency, 0% packet loss
- **LAN**: 1-5ms latency, 0.01% packet loss
- **WAN**: 50-100ms latency, 0.1% packet loss
- **Intercontinental**: 150-300ms latency, 0.5% packet loss
- **Poor**: 200-500ms latency, 2% packet loss

## 📊 Sample Configuration

```python
config = ScalingTestConfig(
    name="comprehensive_scaling_test",
    environment=ScalingEnvironment.LOCAL_SIMULATION,
    node_counts=[10, 25, 50, 100, 200],
    resource_profile=ResourceProfile.STANDARD,
    network_topology=NetworkTopology.MESH,
    network_conditions=[NetworkCondition.WAN],
    test_duration_per_scale=60,
    warmup_duration=15,
    cooldown_duration=10,
    byzantine_ratios=[0.0, 0.1, 0.2],
    target_operations_per_second=10.0,
    enable_resource_monitoring=True,
    max_memory_mb=4096,
    max_cpu_percent=80.0
)
```

## 📈 Results Analysis

### Scaling Efficiency Calculation
```python
# Perfect scaling efficiency = 1.0
# 2x nodes = 2x throughput = 1.0 efficiency
# 2x nodes = 1.5x throughput = 0.75 efficiency
efficiency = (throughput_ratio) / (node_scaling_factor)
```

### Bottleneck Detection
- **Throughput Degradation**: <50% of baseline performance
- **Latency Explosion**: >3x baseline latency
- **CPU Bottleneck**: >80% CPU usage
- **Memory Bottleneck**: >8GB memory usage
- **Network Inefficiency**: <30% network efficiency

### Recommendation Engine
- **Excellent Scaling** (>80% efficiency): Suitable for large deployments
- **Good Scaling** (60-80% efficiency): Optimize for better efficiency
- **Moderate Scaling** (40-60% efficiency): Consider architectural improvements
- **Poor Scaling** (<40% efficiency): Significant optimization needed

## 📊 Result Formats

### JSON Output
```json
{
  "config": {
    "name": "comprehensive_scaling_test",
    "node_counts": [10, 25, 50],
    "environment": "local_simulation"
  },
  "performance": {
    "node_performance": {
      "10": {"operations_per_second": 14.74, "mean_latency_ms": 10.5},
      "25": {"operations_per_second": 18.32, "mean_latency_ms": 15.2}
    },
    "scaling_efficiency": {
      "10": 1.000,
      "25": 0.744
    },
    "recommended_max_nodes": 200
  },
  "analysis": {
    "performance_bottlenecks": ["CPU usage bottleneck"],
    "scaling_recommendations": [
      "Good scaling characteristics - optimize for better efficiency",
      "Consider implementing adaptive consensus based on network size"
    ]
  }
}
```

### CSV Output
```csv
node_count,operations_per_second,mean_latency_ms,p95_latency_ms,success_rate,scaling_efficiency
10,14.74,10.47,18.2,1.000,1.000
25,18.32,15.23,24.8,0.973,0.744
50,22.15,22.45,35.6,0.951,0.534
```

## 🔍 Performance Monitoring

### Resource Monitoring
```python
# Automatic monitoring of:
- CPU usage (system and process)
- Memory usage (system and process)
- Network I/O (bytes sent/received)
- Disk I/O (read/write operations)
```

### Real-time Metrics
- Operations per second
- Success/failure rates  
- Latency percentiles
- Resource utilization
- Network efficiency

## 🛠️ Advanced Usage

### Custom Configuration
```python
# Create custom scaling test
custom_config = ScalingTestConfig(
    name="custom_test",
    node_counts=[10, 50, 100, 500, 1000],
    resource_profile=ResourceProfile.HEAVY,
    network_topology=NetworkTopology.HIERARCHICAL,
    byzantine_ratios=[0.0, 0.15, 0.25],
    enable_fault_injection=True,
    max_memory_mb=8192,
    save_detailed_logs=True
)

# Run test
controller = ScalingTestController()
result = await controller.run_comprehensive_scaling_test(custom_config)
```

### Batch Testing
```python
# Create multiple configurations
configs = controller.create_comprehensive_scaling_configs()

# Filter specific test types
network_tests = [c for c in configs if c.metadata.get("test_type") == "network_impact"]

# Run subset
for config in network_tests:
    result = await controller.run_comprehensive_scaling_test(config)
    controller.save_scaling_results(result)
```

## 📁 Output Structure

```
scaling_test_results/
├── scaling_test_comprehensive_scaling_20250623_160404.json
├── scaling_test_comprehensive_scaling_20250623_160404.csv
├── scaling_test_network_impact_wan_20250623_161205.json
├── scaling_test_network_impact_wan_20250623_161205.csv
└── ...
```

## 🎯 Use Cases

### 1. Production Capacity Planning
- Determine optimal node count for target throughput
- Identify resource requirements at scale
- Plan gradual scaling strategy

### 2. Architecture Optimization
- Compare network topology performance
- Identify scaling bottlenecks
- Validate consensus algorithm efficiency

### 3. Investment Demonstrations
- Show scaling characteristics to investors
- Demonstrate performance under adversarial conditions
- Provide empirical scaling evidence

### 4. Research and Development
- Test new consensus algorithms
- Analyze Byzantine fault tolerance
- Optimize network protocols

## 🔧 Integration with Dashboard

The scaling test controller integrates seamlessly with the Streamlit dashboard:

```python
# Add scaling results to dashboard
from prsm.dashboard.performance_dashboard import PerformanceDashboard

dashboard = PerformanceDashboard()
dashboard.load_scaling_results(result)
```

## 📚 API Reference

### ScalingTestController
- `create_comprehensive_scaling_configs()`: Generate standard test configurations
- `run_comprehensive_scaling_test(config)`: Execute full scaling test
- `simulate_scaled_consensus_operation(config, nodes, byzantine_ratio)`: Single operation simulation
- `save_scaling_results(result)`: Persist results to JSON/CSV

### Analysis Functions
- `_calculate_scaling_efficiency()`: Compute scaling efficiency metrics
- `_find_throughput_saturation()`: Identify saturation point
- `_find_latency_degradation()`: Find latency degradation threshold
- `_identify_bottlenecks()`: Detect performance constraints
- `_generate_scaling_recommendations()`: AI-generated optimization advice

## 🔗 Related Documentation
- [Performance Benchmarking](./README.md)
- [Streamlit Dashboard](../dashboard/README.md)
- [Post-Quantum Cryptography](../cryptography/README.md)
- [PRSM Architecture](../../README.md)

## 📄 License
This scaling test controller is part of the PRSM project and follows the same licensing terms.
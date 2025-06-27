# PRSM Performance Dashboard

## 📊 Overview

The PRSM Performance Dashboard is a comprehensive Streamlit-based web application for monitoring, analyzing, and visualizing PRSM's performance metrics in real-time.

## 🎯 Features

### Real-Time Monitoring
- **Live Benchmark Execution**: Run benchmarks directly from the dashboard
- **Performance Metrics**: Real-time display of throughput, latency, and success rates
- **System Health**: Monitor consensus health, network status, and security metrics

### Interactive Analysis
- **Throughput Analysis**: Visualize operations per second over time and across node counts
- **Latency Distribution**: Analyze mean, P95, and P99 latency percentiles
- **Scaling Efficiency**: Track how performance scales with node count
- **Network Impact**: Compare performance across different network conditions

### Historical Tracking
- **Benchmark History**: View past benchmark results and trends
- **Performance Regression**: Detect performance degradations over time
- **Comparative Analysis**: Compare different benchmark configurations

### Visualization Types
- **Time Series Charts**: Track metrics over time
- **Scaling Analysis**: Node count vs performance efficiency
- **Distribution Charts**: Latency percentiles and success rate distributions
- **Network Comparison**: Performance across different network conditions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install required packages
pip install streamlit plotly pandas numpy

# Or install from requirements
pip install -r prsm/dashboard/requirements.txt
```

### 2. Launch Dashboard
```bash
# Option 1: Using the launcher script
python run_dashboard.py

# Option 2: Direct Streamlit command
streamlit run prsm/dashboard/performance_dashboard.py

# Option 3: With custom port
streamlit run prsm/dashboard/performance_dashboard.py --server.port 8502
```

### 3. Access Dashboard
Open your browser to: http://localhost:8501

## 📋 Dashboard Sections

### 🏠 Overview
- **Performance Summary**: Key metrics at a glance
- **Recent Trends**: Latest benchmark results
- **Quick Actions**: Run demo benchmarks

### 🧪 Live Benchmarks
- **Benchmark Configuration**: Set up custom benchmark parameters
- **Real-Time Execution**: Watch benchmarks run live
- **Immediate Results**: See results as soon as benchmarks complete

### 📈 Historical Analysis
- **Trend Analysis**: Long-term performance trends
- **Scaling Studies**: How performance changes with scale
- **Network Impact**: Performance across different network conditions

### 💚 System Health
- **Consensus Status**: Health of the consensus mechanism
- **Network Status**: P2P network connectivity and peer count
- **Security Status**: Post-quantum cryptography status
- **Performance Status**: Overall system performance health

## 🔧 Configuration

### Benchmark Parameters
- **Benchmark Type**: consensus_scaling, network_throughput, post_quantum_overhead, etc.
- **Node Count**: 5-1000 nodes (configurable range)
- **Duration**: 5-300 seconds
- **Target Operations/Sec**: 1-100 ops/sec
- **Network Condition**: ideal, lan, wan, intercontinental, poor
- **Post-Quantum**: Enable/disable post-quantum signature overhead

### Network Conditions
- **Ideal**: 0ms latency, 0% packet loss
- **LAN**: 1-5ms latency, 0.01% packet loss  
- **WAN**: 50-100ms latency, 0.1% packet loss
- **Intercontinental**: 150-300ms latency, 0.5% packet loss
- **Poor**: 200-500ms latency, 2% packet loss

## 📊 Metrics Explained

### Throughput Metrics
- **Operations per Second**: Total operations completed per second
- **Operations per Node per Second**: Per-node efficiency metric
- **Success Rate**: Percentage of successful operations

### Latency Metrics
- **Mean Latency**: Average operation completion time
- **P95 Latency**: 95th percentile latency (95% of operations complete faster)
- **P99 Latency**: 99th percentile latency (99% of operations complete faster)
- **Standard Deviation**: Variability in operation times

### Scaling Metrics
- **Scaling Efficiency**: How well performance scales with node count
- **Throughput Ratio**: Relative performance compared to baseline
- **Linear Scaling Factor**: Ideal vs actual scaling performance

### Network Metrics
- **Network Overhead**: Simulated network delay impact
- **Bandwidth Usage**: Estimated network bandwidth consumption
- **Message Success Rate**: Network-level message delivery success

## 🧪 Running Benchmarks

### Quick Demo
1. Navigate to "Live Benchmarks" section
2. Click "📊 Run Quick Demo" for a fast 5-second benchmark
3. View results immediately in the dashboard

### Custom Benchmark
1. Configure benchmark parameters in the "Benchmark Configuration" section
2. Set node count, duration, target ops/sec, and network conditions
3. Click "🚀 Run Benchmark" to start execution
4. Monitor real-time progress and metrics
5. View detailed results upon completion

### Benchmark Types
- **Consensus Scaling**: Tests consensus performance as node count increases
- **Network Throughput**: Tests network message passing capabilities
- **Post-Quantum Overhead**: Measures impact of post-quantum signatures
- **Latency Distribution**: Analyzes latency characteristics under load
- **Stress Test**: High-load testing with Byzantine nodes

## 📁 Data Storage

### Results Location
- **Benchmark Results**: `benchmark_results/`
- **Dashboard Results**: `dashboard_results/`
- **Test Results**: `test_dashboard_results/`

### File Formats
- **JSON**: Detailed benchmark results with full metadata
- **CSV**: Summary results for easy analysis in external tools

### Result Structure
```json
{
  "benchmark_suite_version": "1.0",
  "timestamp": "20250623_155506",
  "total_benchmarks": 3,
  "results": [
    {
      "config": {
        "name": "demo_small_consensus",
        "benchmark_type": "consensus_scaling",
        "node_count": 10,
        "duration_seconds": 5,
        "network_condition": "lan"
      },
      "metrics": {
        "operations_per_second": 7.93,
        "mean_latency_ms": 19.31,
        "p95_latency_ms": 25.45,
        "consensus_success_rate": 0.975
      },
      "timing": {
        "start_time": "2025-06-23T15:49:36+00:00",
        "end_time": "2025-06-23T15:49:41+00:00",
        "duration_seconds": 5.01
      }
    }
  ]
}
```

## 🔍 Troubleshooting

### Common Issues

**Dashboard won't start**
- Ensure Streamlit is installed: `pip install streamlit`
- Check Python path includes PRSM: `export PYTHONPATH=/path/to/PRSM`

**No benchmark data displayed**
- Run at least one benchmark to generate data
- Check that results are being saved to the correct directory

**Charts not loading**
- Ensure Plotly is installed: `pip install plotly`
- Check browser console for JavaScript errors

**Benchmark failures**
- Verify PRSM components are properly installed
- Check that benchmark collector is accessible

### Performance Tips
- Use shorter durations (5-30s) for interactive testing
- Use lower node counts (5-50) for faster benchmarks
- Enable auto-refresh cautiously to avoid performance impact

## 🛠️ Development

### Adding New Visualizations
1. Create new chart functions in `PerformanceDashboard` class
2. Add to appropriate dashboard section
3. Test with sample data

### Custom Metrics
1. Extend the benchmark results structure
2. Update data processing functions
3. Create corresponding visualizations

### Integration Points
- **Benchmark Suite**: `comprehensive_performance_benchmark.py`
- **Performance Collector**: `prsm/performance/benchmark_collector.py`
- **Orchestrator**: `prsm/performance/benchmark_orchestrator.py`

## 📚 API Reference

### PerformanceDashboard Class
- `load_historical_results()`: Load saved benchmark results
- `create_performance_overview()`: Generate performance summary
- `create_throughput_analysis()`: Create throughput visualizations
- `create_latency_analysis()`: Create latency charts
- `create_scaling_analysis()`: Generate scaling efficiency analysis
- `run_benchmark_async()`: Execute benchmarks asynchronously

### Utility Functions
- Chart generation using Plotly
- Data processing with Pandas
- Statistical analysis with NumPy

## 🔗 Related Documentation
- [Performance Benchmarking Guide](../performance/README.md)
- [PRSM Architecture Overview](../../README.md)
- [Post-Quantum Cryptography](../cryptography/README.md)

## 📄 License
This dashboard is part of the PRSM project and follows the same licensing terms.
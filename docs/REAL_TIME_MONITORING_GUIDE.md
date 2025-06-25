# PRSM Real-Time Monitoring Dashboard Guide

## Overview

The PRSM Real-Time Monitoring Dashboard provides comprehensive visibility into system performance, network health, and AI model operations. This enterprise-grade monitoring solution enables administrators, developers, and stakeholders to track PRSM operations in real-time.

## Features

### 🖥️ System Performance Monitoring
- **CPU Usage**: Real-time CPU utilization tracking
- **Memory Usage**: Memory consumption and available resources
- **Disk Usage**: Storage utilization monitoring
- **Network Activity**: Bandwidth usage and connection tracking
- **Active Connections**: Current network connections

### 🌐 P2P Network Monitoring
- **Node Status**: Active vs total nodes in the network
- **Connection Health**: Total connections and topology
- **Message Flow**: Message rate and propagation metrics
- **Consensus Activity**: Consensus proposals and voting status
- **Network Latency**: Average response times across nodes

### 🤖 AI Model Performance
- **Model Inventory**: Total and active AI models
- **Inference Metrics**: Successful inference counts and rates
- **Confidence Tracking**: Average model confidence scores
- **Model Distribution**: Framework and type distribution
- **Performance Trends**: Historical inference performance

### 🚨 Alert System
- **Threshold Monitoring**: Automatic alerts for performance thresholds
- **Component-Specific Alerts**: System, network, and AI alerts
- **Alert Levels**: Info, warning, error, and critical classifications
- **Historical Alert Tracking**: Alert history and patterns

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                 PRSM Monitoring Dashboard                   │
├─────────────────┬─────────────────┬─────────────────────────┤
│ MetricsCollector│ Dashboard UI    │ Alert System            │
│                 │                 │                         │
│ • System Metrics│ • Flask Web UI  │ • Threshold Monitoring  │
│ • Network Stats │ • Real-time     │ • Component Alerts      │
│ • AI Performance│   Updates       │ • Historical Tracking   │
│ • Data Storage  │ • Interactive   │ • Notification System   │
│                 │   Charts        │                         │
└─────────────────┴─────────────────┴─────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                             │
│ • psutil (System) • P2P Network • AI Models • Logs         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Metrics Collection**: Real-time gathering from multiple sources
2. **Data Processing**: Aggregation and threshold checking
3. **Storage**: In-memory deque with configurable history limits
4. **Visualization**: Real-time charts and status displays
5. **Alerting**: Automatic notification generation

## Installation & Setup

### Prerequisites

```bash
# Core requirements (included in PRSM)
pip install psutil

# Optional: Full web dashboard
pip install flask flask-socketio

# Optional: Enhanced charting
pip install chart.js
```

### Quick Start

```bash
# Start dashboard with demo network
python dashboard/dashboard_demo.py

# Or start directly
python dashboard/real_time_monitoring_dashboard.py

# Custom configuration
python dashboard/real_time_monitoring_dashboard.py --host 0.0.0.0 --port 8080 --no-demo
```

## Configuration

### Environment Variables

```bash
# Dashboard configuration
export PRSM_DASHBOARD_HOST="localhost"
export PRSM_DASHBOARD_PORT="5000"
export PRSM_DASHBOARD_DEBUG="false"

# Metrics collection
export PRSM_METRICS_INTERVAL="1.0"  # seconds
export PRSM_METRICS_HISTORY="1000"  # data points

# Alert thresholds
export PRSM_CPU_THRESHOLD="80.0"      # percent
export PRSM_MEMORY_THRESHOLD="85.0"   # percent
export PRSM_LATENCY_THRESHOLD="100.0" # milliseconds
```

### Custom Thresholds

```python
# Custom threshold configuration
baseline_metrics = {
    "cpu_threshold": 75.0,
    "memory_threshold": 80.0,
    "disk_threshold": 90.0,
    "network_latency_threshold": 150.0,
    "inference_time_threshold": 3.0,
    "confidence_threshold": 0.8
}
```

## Usage Examples

### Basic Monitoring

```python
from dashboard.real_time_monitoring_dashboard import PRSMMonitoringDashboard

# Start monitoring with defaults
dashboard = PRSMMonitoringDashboard()
await dashboard.start_monitoring()
```

### With Custom Network

```python
from demos.enhanced_p2p_ai_demo import EnhancedP2PNetworkDemo

# Use existing network
network = EnhancedP2PNetworkDemo()
dashboard = PRSMMonitoringDashboard()
dashboard.network_demo = network
await dashboard.start_monitoring()
```

### Metrics-Only Collection

```python
from dashboard.real_time_monitoring_dashboard import MetricsCollector

# Collect metrics without UI
collector = MetricsCollector()
await collector.start_collection()

# Get latest data
metrics = collector.get_latest_metrics()
print(f"CPU: {metrics['system']['cpu_percent']:.1f}%")
```

## Web Dashboard Interface

### Dashboard Features

When Flask is available, the web dashboard provides:

- **Real-time Charts**: Live updating performance graphs
- **Interactive Metrics**: Clickable components and drill-down
- **Alert Notifications**: In-browser alert system
- **Historical Views**: Configurable time ranges
- **Export Capabilities**: Data export for reporting

### URL Endpoints

```
http://localhost:5000/                  # Main dashboard
http://localhost:5000/api/metrics       # Current metrics JSON
http://localhost:5000/api/historical/   # Historical data
```

### WebSocket Events

```javascript
// Connect to real-time updates
const socket = io('http://localhost:5000');

socket.on('metrics_update', (data) => {
    console.log('Real-time metrics:', data);
});

socket.on('alert', (alert) => {
    console.log('New alert:', alert);
});
```

## Metrics Reference

### System Metrics

```json
{
  "timestamp": 1750535000.0,
  "cpu_percent": 45.2,
  "memory_percent": 67.8,
  "memory_used_gb": 8.4,
  "memory_total_gb": 16.0,
  "disk_percent": 32.1,
  "network_sent_mb": 1024.5,
  "network_recv_mb": 2048.3,
  "active_connections": 12
}
```

### Network Metrics

```json
{
  "timestamp": 1750535000.0,
  "total_nodes": 5,
  "active_nodes": 4,
  "total_connections": 8,
  "total_messages": 1250,
  "consensus_proposals": 3,
  "average_latency": 75.5,
  "message_rate": 12.3
}
```

### AI Metrics

```json
{
  "timestamp": 1750535000.0,
  "total_models": 12,
  "active_models": 10,
  "total_inferences": 5678,
  "successful_inferences": 5580,
  "average_inference_time": 0.145,
  "average_confidence": 0.87,
  "model_distribution": {
    "pytorch": 6,
    "tensorflow": 3,
    "custom": 3
  }
}
```

## Alert Configuration

### Alert Levels

- **Info**: Informational messages (green)
- **Warning**: Performance concerns (yellow)
- **Error**: Component failures (orange)
- **Critical**: System-wide issues (red)

### Default Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| CPU Usage | 80% | 95% |
| Memory Usage | 85% | 95% |
| Disk Usage | 90% | 98% |
| Network Latency | 100ms | 500ms |
| AI Confidence | <70% | <50% |

### Custom Alert Rules

```python
# Add custom alert rule
async def custom_alert_check(metrics):
    if metrics['ai']['success_rate'] < 0.9:
        await collector._create_alert(
            "warning", "ai",
            f"Low AI success rate: {metrics['ai']['success_rate']:.1%}",
            {"success_rate": metrics['ai']['success_rate']}
        )
```

## Performance Optimization

### Resource Usage

- **Memory**: ~50MB base + 1KB per metric data point
- **CPU**: <5% overhead for metrics collection
- **Network**: ~1KB/s for metric updates
- **Disk**: Log files only (configurable)

### Scaling Considerations

```python
# Large deployments
collector = MetricsCollector(max_history=10000)  # More history
collector.collection_interval = 5.0              # Less frequent

# High-frequency monitoring
collector = MetricsCollector(max_history=500)    # Less history
collector.collection_interval = 0.5              # More frequent
```

## Troubleshooting

### Common Issues

#### Dashboard Not Loading
```bash
# Check if Flask is installed
pip install flask flask-socketio

# Check port availability
netstat -an | grep :5000

# Use alternative port
python dashboard/real_time_monitoring_dashboard.py --port 8080
```

#### Missing Metrics
```bash
# Check psutil installation
pip install psutil

# Verify permissions
# Some metrics require elevated permissions

# Check network demo
python demos/enhanced_p2p_ai_demo.py
```

#### High Resource Usage
```python
# Reduce collection frequency
collector.collection_interval = 5.0

# Limit history
collector = MetricsCollector(max_history=100)

# Disable network simulation
dashboard.start_monitoring(with_demo=False)
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug metrics
collector.debug_mode = True
```

## Integration

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Health Check
  run: |
    python dashboard/dashboard_test.py
    curl -f http://localhost:5000/api/metrics
```

### Monitoring Integration

```python
# Prometheus integration
from prometheus_client import Counter, Histogram

inference_counter = Counter('prsm_inferences_total')
latency_histogram = Histogram('prsm_latency_seconds')
```

### API Integration

```python
# Custom monitoring endpoint
@app.route('/health')
def health_check():
    metrics = collector.get_latest_metrics()
    return {
        "status": "healthy",
        "cpu_ok": metrics['system']['cpu_percent'] < 80,
        "memory_ok": metrics['system']['memory_percent'] < 85,
        "network_ok": len(metrics['alerts']) == 0
    }
```

## Security Considerations

### Access Control

```python
# Add authentication middleware
@app.before_request
def authenticate():
    auth_header = request.headers.get('Authorization')
    if not validate_token(auth_header):
        abort(401)
```

### Data Privacy

- Metrics contain system performance data only
- No sensitive user data or model weights exposed
- Optional data anonymization available
- Configurable data retention policies

### Network Security

```python
# HTTPS configuration
app.run(ssl_context='adhoc', host='0.0.0.0', port=443)

# CORS configuration
CORS(app, origins=['https://trusted-domain.com'])
```

## Advanced Features

### Custom Metrics

```python
# Add application-specific metrics
class CustomMetricsCollector(MetricsCollector):
    async def _collect_custom_metrics(self):
        # Your custom collection logic
        custom_data = await get_application_metrics()
        self.custom_metrics.append(custom_data)
```

### Plugin System

```python
# Dashboard plugins
class PluginManager:
    def register_plugin(self, plugin):
        self.plugins.append(plugin)
    
    async def run_plugins(self, metrics):
        for plugin in self.plugins:
            await plugin.process(metrics)
```

### Multi-Instance Monitoring

```python
# Monitor multiple PRSM instances
class MultiInstanceDashboard:
    def __init__(self, instances):
        self.instances = instances
        self.collectors = {}
    
    async def start_monitoring_all(self):
        for instance in self.instances:
            collector = MetricsCollector()
            self.collectors[instance.id] = collector
            await collector.start_collection(instance)
```

## Best Practices

### Production Deployment

1. **Resource Allocation**: Allocate sufficient resources for monitoring
2. **Data Retention**: Configure appropriate history limits
3. **Alert Tuning**: Adjust thresholds based on baseline performance
4. **Backup Monitoring**: Implement monitoring of the monitoring system
5. **Documentation**: Maintain operational runbooks

### Performance Monitoring

1. **Baseline Establishment**: Establish performance baselines
2. **Trend Analysis**: Monitor performance trends over time
3. **Capacity Planning**: Use metrics for capacity planning
4. **Anomaly Detection**: Implement anomaly detection algorithms
5. **Predictive Analytics**: Use historical data for predictions

### Alert Management

1. **Alert Fatigue**: Avoid excessive alerting
2. **Escalation Policies**: Implement alert escalation
3. **Root Cause Analysis**: Include context in alerts
4. **Alert Correlation**: Group related alerts
5. **Response Procedures**: Document response procedures

## Conclusion

The PRSM Real-Time Monitoring Dashboard provides comprehensive visibility into system operations, enabling proactive monitoring and rapid issue resolution. Its modular architecture supports both simple deployments and complex enterprise environments.

For additional support or feature requests, please refer to the main PRSM documentation or open an issue in the project repository.
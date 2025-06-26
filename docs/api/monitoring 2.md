# Monitoring & Analytics API

Track performance, system metrics, and operational insights across the PRSM network and applications.

## 🎯 Overview

The Monitoring & Analytics API provides comprehensive observability into PRSM operations including real-time metrics, performance analytics, error tracking, resource utilization, and business intelligence.

## 📋 Base URL

```
https://api.prsm.ai/v1/monitoring
```

## 🔐 Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.prsm.ai/v1/monitoring
```

## 🚀 Quick Start

### Get System Health

```python
import prsm

client = prsm.Client(api_key="your-api-key")

# Check overall system health
health = client.monitoring.system_health()
print(f"System Status: {health.status}")
print(f"Uptime: {health.uptime_percentage}%")
print(f"Response Time: {health.avg_response_time}ms")
```

## 📊 Endpoints

### GET /monitoring/health
Get overall system health and status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_percentage": 99.97,
  "response_time_ms": 45,
  "active_nodes": 1247,
  "total_requests_24h": 2847392,
  "error_rate_24h": 0.003,
  "components": {
    "api_gateway": {"status": "healthy", "response_time_ms": 12},
    "inference_engine": {"status": "healthy", "response_time_ms": 234},
    "p2p_network": {"status": "healthy", "active_peers": 1247},
    "data_storage": {"status": "healthy", "utilization_percentage": 67},
    "token_system": {"status": "healthy", "transaction_rate": 45.2}
  }
}
```

### GET /monitoring/metrics
Get detailed performance metrics.

**Query Parameters:**
- `timeframe`: Time period (1h, 24h, 7d, 30d)
- `granularity`: Data granularity (1m, 5m, 1h, 1d)
- `metrics`: Comma-separated list of specific metrics
- `component`: Filter by system component

**Response:**
```json
{
  "timeframe": "24h",
  "granularity": "1h",
  "data_points": [
    {
      "timestamp": "2024-01-15T09:00:00Z",
      "api_requests": 118492,
      "response_time_avg": 42,
      "response_time_p95": 156,
      "response_time_p99": 287,
      "error_rate": 0.002,
      "cpu_utilization": 0.34,
      "memory_utilization": 0.58,
      "network_throughput_mbps": 234.5,
      "active_users": 3421
    }
  ],
  "summary": {
    "total_requests": 2847392,
    "avg_response_time": 45,
    "peak_response_time": 312,
    "availability": 99.97,
    "error_count": 8542
  }
}
```

### GET /monitoring/performance
Get performance analytics and insights.

**Response:**
```json
{
  "performance_score": 94.5,
  "bottlenecks": [
    {
      "component": "inference_engine",
      "severity": "medium",
      "description": "GPU utilization reaching 85% during peak hours",
      "recommendation": "Consider adding more GPU resources"
    }
  ],
  "trends": {
    "response_time": {
      "direction": "improving",
      "change_percentage": -12.4,
      "period": "7d"
    },
    "throughput": {
      "direction": "increasing", 
      "change_percentage": 23.7,
      "period": "7d"
    }
  },
  "capacity_analysis": {
    "current_utilization": 0.67,
    "projected_capacity_limit": "2024-03-15T00:00:00Z",
    "scaling_recommendations": [
      "Add 2 additional inference nodes",
      "Increase storage capacity by 50%"
    ]
  }
}
```

### POST /monitoring/alerts
Create custom monitoring alerts.

**Request Body:**
```json
{
  "name": "High Error Rate Alert",
  "description": "Alert when error rate exceeds 1%",
  "conditions": [
    {
      "metric": "error_rate",
      "operator": "greater_than",
      "threshold": 0.01,
      "timeframe": "5m"
    }
  ],
  "notifications": [
    {
      "type": "email",
      "recipients": ["admin@company.com", "ops@company.com"]
    },
    {
      "type": "webhook",
      "url": "https://your-app.com/alerts",
      "headers": {"Authorization": "Bearer your-token"}
    },
    {
      "type": "slack",
      "channel": "#alerts",
      "webhook_url": "https://hooks.slack.com/..."
    }
  ],
  "severity": "high",
  "enabled": true
}
```

**Response:**
```json
{
  "alert_id": "alert_abc123",
  "name": "High Error Rate Alert",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "last_triggered": null,
  "trigger_count": 0
}
```

### GET /monitoring/alerts
List all monitoring alerts.

**Response:**
```json
{
  "alerts": [
    {
      "alert_id": "alert_abc123",
      "name": "High Error Rate Alert",
      "status": "active",
      "last_triggered": "2024-01-14T15:23:00Z",
      "trigger_count": 3,
      "severity": "high"
    }
  ],
  "total_alerts": 12,
  "active_alerts": 8,
  "triggered_last_24h": 2
}
```

## 📈 Real-time Monitoring

### Live Metrics Stream

```python
# Subscribe to real-time metrics
stream = client.monitoring.metrics_stream(
    metrics=["api_requests", "response_time", "error_rate"],
    interval_seconds=10
)

@stream.on_data
def handle_metrics(data):
    print(f"API Requests: {data.api_requests}/min")
    print(f"Response Time: {data.response_time}ms")
    print(f"Error Rate: {data.error_rate}%")

stream.start()
```

### Dashboard Integration

```python
# Get dashboard-ready data
dashboard_data = client.monitoring.dashboard_data(
    dashboard_id="main_dashboard",
    refresh_interval=30
)

# Custom dashboard widgets
widgets = client.monitoring.create_dashboard(
    name="Custom Operations Dashboard",
    widgets=[
        {
            "type": "line_chart",
            "title": "API Response Time",
            "metric": "response_time",
            "timeframe": "24h"
        },
        {
            "type": "gauge",
            "title": "System Health Score",
            "metric": "health_score",
            "min_value": 0,
            "max_value": 100
        },
        {
            "type": "counter",
            "title": "Total Requests Today",
            "metric": "requests_count",
            "timeframe": "1d"
        }
    ]
)
```

## 🔍 Application Performance Monitoring

### Custom Application Metrics

```python
# Track custom application metrics
client.monitoring.track_metric(
    name="user_conversion_rate",
    value=0.234,
    tags={
        "environment": "production",
        "feature": "signup_flow",
        "version": "v2.1.0"
    }
)

# Track business events
client.monitoring.track_event(
    event="model_inference_completed",
    properties={
        "model": "gpt-4",
        "tokens": 150,
        "latency_ms": 234,
        "user_id": "user_123",
        "success": True
    }
)
```

### Error Tracking

```python
# Report errors with context
client.monitoring.report_error(
    error_message="Model inference timeout",
    error_type="TimeoutError",
    stack_trace="...",
    context={
        "model": "gpt-4",
        "request_id": "req_123",
        "user_id": "user_456"
    },
    severity="error"
)

# Get error analytics
error_analysis = client.monitoring.error_analysis(
    timeframe="7d",
    group_by=["error_type", "component"]
)
```

### Performance Profiling

```python
# Profile application performance
with client.monitoring.profile("inference_pipeline"):
    result = run_inference_pipeline()

# Get profiling results
profiling_data = client.monitoring.get_profiling_data(
    operation="inference_pipeline",
    timeframe="1h"
)
```

## 📊 Business Intelligence

### Usage Analytics

```python
# Get comprehensive usage analytics
analytics = client.monitoring.usage_analytics(
    timeframe="30d",
    breakdown_by=["user_type", "model", "region"]
)

print(f"Total API calls: {analytics.total_api_calls}")
print(f"Unique users: {analytics.unique_users}")
print(f"Average session duration: {analytics.avg_session_duration}")
```

### Revenue Metrics

```python
# Track revenue and financial metrics
revenue_metrics = client.monitoring.revenue_metrics(
    timeframe="30d",
    currency="USD"
)

print(f"Monthly recurring revenue: ${revenue_metrics.mrr}")
print(f"Customer acquisition cost: ${revenue_metrics.cac}")
print(f"Lifetime value: ${revenue_metrics.ltv}")
```

### User Behavior Analysis

```python
# Analyze user behavior patterns
behavior = client.monitoring.user_behavior(
    timeframe="7d",
    cohort="new_users",
    metrics=["retention", "engagement", "feature_adoption"]
)
```

## 🔧 Infrastructure Monitoring

### Resource Utilization

```python
# Monitor infrastructure resources
resources = client.monitoring.resource_utilization(
    timeframe="24h",
    components=["compute", "storage", "network"]
)

print(f"CPU utilization: {resources.cpu.avg_utilization}%")
print(f"Memory utilization: {resources.memory.avg_utilization}%")
print(f"Storage utilization: {resources.storage.utilization}%")
```

### Network Monitoring

```python
# Monitor network performance
network = client.monitoring.network_metrics(
    timeframe="1h",
    include_peer_details=True
)

print(f"Network latency: {network.avg_latency}ms")
print(f"Bandwidth utilization: {network.bandwidth_utilization}%")
print(f"Packet loss rate: {network.packet_loss_rate}%")
```

### Database Performance

```python
# Monitor database performance
db_metrics = client.monitoring.database_metrics(
    databases=["primary", "analytics", "cache"],
    timeframe="24h"
)

for db_name, metrics in db_metrics.items():
    print(f"Database {db_name}:")
    print(f"  Query response time: {metrics.avg_query_time}ms")
    print(f"  Connection pool usage: {metrics.connection_pool_usage}%")
    print(f"  Slow queries: {metrics.slow_query_count}")
```

## 🎯 Custom Metrics and KPIs

### Define Custom KPIs

```python
# Create custom KPI definitions
kpi = client.monitoring.create_kpi(
    name="Model Accuracy Rate",
    description="Percentage of accurate model predictions",
    calculation={
        "numerator": "accurate_predictions",
        "denominator": "total_predictions",
        "format": "percentage"
    },
    target_value=0.95,
    trend_direction="higher_is_better"
)
```

### Business Metrics Dashboard

```python
# Get business-focused metrics
business_metrics = client.monitoring.business_dashboard(
    timeframe="30d",
    metrics=[
        "monthly_active_users",
        "revenue_per_user",
        "churn_rate",
        "feature_adoption_rate",
        "support_ticket_volume"
    ]
)
```

## 🚨 Alerting and Notifications

### Advanced Alert Rules

```python
# Create complex alerting rules
advanced_alert = client.monitoring.create_alert(
    name="Anomaly Detection Alert",
    conditions=[
        {
            "type": "anomaly_detection",
            "metric": "api_response_time",
            "sensitivity": "high",
            "minimum_deviation": 2.0
        },
        {
            "type": "compound",
            "logic": "AND",
            "sub_conditions": [
                {"metric": "error_rate", "operator": ">", "value": 0.05},
                {"metric": "cpu_utilization", "operator": ">", "value": 0.8}
            ]
        }
    ],
    escalation_policy={
        "levels": [
            {"delay_minutes": 0, "notify": ["oncall_primary"]},
            {"delay_minutes": 15, "notify": ["oncall_secondary"]},
            {"delay_minutes": 30, "notify": ["management"]}
        ]
    }
)
```

### Notification Channels

```python
# Configure notification channels
notification_config = client.monitoring.configure_notifications(
    channels=[
        {
            "type": "pagerduty",
            "integration_key": "your-pagerduty-key",
            "severity_mapping": {
                "critical": "error",
                "high": "warning",
                "medium": "info"
            }
        },
        {
            "type": "teams",
            "webhook_url": "https://outlook.office.com/webhook/...",
            "format": "adaptive_card"
        }
    ]
)
```

## 📋 Reporting and Exports

### Automated Reports

```python
# Schedule automated reports
report_schedule = client.monitoring.schedule_report(
    name="Weekly Performance Report",
    template="performance_summary",
    schedule="0 9 * * 1",  # Every Monday at 9 AM
    recipients=["team@company.com"],
    format="pdf",
    include_charts=True
)
```

### Data Export

```python
# Export monitoring data
export = client.monitoring.export_data(
    timeframe="30d",
    metrics=["all"],
    format="csv",
    compression="gzip",
    destination="s3://your-bucket/monitoring-exports/"
)
```

### Compliance Reporting

```python
# Generate compliance reports
compliance_report = client.monitoring.compliance_report(
    standard="SOC2",
    timeframe="quarter",
    include_evidence=True,
    attestation_required=True
)
```

## 🔍 Log Management

### Centralized Logging

```python
# Query centralized logs
logs = client.monitoring.query_logs(
    query="level:ERROR AND component:inference",
    timeframe="1h",
    limit=100,
    sort="timestamp desc"
)

# Set up log streaming
log_stream = client.monitoring.log_stream(
    filters=["level:ERROR", "level:WARN"],
    components=["api", "inference", "p2p"]
)
```

### Log Analytics

```python
# Analyze log patterns
log_analytics = client.monitoring.log_analytics(
    timeframe="24h",
    analysis_type="pattern_detection",
    group_by=["component", "error_type"]
)
```

## 🧪 Testing and Monitoring

### Synthetic Monitoring

```python
# Set up synthetic tests
synthetic_test = client.monitoring.create_synthetic_test(
    name="API Health Check",
    type="http",
    url="https://api.prsm.ai/v1/health",
    frequency_minutes=5,
    locations=["us-east", "us-west", "eu-central"],
    assertions=[
        {"type": "status_code", "value": 200},
        {"type": "response_time", "operator": "<", "value": 1000}
    ]
)
```

### Load Testing Integration

```python
# Monitor during load tests
load_test_monitoring = client.monitoring.monitor_load_test(
    test_id="load_test_123",
    duration_minutes=30,
    target_metrics=["response_time", "error_rate", "throughput"]
)
```

## 📞 Support

- **Monitoring Issues**: monitoring@prsm.ai
- **Alert Configuration**: alerts@prsm.ai
- **Performance**: performance@prsm.ai
- **Data Export**: data-export@prsm.ai
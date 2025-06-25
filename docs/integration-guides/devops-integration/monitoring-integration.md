# Monitoring Integration Guide

Integrate comprehensive monitoring and observability into your PRSM deployment with Prometheus, Grafana, and modern observability tools.

## 🎯 Overview

This guide covers setting up production-grade monitoring for PRSM including metrics collection, alerting, dashboards, and distributed tracing for optimal system observability.

## 📋 Prerequisites

- PRSM instance running
- Docker and Docker Compose
- Basic knowledge of monitoring and observability concepts
- Network access for monitoring tools

## 🚀 Prometheus Integration

### 1. Prometheus Setup

```yaml
# monitoring/docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prsm-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=15d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: prsm-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:latest
    container_name: prsm-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:

networks:
  monitoring:
    driver: bridge
```

### 2. Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # PRSM API metrics
  - job_name: 'prsm-api'
    static_configs:
      - targets: ['prsm-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s

  # PRSM Worker metrics
  - job_name: 'prsm-workers'
    static_configs:
      - targets: ['prsm-worker-1:8001', 'prsm-worker-2:8002']
    metrics_path: '/metrics'
    scrape_interval: 30s

  # Database metrics
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  # System metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # Nginx metrics
  - job_name: 'nginx-exporter'
    static_configs:
      - targets: ['nginx-exporter:9113']

  # Custom application metrics
  - job_name: 'prsm-custom-metrics'
    static_configs:
      - targets: ['prsm-api:8000']
    metrics_path: '/custom-metrics'
    scrape_interval: 60s
```

### 3. PRSM Metrics Integration

```python
# prsm/monitoring/metrics.py
import time
import logging
from typing import Dict, Any, Optional
from functools import wraps
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST, start_http_server
)
import asyncio
import psutil

logger = logging.getLogger(__name__)

# Create custom registry for PRSM metrics
PRSM_REGISTRY = CollectorRegistry()

# Core API metrics
prsm_requests_total = Counter(
    'prsm_requests_total',
    'Total number of PRSM API requests',
    ['method', 'endpoint', 'status_code'],
    registry=PRSM_REGISTRY
)

prsm_request_duration_seconds = Histogram(
    'prsm_request_duration_seconds',
    'PRSM API request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float('inf')),
    registry=PRSM_REGISTRY
)

prsm_query_processing_seconds = Histogram(
    'prsm_query_processing_seconds',
    'Time spent processing PRSM queries',
    ['query_type', 'model'],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0, float('inf')),
    registry=PRSM_REGISTRY
)

prsm_active_connections = Gauge(
    'prsm_active_connections',
    'Number of active WebSocket connections',
    registry=PRSM_REGISTRY
)

prsm_cache_hits_total = Counter(
    'prsm_cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=PRSM_REGISTRY
)

prsm_cache_misses_total = Counter(
    'prsm_cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=PRSM_REGISTRY
)

prsm_errors_total = Counter(
    'prsm_errors_total',
    'Total number of errors',
    ['error_type', 'component'],
    registry=PRSM_REGISTRY
)

prsm_token_usage_total = Counter(
    'prsm_token_usage_total',
    'Total tokens consumed',
    ['model', 'user_id', 'token_type'],
    registry=PRSM_REGISTRY
)

prsm_memory_usage_bytes = Gauge(
    'prsm_memory_usage_bytes',
    'Memory usage in bytes',
    ['component'],
    registry=PRSM_REGISTRY
)

prsm_cpu_usage_percent = Gauge(
    'prsm_cpu_usage_percent',
    'CPU usage percentage',
    ['component'],
    registry=PRSM_REGISTRY
)

prsm_queue_size = Gauge(
    'prsm_queue_size',
    'Number of items in processing queue',
    ['queue_type'],
    registry=PRSM_REGISTRY
)

class MetricsCollector:
    """Centralized metrics collection for PRSM."""
    
    def __init__(self):
        self.start_time = time.time()
        self.active_requests = 0
        self.system_metrics_enabled = True
        
    def track_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Track API request metrics."""
        prsm_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        prsm_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def track_query_processing(self, query_type: str, model: str, duration: float):
        """Track query processing metrics."""
        prsm_query_processing_seconds.labels(
            query_type=query_type,
            model=model
        ).observe(duration)
    
    def track_cache_operation(self, cache_type: str, hit: bool):
        """Track cache hit/miss metrics."""
        if hit:
            prsm_cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            prsm_cache_misses_total.labels(cache_type=cache_type).inc()
    
    def track_error(self, error_type: str, component: str):
        """Track error metrics."""
        prsm_errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def track_token_usage(self, model: str, user_id: str, token_type: str, count: int):
        """Track token usage metrics."""
        prsm_token_usage_total.labels(
            model=model,
            user_id=user_id,
            token_type=token_type
        ).inc(count)
    
    def update_active_connections(self, count: int):
        """Update active connections gauge."""
        prsm_active_connections.set(count)
    
    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size metrics."""
        prsm_queue_size.labels(queue_type=queue_type).set(size)
    
    async def collect_system_metrics(self):
        """Collect system-level metrics."""
        if not self.system_metrics_enabled:
            return
        
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            prsm_memory_usage_bytes.labels(component='system').set(memory_info.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            prsm_cpu_usage_percent.labels(component='system').set(cpu_percent)
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss
            process_cpu = process.cpu_percent()
            
            prsm_memory_usage_bytes.labels(component='prsm').set(process_memory)
            prsm_cpu_usage_percent.labels(component='prsm').set(process_cpu)
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    async def start_metrics_collection(self, interval: int = 30):
        """Start periodic metrics collection."""
        while True:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(interval)

# Global metrics collector
metrics_collector = MetricsCollector()

def track_requests(func):
    """Decorator to track request metrics."""
    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        start_time = time.time()
        status_code = 200
        
        try:
            response = await func(request, *args, **kwargs)
            if hasattr(response, 'status_code'):
                status_code = response.status_code
            return response
        except Exception as e:
            status_code = 500
            metrics_collector.track_error(
                error_type=type(e).__name__,
                component='api'
            )
            raise
        finally:
            duration = time.time() - start_time
            metrics_collector.track_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                duration=duration
            )
    
    return wrapper

def track_query_processing(query_type: str, model: str):
    """Decorator to track query processing metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics_collector.track_query_processing(query_type, model, duration)
        
        return wrapper
    return decorator

# FastAPI integration
from fastapi import FastAPI, Response

def setup_metrics_endpoint(app: FastAPI):
    """Setup metrics endpoint for Prometheus scraping."""
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        return Response(
            generate_latest(PRSM_REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "uptime": time.time() - metrics_collector.start_time,
            "timestamp": time.time()
        }

# Custom metrics endpoint
async def get_custom_metrics() -> Dict[str, Any]:
    """Get custom application metrics."""
    return {
        "active_requests": metrics_collector.active_requests,
        "uptime_seconds": time.time() - metrics_collector.start_time,
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_usage_percent": psutil.Process().cpu_percent(),
    }
```

### 4. Alert Rules

```yaml
# monitoring/prometheus/rules/prsm-alerts.yml
groups:
  - name: prsm-api
    rules:
      - alert: PRSMHighErrorRate
        expr: rate(prsm_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in PRSM API"
          description: "PRSM API error rate is {{ $value }} errors per second"

      - alert: PRSMHighLatency
        expr: histogram_quantile(0.95, rate(prsm_request_duration_seconds_bucket[5m])) > 5
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High latency in PRSM API"
          description: "95th percentile latency is {{ $value }} seconds"

      - alert: PRSMQueryProcessingTimeout
        expr: histogram_quantile(0.95, rate(prsm_query_processing_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "PRSM query processing is too slow"
          description: "95th percentile query processing time is {{ $value }} seconds"

      - alert: PRSMHighMemoryUsage
        expr: prsm_memory_usage_bytes{component="prsm"} / 1024 / 1024 / 1024 > 4
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage in PRSM"
          description: "PRSM is using {{ $value }}GB of memory"

      - alert: PRSMHighCPUUsage
        expr: prsm_cpu_usage_percent{component="prsm"} > 80
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage in PRSM"
          description: "PRSM CPU usage is {{ $value }}%"

      - alert: PRSMCacheMissRate
        expr: rate(prsm_cache_misses_total[5m]) / (rate(prsm_cache_hits_total[5m]) + rate(prsm_cache_misses_total[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High cache miss rate"
          description: "Cache miss rate is {{ $value | humanizePercentage }}"

  - name: prsm-infrastructure
    rules:
      - alert: PRSMDatabaseDown
        expr: up{job="postgres-exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PRSM database is down"
          description: "PostgreSQL database is not responding"

      - alert: PRSMRedisDown
        expr: up{job="redis-exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PRSM Redis is down"
          description: "Redis cache is not responding"

      - alert: PRSMDiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 20
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space on PRSM server"
          description: "Disk space is {{ $value }}% full"
```

## 📊 Grafana Dashboards

### 1. PRSM Overview Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "PRSM Overview",
    "tags": ["prsm", "overview"],
    "timezone": "browser",
    "refresh": "30s",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(prsm_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(prsm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(prsm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(prsm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "99th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(prsm_errors_total[5m])",
            "legendFormat": "{{error_type}} in {{component}}"
          }
        ],
        "yAxes": [
          {
            "label": "Errors/sec"
          }
        ]
      },
      {
        "id": 4,
        "title": "Active Connections",
        "type": "singlestat",
        "targets": [
          {
            "expr": "prsm_active_connections"
          }
        ]
      },
      {
        "id": 5,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "prsm_memory_usage_bytes{component=\"prsm\"} / 1024 / 1024",
            "legendFormat": "PRSM Memory (MB)"
          }
        ],
        "yAxes": [
          {
            "label": "MB"
          }
        ]
      },
      {
        "id": 6,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "prsm_cpu_usage_percent{component=\"prsm\"}",
            "legendFormat": "PRSM CPU %"
          }
        ],
        "yAxes": [
          {
            "label": "Percent",
            "min": 0,
            "max": 100
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    }
  }
}
```

### 2. Grafana Provisioning

```yaml
# monitoring/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

```yaml
# monitoring/grafana/provisioning/dashboards/prsm.yml
apiVersion: 1

providers:
  - name: 'PRSM'
    orgId: 1
    folder: 'PRSM'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

## 🔔 Alertmanager Configuration

### Alert Routing and Notifications

```yaml
# monitoring/alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@prsm.ai'
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://prsm-api:8000/webhooks/alerts'
        send_resolved: true

  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#prsm-critical'
        title: 'PRSM Critical Alert'
        text: >-
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    email_configs:
      - to: 'oncall@prsm.ai'
        subject: 'PRSM Critical Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Labels: {{ .Labels }}
          {{ end }}

  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#prsm-alerts'
        title: 'PRSM Warning Alert'
        text: >-
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

## 🔍 Distributed Tracing

### Jaeger Integration

```python
# prsm/tracing/jaeger_integration.py
import logging
import time
from typing import Dict, Any, Optional
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

logger = logging.getLogger(__name__)

class PRSMTracing:
    """Distributed tracing setup for PRSM."""
    
    def __init__(
        self,
        service_name: str = "prsm-api",
        jaeger_endpoint: str = "http://localhost:14268/api/traces",
        environment: str = "production"
    ):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.environment = environment
        self.tracer = None
    
    def initialize_tracing(self):
        """Initialize distributed tracing."""
        try:
            # Set up tracer provider
            trace.set_tracer_provider(TracerProvider())
            
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
                collector_endpoint=self.jaeger_endpoint,
            )
            
            # Set up span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(
                __name__,
                version="1.0.0",
            )
            
            logger.info("Distributed tracing initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
    
    def instrument_app(self, app):
        """Instrument FastAPI application."""
        try:
            # Instrument FastAPI
            FastAPIInstrumentor.instrument_app(app)
            
            # Instrument HTTP requests
            RequestsInstrumentor().instrument()
            
            # Instrument database connections
            SQLAlchemyInstrumentor().instrument()
            
            # Instrument Redis
            RedisInstrumentor().instrument()
            
            logger.info("Application instrumented for tracing")
            
        except Exception as e:
            logger.error(f"Failed to instrument application: {e}")
    
    def create_span(
        self,
        name: str,
        attributes: Dict[str, Any] = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL
    ):
        """Create a new span."""
        if not self.tracer:
            return trace.get_tracer(__name__).start_span(name)
        
        span = self.tracer.start_span(name, kind=kind)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        return span
    
    def trace_query_processing(self, query_id: str, user_id: str, model: str):
        """Create span for query processing."""
        return self.create_span(
            "prsm.query.process",
            attributes={
                "query.id": query_id,
                "user.id": user_id,
                "model.name": model,
                "service.name": self.service_name,
                "environment": self.environment
            }
        )
    
    def trace_database_operation(self, operation: str, table: str):
        """Create span for database operations."""
        return self.create_span(
            f"db.{operation}",
            attributes={
                "db.operation": operation,
                "db.table": table,
                "db.system": "postgresql"
            },
            kind=trace.SpanKind.CLIENT
        )
    
    def trace_cache_operation(self, operation: str, key: str):
        """Create span for cache operations."""
        return self.create_span(
            f"cache.{operation}",
            attributes={
                "cache.operation": operation,
                "cache.key": key,
                "cache.system": "redis"
            },
            kind=trace.SpanKind.CLIENT
        )
    
    def trace_external_api_call(self, service: str, endpoint: str):
        """Create span for external API calls."""
        return self.create_span(
            f"http.{service}",
            attributes={
                "http.service": service,
                "http.endpoint": endpoint,
                "http.method": "POST"
            },
            kind=trace.SpanKind.CLIENT
        )

# Global tracing instance
prsm_tracing = PRSMTracing()

def trace_function(span_name: str, attributes: Dict[str, Any] = None):
    """Decorator to trace function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with prsm_tracing.create_span(span_name, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        return wrapper
    return decorator

async def trace_async_function(span_name: str, attributes: Dict[str, Any] = None):
    """Decorator to trace async function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with prsm_tracing.create_span(span_name, attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        return wrapper
    return decorator
```

### Docker Compose for Tracing Stack

```yaml
# monitoring/docker-compose.tracing.yml
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Jaeger HTTP collector
      - "6831:6831/udp"  # Jaeger agent
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411
    networks:
      - monitoring

  zipkin:
    image: openzipkin/zipkin:latest
    ports:
      - "9411:9411"
    networks:
      - monitoring

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yml"]
    volumes:
      - ./otel/otel-collector-config.yml:/etc/otel-collector-config.yml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
      - "8888:8888"   # Prometheus metrics
      - "8889:8889"   # Prometheus exporter metrics
    networks:
      - monitoring

networks:
  monitoring:
    external: true
```

## 📈 Application Performance Monitoring

### Custom APM Integration

```python
# prsm/monitoring/apm.py
import asyncio
import logging
import time
import traceback
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from prsm.core.cache import redis_cache

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: float
    metric_name: str
    value: float
    labels: Dict[str, str]
    unit: str = "ms"

@dataclass
class ErrorEvent:
    """Error event data structure."""
    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    component: str = "unknown"

class APMCollector:
    """Application Performance Monitoring collector."""
    
    def __init__(self):
        self.metrics_buffer: List[PerformanceMetric] = []
        self.errors_buffer: List[ErrorEvent] = []
        self.buffer_size = 1000
        self.flush_interval = 60  # seconds
        self.running = False
    
    async def start(self):
        """Start APM collection."""
        self.running = True
        asyncio.create_task(self._flush_loop())
        logger.info("APM collector started")
    
    async def stop(self):
        """Stop APM collection."""
        self.running = False
        await self._flush_buffers()
        logger.info("APM collector stopped")
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        unit: str = "ms"
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_name=name,
            value=value,
            labels=labels or {},
            unit=unit
        )
        
        self.metrics_buffer.append(metric)
        
        if len(self.metrics_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_metrics())
    
    def record_error(
        self,
        error: Exception,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        component: str = "unknown"
    ):
        """Record an error event."""
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            user_id=user_id,
            request_id=request_id,
            component=component
        )
        
        self.errors_buffer.append(error_event)
        
        if len(self.errors_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_errors())
    
    async def _flush_loop(self):
        """Periodic flush of buffers."""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffers()
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
    
    async def _flush_buffers(self):
        """Flush all buffers."""
        await self._flush_metrics()
        await self._flush_errors()
    
    async def _flush_metrics(self):
        """Flush metrics buffer."""
        if not self.metrics_buffer:
            return
        
        try:
            # Store metrics in Redis for analysis
            metrics_data = [asdict(metric) for metric in self.metrics_buffer]
            
            # Store in time-series format
            current_minute = int(time.time() // 60) * 60
            key = f"apm:metrics:{current_minute}"
            
            await redis_cache.set(
                key,
                json.dumps(metrics_data),
                ttl=86400  # 24 hours
            )
            
            logger.debug(f"Flushed {len(self.metrics_buffer)} metrics")
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
    
    async def _flush_errors(self):
        """Flush errors buffer."""
        if not self.errors_buffer:
            return
        
        try:
            # Store errors in Redis for analysis
            errors_data = [asdict(error) for error in self.errors_buffer]
            
            # Store with timestamp key
            timestamp = int(time.time())
            key = f"apm:errors:{timestamp}"
            
            await redis_cache.set(
                key,
                json.dumps(errors_data),
                ttl=604800  # 7 days
            )
            
            logger.debug(f"Flushed {len(self.errors_buffer)} errors")
            self.errors_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush errors: {e}")
    
    async def get_metrics_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metrics summary for time range."""
        try:
            start_ts = int(start_time.timestamp() // 60) * 60
            end_ts = int(end_time.timestamp() // 60) * 60
            
            all_metrics = []
            
            # Fetch metrics for time range
            for ts in range(start_ts, end_ts + 60, 60):
                key = f"apm:metrics:{ts}"
                data = await redis_cache.get(key)
                
                if data:
                    metrics = json.loads(data)
                    if metric_name:
                        metrics = [m for m in metrics if m["metric_name"] == metric_name]
                    all_metrics.extend(metrics)
            
            if not all_metrics:
                return {"count": 0, "avg": 0, "min": 0, "max": 0}
            
            values = [m["value"] for m in all_metrics]
            
            return {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p50": sorted(values)[len(values) // 2],
                "p95": sorted(values)[int(len(values) * 0.95)],
                "p99": sorted(values)[int(len(values) * 0.99)]
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    async def get_error_summary(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get error summary for time range."""
        try:
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            pattern = "apm:errors:*"
            error_keys = await redis_cache._async_redis.keys(pattern)
            
            all_errors = []
            error_counts = {}
            
            for key in error_keys:
                # Extract timestamp from key
                try:
                    key_ts = int(key.split(":")[-1])
                    if start_ts <= key_ts <= end_ts:
                        data = await redis_cache.get(key)
                        if data:
                            errors = json.loads(data)
                            all_errors.extend(errors)
                            
                            for error in errors:
                                error_type = error["error_type"]
                                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                except ValueError:
                    continue
            
            return {
                "total_errors": len(all_errors),
                "error_types": error_counts,
                "error_rate": len(all_errors) / ((end_ts - start_ts) / 3600),  # errors per hour
                "top_errors": sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
        except Exception as e:
            logger.error(f"Failed to get error summary: {e}")
            return {}

# Global APM collector
apm_collector = APMCollector()

def monitor_performance(metric_name: str, labels: Dict[str, str] = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000  # ms
                apm_collector.record_metric(metric_name, duration, labels)
                return result
            except Exception as e:
                apm_collector.record_error(e, component=func.__module__)
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000  # ms
                apm_collector.record_metric(metric_name, duration, labels)
                return result
            except Exception as e:
                apm_collector.record_error(e, component=func.__module__)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

## 📊 Log Management

### Structured Logging with ELK Stack

```python
# prsm/logging/structured_logging.py
import logging
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import traceback
from pythonjsonlogger import jsonlogger

class PRSMJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for PRSM logs."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        super(PRSMJSONFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add service information
        log_record['service'] = 'prsm-api'
        log_record['version'] = '1.0.0'
        
        # Add trace information if available
        if hasattr(record, 'trace_id'):
            log_record['trace_id'] = record.trace_id
        if hasattr(record, 'span_id'):
            log_record['span_id'] = record.span_id
        
        # Add user context if available
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        
        # Add component information
        log_record['component'] = record.name
        
        # Add exception information
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

class PRSMLogger:
    """Centralized logging for PRSM."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup structured logging."""
        if not self.logger.handlers:
            # Console handler with JSON formatting
            console_handler = logging.StreamHandler()
            json_formatter = PRSMJSONFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
            console_handler.setFormatter(json_formatter)
            self.logger.addHandler(console_handler)
            
            # Set log level
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        extra = self._build_extra(**kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        extra = self._build_extra(**kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with context."""
        extra = self._build_extra(**kwargs)
        if error:
            self.logger.error(message, exc_info=error, extra=extra)
        else:
            self.logger.error(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        extra = self._build_extra(**kwargs)
        self.logger.debug(message, extra=extra)
    
    def _build_extra(self, **kwargs) -> Dict[str, Any]:
        """Build extra context for log record."""
        extra = {}
        
        # Add provided context
        for key, value in kwargs.items():
            extra[key] = value
        
        return extra

def get_logger(name: str) -> PRSMLogger:
    """Get PRSM logger instance."""
    return PRSMLogger(name)
```

### Logstash Configuration

```ruby
# monitoring/logstash/pipeline/prsm-logs.conf
input {
  beats {
    port => 5044
  }
  
  http {
    port => 8080
  }
}

filter {
  if [fields][service] == "prsm" {
    json {
      source => "message"
    }
    
    # Parse timestamp
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    # Add geo location for IP addresses
    if [client_ip] {
      geoip {
        source => "client_ip"
        target => "geoip"
      }
    }
    
    # Categorize log levels
    if [level] == "ERROR" {
      mutate {
        add_tag => ["error"]
      }
    }
    
    if [level] == "WARNING" {
      mutate {
        add_tag => ["warning"]
      }
    }
    
    # Extract user information
    if [user_id] {
      mutate {
        add_field => { "user_category" => "authenticated" }
      }
    } else {
      mutate {
        add_field => { "user_category" => "anonymous" }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "prsm-logs-%{+YYYY.MM.dd}"
  }
  
  # Send critical errors to alerting
  if "error" in [tags] and [level] == "ERROR" {
    http {
      url => "http://alertmanager:9093/api/v1/alerts"
      http_method => "post"
      format => "json"
      mapping => {
        "alerts" => [{
          "labels" => {
            "alertname" => "PRSMApplicationError"
            "severity" => "warning"
            "service" => "%{service}"
            "component" => "%{component}"
          }
          "annotations" => {
            "summary" => "Application error in %{component}"
            "description" => "%{message}"
          }
        }]
      }
    }
  }
  
  stdout { 
    codec => rubydebug 
  }
}
```

## 📋 Deployment and Production Setup

### Complete Monitoring Stack

```bash
#!/bin/bash
# scripts/deploy-monitoring.sh

set -e

echo "Deploying PRSM Monitoring Stack..."

# Create monitoring network
docker network create monitoring 2>/dev/null || true

# Deploy Prometheus stack
echo "Starting Prometheus..."
docker-compose -f monitoring/docker-compose.yml up -d prometheus

# Wait for Prometheus to be ready
echo "Waiting for Prometheus to be ready..."
timeout 60 bash -c 'until curl -f http://localhost:9090/-/ready; do sleep 2; done'

# Deploy Grafana
echo "Starting Grafana..."
docker-compose -f monitoring/docker-compose.yml up -d grafana

# Wait for Grafana to be ready
echo "Waiting for Grafana to be ready..."
timeout 60 bash -c 'until curl -f http://localhost:3000/api/health; do sleep 2; done'

# Deploy Alertmanager
echo "Starting Alertmanager..."
docker-compose -f monitoring/docker-compose.yml up -d alertmanager

# Deploy Jaeger for tracing
echo "Starting Jaeger..."
docker-compose -f monitoring/docker-compose.tracing.yml up -d jaeger

# Deploy exporters
echo "Starting exporters..."
docker-compose -f monitoring/docker-compose.yml up -d \
  node-exporter \
  cadvisor \
  postgres-exporter \
  redis-exporter

echo "Monitoring stack deployed successfully!"
echo ""
echo "Access URLs:"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo "  Alertmanager: http://localhost:9093"
echo "  Jaeger:     http://localhost:16686"
echo ""
echo "Import Grafana dashboards from: monitoring/grafana/dashboards/"
```

### Health Check Script

```python
# scripts/health_check.py
import asyncio
import aiohttp
import sys
from typing import Dict, Any, List

class HealthChecker:
    """Health check for PRSM monitoring stack."""
    
    def __init__(self):
        self.services = {
            'prsm-api': 'http://localhost:8000/health',
            'prometheus': 'http://localhost:9090/-/ready',
            'grafana': 'http://localhost:3000/api/health',
            'alertmanager': 'http://localhost:9093/-/ready',
            'jaeger': 'http://localhost:16686/api/services'
        }
        
    async def check_service(self, name: str, url: str) -> Dict[str, Any]:
        """Check health of a single service."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return {
                            'service': name,
                            'status': 'healthy',
                            'response_time': response.headers.get('X-Response-Time', 'N/A')
                        }
                    else:
                        return {
                            'service': name,
                            'status': 'unhealthy',
                            'error': f'HTTP {response.status}'
                        }
        except Exception as e:
            return {
                'service': name,
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_all_services(self) -> List[Dict[str, Any]]:
        """Check health of all services."""
        tasks = [
            self.check_service(name, url)
            for name, url in self.services.items()
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def print_results(self, results: List[Dict[str, Any]]):
        """Print health check results."""
        print("PRSM Monitoring Stack Health Check")
        print("=" * 40)
        
        healthy_count = 0
        
        for result in results:
            status = result['status']
            service = result['service']
            
            if status == 'healthy':
                print(f"✅ {service}: {status}")
                healthy_count += 1
            else:
                print(f"❌ {service}: {status} - {result.get('error', '')}")
        
        print("=" * 40)
        print(f"Healthy services: {healthy_count}/{len(results)}")
        
        if healthy_count == len(results):
            print("🎉 All services are healthy!")
            return 0
        else:
            print("⚠️  Some services are unhealthy!")
            return 1

async def main():
    checker = HealthChecker()
    results = await checker.check_all_services()
    exit_code = checker.print_results(results)
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())
```

---

**Need help with monitoring integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).
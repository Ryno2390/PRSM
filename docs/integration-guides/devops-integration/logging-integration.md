# Logging Integration Guide

Integrate centralized logging and log management into your PRSM deployment with ELK Stack, Fluentd, and modern log aggregation tools.

## 🎯 Overview

This guide covers setting up production-grade logging for PRSM including structured logging, log aggregation, search capabilities, and log-based alerting.

## 📋 Prerequisites

- PRSM instance running
- Docker and Docker Compose
- Basic knowledge of logging and log management
- Sufficient storage for log retention

## 🚀 ELK Stack Integration

### 1. Elasticsearch, Logstash, Kibana Setup

```yaml
# logging/docker-compose.elk.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    container_name: prsm-elasticsearch
    environment:
      - node.name=elasticsearch
      - cluster.name=prsm-logs
      - discovery.type=single-node
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - xpack.security.enabled=false
      - xpack.security.enrollment.enabled=false
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
      - ./elasticsearch/config:/usr/share/elasticsearch/config
    ports:
      - "9200:9200"
      - "9300:9300"
    networks:
      - logging
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    container_name: prsm-logstash
    volumes:
      - ./logstash/config/logstash.yml:/usr/share/logstash/config/logstash.yml:ro
      - ./logstash/pipeline:/usr/share/logstash/pipeline:ro
    ports:
      - "5044:5044"  # Beats input
      - "5000:5000"  # TCP input
      - "9600:9600"  # Logstash monitoring
    environment:
      - "LS_JAVA_OPTS=-Xmx1g -Xms1g"
    networks:
      - logging
    depends_on:
      elasticsearch:
        condition: service_healthy

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    container_name: prsm-kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
      - ELASTICSEARCH_USERNAME=kibana_system
      - ELASTICSEARCH_PASSWORD=kibana_password
    volumes:
      - ./kibana/config:/usr/share/kibana/config
    networks:
      - logging
    depends_on:
      elasticsearch:
        condition: service_healthy

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.5.0
    container_name: prsm-filebeat
    user: root
    volumes:
      - ./filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - filebeat_data:/usr/share/filebeat/data
    networks:
      - logging
    depends_on:
      - logstash

volumes:
  elasticsearch_data:
  filebeat_data:

networks:
  logging:
    driver: bridge
```

### 2. Logstash Configuration

```ruby
# logging/logstash/pipeline/prsm.conf
input {
  beats {
    port => 5044
  }
  
  tcp {
    port => 5000
    codec => json_lines
  }
  
  http {
    port => 8080
    codec => json
  }
}

filter {
  # Parse timestamp
  if [timestamp] {
    date {
      match => [ "timestamp", "ISO8601" ]
    }
  }
  
  # Handle PRSM application logs
  if [fields][service] == "prsm" or [service] == "prsm" {
    # Parse JSON logs
    if [message] =~ /^\{.*\}$/ {
      json {
        source => "message"
      }
    }
    
    # Extract request information
    if [request_id] {
      mutate {
        add_field => { "[@metadata][request_tracking]" => true }
      }
    }
    
    # Categorize log levels
    if [level] {
      mutate {
        lowercase => [ "level" ]
      }
      
      if [level] == "error" {
        mutate {
          add_tag => ["error", "alert"]
        }
      } else if [level] == "warning" or [level] == "warn" {
        mutate {
          add_tag => ["warning"]
        }
      } else if [level] == "info" {
        mutate {
          add_tag => ["info"]
        }
      } else if [level] == "debug" {
        mutate {
          add_tag => ["debug"]
        }
      }
    }
    
    # Parse user agent
    if [user_agent] {
      useragent {
        source => "user_agent"
        target => "ua"
      }
    }
    
    # GeoIP lookup for client IPs
    if [client_ip] and [client_ip] !~ /^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)/ {
      geoip {
        source => "client_ip"
        target => "geoip"
      }
    }
    
    # Parse query information
    if [query_type] {
      mutate {
        add_field => { "[@metadata][query_log]" => true }
      }
    }
    
    # Handle exception information
    if [exception] {
      mutate {
        add_tag => ["exception"]
      }
      
      # Extract exception details
      if [exception][type] {
        mutate {
          add_field => { "exception_type" => "%{[exception][type]}" }
        }
      }
      
      if [exception][message] {
        mutate {
          add_field => { "exception_message" => "%{[exception][message]}" }
        }
      }
    }
    
    # Performance metrics extraction
    if [duration] {
      mutate {
        convert => { "duration" => "float" }
      }
      
      if [duration] > 5000 {
        mutate {
          add_tag => ["slow_request"]
        }
      }
    }
    
    # User activity tracking
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
  
  # Handle Docker container logs
  if [container] {
    # Extract container information
    if [container][name] {
      mutate {
        add_field => { "container_name" => "%{[container][name]}" }
      }
    }
    
    if [container][image] {
      mutate {
        add_field => { "container_image" => "%{[container][image]}" }
      }
    }
  }
  
  # Handle Nginx access logs
  if [fields][logtype] == "nginx" {
    grok {
      match => { "message" => "%{NGINXACCESS}" }
    }
    
    if [response] {
      mutate {
        convert => { "response" => "integer" }
      }
      
      if [response] >= 400 {
        mutate {
          add_tag => ["http_error"]
        }
      }
      
      if [response] >= 500 {
        mutate {
          add_tag => ["server_error"]
        }
      }
    }
  }
  
  # Clean up unnecessary fields
  mutate {
    remove_field => [ "agent", "ecs", "host", "input" ]
  }
}

output {
  # Main Elasticsearch output
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "prsm-%{+YYYY.MM.dd}"
    template_name => "prsm"
    template => "/usr/share/logstash/templates/prsm-template.json"
    template_overwrite => true
  }
  
  # Separate index for errors
  if "error" in [tags] {
    elasticsearch {
      hosts => ["elasticsearch:9200"]
      index => "prsm-errors-%{+YYYY.MM.dd}"
    }
  }
  
  # Separate index for performance data
  if [duration] {
    elasticsearch {
      hosts => ["elasticsearch:9200"]
      index => "prsm-performance-%{+YYYY.MM.dd}"
    }
  }
  
  # Send critical errors to external alerting
  if "error" in [tags] and [level] == "error" {
    http {
      url => "http://alertmanager:9093/api/v1/alerts"
      http_method => "post"
      format => "json"
      mapping => {
        "alerts" => [{
          "labels" => {
            "alertname" => "PRSMLogError"
            "severity" => "warning"
            "service" => "%{service}"
            "component" => "%{component}"
            "instance" => "%{host}"
          }
          "annotations" => {
            "summary" => "Error in %{component}"
            "description" => "%{message}"
            "timestamp" => "%{@timestamp}"
          }
        }]
      }
    }
  }
  
  # Debug output (comment out in production)
  # stdout { 
  #   codec => rubydebug 
  # }
}
```

### 3. Filebeat Configuration

```yaml
# logging/filebeat/filebeat.yml
filebeat.inputs:
  # Docker container logs
  - type: container
    paths:
      - '/var/lib/docker/containers/*/*.log'
    processors:
      - add_docker_metadata:
          host: "unix:///var/run/docker.sock"
      - decode_json_fields:
          fields: ["message"]
          target: ""
          overwrite_keys: true

  # PRSM application logs
  - type: log
    enabled: true
    paths:
      - /app/logs/prsm*.log
    fields:
      service: prsm
      logtype: application
    fields_under_root: true
    multiline.pattern: '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
    multiline.negate: true
    multiline.match: after

  # Nginx access logs
  - type: log
    enabled: true
    paths:
      - /var/log/nginx/access.log
    fields:
      service: nginx
      logtype: nginx
    fields_under_root: true

  # System logs
  - type: log
    enabled: true
    paths:
      - /var/log/syslog
      - /var/log/auth.log
    fields:
      service: system
      logtype: system
    fields_under_root: true

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
  - add_cloud_metadata: ~
  - add_kubernetes_metadata: ~

output.logstash:
  hosts: ["logstash:5044"]

setup.kibana:
  host: "kibana:5601"

logging.level: info
logging.to_files: true
logging.files:
  path: /usr/share/filebeat/logs
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

## 🔧 PRSM Structured Logging

### Advanced Logging Implementation

```python
# prsm/logging/advanced_logger.py
import logging
import json
import time
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    component: str
    service: str = "prsm"
    version: str = "1.0.0"
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    method: Optional[str] = None
    endpoint: Optional[str] = None
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    query_type: Optional[str] = None
    model: Optional[str] = None
    token_count: Optional[int] = None
    exception: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class PRSMStructuredLogger:
    """Advanced structured logger for PRSM."""
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        output_format: str = "json",
        log_file: Optional[str] = None
    ):
        self.name = name
        self.level = level
        self.output_format = output_format
        self.log_file = log_file
        self.logger = self._setup_logger()
        
        # Performance tracking
        self._request_start_times: Dict[str, float] = {}
        
        # Log aggregation buffer
        self._log_buffer: List[LogEntry] = []
        self._buffer_size = 100
        self._flush_interval = 60
        
    def _setup_logger(self) -> logging.Logger:
        """Setup the underlying logger."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level.value))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        if self.output_format == "json":
            formatter = self._create_json_formatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _create_json_formatter(self):
        """Create JSON formatter for structured logging."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = LogEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    level=record.levelname,
                    message=record.getMessage(),
                    component=record.name,
                    request_id=request_id_var.get() or None,
                    user_id=user_id_var.get() or None,
                    trace_id=trace_id_var.get() or None,
                    metadata=getattr(record, 'metadata', None)
                )
                
                # Add exception information
                if record.exc_info:
                    log_entry.exception = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': traceback.format_exception(*record.exc_info)
                    }
                
                # Convert to JSON
                return json.dumps(asdict(log_entry), default=str)
        
        return JSONFormatter()
    
    def start_request(self, request_id: str = None, user_id: str = None):
        """Start tracking a request."""
        req_id = request_id or str(uuid.uuid4())
        request_id_var.set(req_id)
        
        if user_id:
            user_id_var.set(user_id)
        
        self._request_start_times[req_id] = time.time()
        
        self.info(
            "Request started",
            metadata={
                "event": "request_start",
                "request_id": req_id,
                "user_id": user_id
            }
        )
        
        return req_id
    
    def end_request(
        self,
        request_id: str = None,
        status_code: int = None,
        method: str = None,
        endpoint: str = None
    ):
        """End request tracking."""
        req_id = request_id or request_id_var.get()
        
        if req_id in self._request_start_times:
            duration = time.time() - self._request_start_times[req_id]
            del self._request_start_times[req_id]
        else:
            duration = None
        
        self.info(
            "Request completed",
            metadata={
                "event": "request_end",
                "request_id": req_id,
                "duration": duration,
                "status_code": status_code,
                "method": method,
                "endpoint": endpoint
            }
        )
    
    def log_query(
        self,
        query_type: str,
        model: str,
        duration: float,
        token_count: int = None,
        success: bool = True
    ):
        """Log query processing information."""
        self.info(
            f"Query processed: {query_type}",
            metadata={
                "event": "query_processed",
                "query_type": query_type,
                "model": model,
                "duration": duration,
                "token_count": token_count,
                "success": success
            }
        )
    
    def log_error(
        self,
        message: str,
        error: Exception = None,
        component: str = None,
        **metadata
    ):
        """Log error with full context."""
        extra_metadata = {
            "event": "error",
            "component": component or self.name,
            **metadata
        }
        
        if error:
            self.logger.error(
                message,
                exc_info=error,
                extra={"metadata": extra_metadata}
            )
        else:
            self.logger.error(
                message,
                extra={"metadata": extra_metadata}
            )
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        tags: Dict[str, str] = None
    ):
        """Log performance metrics."""
        self.info(
            f"Performance metric: {metric_name}",
            metadata={
                "event": "performance_metric",
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "tags": tags or {}
            }
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Log security events."""
        self.warning(
            f"Security event: {event_type}",
            metadata={
                "event": "security",
                "event_type": event_type,
                "severity": severity,
                "details": details
            }
        )
    
    def log_user_action(
        self,
        action: str,
        resource: str = None,
        details: Dict[str, Any] = None
    ):
        """Log user actions for audit trail."""
        self.info(
            f"User action: {action}",
            metadata={
                "event": "user_action",
                "action": action,
                "resource": resource,
                "details": details or {}
            }
        )
    
    def info(self, message: str, metadata: Dict[str, Any] = None):
        """Log info message."""
        self.logger.info(message, extra={"metadata": metadata})
    
    def warning(self, message: str, metadata: Dict[str, Any] = None):
        """Log warning message."""
        self.logger.warning(message, extra={"metadata": metadata})
    
    def error(self, message: str, error: Exception = None, metadata: Dict[str, Any] = None):
        """Log error message."""
        if error:
            self.logger.error(message, exc_info=error, extra={"metadata": metadata})
        else:
            self.logger.error(message, extra={"metadata": metadata})
    
    def debug(self, message: str, metadata: Dict[str, Any] = None):
        """Log debug message."""
        self.logger.debug(message, extra={"metadata": metadata})

# Global logger instances
app_logger = PRSMStructuredLogger("prsm.app")
query_logger = PRSMStructuredLogger("prsm.query")
auth_logger = PRSMStructuredLogger("prsm.auth")
api_logger = PRSMStructuredLogger("prsm.api")
db_logger = PRSMStructuredLogger("prsm.database")

def get_logger(name: str) -> PRSMStructuredLogger:
    """Get or create logger instance."""
    return PRSMStructuredLogger(name)

# Decorators for automatic logging
def log_function_call(logger: PRSMStructuredLogger = None):
    """Decorator to log function calls."""
    def decorator(func):
        func_logger = logger or get_logger(f"prsm.{func.__module__}")
        
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger.debug(
                f"Function called: {func.__name__}",
                metadata={
                    "event": "function_call",
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                func_logger.debug(
                    f"Function completed: {func.__name__}",
                    metadata={
                        "event": "function_complete",
                        "function": func.__name__,
                        "duration": duration,
                        "success": True
                    }
                )
                
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                
                func_logger.error(
                    f"Function failed: {func.__name__}",
                    error=e,
                    metadata={
                        "event": "function_error",
                        "function": func.__name__,
                        "duration": duration,
                        "success": False
                    }
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger.debug(
                f"Function called: {func.__name__}",
                metadata={
                    "event": "function_call",
                    "function": func.__name__,
                    "module": func.__module__
                }
            )
            
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                func_logger.debug(
                    f"Function completed: {func.__name__}",
                    metadata={
                        "event": "function_complete",
                        "function": func.__name__,
                        "duration": duration,
                        "success": True
                    }
                )
                
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                
                func_logger.error(
                    f"Function failed: {func.__name__}",
                    error=e,
                    metadata={
                        "event": "function_error",
                        "function": func.__name__,
                        "duration": duration,
                        "success": False
                    }
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def log_api_call(logger: PRSMStructuredLogger = api_logger):
    """Decorator to log API calls."""
    def decorator(func):
        async def wrapper(request, *args, **kwargs):
            # Start request tracking
            request_id = str(uuid.uuid4())
            request_id_var.set(request_id)
            
            # Extract request information
            method = request.method
            endpoint = str(request.url.path)
            user_agent = request.headers.get("user-agent", "")
            client_ip = request.client.host if request.client else ""
            
            start_time = time.time()
            
            logger.info(
                f"API request: {method} {endpoint}",
                metadata={
                    "event": "api_request",
                    "method": method,
                    "endpoint": endpoint,
                    "user_agent": user_agent,
                    "client_ip": client_ip,
                    "request_id": request_id
                }
            )
            
            try:
                response = await func(request, *args, **kwargs)
                duration = (time.time() - start_time) * 1000
                status_code = getattr(response, 'status_code', 200)
                
                logger.info(
                    f"API response: {method} {endpoint} - {status_code}",
                    metadata={
                        "event": "api_response",
                        "method": method,
                        "endpoint": endpoint,
                        "status_code": status_code,
                        "duration": duration,
                        "request_id": request_id
                    }
                )
                
                return response
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                
                logger.error(
                    f"API error: {method} {endpoint}",
                    error=e,
                    metadata={
                        "event": "api_error",
                        "method": method,
                        "endpoint": endpoint,
                        "duration": duration,
                        "request_id": request_id
                    }
                )
                raise
        
        return wrapper
    return decorator
```

## 📊 Fluentd Integration

### Alternative Log Aggregation

```yaml
# logging/docker-compose.fluentd.yml
version: '3.8'

services:
  fluentd:
    image: fluent/fluentd:v1.16-1
    container_name: prsm-fluentd
    ports:
      - "24224:24224"
      - "24224:24224/udp"
    volumes:
      - ./fluentd/conf:/fluentd/etc
      - ./fluentd/logs:/var/log
    environment:
      - FLUENTD_CONF=fluent.conf
    networks:
      - logging

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    container_name: prsm-elasticsearch-fluentd
    environment:
      - node.name=elasticsearch
      - cluster.name=prsm-fluentd
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_fluentd_data:/usr/share/elasticsearch/data
    ports:
      - "9201:9200"
    networks:
      - logging

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    container_name: prsm-kibana-fluentd
    ports:
      - "5602:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    networks:
      - logging
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_fluentd_data:

networks:
  logging:
    driver: bridge
```

### Fluentd Configuration

```ruby
# logging/fluentd/conf/fluent.conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<source>
  @type http
  port 9880
  bind 0.0.0.0
</source>

<source>
  @type tail
  path /var/log/prsm/*.log
  pos_file /var/log/fluentd-prsm.log.pos
  tag prsm.app
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%LZ
</source>

<filter prsm.**>
  @type record_transformer
  <record>
    hostname ${hostname}
    service prsm
    environment ${ENV["ENVIRONMENT"] || "production"}
    version ${ENV["PRSM_VERSION"] || "1.0.0"}
  </record>
</filter>

<filter prsm.**>
  @type grep
  <exclude>
    key level
    pattern DEBUG
  </exclude>
</filter>

<filter prsm.**>
  @type parser
  key_name message
  reserve_data true
  <parse>
    @type json
  </parse>
</filter>

<filter prsm.**>
  @type record_transformer
  enable_ruby
  <record>
    timestamp ${Time.now.utc.iso8601}
  </record>
</filter>

<match prsm.error>
  @type copy
  <store>
    @type elasticsearch
    host elasticsearch
    port 9200
    index_name prsm-errors
    type_name _doc
    logstash_format true
    logstash_prefix prsm-errors
    logstash_dateformat %Y.%m.%d
    time_key @timestamp
    flush_interval 10s
  </store>
  <store>
    @type slack
    webhook_url "#{ENV['SLACK_WEBHOOK_URL']}"
    channel "#prsm-alerts"
    username fluentd
    icon_emoji :exclamation:
    title "PRSM Error Alert"
    message "Error in %s: %s" % [record['component'], record['message']]
    flush_interval 30s
  </store>
</match>

<match prsm.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name prsm
  type_name _doc
  logstash_format true
  logstash_prefix prsm
  logstash_dateformat %Y.%m.%d
  time_key @timestamp
  flush_interval 10s
  <buffer>
    flush_thread_count 2
    flush_interval 5s
    chunk_limit_size 2M
    queue_limit_length 32
    retry_max_interval 30
    retry_forever true
  </buffer>
</match>
```

## 🔍 Log Analysis and Search

### Kibana Dashboard Configuration

```json
{
  "version": "8.5.0",
  "objects": [
    {
      "id": "prsm-overview-dashboard",
      "type": "dashboard",
      "attributes": {
        "title": "PRSM Log Overview",
        "hits": 0,
        "description": "Overview of PRSM application logs",
        "panelsJSON": "[{\"version\":\"8.5.0\",\"type\":\"visualization\",\"gridData\":{\"x\":0,\"y\":0,\"w\":24,\"h\":15,\"i\":\"1\"},\"panelIndex\":\"1\",\"embeddableConfig\":{},\"panelRefName\":\"panel_1\"}]",
        "timeRestore": false,
        "timeTo": "now",
        "timeFrom": "now-24h",
        "refreshInterval": {
          "pause": false,
          "value": 30000
        },
        "kibanaSavedObjectMeta": {
          "searchSourceJSON": "{\"query\":{\"match_all\":{}},\"filter\":[]}"
        }
      }
    },
    {
      "id": "prsm-error-logs",
      "type": "search",
      "attributes": {
        "title": "PRSM Error Logs",
        "description": "All error logs from PRSM application",
        "hits": 0,
        "columns": ["@timestamp", "level", "component", "message", "user_id", "request_id"],
        "sort": [["@timestamp", "desc"]],
        "kibanaSavedObjectMeta": {
          "searchSourceJSON": "{\"index\":\"prsm-*\",\"query\":{\"match\":{\"level\":\"ERROR\"}},\"filter\":[]}"
        }
      }
    },
    {
      "id": "prsm-performance-logs",
      "type": "search",
      "attributes": {
        "title": "PRSM Performance Logs",
        "description": "Performance-related logs from PRSM",
        "hits": 0,
        "columns": ["@timestamp", "component", "duration", "query_type", "model"],
        "sort": [["duration", "desc"]],
        "kibanaSavedObjectMeta": {
          "searchSourceJSON": "{\"index\":\"prsm-*\",\"query\":{\"bool\":{\"must\":[{\"exists\":{\"field\":\"duration\"}},{\"range\":{\"duration\":{\"gte\":1000}}}]}},\"filter\":[]}"
        }
      }
    }
  ]
}
```

### Log Analysis Queries

```python
# logging/log_analyzer.py
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from elasticsearch import AsyncElasticsearch
import pandas as pd

class LogAnalyzer:
    """Advanced log analysis for PRSM."""
    
    def __init__(self, elasticsearch_url: str = "http://localhost:9200"):
        self.es = AsyncElasticsearch([elasticsearch_url])
        self.index_pattern = "prsm-*"
    
    async def get_error_summary(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get summary of errors in time range."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"level": "ERROR"}},
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "error_types": {
                    "terms": {
                        "field": "exception_type.keyword",
                        "size": 10
                    }
                },
                "components": {
                    "terms": {
                        "field": "component.keyword",
                        "size": 10
                    }
                },
                "hourly_distribution": {
                    "date_histogram": {
                        "field": "@timestamp",
                        "calendar_interval": "hour"
                    }
                }
            }
        }
        
        result = await self.es.search(
            index=self.index_pattern,
            body=query,
            size=0
        )
        
        return {
            "total_errors": result["hits"]["total"]["value"],
            "error_types": result["aggregations"]["error_types"]["buckets"],
            "components": result["aggregations"]["components"]["buckets"],
            "hourly_distribution": result["aggregations"]["hourly_distribution"]["buckets"]
        }
    
    async def get_performance_stats(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get performance statistics."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "duration"}},
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "avg_duration": {"avg": {"field": "duration"}},
                "max_duration": {"max": {"field": "duration"}},
                "min_duration": {"min": {"field": "duration"}},
                "percentiles": {
                    "percentiles": {
                        "field": "duration",
                        "percents": [50, 90, 95, 99]
                    }
                },
                "by_endpoint": {
                    "terms": {
                        "field": "endpoint.keyword",
                        "size": 10
                    },
                    "aggs": {
                        "avg_duration": {"avg": {"field": "duration"}}
                    }
                }
            }
        }
        
        result = await self.es.search(
            index=self.index_pattern,
            body=query,
            size=0
        )
        
        return {
            "total_requests": result["hits"]["total"]["value"],
            "avg_duration": result["aggregations"]["avg_duration"]["value"],
            "max_duration": result["aggregations"]["max_duration"]["value"],
            "min_duration": result["aggregations"]["min_duration"]["value"],
            "percentiles": result["aggregations"]["percentiles"]["values"],
            "by_endpoint": result["aggregations"]["by_endpoint"]["buckets"]
        }
    
    async def get_user_activity(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get user activity logs."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}},
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [{"@timestamp": {"order": "desc"}}]
        }
        
        result = await self.es.search(
            index=self.index_pattern,
            body=query,
            size=1000
        )
        
        return [hit["_source"] for hit in result["hits"]["hits"]]
    
    async def find_slow_queries(
        self,
        threshold_ms: int = 5000,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[Dict[str, Any]]:
        """Find slow queries above threshold."""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "query_type"}},
                        {"range": {"duration": {"gte": threshold_ms}}},
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [{"duration": {"order": "desc"}}]
        }
        
        result = await self.es.search(
            index=self.index_pattern,
            body=query,
            size=100
        )
        
        return [hit["_source"] for hit in result["hits"]["hits"]]
    
    async def get_security_events(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get security-related events."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"term": {"event": "security"}},
                                    {"match": {"message": "authentication"}},
                                    {"match": {"message": "authorization"}},
                                    {"match": {"message": "suspicious"}}
                                ]
                            }
                        },
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [{"@timestamp": {"order": "desc"}}]
        }
        
        result = await self.es.search(
            index=self.index_pattern,
            body=query,
            size=1000
        )
        
        return [hit["_source"] for hit in result["hits"]["hits"]]
    
    async def generate_daily_report(
        self,
        date: datetime = None
    ) -> Dict[str, Any]:
        """Generate daily log analysis report."""
        if not date:
            date = datetime.utcnow().date()
        
        start_time = datetime.combine(date, datetime.min.time())
        end_time = start_time + timedelta(days=1)
        
        # Gather all statistics
        error_summary = await self.get_error_summary(start_time, end_time)
        performance_stats = await self.get_performance_stats(start_time, end_time)
        slow_queries = await self.find_slow_queries(5000, start_time, end_time)
        security_events = await self.get_security_events(start_time, end_time)
        
        return {
            "date": date.isoformat(),
            "summary": {
                "total_errors": error_summary["total_errors"],
                "total_requests": performance_stats["total_requests"],
                "avg_response_time": performance_stats["avg_duration"],
                "slow_queries_count": len(slow_queries),
                "security_events_count": len(security_events)
            },
            "errors": error_summary,
            "performance": performance_stats,
            "slow_queries": slow_queries[:10],  # Top 10
            "security_events": security_events[:20]  # Top 20
        }
    
    async def close(self):
        """Close Elasticsearch connection."""
        await self.es.close()

# Usage example
async def main():
    analyzer = LogAnalyzer()
    
    try:
        # Generate report for yesterday
        yesterday = datetime.utcnow().date() - timedelta(days=1)
        report = await analyzer.generate_daily_report(yesterday)
        
        print(json.dumps(report, indent=2, default=str))
        
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## 📋 Production Deployment

### Complete Logging Stack Deployment

```bash
#!/bin/bash
# scripts/deploy-logging.sh

set -e

echo "Deploying PRSM Logging Stack..."

# Create necessary directories
mkdir -p logging/{elasticsearch,logstash,kibana,filebeat}
mkdir -p logging/elasticsearch/config
mkdir -p logging/logstash/{config,pipeline}
mkdir -p logging/kibana/config
mkdir -p logging/filebeat

# Set proper permissions
sudo chown -R 1000:1000 logging/elasticsearch
sudo chown -R 1000:1000 logging/kibana

# Create logging network
docker network create logging 2>/dev/null || true

# Deploy Elasticsearch first
echo "Starting Elasticsearch..."
docker-compose -f logging/docker-compose.elk.yml up -d elasticsearch

# Wait for Elasticsearch to be ready
echo "Waiting for Elasticsearch..."
timeout 120 bash -c 'until curl -f http://localhost:9200/_cluster/health; do sleep 5; done'

# Deploy Logstash
echo "Starting Logstash..."
docker-compose -f logging/docker-compose.elk.yml up -d logstash

# Deploy Kibana
echo "Starting Kibana..."
docker-compose -f logging/docker-compose.elk.yml up -d kibana

# Wait for Kibana to be ready
echo "Waiting for Kibana..."
timeout 120 bash -c 'until curl -f http://localhost:5601/api/status; do sleep 5; done'

# Deploy Filebeat
echo "Starting Filebeat..."
docker-compose -f logging/docker-compose.elk.yml up -d filebeat

# Setup Kibana index patterns
echo "Setting up Kibana index patterns..."
curl -X POST "localhost:5601/api/saved_objects/index-pattern/prsm-*" \
  -H "Content-Type: application/json" \
  -H "kbn-xsrf: true" \
  -d '{
    "attributes": {
      "title": "prsm-*",
      "timeFieldName": "@timestamp"
    }
  }'

echo "Logging stack deployed successfully!"
echo ""
echo "Access URLs:"
echo "  Elasticsearch: http://localhost:9200"
echo "  Kibana:       http://localhost:5601"
echo "  Logstash:     http://localhost:9600"
echo ""
echo "Import dashboards from: logging/kibana/dashboards/"
```

---

**Need help with logging integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).
# Configuration Deep Dive

This tutorial covers comprehensive configuration options for PRSM, including environment setup, authentication, and advanced configuration patterns.

## Overview

PRSM supports multiple configuration methods:
- Environment variables
- Configuration files
- Programmatic configuration
- Runtime configuration updates

## Environment Variables

### Core Configuration

```bash
# API Configuration
export PRSM_API_KEY="your-api-key"
export PRSM_BASE_URL="https://api.prsm.network"
export PRSM_API_VERSION="v1"

# Performance Configuration
export PRSM_TIMEOUT="30"
export PRSM_MAX_RETRIES="3"
export PRSM_RETRY_DELAY="1.0"

# Cost Management
export PRSM_DAILY_BUDGET="100.0"
export PRSM_COST_ALERTS="true"
export PRSM_AUTO_OPTIMIZE="true"

# Logging
export PRSM_LOG_LEVEL="INFO"
export PRSM_LOG_FORMAT="json"
export PRSM_LOG_FILE="/var/log/prsm.log"
```

### Development vs Production

```bash
# Development
export PRSM_ENVIRONMENT="development"
export PRSM_DEBUG="true"
export PRSM_CACHE_ENABLED="false"

# Production
export PRSM_ENVIRONMENT="production"
export PRSM_DEBUG="false"
export PRSM_CACHE_ENABLED="true"
export PRSM_CACHE_TTL="3600"
```

## Configuration Files

### YAML Configuration

Create `prsm.yaml`:

```yaml
api:
  base_url: "https://api.prsm.network"
  version: "v1"
  timeout: 30
  max_retries: 3

authentication:
  api_key: "${PRSM_API_KEY}"
  auto_refresh: true

cost_management:
  daily_budget: 100.0
  alert_thresholds: [0.5, 0.8, 0.95]
  auto_optimize: true
  preferred_providers: ["openai", "anthropic"]

performance:
  cache_enabled: true
  cache_ttl: 3600
  connection_pool_size: 10
  request_timeout: 30

logging:
  level: "INFO"
  format: "json"
  handlers:
    - type: "file"
      filename: "/var/log/prsm.log"
    - type: "console"
      
models:
  default: "gpt-4"
  fallbacks: ["claude-3", "gpt-3.5-turbo"]
  
  routing:
    strategy: "cost_optimized"
    quality_threshold: 0.8
    latency_threshold: 5.0
```

### JSON Configuration

Create `prsm.json`:

```json
{
  "api": {
    "base_url": "https://api.prsm.network",
    "version": "v1",
    "timeout": 30,
    "max_retries": 3
  },
  "authentication": {
    "api_key": "${PRSM_API_KEY}",
    "auto_refresh": true
  },
  "cost_management": {
    "daily_budget": 100.0,
    "alert_thresholds": [0.5, 0.8, 0.95],
    "auto_optimize": true
  },
  "models": {
    "default": "gpt-4",
    "fallbacks": ["claude-3", "gpt-3.5-turbo"]
  }
}
```

## Programmatic Configuration

### Python SDK

```python
from prsm_sdk import PRSMClient, Config

# Basic configuration
config = Config(
    api_key="your-api-key",
    base_url="https://api.prsm.network",
    timeout=30.0,
    max_retries=3
)

client = PRSMClient(config=config)

# Advanced configuration
advanced_config = Config(
    api_key="your-api-key",
    
    # Performance settings
    timeout=30.0,
    max_retries=3,
    retry_delay=1.0,
    connection_pool_size=10,
    
    # Cost management
    daily_budget=100.0,
    auto_optimize=True,
    cost_alert_thresholds=[0.5, 0.8, 0.95],
    
    # Model preferences
    default_model="gpt-4",
    fallback_models=["claude-3", "gpt-3.5-turbo"],
    routing_strategy="cost_optimized",
    
    # Caching
    cache_enabled=True,
    cache_ttl=3600,
    
    # Logging
    log_level="INFO",
    log_format="json"
)

client = PRSMClient(config=advanced_config)
```

### JavaScript SDK

```javascript
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({
  apiKey: 'your-api-key',
  baseURL: 'https://api.prsm.network',
  
  // Performance settings
  timeout: 30000,
  maxRetries: 3,
  retryDelay: 1000,
  
  // Cost management
  dailyBudget: 100.0,
  autoOptimize: true,
  costAlertThresholds: [0.5, 0.8, 0.95],
  
  // Model preferences
  defaultModel: 'gpt-4',
  fallbackModels: ['claude-3', 'gpt-3.5-turbo'],
  routingStrategy: 'cost_optimized',
  
  // Caching
  cacheEnabled: true,
  cacheTTL: 3600,
  
  // Logging
  logLevel: 'info'
});
```

## Advanced Configuration Patterns

### Environment-Specific Configuration

```python
import os
from prsm_sdk import PRSMClient, Config

def create_client():
    env = os.getenv('PRSM_ENVIRONMENT', 'development')
    
    if env == 'development':
        config = Config(
            api_key=os.getenv('PRSM_API_KEY'),
            base_url="https://dev-api.prsm.network",
            timeout=60.0,  # Longer timeout for dev
            debug=True,
            cache_enabled=False
        )
    elif env == 'staging':
        config = Config(
            api_key=os.getenv('PRSM_API_KEY'),
            base_url="https://staging-api.prsm.network",
            timeout=30.0,
            cache_enabled=True,
            cache_ttl=1800  # 30 minutes
        )
    else:  # production
        config = Config(
            api_key=os.getenv('PRSM_API_KEY'),
            base_url="https://api.prsm.network",
            timeout=30.0,
            max_retries=5,
            cache_enabled=True,
            cache_ttl=3600,  # 1 hour
            daily_budget=1000.0,
            auto_optimize=True
        )
    
    return PRSMClient(config=config)

client = create_client()
```

### Dynamic Configuration Updates

```python
# Update configuration at runtime
await client.config.update(
    daily_budget=150.0,
    auto_optimize=False
)

# Get current configuration
current_config = await client.config.get()
print(f"Current budget: ${current_config.daily_budget}")

# Reset to defaults
await client.config.reset()
```

### Configuration Validation

```python
from prsm_sdk import Config, ConfigValidationError

try:
    config = Config(
        api_key="invalid-key-format",
        timeout=-1,  # Invalid negative timeout
        daily_budget="not-a-number"  # Invalid type
    )
except ConfigValidationError as e:
    print(f"Configuration error: {e.message}")
    for error in e.validation_errors:
        print(f"  {error.field}: {error.message}")
```

## Security Configuration

### API Key Management

```python
# Using environment variables (recommended)
config = Config(
    api_key=os.getenv('PRSM_API_KEY')
)

# Using key management service
from your_key_manager import get_secret

config = Config(
    api_key=get_secret('prsm-api-key')
)

# Key rotation support
config = Config(
    api_key=os.getenv('PRSM_API_KEY'),
    api_key_rotation_enabled=True,
    api_key_rotation_interval=86400  # 24 hours
)
```

### SSL/TLS Configuration

```python
config = Config(
    api_key="your-api-key",
    ssl_verify=True,
    ssl_cert_path="/path/to/cert.pem",
    ssl_key_path="/path/to/key.pem",
    ssl_ca_path="/path/to/ca.pem"
)
```

## Monitoring and Observability

### Metrics Configuration

```python
config = Config(
    api_key="your-api-key",
    
    # Metrics collection
    metrics_enabled=True,
    metrics_endpoint="http://prometheus:9090",
    metrics_interval=60,
    
    # Custom labels
    metrics_labels={
        "service": "my-ai-app",
        "environment": "production",
        "version": "1.2.3"
    }
)
```

### Tracing Configuration

```python
config = Config(
    api_key="your-api-key",
    
    # Distributed tracing
    tracing_enabled=True,
    tracing_endpoint="http://jaeger:14268",
    tracing_sample_rate=0.1,  # 10% sampling
    
    # Custom trace attributes
    trace_attributes={
        "service.name": "my-ai-service",
        "service.version": "1.2.3"
    }
)
```

## Configuration Best Practices

### 1. Environment Variable Priority

```python
# Priority order (highest to lowest):
# 1. Programmatic configuration
# 2. Environment variables
# 3. Configuration file
# 4. Default values

config = Config(
    api_key="programmatic-key",  # Highest priority
    # PRSM_API_KEY env var would be ignored
    config_file="prsm.yaml"      # Lowest priority
)
```

### 2. Sensitive Data Handling

```bash
# Use environment variables for sensitive data
export PRSM_API_KEY="sk-..."
export PRSM_DATABASE_URL="postgresql://..."

# Never commit sensitive data to version control
echo "PRSM_API_KEY=sk-..." >> .env
echo ".env" >> .gitignore
```

### 3. Configuration Validation

```python
def validate_config(config):
    """Validate configuration before use"""
    if not config.api_key:
        raise ValueError("API key is required")
    
    if config.timeout <= 0:
        raise ValueError("Timeout must be positive")
    
    if config.daily_budget < 0:
        raise ValueError("Daily budget cannot be negative")
    
    return True

# Use validation
config = Config.from_file("prsm.yaml")
validate_config(config)
client = PRSMClient(config=config)
```

### 4. Configuration Testing

```python
import pytest
from prsm_sdk import Config

def test_config_defaults():
    config = Config(api_key="test-key")
    assert config.timeout == 30.0
    assert config.max_retries == 3
    assert config.cache_enabled is True

def test_config_from_env():
    os.environ['PRSM_TIMEOUT'] = '60'
    config = Config.from_env()
    assert config.timeout == 60.0

def test_config_validation():
    with pytest.raises(ConfigValidationError):
        Config(api_key="", timeout=-1)
```

## Troubleshooting Configuration

### Common Issues

1. **API Key Not Found**
   ```bash
   # Check environment variable
   echo $PRSM_API_KEY
   
   # Verify format
   if [[ $PRSM_API_KEY == prsm_* ]]; then
     echo "Valid format"
   else
     echo "Invalid format - should start with 'prsm_'"
   fi
   ```

2. **Configuration File Not Loaded**
   ```python
   # Debug configuration loading
   config = Config.from_file("prsm.yaml", debug=True)
   # Will print which files were attempted and why they failed
   ```

3. **Environment Variable Override Not Working**
   ```python
   # Check environment variable precedence
   config = Config(api_key="code-value")
   print(f"API Key source: {config.get_source('api_key')}")
   # Should show 'programmatic' > 'environment' > 'file' > 'default'
   ```

## Next Steps

- [Query Processing](#query-processing) - Learn about request processing
- [Hands-On Exercises](#hands-on-exercises) - Practice configuration scenarios
- [API Fundamentals](./api-fundamentals.md) - Core API concepts

## Query Processing

Advanced query handling, context management, and response optimization are key components of PRSM's architecture:

### Context Management
- Maintain conversation context across requests
- Handle memory and context limits efficiently  
- Optimize context for performance

### Response Optimization
- Stream responses for better user experience
- Handle rate limiting and retries
- Optimize response formatting

## Hands-On Exercises

Practice these configuration scenarios to reinforce your learning:

### Exercise 1: Environment Setup
Configure your development environment with proper API keys and settings.

### Exercise 2: Advanced Configuration
Set up multiple environments (dev, staging, production) with different configurations.

### Exercise 3: Performance Tuning
Optimize configuration settings for your specific use case and performance requirements.
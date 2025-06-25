# PRSM Persistent Test Environment

🎯 **Purpose**: Create and manage persistent test environments for PRSM system validation, development, and continuous integration.

## Overview

The Persistent Test Environment provides a comprehensive framework for:

- 🏗️ **Environment Management**: Create, configure, and manage isolated test environments
- 🧪 **Test Execution**: Run comprehensive test suites with real-time monitoring
- 📊 **Performance Monitoring**: Continuous health checks and performance benchmarking
- 💾 **Data Management**: Persistent test data with automated generation and cleanup
- 🔄 **CI/CD Integration**: Support for automated testing in pipelines

## Key Features

### 🚀 Environment Lifecycle Management
- Automated setup and teardown of test environments
- Persistent state across test runs
- Service dependency management
- Configuration management and versioning

### 🧪 Comprehensive Testing
- System health tests
- Integration tests  
- Performance benchmarks
- Stress testing
- Security validation

### 📈 Monitoring & Metrics
- Real-time health monitoring
- Performance metrics collection
- Resource usage tracking
- Automated alerting and recovery

### 🔒 Test Isolation
- Isolated test execution contexts
- Automatic cleanup after tests
- Temporary data management
- State isolation between tests

## Quick Start

### 1. Create a Test Environment

```python
from tests.environment import create_test_environment, TestEnvironmentConfig

# Configure environment
config = TestEnvironmentConfig(
    environment_id="my_test_env",
    persistent_data=True,
    performance_monitoring=True
)

# Create environment
env = await create_test_environment(config)
```

### 2. Run Tests

```python
from tests.environment import TestRunner

# Create test runner
runner = TestRunner()

# Run comprehensive tests
results = await runner.run_comprehensive_tests("my_test_env")

print(f"Success Rate: {results['summary']['success_rate']:.1%}")
```

### 3. Use Test Context

```python
# Isolated test execution
async with env.test_context("my_test") as ctx:
    # Test code here - automatic cleanup after
    test_id = ctx['test_id']
    temp_dir = ctx['temp_dir']
    # ... test operations ...
```

## Command Line Interface

### Create Environment
```bash
python tests/environment/test_runner.py create my_environment
```

### List Environments
```bash
python tests/environment/test_runner.py list
```

### Run Tests
```bash
# Run all test suites
python tests/environment/test_runner.py test my_environment

# Run specific suites
python tests/environment/test_runner.py test my_environment --suites system_health integration
```

## Test Suites Available

### 🔧 System Health Tests
- Component health verification
- Service connectivity tests
- Data integrity checks
- Resource usage monitoring

### 🔗 Integration Tests  
- Database operations
- FTNS tokenomics operations
- Agent framework functionality
- API endpoint validation
- Cross-service integration

### ⚡ Performance Tests
- Database performance benchmarks
- FTNS operation benchmarks
- Memory usage validation
- Response time measurements

### 💪 Stress Tests
- Concurrent user simulation
- Resource limit testing
- System stability under load

### 🔒 Security Tests
- Authentication verification
- Authorization validation
- Data protection checks

## Configuration Options

```python
config = TestEnvironmentConfig(
    environment_id="custom_env",           # Unique environment identifier
    base_directory=Path("./test_envs"),    # Base directory for environments
    persistent_data=True,                  # Keep data between runs
    auto_cleanup=False,                    # Automatic cleanup on exit
    health_check_interval=30,              # Health check frequency (seconds)
    performance_monitoring=True,           # Enable performance monitoring
    isolated_services=True,                # Use isolated service instances
    test_data_seed="custom_seed"           # Seed for test data generation
)
```

## Directory Structure

When you create an environment, it creates the following structure:

```
test_environments/
└── my_environment/
    ├── config/                 # Environment configuration files
    │   ├── environment.json    # Main environment config
    │   ├── database.json      # Database configuration
    │   ├── redis.json         # Redis configuration
    │   └── vector_db.json     # Vector database config
    ├── data/                  # Persistent data storage
    │   ├── database/          # Database files
    │   ├── redis/             # Redis data
    │   ├── vector_db/         # Vector database data
    │   ├── ipfs/              # IPFS data
    │   └── test_data_info.json # Test data metadata
    ├── logs/                  # Log files
    │   ├── services/          # Service logs
    │   ├── tests/             # Test execution logs
    │   └── performance/       # Performance monitoring logs
    └── temp/                  # Temporary files (cleaned up)
```

## Demo

Run the comprehensive demo to see all features:

```bash
python tests/environment/demo_persistent_environment.py
```

This demo will:
1. ✅ Create a persistent test environment
2. 📊 Show environment status and health
3. 🧪 Demonstrate test isolation
4. 💾 Show test data management
5. 🔧 Run comprehensive tests
6. 📈 Display performance monitoring
7. 💾 Demonstrate persistence features
8. ⏱️ Show real-time monitoring

## Advanced Usage

### Custom Test Data Generation

```python
# Generate custom test data
async def generate_custom_data(env):
    # Add custom test data generators
    env.test_data_generators['custom_users'] = custom_user_generator
    await env._generate_test_data()
```

### Environment Monitoring

```python
# Get detailed environment status
status = await env.get_environment_status()

print(f"Status: {status['status']}")
print(f"Components: {status['components_status']}")
print(f"Metrics: {status['performance_metrics']}")
```

### Custom Health Checks

```python
# Add custom health checks
async def custom_health_check(env):
    # Your custom health check logic
    return True

# Integrate with environment
env.custom_health_checks.append(custom_health_check)
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: PRSM Test Environment
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Create test environment
        run: python tests/environment/test_runner.py create ci_environment
      
      - name: Run comprehensive tests
        run: python tests/environment/test_runner.py test ci_environment
```

## Benefits

### 🎯 For Development
- **Consistent Testing**: Same environment across all developers
- **Fast Setup**: Automated environment provisioning
- **Real Dependencies**: Tests against actual PRSM components
- **Isolation**: No interference between test runs

### 🚀 For CI/CD
- **Reproducible**: Identical environments every time
- **Comprehensive**: Full system validation
- **Fast Feedback**: Quick identification of issues
- **Evidence**: Detailed test reports and metrics

### 📊 For Operations
- **Monitoring**: Continuous health and performance tracking
- **Debugging**: Persistent environments for investigation
- **Benchmarking**: Performance regression detection
- **Validation**: Production-like testing scenarios

## Troubleshooting

### Environment Creation Issues
```python
# Check service health
await env._comprehensive_health_check()

# View detailed status
status = await env.get_environment_status()
print(status['errors'])
```

### Performance Issues
```python
# Monitor resource usage
metrics = await env._collect_performance_metrics()
print(f"Memory: {metrics['memory_usage_mb']} MB")
print(f"CPU: {metrics['cpu_percent']}%")
```

### Test Failures
```python
# Run individual test suites
runner = TestRunner()
health_results = await runner._run_system_health_tests(env)
integration_results = await runner._run_integration_tests(env)
```

## Contributing

To extend the test environment:

1. **Add Test Suites**: Create new test suite methods in `TestRunner`
2. **Custom Monitoring**: Add performance metrics collection
3. **Service Integration**: Add support for new PRSM services
4. **Test Data**: Create specialized test data generators

Example:
```python
async def _run_custom_tests(self, env: PersistentTestEnvironment) -> Dict[str, Any]:
    # Your custom test suite implementation
    return {"success": True, "duration": 10.0}

# Add to available suites
available_suites["custom"] = self._run_custom_tests
```

---

🎉 **Ready to test PRSM with confidence!** The persistent test environment provides everything you need for comprehensive, reliable, and efficient testing of the PRSM system.
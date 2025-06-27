# PRSM Test Suite

## Overview

This directory contains the comprehensive test suite for PRSM (Protocol for Recursive Scientific Modeling). The test suite has been improved to address external audit findings regarding test consistency and quality.

## Test Suite Improvements

### ✅ Standardized Test Framework
- **Migrated from informal scripts to pytest** - All new tests follow pytest conventions
- **Added proper assertions** - Replaced print statements with formal assert statements
- **Implemented shared fixtures** - Common test setup in `conftest.py`
- **Added test categorization** - Markers for unit, integration, performance tests

### ✅ Improved Test Files

#### Newly Converted Tests
- `test_adaptive_consensus_engine.py` - Converted from `simple_adaptive_test.py`
- `test_project_foundation.py` - Converted from `test_foundation.py`
- `test_rlt_compiler_improved.py` - Improved version of `test_rlt_enhanced_compiler.py`

#### Shared Infrastructure
- `conftest.py` - Shared fixtures and test configuration
- `pytest.ini` - Pytest configuration and standards

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_adaptive_consensus_engine.py

# Run tests by marker
pytest -m unit           # Run only unit tests
pytest -m integration    # Run only integration tests
pytest -m "not slow"     # Skip slow tests
```

### Test Categories

#### Unit Tests (`@pytest.mark.unit`)
- Test individual components in isolation
- Fast execution (< 1 second per test)
- Use mocking for external dependencies

#### Integration Tests (`@pytest.mark.integration`)
- Test component interactions
- Moderate execution time (1-10 seconds)
- Test realistic workflows

#### Performance Tests (`@pytest.mark.performance`)
- Benchmark and load testing
- Longer execution time (10+ seconds)
- Validate performance claims

#### Network Tests (`@pytest.mark.network`)
- Test P2P network functionality
- Require network simulation
- May be skipped in some environments

## Test Quality Standards

### ✅ Good Test Practices
```python
import pytest

class TestMyComponent:
    """Test suite for MyComponent functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data"""
        return {"key": "value"}
    
    def test_component_creation(self, sample_data):
        """Test component creation with sample data"""
        component = MyComponent(sample_data)
        
        # Use formal assertions
        assert component is not None
        assert component.data == sample_data
        
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test asynchronous operations"""
        result = await my_async_function()
        assert result.success is True
```

### ❌ Avoid These Patterns
```python
# Don't use print statements for validation
def test_something():
    result = my_function()
    print(f"Result: {result}")  # ❌ No validation
    
# Don't use manual success/failure tracking
def test_something():
    try:
        my_function()
        return True  # ❌ Use assertions instead
    except:
        return False

# Don't write scripts instead of tests
def main():  # ❌ Should be test functions
    test_feature_1()
    test_feature_2()
```

## Fixtures and Utilities

### Available Fixtures (from conftest.py)
- `sample_peer_nodes` - Basic peer network for testing
- `large_peer_network` - Larger network for scalability tests
- `mock_ftns_service` - Mocked tokenomics service
- `test_config` - Test configuration dictionary
- `mock_network_conditions` - Various network scenarios
- `test_helpers` - Helper functions for common assertions

### Helper Functions
```python
def test_consensus_result(test_helpers):
    result = achieve_consensus()
    test_helpers.assert_consensus_result_valid(result)
```

## Legacy Test Files

Some test files in the repository still use the old informal script pattern. These are being gradually converted:

### Files Still Needing Conversion
- `test_dashboard.py` - Uses print statements
- `standalone_pq_test.py` - Manual verification
- Various scripts in `scripts_integration/` - Mixed quality

### Migration Strategy
1. **Preserve existing functionality** - Don't break working tests
2. **Add proper assertions** - Replace print statements
3. **Use pytest fixtures** - Eliminate duplicate setup code
4. **Add test categorization** - Use appropriate markers
5. **Improve documentation** - Clear test descriptions

## Contributing to Tests

### Adding New Tests
1. **Follow naming conventions** - `test_*.py` for files, `test_*` for functions
2. **Use appropriate markers** - Categorize tests properly
3. **Write clear docstrings** - Explain what the test validates
4. **Use fixtures** - Leverage shared setup code
5. **Add parametrized tests** - Test multiple scenarios efficiently

### Test Review Checklist
- [ ] Uses proper pytest conventions
- [ ] Has formal assert statements (no print debugging)
- [ ] Uses appropriate fixtures for setup
- [ ] Has clear, descriptive test names
- [ ] Includes docstrings explaining test purpose
- [ ] Uses appropriate test markers
- [ ] Follows error handling best practices

## Performance Testing

Performance tests validate the technical claims made in documentation:

```python
@pytest.mark.performance
def test_consensus_performance():
    """Validate consensus performance under load"""
    start_time = time.time()
    result = achieve_consensus_with_load()
    duration = time.time() - start_time
    
    assert result.success is True
    assert duration < 5.0  # Performance requirement
```

## Continuous Integration

The test suite is designed to work with CI/CD systems:

```bash
# Quick test run for CI
pytest -m "unit and not slow" --tb=line

# Full test suite for nightly builds
pytest -m "unit or integration" --cov=prsm
```

This improved test suite addresses the external audit findings about test consistency while maintaining comprehensive coverage of PRSM functionality.
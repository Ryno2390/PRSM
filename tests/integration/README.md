# Integration Testing for P2P Secure Collaboration Platform

This directory contains comprehensive integration tests for the PRSM P2P Secure Collaboration Platform, validating the integration between all major components including P2P networking, post-quantum security, cryptographic sharding, and user interfaces.

## Test Suite Overview

### 🔗 P2P Integration Tests (`test_p2p_integration.py`)
Tests the complete P2P network stack integration:
- **P2P Network Layer**: Node discovery, shard distribution, bandwidth optimization, reputation system, fallback storage
- **Security Components**: Post-quantum key management, access control, reconstruction engine, integrity validation
- **End-to-End Workflows**: University-industry collaboration, multi-institutional grant writing

### 🖥️ UI Integration Tests (`test_ui_integration.py`)
Tests UI component integration with backend systems:
- **Dashboard Loading**: P2P network dashboard, security status indicators, shard distribution visualization
- **Interactive Features**: Real-time updates, user interactions, responsive design
- **API Integration**: Backend API connections, WebSocket communication, data synchronization
- **Accessibility**: Keyboard navigation, ARIA labels, mobile responsiveness

### ⚡ Performance Integration Tests (`test_performance_integration.py`)
Tests performance characteristics under realistic load:
- **File Operations**: Sharding and reconstruction performance across different file sizes
- **Network Operations**: Peer discovery, shard distribution, network optimization
- **Security Operations**: Key generation, access authorization, integrity validation
- **Scalability**: Performance scaling with increasing peer counts, memory usage under load

## Quick Start

### Prerequisites

Install required dependencies:
```bash
pip install pytest pytest-asyncio psutil selenium
```

For UI tests, install Chrome or Chromium:
```bash
# Ubuntu/Debian
sudo apt-get install chromium-browser

# macOS
brew install --cask google-chrome

# Or use existing Chrome installation
```

### Running Tests

#### Run All Integration Tests
```bash
python run_integration_tests.py
```

#### Run Specific Test Suites
```bash
# P2P integration tests only
python run_integration_tests.py --test-type p2p

# UI integration tests only  
python run_integration_tests.py --test-type ui

# Performance tests only
python run_integration_tests.py --test-type performance
```

#### Quick Smoke Test
```bash
python run_integration_tests.py --smoke-test
```

#### Verbose Output
```bash
python run_integration_tests.py --verbose
```

## Test Configuration

### Test Fixtures (`conftest.py`)

The test suite uses shared fixtures for:
- **Mock Network Peers**: Simulated university and research institution nodes
- **Mock Users**: Different user roles (PI, industry partner, collaborator, tech transfer)
- **Security Configuration**: Post-quantum settings, access control rules, sharding parameters
- **Performance Benchmarks**: Target and maximum execution times for operations
- **Sample Files**: Test files of various sizes and security levels

### Mock Data

Tests use realistic mock data representing:
- **Academic Institutions**: Stanford, MIT, Duke, UNC, Oxford, ETH Zurich, Tokyo Tech
- **Industry Partners**: SAS Institute, IBM Research
- **File Types**: Research papers, proprietary algorithms, datasets, grant proposals
- **Security Levels**: High (7-shard), Medium (5-shard), Standard (3-shard)

## Test Scenarios

### 🎓 University-Industry Collaboration
Tests secure IP evaluation workflow:
1. UNC researcher uploads proprietary quantum algorithm
2. File is sharded with high security (7 shards) across trusted nodes
3. SAS Institute requests evaluation access
4. Multi-signature approval process (PI + Tech Transfer Office)
5. Secure file reconstruction for authorized evaluation
6. Integrity validation throughout process

### 🔬 Multi-Institutional Research
Tests grant collaboration workflow:
1. Multi-university team (UNC, Duke, NC State, IBM) creates NSF proposal
2. Medium security sharding (5 shards) for collaborative editing
3. All team members receive access permissions
4. Real-time collaboration and version control
5. Final submission with institutional approvals

### 🔒 Security Threat Response
Tests security incident handling:
1. Threat detection system identifies suspicious activity
2. Automated threat analysis and risk assessment
3. Security response workflow (block, quarantine, investigate)
4. Integrity validation of affected shards
5. Network optimization after threat mitigation

## Performance Benchmarks

### File Operations
- **Small files** (<1MB): Shard in <1s, reconstruct in <0.5s
- **Medium files** (1-10MB): Shard in <5s, reconstruct in <2s  
- **Large files** (>10MB): Shard in <15s, reconstruct in <8s

### Network Operations
- **Peer discovery**: <2s for 50+ peers
- **Shard distribution**: <10s for 7-shard high security
- **Integrity validation**: <5s for complete file validation

### Security Operations
- **Post-quantum key generation**: <2s for Kyber-1024
- **Access authorization**: <1s for multi-signature approval
- **Digital signature verification**: <0.5s per signature

## Continuous Integration

### GitHub Actions Integration
```yaml
name: Integration Tests
on: [push, pull_request]
jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio psutil selenium
      - name: Run integration tests
        run: python tests/integration/run_integration_tests.py --generate-report
      - name: Upload test reports
        uses: actions/upload-artifact@v3
        with:
          name: integration-test-reports
          path: tests/integration/reports/
```

### Test Reporting

Tests generate comprehensive reports:
- **JUnit XML**: Compatible with CI/CD systems and test dashboards
- **JSON Summary**: Detailed test results with performance metrics
- **HTML Dashboard**: Visual test results and trends (optional)

## Extending Tests

### Adding New Test Cases

1. **Create test file**: Follow naming convention `test_[component]_integration.py`
2. **Use existing fixtures**: Leverage shared fixtures for consistency
3. **Follow test patterns**: Use established patterns for async tests, mocking, assertions
4. **Add performance benchmarks**: Include timing and resource usage validation
5. **Document scenarios**: Add clear documentation for test scenarios

### Example Test Structure
```python
class TestNewComponentIntegration:
    """Integration tests for new component"""
    
    @pytest.fixture
    def component_instance(self):
        """Initialize component for testing"""
        return NewComponent()
    
    @pytest.mark.asyncio
    async def test_component_workflow(self, component_instance, mock_dependencies):
        """Test complete component workflow"""
        # Arrange
        setup_data = create_test_data()
        
        # Act
        result = await component_instance.process(setup_data)
        
        # Assert
        assert result.success
        assert result.meets_requirements()
        
        # Performance verification
        assert result.duration < PERFORMANCE_BENCHMARK
```

### Mock Guidelines

- **Realistic behavior**: Mocks should simulate realistic timing, errors, edge cases
- **Configurable responses**: Use fixtures to configure mock responses for different scenarios
- **Resource simulation**: Include memory usage, network latency, processing time
- **Error conditions**: Test failure scenarios and recovery mechanisms

## Troubleshooting

### Common Issues

**Test Import Errors**:
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/path/to/PRSM:$PYTHONPATH
python run_integration_tests.py
```

**UI Test Failures**:
```bash
# Install Chrome/Chromium for Selenium
sudo apt-get install chromium-browser

# Or run without UI tests
python run_integration_tests.py --test-type p2p
```

**Performance Test Failures**:
```bash
# Adjust benchmarks for slower systems
# Edit performance_benchmarks in conftest.py
```

**Memory Issues**:
```bash
# Run with memory profiling
python -m pytest tests/integration/test_performance_integration.py::TestScalabilityPerformance --verbose
```

### Debug Mode

Enable detailed logging:
```bash
PYTHONPATH=/path/to/PRSM python -m pytest tests/integration/ -v -s --log-cli-level=DEBUG
```

## Security Considerations

### Test Data Security
- All test data uses mock/simulated content
- No real proprietary information in test files
- Cryptographic operations use test keys only
- Network tests use local/mock endpoints

### Test Isolation
- Each test runs in isolated environment
- Temporary directories cleaned up after tests
- Mock components prevent external network calls
- Database operations use in-memory stores

## Contributing

1. **Follow test patterns**: Use established conventions for async tests, fixtures, assertions
2. **Add comprehensive coverage**: Include happy path, error conditions, edge cases
3. **Performance validation**: Include timing and resource usage checks
4. **Clear documentation**: Document test scenarios, expected behavior, dependencies
5. **Review checklist**: Ensure tests are deterministic, isolated, and maintainable

For questions or issues, please refer to the main PRSM documentation or create an issue in the project repository.
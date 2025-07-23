#!/usr/bin/env python3
"""
Shared pytest configuration and fixtures for PRSM test suite
Provides common setup, teardown, and utility fixtures for all tests
"""

import pytest
import asyncio
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Add PRSM to path for all tests
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
try:
    from prsm.core.models import PeerNode
    from prsm.core.config import PRSMConfig
    from prsm.tokenomics.ftns_service import FTNSService
except ImportError as e:
    # If imports fail, we'll skip tests that require them
    pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def project_root():
    """Fixture providing the project root directory"""
    return PROJECT_ROOT


@pytest.fixture
def sample_peer_nodes():
    """Fixture providing a set of sample peer nodes for testing"""
    peer_nodes = []
    for i in range(5):
        peer = PeerNode(
            node_id=f"test_node_{i}",
            peer_id=f"test_peer_{i}",
            multiaddr=f"/ip4/127.0.0.1/tcp/{9000+i}",
            reputation_score=0.8,
            active=True
        )
        peer_nodes.append(peer)
    return peer_nodes


@pytest.fixture
def large_peer_network():
    """Fixture providing a larger network of peer nodes for scalability testing"""
    peer_nodes = []
    for i in range(20):
        reputation = 0.9 if i < 15 else 0.3  # Some nodes with lower reputation
        peer = PeerNode(
            node_id=f"large_node_{i}",
            peer_id=f"large_peer_{i}",
            multiaddr=f"/ip4/127.0.0.1/tcp/{8000+i}",
            reputation_score=reputation,
            active=i < 18  # Some inactive nodes
        )
        peer_nodes.append(peer)
    return peer_nodes


@pytest.fixture
def mock_ftns_service():
    """Fixture providing a mocked FTNS service for testing without dependencies"""
    mock_service = Mock(spec=FTNSService)
    mock_service.get_balance.return_value = 100.0
    mock_service.transfer.return_value = True
    mock_service.create_transaction.return_value = {
        "transaction_id": "test_tx_123",
        "status": "confirmed",
        "amount": 10.0
    }
    return mock_service


@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    config = {
        "test_mode": True,
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/15",  # Test database
        "log_level": "DEBUG",
        "network_size": 5,
        "consensus_timeout": 5.0,
        "max_retries": 3
    }
    return config


@pytest.fixture
def temp_directory(tmp_path):
    """Fixture providing a temporary directory for file operations"""
    return tmp_path


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Auto-use fixture to configure logging for tests"""
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Reduce noise from external libraries during testing
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    yield
    
    # Cleanup after test
    logging.getLogger().handlers.clear()


@pytest.fixture
def mock_network_conditions():
    """Fixture providing various network condition scenarios for testing"""
    return {
        "optimal": {
            "latency_ms": 10,
            "throughput_ops": 25,
            "failure_rate": 0.01,
            "byzantine_percentage": 0.0
        },
        "congested": {
            "latency_ms": 300,
            "throughput_ops": 3,
            "failure_rate": 0.05,
            "byzantine_percentage": 0.0
        },
        "unreliable": {
            "latency_ms": 150,
            "throughput_ops": 8,
            "failure_rate": 0.25,
            "byzantine_percentage": 0.2
        }
    }


@pytest.fixture
def sample_consensus_proposal():
    """Fixture providing a sample consensus proposal for testing"""
    return {
        "action": "test_consensus",
        "data": {
            "operation": "transfer",
            "amount": 50.0,
            "from": "test_user_1",
            "to": "test_user_2"
        },
        "timestamp": 1640995200,  # Fixed timestamp for deterministic testing
        "proposer": "test_node_0"
    }


@pytest.fixture
def performance_test_data():
    """Fixture providing data for performance testing"""
    return {
        "latency_samples": [10, 12, 15, 8, 20, 25, 18, 14, 16, 11],
        "throughput_samples": [20, 22, 18, 25, 19, 21, 23, 20, 24, 22],
        "success_rates": [0.95, 0.98, 0.92, 0.97, 0.94, 0.96, 0.99, 0.93, 0.95, 0.98]
    }


# Pytest markers for categorizing tests
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance/benchmark tests"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network simulation"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Add markers based on test/file names
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "network" in item.nodeid:
            item.add_marker(pytest.mark.network)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(slow_keyword in item.nodeid.lower() for slow_keyword in 
               ["slow", "large", "comprehensive", "stress"]):
            item.add_marker(pytest.mark.slow)


# Helper functions available to all tests
class TestHelpers:
    """Collection of helper functions for tests"""
    
    @staticmethod
    def assert_consensus_result_valid(result):
        """Assert that a consensus result has expected structure"""
        assert result is not None
        assert hasattr(result, 'consensus_achieved')
        assert isinstance(result.consensus_achieved, bool)
        if hasattr(result, 'votes'):
            assert isinstance(result.votes, (dict, list))
    
    @staticmethod
    def assert_peer_node_valid(peer_node):
        """Assert that a peer node has expected structure"""
        assert peer_node is not None
        assert hasattr(peer_node, 'node_id')
        assert hasattr(peer_node, 'peer_id')
        assert hasattr(peer_node, 'multiaddr')
        assert hasattr(peer_node, 'reputation_score')
        assert 0 <= peer_node.reputation_score <= 1
    
    @staticmethod
    def assert_network_metrics_valid(metrics):
        """Assert that network metrics have expected structure"""
        assert metrics is not None
        assert 'avg_latency_ms' in metrics
        assert 'avg_throughput' in metrics
        assert 'failure_rate' in metrics
        assert metrics['avg_latency_ms'] >= 0
        assert metrics['avg_throughput'] >= 0
        assert 0 <= metrics['failure_rate'] <= 1


@pytest.fixture
def test_helpers():
    """Fixture providing test helper functions"""
    return TestHelpers()
"""
Minimal Pytest Configuration
=============================

Simplified configuration that avoids complex imports while still providing
essential testing fixtures.
"""

import pytest
import asyncio
import sys
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

# Add PRSM to path for all tests
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def project_root():
    """Fixture providing the project root directory"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    config = {
        "test_mode": True,
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/15",
        "log_level": "DEBUG",
        "network_size": 5,
        "consensus_timeout": 5.0,
        "max_retries": 3
    }
    return config


@pytest.fixture
def temp_directory(tmp_path):
    """Temporary directory for file operations"""
    return tmp_path


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Auto-use fixture to configure logging for tests"""
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


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Auto-use fixture to set up test environment"""
    # Set environment variables for testing
    os.environ["PRSM_ENVIRONMENT"] = "test"
    os.environ["PRSM_LOG_LEVEL"] = "DEBUG"
    os.environ["PRSM_DATABASE_URL"] = "sqlite:///:memory:"
    
    yield
    
    # Cleanup environment
    test_env_vars = ["PRSM_ENVIRONMENT", "PRSM_LOG_LEVEL", "PRSM_DATABASE_URL"]
    for var in test_env_vars:
        os.environ.pop(var, None)


# Mock fixtures for when imports fail
@pytest.fixture
def sample_peer_nodes():
    """Sample peer nodes for testing"""
    peer_nodes = []
    for i in range(5):
        peer = Mock()
        peer.node_id = f"test_node_{i}"
        peer.peer_id = f"test_peer_{i}"
        peer.multiaddr = f"/ip4/127.0.0.1/tcp/{9000+i}"
        peer.reputation_score = 0.8
        peer.active = True
        peer_nodes.append(peer)
    return peer_nodes


@pytest.fixture
def mock_ftns_service():
    """Mock FTNS service for testing"""
    mock_service = Mock()
    mock_service.get_balance.return_value = Decimal("100.0")
    mock_service.transfer.return_value = True
    mock_service.create_transaction.return_value = {
        "transaction_id": "test_tx_123",
        "status": "confirmed",
        "amount": Decimal("10.0")
    }
    return mock_service


# Test helpers
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


@pytest.fixture
def test_helpers():
    """Fixture providing test helper functions"""
    return TestHelpers()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    markers = [
        "slow: marks tests as slow (may take several seconds)",
        "integration: marks tests as integration tests",
        "unit: marks tests as unit tests", 
        "performance: marks tests as performance/benchmark tests",
        "network: marks tests that require network simulation",
        "api: marks tests that test API endpoints"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)


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


@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Session-wide setup and cleanup"""
    print("\n🚀 Starting PRSM test session...")
    
    # Session setup
    yield
    
    # Session cleanup
    print("✅ PRSM test session completed.")
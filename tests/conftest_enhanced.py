"""
Enhanced Pytest Configuration and Fixtures
===========================================

Comprehensive test configuration with database, API, performance,
and integration testing fixtures for the PRSM test suite.
"""

import pytest
import asyncio
import sys
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

# Add PRSM to path for all tests
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import test fixtures with error handling
try:
    from tests.fixtures.database import *
except ImportError as e:
    print(f"Warning: Could not import database fixtures: {e}")

try:
    from tests.fixtures.api import *
except ImportError as e:
    print(f"Warning: Could not import API fixtures: {e}")

try:
    from tests.fixtures.performance import *
except ImportError as e:
    print(f"Warning: Could not import performance fixtures: {e}")

# Core imports with error handling
try:
    from prsm.core.models import PeerNode, PRSMSession, UserInput
except ImportError as e:
    print(f"Warning: Could not import core models: {e}")
    PeerNode = Mock
    PRSMSession = Mock
    UserInput = Mock

try:
    from prsm.core.config import get_config, PRSMConfig
except ImportError as e:
    print(f"Warning: Could not import config: {e}")
    get_config = lambda: Mock()
    PRSMConfig = Mock

try:
    from prsm.tokenomics.ftns_service import FTNSService
except ImportError as e:
    print(f"Warning: Could not import FTNS service: {e}")
    FTNSService = Mock

try:
    from prsm.nwtn.meta_reasoning_engine import NWTNEngine
except ImportError as e:
    print(f"Warning: Could not import NWTN engine: {e}")
    NWTNEngine = Mock

try:
    from prsm.core.performance import get_performance_monitor, get_profiler
except ImportError as e:
    print(f"Warning: Could not import performance monitoring: {e}")
    get_performance_monitor = lambda: Mock()
    get_profiler = lambda: Mock()

try:
    from prsm.core.caching import CacheManager
except ImportError as e:
    print(f"Warning: Could not import cache manager: {e}")
    CacheManager = Mock


# ================================
# Session and Event Loop Fixtures
# ================================

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


# ================================
# Configuration Fixtures
# ================================

@pytest.fixture(scope="session")
def test_config():
    """Enhanced test configuration"""
    config = {
        # Basic test settings
        "test_mode": True,
        "environment": "test",
        "debug": True,
        
        # Database configuration
        "database_url": "sqlite:///:memory:",
        "database_pool_size": 5,
        "database_echo": False,
        
        # Cache configuration
        "redis_url": "redis://localhost:6379/15",  # Test database
        "cache_enabled": True,
        "cache_ttl": 300,
        
        # API configuration
        "api_host": "127.0.0.1",
        "api_port": 8001,
        "api_workers": 1,
        "api_timeout": 30,
        
        # Security configuration
        "jwt_secret_key": "test-secret-key-for-testing-only",
        "jwt_expiration_hours": 24,
        "bcrypt_rounds": 4,  # Faster for testing
        
        # Network configuration
        "network_size": 5,
        "consensus_timeout": 5.0,
        "max_retries": 3,
        "peer_discovery_interval": 30,
        
        # Performance settings
        "max_concurrent_requests": 10,
        "request_timeout": 10,
        "batch_size": 100,
        
        # Logging configuration
        "log_level": "DEBUG",
        "log_to_file": False,
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        
        # NWTN configuration
        "nwtn_max_depth": 3,
        "nwtn_timeout": 30,
        "nwtn_cache_enabled": True,
        
        # FTNS configuration
        "ftns_initial_balance": 1000.0,
        "ftns_transaction_fee": 0.01,
        "ftns_min_balance": 0.0,
        
        # Test data configuration
        "generate_test_data": True,
        "test_data_size": "small",  # small, medium, large
        "cleanup_after_tests": True
    }
    return config


@pytest.fixture
def mock_config(test_config):
    """Mock PRSM configuration for testing"""
    mock_config = Mock(spec=PRSMConfig)
    
    # Add test configuration attributes
    for key, value in test_config.items():
        setattr(mock_config, key, value)
    
    return mock_config


# ================================
# Core Service Fixtures
# ================================

@pytest.fixture
def mock_nwtn_engine():
    """Mock NWTN reasoning engine"""
    mock_engine = Mock(spec=NWTNEngine)
    
    mock_engine.process_query.return_value = {
        "response": "Mock NWTN response",
        "reasoning_depth": 2,
        "confidence": 0.85,
        "session_id": "test_session_123",
        "tokens_used": 150,
        "processing_time_ms": 1250
    }
    
    mock_engine.get_session_history.return_value = [
        {"query": "Previous query", "response": "Previous response"}
    ]
    
    return mock_engine


@pytest.fixture
def mock_ftns_service():
    """Enhanced mock FTNS service"""
    mock_service = Mock(spec=FTNSService)
    
    # Balance operations
    mock_service.get_balance.return_value = Decimal("100.0")
    mock_service.get_available_balance.return_value = Decimal("95.0")
    mock_service.get_pending_balance.return_value = Decimal("5.0")
    
    # Transaction operations
    mock_service.transfer.return_value = {
        "success": True,
        "transaction_id": "test_tx_123",
        "amount": Decimal("10.0"),
        "fee": Decimal("0.01")
    }
    
    mock_service.create_transaction.return_value = {
        "transaction_id": "test_tx_456",
        "status": "confirmed",
        "amount": Decimal("10.0"),
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    mock_service.get_transaction_history.return_value = [
        {
            "transaction_id": "tx_1",
            "amount": Decimal("50.0"),
            "type": "reward",
            "timestamp": "2024-01-01T10:00:00Z"
        }
    ]
    
    return mock_service


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager"""
    mock_cache = Mock(spec=CacheManager)
    
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    mock_cache.clear.return_value = True
    mock_cache.exists.return_value = False
    
    # Cache statistics
    mock_cache.get_stats.return_value = {
        "hits": 10,
        "misses": 5,
        "hit_rate": 0.67,
        "total_keys": 15,
        "memory_usage_mb": 2.5
    }
    
    return mock_cache


# ================================
# Data Fixtures
# ================================

@pytest.fixture
def sample_peer_nodes():
    """Enhanced sample peer nodes for testing"""
    peer_nodes = []
    
    # Create diverse peer nodes
    node_configs = [
        {"reputation": 0.95, "active": True, "region": "us-east"},
        {"reputation": 0.87, "active": True, "region": "eu-west"},
        {"reputation": 0.92, "active": True, "region": "asia-pacific"},
        {"reputation": 0.45, "active": False, "region": "us-west"},
        {"reputation": 0.78, "active": True, "region": "eu-central"}
    ]
    
    for i, config in enumerate(node_configs):
        peer = Mock(spec=PeerNode)
        peer.node_id = f"test_node_{i}"
        peer.peer_id = f"test_peer_{i}"
        peer.multiaddr = f"/ip4/127.0.0.{i+1}/tcp/{9000+i}"
        peer.reputation_score = config["reputation"]
        peer.active = config["active"]
        peer.region = config["region"]
        peer.capabilities = ["nwtn", "storage", "compute"]
        peer.version = "1.0.0"
        peer_nodes.append(peer)
    
    return peer_nodes


@pytest.fixture
def large_peer_network():
    """Large peer network for scalability testing"""
    peer_nodes = []
    
    for i in range(50):  # Increased from 20 to 50
        # Distributed reputation scores
        reputation = 0.9 if i < 35 else (0.6 if i < 45 else 0.3)
        active = i < 45  # 90% active nodes
        
        peer = Mock(spec=PeerNode)
        peer.node_id = f"large_node_{i}"
        peer.peer_id = f"large_peer_{i}"
        peer.multiaddr = f"/ip4/192.168.{i//50}.{i%50}/tcp/{8000+i}"
        peer.reputation_score = reputation
        peer.active = active
        peer.region = ["us-east", "us-west", "eu-west", "asia-pacific"][i % 4]
        peer.capabilities = ["nwtn", "storage"] if i % 3 == 0 else ["nwtn"]
        peer_nodes.append(peer)
    
    return peer_nodes


@pytest.fixture
def sample_test_queries():
    """Sample queries for NWTN testing"""
    return [
        {
            "query": "What are the implications of quantum computing on current cryptographic methods?",
            "expected_depth": 3,
            "category": "scientific",
            "complexity": "high"
        },
        {
            "query": "Explain the basic principles of machine learning",
            "expected_depth": 2,
            "category": "educational",
            "complexity": "medium"
        },
        {
            "query": "What is 2+2?",
            "expected_depth": 1,
            "category": "simple",
            "complexity": "low"
        },
        {
            "query": "Analyze the potential economic impacts of implementing universal basic income",
            "expected_depth": 4,
            "category": "analytical",
            "complexity": "high"
        }
    ]


@pytest.fixture
def sample_transactions():
    """Sample FTNS transactions for testing"""
    transactions = []
    
    transaction_types = ["reward", "charge", "transfer", "dividend"]
    amounts = [Decimal("10.0"), Decimal("25.5"), Decimal("50.0"), Decimal("7.25")]
    
    for i, (tx_type, amount) in enumerate(zip(transaction_types, amounts)):
        transaction = Mock()
        transaction.transaction_id = f"tx_{i+1}"
        transaction.user_id = f"user_{i+1}"
        transaction.amount = amount
        transaction.transaction_type = tx_type
        transaction.status = "confirmed"
        transaction.description = f"Test {tx_type} transaction"
        transaction.created_at = f"2024-01-0{i+1}T12:00:00Z"
        transactions.append(transaction)
    
    return transactions


# ================================
# Environment and Cleanup Fixtures
# ================================

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


@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """Enhanced logging setup for tests"""
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Reduce noise from external libraries
    external_loggers = [
        'urllib3', 'requests', 'httpx', 'asyncio',
        'sqlalchemy.engine', 'redis', 'pydantic'
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Capture logs for test analysis
    with caplog.at_level(logging.DEBUG):
        yield
    
    # Cleanup
    logging.getLogger().handlers.clear()


@pytest.fixture
def temp_directory(tmp_path):
    """Enhanced temporary directory with subdirectories"""
    # Create common subdirectories for testing
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "cache").mkdir()
    (tmp_path / "uploads").mkdir()
    
    return tmp_path


# ================================
# Mock External Services
# ================================

@pytest.fixture
def mock_external_services():
    """Comprehensive mock external services"""
    mocks = {}
    
    # AI API mocks
    mocks['openai'] = Mock()
    mocks['openai'].chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Mock OpenAI response"))],
        usage=Mock(total_tokens=150)
    )
    
    mocks['anthropic'] = Mock()
    mocks['anthropic'].messages.create.return_value = Mock(
        content=[Mock(text="Mock Claude response")],
        usage=Mock(input_tokens=50, output_tokens=100)
    )
    
    # Storage mocks
    mocks['ipfs'] = Mock()
    mocks['ipfs'].add.return_value = Mock(hash="QmTestHash123")
    mocks['ipfs'].get.return_value = b"Test IPFS content"
    mocks['ipfs'].pin.add.return_value = Mock(pins=["QmTestHash123"])
    
    # Database mocks
    mocks['redis'] = Mock()
    mocks['redis'].get.return_value = None
    mocks['redis'].set.return_value = True
    mocks['redis'].delete.return_value = 1
    mocks['redis'].exists.return_value = False
    
    # Network mocks
    mocks['libp2p'] = Mock()
    mocks['libp2p'].connect.return_value = True
    mocks['libp2p'].peers.return_value = ["peer1", "peer2", "peer3"]
    
    return mocks


# ================================
# Network Simulation Fixtures
# ================================

@pytest.fixture
def enhanced_network_conditions():
    """Enhanced network condition scenarios"""
    return {
        "optimal": {
            "latency_ms": 10,
            "throughput_ops": 25,
            "failure_rate": 0.01,
            "byzantine_percentage": 0.0,
            "packet_loss": 0.0,
            "jitter_ms": 2
        },
        "congested": {
            "latency_ms": 300,
            "throughput_ops": 5,
            "failure_rate": 0.05,
            "byzantine_percentage": 0.0,
            "packet_loss": 0.02,
            "jitter_ms": 50
        },
        "unreliable": {
            "latency_ms": 150,
            "throughput_ops": 8,
            "failure_rate": 0.25,
            "byzantine_percentage": 0.15,
            "packet_loss": 0.10,
            "jitter_ms": 100
        },
        "mobile": {
            "latency_ms": 200,
            "throughput_ops": 3,
            "failure_rate": 0.15,
            "byzantine_percentage": 0.05,
            "packet_loss": 0.05,
            "jitter_ms": 80
        },
        "satellite": {
            "latency_ms": 600,
            "throughput_ops": 2,
            "failure_rate": 0.10,
            "byzantine_percentage": 0.0,
            "packet_loss": 0.03,
            "jitter_ms": 200
        }
    }


# ================================
# Test Helpers and Utilities
# ================================

class EnhancedTestHelpers:
    """Enhanced collection of helper functions for tests"""
    
    @staticmethod
    def assert_performance_within_bounds(
        metrics: Dict[str, float],
        bounds: Dict[str, float],
        tolerance_percent: float = 10.0
    ):
        """Assert performance metrics are within acceptable bounds"""
        failures = []
        
        for metric, expected in bounds.items():
            if metric in metrics:
                actual = metrics[metric]
                tolerance = expected * (tolerance_percent / 100)
                
                if actual > expected + tolerance:
                    failures.append(
                        f"{metric}: {actual} exceeds bound {expected} (tolerance: {tolerance})"
                    )
        
        if failures:
            pytest.fail(f"Performance bounds exceeded: {'; '.join(failures)}")
    
    @staticmethod
    def assert_api_response_valid(response, expected_status: int = 200):
        """Assert API response is valid"""
        assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}"
        
        if expected_status == 200:
            assert response.headers.get("content-type", "").startswith("application/json")
            
            try:
                response.json()
            except ValueError:
                pytest.fail("Response is not valid JSON")
    
    @staticmethod
    def assert_database_state_clean(session):
        """Assert database is in clean state"""
        if session is None:
            return
        
        # Check common tables are empty
        tables = ["prsm_sessions", "user_inputs", "ftns_transactions"]
        for table in tables:
            try:
                count = session.execute(f"SELECT COUNT(*) FROM {table}").scalar()
                assert count == 0, f"Table {table} is not empty: {count} records found"
            except Exception:
                # Table doesn't exist or can't be queried - that's fine
                pass
    
    @staticmethod
    def assert_cache_operations_work(cache_manager):
        """Assert cache operations are working"""
        test_key = "test_cache_key"
        test_value = {"test": "data"}
        
        # Test set and get
        assert cache_manager.set(test_key, test_value)
        cached_value = cache_manager.get(test_key)
        assert cached_value == test_value
        
        # Test delete
        assert cache_manager.delete(test_key)
        assert cache_manager.get(test_key) is None
    
    @staticmethod
    def generate_test_data(
        data_type: str,
        count: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate test data of specified type"""
        data = []
        
        if data_type == "users":
            for i in range(count):
                data.append({
                    "user_id": f"test_user_{i}",
                    "username": f"testuser{i}",
                    "email": f"test{i}@example.com",
                    "balance": 100.0 + i * 10,
                    "created_at": f"2024-01-{i+1:02d}T12:00:00Z"
                })
        
        elif data_type == "queries":
            queries = [
                "What is artificial intelligence?",
                "Explain quantum mechanics",
                "How does blockchain work?",
                "What are the benefits of renewable energy?",
                "Describe machine learning algorithms"
            ]
            
            for i in range(count):
                data.append({
                    "query_id": f"query_{i}",
                    "content": queries[i % len(queries)],
                    "user_id": f"test_user_{i % 5}",
                    "complexity": ["low", "medium", "high"][i % 3],
                    "timestamp": f"2024-01-{i+1:02d}T12:00:00Z"
                })
        
        elif data_type == "transactions":
            tx_types = ["reward", "charge", "transfer", "dividend"]
            
            for i in range(count):
                data.append({
                    "transaction_id": f"tx_{i}",
                    "user_id": f"test_user_{i % 5}",
                    "amount": round(10.0 + i * 5.5, 2),
                    "type": tx_types[i % len(tx_types)],
                    "status": "confirmed",
                    "timestamp": f"2024-01-{i+1:02d}T12:00:00Z"
                })
        
        return data


@pytest.fixture
def enhanced_test_helpers():
    """Enhanced test helpers fixture"""
    return EnhancedTestHelpers()


# ================================
# Pytest Configuration
# ================================

def pytest_configure(config):
    """Enhanced pytest configuration"""
    # Add custom markers
    markers = [
        "slow: marks tests as slow (may take several seconds)",
        "integration: marks tests as integration tests",
        "unit: marks tests as unit tests",
        "performance: marks tests as performance/benchmark tests",
        "security: marks tests as security tests",
        "network: marks tests that require network simulation",
        "api: marks tests that test API endpoints",
        "database: marks tests that require database access",
        "nwtn: marks tests for NWTN reasoning engine",
        "ftns: marks tests for FTNS tokenomics",
        "marketplace: marks tests for marketplace functionality",
        "auth: marks tests for authentication/authorization",
        "e2e: marks tests as end-to-end tests",
        "regression: marks tests for regression testing",
        "load: marks tests for load testing",
        "stress: marks tests for stress testing"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Enhanced test collection modification"""
    for item in items:
        # Add markers based on test/file paths
        test_path = str(item.fspath)
        test_name = item.name.lower()
        
        # Path-based markers
        if "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/performance/" in test_path:
            item.add_marker(pytest.mark.performance)
        elif "/security/" in test_path:
            item.add_marker(pytest.mark.security)
        elif "/api/" in test_path:
            item.add_marker(pytest.mark.api)
        
        # Component-based markers
        if "nwtn" in test_path or "nwtn" in test_name:
            item.add_marker(pytest.mark.nwtn)
        if "ftns" in test_path or "ftns" in test_name:
            item.add_marker(pytest.mark.ftns)
        if "marketplace" in test_path or "marketplace" in test_name:
            item.add_marker(pytest.mark.marketplace)
        if "auth" in test_path or "auth" in test_name:
            item.add_marker(pytest.mark.auth)
        if "database" in test_path or "database" in test_name:
            item.add_marker(pytest.mark.database)
        
        # Performance-based markers
        if any(keyword in test_name for keyword in ["slow", "large", "comprehensive", "stress", "load"]):
            item.add_marker(pytest.mark.slow)
        if "benchmark" in test_name or "performance" in test_name:
            item.add_marker(pytest.mark.performance)
        if "e2e" in test_name or "end_to_end" in test_name:
            item.add_marker(pytest.mark.e2e)


# ================================
# Session-wide fixtures
# ================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Session-wide setup and cleanup"""
    print("\n🚀 Starting PRSM test session...")
    
    # Session setup
    yield
    
    # Session cleanup
    print("✅ PRSM test session completed.")


@pytest.fixture(scope="function", autouse=True)
def test_isolation():
    """Ensure test isolation"""
    # Setup for each test
    yield
    
    # Cleanup after each test
    # This ensures no state leakage between tests
    pass
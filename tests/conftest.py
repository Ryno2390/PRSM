"""
Minimal Pytest Configuration
=============================

Simplified configuration that avoids complex imports while still providing
essential testing fixtures.
"""

import pytest
import pytest_asyncio
import asyncio
import sys
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from decimal import Decimal
from collections import defaultdict

# Add PRSM to path for all tests
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# MOCK EXTERNAL SERVICES - AUTO-USE FIXTURES
# ============================================================================

class FakeRedisPipeline:
    """Fake Redis pipeline for testing"""
    
    def __init__(self, redis):
        self.redis = redis
        self.commands = []
    
    def zadd(self, key, mapping):
        """Add to sorted set"""
        self.commands.append(('zadd', key, mapping))
        return self
    
    def zremrangebyscore(self, key, min_score, max_score):
        """Remove range by score"""
        self.commands.append(('zremrangebyscore', key, min_score, max_score))
        return self
    
    def zcard(self, key):
        """Get sorted set cardinality"""
        self.commands.append(('zcard', key))
        return self
    
    def expire(self, key, seconds):
        """Set key expiration"""
        self.commands.append(('expire', key, seconds))
        return self
    
    def set(self, key, value, ex=None):
        """Set value"""
        self.commands.append(('set', key, value, ex))
        return self
    
    def get(self, key):
        """Get value"""
        self.commands.append(('get', key))
        return self
    
    async def execute(self):
        """Execute all pipeline commands"""
        results = []
        for cmd in self.commands:
            if cmd[0] == 'zadd':
                results.append(len(cmd[2]))
            elif cmd[0] == 'zremrangebyscore':
                results.append(0)
            elif cmd[0] == 'zcard':
                results.append(0)
            elif cmd[0] == 'expire':
                results.append(True)
            elif cmd[0] == 'set':
                results.append(True)
            elif cmd[0] == 'get':
                results.append(None)
            else:
                results.append(True)
        self.commands = []
        return results


class FakeRedis:
    """In-memory fake Redis client for testing"""
    
    def __init__(self):
        self._data = {}
        self._expirations = {}
        self._sorted_sets = defaultdict(dict)
        
    async def get(self, key):
        """Get value from fake Redis"""
        return self._data.get(key)
    
    async def set(self, key, value, ex=None, nx=False, xx=False):
        """Set value in fake Redis"""
        self._data[key] = value
        if ex:
            self._expirations[key] = ex
        return True
    
    async def delete(self, *keys):
        """Delete keys from fake Redis"""
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count
    
    async def exists(self, key):
        """Check if key exists"""
        return 1 if key in self._data else 0
    
    async def keys(self, pattern="*"):
        """Get keys matching pattern"""
        return list(self._data.keys())
    
    async def zadd(self, key, mapping):
        """Add to sorted set"""
        if key not in self._sorted_sets:
            self._sorted_sets[key] = {}
        self._sorted_sets[key].update(mapping)
        return len(mapping)
    
    async def zremrangebyscore(self, key, min_score, max_score):
        """Remove range by score from sorted set"""
        if key not in self._sorted_sets:
            return 0
        removed = 0
        to_remove = []
        for member, score in self._sorted_sets[key].items():
            if min_score <= score <= max_score:
                to_remove.append(member)
                removed += 1
        for member in to_remove:
            del self._sorted_sets[key][member]
        return removed
    
    async def zcard(self, key):
        """Get sorted set cardinality"""
        return len(self._sorted_sets.get(key, {}))
    
    async def expire(self, key, seconds):
        """Set key expiration"""
        self._expirations[key] = seconds
        return True
    
    async def incr(self, key):
        """Increment key"""
        current = int(self._data.get(key, 0))
        self._data[key] = str(current + 1)
        return current + 1
    
    async def decr(self, key):
        """Decrement key"""
        current = int(self._data.get(key, 0))
        self._data[key] = str(current - 1)
        return current - 1
    
    async def lpush(self, key, *values):
        """Push to list (left)"""
        if key not in self._data:
            self._data[key] = []
        for value in values:
            self._data[key].insert(0, value)
        return len(self._data[key])
    
    async def rpush(self, key, *values):
        """Push to list (right)"""
        if key not in self._data:
            self._data[key] = []
        self._data[key].extend(values)
        return len(self._data[key])
    
    async def lrange(self, key, start, stop):
        """Get list range"""
        if key not in self._data:
            return []
        return self._data[key][start:stop+1] if stop >= 0 else self._data[key][start:]
    
    def pipeline(self):
        """Create a pipeline"""
        return FakeRedisPipeline(self)
    
    async def close(self):
        """Close connection (no-op for fake)"""
        pass
    
    async def ping(self):
        """Ping server"""
        return True
    
    def __await__(self):
        """Make FakeRedis awaitable"""
        async def _impl():
            return self
        return _impl().__await__()


class FakeAsyncPGConnection:
    """Fake asyncpg connection for testing"""
    
    def __init__(self):
        self._data = defaultdict(list)
        self._closed = False
        
    async def execute(self, query, *args):
        """Execute query"""
        return "SUCCESS"
    
    async def fetch(self, query, *args):
        """Fetch rows"""
        return []
    
    async def fetchrow(self, query, *args):
        """Fetch single row"""
        return None
    
    async def fetchval(self, query, *args):
        """Fetch single value"""
        return None
    
    async def close(self):
        """Close connection"""
        self._closed = True
    
    def transaction(self):
        """Create transaction context manager"""
        return FakeAsyncPGTransaction(self)


class FakeAsyncPGTransaction:
    """Fake asyncpg transaction"""
    
    def __init__(self, connection):
        self.connection = connection
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture(autouse=True)
def mock_redis():
    """Auto-use fixture to mock Redis connections"""
    fake_redis_instance = FakeRedis()
    
    # Mock redis.asyncio.Redis
    with patch('redis.asyncio.Redis') as mock_async_redis, \
         patch('redis.asyncio.from_url') as mock_async_from_url, \
         patch('redis.Redis') as mock_sync_redis, \
         patch('redis.from_url') as mock_sync_from_url:
        
        # Return fake Redis for all connection methods
        mock_async_redis.return_value = fake_redis_instance
        mock_async_from_url.return_value = fake_redis_instance
        mock_sync_redis.return_value = fake_redis_instance
        mock_sync_from_url.return_value = fake_redis_instance
        
        yield fake_redis_instance


@pytest.fixture(autouse=True)
def mock_asyncpg():
    """Auto-use fixture to mock asyncpg connections"""
    fake_conn = FakeAsyncPGConnection()
    
    async def fake_connect(*args, **kwargs):
        return fake_conn
    
    with patch('asyncpg.connect', side_effect=fake_connect), \
         patch('asyncpg.create_pool') as mock_pool:
        
        # Mock pool
        mock_pool_instance = AsyncMock()
        mock_pool_instance.acquire.return_value.__aenter__.return_value = fake_conn
        mock_pool_instance.close = AsyncMock()
        mock_pool.return_value = mock_pool_instance
        
        yield fake_conn


@pytest.fixture(scope="session")
def test_database_url():
    """Test database URL - in-memory SQLite"""
    return "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def test_sync_database_url():
    """Test database URL for sync operations - in-memory SQLite"""
    return "sqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def test_async_engine(test_database_url):
    """Create async test database engine"""
    try:
        from sqlalchemy import JSON
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.dialects.postgresql import JSONB
        from prsm.core.database import Base
        
        # Replace JSONB columns with JSON for SQLite compatibility
        for table in Base.metadata.tables.values():
            for column in table.columns:
                if isinstance(column.type, JSONB):
                    column.type = JSON()
        
        engine = create_async_engine(
            test_database_url,
            echo=False,
            connect_args={"check_same_thread": False}
        )
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        yield engine
        
        # Cleanup
        await engine.dispose()
    except ImportError:
        # If imports fail, provide a mock
        mock_engine = AsyncMock()
        yield mock_engine


@pytest.fixture(scope="session")
def test_sync_engine(test_sync_database_url):
    """Create sync test database engine"""
    try:
        from sqlalchemy import create_engine, JSON
        from sqlalchemy.pool import StaticPool
        from sqlalchemy.dialects.postgresql import JSONB
        from prsm.core.database import Base
        
        # Replace JSONB columns with JSON for SQLite compatibility
        for table in Base.metadata.tables.values():
            for column in table.columns:
                if isinstance(column.type, JSONB):
                    column.type = JSON()
        
        engine = create_engine(
            test_sync_database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        yield engine
        
        # Cleanup
        engine.dispose()
    except ImportError:
        # If imports fail, provide a mock
        mock_engine = Mock()
        yield mock_engine


@pytest_asyncio.fixture
async def test_async_session(test_async_engine):
    """Provide async database session with automatic rollback"""
    try:
        from sqlalchemy.ext.asyncio import AsyncSession
        
        async with AsyncSession(test_async_engine, expire_on_commit=False) as session:
            async with session.begin():
                yield session
                # Transaction will auto-rollback when exiting context
    except ImportError:
        # If imports fail, provide a mock
        mock_session = AsyncMock()
        yield mock_session


@pytest.fixture
def test_session(test_sync_engine):
    """Provide sync database session with automatic rollback"""
    try:
        from sqlalchemy.orm import Session
        
        with Session(test_sync_engine) as session:
            with session.begin():
                yield session
                # Transaction will auto-rollback when exiting context
    except ImportError:
        # If imports fail, provide a mock
        mock_session = Mock()
        yield mock_session


@pytest_asyncio.fixture
async def async_db_session(test_async_session):
    """Alias for test_async_session for compatibility"""
    return test_async_session


@pytest.fixture
def db_session(test_session):
    """Alias for test_session for compatibility"""
    return test_session


@pytest.fixture(autouse=True)
def mock_http_requests():
    """Auto-use fixture to mock HTTP requests (aiohttp, httpx, requests)"""
    
    # Mock aiohttp ClientSession
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"status": "ok"})
    mock_response.text = AsyncMock(return_value="OK")
    mock_response.read = AsyncMock(return_value=b"OK")
    
    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_session.put.return_value.__aenter__.return_value = mock_response
    mock_session.delete.return_value.__aenter__.return_value = mock_response
    mock_session.close = AsyncMock()
    
    # Mock httpx AsyncClient
    mock_httpx_response = MagicMock()
    mock_httpx_response.status_code = 200
    mock_httpx_response.json.return_value = {"status": "ok"}
    mock_httpx_response.text = "OK"
    mock_httpx_response.content = b"OK"
    
    mock_httpx_client = AsyncMock()
    mock_httpx_client.get = AsyncMock(return_value=mock_httpx_response)
    mock_httpx_client.post = AsyncMock(return_value=mock_httpx_response)
    mock_httpx_client.put = AsyncMock(return_value=mock_httpx_response)
    mock_httpx_client.delete = AsyncMock(return_value=mock_httpx_response)
    mock_httpx_client.aclose = AsyncMock()
    
    # Mock requests (synchronous)
    mock_sync_response = MagicMock()
    mock_sync_response.status_code = 200
    mock_sync_response.json.return_value = {"status": "ok"}
    mock_sync_response.text = "OK"
    mock_sync_response.content = b"OK"
    
    with patch('aiohttp.ClientSession', return_value=mock_session), \
         patch('httpx.AsyncClient', return_value=mock_httpx_client), \
         patch('httpx.Client') as mock_httpx_sync, \
         patch('requests.get', return_value=mock_sync_response), \
         patch('requests.post', return_value=mock_sync_response), \
         patch('requests.put', return_value=mock_sync_response), \
         patch('requests.delete', return_value=mock_sync_response):
        
        mock_httpx_sync.return_value.get.return_value = mock_httpx_response
        mock_httpx_sync.return_value.post.return_value = mock_httpx_response
        
        yield {
            'aiohttp': mock_session,
            'httpx': mock_httpx_client,
            'requests': mock_sync_response
        }


@pytest.fixture(scope="session", autouse=True)
def mock_external_connections_early():
    """Very early fixture to mock external connections before test collection"""
    # Mock subprocess calls that might try to connect
    # NOTE: We don't mock socket.socket as it breaks asyncio event loop initialization
    with patch('subprocess.run') as mock_run, \
         patch('subprocess.Popen') as mock_popen:
        
        mock_run.return_value = Mock(returncode=0, stdout=b'', stderr=b'')
        mock_popen.return_value = Mock(returncode=0, communicate=lambda: (b'', b''))
        
        yield


@pytest.fixture(autouse=True)
def mock_time_sleep():
    """Auto-use fixture to mock time.sleep in tests to prevent actual delays"""
    
    # Mock time.sleep to be instant
    def fake_sleep(seconds):
        """Fake sleep that doesn't actually sleep"""
        pass
    
    with patch('time.sleep', side_effect=fake_sleep):
        yield


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    """Auto-use fixture to mock asyncio.sleep to prevent actual delays"""
    
    # Import the original sleep function before patching
    import asyncio
    _original_sleep = asyncio.sleep
    
    async def fake_async_sleep(seconds):
        """Fake async sleep that doesn't actually sleep"""
        # Yield control to event loop but don't actually wait
        await _original_sleep(0)
    
    with patch('asyncio.sleep', side_effect=fake_async_sleep):
        yield


@pytest.fixture(autouse=True)
def mock_openai_clients():
    """Auto-use fixture to mock OpenAI and LLM clients"""
    
    # Mock OpenAI response
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Mocked LLM response"
    mock_completion.choices[0].text = "Mocked LLM response"
    mock_completion.usage = MagicMock()
    mock_completion.usage.total_tokens = 100
    
    mock_openai_client = AsyncMock()
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    mock_openai_client.completions.create = AsyncMock(return_value=mock_completion)
    
    # Mock Anthropic Claude
    mock_anthropic_response = MagicMock()
    mock_anthropic_response.content = [MagicMock()]
    mock_anthropic_response.content[0].text = "Mocked Claude response"
    
    mock_anthropic_client = AsyncMock()
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_anthropic_response)
    
    with patch('openai.AsyncOpenAI', return_value=mock_openai_client), \
         patch('openai.OpenAI') as mock_sync_openai, \
         patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client), \
         patch('anthropic.Anthropic') as mock_sync_anthropic:
        
        # Mock sync versions too
        mock_sync_openai.return_value.chat.completions.create.return_value = mock_completion
        mock_sync_anthropic.return_value.messages.create.return_value = mock_anthropic_response
        
        yield {
            'openai': mock_openai_client,
            'anthropic': mock_anthropic_client,
            'completion': mock_completion
        }


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
    os.environ["SKIP_REDIS_TESTS"] = "true"
    os.environ["SKIP_POSTGRES_TESTS"] = "true"
    os.environ["SKIP_INTEGRATION_TESTS"] = "true"
    
    # JWT/Auth configuration for tests
    os.environ["PRSM_JWT_SECRET"] = "test-secret-key-for-testing-only-minimum-32-chars-required-here"
    os.environ["PRSM_JWT_ALGORITHM"] = "HS256"
    os.environ["PRSM_SECRET_KEY"] = "test-secret-key-for-testing-only-minimum-32-chars-required-here"
    
    yield
    
    # Cleanup environment
    test_env_vars = [
        "PRSM_ENVIRONMENT", 
        "PRSM_LOG_LEVEL", 
        "PRSM_DATABASE_URL",
        "SKIP_REDIS_TESTS",
        "SKIP_POSTGRES_TESTS",
        "SKIP_INTEGRATION_TESTS",
        "PRSM_JWT_SECRET",
        "PRSM_JWT_ALGORITHM",
        "PRSM_SECRET_KEY"
    ]
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


# Database test factory
class DatabaseTestFactory:
    """Factory for creating test database objects"""
    
    @staticmethod
    def create_prsm_session(session_id=None, user_id="test_user", status="pending", **kwargs):
        """Create test PRSM session"""
        try:
            from prsm.core.database import PRSMSessionModel
            import uuid
            from datetime import datetime, timezone
            
            return PRSMSessionModel(
                session_id=session_id or uuid.uuid4(),
                user_id=user_id,
                status=status,
                created_at=datetime.now(timezone.utc),
                **kwargs
            )
        except ImportError:
            return Mock(session_id=session_id, user_id=user_id, status=status, **kwargs)
    
    @staticmethod
    def create_ftns_transaction(transaction_id=None, from_user=None, to_user=None, user_id=None, amount=10.0, transaction_type="reward", **kwargs):
        """Create test FTNS transaction"""
        try:
            from prsm.core.database import FTNSTransactionModel
            import uuid
            from datetime import datetime, timezone
            
            # Handle user_id alias for to_user
            if user_id and not to_user:
                to_user = user_id
            elif not to_user:
                to_user = "test_user"
            
            # Remove user_id from kwargs if present to avoid conflict
            kwargs.pop('user_id', None)
            
            return FTNSTransactionModel(
                transaction_id=transaction_id or uuid.uuid4(),
                from_user=from_user,
                to_user=to_user,
                amount=amount,
                transaction_type=transaction_type,
                description=kwargs.get('description', 'Test transaction'),
                created_at=datetime.now(timezone.utc),
                **{k: v for k, v in kwargs.items() if k != 'description'}
            )
        except ImportError:
            return Mock(transaction_id=transaction_id, to_user=to_user or user_id, amount=amount, **kwargs)
    
    @staticmethod
    def create_user_input(input_id=None, user_id="test_user", content="Test query", **kwargs):
        """Create test user input"""
        return Mock(input_id=input_id, user_id=user_id, content=content, **kwargs)


@pytest.fixture
def db_factory():
    """Database test factory fixture"""
    return DatabaseTestFactory()


@pytest.fixture(autouse=True)
def mock_jwt_handler_init():
    """Auto-use fixture to prevent JWT handler from initializing during test collection"""
    # Mock the JWT handler's initialize method to prevent async setup during tests
    # This prevents the JWT handler from trying to connect to real database/redis
    # Individual tests can still call initialize() if needed
    yield  # Just provide a placeholder - JWT handler already handles None settings gracefully


@pytest.fixture(autouse=True)
def mock_audit_logger():
    """Auto-use fixture to mock audit logger for all tests"""
    # Mock audit logger methods to prevent actual logging during tests
    mock_logger = AsyncMock()
    mock_logger.log_security_event = AsyncMock()
    mock_logger.log_auth_event = AsyncMock()  # Alias for compatibility
    mock_logger.log = AsyncMock()
    
    with patch('prsm.core.auth.auth_manager.audit_logger', mock_logger):
        yield mock_logger


@pytest.fixture
def performance_runner():
    """Performance test runner fixture"""
    class PerformanceMetrics:
        def __init__(self, execution_time_ms, error_rate, throughput_ops_per_sec=None):
            self.execution_time_ms = execution_time_ms
            self.error_rate = error_rate
            self.throughput_ops_per_sec = throughput_ops_per_sec or 0
    
    class PerformanceRunner:
        def __init__(self):
            self.results = []
        
        def run_performance_test(self, func, iterations=1, warmup_iterations=0):
            """Run performance test with multiple iterations"""
            import time
            
            # Warmup runs
            for _ in range(warmup_iterations):
                try:
                    func()
                except Exception:
                    pass
            
            # Actual test runs
            execution_times = []
            errors = 0
            
            for _ in range(iterations):
                start = time.time()
                try:
                    func()
                except Exception:
                    errors += 1
                elapsed = (time.time() - start) * 1000  # Convert to ms
                execution_times.append(elapsed)
            
            # Calculate metrics
            avg_time_ms = sum(execution_times) / len(execution_times) if execution_times else 0
            error_rate = errors / iterations if iterations > 0 else 0
            throughput = iterations / (sum(execution_times) / 1000) if sum(execution_times) > 0 else 0
            
            return PerformanceMetrics(
                execution_time_ms=avg_time_ms,
                error_rate=error_rate,
                throughput_ops_per_sec=throughput
            )
        
        def measure(self, func, *args, **kwargs):
            """Measure execution time of a function"""
            import time
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            self.results.append(elapsed)
            return result, elapsed
        
        async def measure_async(self, func, *args, **kwargs):
            """Measure execution time of an async function"""
            import time
            start = time.time()
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            self.results.append(elapsed)
            return result, elapsed
        
        def get_stats(self):
            """Get performance statistics"""
            if not self.results:
                return {"avg": 0, "min": 0, "max": 0, "total": 0}
            return {
                "avg": sum(self.results) / len(self.results),
                "min": min(self.results),
                "max": max(self.results),
                "total": sum(self.results),
                "count": len(self.results)
            }
    
    return PerformanceRunner()


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
"""
Security Tests Configuration
============================

Minimal pytest configuration for security tests that avoids
dependencies on modules that may not be available in all environments.
"""

import pytest
import pytest_asyncio
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Add PRSM to path for all tests
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# BASIC FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# OVERRIDE FIXTURES
# These fixtures override the parent conftest's autouse fixtures to prevent
# import errors from modules that may not be installed.
# =============================================================================

@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Override parent conftest's mock_external_dependencies."""
    yield


@pytest.fixture(autouse=True)
def mock_audit_logger():
    """Override parent conftest's mock_audit_logger."""
    mock_logger = AsyncMock()
    mock_logger.log_security_event = AsyncMock()
    mock_logger.log_auth_event = AsyncMock()
    mock_logger.log = AsyncMock()
    yield mock_logger


@pytest.fixture(autouse=True)
def mock_http_requests():
    """Override parent conftest's mock_http_requests."""
    yield None


@pytest.fixture(autouse=True)
def mock_openai_clients():
    """Override parent conftest's mock_openai_clients to prevent openai import errors."""
    yield None


@pytest.fixture(autouse=True)
def mock_redis():
    """Override parent conftest's mock_redis to prevent redis import errors."""
    yield None


@pytest.fixture(autouse=True)
def mock_asyncpg():
    """Override parent conftest's mock_asyncpg to prevent asyncpg import errors."""
    yield None


@pytest.fixture(autouse=True)
def mock_time_sleep():
    """Override parent conftest's mock_time_sleep."""
    yield None


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    """Override parent conftest's mock_asyncio_sleep."""
    yield None


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Override parent conftest's setup_test_logging."""
    yield None


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Override parent conftest's setup_test_environment."""
    yield None


@pytest.fixture(autouse=True)
def mock_jwt_handler_init():
    """Override parent conftest's mock_jwt_handler_init."""
    yield None


@pytest.fixture(scope="session", autouse=True)
def mock_external_connections_early():
    """Override parent conftest's mock_external_connections_early."""
    yield None


@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Override parent conftest's setup_test_session."""
    yield None

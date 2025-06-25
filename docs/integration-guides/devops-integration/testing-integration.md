# Testing Integration Guide

Integrate comprehensive testing frameworks and strategies into your PRSM development workflow including unit tests, integration tests, and end-to-end testing.

## 🎯 Overview

This guide covers setting up robust testing for PRSM including pytest for Python, Jest for JavaScript, automated testing pipelines, and testing best practices.

## 📋 Prerequisites

- PRSM development environment
- Testing frameworks installed
- Basic knowledge of testing concepts
- CI/CD pipeline setup

## 🚀 Python Testing with Pytest

### 1. Pytest Configuration

```ini
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    --strict-markers
    --strict-config
    --cov=prsm
    --cov-report=term-missing:skip-covered
    --cov-report=html:htmlcov
    --cov-report=xml
    --cov-fail-under=80
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    api: marks tests as API tests
    database: marks tests as database tests
    cache: marks tests as cache tests
    security: marks tests as security tests
    performance: marks tests as performance tests
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning
```

### 2. Test Configuration and Fixtures

```python
# tests/conftest.py
import pytest
import asyncio
import tempfile
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
import aioredis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from prsm.main import create_app
from prsm.core.database import get_database_session, Base
from prsm.core.cache import redis_cache
from prsm.core.config import Settings
from prsm.models.user import User
from prsm.models.query import Query

# Test settings
@pytest.fixture
def test_settings():
    """Test configuration settings."""
    return Settings(
        database_url="sqlite:///./test.db",
        redis_url="redis://localhost:6379/15",
        environment="test",
        secret_key="test-secret-key",
        debug=True
    )

# Database fixtures
@pytest.fixture
def test_db_engine(test_settings):
    """Create test database engine."""
    engine = create_engine(
        test_settings.database_url,
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_db_session(test_db_engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_db_engine
    )
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
async def test_redis():
    """Create test Redis connection."""
    redis = aioredis.from_url("redis://localhost:6379/15", decode_responses=True)
    yield redis
    await redis.flushdb()
    await redis.close()

# Application fixtures
@pytest.fixture
def test_app(test_settings, test_db_session, test_redis):
    """Create test FastAPI application."""
    app = create_app(test_settings)
    
    # Override dependencies
    app.dependency_overrides[get_database_session] = lambda: test_db_session
    
    return app

@pytest.fixture
def test_client(test_app):
    """Create test client."""
    return TestClient(test_app)

@pytest.fixture
async def async_test_client(test_app):
    """Create async test client."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client

# User fixtures
@pytest.fixture
def test_user(test_db_session):
    """Create test user."""
    user = User(
        id="test-user-123",
        email="test@example.com",
        username="testuser",
        is_active=True,
        is_verified=True
    )
    test_db_session.add(user)
    test_db_session.commit()
    test_db_session.refresh(user)
    return user

@pytest.fixture
def test_admin_user(test_db_session):
    """Create test admin user."""
    user = User(
        id="admin-user-123",
        email="admin@example.com",
        username="adminuser",
        is_active=True,
        is_verified=True,
        is_admin=True
    )
    test_db_session.add(user)
    test_db_session.commit()
    test_db_session.refresh(user)
    return user

# Authentication fixtures
@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers."""
    # Mock JWT token creation
    token = "test-jwt-token"
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def admin_auth_headers(test_admin_user):
    """Create admin authentication headers."""
    token = "admin-jwt-token"
    return {"Authorization": f"Bearer {token}"}

# Mock fixtures
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = AsyncMock()
    
    # Mock completion response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test AI response"))
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15
    )
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    mock_service = AsyncMock()
    mock_service.create_embedding.return_value = [0.1, 0.2, 0.3] * 128  # 384-dim vector
    return mock_service

@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    mock_store = AsyncMock()
    mock_store.search_similar_documents.return_value = [
        {
            "id": "doc1",
            "content": "Test document content",
            "score": 0.95,
            "metadata": {"type": "test"}
        }
    ]
    return mock_store

# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield
    # Clean up any test files
    test_files = ["test.db", "test.log"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
```

### 3. Unit Tests

```python
# tests/unit/test_query_processing.py
import pytest
from unittest.mock import AsyncMock, patch
from prsm.core.query_processor import QueryProcessor
from prsm.models.query import QueryRequest, QueryResponse
from prsm.core.config import Settings

class TestQueryProcessor:
    """Test query processing functionality."""
    
    @pytest.fixture
    def query_processor(self, test_settings, mock_openai_client):
        """Create query processor with mocked dependencies."""
        processor = QueryProcessor(test_settings)
        processor.openai_client = mock_openai_client
        return processor
    
    @pytest.mark.asyncio
    async def test_simple_query_processing(self, query_processor, mock_openai_client):
        """Test basic query processing."""
        request = QueryRequest(
            prompt="What is the meaning of life?",
            user_id="test-user",
            max_tokens=100
        )
        
        # Mock the OpenAI response
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "42"
        
        response = await query_processor.process_query(request)
        
        assert isinstance(response, QueryResponse)
        assert response.final_answer == "42"
        assert response.user_id == "test-user"
        assert response.token_usage > 0
        
        # Verify OpenAI was called correctly
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        assert "What is the meaning of life?" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_query_with_context(self, query_processor, mock_openai_client):
        """Test query processing with context."""
        request = QueryRequest(
            prompt="Continue this conversation",
            user_id="test-user",
            context="Previous: Hello, how are you?"
        )
        
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "I'm doing well!"
        
        response = await query_processor.process_query(request)
        
        assert response.final_answer == "I'm doing well!"
        
        # Verify context was included
        call_args = mock_openai_client.chat.completions.create.call_args
        assert "Previous: Hello, how are you?" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_query_error_handling(self, query_processor, mock_openai_client):
        """Test error handling in query processing."""
        request = QueryRequest(
            prompt="Test error",
            user_id="test-user"
        )
        
        # Mock an error
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception) as exc_info:
            await query_processor.process_query(request)
        
        assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_token_counting(self, query_processor, mock_openai_client):
        """Test token usage counting."""
        request = QueryRequest(
            prompt="Count my tokens",
            user_id="test-user"
        )
        
        # Mock token usage
        mock_openai_client.chat.completions.create.return_value.usage.total_tokens = 25
        
        response = await query_processor.process_query(request)
        
        assert response.token_usage == 25

# tests/unit/test_user_management.py
import pytest
from prsm.services.user_service import UserService
from prsm.models.user import User, UserCreate, UserUpdate
from prsm.core.exceptions import UserNotFoundError, UserAlreadyExistsError

class TestUserService:
    """Test user management functionality."""
    
    @pytest.fixture
    def user_service(self, test_db_session):
        """Create user service with test database."""
        return UserService(test_db_session)
    
    @pytest.mark.asyncio
    async def test_create_user(self, user_service):
        """Test user creation."""
        user_data = UserCreate(
            email="newuser@example.com",
            username="newuser",
            password="securepassword123"
        )
        
        user = await user_service.create_user(user_data)
        
        assert user.email == "newuser@example.com"
        assert user.username == "newuser"
        assert user.is_active is True
        assert user.password_hash is not None
        assert user.password_hash != "securepassword123"  # Should be hashed
    
    @pytest.mark.asyncio
    async def test_create_duplicate_user(self, user_service, test_user):
        """Test creating duplicate user raises error."""
        user_data = UserCreate(
            email=test_user.email,
            username="differentusername",
            password="password123"
        )
        
        with pytest.raises(UserAlreadyExistsError):
            await user_service.create_user(user_data)
    
    @pytest.mark.asyncio
    async def test_get_user_by_id(self, user_service, test_user):
        """Test getting user by ID."""
        user = await user_service.get_user_by_id(test_user.id)
        
        assert user.id == test_user.id
        assert user.email == test_user.email
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, user_service):
        """Test getting non-existent user raises error."""
        with pytest.raises(UserNotFoundError):
            await user_service.get_user_by_id("nonexistent-id")
    
    @pytest.mark.asyncio
    async def test_update_user(self, user_service, test_user):
        """Test updating user information."""
        update_data = UserUpdate(
            username="updatedusername",
            email="updated@example.com"
        )
        
        updated_user = await user_service.update_user(test_user.id, update_data)
        
        assert updated_user.username == "updatedusername"
        assert updated_user.email == "updated@example.com"
    
    @pytest.mark.asyncio
    async def test_deactivate_user(self, user_service, test_user):
        """Test deactivating user."""
        await user_service.deactivate_user(test_user.id)
        
        user = await user_service.get_user_by_id(test_user.id)
        assert user.is_active is False

# tests/unit/test_cache_service.py
import pytest
from unittest.mock import AsyncMock
from prsm.core.cache import CacheService

class TestCacheService:
    """Test cache functionality."""
    
    @pytest.fixture
    def cache_service(self, test_redis):
        """Create cache service with test Redis."""
        service = CacheService()
        service.redis = test_redis
        return service
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache_service):
        """Test basic cache operations."""
        await cache_service.set("test_key", "test_value")
        value = await cache_service.get("test_key")
        
        assert value == "test_value"
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_service):
        """Test cache expiration."""
        await cache_service.set("temp_key", "temp_value", ttl=1)
        
        # Immediate retrieval should work
        value = await cache_service.get("temp_key")
        assert value == "temp_value"
        
        # After expiration, should return None
        await asyncio.sleep(2)
        expired_value = await cache_service.get("temp_key")
        assert expired_value is None
    
    @pytest.mark.asyncio
    async def test_cache_delete(self, cache_service):
        """Test cache deletion."""
        await cache_service.set("delete_key", "delete_value")
        
        # Verify it exists
        value = await cache_service.get("delete_key")
        assert value == "delete_value"
        
        # Delete and verify
        await cache_service.delete("delete_key")
        deleted_value = await cache_service.get("delete_key")
        assert deleted_value is None
    
    @pytest.mark.asyncio
    async def test_cache_json_serialization(self, cache_service):
        """Test caching complex objects."""
        test_data = {
            "key1": "value1",
            "key2": ["item1", "item2"],
            "key3": {"nested": "object"}
        }
        
        await cache_service.set("json_key", test_data)
        retrieved_data = await cache_service.get("json_key")
        
        assert retrieved_data == test_data
```

### 4. Integration Tests

```python
# tests/integration/test_api_endpoints.py
import pytest
from fastapi import status

class TestAPIEndpoints:
    """Test API endpoint integration."""
    
    @pytest.mark.asyncio
    async def test_health_check(self, async_test_client):
        """Test health check endpoint."""
        response = await async_test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_query_endpoint(self, async_test_client, auth_headers, mock_openai_client):
        """Test query processing endpoint."""
        query_data = {
            "prompt": "What is AI?",
            "max_tokens": 100
        }
        
        response = await async_test_client.post(
            "/api/v1/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "final_answer" in data
        assert "query_id" in data
    
    @pytest.mark.asyncio
    async def test_query_without_auth(self, async_test_client):
        """Test query endpoint without authentication."""
        query_data = {
            "prompt": "Test without auth",
            "max_tokens": 50
        }
        
        response = await async_test_client.post(
            "/api/v1/query",
            json=query_data
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_user_registration(self, async_test_client):
        """Test user registration endpoint."""
        user_data = {
            "email": "newuser@test.com",
            "username": "newuser",
            "password": "SecurePassword123!"
        }
        
        response = await async_test_client.post(
            "/api/v1/auth/register",
            json=user_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert "password" not in data
    
    @pytest.mark.asyncio
    async def test_user_login(self, async_test_client, test_user):
        """Test user login endpoint."""
        login_data = {
            "email": test_user.email,
            "password": "testpassword"  # This should match the test user's password
        }
        
        response = await async_test_client.post(
            "/api/v1/auth/login",
            json=login_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
    
    @pytest.mark.asyncio
    async def test_get_user_profile(self, async_test_client, auth_headers):
        """Test getting user profile."""
        response = await async_test_client.get(
            "/api/v1/users/me",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert "email" in data
        assert "username" in data

# tests/integration/test_database_operations.py
import pytest
from sqlalchemy.orm import Session
from prsm.models.user import User
from prsm.models.query import Query
from prsm.repositories.user_repository import UserRepository
from prsm.repositories.query_repository import QueryRepository

class TestDatabaseOperations:
    """Test database integration."""
    
    @pytest.fixture
    def user_repository(self, test_db_session):
        return UserRepository(test_db_session)
    
    @pytest.fixture
    def query_repository(self, test_db_session):
        return QueryRepository(test_db_session)
    
    def test_user_crud_operations(self, user_repository):
        """Test user CRUD operations."""
        # Create
        user_data = {
            "email": "crud@test.com",
            "username": "cruduser",
            "password_hash": "hashed_password"
        }
        user = user_repository.create(user_data)
        assert user.id is not None
        
        # Read
        retrieved_user = user_repository.get_by_id(user.id)
        assert retrieved_user.email == user_data["email"]
        
        # Update
        updated_data = {"username": "updated_username"}
        updated_user = user_repository.update(user.id, updated_data)
        assert updated_user.username == "updated_username"
        
        # Delete
        user_repository.delete(user.id)
        deleted_user = user_repository.get_by_id(user.id)
        assert deleted_user is None
    
    def test_query_storage_and_retrieval(self, query_repository, test_user):
        """Test query storage and retrieval."""
        query_data = {
            "prompt": "Test query",
            "response": "Test response",
            "user_id": test_user.id,
            "model": "gpt-3.5-turbo",
            "token_usage": 100
        }
        
        # Store query
        query = query_repository.create(query_data)
        assert query.id is not None
        
        # Retrieve query
        retrieved_query = query_repository.get_by_id(query.id)
        assert retrieved_query.prompt == query_data["prompt"]
        assert retrieved_query.user_id == test_user.id
        
        # Get user queries
        user_queries = query_repository.get_by_user_id(test_user.id)
        assert len(user_queries) == 1
        assert user_queries[0].id == query.id
    
    def test_database_relationships(self, test_db_session, test_user):
        """Test database relationships."""
        # Create queries for the user
        query1 = Query(
            prompt="First query",
            response="First response",
            user_id=test_user.id,
            model="gpt-3.5-turbo"
        )
        query2 = Query(
            prompt="Second query",
            response="Second response",
            user_id=test_user.id,
            model="gpt-4"
        )
        
        test_db_session.add(query1)
        test_db_session.add(query2)
        test_db_session.commit()
        
        # Test relationship
        test_db_session.refresh(test_user)
        assert len(test_user.queries) == 2
        
        # Test cascade delete (if configured)
        test_db_session.delete(test_user)
        test_db_session.commit()
        
        # Queries should also be deleted
        remaining_queries = test_db_session.query(Query).filter(
            Query.user_id == test_user.id
        ).all()
        assert len(remaining_queries) == 0
```

### 5. Performance Tests

```python
# tests/performance/test_load_testing.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

class TestPerformance:
    """Performance testing for PRSM."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, async_test_client, auth_headers, mock_openai_client):
        """Test concurrent query processing."""
        num_requests = 50
        max_concurrent = 10
        
        async def make_query(client, headers):
            start_time = time.time()
            response = await client.post(
                "/api/v1/query",
                json={"prompt": "Performance test query", "max_tokens": 50},
                headers=headers
            )
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "duration": end_time - start_time
            }
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_query():
            async with semaphore:
                return await make_query(async_test_client, auth_headers)
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [limited_query() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["status_code"] == 200]
        durations = [r["duration"] for r in successful_requests]
        
        assert len(successful_requests) == num_requests
        assert statistics.mean(durations) < 5.0  # Average response time under 5s
        assert max(durations) < 10.0  # No request takes more than 10s
        assert total_time < 30.0  # All requests complete within 30s
        
        print(f"Performance Results:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful requests: {len(successful_requests)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average response time: {statistics.mean(durations):.2f}s")
        print(f"  Max response time: {max(durations):.2f}s")
        print(f"  Min response time: {min(durations):.2f}s")
        print(f"  Requests per second: {num_requests / total_time:.2f}")
    
    @pytest.mark.performance
    def test_database_query_performance(self, test_db_session):
        """Test database query performance."""
        from prsm.models.user import User
        
        # Create test users
        users = []
        for i in range(1000):
            user = User(
                email=f"user{i}@test.com",
                username=f"user{i}",
                password_hash="hashed_password"
            )
            users.append(user)
        
        # Bulk insert
        start_time = time.time()
        test_db_session.add_all(users)
        test_db_session.commit()
        insert_time = time.time() - start_time
        
        # Query performance
        start_time = time.time()
        retrieved_users = test_db_session.query(User).filter(
            User.email.like("%500%")
        ).all()
        query_time = time.time() - start_time
        
        # Assertions
        assert insert_time < 5.0  # Bulk insert should be fast
        assert query_time < 1.0   # Query should be fast
        assert len(retrieved_users) > 0
        
        print(f"Database Performance:")
        print(f"  Insert time (1000 users): {insert_time:.2f}s")
        print(f"  Query time: {query_time:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_performance(self, test_redis):
        """Test cache performance."""
        num_operations = 1000
        
        # Test SET performance
        start_time = time.time()
        for i in range(num_operations):
            await test_redis.set(f"perf_key_{i}", f"value_{i}")
        set_time = time.time() - start_time
        
        # Test GET performance
        start_time = time.time()
        for i in range(num_operations):
            value = await test_redis.get(f"perf_key_{i}")
            assert value == f"value_{i}"
        get_time = time.time() - start_time
        
        # Assertions
        assert set_time < 2.0  # 1000 SETs should complete quickly
        assert get_time < 1.0  # 1000 GETs should be very fast
        
        print(f"Cache Performance:")
        print(f"  SET operations ({num_operations}): {set_time:.2f}s")
        print(f"  GET operations ({num_operations}): {get_time:.2f}s")
        print(f"  SET ops/sec: {num_operations / set_time:.0f}")
        print(f"  GET ops/sec: {num_operations / get_time:.0f}")
```

## 🔍 JavaScript Testing with Jest

### 1. Jest Configuration

```javascript
// jest.config.js
module.exports = {
  testEnvironment: 'node',
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  testMatch: [
    '**/tests/**/*.test.js',
    '**/tests/**/*.spec.js'
  ],
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  testTimeout: 30000,
  verbose: true,
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/**/*.test.js',
    '!src/config/*.js'
  ],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1'
  }
};
```

### 2. Test Setup and Utilities

```javascript
// tests/setup.js
import { jest } from '@jest/globals';

// Global test setup
beforeAll(async () => {
  // Set test environment variables
  process.env.NODE_ENV = 'test';
  process.env.PRSM_API_URL = 'http://localhost:8000';
  process.env.PRSM_API_KEY = 'test-api-key';
});

afterAll(async () => {
  // Cleanup after all tests
});

beforeEach(() => {
  // Clear all mocks before each test
  jest.clearAllMocks();
});

// Global test utilities
global.createMockResponse = (data, status = 200) => ({
  data,
  status,
  statusText: status === 200 ? 'OK' : 'Error',
  headers: {},
  config: {}
});

global.createMockError = (message, status = 500) => {
  const error = new Error(message);
  error.response = {
    status,
    data: { error: message }
  };
  return error;
};
```

### 3. Frontend Component Tests

```javascript
// tests/components/ChatInterface.test.js
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest } from '@jest/globals';
import ChatInterface from '@/components/ChatInterface';
import { PRSMClient } from '@/services/PRSMClient';

// Mock the PRSM client
jest.mock('@/services/PRSMClient');

describe('ChatInterface', () => {
  let mockPRSMClient;
  
  beforeEach(() => {
    mockPRSMClient = {
      sendMessage: jest.fn(),
      streamMessage: jest.fn(),
      getHistory: jest.fn()
    };
    PRSMClient.mockImplementation(() => mockPRSMClient);
  });

  test('renders chat interface correctly', () => {
    render(<ChatInterface />);
    
    expect(screen.getByPlaceholderText('Type your message...')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
    expect(screen.getByText('PRSM Assistant')).toBeInTheDocument();
  });

  test('sends message when send button is clicked', async () => {
    mockPRSMClient.sendMessage.mockResolvedValue({
      id: 'msg-123',
      content: 'Hello! How can I help you?',
      timestamp: new Date().toISOString()
    });

    render(<ChatInterface />);
    
    const input = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByRole('button', { name: /send/i });
    
    await userEvent.type(input, 'Hello PRSM');
    fireEvent.click(sendButton);
    
    await waitFor(() => {
      expect(mockPRSMClient.sendMessage).toHaveBeenCalledWith('Hello PRSM');
    });
    
    expect(screen.getByText('Hello PRSM')).toBeInTheDocument();
    expect(screen.getByText('Hello! How can I help you?')).toBeInTheDocument();
  });

  test('handles API errors gracefully', async () => {
    mockPRSMClient.sendMessage.mockRejectedValue(
      createMockError('API Error', 500)
    );

    render(<ChatInterface />);
    
    const input = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByRole('button', { name: /send/i });
    
    await userEvent.type(input, 'Test message');
    fireEvent.click(sendButton);
    
    await waitFor(() => {
      expect(screen.getByText(/error.*occurred/i)).toBeInTheDocument();
    });
  });

  test('supports streaming responses', async () => {
    const mockStream = {
      async *[Symbol.asyncIterator]() {
        yield { content: 'Hello' };
        yield { content: ' there!' };
        yield { content: ' How can I help?' };
      }
    };
    
    mockPRSMClient.streamMessage.mockReturnValue(mockStream);

    render(<ChatInterface streamMode={true} />);
    
    const input = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByRole('button', { name: /send/i });
    
    await userEvent.type(input, 'Stream test');
    fireEvent.click(sendButton);
    
    await waitFor(() => {
      expect(screen.getByText('Hello there! How can I help?')).toBeInTheDocument();
    });
  });

  test('loads conversation history on mount', async () => {
    const mockHistory = [
      { id: '1', content: 'Previous message', sender: 'user' },
      { id: '2', content: 'Previous response', sender: 'assistant' }
    ];
    
    mockPRSMClient.getHistory.mockResolvedValue(mockHistory);

    render(<ChatInterface />);
    
    await waitFor(() => {
      expect(screen.getByText('Previous message')).toBeInTheDocument();
      expect(screen.getByText('Previous response')).toBeInTheDocument();
    });
  });
});
```

### 4. API Client Tests

```javascript
// tests/services/PRSMClient.test.js
import { jest } from '@jest/globals';
import axios from 'axios';
import { PRSMClient } from '@/services/PRSMClient';

jest.mock('axios');

describe('PRSMClient', () => {
  let client;
  
  beforeEach(() => {
    client = new PRSMClient({
      apiUrl: 'http://localhost:8000',
      apiKey: 'test-key'
    });
    
    // Reset axios mock
    axios.create.mockReturnValue(axios);
  });

  describe('sendMessage', () => {
    test('sends message successfully', async () => {
      const mockResponse = createMockResponse({
        query_id: 'query-123',
        final_answer: 'Test response',
        timestamp: '2024-01-01T00:00:00Z'
      });
      
      axios.post.mockResolvedValue(mockResponse);
      
      const result = await client.sendMessage('Test message');
      
      expect(axios.post).toHaveBeenCalledWith('/api/v1/query', {
        prompt: 'Test message',
        max_tokens: 2000,
        temperature: 0.7
      });
      
      expect(result).toEqual({
        id: 'query-123',
        content: 'Test response',
        timestamp: '2024-01-01T00:00:00Z'
      });
    });

    test('handles API errors', async () => {
      axios.post.mockRejectedValue(createMockError('Rate limit exceeded', 429));
      
      await expect(client.sendMessage('Test')).rejects.toThrow('Rate limit exceeded');
    });

    test('includes user context when provided', async () => {
      const mockResponse = createMockResponse({
        query_id: 'query-123',
        final_answer: 'Response with context'
      });
      
      axios.post.mockResolvedValue(mockResponse);
      
      await client.sendMessage('Test', {
        userId: 'user-123',
        context: 'Previous conversation'
      });
      
      expect(axios.post).toHaveBeenCalledWith('/api/v1/query', {
        prompt: 'Test',
        max_tokens: 2000,
        temperature: 0.7,
        user_id: 'user-123',
        context: 'Previous conversation'
      });
    });
  });

  describe('streamMessage', () => {
    test('handles streaming responses', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue('data: {"content": "Hello"}\n\n');
          controller.enqueue('data: {"content": " World"}\n\n');
          controller.enqueue('data: [DONE]\n\n');
          controller.close();
        }
      });
      
      axios.post.mockResolvedValue({
        data: mockStream,
        headers: { 'content-type': 'text/event-stream' }
      });
      
      const results = [];
      for await (const chunk of client.streamMessage('Test stream')) {
        results.push(chunk);
      }
      
      expect(results).toEqual([
        { content: 'Hello' },
        { content: ' World' }
      ]);
    });
  });

  describe('authentication', () => {
    test('includes API key in requests', async () => {
      const mockResponse = createMockResponse({ success: true });
      axios.post.mockResolvedValue(mockResponse);
      
      await client.sendMessage('Test');
      
      expect(axios.post).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(Object),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-key'
          })
        })
      );
    });

    test('handles authentication errors', async () => {
      axios.post.mockRejectedValue(createMockError('Unauthorized', 401));
      
      await expect(client.sendMessage('Test')).rejects.toThrow('Unauthorized');
    });
  });
});
```

### 5. End-to-End Tests

```javascript
// tests/e2e/chat-flow.test.js
import { test, expect } from '@playwright/test';

test.describe('PRSM Chat Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the application
    await page.goto('http://localhost:3000');
    
    // Login if required
    await page.fill('input[name="email"]', 'test@example.com');
    await page.fill('input[name="password"]', 'testpassword');
    await page.click('button[type="submit"]');
    
    // Wait for dashboard to load
    await page.waitForSelector('[data-testid="chat-interface"]');
  });

  test('user can send a message and receive a response', async ({ page }) => {
    // Type a message
    const messageInput = page.locator('input[placeholder*="Type your message"]');
    await messageInput.fill('What is artificial intelligence?');
    
    // Send the message
    await page.click('button:has-text("Send")');
    
    // Verify message appears in chat
    await expect(page.locator('text=What is artificial intelligence?')).toBeVisible();
    
    // Wait for response
    await page.waitForSelector('[data-testid="assistant-message"]', { timeout: 10000 });
    
    // Verify response is displayed
    const response = page.locator('[data-testid="assistant-message"]').last();
    await expect(response).toBeVisible();
    await expect(response).toContainText(/artificial intelligence|AI/i);
  });

  test('user can view conversation history', async ({ page }) => {
    // Send multiple messages
    const messages = [
      'Hello PRSM',
      'How are you?',
      'Tell me a joke'
    ];
    
    for (const message of messages) {
      await page.fill('input[placeholder*="Type your message"]', message);
      await page.click('button:has-text("Send")');
      await page.waitForSelector(`text=${message}`);
      await page.waitForSelector('[data-testid="assistant-message"]');
    }
    
    // Verify all messages are in history
    for (const message of messages) {
      await expect(page.locator(`text=${message}`)).toBeVisible();
    }
    
    // Count messages (user + assistant messages)
    const messageCount = await page.locator('[data-testid*="message"]').count();
    expect(messageCount).toBeGreaterThanOrEqual(messages.length * 2);
  });

  test('handles network errors gracefully', async ({ page }) => {
    // Intercept API calls and simulate network error
    await page.route('**/api/v1/query', route => {
      route.abort('internetdisconnected');
    });
    
    // Try to send a message
    await page.fill('input[placeholder*="Type your message"]', 'Test message');
    await page.click('button:has-text("Send")');
    
    // Verify error message is shown
    await expect(page.locator('text*=network error')).toBeVisible({ timeout: 5000 });
    
    // Verify retry functionality
    await page.click('button:has-text("Retry")');
    await expect(page.locator('text*=network error')).toBeVisible();
  });

  test('streaming responses work correctly', async ({ page }) => {
    // Enable streaming mode
    await page.check('input[type="checkbox"]:near(:text("Stream responses"))');
    
    // Send a message
    await page.fill('input[placeholder*="Type your message"]', 'Write a short story');
    await page.click('button:has-text("Send")');
    
    // Wait for streaming response to start
    await page.waitForSelector('[data-testid="streaming-indicator"]');
    
    // Wait for response to complete
    await page.waitForSelector('[data-testid="assistant-message"]:not(:has([data-testid="streaming-indicator"]))', { timeout: 30000 });
    
    // Verify response content
    const response = page.locator('[data-testid="assistant-message"]').last();
    const responseText = await response.textContent();
    expect(responseText.length).toBeGreaterThan(50); // Expect substantial response
  });
});
```

## 🔄 CI/CD Testing Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-python:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: prsm_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 prsm tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 prsm tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type check with mypy
      run: |
        mypy prsm
    
    - name: Security check with bandit
      run: |
        bandit -r prsm
    
    - name: Test with pytest
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/prsm_test
        REDIS_URL: redis://localhost:6379/15
        ENVIRONMENT: test
      run: |
        pytest tests/ -v --cov=prsm --cov-report=xml --cov-report=term
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: python
        name: python-tests

  test-javascript:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: ["16", "18", "20"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json
    
    - name: Install dependencies
      working-directory: ./frontend
      run: npm ci
    
    - name: Lint with ESLint
      working-directory: ./frontend
      run: npm run lint
    
    - name: Type check with TypeScript
      working-directory: ./frontend
      run: npm run type-check
    
    - name: Run tests
      working-directory: ./frontend
      run: npm run test -- --coverage --watchAll=false
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./frontend/coverage/lcov.info
        flags: javascript
        name: javascript-tests

  e2e-tests:
    runs-on: ubuntu-latest
    needs: [test-python, test-javascript]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
        cache: 'pip'
    
    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: "18"
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json
    
    - name: Install Python dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Install Node.js dependencies
      working-directory: ./frontend
      run: npm ci
    
    - name: Install Playwright
      working-directory: ./frontend
      run: npx playwright install --with-deps
    
    - name: Start backend services
      run: |
        docker-compose -f docker-compose.test.yml up -d
    
    - name: Build frontend
      working-directory: ./frontend
      run: npm run build
    
    - name: Start frontend
      working-directory: ./frontend
      run: npm run preview &
    
    - name: Wait for services
      run: |
        timeout 60 bash -c 'until curl -f http://localhost:8000/health; do sleep 2; done'
        timeout 60 bash -c 'until curl -f http://localhost:3000; do sleep 2; done'
    
    - name: Run E2E tests
      working-directory: ./frontend
      run: npx playwright test
    
    - name: Upload E2E test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: e2e-test-results
        path: frontend/test-results/
        retention-days: 30

  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

---

**Need help with testing integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).
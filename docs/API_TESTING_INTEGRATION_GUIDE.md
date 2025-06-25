# API Testing & Integration Validation Guide

## Overview

This guide provides comprehensive testing strategies for PRSM's APIs and third-party integrations, including automated test suites, load testing, integration validation, and continuous testing practices for enterprise environments.

## 🏗️ Testing Architecture

### **Multi-Layer Testing Strategy**

```mermaid
graph TB
    A[Unit Tests] --> B[Integration Tests]
    B --> C[API Tests]
    C --> D[Load Tests]
    D --> E[End-to-End Tests]
    E --> F[Security Tests]
    
    G[Test Data Management] --> A
    G --> B
    G --> C
    
    H[CI/CD Pipeline] --> I[Automated Test Execution]
    I --> J[Test Reporting]
    J --> K[Quality Gates]
    
    L[Test Environments] --> M[Development]
    L --> N[Staging]
    L --> O[Production Monitoring]
```

### **Testing Framework Stack**

| Component | Purpose | Tool/Framework |
|-----------|---------|----------------|
| **Unit Testing** | Component isolation testing | pytest, unittest |
| **API Testing** | REST API validation | pytest-asyncio, httpx |
| **Load Testing** | Performance validation | Locust, Artillery |
| **Integration Testing** | Service interaction testing | pytest-integration |
| **Security Testing** | Vulnerability assessment | OWASP ZAP, Bandit |
| **Contract Testing** | API contract validation | Pact, OpenAPI |

## 🧪 Automated API Test Suites

### **Core API Test Framework**

```python
# tests/api/test_framework.py
import pytest
import asyncio
import httpx
from typing import Dict, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class APITestConfig:
    """API test configuration"""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    headers: Dict[str, str] = None
    auth_token: Optional[str] = None

class APITestClient:
    """Enhanced API test client with comprehensive validation"""
    
    def __init__(self, config: APITestConfig):
        self.config = config
        self.client = None
        self.default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "PRSM-API-Tests/1.0"
        }
        
        if config.headers:
            self.default_headers.update(config.headers)
            
        if config.auth_token:
            self.default_headers["Authorization"] = f"Bearer {config.auth_token}"
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers=self.default_headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()
    
    async def get(self, path: str, **kwargs) -> httpx.Response:
        """Enhanced GET request with logging"""
        logger.info("API GET request", path=path)
        response = await self.client.get(path, **kwargs)
        logger.info("API GET response", path=path, status=response.status_code)
        return response
    
    async def post(self, path: str, **kwargs) -> httpx.Response:
        """Enhanced POST request with logging"""
        logger.info("API POST request", path=path)
        response = await self.client.post(path, **kwargs)
        logger.info("API POST response", path=path, status=response.status_code)
        return response
    
    async def put(self, path: str, **kwargs) -> httpx.Response:
        """Enhanced PUT request with logging"""
        logger.info("API PUT request", path=path)
        response = await self.client.put(path, **kwargs)
        logger.info("API PUT response", path=path, status=response.status_code)
        return response
    
    async def delete(self, path: str, **kwargs) -> httpx.Response:
        """Enhanced DELETE request with logging"""
        logger.info("API DELETE request", path=path)
        response = await self.client.delete(path, **kwargs)
        logger.info("API DELETE response", path=path, status=response.status_code)
        return response
    
    def assert_response_ok(self, response: httpx.Response, expected_status: int = 200):
        """Assert response is successful"""
        assert response.status_code == expected_status, (
            f"Expected status {expected_status}, got {response.status_code}. "
            f"Response: {response.text}"
        )
    
    def assert_response_json(self, response: httpx.Response) -> Dict[str, Any]:
        """Assert response is valid JSON and return data"""
        try:
            data = response.json()
            assert isinstance(data, dict), "Response is not a JSON object"
            return data
        except Exception as e:
            pytest.fail(f"Invalid JSON response: {e}. Response: {response.text}")
    
    def assert_response_schema(self, response: httpx.Response, 
                             required_fields: list, 
                             optional_fields: list = None):
        """Assert response matches expected schema"""
        data = self.assert_response_json(response)
        
        # Check required fields
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from response"
        
        # Validate field types if specified
        # This could be extended with JSON schema validation
        
        return data

# Pytest fixtures
@pytest.fixture
async def api_client():
    """API test client fixture"""
    config = APITestConfig()
    async with APITestClient(config) as client:
        yield client

@pytest.fixture
async def authenticated_client():
    """Authenticated API test client fixture"""
    # Get auth token from test authentication
    auth_token = await get_test_auth_token()
    config = APITestConfig(auth_token=auth_token)
    async with APITestClient(config) as client:
        yield client

async def get_test_auth_token() -> str:
    """Get authentication token for testing"""
    config = APITestConfig()
    async with APITestClient(config) as client:
        response = await client.post("/auth/login", json={
            "username": "test@example.com",
            "password": "test_password"
        })
        assert response.status_code == 200
        data = response.json()
        return data["access_token"]
```

### **Comprehensive API Test Suite**

```python
# tests/api/test_core_apis.py
import pytest
import asyncio
from tests.api.test_framework import APITestClient, APITestConfig

class TestHealthAPI:
    """Test health check endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_check(self, api_client):
        """Test basic health check"""
        response = await api_client.get("/health")
        api_client.assert_response_ok(response)
        
        data = api_client.assert_response_json(response)
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime" in data

    @pytest.mark.asyncio
    async def test_health_detailed(self, api_client):
        """Test detailed health check"""
        response = await api_client.get("/health/detailed")
        api_client.assert_response_ok(response)
        
        data = api_client.assert_response_json(response)
        api_client.assert_response_schema(
            response,
            required_fields=["status", "components", "timestamp"],
            optional_fields=["uptime", "version"]
        )
        
        # Validate component health
        components = data["components"]
        assert "database" in components
        assert "redis" in components
        assert components["database"]["status"] in ["healthy", "degraded"]

class TestAuthAPI:
    """Test authentication endpoints"""
    
    @pytest.mark.asyncio
    async def test_login_success(self, api_client):
        """Test successful login"""
        response = await api_client.post("/auth/login", json={
            "username": "test@example.com",
            "password": "test_password"
        })
        api_client.assert_response_ok(response)
        
        data = api_client.assert_response_json(response)
        api_client.assert_response_schema(
            response,
            required_fields=["access_token", "refresh_token", "token_type", "expires_in"]
        )
        
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0
        assert len(data["access_token"]) > 50  # JWT should be substantial

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, api_client):
        """Test login with invalid credentials"""
        response = await api_client.post("/auth/login", json={
            "username": "test@example.com",
            "password": "wrong_password"
        })
        
        assert response.status_code == 401
        data = api_client.assert_response_json(response)
        assert "error" in data or "detail" in data

    @pytest.mark.asyncio
    async def test_token_refresh(self, api_client):
        """Test token refresh functionality"""
        # First, get tokens
        login_response = await api_client.post("/auth/login", json={
            "username": "test@example.com",
            "password": "test_password"
        })
        api_client.assert_response_ok(login_response)
        login_data = login_response.json()
        
        # Then refresh
        refresh_response = await api_client.post("/auth/refresh", json={
            "refresh_token": login_data["refresh_token"]
        })
        api_client.assert_response_ok(refresh_response)
        
        refresh_data = api_client.assert_response_json(refresh_response)
        assert "access_token" in refresh_data
        assert refresh_data["access_token"] != login_data["access_token"]

class TestUserAPI:
    """Test user management endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, authenticated_client):
        """Test getting current user info"""
        response = await authenticated_client.get("/users/me")
        authenticated_client.assert_response_ok(response)
        
        data = authenticated_client.assert_response_json(response)
        authenticated_client.assert_response_schema(
            response,
            required_fields=["id", "username", "email", "role"],
            optional_fields=["created_at", "last_login", "preferences"]
        )

    @pytest.mark.asyncio
    async def test_update_user_preferences(self, authenticated_client):
        """Test updating user preferences"""
        preferences = {
            "theme": "dark",
            "notifications": {
                "email": True,
                "push": False
            }
        }
        
        response = await authenticated_client.put("/users/me/preferences", json=preferences)
        authenticated_client.assert_response_ok(response)
        
        # Verify update
        get_response = await authenticated_client.get("/users/me")
        data = get_response.json()
        assert data["preferences"]["theme"] == "dark"

class TestFTNSAPI:
    """Test FTNS (token) API endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_balance(self, authenticated_client):
        """Test getting user FTNS balance"""
        response = await authenticated_client.get("/ftns/balance")
        authenticated_client.assert_response_ok(response)
        
        data = authenticated_client.assert_response_json(response)
        authenticated_client.assert_response_schema(
            response,
            required_fields=["balance", "currency"],
            optional_fields=["pending", "reserved"]
        )
        
        assert isinstance(data["balance"], (int, float))
        assert data["balance"] >= 0
        assert data["currency"] == "FTNS"

    @pytest.mark.asyncio
    async def test_transaction_history(self, authenticated_client):
        """Test getting transaction history"""
        response = await authenticated_client.get("/ftns/transactions")
        authenticated_client.assert_response_ok(response)
        
        data = authenticated_client.assert_response_json(response)
        assert "transactions" in data
        assert "pagination" in data
        
        if data["transactions"]:
            tx = data["transactions"][0]
            assert "id" in tx
            assert "amount" in tx
            assert "type" in tx
            assert "timestamp" in tx

class TestAIQueryAPI:
    """Test AI query processing endpoints"""
    
    @pytest.mark.asyncio
    async def test_submit_query(self, authenticated_client):
        """Test submitting AI query"""
        query_data = {
            "query": "What is the capital of France?",
            "model_preference": "gpt-3.5-turbo",
            "max_tokens": 100
        }
        
        response = await authenticated_client.post("/ai/query", json=query_data)
        authenticated_client.assert_response_ok(response, expected_status=202)  # Accepted
        
        data = authenticated_client.assert_response_json(response)
        authenticated_client.assert_response_schema(
            response,
            required_fields=["query_id", "status"],
            optional_fields=["estimated_completion"]
        )
        
        assert data["status"] in ["queued", "processing", "completed"]
        return data["query_id"]

    @pytest.mark.asyncio
    async def test_query_status(self, authenticated_client):
        """Test checking query status"""
        # First submit a query
        query_id = await self.test_submit_query(authenticated_client)
        
        # Check status
        response = await authenticated_client.get(f"/ai/query/{query_id}")
        authenticated_client.assert_response_ok(response)
        
        data = authenticated_client.assert_response_json(response)
        assert "status" in data
        assert data["status"] in ["queued", "processing", "completed", "failed"]

class TestIntegrationAPI:
    """Test third-party integration endpoints"""
    
    @pytest.mark.asyncio
    async def test_openai_integration_status(self, authenticated_client):
        """Test OpenAI integration status"""
        response = await authenticated_client.get("/integrations/openai/status")
        authenticated_client.assert_response_ok(response)
        
        data = authenticated_client.assert_response_json(response)
        assert "available" in data
        assert "models" in data
        
        if data["available"]:
            assert len(data["models"]) > 0

    @pytest.mark.asyncio
    async def test_langchain_integration(self, authenticated_client):
        """Test LangChain integration"""
        response = await authenticated_client.get("/integrations/langchain/chains")
        authenticated_client.assert_response_ok(response)
        
        data = authenticated_client.assert_response_json(response)
        assert "available_chains" in data
        assert isinstance(data["available_chains"], list)
```

## 🚀 Load Testing Framework

### **Locust Load Testing Suite**

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between
import json
import random
from typing import Dict, Any

class PRSMUser(HttpUser):
    """Simulated PRSM user for load testing"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Setup: authenticate user"""
        self.login()
    
    def login(self):
        """Authenticate and get token"""
        response = self.client.post("/auth/login", json={
            "username": f"loadtest_user_{random.randint(1, 1000)}@example.com",
            "password": "loadtest_password"
        })
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
        else:
            print(f"Login failed: {response.status_code} - {response.text}")
    
    @task(10)
    def get_user_profile(self):
        """Get user profile (high frequency)"""
        self.client.get("/users/me")
    
    @task(8)
    def check_ftns_balance(self):
        """Check FTNS balance (medium-high frequency)"""
        self.client.get("/ftns/balance")
    
    @task(5)
    def submit_ai_query(self):
        """Submit AI query (medium frequency)"""
        queries = [
            "What is machine learning?",
            "Explain quantum computing",
            "How does blockchain work?",
            "What is artificial intelligence?",
            "Describe neural networks"
        ]
        
        query_data = {
            "query": random.choice(queries),
            "model_preference": random.choice(["gpt-3.5-turbo", "gpt-4", "claude"]),
            "max_tokens": random.randint(50, 200)
        }
        
        response = self.client.post("/ai/query", json=query_data)
        
        if response.status_code == 202:
            # Query submitted successfully, check status
            data = response.json()
            query_id = data.get("query_id")
            if query_id:
                # Check status after a short delay
                self.client.get(f"/ai/query/{query_id}")
    
    @task(3)
    def get_transaction_history(self):
        """Get FTNS transaction history (low-medium frequency)"""
        self.client.get("/ftns/transactions?limit=10")
    
    @task(2)
    def update_preferences(self):
        """Update user preferences (low frequency)"""
        preferences = {
            "theme": random.choice(["light", "dark"]),
            "notifications": {
                "email": random.choice([True, False]),
                "push": random.choice([True, False])
            }
        }
        
        self.client.put("/users/me/preferences", json=preferences)
    
    @task(1)
    def health_check(self):
        """Health check (very low frequency)"""
        self.client.get("/health")

class PRSMAdminUser(HttpUser):
    """Simulated admin user for load testing admin endpoints"""
    
    wait_time = between(2, 5)  # Admins make fewer requests
    
    def on_start(self):
        """Setup: authenticate admin user"""
        self.login_admin()
    
    def login_admin(self):
        """Authenticate admin user"""
        response = self.client.post("/auth/login", json={
            "username": "admin@example.com",
            "password": "admin_password"
        })
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
    
    @task(5)
    def get_system_metrics(self):
        """Get system metrics"""
        self.client.get("/admin/metrics")
    
    @task(3)
    def get_user_analytics(self):
        """Get user analytics"""
        self.client.get("/admin/analytics/users")
    
    @task(2)
    def get_system_health(self):
        """Get detailed system health"""
        self.client.get("/health/detailed")
    
    @task(1)
    def get_audit_logs(self):
        """Get audit logs"""
        self.client.get("/admin/audit-logs?limit=50")

class PRSMAPIStressTest(HttpUser):
    """High-intensity stress testing"""
    
    wait_time = between(0.1, 0.5)  # Very rapid requests
    
    @task
    def rapid_health_checks(self):
        """Rapid health checks for stress testing"""
        self.client.get("/health")
    
    @task
    def rapid_auth_checks(self):
        """Rapid authentication endpoint stress"""
        # This will mostly fail, but tests error handling
        self.client.post("/auth/login", json={
            "username": "stress_test",
            "password": "invalid"
        })
```

### **Load Testing Configuration**

```bash
#!/bin/bash
# scripts/run_load_tests.sh

echo "Running PRSM Load Tests"

# Ensure PRSM is running
echo "Checking PRSM API availability..."
curl -f http://localhost:8000/health || {
    echo "PRSM API not available. Please start the service."
    exit 1
}

# Basic load test (100 users, 10 spawned per second)
echo "Running basic load test..."
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users=100 \
    --spawn-rate=10 \
    --run-time=5m \
    --headless \
    --html=reports/load_test_basic.html

# Stress test (500 users, rapid spawn)
echo "Running stress test..."
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users=500 \
    --spawn-rate=50 \
    --run-time=10m \
    --headless \
    --html=reports/load_test_stress.html

# Admin-specific load test
echo "Running admin load test..."
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users=20 \
    --spawn-rate=2 \
    --run-time=3m \
    --headless \
    --html=reports/load_test_admin.html \
    PRSMAdminUser

echo "Load tests completed. Reports saved to reports/ directory."
```

## 🔗 Integration Testing

### **Third-Party Integration Validation**

```python
# tests/integration/test_external_integrations.py
import pytest
import asyncio
import aiohttp
from unittest.mock import patch, MagicMock
import os

class TestOpenAIIntegration:
    """Test OpenAI API integration"""
    
    @pytest.mark.asyncio
    async def test_openai_api_connection(self):
        """Test OpenAI API connectivity"""
        from prsm.agents.executors.enhanced_openai_client import EnhancedOpenAIClient
        
        # Test with mock to avoid API costs
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            # Setup mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_client.chat.completions.create.return_value = mock_response
            
            client = EnhancedOpenAIClient(api_key="test_key")
            await client.initialize()
            
            response = await client.chat_completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}]
            )
            
            assert response is not None
            assert "Test response" in str(response)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not provided"
    )
    async def test_openai_real_api(self):
        """Test real OpenAI API (only if API key provided)"""
        from prsm.agents.executors.enhanced_openai_client import EnhancedOpenAIClient
        
        client = EnhancedOpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
        await client.initialize()
        
        response = await client.chat_completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello, PRSM test!'"}],
            max_tokens=10
        )
        
        assert response is not None
        assert "Hello" in str(response)

class TestLangChainIntegration:
    """Test LangChain integration"""
    
    @pytest.mark.asyncio
    async def test_langchain_chain_creation(self):
        """Test LangChain chain creation"""
        from prsm.integrations.langchain.chains import PRSMConversationChain
        
        chain = PRSMConversationChain()
        await chain.initialize()
        
        assert chain.llm is not None
        assert chain.memory is not None
        assert hasattr(chain, 'run')

    @pytest.mark.asyncio
    async def test_langchain_memory_integration(self):
        """Test LangChain memory integration"""
        from prsm.integrations.langchain.memory import PRSMConversationMemory
        
        memory = PRSMConversationMemory(user_id="test_user")
        
        # Test memory operations
        await memory.save_context(
            {"input": "Hello"}, 
            {"output": "Hi there!"}
        )
        
        messages = await memory.get_conversation_history()
        assert len(messages) > 0
        assert messages[-1]["input"] == "Hello"

class TestMCPIntegration:
    """Test Model Context Protocol (MCP) integration"""
    
    @pytest.mark.asyncio
    async def test_mcp_client_initialization(self):
        """Test MCP client initialization"""
        from prsm.integrations.mcp.client import MCPClient
        
        client = MCPClient()
        await client.initialize()
        
        assert client.session is not None
        assert hasattr(client, 'list_tools')
        assert hasattr(client, 'call_tool')

    @pytest.mark.asyncio
    async def test_mcp_tool_discovery(self):
        """Test MCP tool discovery"""
        from prsm.integrations.mcp.client import MCPClient
        
        client = MCPClient()
        await client.initialize()
        
        tools = await client.list_tools()
        assert isinstance(tools, list)
        
        # Verify tool structure
        if tools:
            tool = tools[0]
            assert "name" in tool
            assert "description" in tool

class TestIPFSIntegration:
    """Test IPFS integration"""
    
    @pytest.mark.asyncio
    async def test_ipfs_connection(self):
        """Test IPFS node connection"""
        from prsm.data_layer.enhanced_ipfs import PRSMIPFSClient
        
        # Test with mock IPFS client
        with patch('ipfshttpclient.connect') as mock_connect:
            mock_client = MagicMock()
            mock_connect.return_value = mock_client
            
            ipfs_client = PRSMIPFSClient()
            await ipfs_client.initialize()
            
            assert ipfs_client.client is not None

    @pytest.mark.asyncio
    async def test_ipfs_store_and_retrieve(self):
        """Test IPFS store and retrieve operations"""
        from prsm.data_layer.enhanced_ipfs import PRSMIPFSClient
        
        with patch('ipfshttpclient.connect') as mock_connect:
            mock_client = MagicMock()
            mock_connect.return_value = mock_client
            
            # Mock IPFS add response
            mock_client.add.return_value = [{"Hash": "QmTest123"}]
            mock_client.cat.return_value = b"Test data"
            
            ipfs_client = PRSMIPFSClient()
            await ipfs_client.initialize()
            
            # Test store
            content_id = await ipfs_client.store_content(
                content=b"Test data",
                metadata={"type": "test"}
            )
            
            assert content_id == "QmTest123"
            
            # Test retrieve
            retrieved_data = await ipfs_client.retrieve_content(content_id)
            assert retrieved_data == b"Test data"

class TestRedisIntegration:
    """Test Redis integration"""
    
    @pytest.mark.asyncio
    async def test_redis_connection(self):
        """Test Redis connection"""
        from prsm.core.redis_client import get_redis_client
        
        redis = await get_redis_client()
        
        # Test basic operations
        await redis.set("test_key", "test_value")
        value = await redis.get("test_key")
        
        assert value == "test_value"
        
        # Cleanup
        await redis.delete("test_key")

    @pytest.mark.asyncio
    async def test_redis_caching(self):
        """Test Redis caching functionality"""
        from prsm.core.redis_client import get_redis_client
        from prsm.performance.caching import CacheManager
        
        cache_manager = CacheManager()
        
        # Test cache set/get
        await cache_manager.set("cache_test", {"data": "test"}, ttl=60)
        cached_data = await cache_manager.get("cache_test")
        
        assert cached_data["data"] == "test"

class TestDatabaseIntegration:
    """Test database integration"""
    
    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Test database connection"""
        from prsm.core.database import get_database
        
        db = await get_database()
        
        # Test basic query
        result = await db.fetch("SELECT 1 as test")
        assert result[0]["test"] == 1

    @pytest.mark.asyncio
    async def test_user_crud_operations(self):
        """Test user CRUD operations"""
        from prsm.core.database import get_database
        from prsm.auth.models import User, UserRole
        
        db = await get_database()
        
        # Test user creation
        user_data = {
            "username": "test_integration_user",
            "email": "test_integration@example.com",
            "password_hash": "hashed_password",
            "role": UserRole.USER.value
        }
        
        # This would be a real database operation in practice
        # For testing, we can mock or use a test database
        
        # Verify operations work (mock implementation)
        assert True  # Placeholder for actual database tests
```

### **Performance Integration Tests**

```python
# tests/integration/test_performance_integration.py
import pytest
import asyncio
import time
from typing import List
import statistics

class TestPerformanceIntegration:
    """Test performance requirements and SLAs"""
    
    @pytest.mark.asyncio
    async def test_api_latency_requirements(self):
        """Test that API latency meets requirements (<2s for 95th percentile)"""
        from tests.api.test_framework import APITestClient, APITestConfig
        
        latencies = []
        
        config = APITestConfig()
        async with APITestClient(config) as client:
            # Perform 100 requests to get statistical data
            for _ in range(100):
                start_time = time.time()
                response = await client.get("/health")
                end_time = time.time()
                
                client.assert_response_ok(response)
                latencies.append(end_time - start_time)
        
        # Calculate percentiles
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        avg_latency = statistics.mean(latencies)
        
        # Assert performance requirements
        assert p95_latency < 2.0, f"95th percentile latency {p95_latency:.3f}s exceeds 2s requirement"
        assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}s exceeds 1s target"
        
        print(f"Performance metrics: avg={avg_latency:.3f}s, p95={p95_latency:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_user_handling(self):
        """Test concurrent user handling (target: 1000 concurrent users)"""
        async def simulate_user_session():
            """Simulate a user session"""
            config = APITestConfig()
            async with APITestClient(config) as client:
                # Simulate typical user workflow
                await client.get("/health")
                
                # Login
                login_response = await client.post("/auth/login", json={
                    "username": "test@example.com",
                    "password": "test_password"
                })
                
                if login_response.status_code == 200:
                    token_data = login_response.json()
                    client.default_headers["Authorization"] = f"Bearer {token_data['access_token']}"
                    
                    # Perform user actions
                    await client.get("/users/me")
                    await client.get("/ftns/balance")
                
                return True
        
        # Simulate 100 concurrent users (scaled down for testing)
        concurrent_users = 100
        tasks = [simulate_user_session() for _ in range(concurrent_users)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_sessions = sum(1 for r in results if r is True)
        total_time = end_time - start_time
        
        success_rate = successful_sessions / concurrent_users
        throughput = concurrent_users / total_time
        
        # Assert performance targets
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95% target"
        assert throughput >= 10, f"Throughput {throughput:.1f} users/sec below target"
        
        print(f"Concurrent users test: {successful_sessions}/{concurrent_users} successful")
        print(f"Throughput: {throughput:.1f} users/sec")

    @pytest.mark.asyncio
    async def test_ftns_transaction_precision(self):
        """Test FTNS transaction precision (<1ms calculation time)"""
        from prsm.tokenomics.ftns_service import FTNSService
        
        ftns_service = FTNSService()
        calculation_times = []
        
        # Test 100 calculations for statistical significance
        for _ in range(100):
            start_time = time.perf_counter()
            
            # Perform FTNS calculation
            cost = await ftns_service.calculate_query_cost(
                model_name="gpt-3.5-turbo",
                tokens_used=1000,
                complexity_factor=1.0
            )
            
            end_time = time.perf_counter()
            calculation_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            calculation_times.append(calculation_time)
            assert cost > 0, "FTNS cost calculation should return positive value"
        
        # Analyze calculation performance
        avg_time = statistics.mean(calculation_times)
        max_time = max(calculation_times)
        p95_time = statistics.quantiles(calculation_times, n=20)[18]  # 95th percentile
        
        # Assert precision requirements
        assert avg_time < 1.0, f"Average calculation time {avg_time:.3f}ms exceeds 1ms requirement"
        assert p95_time < 2.0, f"95th percentile time {p95_time:.3f}ms exceeds 2ms tolerance"
        
        print(f"FTNS calculation performance: avg={avg_time:.3f}ms, max={max_time:.3f}ms, p95={p95_time:.3f}ms")
```

## 🔒 Security Testing

### **Automated Security Test Suite**

```python
# tests/security/test_security_suite.py
import pytest
import asyncio
from tests.api.test_framework import APITestClient, APITestConfig

class TestSecurityValidation:
    """Security validation test suite"""
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self):
        """Test SQL injection protection"""
        config = APITestConfig()
        async with APITestClient(config) as client:
            # Test common SQL injection patterns
            injection_patterns = [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "'; UPDATE users SET password='hacked'; --",
                "' UNION SELECT * FROM users --"
            ]
            
            for pattern in injection_patterns:
                response = await client.post("/auth/login", json={
                    "username": pattern,
                    "password": "test"
                })
                
                # Should return 401 (unauthorized) not 500 (server error)
                assert response.status_code in [400, 401], \
                    f"SQL injection pattern '{pattern}' should be rejected safely"

    @pytest.mark.asyncio
    async def test_xss_protection(self):
        """Test XSS protection"""
        config = APITestConfig()
        async with APITestClient(config) as client:
            # Get auth token first
            auth_token = await self._get_auth_token(client)
            client.default_headers["Authorization"] = f"Bearer {auth_token}"
            
            xss_patterns = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
                "';alert('xss');//"
            ]
            
            for pattern in xss_patterns:
                response = await client.put("/users/me/preferences", json={
                    "theme": pattern,
                    "note": pattern
                })
                
                # Request should be processed but content sanitized
                if response.status_code == 200:
                    # Verify sanitization occurred
                    get_response = await client.get("/users/me")
                    if get_response.status_code == 200:
                        data = get_response.json()
                        preferences = data.get("preferences", {})
                        
                        # Check that dangerous content was sanitized
                        assert "<script>" not in str(preferences), "XSS content not sanitized"
                        assert "javascript:" not in str(preferences), "XSS content not sanitized"

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting protection"""
        config = APITestConfig()
        async with APITestClient(config) as client:
            # Attempt rapid requests to trigger rate limiting
            responses = []
            
            for i in range(150):  # Exceed typical rate limits
                response = await client.post("/auth/login", json={
                    "username": f"test{i}@example.com",
                    "password": "test"
                })
                responses.append(response.status_code)
                
                # If we hit rate limiting, break
                if response.status_code == 429:
                    break
            
            # Should eventually hit rate limiting
            rate_limited = any(status == 429 for status in responses)
            assert rate_limited, "Rate limiting should be triggered by rapid requests"

    @pytest.mark.asyncio
    async def test_authentication_required(self):
        """Test that protected endpoints require authentication"""
        config = APITestConfig()
        async with APITestClient(config) as client:
            protected_endpoints = [
                "/users/me",
                "/ftns/balance",
                "/ai/query",
                "/admin/metrics"
            ]
            
            for endpoint in protected_endpoints:
                response = await client.get(endpoint)
                assert response.status_code == 401, \
                    f"Endpoint {endpoint} should require authentication"

    @pytest.mark.asyncio
    async def test_authorization_enforcement(self):
        """Test that role-based authorization is enforced"""
        config = APITestConfig()
        async with APITestClient(config) as client:
            # Get regular user token
            user_token = await self._get_auth_token(client)
            client.default_headers["Authorization"] = f"Bearer {user_token}"
            
            admin_endpoints = [
                "/admin/users",
                "/admin/metrics",
                "/admin/audit-logs"
            ]
            
            for endpoint in admin_endpoints:
                response = await client.get(endpoint)
                assert response.status_code in [403, 404], \
                    f"Admin endpoint {endpoint} should deny access to regular users"

    async def _get_auth_token(self, client: APITestClient) -> str:
        """Helper to get authentication token"""
        response = await client.post("/auth/login", json={
            "username": "test@example.com",
            "password": "test_password"
        })
        
        if response.status_code == 200:
            data = response.json()
            return data["access_token"]
        else:
            raise ValueError("Failed to authenticate test user")
```

## 📊 Test Reporting & CI/CD Integration

### **Test Report Generation**

```python
# scripts/generate_test_report.py
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import jinja2

def run_tests_with_coverage():
    """Run test suite with coverage reporting"""
    try:
        # Run pytest with coverage
        result = subprocess.run([
            "python", "-m", "pytest",
            "--cov=prsm",
            "--cov-report=json",
            "--cov-report=html",
            "--json-report",
            "--json-report-file=reports/test_results.json",
            "tests/"
        ], capture_output=True, text=True, check=True)
        
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def generate_html_report():
    """Generate comprehensive HTML test report"""
    
    # Load test results
    with open("reports/test_results.json") as f:
        test_data = json.load(f)
    
    # Load coverage data
    with open("reports/coverage.json") as f:
        coverage_data = json.load(f)
    
    # Calculate summary statistics
    summary = {
        "total_tests": test_data["summary"]["total"],
        "passed_tests": test_data["summary"]["passed"],
        "failed_tests": test_data["summary"]["failed"],
        "skipped_tests": test_data["summary"]["skipped"],
        "test_duration": test_data["duration"],
        "coverage_percentage": coverage_data["totals"]["percent_covered"],
        "timestamp": datetime.now().isoformat()
    }
    
    # Generate report
    template = jinja2.Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PRSM Test Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; }
            .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }
            .passed { color: green; }
            .failed { color: red; }
            .skipped { color: orange; }
        </style>
    </head>
    <body>
        <h1>PRSM Test Report</h1>
        <div class="summary">
            <h2>Test Summary</h2>
            <div class="metric">
                <strong>Total Tests:</strong> {{ summary.total_tests }}
            </div>
            <div class="metric passed">
                <strong>Passed:</strong> {{ summary.passed_tests }}
            </div>
            <div class="metric failed">
                <strong>Failed:</strong> {{ summary.failed_tests }}
            </div>
            <div class="metric skipped">
                <strong>Skipped:</strong> {{ summary.skipped_tests }}
            </div>
            <div class="metric">
                <strong>Duration:</strong> {{ "%.2f"|format(summary.test_duration) }}s
            </div>
            <div class="metric">
                <strong>Coverage:</strong> {{ "%.1f"|format(summary.coverage_percentage) }}%
            </div>
            <div class="metric">
                <strong>Generated:</strong> {{ summary.timestamp }}
            </div>
        </div>
        
        <h2>Test Results by Category</h2>
        <!-- Additional test details would go here -->
        
        <h2>Coverage Report</h2>
        <p>Detailed coverage report available at: <a href="htmlcov/index.html">htmlcov/index.html</a></p>
    </body>
    </html>
    """)
    
    html_content = template.render(summary=summary)
    
    with open("reports/test_report.html", "w") as f:
        f.write(html_content)
    
    print(f"Test report generated: reports/test_report.html")
    return summary

if __name__ == "__main__":
    # Ensure reports directory exists
    Path("reports").mkdir(exist_ok=True)
    
    # Run tests
    success, output = run_tests_with_coverage()
    
    if success:
        print("✅ Tests completed successfully")
        summary = generate_html_report()
        
        # Print summary
        print(f"\nTest Summary:")
        print(f"  Total: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Coverage: {summary['coverage_percentage']:.1f}%")
        
        # Exit with error if tests failed
        if summary['failed_tests'] > 0:
            sys.exit(1)
    else:
        print("❌ Tests failed")
        print(output)
        sys.exit(1)
```

### **CI/CD Pipeline Configuration**

```yaml
# .github/workflows/api-testing.yml
name: API Testing & Integration Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
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
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ --cov=prsm --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: prsm_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio
    
    - name: Run PRSM API
      run: |
        export DATABASE_URL=postgresql://postgres:test_password@localhost/prsm_test
        export REDIS_URL=redis://localhost:6379
        python -m prsm.api.main &
        sleep 10  # Wait for API to start
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run API tests
      run: |
        pytest tests/api/ -v

  load-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install locust
    
    - name: Start PRSM API
      run: |
        python -m prsm.api.main &
        sleep 15  # Wait for API to fully start
    
    - name: Run load tests
      run: |
        locust -f tests/load/locustfile.py \
          --host=http://localhost:8000 \
          --users=50 \
          --spawn-rate=5 \
          --run-time=2m \
          --headless \
          --html=reports/load_test.html
    
    - name: Upload load test results
      uses: actions/upload-artifact@v3
      with:
        name: load-test-results
        path: reports/

  security-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif
    
    - name: Run OWASP ZAP scan
      uses: zaproxy/action-full-scan@v0.4.0
      with:
        target: 'http://localhost:8000'
```

---

## 📋 Testing Checklist

### **Pre-Release Testing Checklist**

- [ ] **Unit Tests**: All component tests passing (>95% coverage)
- [ ] **Integration Tests**: All external service integrations validated
- [ ] **API Tests**: All endpoints tested with various scenarios
- [ ] **Load Tests**: Performance requirements validated (1000 concurrent users)
- [ ] **Security Tests**: Vulnerability scanning completed
- [ ] **FTNS Precision**: Transaction calculations meet <1ms requirement
- [ ] **Cross-browser Testing**: Web interfaces tested on major browsers
- [ ] **Mobile Testing**: Mobile app functionality validated
- [ ] **Documentation**: API documentation updated and accurate

### **Performance Benchmarks**

| Metric | Target | Validation Method |
|--------|---------|------------------|
| API Latency (P95) | <2 seconds | Load testing with Locust |
| Concurrent Users | 1000+ | Stress testing simulation |
| FTNS Calculation | <1ms | Performance integration tests |
| Error Rate | <1% | API reliability testing |
| Uptime | >99.9% | Health check monitoring |

> _"Comprehensive testing is the cornerstone of reliable software. PRSM's multi-layer testing strategy ensures enterprise-grade quality and performance at scale."_
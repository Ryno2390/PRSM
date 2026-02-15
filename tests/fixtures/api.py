"""
API Testing Fixtures
====================

Comprehensive fixtures for testing FastAPI endpoints, authentication,
and API integration scenarios.
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch
import json
from datetime import datetime, timezone, timedelta
import jwt

try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from httpx import AsyncClient
    import pytest_asyncio
    # Don't import create_app directly due to initialization issues
    # from prsm.interface.api.main import create_app
    # from prsm.core.auth.jwt_handler import create_access_token, verify_token
    # from prsm.core.config import get_config
    create_app = None
    create_access_token = None
    verify_token = None
    get_config = None
except ImportError:
    # If imports fail, create mock fixtures
    TestClient = None
    AsyncClient = None
    create_app = None


@pytest.fixture(scope="session")
def test_app():
    """Create test FastAPI application"""
    if create_app is None:
        pytest.skip("FastAPI dependencies not available")
    
    # Create app with test configuration
    app = create_app()
    
    # Override dependencies for testing
    # This would typically override database sessions, auth, etc.
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create test client for synchronous API testing"""
    if TestClient is None:
        pytest.skip("TestClient not available")
    
    with TestClient(test_app) as client:
        yield client


@pytest_asyncio.fixture
async def async_test_client(test_app):
    """Create async test client for async API testing"""
    if AsyncClient is None:
        pytest.skip("AsyncClient not available")
    
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


class AuthTestHelper:
    """Helper for authentication testing"""
    
    def __init__(self):
        self.test_users = {
            "admin": {
                "user_id": "test_admin_user",
                "username": "admin",
                "email": "admin@test.com",
                "roles": ["admin", "user"],
                "permissions": ["read", "write", "admin"]
            },
            "user": {
                "user_id": "test_regular_user", 
                "username": "user",
                "email": "user@test.com",
                "roles": ["user"],
                "permissions": ["read", "write"]
            },
            "readonly": {
                "user_id": "test_readonly_user",
                "username": "readonly",
                "email": "readonly@test.com",
                "roles": ["readonly"],
                "permissions": ["read"]
            }
        }
    
    def create_test_token(
        self,
        user_type: str = "user",
        expires_delta: Optional[timedelta] = None,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create test JWT token"""
        if user_type not in self.test_users:
            raise ValueError(f"Unknown user type: {user_type}")
        
        user_data = self.test_users[user_type].copy()
        
        if additional_claims:
            user_data.update(additional_claims)
        
        if expires_delta is None:
            expires_delta = timedelta(hours=1)
        
        # Create token using actual JWT creation logic
        try:
            return create_access_token(data=user_data, expires_delta=expires_delta)
        except:
            # Fallback to manual JWT creation for testing
            payload = {
                **user_data,
                "exp": datetime.utcnow() + expires_delta,
                "iat": datetime.utcnow(),
                "type": "access"
            }
            return jwt.encode(payload, "test-secret", algorithm="HS256")
    
    def get_auth_headers(self, user_type: str = "user") -> Dict[str, str]:
        """Get authorization headers for test requests"""
        token = self.create_test_token(user_type)
        return {"Authorization": f"Bearer {token}"}
    
    def get_user_data(self, user_type: str) -> Dict[str, Any]:
        """Get user data for testing"""
        return self.test_users.get(user_type, {})


@pytest.fixture
def auth_helper():
    """Authentication helper fixture"""
    return AuthTestHelper()


@pytest.fixture
def admin_headers(auth_helper):
    """Admin authorization headers"""
    return auth_helper.get_auth_headers("admin")


@pytest.fixture  
def user_headers(auth_helper):
    """Regular user authorization headers"""
    return auth_helper.get_auth_headers("user")


@pytest.fixture
def readonly_headers(auth_helper):
    """Readonly user authorization headers"""
    return auth_helper.get_auth_headers("readonly")


@pytest.fixture
def expired_token_headers(auth_helper):
    """Expired token headers for testing authentication failures"""
    token = auth_helper.create_test_token(
        expires_delta=timedelta(seconds=-1)  # Already expired
    )
    return {"Authorization": f"Bearer {token}"}


class APITestDataFactory:
    """Factory for creating API test data"""
    
    @staticmethod
    def create_nwtn_query_request(
        query: str = "Test reasoning query",
        mode: str = "adaptive",
        max_depth: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Create NWTN query request data"""
        return {
            "query": query,
            "mode": mode,
            "max_depth": max_depth,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
    
    @staticmethod
    def create_ftns_transfer_request(
        recipient: str = "test_recipient",
        amount: float = 10.0,
        description: str = "Test transfer",
        **kwargs
    ) -> Dict[str, Any]:
        """Create FTNS transfer request data"""
        return {
            "recipient": recipient,
            "amount": amount,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
    
    @staticmethod
    def create_marketplace_item_request(
        title: str = "Test Item",
        description: str = "Test marketplace item",
        price: float = 50.0,
        category: str = "tools",
        **kwargs
    ) -> Dict[str, Any]:
        """Create marketplace item request data"""
        return {
            "title": title,
            "description": description,
            "price": price,
            "category": category,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
    
    @staticmethod
    def create_user_registration_request(
        username: str = "testuser",
        email: str = "test@example.com",
        password: str = "testpassword123",
        **kwargs
    ) -> Dict[str, Any]:
        """Create user registration request data"""
        return {
            "username": username,
            "email": email,
            "password": password,
            **kwargs
        }


@pytest.fixture
def api_data_factory():
    """API test data factory fixture"""
    return APITestDataFactory()


@pytest.fixture
def mock_external_apis():
    """Mock external API dependencies"""
    mocks = {}
    
    # Mock OpenAI API
    mocks['openai'] = Mock()
    mocks['openai'].chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Mock AI response"))]
    )
    
    # Mock Anthropic API
    mocks['anthropic'] = Mock()
    mocks['anthropic'].messages.create.return_value = Mock(
        content=[Mock(text="Mock Claude response")]
    )
    
    # Mock IPFS API
    mocks['ipfs'] = Mock()
    mocks['ipfs'].add.return_value = Mock(hash="QmTestHash123")
    mocks['ipfs'].get.return_value = b"Test IPFS content"
    
    # Mock Redis
    mocks['redis'] = Mock()
    mocks['redis'].get.return_value = None
    mocks['redis'].set.return_value = True
    
    return mocks


@pytest.fixture
def api_response_schemas():
    """Expected API response schemas for validation"""
    return {
        "nwtn_response": {
            "type": "object",
            "required": ["response", "reasoning_depth", "confidence"],
            "properties": {
                "response": {"type": "string"},
                "reasoning_depth": {"type": "integer"},
                "confidence": {"type": "number"},
                "session_id": {"type": "string"}
            }
        },
        "ftns_balance_response": {
            "type": "object", 
            "required": ["balance", "available_balance"],
            "properties": {
                "balance": {"type": "number"},
                "available_balance": {"type": "number"},
                "pending_balance": {"type": "number"}
            }
        },
        "error_response": {
            "type": "object",
            "required": ["error", "message"],
            "properties": {
                "error": {"type": "string"},
                "message": {"type": "string"},
                "details": {"type": "object"}
            }
        }
    }


@pytest.fixture
def api_performance_monitor():
    """Monitor API performance during tests"""
    class APIPerformanceMonitor:
        def __init__(self):
            self.request_times = []
            self.response_sizes = []
            
        def record_request(self, start_time: float, end_time: float, response_size: int):
            self.request_times.append((end_time - start_time) * 1000)  # Convert to ms
            self.response_sizes.append(response_size)
        
        def get_stats(self):
            if not self.request_times:
                return {"avg_response_time": 0, "total_requests": 0}
            
            return {
                "avg_response_time": sum(self.request_times) / len(self.request_times),
                "max_response_time": max(self.request_times),
                "min_response_time": min(self.request_times),
                "total_requests": len(self.request_times),
                "avg_response_size": sum(self.response_sizes) / len(self.response_sizes) if self.response_sizes else 0
            }
    
    return APIPerformanceMonitor()


# Rate limiting test fixtures

@pytest.fixture
def rate_limit_tester():
    """Helper for testing API rate limiting"""
    class RateLimitTester:
        @staticmethod
        async def test_rate_limit(
            client: AsyncClient,
            endpoint: str,
            method: str = "GET",
            limit: int = 10,
            window_seconds: int = 60,
            headers: Optional[Dict[str, str]] = None
        ):
            """Test rate limiting on an endpoint"""
            responses = []
            
            for i in range(limit + 2):  # Try to exceed limit
                if method.upper() == "GET":
                    response = await client.get(endpoint, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(endpoint, json={}, headers=headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                responses.append({
                    "status_code": response.status_code,
                    "attempt": i + 1,
                    "headers": dict(response.headers)
                })
            
            return responses
    
    return RateLimitTester()


# WebSocket testing fixtures

@pytest.fixture
def websocket_test_helper():
    """Helper for WebSocket API testing"""
    class WebSocketTestHelper:
        @staticmethod
        async def test_websocket_connection(test_client, endpoint: str):
            """Test WebSocket connection"""
            with test_client.websocket_connect(endpoint) as websocket:
                # Test connection
                yield websocket
    
    return WebSocketTestHelper()


# API error simulation fixtures

@pytest.fixture
def api_error_simulator():
    """Simulate various API error conditions"""
    class APIErrorSimulator:
        @staticmethod
        def create_network_error():
            """Simulate network connectivity error"""
            return Exception("Network connection failed")
        
        @staticmethod
        def create_timeout_error():
            """Simulate request timeout"""
            return Exception("Request timed out")
        
        @staticmethod
        def create_server_error():
            """Simulate internal server error"""
            return Exception("Internal server error")
        
        @staticmethod  
        def create_validation_error(field: str, message: str):
            """Simulate validation error"""
            return {
                "error": "validation_error",
                "field": field,
                "message": message
            }
    
    return APIErrorSimulator()


# Integration test fixtures

@pytest.fixture
def end_to_end_test_helper():
    """Helper for end-to-end API testing"""
    class EndToEndTestHelper:
        def __init__(self):
            self.test_session_data = {}
        
        async def simulate_full_user_workflow(
            self,
            client: AsyncClient,
            auth_headers: Dict[str, str]
        ):
            """Simulate complete user workflow"""
            workflow_results = []
            
            # 1. Check initial balance
            balance_response = await client.get("/api/v1/ftns/balance", headers=auth_headers)
            workflow_results.append(("balance_check", balance_response.status_code))
            
            # 2. Submit NWTN query
            query_data = {"query": "Test reasoning query", "mode": "adaptive"}
            nwtn_response = await client.post("/api/v1/nwtn/query", json=query_data, headers=auth_headers)
            workflow_results.append(("nwtn_query", nwtn_response.status_code))
            
            # 3. Check updated balance (after FTNS charge)
            balance_response_2 = await client.get("/api/v1/ftns/balance", headers=auth_headers)
            workflow_results.append(("balance_after_query", balance_response_2.status_code))
            
            return workflow_results
    
    return EndToEndTestHelper()


# Security testing fixtures

@pytest.fixture
def security_test_helper():
    """Helper for API security testing"""
    class SecurityTestHelper:
        @staticmethod
        def create_sql_injection_payloads():
            """Common SQL injection test payloads"""
            return [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'--",
                "' UNION SELECT * FROM users --"
            ]
        
        @staticmethod
        def create_xss_payloads():
            """Common XSS test payloads"""
            return [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
                "';alert('xss');//"
            ]
        
        @staticmethod
        def create_malformed_requests():
            """Malformed request payloads"""
            return [
                {"invalid": "json", "structure": True},
                "",
                "not json at all",
                {"extremely_long_field": "x" * 10000}
            ]
    
    return SecurityTestHelper()
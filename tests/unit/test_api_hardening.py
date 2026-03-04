"""
Tests for API Hardening Module
==============================

Tests for rate limiting, JWT authentication, WebSocket status,
and OpenAPI specification generation.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request, WebSocket
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from prsm.node.api_hardening import (
    APIHardening,
    APISecurityConfig,
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    RateLimitMiddleware,
    JWTAuthMiddleware,
    StatusWebSocket,
    generate_openapi_schema,
    get_current_user,
    require_auth,
)


# ── Rate Limiter Tests ────────────────────────────────────────────────────────

class TestRateLimitConfig:
    """Tests for RateLimitConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_second == 10
        assert config.requests_per_minute == 100
        assert config.requests_per_hour == 1000
        assert config.burst_size == 20
        assert config.enabled is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_second=50,
            requests_per_minute=500,
            requests_per_hour=5000,
            burst_size=100,
            enabled=False
        )
        assert config.requests_per_second == 50
        assert config.requests_per_minute == 500
        assert config.requests_per_hour == 5000
        assert config.burst_size == 100
        assert config.enabled is False
    
    def test_endpoint_limits(self):
        """Test endpoint-specific limits."""
        config = RateLimitConfig()
        assert "/auth/login" in config.endpoint_limits
        assert "/compute/submit" in config.endpoint_limits


class TestRateLimiter:
    """Tests for RateLimiter."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter with default config."""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            enabled=True
        )
        return RateLimiter(config)
    
    def test_get_client_key(self, rate_limiter):
        """Test client key generation."""
        key = rate_limiter._get_client_key("192.168.1.1", "/api/test")
        assert "192.168.1.1" in key
        assert "/api/test" in key
    
    def test_get_endpoint_config(self, rate_limiter):
        """Test endpoint configuration lookup."""
        # Test specific endpoint
        config = rate_limiter._get_endpoint_config("/auth/login")
        assert "requests_per_minute" in config
        
        # Test default endpoint
        config = rate_limiter._get_endpoint_config("/unknown/endpoint")
        assert "requests_per_minute" in config
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, rate_limiter):
        """Test rate limit check when allowed."""
        result = await rate_limiter.check_rate_limit("test_client", "/test")
        assert result.allowed is True
        assert result.limit > 0
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter):
        """Test rate limit check when exceeded."""
        # Make many requests to exceed limit
        for _ in range(15):
            await rate_limiter.check_rate_limit("test_client", "/test")
        
        # Next request should be blocked
        result = await rate_limiter.check_rate_limit("test_client", "/test")
        assert result.allowed is False
        assert result.retry_after is not None
    
    @pytest.mark.asyncio
    async def test_rate_limit_disabled(self):
        """Test rate limiter when disabled."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        
        result = await limiter.check_rate_limit("test_client", "/test")
        assert result.allowed is True
    
    def test_reset_limits(self, rate_limiter):
        """Test resetting rate limits."""
        # Add some entries
        rate_limiter._local_store["rate_limit:test_client:/test"] = [time.time()] * 5
        
        # Reset
        rate_limiter.reset_limits("test_client")
        
        # Check that entries are cleared
        assert len([k for k in rate_limiter._local_store.keys() if "test_client" in k]) == 0


# ── Rate Limit Middleware Tests ────────────────────────────────────────────────

class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.get("/health")
        async def health():
            return {"status": "ok"}
        
        return app
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter."""
        config = RateLimitConfig(
            requests_per_minute=5,
            enabled=True
        )
        return RateLimiter(config)
    
    def test_exempt_endpoints(self, app, rate_limiter):
        """Test that exempt endpoints bypass rate limiting."""
        app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
        client = TestClient(app)
        
        # Health endpoint should always work
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_rate_limit_headers(self, app, rate_limiter):
        """Test that rate limit headers are added."""
        app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
        client = TestClient(app)
        
        response = client.get("/test")
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


# ── JWT Auth Middleware Tests ──────────────────────────────────────────────────

class TestJWTAuthMiddleware:
    """Tests for JWTAuthMiddleware."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        
        @app.get("/")
        async def root():
            return {"message": "root"}
        
        @app.get("/health")
        async def health():
            return {"status": "ok"}
        
        @app.get("/protected")
        async def protected_endpoint(request: Request):
            user = getattr(request.state, "user", None)
            return {"message": "protected", "user": user}
        
        return app
    
    @pytest.fixture
    def middleware(self, app):
        """Create JWT auth middleware."""
        return JWTAuthMiddleware(app, jwt_handler=None)
    
    def test_public_endpoints(self, app, middleware):
        """Test that public endpoints don't require auth."""
        app.add_middleware(JWTAuthMiddleware, jwt_handler=None)
        client = TestClient(app)
        
        # Root endpoint is in PUBLIC_ENDPOINTS
        response = client.get("/")
        assert response.status_code == 200
        
        # Health endpoint is in PUBLIC_ENDPOINTS
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_protected_endpoint_without_token(self, app, middleware):
        """Test protected endpoint without token."""
        app.add_middleware(JWTAuthMiddleware, jwt_handler=None)
        client = TestClient(app)
        
        response = client.get("/protected")
        assert response.status_code == 401
    
    def test_options_request_bypass(self, app, middleware):
        """Test that OPTIONS requests bypass auth."""
        app.add_middleware(JWTAuthMiddleware, jwt_handler=None)
        client = TestClient(app)
        
        response = client.options("/protected")
        # Should not be 401
        assert response.status_code != 401 or response.status_code == 200


# ── Status WebSocket Tests ─────────────────────────────────────────────────────

class TestStatusWebSocket:
    """Tests for StatusWebSocket."""
    
    @pytest.fixture
    def status_websocket(self):
        """Create StatusWebSocket instance."""
        return StatusWebSocket()
    
    def test_initial_state(self, status_websocket):
        """Test initial state."""
        assert status_websocket.get_connection_count() == 0
        assert len(status_websocket.get_client_ids()) == 0
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, status_websocket):
        """Test WebSocket connection and disconnection."""
        mock_websocket = MagicMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        
        # Connect
        await status_websocket.connect(mock_websocket)
        assert status_websocket.get_connection_count() == 1
        
        # Disconnect
        status_websocket.disconnect(mock_websocket)
        # Allow async disconnect to complete
        await asyncio.sleep(0.1)
        assert status_websocket.get_connection_count() == 0
    
    @pytest.mark.asyncio
    async def test_broadcast_status(self, status_websocket):
        """Test broadcasting status to connections."""
        mock_websocket = MagicMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        
        await status_websocket.connect(mock_websocket)
        
        # Broadcast status
        status = {"node_id": "test", "status": "running"}
        await status_websocket.broadcast_status(status)
        
        # Verify message was sent
        assert mock_websocket.send_json.called
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "status_update"
        assert call_args["data"] == status
    
    @pytest.mark.asyncio
    async def test_send_personal_status(self, status_websocket):
        """Test sending personal status message."""
        mock_websocket = MagicMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        
        await status_websocket.connect(mock_websocket, client_id="test_client")
        
        # Send personal message
        message = {"type": "test", "data": "test_data"}
        await status_websocket.send_personal_status(mock_websocket, message)
        
        assert mock_websocket.send_json.called
    
    @pytest.mark.asyncio
    async def test_send_personal_message_by_id(self, status_websocket):
        """Test sending message to specific client by ID."""
        mock_websocket = MagicMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        
        await status_websocket.connect(mock_websocket, client_id="client_123")
        
        # Send message to client by ID
        message = {"type": "notification", "data": "test"}
        result = await status_websocket.send_personal_message("client_123", message)
        
        assert result is True
        assert mock_websocket.send_json.called
    
    @pytest.mark.asyncio
    async def test_status_cache(self, status_websocket):
        """Test that status is cached for new connections."""
        # First connect a client so broadcast has somewhere to go
        mock_websocket1 = MagicMock(spec=WebSocket)
        mock_websocket1.accept = AsyncMock()
        mock_websocket1.send_json = AsyncMock()
        await status_websocket.connect(mock_websocket1)
        
        # Broadcast initial status
        status = {"node_id": "test", "status": "running"}
        await status_websocket.broadcast_status(status)
        
        # Verify status is cached
        assert status_websocket._status_cache == status
        
        # Connect new client
        mock_websocket2 = MagicMock(spec=WebSocket)
        mock_websocket2.accept = AsyncMock()
        mock_websocket2.send_json = AsyncMock()
        
        await status_websocket.connect(mock_websocket2)
        
        # Verify connection message was sent (at least 1 call)
        calls = mock_websocket2.send_json.call_args_list
        assert len(calls) >= 1  # At least connection message
        
        # Verify cached status is still stored
        assert status_websocket._status_cache == status


# ── OpenAPI Schema Tests ──────────────────────────────────────────────────────

class TestOpenAPISchema:
    """Tests for OpenAPI schema generation."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI(
            title="Test API",
            version="1.0.0",
        )
        
        @app.get("/test", tags=["test"])
        async def test_endpoint():
            """Test endpoint."""
            return {"message": "success"}
        
        @app.post("/auth/login", tags=["auth"])
        async def login(username: str, password: str):
            """Login endpoint."""
            return {"token": "test"}
        
        return app
    
    def test_generate_openapi_schema(self, app):
        """Test OpenAPI schema generation."""
        schema = generate_openapi_schema(app)
        
        assert schema is not None
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
    
    def test_security_schemes(self, app):
        """Test that security schemes are added."""
        schema = generate_openapi_schema(app)
        
        assert "securitySchemes" in schema["components"]
        assert "bearerAuth" in schema["components"]["securitySchemes"]
        assert "apiKey" in schema["components"]["securitySchemes"]
    
    def test_global_security(self, app):
        """Test that global security is set."""
        schema = generate_openapi_schema(app)
        
        assert "security" in schema
        assert len(schema["security"]) > 0
    
    def test_common_schemas(self, app):
        """Test that common schemas are added."""
        schema = generate_openapi_schema(app)
        
        assert "ErrorResponse" in schema["components"]["schemas"]
        assert "StatusResponse" in schema["components"]["schemas"]
        assert "RateLimitInfo" in schema["components"]["schemas"]
    
    def test_schema_caching(self, app):
        """Test that schema is cached on app."""
        schema1 = generate_openapi_schema(app)
        schema2 = generate_openapi_schema(app)
        
        # Should return same cached schema
        assert schema1 is schema2


# ── API Hardening Integration Tests ────────────────────────────────────────────

class TestAPIHardening:
    """Tests for APIHardening class."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.get("/health")
        async def health():
            return {"status": "ok"}
        
        return app
    
    @pytest.fixture
    def config(self):
        """Create security configuration."""
        return APISecurityConfig(
            enable_rate_limiting=True,
            enable_jwt_auth=True,
            enable_websocket=True,
            enable_openapi=True,
        )
    
    def test_initialization(self, app, config):
        """Test API hardening initialization."""
        hardening = APIHardening(app, config)
        
        assert hardening.config == config
        assert hardening.rate_limiter is None  # Not initialized yet
        assert hardening.jwt_handler is None
        assert hardening.status_websocket is None
    
    @pytest.mark.asyncio
    async def test_initialize_components(self, app, config):
        """Test component initialization."""
        hardening = APIHardening(app, config)
        await hardening.initialize()
        
        assert hardening.rate_limiter is not None
        assert hardening.status_websocket is not None
    
    def test_apply_middleware(self, app, config):
        """Test middleware application."""
        hardening = APIHardening(app, config)
        hardening.rate_limiter = RateLimiter(RateLimitConfig())
        
        # Should not raise
        hardening.apply_middleware()
    
    def test_setup_openapi(self, app, config):
        """Test OpenAPI setup."""
        hardening = APIHardening(app, config)
        hardening.setup_openapi()
        
        # Check that openapi method is set
        assert app.openapi is not None
    
    def test_get_status_websocket(self, app, config):
        """Test getting status websocket."""
        hardening = APIHardening(app, config)
        hardening.status_websocket = StatusWebSocket()
        
        ws = hardening.get_status_websocket()
        assert ws is not None


# ── Dependency Injection Tests ────────────────────────────────────────────────

class TestDependencyInjection:
    """Tests for dependency injection helpers."""
    
    @pytest.mark.asyncio
    async def test_get_current_user(self):
        """Test getting current user from request."""
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        request.state.user = {"user_id": "123", "username": "test"}
        
        user = await get_current_user(request)
        assert user == {"user_id": "123", "username": "test"}
    
    @pytest.mark.asyncio
    async def test_get_current_user_none(self):
        """Test getting current user when not set."""
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        # Configure MagicMock to return None for user attribute
        request.state.user = None
        
        user = await get_current_user(request)
        assert user is None
    
    @pytest.mark.asyncio
    async def test_require_auth_success(self):
        """Test require_auth with valid user."""
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        request.state.user = {"user_id": "123", "username": "test"}
        
        user = await require_auth(request)
        assert user == {"user_id": "123", "username": "test"}
    
    @pytest.mark.asyncio
    async def test_require_auth_failure(self):
        """Test require_auth without user."""
        from fastapi import HTTPException
        
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        # Configure MagicMock to return None for user attribute
        request.state.user = None
        
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(request)
        
        assert exc_info.value.status_code == 401


# ── Integration Tests ──────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests for API hardening."""
    
    @pytest.fixture
    def hardened_app(self):
        """Create fully hardened FastAPI app."""
        app = FastAPI(
            title="Test API",
            version="1.0.0",
        )
        
        config = APISecurityConfig(
            enable_rate_limiting=True,
            enable_jwt_auth=True,
            enable_websocket=True,
            enable_openapi=True,
            rate_limit_requests_per_minute=100,
        )
        
        hardening = APIHardening(app, config)
        hardening.setup_openapi()
        
        @app.get("/public")
        async def public():
            return {"message": "public"}
        
        @app.get("/protected")
        async def protected(request: Request):
            user = getattr(request.state, "user", None)
            return {"message": "protected", "user": user}
        
        @app.get("/health")
        async def health():
            return {"status": "ok"}
        
        return app, hardening
    
    def test_openapi_endpoint(self, hardened_app):
        """Test OpenAPI endpoint is available."""
        app, _ = hardened_app
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()
    
    def test_docs_endpoint(self, hardened_app):
        """Test docs endpoint is available."""
        app, _ = hardened_app
        client = TestClient(app)
        
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_health_endpoint(self, hardened_app):
        """Test health endpoint works without auth."""
        app, _ = hardened_app
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_public_endpoint(self, hardened_app):
        """Test public endpoint works."""
        app, _ = hardened_app
        client = TestClient(app)
        
        response = client.get("/public")
        assert response.status_code == 200


# ── Rate Limit Result Tests ────────────────────────────────────────────────────

class TestRateLimitResult:
    """Tests for RateLimitResult model."""
    
    def test_allowed_result(self):
        """Test creating an allowed result."""
        result = RateLimitResult(
            allowed=True,
            limit=100,
            remaining=99,
            reset_at=time.time() + 60
        )
        
        assert result.allowed is True
        assert result.limit == 100
        assert result.remaining == 99
        assert result.retry_after is None
    
    def test_blocked_result(self):
        """Test creating a blocked result."""
        result = RateLimitResult(
            allowed=False,
            limit=100,
            remaining=0,
            reset_at=time.time() + 60,
            retry_after=60,
            rule_name="per_minute"
        )
        
        assert result.allowed is False
        assert result.retry_after == 60
        assert result.rule_name == "per_minute"


# ── API Security Config Tests ──────────────────────────────────────────────────

class TestAPISecurityConfig:
    """Tests for APISecurityConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = APISecurityConfig()
        
        assert config.enable_rate_limiting is True
        assert config.enable_jwt_auth is True
        assert config.enable_websocket is True
        assert config.enable_openapi is True
        assert config.rate_limit_requests_per_minute == 100
        assert config.rate_limit_requests_per_hour == 1000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = APISecurityConfig(
            enable_rate_limiting=False,
            enable_jwt_auth=False,
            rate_limit_requests_per_minute=50,
            jwt_required_endpoints=["/admin", "/api"]
        )
        
        assert config.enable_rate_limiting is False
        assert config.enable_jwt_auth is False
        assert config.rate_limit_requests_per_minute == 50
        assert "/admin" in config.jwt_required_endpoints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
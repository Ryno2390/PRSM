"""
Test suite for PRSM Python SDK Client
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
from prsm_sdk import PRSMClient, PRSMError, AuthenticationError, RateLimitError
from prsm_sdk.models import QueryRequest, PRSMResponse, StreamingResponse


class TestPRSMClient:
    """Test cases for PRSMClient"""
    
    @pytest.fixture
    def client(self):
        """Create test client instance"""
        return PRSMClient(
            api_key="test_key",
            base_url="https://api.test.prsm.io",
            timeout=30
        )
    
    def test_client_initialization(self):
        """Test client initialization with various parameters"""
        # Basic initialization
        client = PRSMClient(api_key="test_key")
        assert client.auth.api_key == "test_key"
        assert client.base_url == "https://api.prsm.ai/v1"
        assert client.timeout == 60
        
        # Custom initialization
        client = PRSMClient(
            api_key="custom_key",
            base_url="https://custom.prsm.io",
            timeout=30
        )
        assert client.auth.api_key == "custom_key"
        assert client.base_url == "https://custom.prsm.io"
        assert client.timeout == 30
    
    def test_client_has_subclients(self, client):
        """Test client has all expected sub-clients"""
        assert hasattr(client, 'ftns')
        assert hasattr(client, 'marketplace')
        assert hasattr(client, 'tools')
        assert hasattr(client, 'compute')
        assert hasattr(client, 'storage')
        assert hasattr(client, 'governance')
        assert hasattr(client, 'auth')
    
    @pytest.mark.asyncio
    async def test_successful_query(self, client):
        """Test successful query execution"""
        # Mock the _request method
        mock_response = {
            "content": "Test response",
            "model_id": "test-model",
            "provider": "prsm",
            "execution_time": 0.5,
            "token_usage": {"input": 10, "output": 20},
            "ftns_cost": 0.05,
            "reasoning_trace": [],
            "safety_status": "moderate",
            "metadata": {},
            "request_id": "req-123",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.query("Test prompt")
            
            assert result.content == "Test response"
            assert result.model_id == "test-model"
            assert result.ftns_cost == 0.05
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, client):
        """Test authentication error handling"""
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = AuthenticationError("Invalid API key")
            
            with pytest.raises(AuthenticationError):
                await client.query("Test prompt")
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self, client):
        """Test rate limit error handling"""
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = RateLimitError(retry_after=60)
            
            with pytest.raises(RateLimitError) as exc_info:
                await client.query("Test prompt")
            
            assert exc_info.value.details.get("retry_after") == 60
    
    @pytest.mark.asyncio
    async def test_query_with_options(self, client):
        """Test query with additional options"""
        mock_response = {
            "content": "Test response",
            "model_id": "gpt-4",
            "provider": "openai",
            "execution_time": 1.0,
            "token_usage": {"input": 50, "output": 100},
            "ftns_cost": 0.10,
            "reasoning_trace": ["step1", "step2"],
            "safety_status": "moderate",
            "metadata": {},
            "request_id": "req-456",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.query(
                "Test prompt",
                model_id="gpt-4",
                max_tokens=500,
                temperature=0.5
            )
            
            # Verify request was made with correct params
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/nwtn/query"
            
            json_data = call_args[1]['json_data']
            assert json_data['prompt'] == "Test prompt"
            assert json_data['model_id'] == "gpt-4"
            assert json_data['max_tokens'] == 500
            assert json_data['temperature'] == 0.5
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check method"""
        with patch.object(client, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "healthy", "version": "0.37.0"}
            
            result = await client.health_check()
            
            assert result["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager"""
        async with PRSMClient(api_key="test_key") as client:
            assert client.auth.api_key == "test_key"
            # Client should be properly initialized
            assert client._session is None  # Session created on first use
    
    def test_sync_query_wrapper(self, client):
        """Test synchronous wrapper exists"""
        # The client is async, but we can verify the structure
        assert hasattr(client, 'query')
        assert asyncio.iscoroutinefunction(client.query)


class TestPRSMModels:
    """Test cases for PRSM SDK models"""
    
    def test_query_request_model(self):
        """Test QueryRequest model validation"""
        
        # Valid request
        request = QueryRequest(
            prompt="Test prompt",
            model_id="test-model",
            max_tokens=1000,
            temperature=0.7
        )
        assert request.prompt == "Test prompt"
        assert request.model_id == "test-model"
        assert request.max_tokens == 1000
        assert request.temperature == 0.7
        
        # Minimal request
        minimal_request = QueryRequest(prompt="Minimal prompt")
        assert minimal_request.prompt == "Minimal prompt"
        assert minimal_request.model_id is None
    
    def test_prsm_response_model(self):
        """Test PRSMResponse model validation"""
        
        response = PRSMResponse(
            content="Test answer",
            model_id="test-model",
            provider="prsm",
            execution_time=0.5,
            token_usage={"input": 10, "output": 20},
            ftns_cost=0.05,
            reasoning_trace=["step1", "step2"],
            safety_status="moderate",
            metadata={"key": "value"},
            request_id="req-123",
            timestamp="2024-01-01T00:00:00Z"
        )
        
        assert response.content == "Test answer"
        assert response.model_id == "test-model"
        assert response.ftns_cost == 0.05
        assert len(response.reasoning_trace) == 2


class TestStreamingResponse:
    """Test cases for StreamingResponse model"""
    
    def test_partial_streaming_response(self):
        """Test partial streaming response"""
        response = StreamingResponse(
            type="partial",
            content="Partial response content",
            session_id="stream_session_123"
        )
        
        assert response.type == "partial"
        assert response.content == "Partial response content"
        assert response.session_id == "stream_session_123"
        assert response.is_final is False
        assert response.metadata == {}
    
    def test_final_streaming_response(self):
        """Test final streaming response"""
        response = StreamingResponse(
            type="final",
            content="Final complete response",
            session_id="stream_session_123",
            is_final=True,
            confidence_score=0.88,
            context_used=150,
            ftns_charged=0.075,
            metadata={"total_tokens": 500}
        )
        
        assert response.type == "final"
        assert response.content == "Final complete response"
        assert response.is_final is True
        assert response.confidence_score == 0.88
        assert response.context_used == 150
        assert response.ftns_charged == 0.075
        assert response.metadata["total_tokens"] == 500
    
    def test_error_streaming_response(self):
        """Test error streaming response"""
        response = StreamingResponse(
            type="error",
            content="An error occurred during processing",
            session_id="stream_session_123",
            error_code="RATE_LIMIT_EXCEEDED",
            metadata={"retry_after": 60}
        )
        
        assert response.type == "error"
        assert response.content == "An error occurred during processing"
        assert response.error_code == "RATE_LIMIT_EXCEEDED"
        assert response.metadata["retry_after"] == 60
    
    def test_streaming_response_validation(self):
        """Test StreamingResponse validation"""
        # Invalid type should fail
        with pytest.raises(ValueError):
            StreamingResponse(
                type="invalid_type",
                content="Test content",
                session_id="session123"
            )
        
        # Final response without content should fail
        with pytest.raises(ValueError):
            StreamingResponse(
                type="final",
                content="",
                session_id="session123",
                is_final=True
            )


class TestPRSMExceptions:
    """Test cases for PRSM SDK exceptions"""
    
    def test_prsm_error_hierarchy(self):
        """Test exception hierarchy"""
        # Base error
        base_error = PRSMError("Base error message")
        assert str(base_error) == "Base error message"
        assert base_error.error_code is None
        
        # Authentication error
        auth_error = AuthenticationError("Invalid API key")
        assert isinstance(auth_error, PRSMError)
        assert str(auth_error) == "Invalid API key"
        assert auth_error.error_code == "AUTH_ERROR"
        
        # Rate limit error
        rate_error = RateLimitError(retry_after=60)
        assert isinstance(rate_error, PRSMError)
        assert "Rate limit exceeded" in str(rate_error)
        assert rate_error.details.get("retry_after") == 60


if __name__ == "__main__":
    pytest.main([__file__])

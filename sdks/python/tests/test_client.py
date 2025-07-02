"""
Test suite for PRSM Python SDK Client
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
from prsm_sdk import PRSMClient, PRSMError, AuthenticationError, RateLimitError


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
    
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session"""
        with patch('prsm_sdk.client.aiohttp.ClientSession') as mock:
            session = AsyncMock()
            mock.return_value.__aenter__.return_value = session
            yield session
    
    def test_client_initialization(self):
        """Test client initialization with various parameters"""
        # Basic initialization
        client = PRSMClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.base_url == "https://api.prsm.io"
        assert client.timeout == 60
        
        # Custom initialization
        client = PRSMClient(
            api_key="custom_key",
            base_url="https://custom.prsm.io",
            timeout=30
        )
        assert client.api_key == "custom_key"
        assert client.base_url == "https://custom.prsm.io"
        assert client.timeout == 30
    
    def test_client_headers(self, client):
        """Test client headers generation"""
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_key"
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers
    
    @pytest.mark.asyncio
    async def test_successful_query(self, client, mock_session):
        """Test successful query execution"""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "session_id": "test_session_123",
            "final_answer": "Test response",
            "reasoning_trace": [],
            "confidence_score": 0.95,
            "context_used": 100,
            "ftns_charged": 0.05
        }
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        result = await client.query("Test prompt")
        
        assert result.session_id == "test_session_123"
        assert result.final_answer == "Test response"
        assert result.confidence_score == 0.95
        assert result.context_used == 100
        assert result.ftns_charged == 0.05
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, client, mock_session):
        """Test authentication error handling"""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(AuthenticationError):
            await client.query("Test prompt")
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self, client, mock_session):
        """Test rate limit error handling"""
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(RateLimitError):
            await client.query("Test prompt")
    
    @pytest.mark.asyncio
    async def test_query_with_options(self, client, mock_session):
        """Test query with additional options"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "session_id": "test_session_123",
            "final_answer": "Test response",
            "reasoning_trace": [],
            "confidence_score": 0.95,
            "context_used": 200,
            "ftns_charged": 0.10
        }
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        result = await client.query(
            "Test prompt",
            context_allocation=200,
            preferences={"temperature": 0.7}
        )
        
        # Verify the request was made with correct parameters
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args[1]
        request_data = json.loads(call_kwargs['data'])
        
        assert request_data['prompt'] == "Test prompt"
        assert request_data['context_allocation'] == 200
        assert request_data['preferences'] == {"temperature": 0.7}
    
    @pytest.mark.asyncio
    async def test_streaming_query(self, client):
        """Test streaming query functionality"""
        # Mock WebSocket connection
        with patch('prsm_sdk.client.websockets.connect') as mock_connect:
            mock_ws = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_ws
            
            # Mock streaming responses
            mock_ws.__aiter__.return_value = [
                json.dumps({"type": "partial", "content": "Partial "}),
                json.dumps({"type": "partial", "content": "response"}),
                json.dumps({
                    "type": "final",
                    "session_id": "stream_session_123",
                    "final_answer": "Partial response",
                    "confidence_score": 0.9
                })
            ]
            
            responses = []
            async for response in client.stream("Test streaming prompt"):
                responses.append(response)
            
            assert len(responses) == 3
            assert responses[-1].session_id == "stream_session_123"
            assert responses[-1].final_answer == "Partial response"
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_session):
        """Test client as async context manager"""
        async with PRSMClient(api_key="test_key") as client:
            assert client.api_key == "test_key"
            # Client should be properly initialized and cleaned up
    
    def test_sync_query_wrapper(self, client, mock_session):
        """Test synchronous wrapper for query method"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "session_id": "sync_session_123",
            "final_answer": "Sync response",
            "reasoning_trace": [],
            "confidence_score": 0.85,
            "context_used": 150,
            "ftns_charged": 0.075
        }
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        # Test sync wrapper (if implemented)
        if hasattr(client, 'query_sync'):
            result = client.query_sync("Sync test prompt")
            assert result.session_id == "sync_session_123"
            assert result.final_answer == "Sync response"


class TestPRSMModels:
    """Test cases for PRSM SDK models"""
    
    def test_query_request_model(self):
        """Test QueryRequest model validation"""
        from prsm_sdk.models import QueryRequest
        
        # Valid request
        request = QueryRequest(
            user_id="user123",
            prompt="Test prompt",
            context_allocation=100,
            preferences={"temperature": 0.7}
        )
        assert request.user_id == "user123"
        assert request.prompt == "Test prompt"
        assert request.context_allocation == 100
        assert request.preferences == {"temperature": 0.7}
        
        # Minimal request
        minimal_request = QueryRequest(
            user_id="user456",
            prompt="Minimal prompt"
        )
        assert minimal_request.user_id == "user456"
        assert minimal_request.prompt == "Minimal prompt"
        assert minimal_request.context_allocation is None
    
    def test_prsm_response_model(self):
        """Test PRSMResponse model validation"""
        from prsm_sdk.models import PRSMResponse
        
        response = PRSMResponse(
            session_id="session123",
            user_id="user123",
            final_answer="Test answer",
            confidence_score=0.95,
            context_used=100,
            ftns_charged=0.05,
            sources=["source1", "source2"],
            reasoning_trace=[]
        )
        
        assert response.session_id == "session123"
        assert response.final_answer == "Test answer"
        assert response.confidence_score == 0.95
        assert len(response.sources) == 2


class TestPRSMExceptions:
    """Test cases for PRSM SDK exceptions"""
    
    def test_prsm_error_hierarchy(self):
        """Test exception hierarchy"""
        # Base error
        base_error = PRSMError("Base error message")
        assert str(base_error) == "Base error message"
        
        # Authentication error
        auth_error = AuthenticationError("Invalid API key")
        assert isinstance(auth_error, PRSMError)
        assert str(auth_error) == "Invalid API key"
        
        # Rate limit error
        rate_error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert isinstance(rate_error, PRSMError)
        assert str(rate_error) == "Rate limit exceeded"
        assert rate_error.retry_after == 60


if __name__ == "__main__":
    pytest.main([__file__])
"""
Test suite for PRSM Python SDK Models
"""
import pytest
from datetime import datetime
from prsm_sdk.models import (
    QueryRequest, PRSMResponse, StreamingResponse,
    ModelProvider, SafetyLevel, FTNSBalance, ModelInfo,
    ToolSpec, SafetyStatus, WebSocketMessage, MarketplaceQuery
)


class TestQueryRequest:
    """Test cases for QueryRequest model"""
    
    def test_minimal_query_request(self):
        """Test minimal valid QueryRequest"""
        request = QueryRequest(prompt="Test prompt")
        assert request.prompt == "Test prompt"
        assert request.model_id is None
        assert request.max_tokens == 1000
        assert request.temperature == 0.7
        assert request.system_prompt is None
        assert request.context == {}
        assert request.tools is None
        assert request.safety_level == SafetyLevel.MODERATE
    
    def test_full_query_request(self):
        """Test QueryRequest with all fields"""
        request = QueryRequest(
            prompt="Test prompt",
            model_id="gpt-4",
            max_tokens=2000,
            temperature=0.5,
            system_prompt="You are a helpful assistant",
            context={"user": "test"},
            tools=["calculator", "search"],
            safety_level=SafetyLevel.HIGH
        )
        assert request.prompt == "Test prompt"
        assert request.model_id == "gpt-4"
        assert request.max_tokens == 2000
        assert request.temperature == 0.5
        assert request.system_prompt == "You are a helpful assistant"
        assert request.context == {"user": "test"}
        assert request.tools == ["calculator", "search"]
        assert request.safety_level == SafetyLevel.HIGH
    
    def test_query_request_serialization(self):
        """Test QueryRequest serialization"""
        request = QueryRequest(
            prompt="Test prompt",
            model_id="test-model",
            context={"key": "value"}
        )
        data = request.model_dump()
        assert data["prompt"] == "Test prompt"
        assert data["model_id"] == "test-model"
        assert data["context"] == {"key": "value"}


class TestPRSMResponse:
    """Test cases for PRSMResponse model"""
    
    def test_minimal_prsm_response(self):
        """Test minimal PRSMResponse"""
        response = PRSMResponse(
            content="Test response",
            model_id="test-model",
            provider=ModelProvider.PRSM,
            execution_time=0.5,
            token_usage={"input": 10, "output": 20},
            ftns_cost=0.05,
            safety_status=SafetyLevel.MODERATE,
            request_id="req-123",
            timestamp=datetime.utcnow()
        )
        assert response.content == "Test response"
        assert response.model_id == "test-model"
        assert response.ftns_cost == 0.05
    
    def test_full_prsm_response(self):
        """Test PRSMResponse with all fields"""
        response = PRSMResponse(
            content="Test answer",
            model_id="gpt-4",
            provider=ModelProvider.OPENAI,
            execution_time=1.5,
            token_usage={"input": 100, "output": 200},
            ftns_cost=0.15,
            reasoning_trace=["step1", "step2", "step3"],
            safety_status=SafetyLevel.HIGH,
            metadata={"source": "test"},
            request_id="req-456",
            timestamp=datetime.utcnow()
        )
        
        assert response.content == "Test answer"
        assert response.provider == ModelProvider.OPENAI
        assert len(response.reasoning_trace) == 3
    
    def test_prsm_response_validation(self):
        """Test PRSMResponse validation"""
        # Missing required fields should fail
        with pytest.raises(Exception):
            PRSMResponse(content="Test")


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


class TestEnums:
    """Test cases for enum types"""
    
    def test_model_provider_enum(self):
        """Test ModelProvider enum values"""
        assert ModelProvider.OPENAI == "openai"
        assert ModelProvider.ANTHROPIC == "anthropic"
        assert ModelProvider.PRSM == "prsm"
    
    def test_safety_level_enum(self):
        """Test SafetyLevel enum values"""
        assert SafetyLevel.NONE == "none"
        assert SafetyLevel.LOW == "low"
        assert SafetyLevel.MODERATE == "moderate"
        assert SafetyLevel.HIGH == "high"
        assert SafetyLevel.CRITICAL == "critical"


if __name__ == "__main__":
    pytest.main([__file__])

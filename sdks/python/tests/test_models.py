"""
Test suite for PRSM Python SDK Models
"""
import pytest
from datetime import datetime
from uuid import UUID
from prsm_sdk.models import QueryRequest, PRSMResponse, StreamingResponse


class TestQueryRequest:
    """Test cases for QueryRequest model"""
    
    def test_minimal_query_request(self):
        """Test minimal valid QueryRequest"""
        request = QueryRequest(
            user_id="user123",
            prompt="Test prompt"
        )
        assert request.user_id == "user123"
        assert request.prompt == "Test prompt"
        assert request.context_allocation is None
        assert request.preferences == {}
        assert request.session_id is None
    
    def test_full_query_request(self):
        """Test QueryRequest with all fields"""
        session_id = "session-123-456"
        preferences = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "model_preference": "claude"
        }
        
        request = QueryRequest(
            user_id="user123",
            prompt="Detailed test prompt",
            context_allocation=500,
            preferences=preferences,
            session_id=session_id
        )
        
        assert request.user_id == "user123"
        assert request.prompt == "Detailed test prompt"
        assert request.context_allocation == 500
        assert request.preferences == preferences
        assert request.session_id == session_id
    
    def test_query_request_validation(self):
        """Test QueryRequest validation"""
        # Empty prompt should fail
        with pytest.raises(ValueError):
            QueryRequest(user_id="user123", prompt="")
        
        # Negative context allocation should fail
        with pytest.raises(ValueError):
            QueryRequest(
                user_id="user123",
                prompt="Test",
                context_allocation=-100
            )
        
        # Too large context allocation should fail
        with pytest.raises(ValueError):
            QueryRequest(
                user_id="user123",
                prompt="Test",
                context_allocation=1000000
            )
    
    def test_query_request_serialization(self):
        """Test QueryRequest serialization to dict"""
        request = QueryRequest(
            user_id="user123",
            prompt="Test prompt",
            context_allocation=200,
            preferences={"temperature": 0.8}
        )
        
        data = request.model_dump()
        assert data["user_id"] == "user123"
        assert data["prompt"] == "Test prompt"
        assert data["context_allocation"] == 200
        assert data["preferences"]["temperature"] == 0.8


class TestPRSMResponse:
    """Test cases for PRSMResponse model"""
    
    def test_minimal_prsm_response(self):
        """Test minimal valid PRSMResponse"""
        response = PRSMResponse(
            session_id="session123",
            user_id="user123",
            final_answer="Test answer",
            context_used=100,
            ftns_charged=0.05
        )
        
        assert response.session_id == "session123"
        assert response.user_id == "user123"
        assert response.final_answer == "Test answer"
        assert response.context_used == 100
        assert response.ftns_charged == 0.05
        assert response.confidence_score is None
        assert response.sources == []
        assert response.reasoning_trace == []
        assert response.safety_validated is True
        assert response.metadata == {}
    
    def test_full_prsm_response(self):
        """Test PRSMResponse with all fields"""
        reasoning_trace = [
            {
                "step": 1,
                "agent": "architect",
                "reasoning": "Breaking down the problem"
            },
            {
                "step": 2,
                "agent": "executor",
                "reasoning": "Executing the solution"
            }
        ]
        
        sources = ["source1.pdf", "source2.md", "web_search_results"]
        metadata = {"model_used": "claude-3", "processing_time": 2.5}
        
        response = PRSMResponse(
            session_id="session123",
            user_id="user123",
            final_answer="Comprehensive test answer",
            reasoning_trace=reasoning_trace,
            confidence_score=0.92,
            context_used=250,
            ftns_charged=0.125,
            sources=sources,
            safety_validated=True,
            metadata=metadata
        )
        
        assert response.session_id == "session123"
        assert response.final_answer == "Comprehensive test answer"
        assert len(response.reasoning_trace) == 2
        assert response.confidence_score == 0.92
        assert response.context_used == 250
        assert response.ftns_charged == 0.125
        assert len(response.sources) == 3
        assert response.metadata["model_used"] == "claude-3"
    
    def test_prsm_response_validation(self):
        """Test PRSMResponse validation"""
        # Negative context_used should fail
        with pytest.raises(ValueError):
            PRSMResponse(
                session_id="session123",
                user_id="user123",
                final_answer="Test",
                context_used=-50,
                ftns_charged=0.05
            )
        
        # Negative ftns_charged should fail
        with pytest.raises(ValueError):
            PRSMResponse(
                session_id="session123",
                user_id="user123",
                final_answer="Test",
                context_used=100,
                ftns_charged=-0.01
            )
        
        # Invalid confidence score should fail
        with pytest.raises(ValueError):
            PRSMResponse(
                session_id="session123",
                user_id="user123",
                final_answer="Test",
                context_used=100,
                ftns_charged=0.05,
                confidence_score=1.5  # > 1.0
            )


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


if __name__ == "__main__":
    pytest.main([__file__])
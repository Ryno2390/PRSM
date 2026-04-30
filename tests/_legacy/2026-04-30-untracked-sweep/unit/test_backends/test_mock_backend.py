"""
Unit tests for MockBackend

Tests the mock backend implementation which provides deterministic
responses for testing without requiring API keys.
"""

import pytest

from prsm.compute.nwtn.backends import (
    BackendType,
    GenerateResult,
    EmbedResult,
    TokenUsage,
)
from prsm.compute.nwtn.backends.mock_backend import MockBackend


class TestMockBackend:
    """Tests for MockBackend that don't require API keys"""
    
    @pytest.fixture
    def backend(self):
        """Create a mock backend with no delay for fast tests"""
        return MockBackend(delay_seconds=0)
    
    @pytest.mark.asyncio
    async def test_initialize(self, backend):
        """Test that backend initializes correctly"""
        await backend.initialize()
        assert backend._initialized is True
    
    @pytest.mark.asyncio
    async def test_close(self, backend):
        """Test that backend closes correctly"""
        await backend.initialize()
        await backend.close()
        assert backend._initialized is False
    
    @pytest.mark.asyncio
    async def test_context_manager(self, backend):
        """Test async context manager usage"""
        async with backend:
            assert backend._initialized is True
        assert backend._initialized is False
    
    @pytest.mark.asyncio
    async def test_generate_returns_content(self, backend):
        """Test that generate returns valid content"""
        await backend.initialize()
        result = await backend.generate("test prompt")
        
        assert isinstance(result, GenerateResult)
        assert len(result.content) > 0
        assert result.provider == BackendType.MOCK
        assert result.model_id == "mock-model"
    
    @pytest.mark.asyncio
    async def test_generate_is_deterministic(self, backend):
        """Test that generate returns consistent results for same prompt"""
        await backend.initialize()
        
        result1 = await backend.generate("research topic")
        result2 = await backend.generate("research topic")
        
        assert result1.content == result2.content
    
    @pytest.mark.asyncio
    async def test_generate_research_response(self, backend):
        """Test that research prompts get research responses"""
        await backend.initialize()
        result = await backend.generate("research quantum computing")
        
        assert "research" in result.content.lower() or "finding" in result.content.lower()
    
    @pytest.mark.asyncio
    async def test_generate_coding_response(self, backend):
        """Test that coding prompts get coding responses"""
        await backend.initialize()
        result = await backend.generate("implement a function in python")
        
        assert "```python" in result.content or "def " in result.content
    
    @pytest.mark.asyncio
    async def test_generate_analysis_response(self, backend):
        """Test that analysis prompts get analysis responses"""
        await backend.initialize()
        result = await backend.generate("analyze this data")
        
        assert "analysis" in result.content.lower() or "observation" in result.content.lower()
    
    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, backend):
        """Test generation with system prompt"""
        await backend.initialize()
        result = await backend.generate(
            prompt="test prompt",
            system_prompt="You are a coding assistant"
        )
        
        assert isinstance(result, GenerateResult)
        assert len(result.content) > 0
    
    @pytest.mark.asyncio
    async def test_generate_respects_max_tokens(self, backend):
        """Test that max_tokens truncates response"""
        await backend.initialize()
        result = await backend.generate("research topic", max_tokens=10)
        
        # Response should be truncated (roughly 10 tokens * 4 chars)
        assert len(result.content) <= 50
    
    @pytest.mark.asyncio
    async def test_generate_increments_call_count(self, backend):
        """Test that call_count increments with each call"""
        await backend.initialize()
        
        assert backend.call_count == 0
        await backend.generate("test 1")
        assert backend.call_count == 1
        await backend.generate("test 2")
        assert backend.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generate_token_usage(self, backend):
        """Test that token usage is calculated"""
        await backend.initialize()
        result = await backend.generate("test prompt")
        
        assert result.token_usage.prompt_tokens > 0
        assert result.token_usage.completion_tokens > 0
        assert result.token_usage.total_tokens > 0
    
    @pytest.mark.asyncio
    async def test_generate_metadata(self, backend):
        """Test that metadata is included in result"""
        await backend.initialize()
        result = await backend.generate("test prompt")
        
        assert "mock" in result.metadata
        assert result.metadata["mock"] is True
        assert "prompt_hash" in result.metadata
    
    @pytest.mark.asyncio
    async def test_embed_returns_vector(self, backend):
        """Test that embed returns a valid embedding vector"""
        await backend.initialize()
        result = await backend.embed("test text")
        
        assert isinstance(result, EmbedResult)
        assert len(result.embedding) == 1536  # Same as OpenAI small
        assert result.provider == BackendType.MOCK
        assert result.model_id == "mock-embedding"
    
    @pytest.mark.asyncio
    async def test_embed_is_deterministic(self, backend):
        """Test that embed returns consistent results for same text"""
        await backend.initialize()
        
        result1 = await backend.embed("test text")
        result2 = await backend.embed("test text")
        
        assert result1.embedding == result2.embedding
    
    @pytest.mark.asyncio
    async def test_embed_different_texts_different_vectors(self, backend):
        """Test that different texts produce different embeddings"""
        await backend.initialize()
        
        result1 = await backend.embed("first text")
        result2 = await backend.embed("second text")
        
        assert result1.embedding != result2.embedding
    
    @pytest.mark.asyncio
    async def test_embed_values_in_range(self, backend):
        """Test that embedding values are normalized to [-1, 1]"""
        await backend.initialize()
        result = await backend.embed("test text")
        
        for value in result.embedding:
            assert -1.0 <= value <= 1.0
    
    @pytest.mark.asyncio
    async def test_embed_is_normalized_unit_vector(self, backend):
        """Test that embedding is normalized to unit length for cosine similarity"""
        await backend.initialize()
        result = await backend.embed("test text")
        
        # Calculate magnitude (should be ~1.0 for unit vector)
        magnitude = sum(x * x for x in result.embedding) ** 0.5
        assert 0.99 <= magnitude <= 1.01, f"Expected unit vector, got magnitude {magnitude}"
    
    @pytest.mark.asyncio
    async def test_embed_custom_dimensions(self, backend):
        """Test that embedding respects custom dimensions parameter"""
        await backend.initialize()
        
        # Test with 768 dimensions (like all-mpnet-base-v2)
        result_768 = await backend.embed("test text", dimensions=768)
        assert len(result_768.embedding) == 768
        
        # Test with 3072 dimensions (like text-embedding-3-large)
        result_3072 = await backend.embed("test text", dimensions=3072)
        assert len(result_3072.embedding) == 3072
    
    @pytest.mark.asyncio
    async def test_embed_normalized_with_custom_dimensions(self, backend):
        """Test that custom dimension embeddings are also normalized"""
        await backend.initialize()
        
        for dims in [384, 768, 1536, 3072]:
            result = await backend.embed("test text", dimensions=dims)
            magnitude = sum(x * x for x in result.embedding) ** 0.5
            assert 0.99 <= magnitude <= 1.01, f"Dimensions {dims}: expected unit vector, got magnitude {magnitude}"
    
    @pytest.mark.asyncio
    async def test_embed_metadata_includes_normalized_flag(self, backend):
        """Test that metadata indicates the embedding is normalized"""
        await backend.initialize()
        result = await backend.embed("test text")
        
        assert result.metadata.get("normalized") is True
        assert result.metadata.get("dimensions") == 1536
    
    @pytest.mark.asyncio
    async def test_is_available(self, backend):
        """Test that mock backend is always available"""
        assert await backend.is_available() is True
        await backend.initialize()
        assert await backend.is_available() is True
    
    def test_backend_type(self, backend):
        """Test that backend type is correct"""
        assert backend.backend_type == BackendType.MOCK
    
    def test_models_supported(self, backend):
        """Test that models_supported returns expected models"""
        models = backend.models_supported
        assert "mock-model" in models
        assert "mock-embedding" in models
    
    def test_get_model_info_all(self, backend):
        """Test getting info for all models"""
        info = backend.get_model_info()
        
        assert isinstance(info, dict)
        assert "mock-model" in info
        assert "mock-embedding" in info
    
    def test_get_model_info_specific(self, backend):
        """Test getting info for a specific model"""
        info = backend.get_model_info("mock-model")
        
        assert info["model_id"] == "mock-model"
        assert info["provider"] == "mock"
        assert "context_window" in info
    
    def test_get_model_info_unknown(self, backend):
        """Test getting info for unknown model returns empty dict"""
        info = backend.get_model_info("unknown-model")
        assert info == {}
    
    def test_repr(self, backend):
        """Test string representation"""
        repr_str = repr(backend)
        assert "MockBackend" in repr_str
        assert "delay=" in repr_str


class TestMockBackendWithDelay:
    """Tests for MockBackend with simulated delay"""
    
    @pytest.fixture
    def backend(self):
        """Create a mock backend with small delay"""
        return MockBackend(delay_seconds=0.05)
    
    @pytest.mark.asyncio
    async def test_delay_attribute_set(self, backend):
        """Test that delay attribute is correctly set"""
        assert backend.delay_seconds == 0.05
    
    @pytest.mark.asyncio
    async def test_delay_in_metadata(self, backend):
        """Test that delay is recorded in result metadata"""
        await backend.initialize()
        result = await backend.generate("test prompt")
        
        # The delay_seconds should be in metadata
        assert "delay_seconds" in result.metadata
        assert result.metadata["delay_seconds"] == 0.05
    
    @pytest.mark.asyncio
    async def test_execution_time_in_result(self, backend):
        """Test that execution_time is included in result"""
        await backend.initialize()
        result = await backend.generate("test prompt")
        
        assert result.execution_time > 0
        assert isinstance(result.execution_time, float)
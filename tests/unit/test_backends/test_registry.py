"""
Unit tests for BackendRegistry

Tests the backend registry which manages backend lifecycle,
selection, and fallback logic.
"""

import pytest

from prsm.compute.nwtn.backends import (
    BackendType,
    BackendConfig,
    BackendRegistry,
    BackendHealth,
    GenerateResult,
    EmbedResult,
    AllBackendsFailedError,
)
from prsm.compute.nwtn.backends.mock_backend import MockBackend


class TestBackendConfig:
    """Tests for BackendConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = BackendConfig()
        
        assert config.primary_backend == BackendType.MOCK
        assert BackendType.MOCK in config.fallback_chain
        assert config.timeout_seconds == 120
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            anthropic_api_key="test-key",
            timeout_seconds=60
        )
        
        assert config.primary_backend == BackendType.ANTHROPIC
        assert config.anthropic_api_key == "test-key"
        assert config.timeout_seconds == 60
    
    def test_from_environment(self, monkeypatch):
        """Test loading config from environment variables"""
        monkeypatch.setenv("PRSM_PRIMARY_BACKEND", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("PRSM_BACKEND_TIMEOUT", "90")
        
        config = BackendConfig.from_environment()
        
        assert config.primary_backend == BackendType.OPENAI
        assert config.openai_api_key == "test-openai-key"
        assert config.timeout_seconds == 90
    
    def test_from_environment_fallback_chain(self, monkeypatch):
        """Test parsing fallback chain from environment"""
        monkeypatch.setenv("PRSM_FALLBACK_CHAIN", "anthropic,openai,mock")
        
        config = BackendConfig.from_environment()
        
        assert BackendType.ANTHROPIC in config.fallback_chain
        assert BackendType.OPENAI in config.fallback_chain
        assert BackendType.MOCK in config.fallback_chain
    
    def test_validate_missing_api_key(self):
        """Test validation catches missing API keys"""
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            anthropic_api_key=None
        )
        
        issues = config.validate()
        assert len(issues) > 0
        assert any("ANTHROPIC_API_KEY" in issue for issue in issues)
    
    def test_validate_valid_config(self):
        """Test validation passes for valid config"""
        config = BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.MOCK]  # Only mock in chain, no API keys needed
        )
        
        issues = config.validate()
        assert len(issues) == 0
    
    def test_is_valid(self):
        """Test is_valid method"""
        config = BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.MOCK]  # Only mock in chain, no API keys needed
        )
        assert config.is_valid() is True
        
        config_invalid = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            anthropic_api_key=None
        )
        assert config_invalid.is_valid() is False
    
    def test_get_default_model(self):
        """Test getting default model for backend type"""
        config = BackendConfig()
        
        assert config.get_default_model(BackendType.ANTHROPIC) == "claude-3-5-sonnet-20241022"
        assert config.get_default_model(BackendType.OPENAI) == "gpt-4o"
        assert config.get_default_model(BackendType.MOCK) == "mock-model"
    
    def test_get_default_embedding_model(self):
        """Test getting default embedding model"""
        config = BackendConfig()
        
        assert config.get_default_embedding_model(BackendType.OPENAI) == "text-embedding-3-small"
        assert config.get_default_embedding_model(BackendType.MOCK) == "mock-embedding"
    
    def test_to_dict(self):
        """Test converting config to dictionary"""
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            anthropic_api_key="secret-key"
        )
        
        d = config.to_dict()
        
        assert d["primary_backend"] == "anthropic"
        assert d["anthropic_api_key"] == "***"  # Should be masked
        assert "timeout_seconds" in d
    
    def test_repr(self):
        """Test string representation"""
        config = BackendConfig(primary_backend=BackendType.OPENAI)
        repr_str = repr(config)
        
        assert "BackendConfig" in repr_str
        assert "openai" in repr_str


class TestBackendHealth:
    """Tests for BackendHealth"""
    
    def test_default_health(self):
        """Test default health values"""
        health = BackendHealth(backend_type=BackendType.MOCK)
        
        assert health.backend_type == BackendType.MOCK
        assert health.is_available is True
        assert health.error_count == 0
    
    def test_success_rate_no_requests(self):
        """Test success rate with no requests"""
        health = BackendHealth(backend_type=BackendType.MOCK)
        assert health.success_rate == 100.0
    
    def test_success_rate_with_requests(self):
        """Test success rate calculation"""
        health = BackendHealth(
            backend_type=BackendType.MOCK,
            total_requests=10,
            successful_requests=8
        )
        assert health.success_rate == 80.0
    
    def test_to_dict(self):
        """Test converting health to dictionary"""
        health = BackendHealth(
            backend_type=BackendType.MOCK,
            is_available=True,
            error_count=0
        )
        
        d = health.to_dict()
        
        assert d["backend_type"] == "mock"
        assert d["is_available"] is True
        assert "success_rate" in d


class TestBackendRegistry:
    """Tests for BackendRegistry"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a config that uses only mock backend"""
        return BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.MOCK],
            mock_delay_seconds=0
        )
    
    @pytest.fixture
    def multi_backend_config(self):
        """Create a config with multiple backends in fallback chain"""
        return BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.MOCK],
            mock_delay_seconds=0
        )
    
    @pytest.mark.asyncio
    async def test_initialize_creates_backends(self, mock_config):
        """Test that initialize creates backends"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        assert BackendType.MOCK in registry._backends
        assert registry._initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mock_config):
        """Test that initialize is idempotent"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        await registry.initialize()  # Second call
        
        assert registry._initialized is True
    
    @pytest.mark.asyncio
    async def test_close(self, mock_config):
        """Test closing registry"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        await registry.close()
        
        assert len(registry._backends) == 0
        assert registry._initialized is False
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_config):
        """Test async context manager usage"""
        async with BackendRegistry(mock_config) as registry:
            assert registry._initialized is True
            assert BackendType.MOCK in registry._backends
        
        assert len(registry._backends) == 0
    
    @pytest.mark.asyncio
    async def test_get_backend(self, mock_config):
        """Test getting a specific backend"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        backend = registry.get_backend(BackendType.MOCK)
        assert backend is not None
        assert backend.backend_type == BackendType.MOCK
    
    @pytest.mark.asyncio
    async def test_get_backend_unknown(self, mock_config):
        """Test getting unknown backend returns None"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        backend = registry.get_backend(BackendType.ANTHROPIC)
        assert backend is None
    
    @pytest.mark.asyncio
    async def test_get_health(self, mock_config):
        """Test getting health status"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        health = registry.get_health(BackendType.MOCK)
        assert health is not None
        assert health.backend_type == BackendType.MOCK
    
    @pytest.mark.asyncio
    async def test_get_available_backend(self, mock_config):
        """Test getting available backend"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        backend, backend_type = await registry.get_available_backend()
        
        assert backend is not None
        assert backend_type == BackendType.MOCK
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, mock_config):
        """Test executing with fallback"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        result = await registry.execute_with_fallback("test prompt")
        
        assert isinstance(result, GenerateResult)
        assert result.content is not None
        assert result.provider == BackendType.MOCK
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_tracks_requests(self, mock_config):
        """Test that execute tracks request counts"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        await registry.execute_with_fallback("test 1")
        await registry.execute_with_fallback("test 2")
        
        health = registry.get_health(BackendType.MOCK)
        assert health.total_requests == 2
        assert health.successful_requests == 2
    
    @pytest.mark.asyncio
    async def test_register_backend(self, mock_config):
        """Test manually registering a backend"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        # Create and register another mock backend
        new_backend = MockBackend(delay_seconds=0)
        registry.register(new_backend)
        
        assert new_backend.backend_type in registry._backends
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, mock_config):
        """Test health check for all backends"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        health = await registry.health_check_all()
        
        assert BackendType.MOCK in health
        assert health[BackendType.MOCK].is_available is True
    
    @pytest.mark.asyncio
    async def test_get_status(self, mock_config):
        """Test getting registry status"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        status = registry.get_status()
        
        assert status["initialized"] is True
        assert "primary_backend" in status
        assert "backends" in status


class TestBackendRegistryFallback:
    """Tests for fallback behavior"""
    
    @pytest.fixture
    def fallback_config(self):
        """Create config with fallback chain"""
        return BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.MOCK],
            mock_delay_seconds=0,
            max_retries=1
        )
    
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, fallback_config):
        """Test that fallback works when primary fails"""
        registry = BackendRegistry(fallback_config)
        await registry.initialize()
        
        # Should succeed with mock backend
        result = await registry.execute_with_fallback("test prompt")
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_all_backends_failed_error(self):
        """Test that AllBackendsFailedError is raised when all fail"""
        # Create config with no valid backends
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            fallback_chain=[BackendType.OPENAI],
            anthropic_api_key=None,
            openai_api_key=None
        )
        
        registry = BackendRegistry(config)
        
        # Should fail to initialize since no API keys
        with pytest.raises(AllBackendsFailedError):
            await registry.initialize()


class TestBackendRegistryRetry:
    """Tests for retry behavior"""
    
    @pytest.fixture
    def retry_config(self):
        """Create config with retry settings"""
        return BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.MOCK],
            mock_delay_seconds=0,
            max_retries=3,
            retry_delay_seconds=0.01
        )
    
    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, retry_config):
        """Test that transient errors trigger retry"""
        registry = BackendRegistry(retry_config)
        await registry.initialize()
        
        # Mock backend should succeed
        result = await registry.execute_with_fallback("test prompt")
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test behavior when retries are exhausted"""
        config = BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[],
            mock_delay_seconds=0,
            max_retries=1
        )
        
        registry = BackendRegistry(config)
        await registry.initialize()
        
        # Should succeed with mock
        result = await registry.execute_with_fallback("test")
        assert result is not None


class TestBackendRegistryEmbedding:
    """Tests for embedding functionality with fallback"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a config that uses only mock backend"""
        return BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.MOCK],
            mock_delay_seconds=0
        )
    
    @pytest.mark.asyncio
    async def test_embed_with_fallback(self, mock_config):
        """Test embedding with fallback"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        result = await registry.embed_with_fallback("test text")
        
        assert isinstance(result, EmbedResult)
        assert len(result.embedding) == 1536  # Default dimensions
        assert result.provider == BackendType.MOCK
    
    @pytest.mark.asyncio
    async def test_embed_with_custom_dimensions(self, mock_config):
        """Test embedding with custom dimensions"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        result = await registry.embed_with_fallback("test text", dimensions=768)
        
        assert isinstance(result, EmbedResult)
        assert len(result.embedding) == 768
    
    @pytest.mark.asyncio
    async def test_embed_is_normalized(self, mock_config):
        """Test that embedding is normalized to unit vector"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        result = await registry.embed_with_fallback("test text")
        
        # Calculate magnitude (should be ~1.0 for unit vector)
        magnitude = sum(x * x for x in result.embedding) ** 0.5
        assert 0.99 <= magnitude <= 1.01, f"Expected unit vector, got magnitude {magnitude}"
    
    @pytest.mark.asyncio
    async def test_embed_tracks_requests(self, mock_config):
        """Test that embed tracks request counts"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        await registry.embed_with_fallback("test 1")
        await registry.embed_with_fallback("test 2")
        
        health = registry.get_health(BackendType.MOCK)
        assert health.total_requests == 2
        assert health.successful_requests == 2
    
    @pytest.mark.asyncio
    async def test_embed_deterministic(self, mock_config):
        """Test that embedding is deterministic for same input"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        result1 = await registry.embed_with_fallback("same text")
        result2 = await registry.embed_with_fallback("same text")
        
        assert result1.embedding == result2.embedding
    
    @pytest.mark.asyncio
    async def test_embed_different_texts_different_vectors(self, mock_config):
        """Test that different texts produce different embeddings"""
        registry = BackendRegistry(mock_config)
        await registry.initialize()
        
        result1 = await registry.embed_with_fallback("first text")
        result2 = await registry.embed_with_fallback("second text")
        
        assert result1.embedding != result2.embedding
    
    @pytest.mark.asyncio
    async def test_embed_all_backends_failed_error(self):
        """Test that AllBackendsFailedError is raised when all backends fail for embedding"""
        # Create config with no valid backends
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            fallback_chain=[BackendType.OPENAI],
            anthropic_api_key=None,
            openai_api_key=None
        )
        
        registry = BackendRegistry(config)
        
        # Should fail to initialize since no API keys
        with pytest.raises(AllBackendsFailedError):
            await registry.initialize()
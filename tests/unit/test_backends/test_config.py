"""
Unit tests for BackendConfig

Tests the configuration management for LLM backends.
"""

import os
import pytest
import tempfile
import json

from prsm.compute.nwtn.backends import BackendType, BackendConfig


class TestBackendConfig:
    """Tests for BackendConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = BackendConfig()
        
        assert config.primary_backend == BackendType.MOCK
        assert BackendType.ANTHROPIC in config.fallback_chain
        assert BackendType.OPENAI in config.fallback_chain
        assert BackendType.LOCAL in config.fallback_chain
        assert BackendType.MOCK in config.fallback_chain
        assert config.timeout_seconds == 120
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.rate_limit_rpm == 60
    
    def test_custom_values(self):
        """Test custom configuration values"""
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            anthropic_api_key="sk-ant-test",
            timeout_seconds=60,
            max_retries=5
        )
        
        assert config.primary_backend == BackendType.ANTHROPIC
        assert config.anthropic_api_key == "sk-ant-test"
        assert config.timeout_seconds == 60
        assert config.max_retries == 5
    
    def test_from_environment_basic(self, monkeypatch):
        """Test loading basic config from environment"""
        monkeypatch.setenv("PRSM_PRIMARY_BACKEND", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        
        config = BackendConfig.from_environment()
        
        assert config.primary_backend == BackendType.ANTHROPIC
        assert config.anthropic_api_key == "test-anthropic-key"
        assert config.openai_api_key == "test-openai-key"
    
    def test_from_environment_timeout(self, monkeypatch):
        """Test loading timeout from environment"""
        monkeypatch.setenv("PRSM_BACKEND_TIMEOUT", "90")
        
        config = BackendConfig.from_environment()
        
        assert config.timeout_seconds == 90
    
    def test_from_environment_retry_settings(self, monkeypatch):
        """Test loading retry settings from environment"""
        monkeypatch.setenv("PRSM_BACKEND_MAX_RETRIES", "5")
        monkeypatch.setenv("PRSM_BACKEND_RETRY_DELAY", "2.0")
        
        config = BackendConfig.from_environment()
        
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0
    
    def test_from_environment_local_settings(self, monkeypatch):
        """Test loading local backend settings from environment"""
        monkeypatch.setenv("PRSM_LOCAL_MODEL_PATH", "/path/to/models")
        monkeypatch.setenv("PRSM_OLLAMA_HOST", "http://custom-host:11434")
        monkeypatch.setenv("PRSM_USE_OLLAMA", "false")
        
        config = BackendConfig.from_environment()
        
        assert config.local_model_path == "/path/to/models"
        assert config.ollama_host == "http://custom-host:11434"
        assert config.use_ollama is False
    
    def test_from_environment_fallback_chain(self, monkeypatch):
        """Test parsing fallback chain from environment"""
        monkeypatch.setenv("PRSM_FALLBACK_CHAIN", "openai,local,mock")
        
        config = BackendConfig.from_environment()
        
        assert config.fallback_chain == [
            BackendType.OPENAI,
            BackendType.LOCAL,
            BackendType.MOCK
        ]
    
    def test_from_environment_invalid_backend_type(self, monkeypatch):
        """Test handling invalid backend type in environment"""
        monkeypatch.setenv("PRSM_PRIMARY_BACKEND", "invalid_backend")
        
        config = BackendConfig.from_environment()
        
        # Should fall back to default
        assert config.primary_backend == BackendType.MOCK
    
    def test_from_environment_mock_delay(self, monkeypatch):
        """Test loading mock delay from environment"""
        monkeypatch.setenv("PRSM_MOCK_DELAY", "0.5")
        
        config = BackendConfig.from_environment()
        
        assert config.mock_delay_seconds == 0.5
    
    def test_from_file_json(self):
        """Test loading config from JSON file"""
        config_data = {
            "primary_backend": "openai",
            "timeout_seconds": 90,
            "max_retries": 5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            
            config = BackendConfig.from_file(f.name)
            
            assert config.primary_backend == BackendType.OPENAI
            assert config.timeout_seconds == 90
            assert config.max_retries == 5
    
    def test_from_file_not_found(self):
        """Test error when config file not found"""
        with pytest.raises(FileNotFoundError):
            BackendConfig.from_file("/nonexistent/config.json")
    
    def test_validate_no_issues(self):
        """Test validation with valid config"""
        config = BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.MOCK]  # Only mock in chain, no API keys needed
        )
        
        issues = config.validate()
        
        assert len(issues) == 0
    
    def test_validate_missing_anthropic_key(self):
        """Test validation catches missing Anthropic key"""
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            anthropic_api_key=None
        )
        
        issues = config.validate()
        
        assert len(issues) > 0
        assert any("ANTHROPIC_API_KEY" in issue for issue in issues)
    
    def test_validate_missing_openai_key(self):
        """Test validation catches missing OpenAI key"""
        config = BackendConfig(
            primary_backend=BackendType.OPENAI,
            openai_api_key=None
        )
        
        issues = config.validate()
        
        assert len(issues) > 0
        assert any("OPENAI_API_KEY" in issue for issue in issues)
    
    def test_validate_fallback_chain_missing_keys(self):
        """Test validation catches missing keys in fallback chain"""
        config = BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.ANTHROPIC, BackendType.OPENAI],
            anthropic_api_key=None,
            openai_api_key=None
        )
        
        issues = config.validate()
        
        # Should have issues for both Anthropic and OpenAI in fallback
        assert any("Anthropic" in issue and "fallback" in issue.lower() for issue in issues)
        assert any("OpenAI" in issue and "fallback" in issue.lower() for issue in issues)
    
    def test_validate_invalid_timeout(self):
        """Test validation catches invalid timeout"""
        config = BackendConfig(timeout_seconds=0)
        
        issues = config.validate()
        
        assert len(issues) > 0
        assert any("timeout" in issue.lower() for issue in issues)
    
    def test_validate_invalid_retries(self):
        """Test validation catches invalid retry count"""
        config = BackendConfig(max_retries=-1)
        
        issues = config.validate()
        
        assert len(issues) > 0
        assert any("retries" in issue.lower() for issue in issues)
    
    def test_is_valid_true(self):
        """Test is_valid returns True for valid config"""
        config = BackendConfig(
            primary_backend=BackendType.MOCK,
            fallback_chain=[BackendType.MOCK]  # Only mock in chain, no API keys needed
        )
        
        assert config.is_valid() is True
    
    def test_is_valid_false(self):
        """Test is_valid returns False for invalid config"""
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            anthropic_api_key=None
        )
        
        assert config.is_valid() is False
    
    def test_get_default_model(self):
        """Test getting default model for each backend type"""
        config = BackendConfig()
        
        assert config.get_default_model(BackendType.ANTHROPIC) == "claude-3-5-sonnet-20241022"
        assert config.get_default_model(BackendType.OPENAI) == "gpt-4o"
        assert config.get_default_model(BackendType.LOCAL) == "llama3.2"
        assert config.get_default_model(BackendType.MOCK) == "mock-model"
    
    def test_get_default_embedding_model(self):
        """Test getting default embedding model for each backend type"""
        config = BackendConfig()
        
        assert config.get_default_embedding_model(BackendType.OPENAI) == "text-embedding-3-small"
        assert config.get_default_embedding_model(BackendType.LOCAL) == "nomic-embed-text"
        assert config.get_default_embedding_model(BackendType.MOCK) == "mock-embedding"
        assert config.get_default_embedding_model(BackendType.ANTHROPIC) is None
    
    def test_to_dict(self):
        """Test converting config to dictionary"""
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            anthropic_api_key="secret-key",
            openai_api_key="another-secret"
        )
        
        d = config.to_dict()
        
        assert d["primary_backend"] == "anthropic"
        assert d["anthropic_api_key"] == "***"  # Should be masked
        assert d["openai_api_key"] == "***"  # Should be masked
        assert "timeout_seconds" in d
        assert "fallback_chain" in d
    
    def test_to_dict_no_keys(self):
        """Test to_dict with no API keys set"""
        config = BackendConfig(primary_backend=BackendType.MOCK)
        
        d = config.to_dict()
        
        assert d["anthropic_api_key"] is None
        assert d["openai_api_key"] is None
    
    def test_repr(self):
        """Test string representation"""
        config = BackendConfig(
            primary_backend=BackendType.OPENAI,
            timeout_seconds=60
        )
        
        repr_str = repr(config)
        
        assert "BackendConfig" in repr_str
        assert "openai" in repr_str
        assert "timeout=60" in repr_str


class TestBackendType:
    """Tests for BackendType enum"""
    
    def test_values(self):
        """Test enum values"""
        assert BackendType.ANTHROPIC.value == "anthropic"
        assert BackendType.OPENAI.value == "openai"
        assert BackendType.LOCAL.value == "local"
        assert BackendType.MOCK.value == "mock"
    
    def test_from_string(self):
        """Test creating from string"""
        assert BackendType("anthropic") == BackendType.ANTHROPIC
        assert BackendType("openai") == BackendType.OPENAI
        assert BackendType("local") == BackendType.LOCAL
        assert BackendType("mock") == BackendType.MOCK
    
    def test_invalid_string(self):
        """Test error on invalid string"""
        with pytest.raises(ValueError):
            BackendType("invalid")
    
    def test_case_sensitivity(self):
        """Test that values are lowercase"""
        for bt in BackendType:
            assert bt.value == bt.value.lower()


class TestBackendConfigIntegration:
    """Integration tests for BackendConfig"""
    
    def test_full_config_round_trip(self):
        """Test creating config with all settings"""
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            fallback_chain=[BackendType.OPENAI, BackendType.LOCAL, BackendType.MOCK],
            anthropic_api_key="sk-ant-test",
            openai_api_key="sk-test",
            local_model_path="/models",
            ollama_host="http://localhost:11434",
            use_ollama=True,
            timeout_seconds=90,
            max_retries=5,
            retry_delay_seconds=2.0,
            retry_exponential_base=3.0,
            rate_limit_rpm=100,
            rate_limit_tpm=200000,
            default_max_tokens=2000,
            default_temperature=0.5,
            mock_delay_seconds=0.05
        )
        
        assert config.primary_backend == BackendType.ANTHROPIC
        assert len(config.fallback_chain) == 3
        assert config.anthropic_api_key == "sk-ant-test"
        assert config.openai_api_key == "sk-test"
        assert config.local_model_path == "/models"
        assert config.ollama_host == "http://localhost:11434"
        assert config.use_ollama is True
        assert config.timeout_seconds == 90
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0
        assert config.retry_exponential_base == 3.0
        assert config.rate_limit_rpm == 100
        assert config.rate_limit_tpm == 200000
        assert config.default_max_tokens == 2000
        assert config.default_temperature == 0.5
        assert config.mock_delay_seconds == 0.05
    
    def test_environment_override_defaults(self, monkeypatch):
        """Test that environment variables override defaults"""
        # Set all environment variables
        monkeypatch.setenv("PRSM_PRIMARY_BACKEND", "local")
        monkeypatch.setenv("PRSM_FALLBACK_CHAIN", "mock")
        monkeypatch.setenv("PRSM_BACKEND_TIMEOUT", "180")
        monkeypatch.setenv("PRSM_BACKEND_MAX_RETRIES", "10")
        monkeypatch.setenv("PRSM_BACKEND_RETRY_DELAY", "3.0")
        monkeypatch.setenv("PRSM_RATE_LIMIT_RPM", "120")
        
        config = BackendConfig.from_environment()
        
        assert config.primary_backend == BackendType.LOCAL
        assert config.fallback_chain == [BackendType.MOCK]
        assert config.timeout_seconds == 180
        assert config.max_retries == 10
        assert config.retry_delay_seconds == 3.0
        assert config.rate_limit_rpm == 120
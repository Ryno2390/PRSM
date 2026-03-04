#!/usr/bin/env python3
"""
Backend Configuration
=====================

Configuration management for LLM backends. Supports loading from:
- Environment variables (preferred for secrets)
- Configuration files (YAML/JSON)
- Direct instantiation

Classes:
    BackendConfig: Main configuration dataclass
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .base import BackendType


@dataclass
class BackendConfig:
    """
    Configuration for LLM backends.
    
    This configuration can be loaded from:
    - Environment variables (preferred for secrets)
    - Configuration file (YAML/JSON)
    - Direct instantiation
    
    Attributes:
        primary_backend: The primary backend to use
        fallback_chain: List of backends to try if primary fails
        anthropic_api_key: API key for Anthropic
        openai_api_key: API key for OpenAI
        local_model_path: Path to local model (for transformers)
        ollama_host: Host URL for Ollama server
        use_ollama: Whether to use Ollama vs transformers
        timeout_seconds: Request timeout in seconds
        max_retries: Maximum retry attempts
        retry_delay_seconds: Initial delay between retries
        retry_exponential_base: Multiplier for exponential backoff
        rate_limit_rpm: Rate limit in requests per minute
        rate_limit_tpm: Rate limit in tokens per minute
        default_model: Default model for each backend type
        default_embedding_model: Default embedding model for each backend
        default_max_tokens: Default max tokens for generation
        default_temperature: Default temperature for generation
        mock_delay_seconds: Simulated delay for mock backend
    
    Example:
        # Load from environment
        config = BackendConfig.from_environment()
        
        # Load from file
        config = BackendConfig.from_file("config.yaml")
        
        # Direct instantiation
        config = BackendConfig(
            primary_backend=BackendType.ANTHROPIC,
            anthropic_api_key="sk-ant-..."
        )
    """
    
    # Primary backend selection
    primary_backend: BackendType = BackendType.MOCK
    
    # Fallback chain (tried in order if primary fails)
    fallback_chain: List[BackendType] = field(default_factory=lambda: [
        BackendType.ANTHROPIC,
        BackendType.OPENAI,
        BackendType.LOCAL,
        BackendType.MOCK
    ])
    
    # API Keys (loaded from environment)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Local backend settings
    local_model_path: Optional[str] = None
    ollama_host: str = "http://localhost:11434"
    use_ollama: bool = True
    
    # Timeout and retry settings
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_exponential_base: float = 2.0
    
    # Rate limiting
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_tpm: int = 100000  # Tokens per minute
    
    # Model defaults
    default_model: Dict[BackendType, str] = field(default_factory=lambda: {
        BackendType.ANTHROPIC: "claude-3-5-sonnet-20241022",
        BackendType.OPENAI: "gpt-4o",
        BackendType.LOCAL: "llama3.2",
        BackendType.MOCK: "mock-model"
    })
    
    default_embedding_model: Dict[BackendType, str] = field(default_factory=lambda: {
        BackendType.OPENAI: "text-embedding-3-small",
        BackendType.LOCAL: "nomic-embed-text",
        BackendType.MOCK: "mock-embedding"
    })
    
    # Generation defaults
    default_max_tokens: int = 1000
    default_temperature: float = 0.7
    
    # Mock backend settings (for testing)
    mock_delay_seconds: float = 0.1
    
    @classmethod
    def from_environment(cls) -> "BackendConfig":
        """
        Load configuration from environment variables.
        
        Environment variables:
            PRSM_PRIMARY_BACKEND: Primary backend type (anthropic, openai, local, mock)
            ANTHROPIC_API_KEY: Anthropic API key
            OPENAI_API_KEY: OpenAI API key
            PRSM_LOCAL_MODEL_PATH: Path to local model
            PRSM_OLLAMA_HOST: Ollama server host
            PRSM_USE_OLLAMA: Whether to use Ollama (true/false)
            PRSM_BACKEND_TIMEOUT: Request timeout in seconds
            PRSM_BACKEND_MAX_RETRIES: Maximum retry attempts
            PRSM_BACKEND_RETRY_DELAY: Initial retry delay in seconds
            PRSM_RATE_LIMIT_RPM: Rate limit in requests per minute
            PRSM_MOCK_DELAY: Mock backend delay in seconds
        
        Returns:
            BackendConfig: Configuration loaded from environment
        """
        
        def get_backend_type(key: str, default: BackendType) -> BackendType:
            value = os.getenv(key, default.value)
            try:
                return BackendType(value.lower())
            except ValueError:
                return default
        
        def get_bool_env(key: str, default: bool) -> bool:
            value = os.getenv(key, str(default).lower())
            return value.lower() in ("true", "1", "yes", "on")
        
        def get_int_env(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        def get_float_env(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        def parse_fallback_chain(env_value: Optional[str]) -> List[BackendType]:
            if not env_value:
                return [
                    BackendType.ANTHROPIC,
                    BackendType.OPENAI,
                    BackendType.LOCAL,
                    BackendType.MOCK
                ]
            
            backends = []
            for part in env_value.split(","):
                part = part.strip().lower()
                try:
                    backends.append(BackendType(part))
                except ValueError:
                    continue
            return backends if backends else [BackendType.MOCK]
        
        return cls(
            primary_backend=get_backend_type("PRSM_PRIMARY_BACKEND", BackendType.MOCK),
            fallback_chain=parse_fallback_chain(os.getenv("PRSM_FALLBACK_CHAIN")),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            local_model_path=os.getenv("PRSM_LOCAL_MODEL_PATH"),
            ollama_host=os.getenv("PRSM_OLLAMA_HOST", "http://localhost:11434"),
            use_ollama=get_bool_env("PRSM_USE_OLLAMA", True),
            timeout_seconds=get_int_env("PRSM_BACKEND_TIMEOUT", 120),
            max_retries=get_int_env("PRSM_BACKEND_MAX_RETRIES", 3),
            retry_delay_seconds=get_float_env("PRSM_BACKEND_RETRY_DELAY", 1.0),
            rate_limit_rpm=get_int_env("PRSM_RATE_LIMIT_RPM", 60),
            mock_delay_seconds=get_float_env("PRSM_MOCK_DELAY", 0.1)
        )
    
    @classmethod
    def from_file(cls, path: str) -> "BackendConfig":
        """
        Load configuration from YAML or JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            BackendConfig: Configuration loaded from file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            if path.endswith('.json'):
                data = json.load(f)
            else:
                # Try to load YAML, fall back to JSON
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    # If PyYAML not installed, try JSON
                    f.seek(0)
                    data = json.load(f)
        
        # Convert backend type strings to enums
        if 'primary_backend' in data:
            data['primary_backend'] = BackendType(data['primary_backend'])
        if 'fallback_chain' in data:
            data['fallback_chain'] = [BackendType(b) for b in data['fallback_chain']]
        
        # Convert default_model dict keys to enums
        if 'default_model' in data:
            data['default_model'] = {
                BackendType(k): v for k, v in data['default_model'].items()
            }
        if 'default_embedding_model' in data:
            data['default_embedding_model'] = {
                BackendType(k): v for k, v in data['default_embedding_model'].items()
            }
        
        return cls(**data)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List[str]: List of validation issues (empty if valid)
        """
        issues = []
        
        # Check API keys for cloud providers
        if self.primary_backend == BackendType.ANTHROPIC and not self.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY not set but Anthropic is primary backend")
        
        if self.primary_backend == BackendType.OPENAI and not self.openai_api_key:
            issues.append("OPENAI_API_KEY not set but OpenAI is primary backend")
        
        # Validate fallback chain
        for backend in self.fallback_chain:
            if backend == BackendType.ANTHROPIC and not self.anthropic_api_key:
                issues.append(f"Anthropic in fallback chain but ANTHROPIC_API_KEY not set")
            if backend == BackendType.OPENAI and not self.openai_api_key:
                issues.append(f"OpenAI in fallback chain but OPENAI_API_KEY not set")
        
        # Validate timeout
        if self.timeout_seconds < 1:
            issues.append(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")
        
        # Validate retries
        if self.max_retries < 0:
            issues.append(f"max_retries must be >= 0, got {self.max_retries}")
        
        # Validate rate limits
        if self.rate_limit_rpm < 1:
            issues.append(f"rate_limit_rpm must be >= 1, got {self.rate_limit_rpm}")
        
        return issues
    
    def is_valid(self) -> bool:
        """
        Check if configuration is valid.
        
        Returns:
            bool: True if configuration has no issues
        """
        return len(self.validate()) == 0
    
    def get_default_model(self, backend_type: BackendType) -> str:
        """
        Get the default model for a backend type.
        
        Args:
            backend_type: The backend type
            
        Returns:
            str: The default model ID for that backend
        """
        return self.default_model.get(backend_type, "unknown")
    
    def get_default_embedding_model(self, backend_type: BackendType) -> Optional[str]:
        """
        Get the default embedding model for a backend type.
        
        Args:
            backend_type: The backend type
            
        Returns:
            Optional[str]: The default embedding model ID, or None if not supported
        """
        return self.default_embedding_model.get(backend_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            "primary_backend": self.primary_backend.value,
            "fallback_chain": [b.value for b in self.fallback_chain],
            "anthropic_api_key": "***" if self.anthropic_api_key else None,
            "openai_api_key": "***" if self.openai_api_key else None,
            "local_model_path": self.local_model_path,
            "ollama_host": self.ollama_host,
            "use_ollama": self.use_ollama,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "retry_exponential_base": self.retry_exponential_base,
            "rate_limit_rpm": self.rate_limit_rpm,
            "rate_limit_tpm": self.rate_limit_tpm,
            "default_model": {k.value: v for k, v in self.default_model.items()},
            "default_embedding_model": {k.value: v for k, v in self.default_embedding_model.items()},
            "default_max_tokens": self.default_max_tokens,
            "default_temperature": self.default_temperature,
            "mock_delay_seconds": self.mock_delay_seconds
        }
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return (
            f"BackendConfig("
            f"primary={self.primary_backend.value}, "
            f"fallback={[b.value for b in self.fallback_chain]}, "
            f"timeout={self.timeout_seconds}s, "
            f"retries={self.max_retries})"
        )
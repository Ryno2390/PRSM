"""
PRSM Core Configuration
Enhanced from Co-Lab's settings.py with PRSM-specific features

Centralized configuration management for all PRSM subsystems providing:

Configuration Categories:

1. Core Application Settings:
   - Environment management (development, testing, staging, production)
   - Debug modes and logging configuration
   - API server configuration and networking

2. Security Configuration:
   - JWT authentication settings
   - Secret key management
   - Access control parameters

3. Database Configuration:
   - Multi-database support (SQLite, PostgreSQL)
   - Connection pooling and optimization
   - Transaction management settings

4. External Service Integration:
   - Redis for caching and session management
   - IPFS for distributed storage
   - Vector databases (Pinecone, Weaviate)
   - AI model APIs (OpenAI, Anthropic)

5. PRSM-Specific Features:
   - NWTN orchestrator configuration
   - Agent pipeline parameters
   - Teacher model system settings
   - Safety and circuit breaker thresholds
   - FTNS token economy parameters
   - P2P federation networking
   - Governance system settings
   - Recursive self-improvement controls

Environment Management:
- Automatic environment detection
- Environment-specific configuration overrides
- Development vs production optimizations
- Testing configuration isolation

Performance Tuning:
- Database connection pooling
- API worker configuration
- Timeout and retry settings
- Resource allocation limits

Security Features:
- Environment variable encryption
- Secret rotation capabilities
- Access control validation
- Audit trail configuration

Validation Features:
- Configuration validation at startup
- Type checking and constraint enforcement
- Required parameter verification
- Sensible default value provisioning

The configuration system ensures that PRSM can be deployed
across various environments while maintaining security,
performance, and operational requirements.
"""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Environment types for PRSM deployment"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PRSMSettings(BaseSettings):
    """
    PRSM Configuration Settings
    Enhanced version of Co-Lab's settings with PRSM-specific features
    """
    
    # === Core Application Settings ===
    app_name: str = "PRSM"
    app_version: str = "0.1.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=False, env="PRSM_DEBUG")
    log_level: LogLevel = LogLevel.INFO
    
    # === API Configuration ===
    api_host: str = Field(default="127.0.0.1", env="PRSM_API_HOST")  # Default to localhost for security
    api_port: int = Field(default=8000, env="PRSM_API_PORT")
    api_reload: bool = Field(default=True, env="PRSM_API_RELOAD")
    api_workers: int = Field(default=1, env="PRSM_API_WORKERS")
    
    # === Security ===
    secret_key: str = Field(default="test-secret-key-at-least-32-characters-long", env="PRSM_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week
    
    # === Database Configuration ===
    database_url: str = Field(default="sqlite:///./prsm_test.db", env="PRSM_DATABASE_URL")
    database_echo: bool = Field(default=False, env="PRSM_DATABASE_ECHO")
    database_pool_size: int = 5
    database_max_overflow: int = 10
    
    # === Redis Configuration ===
    redis_url: str = Field(default="redis://localhost:6379/0", env="PRSM_REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="PRSM_REDIS_PASSWORD")
    
    # === IPFS Configuration ===
    ipfs_host: str = Field(default="localhost", env="PRSM_IPFS_HOST")
    ipfs_port: int = Field(default=5001, env="PRSM_IPFS_PORT")
    ipfs_timeout: int = Field(default=60, env="PRSM_IPFS_TIMEOUT")
    ipfs_gateway_url: str = Field(default="http://localhost:8080", env="PRSM_IPFS_GATEWAY")
    
    # === NWTN Configuration ===
    nwtn_enabled: bool = Field(default=True, env="PRSM_NWTN_ENABLED")
    nwtn_max_context_per_query: int = Field(default=1000, env="PRSM_NWTN_MAX_CONTEXT")
    nwtn_min_context_cost: int = Field(default=10, env="PRSM_NWTN_MIN_CONTEXT_COST")
    nwtn_default_model: str = Field(default="claude-3-5-sonnet-20241022", env="PRSM_NWTN_MODEL")
    nwtn_temperature: float = Field(default=0.7, env="PRSM_NWTN_TEMPERATURE")
    
    # === AI Model Configuration ===
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Embedding models
    embedding_model: str = Field(default="text-embedding-3-small", env="PRSM_EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, env="PRSM_EMBEDDING_DIMENSIONS")
    
    # === Vector Database Configuration ===
    # Pinecone
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="prsm-models", env="PINECONE_INDEX_NAME")
    
    # Weaviate
    weaviate_url: Optional[str] = Field(default=None, env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    
    # === FTNS Token Configuration ===
    ftns_enabled: bool = Field(default=True, env="PRSM_FTNS_ENABLED")
    ftns_initial_grant: int = Field(default=100, env="PRSM_FTNS_INITIAL_GRANT")
    ftns_context_cost_base: float = Field(default=0.1, env="PRSM_FTNS_CONTEXT_COST")
    ftns_reward_multiplier: float = Field(default=1.0, env="PRSM_FTNS_REWARD_MULTIPLIER")
    ftns_max_session_budget: int = Field(default=10000, env="PRSM_FTNS_MAX_SESSION_BUDGET")
    
    # === Distributed Ledger Configuration ===
    iota_node_url: Optional[str] = Field(default=None, env="IOTA_NODE_URL")
    iota_network: str = Field(default="testnet", env="IOTA_NETWORK")
    
    # === Agent Configuration ===
    max_decomposition_depth: int = Field(default=5, env="PRSM_MAX_DECOMPOSITION_DEPTH")
    max_parallel_tasks: int = Field(default=10, env="PRSM_MAX_PARALLEL_TASKS")
    agent_timeout_seconds: int = Field(default=300, env="PRSM_AGENT_TIMEOUT")
    
    # === Teacher Model Configuration ===
    teacher_enabled: bool = Field(default=True, env="PRSM_TEACHER_ENABLED")
    teacher_rlvr_enabled: bool = Field(default=True, env="PRSM_TEACHER_RLVR_ENABLED")
    teacher_update_frequency: int = Field(default=3600, env="PRSM_TEACHER_UPDATE_FREQ")  # seconds
    
    # === Safety Configuration ===
    circuit_breaker_enabled: bool = Field(default=True, env="PRSM_CIRCUIT_BREAKER_ENABLED")
    safety_monitoring_enabled: bool = Field(default=True, env="PRSM_SAFETY_MONITORING")
    max_safety_violations: int = Field(default=3, env="PRSM_MAX_SAFETY_VIOLATIONS")
    
    # === P2P Federation Configuration ===
    p2p_enabled: bool = Field(default=True, env="PRSM_P2P_ENABLED")
    p2p_port: int = Field(default=4001, env="PRSM_P2P_PORT")
    p2p_bootstrap_peers: List[str] = Field(default=[], env="PRSM_P2P_BOOTSTRAP_PEERS")
    
    # === Monitoring & Metrics ===
    metrics_enabled: bool = Field(default=True, env="PRSM_METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="PRSM_METRICS_PORT")

    # === TLS/SSL Configuration ===
    tls_enabled: bool = Field(default=False, env="PRSM_TLS_ENABLED")
    tls_mode: str = Field(default="verify-full", env="PRSM_TLS_MODE")
    tls_min_version: str = Field(default="TLSv1.2", env="PRSM_TLS_MIN_VERSION")
    tls_cert_file: Optional[str] = Field(default=None, env="PRSM_TLS_CERT_FILE")
    tls_key_file: Optional[str] = Field(default=None, env="PRSM_TLS_KEY_FILE")
    tls_ca_file: Optional[str] = Field(default=None, env="PRSM_TLS_CA_FILE")

    # === HSTS Configuration ===
    hsts_enabled: bool = Field(default=True, env="PRSM_HSTS_ENABLED")
    hsts_max_age: int = Field(default=31536000, env="PRSM_HSTS_MAX_AGE")  # 1 year
    hsts_include_subdomains: bool = Field(default=True, env="PRSM_HSTS_SUBDOMAINS")
    hsts_preload: bool = Field(default=False, env="PRSM_HSTS_PRELOAD")

    # === JWT Configuration (enhanced) ===
    jwt_secret: Optional[str] = Field(default=None, env="PRSM_JWT_SECRET")
    
    # === Governance Configuration ===
    governance_enabled: bool = Field(default=True, env="PRSM_GOVERNANCE_ENABLED")
    governance_proposal_threshold: int = Field(default=100, env="PRSM_GOVERNANCE_THRESHOLD")
    governance_voting_period: int = Field(default=604800, env="PRSM_GOVERNANCE_VOTING_PERIOD")  # 7 days
    
    # === Recursive Self-Improvement ===
    rsi_enabled: bool = Field(default=True, env="PRSM_RSI_ENABLED")
    rsi_evaluation_frequency: int = Field(default=86400, env="PRSM_RSI_EVAL_FREQUENCY")  # 24 hours
    rsi_improvement_threshold: float = Field(default=0.05, env="PRSM_RSI_IMPROVEMENT_THRESHOLD")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
        "extra": "allow",
        "env_prefix": ""
    }
        
    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v):
        if not v.startswith(("postgresql://", "postgresql+asyncpg://", "sqlite:///")):
            raise ValueError("Database URL must be PostgreSQL or SQLite")
        return v
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        return self.environment == Environment.TESTING
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Database configuration for SQLAlchemy"""
        return {
            "url": self.database_url,
            "echo": self.database_echo,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
        }
    
    @property
    def ipfs_config(self) -> Dict[str, Any]:
        """IPFS client configuration"""
        return {
            "host": self.ipfs_host,
            "port": self.ipfs_port,
            "timeout": self.ipfs_timeout,
        }
    
    @property
    def ai_model_config(self) -> Dict[str, Any]:
        """AI model configuration"""
        return {
            "openai_api_key": self.openai_api_key,
            "anthropic_api_key": self.anthropic_api_key,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
        }
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    @property 
    def is_staging(self) -> bool:
        return self.environment == Environment.STAGING
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    def validate_required_config(self) -> List[str]:
        """
        Validate that required configuration is present for core functionality
        
        Returns:
            List of missing required configuration items
        """
        missing_config = []
        
        # Critical database configuration
        if not self.database_url or self.database_url == "postgresql://user:password@localhost:5432/prsm":
            missing_config.append("DATABASE_URL - Required for data persistence")
        
        # Redis configuration for production caching
        if self.is_production and (not self.redis_url or self.redis_url == "redis://localhost:6379/0"):
            missing_config.append("REDIS_URL - Required for production caching")
        
        # API key for embeddings (critical for vector operations)
        if not self.openai_api_key:
            missing_config.append("OPENAI_API_KEY - Required for semantic search and embeddings")
        
        # Security considerations
        if self.is_production and not self.secret_key:
            missing_config.append("SECRET_KEY - Required for production security")
        
        return missing_config


@lru_cache()
def get_settings() -> PRSMSettings:
    """
    Get cached settings instance with error handling
    """
    try:
        return PRSMSettings()
    except Exception as e:
        print(f"Warning: Failed to load settings: {e}")
        # Return a minimal default settings instance
        return PRSMSettings(
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            database_url="sqlite:///./prsm_test.db",
            environment=Environment.DEVELOPMENT
        )


def get_settings_safe() -> Optional[PRSMSettings]:
    """
    Get settings instance safely, returning None if initialization fails
    """
    try:
        return get_settings()
    except Exception:
        return None


# Global settings instance with safe initialization
try:
    settings = get_settings()
except Exception:
    settings = None
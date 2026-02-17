"""
Configuration Schemas
======================

Pydantic schemas for validating all PRSM configuration sections.
"""

from pydantic import BaseModel, Field, validator, model_validator
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal
from enum import Enum
import os
from pathlib import Path


class LogLevelEnum(str, Enum):
    """Logging level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseTypeEnum(str, Enum):
    """Database type enumeration"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"


class BaseConfigSchema(BaseModel):
    """Base configuration schema with common validation"""
    
    class Config:
        validate_assignment = True
        use_enum_values = True
        allow_population_by_field_name = True
        extra = "forbid"


# Core component configurations
class NWTNConfig(BaseConfigSchema):
    """NWTN reasoning system configuration"""
    
    # Processing settings
    max_concurrent_queries: int = Field(10, ge=1, le=100, description="Maximum concurrent queries")
    default_thinking_mode: str = Field("intermediate", description="Default thinking mode")
    default_verbosity: str = Field("standard", description="Default verbosity level")
    max_processing_time: int = Field(3600, ge=60, le=7200, description="Max processing time in seconds")
    
    # Analogical reasoning
    analogical_chain_max_depth: int = Field(6, ge=1, le=10, description="Max analogical chain depth")
    analogical_similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    enable_breakthrough_discovery: bool = Field(False, description="Enable breakthrough discovery mode")
    
    # Reasoning engines
    enabled_engines: List[str] = Field(
        ["logical", "creative", "analytical", "intuitive", "analogical"],
        description="Enabled reasoning engines"
    )
    engine_weights: Dict[str, float] = Field(
        {
            "logical": 1.0,
            "creative": 0.8,
            "analytical": 1.0,
            "intuitive": 0.6,
            "analogical": 0.7
        },
        description="Engine weighting factors"
    )
    
    # Performance settings
    cache_enabled: bool = Field(True, description="Enable result caching")
    cache_ttl_seconds: int = Field(3600, ge=60, description="Cache TTL in seconds")
    enable_parallel_processing: bool = Field(True, description="Enable parallel engine processing")
    
    @validator('enabled_engines')
    def validate_engines(cls, v):
        allowed_engines = ["logical", "creative", "analytical", "intuitive", "analogical"]
        for engine in v:
            if engine not in allowed_engines:
                raise ValueError(f"Invalid engine: {engine}. Allowed: {allowed_engines}")
        return v
    
    @validator('engine_weights')
    def validate_weights(cls, v):
        for engine, weight in v.items():
            if not 0.0 <= weight <= 2.0:
                raise ValueError(f"Engine weight for {engine} must be between 0.0 and 2.0")
        return v


class TokenomicsConfig(BaseConfigSchema):
    """FTNS tokenomics system configuration"""
    
    # Token settings
    initial_supply: Decimal = Field(Decimal("1000000"), gt=0, description="Initial FTNS supply")
    min_transaction_amount: Decimal = Field(Decimal("0.01"), gt=0, description="Minimum transaction amount")
    max_transaction_amount: Decimal = Field(Decimal("10000"), gt=0, description="Maximum transaction amount")
    
    # Pricing model
    base_query_cost: Decimal = Field(Decimal("1.0"), gt=0, description="Base cost per query")
    thinking_mode_multipliers: Dict[str, float] = Field(
        {
            "quick": 0.5,
            "intermediate": 1.0,
            "deep": 2.0
        },
        description="Thinking mode cost multipliers"
    )
    verbosity_multipliers: Dict[str, float] = Field(
        {
            "brief": 0.8,
            "standard": 1.0,
            "detailed": 1.3,
            "comprehensive": 1.6,
            "academic": 2.0
        },
        description="Verbosity level cost multipliers"
    )
    
    # User tier pricing
    tier_discounts: Dict[str, float] = Field(
        {
            "basic": 0.0,
            "standard": 0.1,
            "premium": 0.2,
            "enterprise": 0.3,
            "research": 0.5
        },
        description="Discount rates by user tier"
    )
    
    # Market dynamics
    enable_dynamic_pricing: bool = Field(True, description="Enable dynamic pricing")
    demand_surge_threshold: float = Field(0.8, ge=0.5, le=1.0, description="Demand surge threshold")
    surge_multiplier: float = Field(1.5, ge=1.0, le=3.0, description="Surge pricing multiplier")
    supply_adjustment_rate: float = Field(0.001, ge=0.0, le=0.01, description="Supply adjustment rate")
    
    # Transaction settings
    enable_micropayments: bool = Field(True, description="Enable micropayments")
    batch_transaction_threshold: int = Field(100, ge=1, description="Batch transaction threshold")
    transaction_fee_rate: float = Field(0.01, ge=0.0, le=0.1, description="Transaction fee rate")
    
    @validator('thinking_mode_multipliers')
    def validate_thinking_multipliers(cls, v):
        required_modes = ["quick", "intermediate", "deep"]
        for mode in required_modes:
            if mode not in v:
                raise ValueError(f"Missing thinking mode multiplier: {mode}")
        return v


class MarketplaceConfig(BaseConfigSchema):
    """Marketplace system configuration"""
    
    # Asset management
    max_assets_per_user: int = Field(100, ge=1, le=1000, description="Max assets per user")
    max_asset_size_mb: int = Field(100, ge=1, le=1000, description="Max asset size in MB")
    supported_asset_types: List[str] = Field(
        ["ai_model", "dataset", "tool", "service", "workflow", "knowledge_resource"],
        description="Supported asset types"
    )
    
    # Search and discovery
    search_results_per_page: int = Field(20, ge=1, le=100, description="Search results per page")
    max_search_term_length: int = Field(200, ge=1, le=500, description="Max search term length")
    enable_fuzzy_search: bool = Field(True, description="Enable fuzzy search")
    search_index_refresh_interval: int = Field(300, ge=60, description="Search index refresh interval")
    
    # Quality and rating
    min_quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum quality score")
    enable_user_ratings: bool = Field(True, description="Enable user ratings")
    min_ratings_for_score: int = Field(3, ge=1, description="Minimum ratings for quality score")
    
    # Monetization
    marketplace_fee_rate: float = Field(0.05, ge=0.0, le=0.2, description="Marketplace fee rate")
    creator_revenue_share: float = Field(0.8, ge=0.5, le=0.95, description="Creator revenue share")
    enable_revenue_sharing: bool = Field(True, description="Enable revenue sharing")
    
    @validator('creator_revenue_share')
    def validate_revenue_share(cls, v, values):
        marketplace_fee = values.get('marketplace_fee_rate', 0.05)
        if v + marketplace_fee > 1.0:
            raise ValueError("Creator revenue share + marketplace fee cannot exceed 100%")
        return v


class DatabaseConfig(BaseConfigSchema):
    """Database configuration"""
    
    # Connection settings
    type: DatabaseTypeEnum = Field(DatabaseTypeEnum.SQLITE, description="Database type")
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, ge=1, le=65535, description="Database port")
    database: str = Field("prsm", description="Database name")
    username: Optional[str] = Field(None, description="Database username")
    password: Optional[str] = Field(None, description="Database password")
    
    # Connection pooling
    pool_size: int = Field(20, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(30, ge=0, le=100, description="Max pool overflow")
    pool_timeout: int = Field(30, ge=5, le=300, description="Pool timeout in seconds")
    pool_recycle: int = Field(3600, ge=300, description="Pool recycle time in seconds")
    
    # Performance settings
    echo_sql: bool = Field(False, description="Echo SQL queries to logs")
    query_timeout: int = Field(30, ge=5, le=300, description="Query timeout in seconds")
    enable_query_cache: bool = Field(True, description="Enable query result caching")
    cache_size: int = Field(1000, ge=100, le=10000, description="Query cache size")
    
    # Backup and maintenance
    backup_enabled: bool = Field(True, description="Enable automatic backups")
    backup_interval_hours: int = Field(24, ge=1, le=168, description="Backup interval in hours")
    backup_retention_days: int = Field(30, ge=1, le=365, description="Backup retention in days")
    
    @validator('port')
    def validate_port(cls, v, values):
        db_type = values.get('type')
        if db_type == DatabaseTypeEnum.POSTGRESQL and v != 5432:
            # Custom PostgreSQL port is allowed but logged
            pass
        elif db_type == DatabaseTypeEnum.MYSQL and v not in [3306, 3307]:
            # Custom MySQL port warning
            pass
        return v


class SecurityConfig(BaseConfigSchema):
    """Security configuration"""
    
    # Authentication
    jwt_secret_key: str = Field(default="change-me-to-a-random-string-at-least-32-chars", min_length=32, description="JWT secret key")
    jwt_expiry_hours: int = Field(24, ge=1, le=168, description="JWT expiry in hours")
    enable_refresh_tokens: bool = Field(True, description="Enable refresh tokens")
    refresh_token_expiry_days: int = Field(30, ge=1, le=90, description="Refresh token expiry in days")
    
    # API security
    enable_rate_limiting: bool = Field(True, description="Enable API rate limiting")
    rate_limit_per_minute: int = Field(100, ge=10, le=10000, description="Requests per minute limit")
    enable_cors: bool = Field(True, description="Enable CORS")
    allowed_origins: List[str] = Field(["*"], description="Allowed CORS origins")
    
    # Input validation
    max_request_size_mb: int = Field(10, ge=1, le=100, description="Max request size in MB")
    enable_input_sanitization: bool = Field(True, description="Enable input sanitization")
    blocked_patterns: List[str] = Field(
        ["<script", "javascript:", "vbscript:", "onload=", "onerror="],
        description="Blocked input patterns"
    )
    
    # Encryption
    encryption_algorithm: str = Field("AES-256-GCM", description="Encryption algorithm")
    enable_field_encryption: bool = Field(True, description="Enable sensitive field encryption")
    encrypted_fields: List[str] = Field(
        ["password", "api_key", "secret", "token"],
        description="Fields to encrypt"
    )
    
    # Audit and monitoring
    enable_audit_logging: bool = Field(True, description="Enable audit logging")
    log_failed_attempts: bool = Field(True, description="Log failed authentication attempts")
    failed_attempt_threshold: int = Field(5, ge=1, le=20, description="Failed attempt threshold")
    account_lockout_duration_minutes: int = Field(30, ge=5, le=1440, description="Account lockout duration")
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v


class APIConfig(BaseConfigSchema):
    """API server configuration"""
    
    # Server settings
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, ge=1, le=65535, description="Server port")
    debug: bool = Field(False, description="Debug mode")
    reload: bool = Field(False, description="Auto-reload on changes")
    
    # Performance
    workers: int = Field(4, ge=1, le=32, description="Number of worker processes")
    max_connections: int = Field(1000, ge=100, le=10000, description="Max concurrent connections")
    keepalive_timeout: int = Field(5, ge=1, le=60, description="Keep-alive timeout in seconds")
    client_timeout: int = Field(60, ge=10, le=300, description="Client timeout in seconds")
    
    # Request handling
    max_request_size: int = Field(10485760, ge=1024, description="Max request size in bytes")  # 10MB
    request_timeout: int = Field(300, ge=30, le=3600, description="Request timeout in seconds")
    enable_gzip: bool = Field(True, description="Enable gzip compression")
    
    # API versioning
    api_version: str = Field("v1", description="API version")
    enable_versioning: bool = Field(True, description="Enable API versioning")
    deprecated_versions: List[str] = Field([], description="Deprecated API versions")
    
    # Documentation
    enable_docs: bool = Field(True, description="Enable API documentation")
    docs_url: str = Field("/docs", description="Documentation URL path")
    redoc_url: str = Field("/redoc", description="ReDoc URL path")
    openapi_url: str = Field("/openapi.json", description="OpenAPI schema URL")


class LoggingConfig(BaseConfigSchema):
    """Logging configuration"""
    
    # General settings
    level: LogLevelEnum = Field(LogLevelEnum.INFO, description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    date_format: str = Field("%Y-%m-%d %H:%M:%S", description="Date format string")
    
    # File logging
    enable_file_logging: bool = Field(True, description="Enable file logging")
    log_file_path: str = Field("logs/prsm.log", description="Log file path")
    max_file_size_mb: int = Field(100, ge=1, le=1000, description="Max log file size in MB")
    backup_count: int = Field(5, ge=1, le=20, description="Number of backup log files")
    
    # Console logging
    enable_console_logging: bool = Field(True, description="Enable console logging")
    console_level: LogLevelEnum = Field(LogLevelEnum.INFO, description="Console logging level")
    
    # Structured logging
    enable_json_logging: bool = Field(False, description="Enable JSON structured logging")
    include_extra_fields: bool = Field(True, description="Include extra fields in logs")
    
    # Component-specific logging
    component_levels: Dict[str, str] = Field(
        {
            "nwtn": "INFO",
            "tokenomics": "INFO", 
            "marketplace": "INFO",
            "database": "WARNING",
            "security": "INFO"
        },
        description="Per-component logging levels"
    )
    
    # Performance logging
    enable_performance_logging: bool = Field(True, description="Enable performance logging")
    slow_query_threshold_ms: int = Field(1000, ge=100, description="Slow query threshold in ms")
    
    @validator('log_file_path')
    def validate_log_path(cls, v):
        log_dir = Path(v).parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        return v


class SystemConfig(BaseConfigSchema):
    """System-wide configuration"""
    
    # Environment
    environment: str = Field("development", description="Environment name")
    debug: bool = Field(False, description="System debug mode")
    testing: bool = Field(False, description="Testing mode")
    
    # Paths
    data_directory: str = Field("data", description="Data directory path")
    temp_directory: str = Field("temp", description="Temporary directory path")
    log_directory: str = Field("logs", description="Log directory path")
    cache_directory: str = Field("cache", description="Cache directory path")
    
    # Resource limits
    max_memory_usage_mb: int = Field(8192, ge=512, description="Max memory usage in MB")
    max_cpu_usage_percent: int = Field(80, ge=10, le=100, description="Max CPU usage percentage")
    max_disk_usage_gb: int = Field(100, ge=1, description="Max disk usage in GB")
    
    # Monitoring
    enable_health_checks: bool = Field(True, description="Enable health checks")
    health_check_interval: int = Field(60, ge=10, description="Health check interval in seconds")
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    metrics_port: int = Field(9090, ge=1024, le=65535, description="Metrics server port")
    
    @validator('data_directory', 'temp_directory', 'log_directory', 'cache_directory')
    def validate_directories(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class P2PConfig(BaseConfigSchema):
    """P2P networking configuration"""

    enabled: bool = Field(True, description="Enable P2P networking")
    listen_host: str = Field("0.0.0.0", description="P2P listen host")
    listen_port: int = Field(9001, ge=1024, le=65535, description="P2P listen port")
    bootstrap_nodes: List[str] = Field([], description="Bootstrap node addresses (host:port)")
    max_peers: int = Field(50, ge=1, le=500, description="Maximum peer connections")
    gossip_fanout: int = Field(3, ge=1, le=10, description="Gossip fanout (peers per message)")
    gossip_ttl: int = Field(5, ge=1, le=20, description="Gossip message TTL (hops)")
    heartbeat_interval: float = Field(30.0, ge=5.0, le=300.0, description="Heartbeat interval in seconds")


# Main configuration schema
class PRSMConfig(BaseConfigSchema):
    """Main PRSM configuration schema"""
    
    # Component configurations
    nwtn: NWTNConfig = Field(default_factory=NWTNConfig, description="NWTN configuration")
    tokenomics: TokenomicsConfig = Field(default_factory=TokenomicsConfig, description="Tokenomics configuration")
    marketplace: MarketplaceConfig = Field(default_factory=MarketplaceConfig, description="Marketplace configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    api: APIConfig = Field(default_factory=APIConfig, description="API configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    system: SystemConfig = Field(default_factory=SystemConfig, description="System configuration")
    p2p: P2PConfig = Field(default_factory=P2PConfig, description="P2P networking configuration")

    # Global settings
    app_name: str = Field("PRSM", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    
    # === Flat attribute compatibility ===
    # These properties allow code that expects PRSMSettings-style flat access
    # (settings.environment, settings.debug, etc.) to work with PRSMConfig.

    @property
    def environment(self):
        """Return environment as an object with .value for compatibility."""
        env_str = self.system.environment
        class _Env:
            def __init__(self, v): self.value = v
            def __str__(self): return self.value
            def __eq__(self, other): return self.value == (other.value if hasattr(other, 'value') else other)
        return _Env(env_str)

    @property
    def debug(self):
        return self.system.debug

    @property
    def is_production(self):
        return self.system.environment == "production"

    @property
    def is_development(self):
        return self.system.environment == "development" or self.system.environment != "production"

    @property
    def is_testing(self):
        return self.system.testing

    @property
    def jwt_algorithm(self):
        return "HS256"

    @property
    def jwt_secret(self):
        return self.security.jwt_secret_key if self.security else None

    @property
    def secret_key(self):
        return self.security.jwt_secret_key if self.security else "dev-secret-key"

    @property
    def database_url(self):
        return f"{self.database.type.value}://{self.database.host}:{self.database.port}/{self.database.name}" if self.database else "sqlite:///prsm.db"

    @property
    def redis_url(self):
        return "redis://localhost:6379/0"

    @property
    def ipfs_host(self):
        return "127.0.0.1"

    @property
    def ipfs_port(self):
        return 5001

    @property
    def nwtn_enabled(self):
        return True

    @property
    def nwtn_default_model(self):
        return self.nwtn.default_model if self.nwtn and hasattr(self.nwtn, 'default_model') else "default"

    @property
    def ftns_enabled(self):
        return True

    @property
    def p2p_enabled(self):
        return self.p2p.enabled if self.p2p else False

    @property
    def governance_enabled(self):
        return True

    @property
    def rsi_enabled(self):
        return False

    @property
    def ftns_initial_grant(self):
        return 100

    @property
    def embedding_model(self):
        return "text-embedding-3-small"

    @property
    def embedding_dimensions(self):
        return 1536

    def validate_required_config(self):
        """Validate required configuration, returns list of missing items."""
        missing = []
        if self.security and self.security.jwt_secret_key == "change-me-to-a-random-string-at-least-32-chars":
            missing.append("PRSM_SECRET_KEY (using default)")
        return missing

    @model_validator(mode='after')
    def validate_component_consistency(self):
        """Validate consistency between component configurations"""
        
        # Ensure security JWT secret is provided
        if self.security and not self.security.jwt_secret_key:
            raise ValueError("Security JWT secret key is required")
        
        # Validate database and API consistency
        if self.api and self.database:
            # Ensure sufficient database connections for API workers
            if self.database.pool_size < self.api.workers:
                raise ValueError("Database pool size should be at least equal to API workers")
        
        return self
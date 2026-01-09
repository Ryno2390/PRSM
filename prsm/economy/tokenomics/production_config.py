"""
FTNS Production Configuration
Production-ready configuration and deployment utilities for FTNS tokenomics system

This module provides comprehensive production configuration including:
- Database connection management with connection pooling
- Environment-based configuration with security best practices
- Monitoring and observability setup
- Performance optimization settings
- Emergency response configuration
- Backup and recovery procedures
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import timedelta

import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool


@dataclass
class DatabaseConfig:
    """Database configuration for production"""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    echo: bool = False
    echo_pool: bool = False
    
    # Connection retry settings
    connect_retries: int = 3
    retry_delay: float = 1.0
    
    # Performance tuning
    statement_timeout: int = 60000  # 60 seconds
    lock_timeout: int = 30000       # 30 seconds


@dataclass
class SecurityConfig:
    """Security configuration for production"""
    secret_key: str
    encryption_key: str
    jwt_secret: Optional[str] = None
    
    # Access control
    api_rate_limit: int = 1000  # requests per hour
    max_concurrent_users: int = 1000
    
    # Audit settings
    audit_log_enabled: bool = True
    sensitive_data_masking: bool = True
    
    # Emergency access
    emergency_override_enabled: bool = True
    emergency_contacts: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    log_level: str = "INFO"
    structured_logging: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    
    # Health checks
    health_check_interval: int = 60  # seconds
    deep_health_check_interval: int = 300  # 5 minutes
    
    # Performance monitoring
    slow_query_threshold: float = 1.0  # seconds
    memory_usage_threshold: float = 0.8  # 80%
    
    # Alerting
    alert_webhook_url: Optional[str] = None
    alert_email: Optional[str] = None


@dataclass
class EconomicConfig:
    """Economic parameters for production"""
    # Phase 2: Dynamic Supply
    initial_appreciation_rate: float = 0.5  # 50% annually
    target_appreciation_rate: float = 0.02  # 2% annually
    max_daily_adjustment: float = 0.001     # 0.1% daily
    volatility_damping_factor: float = 0.5
    
    # Phase 3: Anti-Hoarding
    target_velocity: float = 1.2  # Monthly velocity target
    base_demurrage_rate: float = 0.002    # 0.2% monthly
    max_demurrage_rate: float = 0.01      # 1.0% monthly
    grace_period_days: int = 90
    min_fee_threshold: Decimal = Decimal("0.001")
    
    # Phase 4: Emergency Protocols
    price_crash_threshold: float = 0.4     # 40% drop
    volume_spike_threshold: float = 5.0    # 5x normal
    oracle_deviation_threshold: float = 0.1  # 10% deviation
    emergency_response_enabled: bool = True
    
    # Governance integration
    governance_required_threshold: float = 0.05  # 5% supply change
    fast_track_voting_hours: int = 6


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes
    cache_max_size: int = 10000
    
    # Background tasks
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # 5 minutes
    
    # Database optimization
    batch_size: int = 1000
    bulk_operation_chunk_size: int = 500
    
    # Memory management
    gc_threshold: int = 1000000  # Force GC after 1M operations
    max_memory_usage: int = 8 * 1024 * 1024 * 1024  # 8GB


class ProductionConfig:
    """Production configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment and config file"""
        
        # Database configuration
        self.database = DatabaseConfig(
            url=self._get_env("FTNS_DATABASE_URL", required=True),
            pool_size=int(self._get_env("FTNS_DATABASE_POOL_SIZE", "20")),
            max_overflow=int(self._get_env("FTNS_DATABASE_MAX_OVERFLOW", "30")),
            pool_timeout=int(self._get_env("FTNS_DATABASE_POOL_TIMEOUT", "30")),
            echo=self._get_env("FTNS_DATABASE_ECHO", "false").lower() == "true"
        )
        
        # Security configuration
        self.security = SecurityConfig(
            secret_key=self._get_env("FTNS_SECRET_KEY", required=True),
            encryption_key=self._get_env("FTNS_ENCRYPTION_KEY", required=True),
            jwt_secret=self._get_env("FTNS_JWT_SECRET"),
            api_rate_limit=int(self._get_env("FTNS_API_RATE_LIMIT", "1000")),
            max_concurrent_users=int(self._get_env("FTNS_MAX_CONCURRENT_USERS", "1000")),
            emergency_contacts=self._get_env("FTNS_EMERGENCY_CONTACTS", "").split(",")
        )
        
        # Monitoring configuration
        self.monitoring = MonitoringConfig(
            log_level=self._get_env("FTNS_LOG_LEVEL", "INFO"),
            metrics_enabled=self._get_env("FTNS_METRICS_ENABLED", "true").lower() == "true",
            health_check_interval=int(self._get_env("FTNS_HEALTH_CHECK_INTERVAL", "60")),
            alert_webhook_url=self._get_env("FTNS_ALERT_WEBHOOK_URL"),
            alert_email=self._get_env("FTNS_ALERT_EMAIL")
        )
        
        # Economic configuration
        self.economic = EconomicConfig(
            initial_appreciation_rate=float(self._get_env("FTNS_INITIAL_APPRECIATION_RATE", "0.5")),
            target_appreciation_rate=float(self._get_env("FTNS_TARGET_APPRECIATION_RATE", "0.02")),
            max_daily_adjustment=float(self._get_env("FTNS_MAX_DAILY_ADJUSTMENT", "0.001")),
            target_velocity=float(self._get_env("FTNS_TARGET_VELOCITY", "1.2")),
            grace_period_days=int(self._get_env("FTNS_GRACE_PERIOD_DAYS", "90")),
            emergency_response_enabled=self._get_env("FTNS_EMERGENCY_RESPONSE_ENABLED", "true").lower() == "true"
        )
        
        # Performance configuration
        self.performance = PerformanceConfig(
            cache_enabled=self._get_env("FTNS_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(self._get_env("FTNS_CACHE_TTL", "300")),
            max_concurrent_tasks=int(self._get_env("FTNS_MAX_CONCURRENT_TASKS", "10")),
            batch_size=int(self._get_env("FTNS_BATCH_SIZE", "1000"))
        )
        
        # Environment-specific settings
        self.environment = self._get_env("FTNS_ENVIRONMENT", "production")
        self.debug = self._get_env("FTNS_DEBUG", "false").lower() == "true"
        self.testing = self._get_env("FTNS_TESTING", "false").lower() == "true"
    
    def _get_env(self, key: str, default: Optional[str] = None, required: bool = False) -> str:
        """Get environment variable with validation"""
        value = os.getenv(key, default)
        
        if required and value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        
        return value
    
    async def create_database_engine(self):
        """Create production database engine with optimal settings"""
        
        engine = create_async_engine(
            self.database.url,
            poolclass=QueuePool,
            pool_size=self.database.pool_size,
            max_overflow=self.database.max_overflow,
            pool_timeout=self.database.pool_timeout,
            pool_recycle=self.database.pool_recycle,
            echo=self.database.echo,
            echo_pool=self.database.echo_pool,
            
            # Performance optimizations
            connect_args={
                "statement_timeout": self.database.statement_timeout,
                "lock_timeout": self.database.lock_timeout,
                "application_name": "ftns_tokenomics"
            }
        )
        
        return engine
    
    async def create_session_factory(self):
        """Create session factory for database operations"""
        engine = await self.create_database_engine()
        
        return sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    def setup_logging(self):
        """Configure structured logging for production"""
        
        if self.monitoring.structured_logging:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True
            )
        
        # Configure standard logging
        logging.basicConfig(
            level=getattr(logging, self.monitoring.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Suppress noisy loggers
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
    
    def get_health_check_config(self) -> Dict[str, Any]:
        """Get health check configuration"""
        return {
            "enabled": True,
            "interval": self.monitoring.health_check_interval,
            "deep_check_interval": self.monitoring.deep_health_check_interval,
            "checks": {
                "database": True,
                "memory": True,
                "disk": True,
                "external_services": True
            },
            "thresholds": {
                "response_time": 1.0,  # seconds
                "memory_usage": 0.8,   # 80%
                "disk_usage": 0.9,     # 90%
                "error_rate": 0.01     # 1%
            }
        }
    
    def get_backup_config(self) -> Dict[str, Any]:
        """Get backup and recovery configuration"""
        return {
            "enabled": True,
            "schedule": "0 2 * * *",  # Daily at 2 AM
            "retention_days": 30,
            "compression": True,
            "encryption": True,
            "destinations": [
                {
                    "type": "s3",
                    "bucket": self._get_env("FTNS_BACKUP_S3_BUCKET"),
                    "region": self._get_env("FTNS_BACKUP_S3_REGION", "us-east-1")
                },
                {
                    "type": "local",
                    "path": self._get_env("FTNS_BACKUP_LOCAL_PATH", "/var/backups/ftns")
                }
            ]
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get comprehensive monitoring configuration"""
        return {
            "metrics": {
                "enabled": self.monitoring.metrics_enabled,
                "port": 9090,
                "path": "/metrics",
                "interval": 15  # seconds
            },
            "tracing": {
                "enabled": self.monitoring.tracing_enabled,
                "service_name": "ftns-tokenomics",
                "sample_rate": 0.1  # 10% sampling
            },
            "alerting": {
                "webhook_url": self.monitoring.alert_webhook_url,
                "email": self.monitoring.alert_email,
                "rules": [
                    {
                        "name": "high_error_rate",
                        "condition": "error_rate > 0.05",
                        "severity": "critical"
                    },
                    {
                        "name": "slow_response_time",
                        "condition": "response_time > 2.0",
                        "severity": "warning"
                    },
                    {
                        "name": "high_memory_usage",
                        "condition": "memory_usage > 0.9",
                        "severity": "warning"
                    },
                    {
                        "name": "emergency_triggered",
                        "condition": "emergency_response_active",
                        "severity": "critical"
                    }
                ]
            }
        }


class DeploymentManager:
    """Production deployment management utilities"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def validate_deployment(self) -> Dict[str, Any]:
        """Validate production deployment readiness"""
        
        validation_results = {
            "database": await self._validate_database(),
            "security": await self._validate_security(),
            "configuration": await self._validate_configuration(),
            "dependencies": await self._validate_dependencies(),
            "performance": await self._validate_performance()
        }
        
        all_passed = all(result["status"] == "pass" for result in validation_results.values())
        
        return {
            "overall_status": "ready" if all_passed else "not_ready",
            "checks": validation_results,
            "timestamp": self._get_timestamp()
        }
    
    async def _validate_database(self) -> Dict[str, Any]:
        """Validate database configuration and connectivity"""
        try:
            engine = await self.config.create_database_engine()
            
            # Test connectivity
            async with engine.begin() as conn:
                result = await conn.execute("SELECT 1")
                await result.fetchone()
            
            await engine.dispose()
            
            return {
                "status": "pass",
                "message": "Database connectivity verified",
                "details": {
                    "pool_size": self.config.database.pool_size,
                    "max_overflow": self.config.database.max_overflow
                }
            }
        
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Database validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_security(self) -> Dict[str, Any]:
        """Validate security configuration"""
        issues = []
        
        # Check required secrets
        if not self.config.security.secret_key:
            issues.append("Secret key not configured")
        
        if not self.config.security.encryption_key:
            issues.append("Encryption key not configured")
        
        # Check key strength
        if len(self.config.security.secret_key) < 32:
            issues.append("Secret key too short (minimum 32 characters)")
        
        # Check emergency contacts
        if not self.config.security.emergency_contacts:
            issues.append("No emergency contacts configured")
        
        return {
            "status": "pass" if not issues else "fail",
            "message": "Security validation completed",
            "details": {"issues": issues}
        }
    
    async def _validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration"""
        issues = []
        
        # Check economic parameters
        if self.config.economic.max_daily_adjustment > 0.01:  # 1%
            issues.append("Max daily adjustment too high (>1%)")
        
        if self.config.economic.target_velocity <= 0:
            issues.append("Invalid target velocity")
        
        # Check performance settings
        if self.config.performance.max_concurrent_tasks > 100:
            issues.append("Too many concurrent tasks configured")
        
        return {
            "status": "pass" if not issues else "warning",
            "message": "Configuration validation completed",
            "details": {"issues": issues}
        }
    
    async def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate external dependencies"""
        try:
            # Check required Python packages
            import asyncpg
            import sqlalchemy
            import structlog
            
            return {
                "status": "pass",
                "message": "All dependencies available",
                "details": {
                    "asyncpg": asyncpg.__version__,
                    "sqlalchemy": sqlalchemy.__version__
                }
            }
        
        except ImportError as e:
            return {
                "status": "fail",
                "message": f"Missing dependency: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance configuration"""
        import psutil
        
        # Check system resources
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        issues = []
        
        if memory.available < 2 * 1024 * 1024 * 1024:  # 2GB
            issues.append("Low available memory (<2GB)")
        
        if disk.free < 10 * 1024 * 1024 * 1024:  # 10GB
            issues.append("Low disk space (<10GB)")
        
        return {
            "status": "pass" if not issues else "warning",
            "message": "Performance validation completed",
            "details": {
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "issues": issues
            }
        }
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    
    async def create_deployment_report(self) -> str:
        """Create comprehensive deployment report"""
        validation = await self.validate_deployment()
        
        report = f"""
FTNS Tokenomics System - Deployment Report
Generated: {validation['timestamp']}
Overall Status: {validation['overall_status'].upper()}

=== VALIDATION RESULTS ===

Database:
  Status: {validation['checks']['database']['status']}
  Message: {validation['checks']['database']['message']}

Security:
  Status: {validation['checks']['security']['status']}
  Message: {validation['checks']['security']['message']}

Configuration:
  Status: {validation['checks']['configuration']['status']}
  Message: {validation['checks']['configuration']['message']}

Dependencies:
  Status: {validation['checks']['dependencies']['status']}
  Message: {validation['checks']['dependencies']['message']}

Performance:
  Status: {validation['checks']['performance']['status']}
  Message: {validation['checks']['performance']['message']}

=== CONFIGURATION SUMMARY ===

Environment: {self.config.environment}
Debug Mode: {self.config.debug}

Database Pool Size: {self.config.database.pool_size}
Max Concurrent Users: {self.config.security.max_concurrent_users}
Health Check Interval: {self.config.monitoring.health_check_interval}s

Economic Parameters:
  Initial Appreciation: {self.config.economic.initial_appreciation_rate:.1%}
  Target Appreciation: {self.config.economic.target_appreciation_rate:.1%}
  Grace Period: {self.config.economic.grace_period_days} days
  Emergency Response: {'Enabled' if self.config.economic.emergency_response_enabled else 'Disabled'}

=== RECOMMENDATIONS ===
"""
        
        if validation['overall_status'] == 'ready':
            report += "✅ System is ready for production deployment.\n"
        else:
            report += "⚠️  Issues found that should be addressed before deployment.\n"
            
            for check_name, check_result in validation['checks'].items():
                if check_result['status'] != 'pass':
                    if 'issues' in check_result['details']:
                        for issue in check_result['details']['issues']:
                            report += f"   - {check_name}: {issue}\n"
        
        return report


# Global production configuration instance
_production_config: Optional[ProductionConfig] = None


def get_production_config() -> ProductionConfig:
    """Get singleton production configuration"""
    global _production_config
    
    if _production_config is None:
        _production_config = ProductionConfig()
    
    return _production_config


def initialize_production_environment():
    """Initialize production environment with all necessary configurations"""
    config = get_production_config()
    
    # Setup logging
    config.setup_logging()
    
    # Log startup information
    logger = structlog.get_logger(__name__)
    logger.info(
        "FTNS production environment initialized",
        environment=config.environment,
        database_pool_size=config.database.pool_size,
        monitoring_enabled=config.monitoring.metrics_enabled,
        emergency_response=config.economic.emergency_response_enabled
    )
    
    return config


# Environment validation script for deployment
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Deployment validation script"""
        config = get_production_config()
        deployment_manager = DeploymentManager(config)
        
        print("FTNS Tokenomics System - Deployment Validation")
        print("=" * 50)
        
        # Run validation
        validation = await deployment_manager.validate_deployment()
        
        # Generate and display report
        report = await deployment_manager.create_deployment_report()
        print(report)
        
        # Exit with appropriate code
        exit_code = 0 if validation['overall_status'] == 'ready' else 1
        exit(exit_code)
    
    asyncio.run(main())
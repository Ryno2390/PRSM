"""
Configuration Validators
========================

Validation system for PRSM configuration with business rule checking,
security validation, and cross-component consistency checks.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import ipaddress
from decimal import Decimal, InvalidOperation

from .schemas import PRSMConfig, BaseConfigSchema
from ..errors.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    component_results: Dict[str, 'ValidationResult'] = field(default_factory=dict)
    
    def add_error(self, error: str, component: Optional[str] = None):
        """Add validation error"""
        if component:
            if component not in self.component_results:
                self.component_results[component] = ValidationResult(is_valid=True)
            self.component_results[component].add_error(error)
            self.component_results[component].is_valid = False
        
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str, component: Optional[str] = None):
        """Add validation warning"""
        if component:
            if component not in self.component_results:
                self.component_results[component] = ValidationResult(is_valid=True)
            self.component_results[component].add_warning(warning)
        
        self.warnings.append(warning)


class BaseValidator:
    """Base validator class"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
    
    def validate(self, config: BaseConfigSchema) -> ValidationResult:
        """Validate configuration"""
        result = ValidationResult(is_valid=True)
        
        # Run all validation methods
        for method_name in dir(self):
            if method_name.startswith('validate_') and method_name != 'validate':
                method = getattr(self, method_name)
                if callable(method):
                    try:
                        method(config, result)
                    except Exception as e:
                        result.add_error(f"Validation method {method_name} failed: {e}")
        
        return result


class NWTNConfigValidator(BaseValidator):
    """Validator for NWTN configuration"""
    
    def __init__(self):
        super().__init__("nwtn")
    
    def validate_reasoning_engines(self, config, result: ValidationResult):
        """Validate reasoning engine configuration"""
        
        # Check enabled engines
        if not config.enabled_engines:
            result.add_error("At least one reasoning engine must be enabled")
        
        # Validate engine weights
        for engine in config.enabled_engines:
            if engine not in config.engine_weights:
                result.add_warning(f"No weight specified for engine '{engine}', using default")
            else:
                weight = config.engine_weights[engine]
                if weight <= 0:
                    result.add_error(f"Engine weight for '{engine}' must be positive")
                elif weight > 2.0:
                    result.add_warning(f"Engine weight for '{engine}' is unusually high: {weight}")
    
    def validate_processing_limits(self, config, result: ValidationResult):
        """Validate processing limits"""
        
        # Check concurrent queries vs processing time
        if config.max_concurrent_queries > 50 and config.max_processing_time > 1800:
            result.add_warning(
                "High concurrent queries with long processing time may cause resource exhaustion"
            )
        
        # Validate analogical chain depth
        if config.analogical_chain_max_depth > 8:
            result.add_warning(
                f"Analogical chain depth of {config.analogical_chain_max_depth} may impact performance"
            )
    
    def validate_cache_settings(self, config, result: ValidationResult):
        """Validate cache configuration"""
        
        if config.cache_enabled:
            if config.cache_ttl_seconds < 300:
                result.add_warning("Very short cache TTL may reduce effectiveness")
            elif config.cache_ttl_seconds > 86400:
                result.add_warning("Very long cache TTL may cause stale results")


class TokenomicsConfigValidator(BaseValidator):
    """Validator for tokenomics configuration"""
    
    def __init__(self):
        super().__init__("tokenomics")
    
    def validate_token_economics(self, config, result: ValidationResult):
        """Validate token economics"""
        
        # Check supply limits
        if config.initial_supply <= 0:
            result.add_error("Initial token supply must be positive")
        
        # Validate transaction amounts
        if config.min_transaction_amount >= config.max_transaction_amount:
            result.add_error("Minimum transaction amount must be less than maximum")
        
        # Check pricing model consistency
        total_discount = sum(config.tier_discounts.values())
        if total_discount > len(config.tier_discounts) * 0.5:
            result.add_warning("Total discounts across tiers are very high")
    
    def validate_pricing_multipliers(self, config, result: ValidationResult):
        """Validate pricing multipliers"""
        
        # Check thinking mode multipliers
        thinking_multipliers = config.thinking_mode_multipliers
        if "intermediate" not in thinking_multipliers or thinking_multipliers["intermediate"] != 1.0:
            result.add_warning("Intermediate thinking mode should have multiplier of 1.0 as baseline")
        
        # Check verbosity multipliers
        verbosity_multipliers = config.verbosity_multipliers
        if "standard" not in verbosity_multipliers or verbosity_multipliers["standard"] != 1.0:
            result.add_warning("Standard verbosity should have multiplier of 1.0 as baseline")
    
    def validate_market_dynamics(self, config, result: ValidationResult):
        """Validate market dynamics settings"""
        
        if config.enable_dynamic_pricing:
            if config.surge_multiplier < 1.0:
                result.add_error("Surge multiplier must be at least 1.0")
            
            if config.demand_surge_threshold >= 1.0:
                result.add_error("Demand surge threshold must be less than 1.0")
            
            if config.supply_adjustment_rate > 0.005:
                result.add_warning("High supply adjustment rate may cause price instability")


class MarketplaceConfigValidator(BaseValidator):
    """Validator for marketplace configuration"""
    
    def __init__(self):
        super().__init__("marketplace")
    
    def validate_asset_limits(self, config, result: ValidationResult):
        """Validate asset limits"""
        
        if config.max_assets_per_user > 1000:
            result.add_warning(f"High asset limit per user: {config.max_assets_per_user}")
        
        if config.max_asset_size_mb > 500:
            result.add_warning(f"Large asset size limit: {config.max_asset_size_mb}MB")
    
    def validate_revenue_model(self, config, result: ValidationResult):
        """Validate revenue sharing model"""
        
        total_share = config.creator_revenue_share + config.marketplace_fee_rate
        if total_share > 1.0:
            result.add_error(
                f"Creator revenue share ({config.creator_revenue_share}) + "
                f"marketplace fee ({config.marketplace_fee_rate}) exceeds 100%"
            )
        elif total_share < 0.9:
            result.add_warning("Low total revenue allocation may indicate missing fees")
    
    def validate_search_settings(self, config, result: ValidationResult):
        """Validate search configuration"""
        
        if config.search_results_per_page > 100:
            result.add_warning("High search results per page may impact performance")
        
        if config.search_index_refresh_interval < 60:
            result.add_warning("Frequent search index refresh may impact performance")


class DatabaseConfigValidator(BaseValidator):
    """Validator for database configuration"""
    
    def __init__(self):
        super().__init__("database")
    
    def validate_connection_settings(self, config, result: ValidationResult):
        """Validate database connection settings"""
        
        # Validate connection pool
        if config.pool_size > config.max_overflow + config.pool_size:
            result.add_error("Pool overflow configuration is inconsistent")
        
        if config.pool_size < 5:
            result.add_warning("Small connection pool may limit performance")
        elif config.pool_size > 100:
            result.add_warning("Large connection pool may waste resources")
        
        # Validate timeouts
        if config.query_timeout > config.pool_timeout:
            result.add_error("Query timeout should not exceed pool timeout")
    
    def validate_performance_settings(self, config, result: ValidationResult):
        """Validate performance settings"""
        
        if config.enable_query_cache and config.cache_size < 100:
            result.add_warning("Small query cache size may reduce effectiveness")
        
        if config.pool_recycle < 1800:
            result.add_warning("Short pool recycle time may cause frequent reconnections")
    
    def validate_backup_settings(self, config, result: ValidationResult):
        """Validate backup configuration"""
        
        if config.backup_enabled:
            if config.backup_interval_hours > 72:
                result.add_warning("Long backup interval may risk data loss")
            
            if config.backup_retention_days < 7:
                result.add_warning("Short backup retention may not provide sufficient recovery window")


class SecurityConfigValidator(BaseValidator):
    """Validator for security configuration"""
    
    def __init__(self):
        super().__init__("security")
    
    def validate_authentication(self, config, result: ValidationResult):
        """Validate authentication settings"""
        
        # JWT secret strength
        if len(config.jwt_secret_key) < 32:
            result.add_error("JWT secret key must be at least 32 characters")
        elif len(config.jwt_secret_key) < 64:
            result.add_warning("JWT secret key should be at least 64 characters for better security")
        
        # Token expiry
        if config.jwt_expiry_hours > 168:  # 7 days
            result.add_warning("Long JWT expiry may pose security risk")
        elif config.jwt_expiry_hours < 1:
            result.add_warning("Very short JWT expiry may impact user experience")
    
    def validate_api_security(self, config, result: ValidationResult):
        """Validate API security settings"""
        
        if not config.enable_rate_limiting:
            result.add_warning("Rate limiting is disabled - this may allow abuse")
        elif config.rate_limit_per_minute > 1000:
            result.add_warning("High rate limit may not prevent abuse effectively")
        
        # CORS validation
        if config.enable_cors and "*" in config.allowed_origins:
            result.add_warning("Wildcard CORS origin allows requests from any domain")
    
    def validate_encryption(self, config, result: ValidationResult):
        """Validate encryption settings"""
        
        if not config.enable_field_encryption:
            result.add_warning("Field encryption is disabled - sensitive data may be at risk")
        
        # Check encryption algorithm
        if "AES" not in config.encryption_algorithm:
            result.add_warning(f"Non-AES encryption algorithm: {config.encryption_algorithm}")
    
    def validate_audit_settings(self, config, result: ValidationResult):
        """Validate audit and monitoring settings"""
        
        if not config.enable_audit_logging:
            result.add_error("Audit logging should be enabled for security compliance")
        
        if config.failed_attempt_threshold > 10:
            result.add_warning("High failed attempt threshold may allow brute force attacks")


class APIConfigValidator(BaseValidator):
    """Validator for API configuration"""
    
    def __init__(self):
        super().__init__("api")
    
    def validate_server_settings(self, config, result: ValidationResult):
        """Validate server settings"""
        
        # Port validation
        if config.port < 1024 and config.port != 80 and config.port != 443:
            result.add_warning("Using privileged port - ensure proper permissions")
        
        # Worker configuration
        if config.workers > 32:
            result.add_warning("High number of workers may cause resource contention")
        elif config.workers < 2:
            result.add_warning("Single worker may not provide adequate performance")
    
    def validate_performance_settings(self, config, result: ValidationResult):
        """Validate performance settings"""
        
        if config.max_connections < config.workers * 10:
            result.add_warning("Low connection limit relative to worker count")
        
        if config.request_timeout < 30:
            result.add_warning("Short request timeout may cause premature disconnections")
        elif config.request_timeout > 600:
            result.add_warning("Long request timeout may allow resource exhaustion")


class LoggingConfigValidator(BaseValidator):
    """Validator for logging configuration"""
    
    def __init__(self):
        super().__init__("logging")
    
    def validate_log_settings(self, config, result: ValidationResult):
        """Validate logging settings"""
        
        # File logging validation
        if config.enable_file_logging:
            log_path = Path(config.log_file_path)
            if not log_path.parent.exists():
                try:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    result.add_error(f"Cannot create log directory: {e}")
            
            if config.max_file_size_mb > 1000:
                result.add_warning("Large log file size may impact performance")
        
        # Log level validation
        if config.level.value == "DEBUG" and not config.enable_file_logging:
            result.add_warning("DEBUG logging without file output may flood console")


class SystemConfigValidator(BaseValidator):
    """Validator for system configuration"""
    
    def __init__(self):
        super().__init__("system")
    
    def validate_resource_limits(self, config, result: ValidationResult):
        """Validate resource limits"""
        
        if config.max_memory_usage_mb < 1024:
            result.add_warning("Low memory limit may cause performance issues")
        
        if config.max_cpu_usage_percent > 90:
            result.add_warning("High CPU limit may impact system stability")
        
        if config.max_disk_usage_gb < 10:
            result.add_warning("Low disk space limit may cause operational issues")
    
    def validate_directories(self, config, result: ValidationResult):
        """Validate directory settings"""
        
        directories = [
            config.data_directory,
            config.temp_directory,
            config.log_directory,
            config.cache_directory
        ]
        
        for directory in directories:
            path = Path(directory)
            try:
                path.mkdir(parents=True, exist_ok=True)
                if not path.is_dir():
                    result.add_error(f"Directory path is not a directory: {directory}")
            except Exception as e:
                result.add_error(f"Cannot create directory {directory}: {e}")


class ConfigValidator:
    """Main configuration validator"""
    
    def __init__(self):
        self.component_validators = {
            'nwtn': NWTNConfigValidator(),
            'tokenomics': TokenomicsConfigValidator(),
            'marketplace': MarketplaceConfigValidator(),
            'database': DatabaseConfigValidator(),
            'security': SecurityConfigValidator(),
            'api': APIConfigValidator(),
            'logging': LoggingConfigValidator(),
            'system': SystemConfigValidator()
        }
    
    def validate(self, config: PRSMConfig) -> ValidationResult:
        """Validate complete PRSM configuration"""
        
        result = ValidationResult(is_valid=True)
        
        # Validate each component
        for component_name, validator in self.component_validators.items():
            component_config = getattr(config, component_name, None)
            if component_config:
                component_result = validator.validate(component_config)
                result.component_results[component_name] = component_result
                
                # Add component errors/warnings to main result
                for error in component_result.errors:
                    result.add_error(f"{component_name}: {error}")
                
                for warning in component_result.warnings:
                    result.add_warning(f"{component_name}: {warning}")
        
        # Cross-component validation
        self._validate_cross_component_consistency(config, result)
        
        return result
    
    def _validate_cross_component_consistency(self, config: PRSMConfig, result: ValidationResult):
        """Validate consistency between components"""
        
        # Database connections vs API workers
        if config.database.pool_size < config.api.workers:
            result.add_warning(
                "Database connection pool smaller than API workers - may cause connection bottlenecks"
            )
        
        # Security settings vs API settings
        if config.security.enable_rate_limiting and config.api.max_connections > config.security.rate_limit_per_minute * 10:
            result.add_warning(
                "High connection limit relative to rate limiting may reduce effectiveness"
            )
        
        # Tokenomics pricing vs marketplace fees
        if hasattr(config.tokenomics, 'transaction_fee_rate') and hasattr(config.marketplace, 'marketplace_fee_rate'):
            total_fees = config.tokenomics.transaction_fee_rate + config.marketplace.marketplace_fee_rate
            if total_fees > 0.15:  # 15%
                result.add_warning(f"Combined fees are high: {total_fees*100:.1f}%")
        
        # Logging settings vs system resources
        if config.logging.enable_file_logging and config.logging.max_file_size_mb > config.system.max_disk_usage_gb * 100:
            result.add_warning("Log file size limit exceeds disk usage limit")


# Utility functions
def validate_config(config: PRSMConfig) -> ValidationResult:
    """Validate PRSM configuration"""
    validator = ConfigValidator()
    return validator.validate(config)


def validate_component_config(component_name: str, component_config: BaseConfigSchema) -> ValidationResult:
    """Validate specific component configuration"""
    validator = ConfigValidator()
    
    if component_name not in validator.component_validators:
        result = ValidationResult(is_valid=False)
        result.add_error(f"Unknown component: {component_name}")
        return result
    
    component_validator = validator.component_validators[component_name]
    return component_validator.validate(component_config)
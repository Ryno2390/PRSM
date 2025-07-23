"""
PRSM Centralized Configuration Management
========================================

Unified configuration system for all PRSM components with
environment-based overrides, validation, and hot reloading.
"""

from .manager import ConfigManager, get_config
from .schemas import *
from .loaders import *
from .validators import *

# Legacy compatibility layer
def get_settings():
    """Legacy compatibility function"""
    return get_config()

def get_settings_safe():
    """Safe settings getter that returns None if configuration fails"""
    try:
        return get_config()
    except Exception:
        return None

# Create settings instance for backward compatibility
try:
    settings = get_config()
except Exception:
    settings = None

# Alias for PRSMConfig for backward compatibility
PRSMSettings = PRSMConfig

__all__ = [
    'ConfigManager',
    'get_config',
    'get_settings',
    'get_settings_safe',
    'settings',
    'PRSMSettings',
    'PRSMConfig',
    'NWTNConfig',
    'TokenomicsConfig', 
    'MarketplaceConfig',
    'DatabaseConfig',
    'SecurityConfig',
    'APIConfig',
    'LoggingConfig',
    'validate_config',
    'load_config_file',
    'load_environment_config'
]
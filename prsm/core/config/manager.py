"""
Configuration Manager
=====================

Central configuration management system with environment overrides,
validation, hot reloading, and secure credential handling.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path
from datetime import datetime, timezone
import threading
from functools import lru_cache
import weakref

from .schemas import PRSMConfig, BaseConfigSchema
from .loaders import ConfigLoader, EnvironmentConfigLoader, FileConfigLoader
from .validators import ConfigValidator
from ..errors.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigManager:
    """Central configuration manager for PRSM"""
    
    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._config: Optional[PRSMConfig] = None
        self._config_sources: Dict[str, Any] = {}
        self._last_reload_time: Optional[datetime] = None
        self._file_watchers: Dict[str, float] = {}  # filename -> last_modified
        self._subscribers: weakref.WeakSet = weakref.WeakSet()
        
        # Configuration loaders
        self._loaders: Dict[str, ConfigLoader] = {
            'environment': EnvironmentConfigLoader(),
            'file': FileConfigLoader()
        }
        
        # Configuration validator
        self._validator = ConfigValidator()
        
        self._lock = threading.RLock()
        self._initialized = True
    
    def load_config(
        self,
        config_file: Optional[str] = None,
        environment_prefix: str = "PRSM_",
        validate: bool = True,
        reload_on_change: bool = False
    ) -> PRSMConfig:
        """Load configuration from multiple sources with validation and file watching.
        
        Args:
            config_file: Path to configuration file (YAML, JSON, or TOML). If None,
                only environment variables and defaults will be used.
            environment_prefix: Prefix for environment variables to include in 
                configuration. Variables like PRSM_NWTN_MAX_QUERIES become 
                nwtn.max_queries in the config.
            validate: Whether to validate the loaded configuration against schemas.
                Validation errors will raise ConfigurationError.
            reload_on_change: Whether to set up file watchers to automatically reload
                configuration when the config file changes on disk.
                
        Returns:
            PRSMConfig: Validated configuration object with all component settings.
            
        Raises:
            ConfigurationError: When configuration file cannot be loaded, parsed,
                or validation fails. Includes specific error details and suggestions
                for fixing common issues.
                
        Note:
            Configuration loading follows this precedence order:
            1. Default values from schemas
            2. Configuration file values
            3. Environment variable overrides
            
            Environment variables with the specified prefix are automatically
            converted to nested configuration keys using underscore separation.
            
        Example:
            >>> manager = ConfigManager()
            >>> config = manager.load_config(
            ...     config_file="/path/to/config.yaml",
            ...     environment_prefix="PRSM_",
            ...     validate=True,
            ...     reload_on_change=True
            ... )
            >>> print(config.nwtn.max_concurrent_queries)
            10
        """
        
        with self._lock:
            config_data = {}
            
            # Load from file if provided
            if config_file:
                try:
                    file_data = self._loaders['file'].load(config_file)
                    config_data.update(file_data)
                    self._config_sources['file'] = config_file
                    
                    # Set up file watching if requested
                    if reload_on_change:
                        self._setup_file_watcher(config_file)
                        
                    logger.info(f"Loaded configuration from file: {config_file}")
                    
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to load configuration file: {config_file}",
                        config_key="config_file",
                        config_value=config_file
                    ) from e
            
            # Load from environment variables
            try:
                env_data = self._loaders['environment'].load(prefix=environment_prefix)
                if env_data:
                    config_data.update(env_data)
                    self._config_sources['environment'] = environment_prefix
                    logger.info(f"Loaded environment configuration with prefix: {environment_prefix}")
            
            except Exception as e:
                logger.warning(f"Failed to load environment configuration: {e}")
            
            # Apply default configuration
            if not config_data:
                logger.info("Using default configuration")
            
            # Create configuration object
            try:
                self._config = PRSMConfig(**config_data)
                self._last_reload_time = datetime.now(timezone.utc)
                
                # Validate configuration if requested
                if validate:
                    validation_result = self._validator.validate(self._config)
                    if not validation_result.is_valid:
                        raise ConfigurationError(
                            f"Configuration validation failed: {validation_result.errors}",
                            config_key="validation",
                            config_value=validation_result.errors
                        )
                    
                    if validation_result.warnings:
                        for warning in validation_result.warnings:
                            logger.warning(f"Configuration warning: {warning}")
                
                # Notify subscribers
                self._notify_subscribers('config_loaded')
                
                logger.info("Configuration loaded and validated successfully")
                return self._config
                
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to create configuration object: {str(e)}",
                    config_key="config_creation",
                    config_value=config_data
                ) from e
    
    def get_config(self) -> Optional[PRSMConfig]:
        """Get current configuration"""
        with self._lock:
            return self._config
    
    def get_component_config(self, component: str) -> Optional[BaseConfigSchema]:
        """Get configuration for specific component"""
        if not self._config:
            return None
        
        return getattr(self._config, component, None)
    
    def update_config(
        self,
        updates: Dict[str, Any],
        component: Optional[str] = None,
        validate: bool = True
    ) -> bool:
        """Update configuration dynamically"""
        
        with self._lock:
            if not self._config:
                raise ConfigurationError(
                    "No configuration loaded",
                    config_key="config_state"
                )
            
            try:
                # Create updated configuration data
                current_data = self._config.dict()
                
                if component:
                    # Update specific component
                    if component not in current_data:
                        raise ConfigurationError(
                            f"Unknown component: {component}",
                            config_key="component",
                            config_value=component
                        )
                    
                    current_data[component].update(updates)
                else:
                    # Update root level
                    current_data.update(updates)
                
                # Create new configuration object
                new_config = PRSMConfig(**current_data)
                
                # Validate if requested
                if validate:
                    validation_result = self._validator.validate(new_config)
                    if not validation_result.is_valid:
                        raise ConfigurationError(
                            f"Configuration update validation failed: {validation_result.errors}",
                            config_key="validation_update",
                            config_value=validation_result.errors
                        )
                
                # Apply the update
                self._config = new_config
                self._last_reload_time = datetime.now(timezone.utc)
                
                # Notify subscribers
                self._notify_subscribers('config_updated', component=component, updates=updates)
                
                logger.info(f"Configuration updated for component: {component or 'root'}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update configuration: {e}")
                return False
    
    def reload_config(self) -> bool:
        """Reload configuration from sources"""
        
        with self._lock:
            try:
                # Reload from original sources
                config_file = self._config_sources.get('file')
                env_prefix = self._config_sources.get('environment', 'PRSM_')
                
                self.load_config(
                    config_file=config_file,
                    environment_prefix=env_prefix,
                    validate=True
                )
                
                logger.info("Configuration reloaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")
                return False
    
    def check_file_changes(self) -> bool:
        """Check if configuration files have changed"""
        
        with self._lock:
            changed = False
            
            for file_path, last_modified in self._file_watchers.items():
                try:
                    current_modified = os.path.getmtime(file_path)
                    if current_modified > last_modified:
                        self._file_watchers[file_path] = current_modified
                        changed = True
                        logger.info(f"Configuration file changed: {file_path}")
                
                except OSError:
                    # File might have been deleted
                    logger.warning(f"Configuration file not accessible: {file_path}")
                    continue
            
            return changed
    
    def auto_reload_if_changed(self) -> bool:
        """Automatically reload configuration if files have changed"""
        
        if self.check_file_changes():
            return self.reload_config()
        
        return False
    
    def _setup_file_watcher(self, file_path: str):
        """Set up file watcher for configuration file"""
        try:
            modified_time = os.path.getmtime(file_path)
            self._file_watchers[file_path] = modified_time
            logger.debug(f"Set up file watcher for: {file_path}")
        
        except OSError as e:
            logger.warning(f"Could not set up file watcher for {file_path}: {e}")
    
    def subscribe(self, callback: callable):
        """Subscribe to configuration change notifications"""
        self._subscribers.add(callback)
    
    def _notify_subscribers(self, event_type: str, **kwargs):
        """Notify subscribers of configuration changes"""
        for callback in self._subscribers:
            try:
                callback(event_type, config=self._config, **kwargs)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def export_config(
        self,
        file_path: str,
        format: str = "yaml",
        include_secrets: bool = False
    ) -> bool:
        """Export current configuration to file"""
        
        if not self._config:
            raise ConfigurationError(
                "No configuration to export",
                config_key="config_state"
            )
        
        try:
            config_data = self._config.dict()
            
            # Remove secrets if requested
            if not include_secrets:
                config_data = self._remove_secrets(config_data)
            
            # Export based on format
            with open(file_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_data, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Configuration exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def _remove_secrets(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from configuration data"""
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        
        def clean_dict(data):
            if isinstance(data, dict):
                cleaned = {}
                for key, value in data.items():
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        cleaned[key] = "***REDACTED***"
                    else:
                        cleaned[key] = clean_dict(value)
                return cleaned
            elif isinstance(data, list):
                return [clean_dict(item) for item in data]
            else:
                return data
        
        return clean_dict(config_data)
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about current configuration"""
        
        with self._lock:
            if not self._config:
                return {"status": "not_loaded"}
            
            return {
                "status": "loaded",
                "last_reload": self._last_reload_time.isoformat() if self._last_reload_time else None,
                "sources": list(self._config_sources.keys()),
                "component_count": len([attr for attr in dir(self._config) 
                                      if not attr.startswith('_') and 
                                      hasattr(getattr(self._config, attr), '__dict__')]),
                "file_watchers": len(self._file_watchers),
                "subscribers": len(self._subscribers)
            }
    
    def validate_current_config(self) -> Dict[str, Any]:
        """Validate current configuration and return results"""
        
        if not self._config:
            return {
                "is_valid": False,
                "errors": ["No configuration loaded"],
                "warnings": []
            }
        
        validation_result = self._validator.validate(self._config)
        
        return {
            "is_valid": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings
        }


# Global configuration manager instance
_config_manager = None


@lru_cache(maxsize=1)
def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Optional[PRSMConfig]:
    """Get current configuration"""
    return get_config_manager().get_config()


def get_component_config(component: str) -> Optional[BaseConfigSchema]:
    """Get configuration for specific component"""
    return get_config_manager().get_component_config(component)


def load_config(
    config_file: Optional[str] = None,
    environment_prefix: str = "PRSM_",
    **kwargs
) -> PRSMConfig:
    """Load configuration from sources"""
    return get_config_manager().load_config(
        config_file=config_file,
        environment_prefix=environment_prefix,
        **kwargs
    )


def reload_config() -> bool:
    """Reload configuration from sources"""
    return get_config_manager().reload_config()


def update_config(
    updates: Dict[str, Any],
    component: Optional[str] = None,
    **kwargs
) -> bool:
    """Update configuration dynamically"""
    return get_config_manager().update_config(
        updates=updates,
        component=component,
        **kwargs
    )
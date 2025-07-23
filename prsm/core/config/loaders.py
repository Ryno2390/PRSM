"""
Configuration Loaders
======================

Loaders for different configuration sources including files,
environment variables, and remote configuration services.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders"""
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> Dict[str, Any]:
        """Load configuration from source"""
        pass
    
    def supports_format(self, format: str) -> bool:
        """Check if loader supports specific format"""
        return False


class FileConfigLoader(ConfigLoader):
    """Load configuration from files (YAML, JSON, TOML)"""
    
    def __init__(self):
        self.supported_formats = {'.yaml', '.yml', '.json', '.toml'}
    
    def load(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Load configuration from file"""
        
        if not file_path:
            return {}
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Configuration path is not a file: {file_path}")
        
        # Determine file format
        file_format = path.suffix.lower()
        
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported configuration file format: {file_format}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if file_format in {'.yaml', '.yml'}:
                    return self._load_yaml(f)
                elif file_format == '.json':
                    return self._load_json(f)
                elif file_format == '.toml':
                    return self._load_toml(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {file_path}: {e}") from e
    
    def supports_format(self, format: str) -> bool:
        """Check if format is supported"""
        return format.lower() in self.supported_formats
    
    def _load_yaml(self, file_handle) -> Dict[str, Any]:
        """Load YAML configuration"""
        try:
            return yaml.safe_load(file_handle) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}") from e
    
    def _load_json(self, file_handle) -> Dict[str, Any]:
        """Load JSON configuration"""
        try:
            return json.load(file_handle) or {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e
    
    def _load_toml(self, file_handle) -> Dict[str, Any]:
        """Load TOML configuration"""
        try:
            import tomli
            return tomli.loads(file_handle.read()) or {}
        except ImportError:
            raise RuntimeError("TOML support requires 'tomli' package. Install with: pip install tomli")
        except Exception as e:
            raise ValueError(f"Invalid TOML format: {e}") from e


class EnvironmentConfigLoader(ConfigLoader):
    """Load configuration from environment variables"""
    
    def __init__(self):
        self.type_converters = {
            'bool': self._convert_bool,
            'int': self._convert_int,
            'float': self._convert_float,
            'list': self._convert_list,
            'dict': self._convert_dict
        }
    
    def load(self, prefix: str = "PRSM_", **kwargs) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        
        config = {}
        
        # Get all environment variables with the specified prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Convert nested keys (e.g., PRSM_NWTN_MAX_QUERIES -> nwtn.max_queries)
                nested_keys = config_key.split('_')
                
                # Convert value to appropriate type
                converted_value = self._convert_value(value)
                
                # Set nested configuration
                self._set_nested_config(config, nested_keys, converted_value)
        
        return config
    
    def _set_nested_config(self, config: Dict[str, Any], keys: list, value: Any):
        """Set nested configuration value"""
        current = config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Convert to dict if it's not already
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate Python type"""
        
        # Check for type hints in the value (e.g., "bool:true", "int:42")
        if ':' in value:
            type_hint, actual_value = value.split(':', 1)
            if type_hint in self.type_converters:
                return self.type_converters[type_hint](actual_value)
        
        # Auto-detect type
        return self._auto_convert(value)
    
    def _auto_convert(self, value: str) -> Any:
        """Automatically convert string to appropriate type"""
        
        # Boolean values
        if value.lower() in ('true', 'false', 'yes', 'no', 'on', 'off', '1', '0'):
            return self._convert_bool(value)
        
        # Numbers
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        
        try:
            return float(value)
        except ValueError:
            pass
        
        # Lists (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # JSON objects/arrays
        if value.startswith(('{', '[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Default to string
        return value
    
    def _convert_bool(self, value: str) -> bool:
        """Convert string to boolean"""
        return value.lower() in ('true', 'yes', 'on', '1')
    
    def _convert_int(self, value: str) -> int:
        """Convert string to integer"""
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Cannot convert '{value}' to integer")
    
    def _convert_float(self, value: str) -> float:
        """Convert string to float"""
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot convert '{value}' to float")
    
    def _convert_list(self, value: str) -> list:
        """Convert string to list"""
        if value.startswith('[') and value.endswith(']'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Comma-separated values
        return [item.strip() for item in value.split(',')]
    
    def _convert_dict(self, value: str) -> dict:
        """Convert string to dictionary"""
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f"Cannot convert '{value}' to dictionary")


class RemoteConfigLoader(ConfigLoader):
    """Load configuration from remote sources (HTTP endpoints)"""
    
    def __init__(self, timeout: int = 30, headers: Optional[Dict[str, str]] = None):
        self.timeout = timeout
        self.headers = headers or {}
    
    def load(self, url: str, **kwargs) -> Dict[str, Any]:
        """Load configuration from remote URL"""
        
        if not url:
            return {}
        
        try:
            # Prepare request
            request = urllib.request.Request(url)
            
            # Add headers
            for key, value in self.headers.items():
                request.add_header(key, value)
            
            # Make request
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read().decode('utf-8')
                
                # Determine content type
                content_type = response.headers.get('content-type', '').lower()
                
                if 'application/json' in content_type:
                    return json.loads(content)
                elif 'application/yaml' in content_type or 'text/yaml' in content_type:
                    return yaml.safe_load(content)
                else:
                    # Try to parse as JSON first, then YAML
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        try:
                            return yaml.safe_load(content)
                        except yaml.YAMLError:
                            raise ValueError(f"Cannot parse remote configuration content")
        
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to load remote configuration from {url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading remote configuration: {e}") from e


class DatabaseConfigLoader(ConfigLoader):
    """Load configuration from database"""
    
    def __init__(self, connection_string: str, table_name: str = "configuration"):
        self.connection_string = connection_string
        self.table_name = table_name
    
    def load(self, query_params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Load configuration from database"""
        
        try:
            # This would require a database connection library
            # For now, we'll raise NotImplementedError
            raise NotImplementedError("Database configuration loader not yet implemented")
            
            # Implementation would look something like:
            # with get_database_connection(self.connection_string) as conn:
            #     cursor = conn.cursor()
            #     cursor.execute(f"SELECT key, value FROM {self.table_name}")
            #     config = {}
            #     for key, value in cursor.fetchall():
            #         config[key] = json.loads(value) if value.startswith(('{', '[')) else value
            #     return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from database: {e}") from e


class CompositeConfigLoader(ConfigLoader):
    """Composite loader that combines multiple configuration sources"""
    
    def __init__(self, loaders: Dict[str, ConfigLoader], merge_strategy: str = "deep"):
        self.loaders = loaders
        self.merge_strategy = merge_strategy
    
    def load(self, sources: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Load configuration from multiple sources"""
        
        config = {}
        
        # Load from each source in order
        for source_name, source_config in sources.items():
            if source_name not in self.loaders:
                logger.warning(f"No loader available for source: {source_name}")
                continue
            
            try:
                loader = self.loaders[source_name]
                source_data = loader.load(**source_config)
                
                if source_data:
                    config = self._merge_configs(config, source_data)
                    logger.debug(f"Loaded configuration from source: {source_name}")
            
            except Exception as e:
                logger.error(f"Failed to load from source {source_name}: {e}")
                # Continue with other sources
                continue
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries"""
        
        if self.merge_strategy == "shallow":
            # Shallow merge - overlay overwrites base
            return {**base, **overlay}
        
        elif self.merge_strategy == "deep":
            # Deep merge - recursively merge nested dictionaries
            return self._deep_merge(base, overlay)
        
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")
    
    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


# Utility functions for common loading patterns
def load_config_file(file_path: str) -> Dict[str, Any]:
    """Load configuration from a single file"""
    loader = FileConfigLoader()
    return loader.load(file_path)


def load_environment_config(prefix: str = "PRSM_") -> Dict[str, Any]:
    """Load configuration from environment variables"""
    loader = EnvironmentConfigLoader()
    return loader.load(prefix=prefix)


def load_remote_config(url: str, **kwargs) -> Dict[str, Any]:
    """Load configuration from remote URL"""
    loader = RemoteConfigLoader(**kwargs)
    return loader.load(url)


def create_composite_loader() -> CompositeConfigLoader:
    """Create a composite loader with common loaders"""
    loaders = {
        'file': FileConfigLoader(),
        'environment': EnvironmentConfigLoader(),
        'remote': RemoteConfigLoader()
    }
    return CompositeConfigLoader(loaders)
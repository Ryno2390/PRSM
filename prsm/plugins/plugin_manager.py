#!/usr/bin/env python3
"""
Plugin Manager System
=====================

Extensible plugin architecture for PRSM components.
"""

import logging
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Set, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin"""
    name: str
    version: str
    description: str
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


class Plugin(ABC):
    """Base class for all plugins"""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Cleanup plugin resources"""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities provided by this plugin"""
        return []
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Get hook functions provided by this plugin"""
        return {}


class PluginRegistry:
    """Registry for managing plugins"""
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        self._enabled_plugins: Set[str] = set()
        self._hooks: Dict[str, List[Callable]] = {}
        self._capabilities: Dict[str, List[str]] = {}
        
    def register_plugin_class(self, plugin_class: Type[Plugin]):
        """Register a plugin class"""
        # Create temporary instance to get metadata
        temp_instance = plugin_class()
        metadata = temp_instance.metadata
        
        self._plugin_classes[metadata.name] = plugin_class
        logger.info(f"Registered plugin class: {metadata.name} v{metadata.version}")
    
    def load_plugin(self, plugin_name: str, **init_kwargs) -> bool:
        """Load and initialize a plugin"""
        if plugin_name in self._plugins:
            logger.warning(f"Plugin {plugin_name} already loaded")
            return True
        
        if plugin_name not in self._plugin_classes:
            logger.error(f"Plugin class {plugin_name} not registered")
            return False
        
        try:
            plugin_class = self._plugin_classes[plugin_name]
            plugin_instance = plugin_class()
            
            # Check dependencies
            if not self._check_dependencies(plugin_instance.metadata):
                return False
            
            # Initialize plugin
            if plugin_instance.initialize(**init_kwargs):
                self._plugins[plugin_name] = plugin_instance
                self._enabled_plugins.add(plugin_name)
                
                # Register hooks
                hooks = plugin_instance.get_hooks()
                for hook_name, hook_func in hooks.items():
                    if hook_name not in self._hooks:
                        self._hooks[hook_name] = []
                    self._hooks[hook_name].append(hook_func)
                
                # Register capabilities
                capabilities = plugin_instance.get_capabilities()
                self._capabilities[plugin_name] = capabilities
                
                logger.info(f"Loaded plugin: {plugin_name}")
                return True
            else:
                logger.error(f"Failed to initialize plugin: {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return False
        
        try:
            plugin = self._plugins[plugin_name]
            
            # Cleanup plugin
            plugin.cleanup()
            
            # Remove from registries
            del self._plugins[plugin_name]
            self._enabled_plugins.discard(plugin_name)
            
            # Remove hooks
            hooks = plugin.get_hooks()
            for hook_name in hooks.keys():
                if hook_name in self._hooks:
                    self._hooks[hook_name] = [
                        func for func in self._hooks[hook_name] 
                        if func not in hooks.values()
                    ]
            
            # Remove capabilities
            if plugin_name in self._capabilities:
                del self._capabilities[plugin_name]
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """Check if plugin dependencies are satisfied"""
        # Check required dependencies
        for dep in metadata.dependencies:
            if dep not in self._plugins:
                logger.error(f"Plugin dependency not satisfied: {dep}")
                return False
        
        # Check optional dependencies
        from .optional_deps import has_optional_dependency
        for opt_dep in metadata.optional_dependencies:
            if not has_optional_dependency(opt_dep):
                logger.warning(f"Optional dependency not available: {opt_dep}")
        
        return True
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a loaded plugin instance"""
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin classes"""
        return list(self._plugin_classes.keys())
    
    def list_loaded_plugins(self) -> List[str]:
        """List all loaded plugins"""
        return list(self._plugins.keys())
    
    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """Check if a plugin is loaded"""
        return plugin_name in self._plugins
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all registered hooks for a given hook name"""
        results = []
        
        if hook_name in self._hooks:
            for hook_func in self._hooks[hook_name]:
                try:
                    result = hook_func(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error executing hook {hook_name}: {e}")
        
        return results
    
    def has_capability(self, capability: str) -> bool:
        """Check if any loaded plugin provides a capability"""
        for plugin_caps in self._capabilities.values():
            if capability in plugin_caps:
                return True
        return False
    
    def get_plugins_with_capability(self, capability: str) -> List[str]:
        """Get plugins that provide a specific capability"""
        result = []
        for plugin_name, capabilities in self._capabilities.items():
            if capability in capabilities:
                result.append(plugin_name)
        return result


class PluginManager:
    """Main plugin manager for PRSM"""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self._plugin_directories: List[Path] = []
        self._auto_discovery_enabled = True
        
        # Add default plugin directories
        self._add_default_directories()
    
    def _add_default_directories(self):
        """Add default plugin directories"""
        # Current package plugins directory
        current_dir = Path(__file__).parent
        plugins_dir = current_dir / "builtin"
        if plugins_dir.exists():
            self._plugin_directories.append(plugins_dir)
        
        # User plugins directory
        try:
            import os
            user_plugins = Path.home() / ".prsm" / "plugins"
            if user_plugins.exists():
                self._plugin_directories.append(user_plugins)
        except Exception:
            pass
    
    def add_plugin_directory(self, directory: Path):
        """Add a directory to search for plugins"""
        if directory.exists() and directory.is_dir():
            self._plugin_directories.append(directory)
            logger.info(f"Added plugin directory: {directory}")
        else:
            logger.warning(f"Plugin directory does not exist: {directory}")
    
    def discover_plugins(self):
        """Auto-discover plugins in registered directories"""
        if not self._auto_discovery_enabled:
            return
        
        for plugin_dir in self._plugin_directories:
            self._scan_directory(plugin_dir)
    
    def _scan_directory(self, directory: Path):
        """Scan a directory for plugins"""
        try:
            for plugin_file in directory.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue
                
                self._load_plugin_module(plugin_file)
        except Exception as e:
            logger.error(f"Error scanning plugin directory {directory}: {e}")
    
    def _load_plugin_module(self, plugin_file: Path):
        """Load a plugin module and register any Plugin classes"""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(
                plugin_file.stem, plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find Plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin):
                    self.registry.register_plugin_class(obj)
                    
        except Exception as e:
            logger.error(f"Error loading plugin module {plugin_file}: {e}")
    
    def enable_plugin(self, plugin_name: str, **init_kwargs) -> bool:
        """Enable a plugin"""
        return self.registry.load_plugin(plugin_name, **init_kwargs)
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        return self.registry.unload_plugin(plugin_name)
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a plugin instance"""
        return self.registry.get_plugin(plugin_name)
    
    def list_available_plugins(self) -> List[Dict[str, Any]]:
        """List all available plugins with metadata"""
        result = []
        
        for plugin_name in self.registry.list_plugins():
            plugin_class = self.registry._plugin_classes[plugin_name]
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            
            result.append({
                "name": metadata.name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "loaded": self.registry.is_plugin_loaded(plugin_name),
                "dependencies": metadata.dependencies,
                "optional_dependencies": metadata.optional_dependencies
            })
        
        return result
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute a plugin hook"""
        return self.registry.execute_hook(hook_name, *args, **kwargs)
    
    def has_capability(self, capability: str) -> bool:
        """Check if any plugin provides a capability"""
        return self.registry.has_capability(capability)
    
    def get_plugins_with_capability(self, capability: str) -> List[str]:
        """Get plugins that provide a capability"""
        return self.registry.get_plugins_with_capability(capability)
    
    def shutdown(self):
        """Shutdown all plugins"""
        for plugin_name in list(self.registry.list_loaded_plugins()):
            self.disable_plugin(plugin_name)


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
        _plugin_manager.discover_plugins()
    return _plugin_manager


def initialize_plugins():
    """Initialize the plugin system"""
    return get_plugin_manager()


# Export public interface
__all__ = [
    'Plugin',
    'PluginMetadata', 
    'PluginRegistry',
    'PluginManager',
    'get_plugin_manager',
    'initialize_plugins'
]
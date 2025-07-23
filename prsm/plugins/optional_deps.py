#!/usr/bin/env python3
"""
Optional Dependencies Management
===============================

Safe handling of optional dependencies with graceful fallbacks.
"""

import logging
import importlib
import sys
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class OptionalDependency:
    """Represents an optional dependency with metadata"""
    name: str
    import_name: str
    description: str
    fallback_available: bool = True
    required_version: Optional[str] = None
    install_command: Optional[str] = None
    
    def __post_init__(self):
        if not self.install_command:
            self.install_command = f"pip install {self.name}"


class DependencyRegistry:
    """Registry for managing optional dependencies"""
    
    def __init__(self):
        self._dependencies: Dict[str, OptionalDependency] = {}
        self._available_cache: Dict[str, bool] = {}
        self._modules_cache: Dict[str, Any] = {}
        
        # Register common optional dependencies
        self._register_common_deps()
    
    def _register_common_deps(self):
        """Register commonly used optional dependencies"""
        common_deps = [
            OptionalDependency(
                name="psutil",
                import_name="psutil",
                description="System and process monitoring",
                install_command="pip install psutil"
            ),
            OptionalDependency(
                name="numpy",
                import_name="numpy",
                description="Numerical computing library",
                install_command="pip install numpy"
            ),
            OptionalDependency(
                name="pandas",
                import_name="pandas", 
                description="Data manipulation and analysis",
                install_command="pip install pandas"
            ),
            OptionalDependency(
                name="matplotlib",
                import_name="matplotlib",
                description="Plotting and visualization",
                install_command="pip install matplotlib"
            ),
            OptionalDependency(
                name="plotly",
                import_name="plotly",
                description="Interactive plotting",
                install_command="pip install plotly"
            ),
            OptionalDependency(
                name="dash",
                import_name="dash",
                description="Web analytics applications",
                install_command="pip install dash"
            ),
            OptionalDependency(
                name="streamlit",
                import_name="streamlit",
                description="Machine learning web apps",
                install_command="pip install streamlit"
            ),
            OptionalDependency(
                name="requests",
                import_name="requests",
                description="HTTP library",
                install_command="pip install requests"
            ),
            OptionalDependency(
                name="aiohttp",
                import_name="aiohttp",
                description="Async HTTP client/server",
                install_command="pip install aiohttp"
            ),
            OptionalDependency(
                name="redis",
                import_name="redis",
                description="Redis client",
                install_command="pip install redis"
            ),
            OptionalDependency(
                name="sqlalchemy",
                import_name="sqlalchemy",
                description="SQL toolkit and ORM",
                install_command="pip install sqlalchemy"
            ),
            OptionalDependency(
                name="openai",
                import_name="openai",
                description="OpenAI API client",
                install_command="pip install openai"
            ),
            OptionalDependency(
                name="anthropic",
                import_name="anthropic",
                description="Anthropic API client",
                install_command="pip install anthropic"
            ),
            OptionalDependency(
                name="google-cloud-aiplatform",
                import_name="google.cloud.aiplatform",
                description="Google Cloud AI Platform",
                install_command="pip install google-cloud-aiplatform"
            )
        ]
        
        for dep in common_deps:
            self.register_dependency(dep)
    
    def register_dependency(self, dependency: OptionalDependency):
        """Register an optional dependency"""
        self._dependencies[dependency.name] = dependency
        logger.debug(f"Registered optional dependency: {dependency.name}")
    
    def is_available(self, name: str) -> bool:
        """Check if an optional dependency is available"""
        if name in self._available_cache:
            return self._available_cache[name]
        
        if name not in self._dependencies:
            logger.warning(f"Unknown dependency: {name}")
            return False
        
        dep = self._dependencies[name]
        try:
            importlib.import_module(dep.import_name)
            self._available_cache[name] = True
            return True
        except ImportError:
            self._available_cache[name] = False
            logger.debug(f"Optional dependency {name} not available")
            return False
    
    def get_module(self, name: str) -> Optional[Any]:
        """Get an optional module if available"""
        if name in self._modules_cache:
            return self._modules_cache[name]
        
        if not self.is_available(name):
            return None
        
        dep = self._dependencies[name]
        try:
            module = importlib.import_module(dep.import_name)
            self._modules_cache[name] = module
            return module
        except ImportError:
            logger.error(f"Failed to import {name} despite availability check")
            return None
    
    def get_dependency_info(self, name: str) -> Optional[OptionalDependency]:
        """Get information about a dependency"""
        return self._dependencies.get(name)
    
    def list_available_dependencies(self) -> List[str]:
        """List all available dependencies"""
        return [name for name in self._dependencies.keys() if self.is_available(name)]
    
    def list_missing_dependencies(self) -> List[str]:
        """List all missing dependencies"""
        return [name for name in self._dependencies.keys() if not self.is_available(name)]
    
    def get_install_instructions(self, name: str) -> Optional[str]:
        """Get installation instructions for a dependency"""
        dep = self.get_dependency_info(name)
        return dep.install_command if dep else None
    
    def clear_cache(self):
        """Clear the dependency cache"""
        self._available_cache.clear()
        self._modules_cache.clear()


# Global registry instance
_registry = DependencyRegistry()


def safe_import(module_name: str, fallback=None) -> Any:
    """Safely import a module with optional fallback"""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        logger.debug(f"Could not import {module_name}: {e}")
        return fallback


def require_optional(dependency_name: str, raise_on_missing: bool = False) -> Optional[Any]:
    """Require an optional dependency, with optional error raising"""
    module = _registry.get_module(dependency_name)
    
    if module is None and raise_on_missing:
        dep_info = _registry.get_dependency_info(dependency_name)
        install_cmd = dep_info.install_command if dep_info else f"pip install {dependency_name}"
        raise ImportError(
            f"Required optional dependency '{dependency_name}' is not available. "
            f"Install it with: {install_cmd}"
        )
    
    return module


def optional_feature(dependency_name: str, fallback_message: str = None):
    """Decorator to mark functions that require optional dependencies"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _registry.is_available(dependency_name):
                dep_info = _registry.get_dependency_info(dependency_name)
                message = fallback_message or f"Feature requires {dependency_name}"
                if dep_info:
                    message += f". Install with: {dep_info.install_command}"
                
                logger.warning(message)
                raise ImportError(message)
            
            return func(*args, **kwargs)
        
        # Add metadata to the function
        wrapper.__optional_dependency__ = dependency_name
        wrapper.__fallback_message__ = fallback_message
        
        return wrapper
    return decorator


def has_optional_dependency(dependency_name: str) -> bool:
    """Check if an optional dependency is available"""
    return _registry.is_available(dependency_name)


def list_optional_dependencies() -> Dict[str, Dict[str, Any]]:
    """List all registered optional dependencies with their status"""
    result = {}
    
    for name, dep in _registry._dependencies.items():
        result[name] = {
            "available": _registry.is_available(name),
            "description": dep.description,
            "install_command": dep.install_command,
            "required_version": dep.required_version
        }
    
    return result


def register_optional_dependency(dependency: OptionalDependency):
    """Register a new optional dependency"""
    _registry.register_dependency(dependency)


class OptionalImporter:
    """Context manager for optional imports with fallbacks"""
    
    def __init__(self, module_name: str, fallback_name: str = None):
        self.module_name = module_name
        self.fallback_name = fallback_name
        self.module = None
        self.available = False
    
    def __enter__(self):
        self.module = safe_import(self.module_name)
        self.available = self.module is not None
        
        if not self.available and self.fallback_name:
            self.module = safe_import(self.fallback_name)
            self.available = self.module is not None
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def __bool__(self):
        return self.available


# Convenience functions for common patterns
def try_import_numpy():
    """Try to import numpy with fallback"""
    return require_optional("numpy")


def try_import_pandas():
    """Try to import pandas with fallback"""
    return require_optional("pandas")


def try_import_matplotlib():
    """Try to import matplotlib with fallback"""
    return require_optional("matplotlib")


def try_import_plotly():
    """Try to import plotly with fallback"""
    return require_optional("plotly")


def try_import_psutil():
    """Try to import psutil with fallback"""
    return require_optional("psutil")


# Export public interface
__all__ = [
    'OptionalDependency',
    'DependencyRegistry', 
    'safe_import',
    'require_optional',
    'optional_feature',
    'has_optional_dependency',
    'list_optional_dependencies',
    'register_optional_dependency',
    'OptionalImporter',
    'try_import_numpy',
    'try_import_pandas', 
    'try_import_matplotlib',
    'try_import_plotly',
    'try_import_psutil'
]
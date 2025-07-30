#!/usr/bin/env python3
"""
PRSM Plugin System
==================

Plugin architecture for optional dependencies and extensible functionality.
Provides safe handling of optional packages and modular extension system.
"""

from .plugin_manager import PluginManager, Plugin, PluginRegistry, get_plugin_manager, initialize_plugins
from .optional_deps import OptionalDependency, require_optional, safe_import, has_optional_dependency

__all__ = [
    'PluginManager',
    'Plugin', 
    'PluginRegistry',
    'get_plugin_manager',
    'initialize_plugins',
    'OptionalDependency',
    'require_optional',
    'safe_import',
    'has_optional_dependency'
]
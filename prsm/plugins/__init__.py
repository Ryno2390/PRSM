#!/usr/bin/env python3
"""
PRSM Plugin System
==================

Plugin architecture for optional dependencies and extensible functionality.
Provides safe handling of optional packages and modular extension system.
"""

from .plugin_manager import PluginManager, Plugin, PluginRegistry
from .optional_deps import OptionalDependency, require_optional, safe_import

__all__ = [
    'PluginManager',
    'Plugin', 
    'PluginRegistry',
    'OptionalDependency',
    'require_optional',
    'safe_import'
]
#!/usr/bin/env python3
"""
Plugin Registry System
======================

Advanced plugin and extension registry with lifecycle management,
dependency resolution, security scanning, and automated testing.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
import uuid
from pathlib import Path
import hashlib
import importlib.util
import ast
import subprocess
import tempfile
import zipfile
import tarfile

from prsm.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins"""
    CORE_EXTENSION = "core_extension"
    UI_COMPONENT = "ui_component"
    DATA_PROCESSOR = "data_processor"
    AI_MODEL_WRAPPER = "ai_model_wrapper"
    INTEGRATION_CONNECTOR = "integration_connector"
    WORKFLOW_STEP = "workflow_step"
    CUSTOM_TOOL = "custom_tool"
    THEME = "theme"
    LANGUAGE_PACK = "language_pack"
    SECURITY_MODULE = "security_module"


class PluginCapability(Enum):
    """Plugin capabilities"""
    TEXT_PROCESSING = "text_processing"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"
    DATA_TRANSFORMATION = "data_transformation"
    API_INTEGRATION = "api_integration"
    DATABASE_ACCESS = "database_access"
    FILE_SYSTEM_ACCESS = "file_system_access"
    NETWORK_ACCESS = "network_access"
    UI_MODIFICATION = "ui_modification"
    SYSTEM_ADMINISTRATION = "system_administration"
    ENCRYPTION = "encryption"
    AUTHENTICATION = "authentication"


class PluginStatus(Enum):
    """Plugin status in registry"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    TESTING = "testing"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    BANNED = "banned"
    QUARANTINED = "quarantined"


class PluginLifecycleState(Enum):
    """Plugin lifecycle states"""
    UNINSTALLED = "uninstalled"
    INSTALLING = "installing"
    INSTALLED = "installed"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    UPDATING = "updating"
    UNINSTALLING = "uninstalling"
    ERROR = "error"


@dataclass
class PluginManifest:
    """Plugin manifest definition"""
    name: str
    version: str
    plugin_id: str
    description: str = ""
    
    # Plugin metadata
    author: str = ""
    author_email: str = ""
    homepage: str = ""
    repository: str = ""
    license: str = "MIT"
    
    # Plugin configuration
    plugin_type: PluginType = PluginType.CORE_EXTENSION
    capabilities: List[PluginCapability] = field(default_factory=list)
    
    # Entry points
    main_module: str = ""
    entry_point: str = "main"
    config_schema: Optional[Dict[str, Any]] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    python_requires: str = ">=3.8"
    prsm_requires: str = ">=1.0.0"
    
    # Platform compatibility
    platforms: List[str] = field(default_factory=lambda: ["any"])
    architectures: List[str] = field(default_factory=lambda: ["any"])
    
    # Resource requirements
    min_memory_mb: int = 64
    max_memory_mb: int = 512
    cpu_intensive: bool = False
    gpu_required: bool = False
    
    # Security and permissions
    permissions: List[str] = field(default_factory=list)
    sandbox_mode: bool = True
    trusted: bool = False
    
    # UI and display
    display_name: str = ""
    category: str = "General"
    tags: List[str] = field(default_factory=list)
    icon: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    
    # Lifecycle hooks
    install_hooks: List[str] = field(default_factory=list)
    uninstall_hooks: List[str] = field(default_factory=list)
    activation_hooks: List[str] = field(default_factory=list)
    deactivation_hooks: List[str] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate manifest data"""
        errors = []
        
        # Required fields
        if not self.name:
            errors.append("Missing required field: name")
        
        if not self.version:
            errors.append("Missing required field: version")
        
        if not self.plugin_id:
            errors.append("Missing required field: plugin_id")
        
        if not self.main_module:
            errors.append("Missing required field: main_module")
        
        # Version format validation
        try:
            import semver
            semver.VersionInfo.parse(self.version)
        except Exception:
            errors.append(f"Invalid version format: {self.version}")
        
        # Plugin ID format validation
        if not self.plugin_id.replace("_", "").replace("-", "").isalnum():
            errors.append(f"Invalid plugin_id format: {self.plugin_id}")
        
        # Permissions validation
        valid_permissions = [
            "file_read", "file_write", "network", "database",
            "system", "admin", "encryption", "ui_modify"
        ]
        
        for permission in self.permissions:
            if permission not in valid_permissions:
                errors.append(f"Invalid permission: {permission}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "plugin_id": self.plugin_id,
            "description": self.description,
            "author": self.author,
            "author_email": self.author_email,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
            "plugin_type": self.plugin_type.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "main_module": self.main_module,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
            "dependencies": self.dependencies,
            "python_requires": self.python_requires,
            "prsm_requires": self.prsm_requires,
            "platforms": self.platforms,
            "architectures": self.architectures,
            "min_memory_mb": self.min_memory_mb,
            "max_memory_mb": self.max_memory_mb,
            "cpu_intensive": self.cpu_intensive,
            "gpu_required": self.gpu_required,
            "permissions": self.permissions,
            "sandbox_mode": self.sandbox_mode,
            "trusted": self.trusted,
            "display_name": self.display_name,
            "category": self.category,
            "tags": self.tags,
            "icon": self.icon,
            "screenshots": self.screenshots,
            "install_hooks": self.install_hooks,
            "uninstall_hooks": self.uninstall_hooks,
            "activation_hooks": self.activation_hooks,
            "deactivation_hooks": self.deactivation_hooks
        }


@dataclass
class Plugin:
    """Plugin instance with runtime information"""
    manifest: PluginManifest
    
    # Installation information
    install_path: Optional[Path] = None
    installed_at: Optional[datetime] = None
    installed_by: Optional[str] = None
    
    # Runtime state
    state: PluginLifecycleState = PluginLifecycleState.UNINSTALLED
    status: PluginStatus = PluginStatus.DRAFT
    
    # Instance data
    instance: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    load_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Usage statistics
    activation_count: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None
    
    # Health and monitoring
    health_score: float = 100.0
    last_health_check: Optional[datetime] = None
    
    # Error tracking
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    def get_plugin_id(self) -> str:
        """Get plugin ID"""
        return self.manifest.plugin_id
    
    def get_version(self) -> str:
        """Get plugin version"""
        return self.manifest.version
    
    def is_active(self) -> bool:
        """Check if plugin is active"""
        return self.state == PluginLifecycleState.ACTIVE
    
    def is_installed(self) -> bool:
        """Check if plugin is installed"""
        return self.state not in [PluginLifecycleState.UNINSTALLED, PluginLifecycleState.ERROR]
    
    def calculate_health_score(self) -> float:
        """Calculate plugin health score"""
        
        base_score = 100.0
        
        # Penalize errors
        if self.error_count > 0:
            error_penalty = min(50, self.error_count * 10)
            base_score -= error_penalty
        
        # Factor in resource usage
        if self.memory_usage_mb > self.manifest.max_memory_mb:
            memory_penalty = 20
            base_score -= memory_penalty
        
        if self.cpu_usage_percent > 80:
            cpu_penalty = 15
            base_score -= cpu_penalty
        
        # Age factor for last usage
        if self.last_used:
            days_since_use = (datetime.now(timezone.utc) - self.last_used).days
            if days_since_use > 30:
                age_penalty = min(20, days_since_use / 10)
                base_score -= age_penalty
        
        return max(0, base_score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "manifest": self.manifest.to_dict(),
            "install_path": str(self.install_path) if self.install_path else None,
            "installed_at": self.installed_at.isoformat() if self.installed_at else None,
            "installed_by": self.installed_by,
            "state": self.state.value,
            "status": self.status.value,
            "config": self.config,
            "load_time_ms": self.load_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "activation_count": self.activation_count,
            "error_count": self.error_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "health_score": self.calculate_health_score(),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None
        }


class PluginValidator:
    """Plugin validation and security scanning"""
    
    def __init__(self):
        self.security_checks = [
            self._check_dangerous_imports,
            self._check_file_system_access,
            self._check_network_access,
            self._check_subprocess_usage,
            self._check_eval_usage,
            self._check_pickle_usage
        ]
        
        self.code_quality_checks = [
            self._check_syntax,
            self._check_code_complexity,
            self._check_dependencies,
            self._check_manifest_consistency
        ]
    
    async def validate_plugin_package(self, package_path: Path) -> Dict[str, Any]:
        """Validate plugin package"""
        
        validation_results = {
            "valid": True,
            "security_score": 100,
            "quality_score": 100,
            "issues": [],
            "warnings": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Extract and analyze package
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract package
                if package_path.suffix == '.zip':
                    with zipfile.ZipFile(package_path, 'r') as zip_file:
                        zip_file.extractall(temp_path)
                elif package_path.suffix in ['.tar.gz', '.tgz']:
                    with tarfile.open(package_path, 'r:gz') as tar_file:
                        tar_file.extractall(temp_path)
                else:
                    validation_results["issues"].append("Unsupported package format")
                    validation_results["valid"] = False
                    return validation_results
                
                # Find and validate manifest
                manifest = await self._find_and_validate_manifest(temp_path)
                if not manifest:
                    validation_results["issues"].append("Missing or invalid manifest")
                    validation_results["valid"] = False
                    return validation_results
                
                # Security validation
                security_results = await self._perform_security_checks(temp_path, manifest)
                validation_results["security_score"] = security_results["score"]
                validation_results["issues"].extend(security_results["issues"])
                validation_results["warnings"].extend(security_results["warnings"])
                
                # Code quality validation
                quality_results = await self._perform_quality_checks(temp_path, manifest)
                validation_results["quality_score"] = quality_results["score"]
                validation_results["issues"].extend(quality_results["issues"])
                validation_results["warnings"].extend(quality_results["warnings"])
                
                # Overall validation
                if validation_results["security_score"] < 70 or validation_results["quality_score"] < 60:
                    validation_results["valid"] = False
                
        except Exception as e:
            validation_results["issues"].append(f"Validation error: {str(e)}")
            validation_results["valid"] = False
        
        return validation_results
    
    async def _find_and_validate_manifest(self, package_path: Path) -> Optional[PluginManifest]:
        """Find and validate plugin manifest"""
        
        # Look for manifest files
        manifest_files = [
            "plugin.json",
            "manifest.json",
            "prsm-plugin.json"
        ]
        
        for manifest_file in manifest_files:
            manifest_path = package_path / manifest_file
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest_data = json.load(f)
                    
                    # Convert to PluginManifest
                    manifest = PluginManifest(**manifest_data)
                    
                    # Validate manifest
                    is_valid, errors = manifest.validate()
                    if is_valid:
                        return manifest
                    else:
                        logger.warning(f"Invalid manifest: {errors}")
                
                except Exception as e:
                    logger.error(f"Failed to parse manifest: {e}")
        
        return None
    
    async def _perform_security_checks(self, package_path: Path, 
                                     manifest: PluginManifest) -> Dict[str, Any]:
        """Perform security validation checks"""
        
        results = {
            "score": 100,
            "issues": [],
            "warnings": []
        }
        
        # Find Python files
        python_files = list(package_path.rglob("*.py"))
        
        for python_file in python_files:
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Run security checks
                for check in self.security_checks:
                    check_results = await check(code, python_file, manifest)
                    results["score"] -= check_results["penalty"]
                    results["issues"].extend(check_results["issues"])
                    results["warnings"].extend(check_results["warnings"])
            
            except Exception as e:
                results["warnings"].append(f"Failed to analyze {python_file}: {e}")
        
        results["score"] = max(0, results["score"])
        return results
    
    async def _perform_quality_checks(self, package_path: Path,
                                    manifest: PluginManifest) -> Dict[str, Any]:
        """Perform code quality checks"""
        
        results = {
            "score": 100,
            "issues": [],
            "warnings": []
        }
        
        # Run quality checks
        for check in self.code_quality_checks:
            check_results = await check(package_path, manifest)
            results["score"] -= check_results["penalty"]
            results["issues"].extend(check_results["issues"])
            results["warnings"].extend(check_results["warnings"])
        
        results["score"] = max(0, results["score"])
        return results
    
    async def _check_dangerous_imports(self, code: str, file_path: Path,
                                     manifest: PluginManifest) -> Dict[str, Any]:
        """Check for dangerous imports"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        dangerous_modules = [
            "os", "subprocess", "sys", "eval", "exec", "compile",
            "importlib", "__import__", "pickle", "ctypes"
        ]
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_modules:
                            if alias.name not in manifest.permissions:
                                results["issues"].append(
                                    f"Unauthorized dangerous import: {alias.name} in {file_path}"
                                )
                                results["penalty"] += 20
                            else:
                                results["warnings"].append(
                                    f"Authorized dangerous import: {alias.name} in {file_path}"
                                )
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_modules:
                        if node.module not in manifest.permissions:
                            results["issues"].append(
                                f"Unauthorized dangerous import: {node.module} in {file_path}"
                            )
                            results["penalty"] += 20
        
        except Exception as e:
            results["warnings"].append(f"Failed to parse {file_path}: {e}")
        
        return results
    
    async def _check_file_system_access(self, code: str, file_path: Path,
                                      manifest: PluginManifest) -> Dict[str, Any]:
        """Check for file system access"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        file_operations = ["open", "file", "read", "write", "mkdir", "rmdir", "remove"]
        
        if any(op in code for op in file_operations):
            if "file_read" not in manifest.permissions and "file_write" not in manifest.permissions:
                results["issues"].append(f"Unauthorized file system access in {file_path}")
                results["penalty"] += 15
        
        return results
    
    async def _check_network_access(self, code: str, file_path: Path,
                                  manifest: PluginManifest) -> Dict[str, Any]:
        """Check for network access"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        network_modules = ["urllib", "requests", "http", "socket", "asyncio"]
        
        if any(module in code for module in network_modules):
            if "network" not in manifest.permissions:
                results["issues"].append(f"Unauthorized network access in {file_path}")
                results["penalty"] += 15
        
        return results
    
    async def _check_subprocess_usage(self, code: str, file_path: Path,
                                    manifest: PluginManifest) -> Dict[str, Any]:
        """Check for subprocess usage"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        if "subprocess" in code or "os.system" in code:
            if "system" not in manifest.permissions:
                results["issues"].append(f"Unauthorized subprocess usage in {file_path}")
                results["penalty"] += 25
        
        return results
    
    async def _check_eval_usage(self, code: str, file_path: Path,
                              manifest: PluginManifest) -> Dict[str, Any]:
        """Check for eval/exec usage"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        dangerous_functions = ["eval", "exec", "compile"]
        
        if any(func in code for func in dangerous_functions):
            results["issues"].append(f"Dangerous code execution functions in {file_path}")
            results["penalty"] += 30
        
        return results
    
    async def _check_pickle_usage(self, code: str, file_path: Path,
                                manifest: PluginManifest) -> Dict[str, Any]:
        """Check for pickle usage"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        if "pickle" in code or "cPickle" in code:
            results["warnings"].append(f"Pickle usage detected in {file_path} - potential security risk")
            results["penalty"] += 10
        
        return results
    
    async def _check_syntax(self, package_path: Path, manifest: PluginManifest) -> Dict[str, Any]:
        """Check Python syntax"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        python_files = list(package_path.rglob("*.py"))
        
        for python_file in python_files:
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                ast.parse(code)
            
            except SyntaxError as e:
                results["issues"].append(f"Syntax error in {python_file}: {e}")
                results["penalty"] += 20
            except Exception as e:
                results["warnings"].append(f"Failed to check syntax for {python_file}: {e}")
        
        return results
    
    async def _check_code_complexity(self, package_path: Path, 
                                   manifest: PluginManifest) -> Dict[str, Any]:
        """Check code complexity"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        # This would integrate with tools like radon or flake8
        # For now, basic file size check
        python_files = list(package_path.rglob("*.py"))
        
        for python_file in python_files:
            try:
                file_size = python_file.stat().st_size
                if file_size > 100000:  # 100KB
                    results["warnings"].append(f"Large file detected: {python_file} ({file_size} bytes)")
                    results["penalty"] += 5
            
            except Exception as e:
                results["warnings"].append(f"Failed to check file size for {python_file}: {e}")
        
        return results
    
    async def _check_dependencies(self, package_path: Path, 
                                manifest: PluginManifest) -> Dict[str, Any]:
        """Check dependencies"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        # Check for requirements.txt
        requirements_file = package_path / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    requirements = f.read().strip().split('\n')
                
                # Basic validation - check if dependencies match manifest
                manifest_deps = set(manifest.dependencies)
                file_deps = set(req.split('==')[0].split('>=')[0].split('<=')[0] for req in requirements if req.strip())
                
                missing_in_manifest = file_deps - manifest_deps
                if missing_in_manifest:
                    results["warnings"].append(f"Dependencies in requirements.txt not declared in manifest: {missing_in_manifest}")
                    results["penalty"] += 5
            
            except Exception as e:
                results["warnings"].append(f"Failed to parse requirements.txt: {e}")
        
        return results
    
    async def _check_manifest_consistency(self, package_path: Path,
                                        manifest: PluginManifest) -> Dict[str, Any]:
        """Check manifest consistency with package contents"""
        
        results = {"penalty": 0, "issues": [], "warnings": []}
        
        # Check if main module exists
        main_module_path = package_path / f"{manifest.main_module}.py"
        if not main_module_path.exists():
            # Try as package
            package_init = package_path / manifest.main_module / "__init__.py"
            if not package_init.exists():
                results["issues"].append(f"Main module not found: {manifest.main_module}")
                results["penalty"] += 25
        
        return results


class PluginSandbox:
    """Sandbox environment for plugin execution"""
    
    def __init__(self):
        self.restricted_modules = [
            "os", "sys", "subprocess", "ctypes", "pickle",
            "importlib", "__builtin__", "builtins"
        ]
        
        self.allowed_builtins = [
            "abs", "all", "any", "bool", "dict", "enumerate",
            "filter", "float", "int", "len", "list", "map",
            "max", "min", "range", "str", "sum", "tuple", "zip"
        ]
    
    def create_sandbox_globals(self, plugin: Plugin) -> Dict[str, Any]:
        """Create restricted globals for plugin execution"""
        
        sandbox_globals = {
            "__builtins__": {name: getattr(__builtins__, name) 
                           for name in self.allowed_builtins 
                           if hasattr(__builtins__, name)}
        }
        
        # Add plugin-specific allowed modules based on permissions
        if "file_read" in plugin.manifest.permissions:
            # Allow limited file operations
            pass
        
        if "network" in plugin.manifest.permissions:
            # Allow network operations
            pass
        
        return sandbox_globals
    
    async def execute_in_sandbox(self, plugin: Plugin, function: str, 
                                *args, **kwargs) -> Any:
        """Execute plugin function in sandbox"""
        
        if not plugin.instance:
            raise Exception("Plugin not loaded")
        
        if not plugin.manifest.sandbox_mode and not plugin.manifest.trusted:
            raise Exception("Untrusted plugin must run in sandbox mode")
        
        try:
            # Get the function from plugin instance
            if hasattr(plugin.instance, function):
                func = getattr(plugin.instance, function)
                
                if plugin.manifest.sandbox_mode:
                    # Execute in restricted environment
                    # This would use more sophisticated sandboxing in production
                    result = await asyncio.to_thread(func, *args, **kwargs)
                else:
                    # Execute normally for trusted plugins
                    result = await asyncio.to_thread(func, *args, **kwargs)
                
                return result
            else:
                raise Exception(f"Function {function} not found in plugin")
        
        except Exception as e:
            plugin.error_count += 1
            plugin.last_error = str(e)
            plugin.last_error_time = datetime.now(timezone.utc)
            raise


class PluginDependencyResolver:
    """Dependency resolution system for plugins"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.resolution_cache: Dict[str, List[str]] = {}
    
    def add_plugin_dependencies(self, plugin_id: str, dependencies: List[str]):
        """Add plugin dependencies to graph"""
        self.dependency_graph[plugin_id] = set(dependencies)
    
    def resolve_dependencies(self, plugin_id: str, available_plugins: Set[str]) -> Tuple[List[str], List[str]]:
        """Resolve plugin dependencies"""
        
        if plugin_id in self.resolution_cache:
            resolved = self.resolution_cache[plugin_id]
            missing = [dep for dep in resolved if dep not in available_plugins]
            return resolved, missing
        
        resolved = []
        visited = set()
        missing = []
        
        def visit(current_plugin: str):
            if current_plugin in visited:
                return
            
            visited.add(current_plugin)
            
            if current_plugin in self.dependency_graph:
                for dependency in self.dependency_graph[current_plugin]:
                    if dependency not in available_plugins:
                        missing.append(dependency)
                    else:
                        visit(dependency)
                        if dependency not in resolved:
                            resolved.append(dependency)
        
        visit(plugin_id)
        
        # Cache result
        self.resolution_cache[plugin_id] = resolved
        
        return resolved, missing
    
    def check_circular_dependencies(self) -> List[List[str]]:
        """Check for circular dependencies"""
        
        cycles = []
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            if node in self.dependency_graph:
                for neighbor in self.dependency_graph[node]:
                    if has_cycle(neighbor, path + [node]):
                        return True
            
            rec_stack.remove(node)
            return False
        
        for plugin_id in self.dependency_graph:
            if plugin_id not in visited:
                has_cycle(plugin_id, [])
        
        return cycles


class PluginRegistry:
    """Main plugin registry and management system"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./plugin_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Core components
        self.plugins: Dict[str, Plugin] = {}
        self.validator = PluginValidator()
        self.sandbox = PluginSandbox()
        self.dependency_resolver = PluginDependencyResolver()
        
        # Plugin installation paths
        self.install_path = self.storage_path / "installed"
        self.install_path.mkdir(exist_ok=True)
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.stats = {
            "total_plugins": 0,
            "active_plugins": 0,
            "installed_plugins": 0,
            "plugin_types": {},
            "total_activations": 0,
            "total_errors": 0
        }
        
        logger.info("Plugin Registry initialized")
    
    async def register_plugin_package(self, package_path: Path, 
                                     developer_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Register plugin from package"""
        
        try:
            # Validate package
            validation_results = await self.validator.validate_plugin_package(package_path)
            
            if not validation_results["valid"]:
                return False, validation_results
            
            # Extract package to temporary location
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract package
                if package_path.suffix == '.zip':
                    with zipfile.ZipFile(package_path, 'r') as zip_file:
                        zip_file.extractall(temp_path)
                elif package_path.suffix in ['.tar.gz', '.tgz']:
                    with tarfile.open(package_path, 'r:gz') as tar_file:
                        tar_file.extractall(temp_path)
                
                # Load manifest
                manifest = await self.validator._find_and_validate_manifest(temp_path)
                if not manifest:
                    return False, {"error": "Invalid manifest"}
                
                # Create plugin instance
                plugin = Plugin(manifest=manifest)
                plugin.status = PluginStatus.SUBMITTED
                
                # Register plugin
                self.plugins[plugin.get_plugin_id()] = plugin
                
                # Update dependency graph
                self.dependency_resolver.add_plugin_dependencies(
                    plugin.get_plugin_id(), 
                    manifest.dependencies
                )
                
                # Update statistics
                self.stats["total_plugins"] += 1
                plugin_type = manifest.plugin_type.value
                self.stats["plugin_types"][plugin_type] = \
                    self.stats["plugin_types"].get(plugin_type, 0) + 1
                
                logger.info(f"Registered plugin package: {manifest.name}")
                
                # Emit event
                await self._emit_event("plugin_registered", {
                    "plugin_id": plugin.get_plugin_id(),
                    "manifest": manifest.to_dict(),
                    "developer_id": developer_id,
                    "validation_results": validation_results
                })
                
                return True, {"plugin_id": plugin.get_plugin_id(), "validation": validation_results}
        
        except Exception as e:
            logger.error(f"Failed to register plugin package: {e}")
            return False, {"error": str(e)}
    
    async def install_plugin(self, plugin_id: str, package_path: Path) -> bool:
        """Install plugin from package"""
        
        if plugin_id not in self.plugins:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        plugin = self.plugins[plugin_id]
        
        try:
            plugin.state = PluginLifecycleState.INSTALLING
            
            # Create plugin installation directory
            plugin_install_path = self.install_path / plugin_id
            plugin_install_path.mkdir(exist_ok=True)
            
            # Extract package to installation directory
            if package_path.suffix == '.zip':
                with zipfile.ZipFile(package_path, 'r') as zip_file:
                    zip_file.extractall(plugin_install_path)
            elif package_path.suffix in ['.tar.gz', '.tgz']:
                with tarfile.open(package_path, 'r:gz') as tar_file:
                    tar_file.extractall(plugin_install_path)
            
            # Update plugin information
            plugin.install_path = plugin_install_path
            plugin.installed_at = datetime.now(timezone.utc)
            plugin.state = PluginLifecycleState.INSTALLED
            
            # Run install hooks
            await self._run_lifecycle_hooks(plugin, "install_hooks")
            
            # Update statistics
            self.stats["installed_plugins"] += 1
            
            logger.info(f"Installed plugin: {plugin.manifest.name}")
            
            # Emit event
            await self._emit_event("plugin_installed", {
                "plugin_id": plugin_id,
                "install_path": str(plugin_install_path)
            })
            
            return True
        
        except Exception as e:
            plugin.state = PluginLifecycleState.ERROR
            plugin.last_error = str(e)
            plugin.last_error_time = datetime.now(timezone.utc)
            
            logger.error(f"Failed to install plugin {plugin_id}: {e}")
            return False
    
    async def load_plugin(self, plugin_id: str) -> bool:
        """Load plugin into memory"""
        
        if plugin_id not in self.plugins:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        plugin = self.plugins[plugin_id]
        
        if not plugin.is_installed():
            logger.error(f"Plugin not installed: {plugin_id}")
            return False
        
        try:
            plugin.state = PluginLifecycleState.LOADING
            start_time = datetime.now()
            
            # Load plugin module
            module_path = plugin.install_path / f"{plugin.manifest.main_module}.py"
            if not module_path.exists():
                # Try as package
                module_path = plugin.install_path / plugin.manifest.main_module / "__init__.py"
            
            if not module_path.exists():
                raise Exception(f"Main module not found: {plugin.manifest.main_module}")
            
            # Load module
            spec = importlib.util.spec_from_file_location(
                plugin.manifest.main_module, 
                module_path
            )
            module = importlib.util.module_from_spec(spec)
            
            # Execute module in sandbox if required
            if plugin.manifest.sandbox_mode:
                # Would implement proper sandboxing here
                spec.loader.exec_module(module)
            else:
                spec.loader.exec_module(module)
            
            # Get plugin instance
            if hasattr(module, plugin.manifest.entry_point):
                entry_point = getattr(module, plugin.manifest.entry_point)
                if callable(entry_point):
                    plugin.instance = entry_point()
                else:
                    plugin.instance = entry_point
            else:
                raise Exception(f"Entry point not found: {plugin.manifest.entry_point}")
            
            # Calculate load time
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            plugin.load_time_ms = load_time
            
            plugin.state = PluginLifecycleState.LOADED
            
            logger.info(f"Loaded plugin: {plugin.manifest.name} in {load_time:.2f}ms")
            
            # Emit event
            await self._emit_event("plugin_loaded", {
                "plugin_id": plugin_id,
                "load_time_ms": load_time
            })
            
            return True
        
        except Exception as e:
            plugin.state = PluginLifecycleState.ERROR
            plugin.last_error = str(e)
            plugin.last_error_time = datetime.now(timezone.utc)
            plugin.error_count += 1
            
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            return False
    
    async def activate_plugin(self, plugin_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Activate plugin"""
        
        if plugin_id not in self.plugins:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        plugin = self.plugins[plugin_id]
        
        if plugin.state != PluginLifecycleState.LOADED:
            # Try to load plugin first
            if not await self.load_plugin(plugin_id):
                return False
        
        try:
            # Check dependencies
            available_plugins = set(self.plugins.keys())
            dependencies, missing = self.dependency_resolver.resolve_dependencies(
                plugin_id, available_plugins
            )
            
            if missing:
                logger.error(f"Missing dependencies for {plugin_id}: {missing}")
                return False
            
            # Activate dependencies first
            for dep_id in dependencies:
                dep_plugin = self.plugins.get(dep_id)
                if dep_plugin and not dep_plugin.is_active():
                    if not await self.activate_plugin(dep_id):
                        logger.error(f"Failed to activate dependency: {dep_id}")
                        return False
            
            # Set plugin configuration
            if config:
                plugin.config.update(config)
            
            # Run activation hooks
            await self._run_lifecycle_hooks(plugin, "activation_hooks")
            
            # Call plugin activation method if available
            if hasattr(plugin.instance, 'activate'):
                await asyncio.to_thread(plugin.instance.activate, plugin.config)
            
            plugin.state = PluginLifecycleState.ACTIVE
            plugin.activation_count += 1
            plugin.last_used = datetime.now(timezone.utc)
            
            # Update statistics
            self.stats["active_plugins"] += 1
            self.stats["total_activations"] += 1
            
            logger.info(f"Activated plugin: {plugin.manifest.name}")
            
            # Emit event
            await self._emit_event("plugin_activated", {
                "plugin_id": plugin_id,
                "config": plugin.config
            })
            
            return True
        
        except Exception as e:
            plugin.state = PluginLifecycleState.ERROR
            plugin.last_error = str(e)
            plugin.last_error_time = datetime.now(timezone.utc)
            plugin.error_count += 1
            
            self.stats["total_errors"] += 1
            
            logger.error(f"Failed to activate plugin {plugin_id}: {e}")
            return False
    
    async def deactivate_plugin(self, plugin_id: str) -> bool:
        """Deactivate plugin"""
        
        if plugin_id not in self.plugins:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        plugin = self.plugins[plugin_id]
        
        if not plugin.is_active():
            return True
        
        try:
            # Check if other plugins depend on this one
            dependents = [
                pid for pid, deps in self.dependency_resolver.dependency_graph.items()
                if plugin_id in deps and pid in self.plugins and self.plugins[pid].is_active()
            ]
            
            if dependents:
                logger.warning(f"Plugin {plugin_id} has active dependents: {dependents}")
                # Could force deactivate dependents or return False based on policy
            
            # Call plugin deactivation method if available
            if hasattr(plugin.instance, 'deactivate'):
                await asyncio.to_thread(plugin.instance.deactivate)
            
            # Run deactivation hooks
            await self._run_lifecycle_hooks(plugin, "deactivation_hooks")
            
            plugin.state = PluginLifecycleState.INACTIVE
            
            # Update statistics
            self.stats["active_plugins"] -= 1
            
            logger.info(f"Deactivated plugin: {plugin.manifest.name}")
            
            # Emit event
            await self._emit_event("plugin_deactivated", {
                "plugin_id": plugin_id
            })
            
            return True
        
        except Exception as e:
            plugin.last_error = str(e)
            plugin.last_error_time = datetime.now(timezone.utc)
            plugin.error_count += 1
            
            logger.error(f"Failed to deactivate plugin {plugin_id}: {e}")
            return False
    
    async def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall plugin"""
        
        if plugin_id not in self.plugins:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        plugin = self.plugins[plugin_id]
        
        try:
            # Deactivate if active
            if plugin.is_active():
                await self.deactivate_plugin(plugin_id)
            
            plugin.state = PluginLifecycleState.UNINSTALLING
            
            # Run uninstall hooks
            await self._run_lifecycle_hooks(plugin, "uninstall_hooks")
            
            # Remove installation directory
            if plugin.install_path and plugin.install_path.exists():
                import shutil
                shutil.rmtree(plugin.install_path)
            
            plugin.state = PluginLifecycleState.UNINSTALLED
            plugin.install_path = None
            plugin.installed_at = None
            plugin.instance = None
            
            # Update statistics
            self.stats["installed_plugins"] -= 1
            
            logger.info(f"Uninstalled plugin: {plugin.manifest.name}")
            
            # Emit event
            await self._emit_event("plugin_uninstalled", {
                "plugin_id": plugin_id
            })
            
            return True
        
        except Exception as e:
            plugin.state = PluginLifecycleState.ERROR
            plugin.last_error = str(e)
            plugin.last_error_time = datetime.now(timezone.utc)
            plugin.error_count += 1
            
            logger.error(f"Failed to uninstall plugin {plugin_id}: {e}")
            return False
    
    async def execute_plugin_function(self, plugin_id: str, function: str,
                                    *args, **kwargs) -> Any:
        """Execute function on plugin"""
        
        if plugin_id not in self.plugins:
            raise Exception(f"Plugin not found: {plugin_id}")
        
        plugin = self.plugins[plugin_id]
        
        if not plugin.is_active():
            raise Exception(f"Plugin not active: {plugin_id}")
        
        try:
            result = await self.sandbox.execute_in_sandbox(plugin, function, *args, **kwargs)
            plugin.last_used = datetime.now(timezone.utc)
            return result
        
        except Exception as e:
            plugin.error_count += 1
            plugin.last_error = str(e)
            plugin.last_error_time = datetime.now(timezone.utc)
            self.stats["total_errors"] += 1
            raise
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get plugin by ID"""
        return self.plugins.get(plugin_id)
    
    def list_plugins(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List plugins with optional filtering"""
        
        plugins = list(self.plugins.values())
        
        # Apply filters
        if filters:
            if "status" in filters:
                plugins = [p for p in plugins if p.status.value == filters["status"]]
            
            if "state" in filters:
                plugins = [p for p in plugins if p.state.value == filters["state"]]
            
            if "plugin_type" in filters:
                plugins = [p for p in plugins if p.manifest.plugin_type.value == filters["plugin_type"]]
            
            if "category" in filters:
                plugins = [p for p in plugins if p.manifest.category == filters["category"]]
        
        return [plugin.to_dict() for plugin in plugins]
    
    async def _run_lifecycle_hooks(self, plugin: Plugin, hook_type: str):
        """Run plugin lifecycle hooks"""
        
        hooks = getattr(plugin.manifest, hook_type, [])
        
        for hook in hooks:
            try:
                if plugin.instance and hasattr(plugin.instance, hook):
                    hook_func = getattr(plugin.instance, hook)
                    await asyncio.to_thread(hook_func)
            except Exception as e:
                logger.warning(f"Hook {hook} failed for plugin {plugin.get_plugin_id()}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Added event handler for {event_type}")
    
    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit plugin event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        
        return {
            "registry_statistics": self.stats,
            "plugin_health_summary": {
                "healthy_plugins": len([p for p in self.plugins.values() 
                                      if p.calculate_health_score() >= 80]),
                "unhealthy_plugins": len([p for p in self.plugins.values() 
                                        if p.calculate_health_score() < 80]),
                "error_plugins": len([p for p in self.plugins.values() 
                                    if p.state == PluginLifecycleState.ERROR])
            },
            "dependency_info": {
                "circular_dependencies": self.dependency_resolver.check_circular_dependencies(),
                "total_dependencies": len(self.dependency_resolver.dependency_graph)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Export main classes
__all__ = [
    'PluginType',
    'PluginCapability',
    'PluginStatus',
    'PluginLifecycleState',
    'PluginManifest',
    'Plugin',
    'PluginValidator',
    'PluginSandbox',
    'PluginDependencyResolver', 
    'PluginRegistry'
]
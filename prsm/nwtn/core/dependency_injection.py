"""
NWTN Dependency Injection System
================================

Dependency injection container and factory system to resolve circular dependencies
and provide loose coupling between NWTN components.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable, Union
from functools import wraps
from dataclasses import dataclass, field

from .interfaces import (
    ReasoningEngineInterface, VoiceboxInterface, MetaReasoningInterface,
    OrchestratorInterface, ReasoningEngineFactory, ComponentFactory,
    DependencyContainer
)
from .types import ReasoningType

logger = logging.getLogger(__name__)
T = TypeVar('T')


@dataclass
class ComponentRegistration:
    """Registration information for a component"""
    interface_type: Type
    implementation: Union[Type, Callable, Any]
    singleton: bool = True
    factory: bool = False
    dependencies: Dict[str, Type] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class NWTNDependencyContainer(DependencyContainer):
    """Advanced dependency injection container for NWTN components"""
    
    def __init__(self):
        self._registrations: Dict[Type, ComponentRegistration] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initializing: set = set()  # Prevent circular initialization
        self._initialized = False
    
    def register(
        self,
        interface_type: Type[T],
        implementation: Union[Type[T], Callable[[], T], T],
        singleton: bool = True,
        dependencies: Optional[Dict[str, Type]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> 'NWTNDependencyContainer':
        """Register a component with the container"""
        registration = ComponentRegistration(
            interface_type=interface_type,
            implementation=implementation,
            singleton=singleton,
            dependencies=dependencies or {},
            config=config or {}
        )
        
        self._registrations[interface_type] = registration
        logger.debug(f"Registered {interface_type.__name__} -> {implementation}")
        
        return self  # Allow chaining
    
    def register_factory(
        self,
        factory_name: str,
        factory_function: Callable,
        dependencies: Optional[Dict[str, Type]] = None
    ) -> 'NWTNDependencyContainer':
        """Register a factory function"""
        self._factories[factory_name] = factory_function
        
        if dependencies:
            # Register factory dependencies
            for dep_name, dep_type in dependencies.items():
                if dep_type not in self._registrations:
                    logger.warning(f"Factory {factory_name} depends on unregistered type {dep_type}")
        
        return self
    
    def resolve(self, interface_type: Type[T]) -> T:
        """Resolve a component instance"""
        if interface_type not in self._registrations:
            raise ValueError(f"No registration found for {interface_type.__name__}")
        
        registration = self._registrations[interface_type]
        
        # Return singleton if already created
        if registration.singleton and interface_type in self._singletons:
            return self._singletons[interface_type]
        
        # Prevent circular initialization
        if interface_type in self._initializing:
            raise ValueError(f"Circular dependency detected for {interface_type.__name__}")
        
        self._initializing.add(interface_type)
        
        try:
            # Create instance
            instance = self._create_instance(registration)
            
            # Store singleton
            if registration.singleton:
                self._singletons[interface_type] = instance
            
            return instance
            
        finally:
            self._initializing.discard(interface_type)
    
    def resolve_factory(self, factory_name: str, **kwargs) -> Any:
        """Resolve using a registered factory"""
        if factory_name not in self._factories:
            raise ValueError(f"No factory registered with name {factory_name}")
        
        factory = self._factories[factory_name]
        return factory(self, **kwargs)
    
    def _create_instance(self, registration: ComponentRegistration) -> Any:
        """Create instance with dependency injection"""
        implementation = registration.implementation
        
        # If it's already an instance, return it
        if not callable(implementation) or hasattr(implementation, '__dict__'):
            return implementation
        
        # Resolve dependencies
        resolved_deps = {}
        for dep_name, dep_type in registration.dependencies.items():
            resolved_deps[dep_name] = self.resolve(dep_type)
        
        # Add configuration
        resolved_deps.update(registration.config)
        
        # Create instance
        try:
            if resolved_deps:
                return implementation(**resolved_deps)
            else:
                return implementation()
        except Exception as e:
            logger.error(f"Failed to create instance of {implementation}: {e}")
            raise
    
    async def initialize_async_components(self) -> None:
        """Initialize components that require async setup"""
        if self._initialized:
            return
        
        initialization_tasks = []
        
        for interface_type, instance in self._singletons.items():
            if hasattr(instance, 'initialize') and asyncio.iscoroutinefunction(instance.initialize):
                initialization_tasks.append(instance.initialize())
        
        if initialization_tasks:
            await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        self._initialized = True
        logger.info("Async component initialization completed")
    
    def configure_dependencies(self, config: Dict[str, Any]) -> None:
        """Configure container with external configuration"""
        for key, value in config.items():
            # Update registration configs
            for registration in self._registrations.values():
                if key in registration.config:
                    registration.config[key] = value


class NWTNComponentFactory(ComponentFactory):
    """Factory for creating NWTN components with dependency injection"""
    
    def __init__(self, container: NWTNDependencyContainer):
        self.container = container
    
    def create_voicebox(self, **kwargs) -> VoiceboxInterface:
        """Create voicebox component"""
        return self.container.resolve_factory('voicebox_factory', **kwargs)
    
    def create_meta_reasoning_engine(self, **kwargs) -> MetaReasoningInterface:
        """Create meta-reasoning engine"""
        return self.container.resolve_factory('meta_reasoning_factory', **kwargs)
    
    def create_orchestrator(self, **kwargs) -> OrchestratorInterface:
        """Create system orchestrator"""
        return self.container.resolve_factory('orchestrator_factory', **kwargs)


class NWTNReasoningEngineFactory(ReasoningEngineFactory):
    """Factory for creating reasoning engines"""
    
    def __init__(self, container: NWTNDependencyContainer):
        self.container = container
        self._engine_mappings = {
            ReasoningType.DEDUCTIVE: 'deductive_engine',
            ReasoningType.INDUCTIVE: 'inductive_engine', 
            ReasoningType.ABDUCTIVE: 'abductive_engine',
            ReasoningType.ANALOGICAL: 'analogical_engine',
            ReasoningType.CAUSAL: 'causal_engine',
            ReasoningType.PROBABILISTIC: 'probabilistic_engine',
            ReasoningType.COUNTERFACTUAL: 'counterfactual_engine'
        }
    
    def create_engine(self, engine_type: ReasoningType) -> ReasoningEngineInterface:
        """Create reasoning engine of specified type"""
        if engine_type not in self._engine_mappings:
            raise ValueError(f"Unknown engine type: {engine_type}")
        
        factory_name = f"{self._engine_mappings[engine_type]}_factory"
        return self.container.resolve_factory(factory_name)
    
    def get_available_engines(self) -> list[ReasoningType]:
        """Get list of available reasoning engine types"""
        return list(self._engine_mappings.keys())


# Decorators for lazy dependency injection
def inject_dependency(dependency_type: Type[T]) -> Callable:
    """Decorator for lazy dependency injection"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get container from args (assuming self has container)
            container = getattr(args[0], 'container', None) or getattr(args[0], 'dependencies', None)
            if not container:
                raise ValueError("No dependency container found")
            
            dependency = container.resolve(dependency_type)
            return await func(*args, dependency=dependency, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            container = getattr(args[0], 'container', None) or getattr(args[0], 'dependencies', None)
            if not container:
                raise ValueError("No dependency container found")
            
            dependency = container.resolve(dependency_type)
            return func(*args, dependency=dependency, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def lazy_import(module_path: str, attribute: str) -> Callable:
    """Lazy import to avoid circular dependencies"""
    def get_attribute():
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attribute)
    
    return get_attribute


# Default container configuration
def create_default_container() -> NWTNDependencyContainer:
    """Create a default configured dependency container"""
    container = NWTNDependencyContainer()
    
    # Register common configurations
    # Note: Actual implementations would be registered when modules are loaded
    
    # Register factory functions (these would be implemented by actual modules)
    container.register_factory(
        'voicebox_factory',
        lambda container, **kwargs: container.resolve(VoiceboxInterface)
    )
    
    container.register_factory(
        'meta_reasoning_factory', 
        lambda container, **kwargs: container.resolve(MetaReasoningInterface)
    )
    
    return container


# Global container instance (can be overridden for testing)
_default_container: Optional[NWTNDependencyContainer] = None


def get_default_container() -> NWTNDependencyContainer:
    """Get the default dependency container"""
    global _default_container
    if _default_container is None:
        _default_container = create_default_container()
    return _default_container


def set_default_container(container: NWTNDependencyContainer) -> None:
    """Set the default dependency container (useful for testing)"""
    global _default_container
    _default_container = container
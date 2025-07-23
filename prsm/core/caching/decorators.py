"""
Cache Decorators
================

Decorators for easy caching of function results with different strategies.
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from typing import Any, Callable, Optional, Union, Dict, TypeVar
from datetime import datetime, timezone

from .cache_manager import get_cache_manager

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

def cache_result(
    cache_name: str = "default",
    ttl_seconds: Optional[int] = None,
    key_prefix: Optional[str] = None,
    include_args: bool = True,
    include_kwargs: bool = True,
    exclude_args: Optional[list] = None,
    condition: Optional[Callable[..., bool]] = None
) -> Callable[[F], F]:
    """
    Decorator to cache function results.
    
    Args:
        cache_name: Name of cache to use
        ttl_seconds: Time to live in seconds
        key_prefix: Prefix for cache keys
        include_args: Whether to include positional args in cache key
        include_kwargs: Whether to include keyword args in cache key
        exclude_args: List of argument indices to exclude from cache key
        condition: Function to determine if result should be cached
        
    Returns:
        Decorated function with caching
        
    Example:
        @cache_result(cache_name="reasoning", ttl_seconds=3600)
        def expensive_reasoning_function(query: str) -> str:
            # Expensive computation
            return result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check condition if provided
            if condition and not condition(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = _generate_cache_key(
                func, args, kwargs, key_prefix, 
                include_args, include_kwargs, exclude_args
            )
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            cached_result = asyncio.run(cache_manager.get(cache_name, cache_key))
            
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key[:32]}...")
                return cached_result
            
            # Execute function
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Cache result if worthwhile
            if _should_cache_result(result, execution_time):
                asyncio.run(cache_manager.set(cache_name, cache_key, result, ttl_seconds))
                logger.debug(f"Cached result for {func.__name__}: {cache_key[:32]}... (exec_time: {execution_time:.3f}s)")
            
            return result
        
        return wrapper
    return decorator


def async_cache_result(
    cache_name: str = "default",
    ttl_seconds: Optional[int] = None,
    key_prefix: Optional[str] = None,
    include_args: bool = True,
    include_kwargs: bool = True,
    exclude_args: Optional[list] = None,
    condition: Optional[Callable[..., bool]] = None
) -> Callable[[F], F]:
    """
    Decorator to cache async function results.
    
    Args:
        cache_name: Name of cache to use
        ttl_seconds: Time to live in seconds
        key_prefix: Prefix for cache keys
        include_args: Whether to include positional args in cache key
        include_kwargs: Whether to include keyword args in cache key
        exclude_args: List of argument indices to exclude from cache key
        condition: Function to determine if result should be cached
        
    Returns:
        Decorated async function with caching
        
    Example:
        @async_cache_result(cache_name="api_response", ttl_seconds=86400)
        async def expensive_api_call(query: str) -> Dict[str, Any]:
            # Expensive API call
            return result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Check condition if provided
            if condition and not condition(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = _generate_cache_key(
                func, args, kwargs, key_prefix,
                include_args, include_kwargs, exclude_args
            )
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            cached_result = await cache_manager.get(cache_name, cache_key)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key[:32]}...")
                return cached_result
            
            # Execute function
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Cache result if worthwhile
            if _should_cache_result(result, execution_time):
                await cache_manager.set(cache_name, cache_key, result, ttl_seconds)
                logger.debug(f"Cached result for {func.__name__}: {cache_key[:32]}... (exec_time: {execution_time:.3f}s)")
            
            return result
        
        return wrapper
    return decorator


def reasoning_cache(
    ttl_seconds: int = 3600,
    include_context: bool = True,
    include_mode: bool = True
) -> Callable[[F], F]:
    """
    Specialized cache decorator for NWTN reasoning operations.
    
    Args:
        ttl_seconds: Cache TTL (default 1 hour)
        include_context: Whether to include reasoning context in cache key
        include_mode: Whether to include thinking mode in cache key
        
    Returns:
        Decorated function with reasoning-specific caching
        
    Example:
        @reasoning_cache(ttl_seconds=7200, include_context=True)
        async def complex_reasoning(query: str, context: Dict, mode: str) -> Dict:
            # Complex reasoning logic
            return reasoning_result
    """
    def condition(*args, **kwargs):
        # Don't cache empty queries or error states
        query = args[0] if args else kwargs.get('query', '')
        return bool(query and len(query.strip()) > 3)
    
    exclude_args = []
    if not include_context:
        exclude_args.append('context')
    if not include_mode:
        exclude_args.append('mode')
    
    return async_cache_result(
        cache_name="reasoning",
        ttl_seconds=ttl_seconds,
        key_prefix="nwtn_reasoning",
        exclude_args=exclude_args,
        condition=condition
    )


def embedding_cache(
    ttl_seconds: int = 604800,  # 7 days
    content_hash: bool = True
) -> Callable[[F], F]:
    """
    Specialized cache decorator for embedding operations.
    
    Args:
        ttl_seconds: Cache TTL (default 7 days)
        content_hash: Whether to use content hash for cache key
        
    Returns:
        Decorated function with embedding-specific caching
        
    Example:
        @embedding_cache(ttl_seconds=604800)
        async def generate_embeddings(texts: List[str]) -> List[List[float]]:
            # Expensive embedding generation
            return embeddings
    """
    def key_generator(func, args, kwargs):
        if content_hash and args:
            # Use content hash for consistent caching
            content = str(args[0])
            content_hash_val = hashlib.sha256(content.encode()).hexdigest()
            return f"embedding_{func.__name__}_{content_hash_val}"
        return None
    
    def condition(*args, **kwargs):
        # Don't cache empty content
        content = args[0] if args else kwargs.get('text', kwargs.get('texts', ''))
        return bool(content)
    
    return async_cache_result(
        cache_name="embedding",
        ttl_seconds=ttl_seconds,
        key_prefix="embedding",
        condition=condition
    )


def api_cache(
    ttl_seconds: int = 86400,  # 24 hours
    include_auth: bool = False,
    rate_limit_key: Optional[str] = None
) -> Callable[[F], F]:
    """
    Specialized cache decorator for API calls.
    
    Args:
        ttl_seconds: Cache TTL (default 24 hours)
        include_auth: Whether to include auth info in cache key
        rate_limit_key: Key for rate limiting (optional)
        
    Returns:
        Decorated function with API-specific caching
        
    Example:
        @api_cache(ttl_seconds=43200, rate_limit_key="anthropic_api")
        async def call_anthropic_api(query: str, model: str) -> Dict:
            # Expensive API call
            return api_response
    """
    exclude_args = []
    if not include_auth:
        exclude_args.extend(['api_key', 'auth', 'token', 'credentials'])
    
    def condition(*args, **kwargs):
        # Check rate limiting if specified
        if rate_limit_key:
            # TODO: Implement rate limiting check
            pass
        return True
    
    return async_cache_result(
        cache_name="api_response",
        ttl_seconds=ttl_seconds,
        key_prefix="api_call",
        exclude_args=exclude_args,
        condition=condition
    )


def db_cache(
    ttl_seconds: int = 1800,  # 30 minutes
    invalidate_on_write: bool = True
) -> Callable[[F], F]:
    """
    Specialized cache decorator for database operations.
    
    Args:
        ttl_seconds: Cache TTL (default 30 minutes)
        invalidate_on_write: Whether to invalidate cache on write operations
        
    Returns:
        Decorated function with database-specific caching
        
    Example:
        @db_cache(ttl_seconds=1800)
        async def get_user_sessions(user_id: str) -> List[Dict]:
            # Database query
            return sessions
    """
    def condition(*args, **kwargs):
        # Don't cache write operations if specified
        if invalidate_on_write:
            func_name = args[0].__class__.__name__ if args else ""
            if any(op in func_name.lower() for op in ['create', 'update', 'delete', 'insert']):
                return False
        return True
    
    return async_cache_result(
        cache_name="database",
        ttl_seconds=ttl_seconds,
        key_prefix="db_query",
        condition=condition
    )


def invalidate_cache(
    cache_name: str,
    key_pattern: Optional[str] = None,
    clear_all: bool = False
) -> Callable[[F], F]:
    """
    Decorator to invalidate cache after function execution.
    
    Args:
        cache_name: Name of cache to invalidate
        key_pattern: Pattern of keys to invalidate (None for function-specific)
        clear_all: Whether to clear entire cache
        
    Returns:
        Decorated function that invalidates cache after execution
        
    Example:
        @invalidate_cache(cache_name="database", key_pattern="user_*")
        async def update_user(user_id: str, data: Dict) -> None:
            # Update user data
            pass
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                
                cache_manager = get_cache_manager()
                if clear_all:
                    await cache_manager.invalidate(cache_name)
                elif key_pattern:
                    # TODO: Implement pattern-based invalidation
                    await cache_manager.invalidate(cache_name, key_pattern)
                else:
                    cache_key = _generate_cache_key(func, args, kwargs)
                    await cache_manager.invalidate(cache_name, cache_key)
                
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                cache_manager = get_cache_manager()
                if clear_all:
                    asyncio.run(cache_manager.invalidate(cache_name))
                elif key_pattern:
                    asyncio.run(cache_manager.invalidate(cache_name, key_pattern))
                else:
                    cache_key = _generate_cache_key(func, args, kwargs)
                    asyncio.run(cache_manager.invalidate(cache_name, cache_key))
                
                return result
            return sync_wrapper
    
    return decorator


def _generate_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    key_prefix: Optional[str] = None,
    include_args: bool = True,
    include_kwargs: bool = True,
    exclude_args: Optional[list] = None
) -> str:
    """Generate cache key from function and arguments"""
    key_parts = []
    
    # Add prefix if provided
    if key_prefix:
        key_parts.append(key_prefix)
    
    # Add function name
    key_parts.append(func.__name__)
    
    # Add module name for uniqueness
    if hasattr(func, '__module__'):
        key_parts.append(func.__module__.split('.')[-1])
    
    # Process arguments
    if include_args and args:
        filtered_args = list(args)
        if exclude_args:
            # Remove excluded argument indices
            for i in sorted(exclude_args, reverse=True):
                if i < len(filtered_args):
                    filtered_args.pop(i)
        
        # Convert args to strings
        arg_strs = []
        for arg in filtered_args:
            if hasattr(arg, '__dict__'):
                # For objects, use class name and key attributes
                arg_str = f"{arg.__class__.__name__}_{id(arg)}"
            else:
                arg_str = str(arg)
            arg_strs.append(arg_str)
        
        if arg_strs:
            key_parts.append('args:' + ','.join(arg_strs))
    
    # Process keyword arguments
    if include_kwargs and kwargs:
        filtered_kwargs = dict(kwargs)
        if exclude_args:
            # Remove excluded keyword arguments
            for key in exclude_args:
                filtered_kwargs.pop(key, None)
        
        # Sort kwargs for consistent keys
        sorted_kwargs = sorted(filtered_kwargs.items())
        kwargs_str = ','.join(f"{k}:{v}" for k, v in sorted_kwargs)
        key_parts.append('kwargs:' + kwargs_str)
    
    # Join all parts and hash for fixed length
    key_string = '|'.join(key_parts)
    
    # Create hash for consistent key length
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()
    
    # Return truncated hash with function name for readability
    return f"{func.__name__}_{key_hash[:16]}"


def _should_cache_result(result: Any, execution_time: float) -> bool:
    """Determine if result should be cached based on various factors"""
    
    # Don't cache None results
    if result is None:
        return False
    
    # Don't cache empty results
    if not result:
        return False
    
    # Cache if execution took significant time (>100ms)
    if execution_time > 0.1:
        return True
    
    # Cache if result is substantial
    try:
        result_size = len(str(result))
        if result_size > 100:  # Substantial result
            return True
    except Exception:
        pass
    
    # Default to caching for consistency
    return True


# Context manager for temporary cache control
class CacheContext:
    """Context manager for temporary cache configuration"""
    
    def __init__(
        self,
        cache_name: str,
        enabled: bool = True,
        ttl_override: Optional[int] = None
    ):
        self.cache_name = cache_name
        self.enabled = enabled
        self.ttl_override = ttl_override
        self.original_config = {}
    
    def __enter__(self):
        # Store original configuration
        cache_manager = get_cache_manager()
        cache = cache_manager.get_cache(self.cache_name)
        
        if cache:
            self.original_config = {
                'enabled': getattr(cache, 'enabled', True),
                'default_ttl': getattr(cache, 'default_ttl', None)
            }
            
            # Apply temporary configuration
            if hasattr(cache, 'enabled'):
                cache.enabled = self.enabled
            if self.ttl_override and hasattr(cache, 'default_ttl'):
                cache.default_ttl = self.ttl_override
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original configuration
        cache_manager = get_cache_manager()
        cache = cache_manager.get_cache(self.cache_name)
        
        if cache and self.original_config:
            if hasattr(cache, 'enabled'):
                cache.enabled = self.original_config['enabled']
            if hasattr(cache, 'default_ttl'):
                cache.default_ttl = self.original_config['default_ttl']
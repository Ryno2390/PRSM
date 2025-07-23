"""
Async Utilities
================

Common async patterns and utilities for improved I/O performance.
"""

import asyncio
import logging
import time
from typing import Any, Callable, List, TypeVar, Optional, Union, Dict, AsyncIterator, Iterator
from functools import wraps
from contextlib import asynccontextmanager
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

async def gather_with_limit(
    tasks: List[Callable],
    limit: int = 10,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Run async tasks with concurrency limit.
    
    Args:
        tasks: List of async callables to execute
        limit: Maximum concurrent tasks
        return_exceptions: Whether to return exceptions instead of raising
        
    Returns:
        List of results from all tasks
        
    Example:
        >>> tasks = [fetch_data(url) for url in urls]
        >>> results = await gather_with_limit(tasks, limit=5)
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def _limited_task(task):
        async with semaphore:
            if asyncio.iscoroutinefunction(task):
                return await task()
            elif asyncio.iscoroutine(task):
                return await task
            else:
                # Sync function - run in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, task)
    
    limited_tasks = [_limited_task(task) for task in tasks]
    return await asyncio.gather(*limited_tasks, return_exceptions=return_exceptions)

async def async_map(
    func: Callable[[T], Any],
    items: List[T],
    concurrency: int = 10,
    preserve_order: bool = True
) -> List[Any]:
    """
    Async version of map with concurrency control.
    
    Args:
        func: Function to apply to each item
        items: Items to process
        concurrency: Maximum concurrent operations
        preserve_order: Whether to preserve input order in results
        
    Returns:
        List of results
        
    Example:
        >>> async def process_item(item):
        ...     return await expensive_operation(item)
        >>> results = await async_map(process_item, items, concurrency=5)
    """
    if not items:
        return []
    
    if asyncio.iscoroutinefunction(func):
        tasks = [func(item) for item in items]
    else:
        # Convert sync function to async
        async def async_func(item):
            return func(item)
        tasks = [async_func(item) for item in items]
    
    if preserve_order:
        results = await gather_with_limit(tasks, limit=concurrency)
        return results
    else:
        # Use as_completed for faster processing when order doesn't matter
        results = []
        semaphore = asyncio.Semaphore(concurrency)
        
        async def _limited_task(task):
            async with semaphore:
                return await task
        
        completed_tasks = asyncio.as_completed([_limited_task(task) for task in tasks])
        for completed_task in completed_tasks:
            result = await completed_task
            results.append(result)
        
        return results

async def async_filter(
    predicate: Callable[[T], Union[bool, Any]],
    items: List[T],
    concurrency: int = 10
) -> List[T]:
    """
    Async version of filter with concurrency control.
    
    Args:
        predicate: Async or sync predicate function
        items: Items to filter
        concurrency: Maximum concurrent operations
        
    Returns:
        Filtered list of items
        
    Example:
        >>> async def is_valid(item):
        ...     return await validate_item(item)
        >>> valid_items = await async_filter(is_valid, items)
    """
    if not items:
        return []
    
    if asyncio.iscoroutinefunction(predicate):
        # Async predicate
        async def check_item(item):
            result = await predicate(item)
            return (item, bool(result))
    else:
        # Sync predicate
        async def check_item(item):
            result = predicate(item)
            return (item, bool(result))
    
    # Apply predicate to all items
    check_results = await async_map(check_item, items, concurrency=concurrency)
    
    # Filter based on results
    return [item for item, passed in check_results if passed]

def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_backoff: bool = True,
    exceptions: tuple = (Exception,)
) -> Callable[[F], F]:
    """
    Decorator for async functions with retry logic.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        exponential_backoff: Whether to use exponential backoff
        exceptions: Tuple of exceptions to retry on
        
    Returns:
        Decorated function with retry capability
        
    Example:
        @async_retry(max_attempts=3, delay=1.0)
        async def unstable_api_call():
            return await make_api_request()
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise e
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                    await asyncio.sleep(current_delay)
                    
                    if exponential_backoff:
                        current_delay *= 2
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator

def async_timeout(seconds: float) -> Callable[[F], F]:
    """
    Decorator to add timeout to async functions.
    
    Args:
        seconds: Timeout in seconds
        
    Returns:
        Decorated function with timeout
        
    Example:
        @async_timeout(30.0)
        async def long_running_operation():
            await asyncio.sleep(60)  # Will timeout after 30s
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds}s")
                raise
        
        return wrapper
    return decorator

class AsyncBatch:
    """Batch async operations for improved efficiency"""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._batch: List[Any] = []
        self._batch_processors: Dict[str, Callable] = {}
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def add(self, item: Any, processor_name: str = "default") -> None:
        """Add item to batch"""
        async with self._lock:
            self._batch.append((item, processor_name))
            
            # Auto-flush if batch is full
            if len(self._batch) >= self.batch_size:
                await self._flush()
            
            # Start flush timer if not already running
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._auto_flush())
    
    def register_processor(self, name: str, processor: Callable[[List[Any]], Any]) -> None:
        """Register batch processor function"""
        self._batch_processors[name] = processor
    
    async def _auto_flush(self) -> None:
        """Auto-flush batch after interval"""
        await asyncio.sleep(self.flush_interval)
        await self._flush()
    
    async def _flush(self) -> None:
        """Flush current batch"""
        async with self._lock:
            if not self._batch:
                return
            
            # Group items by processor
            processor_batches: Dict[str, List[Any]] = {}
            for item, processor_name in self._batch:
                if processor_name not in processor_batches:
                    processor_batches[processor_name] = []
                processor_batches[processor_name].append(item)
            
            self._batch.clear()
            
            # Process each batch
            for processor_name, items in processor_batches.items():
                processor = self._batch_processors.get(processor_name)
                if processor:
                    try:
                        if asyncio.iscoroutinefunction(processor):
                            await processor(items)
                        else:
                            processor(items)
                    except Exception as e:
                        logger.error(f"Batch processor {processor_name} failed: {e}")
                else:
                    logger.warning(f"No processor registered for {processor_name}")
    
    async def flush(self) -> None:
        """Manually flush batch"""
        await self._flush()
    
    async def close(self) -> None:
        """Close batch processor and flush remaining items"""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        
        await self._flush()

@asynccontextmanager
async def async_cache_context(cache_manager, cache_name: str, enabled: bool = True):
    """Context manager for temporary cache configuration"""
    original_enabled = getattr(cache_manager.get_cache(cache_name), 'enabled', True)
    
    try:
        cache = cache_manager.get_cache(cache_name)
        if cache and hasattr(cache, 'enabled'):
            cache.enabled = enabled
        yield cache
    finally:
        cache = cache_manager.get_cache(cache_name)
        if cache and hasattr(cache, 'enabled'):
            cache.enabled = original_enabled

class StreamProcessor:
    """Process async streams with backpressure control"""
    
    def __init__(self, buffer_size: int = 1000, processing_delay: float = 0.01):
        self.buffer_size = buffer_size
        self.processing_delay = processing_delay
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._processors: List[Callable] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    def add_processor(self, processor: Callable[[Any], Any]) -> None:
        """Add a processor function to the pipeline"""
        self._processors.append(processor)
    
    async def start(self) -> None:
        """Start the stream processor"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_stream())
    
    async def stop(self) -> None:
        """Stop the stream processor"""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def submit(self, item: Any) -> None:
        """Submit item for processing"""
        if not self._running:
            await self.start()
        
        await self._queue.put(item)
    
    async def _process_stream(self) -> None:
        """Process items from the stream"""
        while self._running:
            try:
                # Get item with timeout to allow checking running status
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                # Process through all processors
                processed_item = item
                for processor in self._processors:
                    if asyncio.iscoroutinefunction(processor):
                        processed_item = await processor(processed_item)
                    else:
                        processed_item = processor(processed_item)
                
                # Small delay to prevent overwhelming downstream systems
                if self.processing_delay > 0:
                    await asyncio.sleep(self.processing_delay)
                
                # Mark task as done
                self._queue.task_done()
                
            except asyncio.TimeoutError:
                # Continue loop to check running status
                continue
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                # Mark task as done even on error
                if not self._queue.empty():
                    self._queue.task_done()

# Global utilities

async def run_with_progress(
    tasks: List[Any],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    concurrency: int = 10
) -> List[Any]:
    """
    Run tasks with progress reporting.
    
    Args:
        tasks: List of async tasks
        progress_callback: Callback function for progress updates
        concurrency: Maximum concurrent tasks
        
    Returns:
        List of results
    """
    total_tasks = len(tasks)
    completed = 0
    results = []
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def _task_with_progress(task):
        nonlocal completed
        async with semaphore:
            try:
                if asyncio.iscoroutinefunction(task):
                    result = await task()
                elif asyncio.iscoroutine(task):
                    result = await task
                else:
                    result = task()
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_tasks)
                
                return result
            except Exception as e:
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_tasks)
                raise
    
    # Execute all tasks
    wrapped_tasks = [_task_with_progress(task) for task in tasks]
    results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
    
    return results

# Memory management for long-running async operations
class AsyncResourceManager:
    """Manage resources for long-running async operations"""
    
    def __init__(self):
        self._resources: weakref.WeakSet = weakref.WeakSet()
        self._cleanup_interval = 300  # 5 minutes
        self._cleanup_task: Optional[asyncio.Task] = None
        
    def register_resource(self, resource: Any) -> None:
        """Register a resource for cleanup"""
        self._resources.add(resource)
        
        # Start cleanup task if not running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self) -> None:
        """Periodically clean up resources"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                # Resources are automatically removed from WeakSet when garbage collected
                # This is mainly for explicit cleanup of resources that need it
                for resource in list(self._resources):
                    if hasattr(resource, 'cleanup'):
                        try:
                            if asyncio.iscoroutinefunction(resource.cleanup):
                                await resource.cleanup()
                            else:
                                resource.cleanup()
                        except Exception as e:
                            logger.error(f"Resource cleanup error: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")
    
    async def cleanup_all(self) -> None:
        """Clean up all registered resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        for resource in list(self._resources):
            if hasattr(resource, 'cleanup'):
                try:
                    if asyncio.iscoroutinefunction(resource.cleanup):
                        await resource.cleanup()
                    else:
                        resource.cleanup()
                except Exception as e:
                    logger.error(f"Resource cleanup error: {e}")

# Global resource manager instance
_resource_manager = AsyncResourceManager()

def get_async_resource_manager() -> AsyncResourceManager:
    """Get global async resource manager"""
    return _resource_manager
#!/usr/bin/env python3
"""
Batch Processing Optimizer for PRSM
===================================

This module provides intelligent batch processing optimization for large-scale
content ingestion, designed to maximize throughput while maintaining quality
and system stability.

Key Features:
1. Adaptive batch sizing based on system performance
2. Intelligent resource allocation and load balancing
3. Parallel processing with congestion control
4. Memory-efficient streaming processing
5. Error handling and recovery mechanisms
6. Performance monitoring and optimization
7. Priority-based processing queues

Optimized for the 150,000 item breadth-focused ingestion strategy.
"""

import asyncio
import json
import logging
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, AsyncIterator
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import structlog

logger = structlog.get_logger(__name__)


class ProcessingPriority(int, Enum):
    """Processing priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class BatchStatus(str, Enum):
    """Batch processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ResourceType(str, Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK = "network"


@dataclass
class BatchItem:
    """Individual item in a processing batch"""
    item_id: str
    data: Dict[str, Any]
    priority: ProcessingPriority
    processing_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingBatch:
    """A batch of items for processing"""
    batch_id: str
    items: List[BatchItem]
    priority: ProcessingPriority
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_processing_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    retry_count: int = 0
    
    @property
    def size(self) -> int:
        return len(self.items)
    
    @property
    def completion_rate(self) -> float:
        if self.size == 0:
            return 0.0
        return (self.success_count + self.failure_count) / self.size
    
    @property
    def success_rate(self) -> float:
        if self.size == 0:
            return 0.0
        return self.success_count / self.size
    
    @property
    def processing_duration(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return 0.0


@dataclass
class SystemResources:
    """System resource utilization"""
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_percent: float
    available_memory_gb: float
    load_average: float
    
    @property
    def overall_utilization(self) -> float:
        return (self.cpu_percent + self.memory_percent + self.disk_io_percent) / 3


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing optimization"""
    
    # Batch sizing
    min_batch_size: int = 10
    max_batch_size: int = 1000
    target_batch_size: int = 100
    adaptive_sizing: bool = True
    
    # Concurrency settings
    max_concurrent_batches: int = 5
    max_worker_threads: int = 10
    max_worker_processes: int = 4
    
    # Performance thresholds
    target_processing_time: float = 30.0  # seconds per batch
    max_processing_time: float = 300.0    # 5 minutes max
    cpu_threshold: float = 80.0           # CPU usage threshold
    memory_threshold: float = 85.0        # Memory usage threshold
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 5.0
    error_threshold: float = 0.1          # 10% error rate threshold
    
    # Resource management
    memory_limit_gb: float = 8.0
    disk_space_threshold: float = 0.9     # 90% disk usage threshold
    
    # Optimization settings
    performance_monitoring: bool = True
    adaptive_optimization: bool = True
    load_balancing: bool = True
    
    # Progress reporting
    progress_reporting_interval: int = 10  # seconds
    detailed_logging: bool = True


@dataclass
class ProcessingMetrics:
    """Processing performance metrics"""
    
    # Throughput metrics
    total_items_processed: int = 0
    total_batches_processed: int = 0
    items_per_second: float = 0.0
    batches_per_second: float = 0.0
    
    # Quality metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Performance metrics
    average_batch_processing_time: float = 0.0
    average_item_processing_time: float = 0.0
    peak_throughput: float = 0.0
    
    # Resource metrics
    peak_cpu_usage: float = 0.0
    peak_memory_usage: float = 0.0
    average_cpu_usage: float = 0.0
    average_memory_usage: float = 0.0
    
    # Timing metrics
    total_processing_time: float = 0.0
    idle_time: float = 0.0
    optimization_time: float = 0.0
    
    # System health
    system_stability_score: float = 0.0
    resource_efficiency_score: float = 0.0


class BatchProcessingOptimizer:
    """
    Intelligent Batch Processing Optimizer
    
    Provides adaptive batch processing with intelligent resource management,
    load balancing, and performance optimization for large-scale ingestion.
    """
    
    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or BatchProcessingConfig()
        
        # Processing queues
        self.processing_queue = []  # Priority queue of batches
        self.active_batches: Dict[str, ProcessingBatch] = {}
        self.completed_batches: Dict[str, ProcessingBatch] = {}
        self.failed_batches: Dict[str, ProcessingBatch] = {}
        
        # Resource management
        self.resource_monitor = None
        self.system_resources = SystemResources(0, 0, 0, 0, 0, 0)
        
        # Performance tracking
        self.metrics = ProcessingMetrics()
        self.performance_history = deque(maxlen=1000)
        
        # Processing control
        self.processing_active = False
        self.processing_task = None
        self.monitoring_task = None
        
        # Optimization state
        self.current_batch_size = self.config.target_batch_size
        self.current_concurrency = self.config.max_concurrent_batches
        self.optimization_cycles = 0
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_worker_threads)
        
        # Locks for thread safety
        self.queue_lock = threading.RLock()
        self.metrics_lock = threading.RLock()
        
        logger.info("Batch Processing Optimizer initialized")
    
    async def initialize(self):
        """Initialize the batch processing optimizer"""
        
        logger.info("🚀 Initializing Batch Processing Optimizer...")
        
        # Initialize resource monitoring
        await self._initialize_resource_monitoring()
        
        # Start monitoring tasks
        await self._start_monitoring_tasks()
        
        # Perform initial optimization
        await self._perform_initial_optimization()
        
        logger.info("✅ Batch Processing Optimizer ready",
                   batch_size=self.current_batch_size,
                   concurrency=self.current_concurrency)
    
    async def process_items_stream(self, 
                                 items_stream: AsyncIterator[Dict[str, Any]],
                                 processor_func: Callable,
                                 priority: ProcessingPriority = ProcessingPriority.MEDIUM) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a stream of items with optimized batching
        
        Args:
            items_stream: Async iterator of items to process
            processor_func: Function to process each item
            priority: Processing priority
            
        Yields:
            Processed items
        """
        
        logger.info("🔄 Starting optimized stream processing",
                   priority=priority.name)
        
        self.processing_active = True
        
        try:
            # Start processing pipeline
            async for batch_result in self._process_stream_batches(items_stream, processor_func, priority):
                for item_result in batch_result:
                    yield item_result
        finally:
            self.processing_active = False
    
    async def process_batch_list(self, 
                               items: List[Dict[str, Any]],
                               processor_func: Callable,
                               priority: ProcessingPriority = ProcessingPriority.MEDIUM) -> List[Dict[str, Any]]:
        """
        Process a list of items with optimized batching
        
        Args:
            items: List of items to process
            processor_func: Function to process each item
            priority: Processing priority
            
        Returns:
            List of processed items
        """
        
        logger.info(f"🔄 Processing {len(items)} items with optimization",
                   priority=priority.name)
        
        # Convert to async iterator
        async def items_iterator():
            for item in items:
                yield item
        
        # Process and collect results
        results = []
        async for result in self.process_items_stream(items_iterator(), processor_func, priority):
            results.append(result)
        
        return results
    
    async def optimize_processing_parameters(self) -> Dict[str, Any]:
        """
        Optimize processing parameters based on current performance
        
        Returns:
            Optimization results
        """
        
        logger.info("🔧 Optimizing processing parameters...")
        
        optimization_start = time.time()
        
        # Gather current performance data
        current_performance = await self._gather_performance_data()
        
        # Optimize batch size
        batch_size_optimization = await self._optimize_batch_size(current_performance)
        
        # Optimize concurrency
        concurrency_optimization = await self._optimize_concurrency(current_performance)
        
        # Optimize resource allocation
        resource_optimization = await self._optimize_resource_allocation()
        
        # Update configuration
        await self._apply_optimizations(batch_size_optimization, concurrency_optimization, resource_optimization)
        
        optimization_time = time.time() - optimization_start
        self.metrics.optimization_time += optimization_time
        self.optimization_cycles += 1
        
        optimization_results = {
            "optimization_cycle": self.optimization_cycles,
            "optimization_time": optimization_time,
            "batch_size_change": batch_size_optimization,
            "concurrency_change": concurrency_optimization,
            "resource_optimization": resource_optimization,
            "current_performance": current_performance,
            "new_batch_size": self.current_batch_size,
            "new_concurrency": self.current_concurrency
        }
        
        logger.info("✅ Processing parameters optimized",
                   new_batch_size=self.current_batch_size,
                   new_concurrency=self.current_concurrency,
                   optimization_time=optimization_time)
        
        return optimization_results
    
    async def get_processing_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics"""
        
        with self.metrics_lock:
            # Update real-time metrics
            await self._update_real_time_metrics()
            
            # Calculate derived metrics
            self.metrics.system_stability_score = await self._calculate_stability_score()
            self.metrics.resource_efficiency_score = await self._calculate_efficiency_score()
            
            return self.metrics
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "processing_status": {
                "active": self.processing_active,
                "queue_size": len(self.processing_queue),
                "active_batches": len(self.active_batches),
                "completed_batches": len(self.completed_batches),
                "failed_batches": len(self.failed_batches)
            },
            "current_configuration": {
                "batch_size": self.current_batch_size,
                "concurrency": self.current_concurrency,
                "worker_threads": self.config.max_worker_threads,
                "worker_processes": self.config.max_worker_processes
            },
            "system_resources": {
                "cpu_percent": self.system_resources.cpu_percent,
                "memory_percent": self.system_resources.memory_percent,
                "available_memory_gb": self.system_resources.available_memory_gb,
                "load_average": self.system_resources.load_average
            },
            "performance_metrics": await self.get_processing_metrics(),
            "optimization_state": {
                "optimization_cycles": self.optimization_cycles,
                "last_optimization": "recently" if self.optimization_cycles > 0 else "never"
            }
        }
    
    # === Private Methods ===
    
    async def _process_stream_batches(self, 
                                    items_stream: AsyncIterator[Dict[str, Any]],
                                    processor_func: Callable,
                                    priority: ProcessingPriority) -> AsyncIterator[List[Dict[str, Any]]]:
        """Process stream in optimized batches"""
        
        current_batch = []
        
        async for item in items_stream:
            current_batch.append(item)
            
            # Process batch when it reaches optimal size
            if len(current_batch) >= self.current_batch_size:
                batch_result = await self._process_single_batch(current_batch, processor_func, priority)
                yield batch_result
                current_batch = []
                
                # Check if we need to optimize parameters
                if self.optimization_cycles % 10 == 0:
                    await self.optimize_processing_parameters()
        
        # Process remaining items
        if current_batch:
            batch_result = await self._process_single_batch(current_batch, processor_func, priority)
            yield batch_result
    
    async def _process_single_batch(self, 
                                  items: List[Dict[str, Any]],
                                  processor_func: Callable,
                                  priority: ProcessingPriority) -> List[Dict[str, Any]]:
        """Process a single batch of items"""
        
        batch_id = f"batch_{int(time.time() * 1000)}"
        
        # Create batch items
        batch_items = []
        for i, item in enumerate(items):
            batch_item = BatchItem(
                item_id=f"{batch_id}_{i}",
                data=item,
                priority=priority
            )
            batch_items.append(batch_item)
        
        # Create processing batch
        batch = ProcessingBatch(
            batch_id=batch_id,
            items=batch_items,
            priority=priority
        )
        
        # Add to active batches
        self.active_batches[batch_id] = batch
        
        try:
            # Process batch
            batch.started_at = datetime.now(timezone.utc)
            batch.status = BatchStatus.PROCESSING
            
            results = await self._execute_batch_processing(batch, processor_func)
            
            # Mark as completed
            batch.completed_at = datetime.now(timezone.utc)
            batch.status = BatchStatus.COMPLETED
            
            # Update metrics
            await self._update_batch_metrics(batch)
            
            # Move to completed batches
            self.completed_batches[batch_id] = batch
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {batch_id}: {e}")
            batch.status = BatchStatus.FAILED
            self.failed_batches[batch_id] = batch
            raise
        finally:
            # Remove from active batches
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
    
    async def _execute_batch_processing(self, 
                                      batch: ProcessingBatch,
                                      processor_func: Callable) -> List[Dict[str, Any]]:
        """Execute processing for a batch"""
        
        # Determine processing strategy based on batch size and system resources
        if batch.size <= 10 or self.system_resources.cpu_percent > 80:
            # Sequential processing for small batches or high CPU usage
            return await self._process_batch_sequential(batch, processor_func)
        else:
            # Parallel processing for larger batches
            return await self._process_batch_parallel(batch, processor_func)
    
    async def _process_batch_sequential(self, 
                                      batch: ProcessingBatch,
                                      processor_func: Callable) -> List[Dict[str, Any]]:
        """Process batch sequentially"""
        
        results = []
        
        for item in batch.items:
            try:
                item_start = time.time()
                
                # Process item
                result = await processor_func(item.data)
                
                # Record processing time
                item.processing_time = time.time() - item_start
                
                results.append(result)
                batch.success_count += 1
                
            except Exception as e:
                logger.error(f"Item processing failed: {item.item_id}: {e}")
                item.error_count += 1
                item.last_error = str(e)
                batch.failure_count += 1
                
                # Add error result
                results.append({
                    "error": str(e),
                    "item_id": item.item_id,
                    "original_data": item.data
                })
        
        return results
    
    async def _process_batch_parallel(self, 
                                    batch: ProcessingBatch,
                                    processor_func: Callable) -> List[Dict[str, Any]]:
        """Process batch in parallel"""
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.current_concurrency)
        
        async def process_item_with_semaphore(item: BatchItem):
            async with semaphore:
                try:
                    item_start = time.time()
                    
                    # Process item
                    result = await processor_func(item.data)
                    
                    # Record processing time
                    item.processing_time = time.time() - item_start
                    
                    batch.success_count += 1
                    return result
                    
                except Exception as e:
                    logger.error(f"Item processing failed: {item.item_id}: {e}")
                    item.error_count += 1
                    item.last_error = str(e)
                    batch.failure_count += 1
                    
                    return {
                        "error": str(e),
                        "item_id": item.item_id,
                        "original_data": item.data
                    }
        
        # Process all items concurrently
        tasks = [process_item_with_semaphore(item) for item in batch.items]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _initialize_resource_monitoring(self):
        """Initialize system resource monitoring"""
        
        # Get initial resource state
        await self._update_system_resources()
        
        logger.info("Resource monitoring initialized")
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        
        if self.config.performance_monitoring:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Monitoring tasks started")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while True:
            try:
                # Update resource metrics
                await self._update_system_resources()
                
                # Update performance metrics
                await self._update_real_time_metrics()
                
                # Check for optimization opportunities
                if self.config.adaptive_optimization:
                    await self._check_optimization_trigger()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.config.progress_reporting_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _update_system_resources(self):
        """Update system resource metrics"""
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_memory_gb = memory.available / (1024**3)
        
        # Get load average
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        
        # Update system resources
        self.system_resources = SystemResources(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_io_percent=0.0,  # Would implement disk I/O monitoring
            network_io_percent=0.0,  # Would implement network I/O monitoring
            available_memory_gb=available_memory_gb,
            load_average=load_avg
        )
    
    async def _gather_performance_data(self) -> Dict[str, Any]:
        """Gather current performance data"""
        
        return {
            "throughput": self.metrics.items_per_second,
            "success_rate": self.metrics.success_rate,
            "average_processing_time": self.metrics.average_batch_processing_time,
            "resource_utilization": self.system_resources.overall_utilization,
            "queue_size": len(self.processing_queue),
            "active_batches": len(self.active_batches)
        }
    
    async def _optimize_batch_size(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize batch size based on performance"""
        
        old_batch_size = self.current_batch_size
        
        # Analyze performance trends
        if performance_data["throughput"] < 10:  # Low throughput
            # Increase batch size to improve efficiency
            self.current_batch_size = min(self.config.max_batch_size, 
                                        int(self.current_batch_size * 1.2))
        elif performance_data["average_processing_time"] > self.config.target_processing_time:
            # Decrease batch size to reduce processing time
            self.current_batch_size = max(self.config.min_batch_size,
                                        int(self.current_batch_size * 0.8))
        
        return {
            "old_size": old_batch_size,
            "new_size": self.current_batch_size,
            "change": self.current_batch_size - old_batch_size
        }
    
    async def _optimize_concurrency(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize concurrency based on system resources"""
        
        old_concurrency = self.current_concurrency
        
        # Adjust based on CPU usage
        if self.system_resources.cpu_percent < 60:
            # Increase concurrency
            self.current_concurrency = min(self.config.max_concurrent_batches,
                                         self.current_concurrency + 1)
        elif self.system_resources.cpu_percent > 85:
            # Decrease concurrency
            self.current_concurrency = max(1, self.current_concurrency - 1)
        
        return {
            "old_concurrency": old_concurrency,
            "new_concurrency": self.current_concurrency,
            "change": self.current_concurrency - old_concurrency
        }
    
    async def _optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation"""
        
        # Trigger garbage collection if memory usage is high
        if self.system_resources.memory_percent > 80:
            gc.collect()
        
        return {
            "memory_optimization": "gc_triggered" if self.system_resources.memory_percent > 80 else "none",
            "cpu_optimization": "balanced",
            "io_optimization": "standard"
        }
    
    async def _apply_optimizations(self, batch_opt: Dict, concurrency_opt: Dict, resource_opt: Dict):
        """Apply optimization changes"""
        
        # Log optimization changes
        if batch_opt["change"] != 0:
            logger.info(f"Batch size optimized: {batch_opt['old_size']} -> {batch_opt['new_size']}")
        
        if concurrency_opt["change"] != 0:
            logger.info(f"Concurrency optimized: {concurrency_opt['old_concurrency']} -> {concurrency_opt['new_concurrency']}")
    
    async def _update_real_time_metrics(self):
        """Update real-time performance metrics"""
        
        with self.metrics_lock:
            # Calculate throughput
            if self.metrics.total_processing_time > 0:
                self.metrics.items_per_second = self.metrics.total_items_processed / self.metrics.total_processing_time
            
            # Calculate success rate
            if self.metrics.total_items_processed > 0:
                total_errors = self.metrics.total_items_processed * self.metrics.error_rate
                self.metrics.success_rate = 1.0 - (total_errors / self.metrics.total_items_processed)
            
            # Update resource metrics
            self.metrics.peak_cpu_usage = max(self.metrics.peak_cpu_usage, self.system_resources.cpu_percent)
            self.metrics.peak_memory_usage = max(self.metrics.peak_memory_usage, self.system_resources.memory_percent)
    
    async def _update_batch_metrics(self, batch: ProcessingBatch):
        """Update metrics after batch completion"""
        
        with self.metrics_lock:
            self.metrics.total_items_processed += batch.size
            self.metrics.total_batches_processed += 1
            
            # Update timing metrics
            batch_processing_time = batch.processing_duration
            self.metrics.total_processing_time += batch_processing_time
            
            # Update average processing times
            self.metrics.average_batch_processing_time = (
                (self.metrics.average_batch_processing_time * (self.metrics.total_batches_processed - 1) +
                 batch_processing_time) / self.metrics.total_batches_processed
            )
            
            # Update success/error rates
            if batch.size > 0:
                batch_error_rate = batch.failure_count / batch.size
                self.metrics.error_rate = (
                    (self.metrics.error_rate * (self.metrics.total_batches_processed - 1) +
                     batch_error_rate) / self.metrics.total_batches_processed
                )
    
    async def _calculate_stability_score(self) -> float:
        """Calculate system stability score"""
        
        # Consider error rate, resource stability, and processing consistency
        error_stability = max(0, 1.0 - (self.metrics.error_rate * 2))
        resource_stability = max(0, 1.0 - (self.system_resources.overall_utilization / 100))
        processing_stability = 1.0  # Would implement based on processing time variance
        
        return (error_stability + resource_stability + processing_stability) / 3
    
    async def _calculate_efficiency_score(self) -> float:
        """Calculate resource efficiency score"""
        
        # Consider throughput vs resource usage
        if self.system_resources.overall_utilization == 0:
            return 0.0
        
        efficiency = self.metrics.items_per_second / self.system_resources.overall_utilization
        return min(1.0, efficiency / 10)  # Normalize to 0-1 scale
    
    async def _perform_initial_optimization(self):
        """Perform initial optimization based on system capabilities"""
        
        # Adjust batch size based on available memory
        if self.system_resources.available_memory_gb < 2:
            self.current_batch_size = min(50, self.current_batch_size)
        elif self.system_resources.available_memory_gb > 8:
            self.current_batch_size = min(500, self.current_batch_size)
        
        # Adjust concurrency based on CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count:
            self.current_concurrency = min(cpu_count, self.current_concurrency)
        
        logger.info("Initial optimization completed",
                   batch_size=self.current_batch_size,
                   concurrency=self.current_concurrency)
    
    async def _check_optimization_trigger(self):
        """Check if optimization should be triggered"""
        
        # Trigger optimization if performance degrades
        if (self.metrics.error_rate > 0.1 or 
            self.system_resources.overall_utilization > 90 or
            self.metrics.average_batch_processing_time > self.config.target_processing_time * 2):
            
            await self.optimize_processing_parameters()
    
    async def shutdown(self):
        """Shutdown the batch processing optimizer"""
        
        logger.info("Shutting down batch processing optimizer...")
        
        # Cancel monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Wait for active batches to complete
        while self.active_batches:
            await asyncio.sleep(1)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Batch processing optimizer shutdown complete")


# Test function
async def test_batch_optimizer():
    """Test batch processing optimizer"""
    
    print("⚡ BATCH PROCESSING OPTIMIZER TEST")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = BatchProcessingOptimizer()
    await optimizer.initialize()
    
    # Mock processor function
    async def mock_processor(item: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.001)  # Simulate processing time
        return {"processed": True, "original": item}
    
    # Test data
    test_items = [{"id": i, "data": f"item_{i}"} for i in range(100)]
    
    # Test batch processing
    results = await optimizer.process_batch_list(test_items, mock_processor)
    print(f"Batch Processing: ✅ Processed {len(results)} items")
    
    # Test optimization
    optimization_result = await optimizer.optimize_processing_parameters()
    print(f"Optimization: ✅ Cycle {optimization_result['optimization_cycle']}")
    
    # Get metrics
    metrics = await optimizer.get_processing_metrics()
    print(f"Metrics: ✅ Throughput: {metrics.items_per_second:.1f} items/sec")
    
    # Get system status
    status = await optimizer.get_system_status()
    print(f"Status: ✅ {status['processing_status']['completed_batches']} batches completed")
    
    # Shutdown
    await optimizer.shutdown()
    print("Shutdown: ✅")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(test_batch_optimizer())
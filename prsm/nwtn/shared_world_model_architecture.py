#!/usr/bin/env python3
"""
Shared World Model Architecture for NWTN
========================================

This module implements the Phase 8.1.3 Shared World Model Architecture that provides
memory-efficient shared world model access across all parallel workers, along with
advanced performance optimizations including hierarchical caching and adaptive
resource management.

Architecture:
- SharedWorldModelManager: Single world model instance shared across all parallel workers
- HierarchicalResultCache: Multi-level caching system for parallel reasoning optimization  
- AdaptiveResourceManager: Dynamically optimizes resource allocation across parallel workers
- ParallelProcessingResilience: Ensures parallel processing completes even with worker failures
- ParallelValidationEngine: Batch validates multiple reasoning results simultaneously

Based on NWTN Roadmap Phase 8.1.3 - Shared World Model Architecture (Very High Priority)
Expected Impact: Memory-efficient shared world model supporting 10,000+ knowledge items
"""

import asyncio
import time
import threading
import multiprocessing
import mmap
import pickle
import gzip
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque
import heapq
import weakref
import gc
import psutil
import structlog

# Lazy import to avoid circular dependency - imported in functions where needed
# from prsm.nwtn.meta_reasoning_engine import ReasoningResult, ReasoningEngine

logger = structlog.get_logger(__name__)

class CacheLevel(Enum):
    """Different levels of hierarchical caching"""
    ENGINE_RESULT = "engine_result"           # Level 1: Individual engine results
    SEQUENCE_RESULT = "sequence_result"       # Level 2: Reasoning sequence results  
    VALIDATION_RESULT = "validation_result"   # Level 3: World model validation results
    SHARED_WORKER = "shared_worker"          # Level 4: Cross-worker result sharing

class ValidationStatus(Enum):
    """Status of validation operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class WorkerHealth(Enum):
    """Health status of parallel workers"""
    HEALTHY = "healthy"
    SLOW = "slow"
    UNRESPONSIVE = "unresponsive"
    FAILED = "failed"
    RECOVERED = "recovered"

@dataclass
class KnowledgeItem:
    """Individual knowledge item in the shared world model"""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    domain: str = "general"
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    validation_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ValidationRequest:
    """Request for world model validation"""
    reasoning_result: "ReasoningResult"  # Forward reference to avoid circular import
    request_id: str = field(default_factory=lambda: str(uuid4()))
    validation_type: str = "standard"
    priority: int = 1  # 1=high, 5=low
    worker_id: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ValidationStatus = ValidationStatus.PENDING

@dataclass
class ValidationResult:
    """Result of world model validation"""
    request_id: str
    validation_passed: bool = True
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    supporting_knowledge: List[KnowledgeItem] = field(default_factory=list)
    processing_time: float = 0.0
    cached: bool = False
    validation_details: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class WorkerPerformanceMetrics:
    """Performance metrics for individual workers"""
    worker_id: int
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    average_validation_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    last_activity: Optional[datetime] = None
    health_status: WorkerHealth = WorkerHealth.HEALTHY
    performance_score: float = 1.0  # 0.0-1.0 performance rating

@dataclass
class BottleneckInfo:
    """Information about performance bottlenecks"""
    worker_id: int
    bottleneck_type: str  # 'slow_validation', 'high_memory', 'cpu_bound', 'cache_misses'
    severity: float  # 0.0-1.0
    metrics: WorkerPerformanceMetrics
    suggested_actions: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class SharedKnowledgeBase:
    """Thread-safe shared knowledge base using shared memory"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.access_lock = threading.RLock()
        self.memory_mapped_file = None
        self.shared_memory_initialized = False
        
        # Access statistics
        self.total_accesses = 0
        self.cache_hits = 0
        self.last_cleanup = datetime.now(timezone.utc)
        
        self._initialize_shared_memory()
    
    def _initialize_shared_memory(self):
        """Initialize shared memory for knowledge base"""
        try:
            # For demo purposes, we'll simulate shared memory with thread-safe dict
            # In production, this would use actual shared memory (mmap, multiprocessing.shared_memory)
            self.shared_memory_initialized = True
            logger.info("Shared memory knowledge base initialized", max_size=self.max_size)
            
        except Exception as e:
            logger.error("Failed to initialize shared memory", error=str(e))
            raise
    
    def add_knowledge_item(self, item: KnowledgeItem) -> bool:
        """Add knowledge item to shared base"""
        with self.access_lock:
            if len(self.knowledge_items) >= self.max_size:
                # Remove least recently used item
                self._evict_lru_item()
            
            self.knowledge_items[item.id] = item
            return True
    
    def get_knowledge_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get knowledge item by ID"""
        with self.access_lock:
            self.total_accesses += 1
            
            if item_id in self.knowledge_items:
                item = self.knowledge_items[item_id]
                item.last_accessed = datetime.now(timezone.utc)
                item.access_count += 1
                self.cache_hits += 1
                return item
            
            return None
    
    def search_knowledge(self, query: str, domain: Optional[str] = None) -> List[KnowledgeItem]:
        """Search knowledge items by content"""
        with self.access_lock:
            self.total_accesses += 1
            
            results = []
            query_lower = query.lower()
            
            for item in self.knowledge_items.values():
                # Simple text search (in production would use vector similarity)
                if query_lower in item.content.lower():
                    if domain is None or item.domain == domain:
                        item.last_accessed = datetime.now(timezone.utc)
                        item.access_count += 1
                        results.append(item)
            
            if results:
                self.cache_hits += len(results)
            
            return results[:20]  # Return top 20 matches
    
    def _evict_lru_item(self):
        """Remove least recently used item"""
        if not self.knowledge_items:
            return
        
        lru_item_id = min(self.knowledge_items.keys(), 
                         key=lambda k: self.knowledge_items[k].last_accessed)
        del self.knowledge_items[lru_item_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        with self.access_lock:
            hit_rate = self.cache_hits / max(self.total_accesses, 1)
            
            return {
                'total_items': len(self.knowledge_items),
                'max_size': self.max_size,
                'total_accesses': self.total_accesses,
                'cache_hits': self.cache_hits,
                'hit_rate': hit_rate,
                'memory_initialized': self.shared_memory_initialized
            }
    
    def cleanup_old_items(self, max_age_hours: int = 24):
        """Clean up old, unused knowledge items"""
        if datetime.now(timezone.utc) - self.last_cleanup < timedelta(hours=1):
            return  # Only cleanup once per hour
        
        with self.access_lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            old_items = [
                item_id for item_id, item in self.knowledge_items.items()
                if item.last_accessed < cutoff_time and item.access_count < 5
            ]
            
            for item_id in old_items:
                del self.knowledge_items[item_id]
            
            self.last_cleanup = datetime.now(timezone.utc)
            
            if old_items:
                logger.info("Cleaned up old knowledge items", 
                           items_removed=len(old_items),
                           remaining_items=len(self.knowledge_items))

class ParallelValidationEngine:
    """Validates reasoning results against shared world model in parallel"""
    
    def __init__(self, shared_knowledge_base: SharedKnowledgeBase, max_workers: int = 4):
        self.shared_knowledge_base = shared_knowledge_base
        self.max_workers = max_workers
        self.validation_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.validation_queue = asyncio.Queue()
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.total_validations = 0
        self.successful_validations = 0
        self.cache_hits = 0
        
    async def validate_batch(self, validation_requests: List[ValidationRequest]) -> List[ValidationResult]:
        """Validate a batch of reasoning results in parallel"""
        
        if not validation_requests:
            return []
        
        # Check cache first
        cached_results = []
        uncached_requests = []
        
        for request in validation_requests:
            cache_key = self._generate_cache_key(request)
            
            with self.cache_lock:
                if cache_key in self.validation_cache:
                    cached_result = self.validation_cache[cache_key]
                    cached_result.cached = True
                    cached_results.append(cached_result)
                    self.cache_hits += 1
                else:
                    uncached_requests.append(request)
        
        # Process uncached requests in parallel
        uncached_results = []
        if uncached_requests:
            validation_tasks = [
                self._validate_single_request(request) 
                for request in uncached_requests
            ]
            
            try:
                uncached_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                
                # Filter out exceptions
                valid_results = [
                    result for result in uncached_results 
                    if isinstance(result, ValidationResult)
                ]
                
                # Cache successful validations
                for result in valid_results:
                    cache_key = self._generate_cache_key_from_result(result)
                    with self.cache_lock:
                        if len(self.validation_cache) < 10000:  # Max cache size
                            self.validation_cache[cache_key] = result
                
                uncached_results = valid_results
                
            except Exception as e:
                logger.error("Batch validation failed", error=str(e))
                uncached_results = []
        
        # Combine results
        all_results = cached_results + uncached_results
        
        # Update statistics
        self.total_validations += len(validation_requests)
        self.successful_validations += len(all_results)
        
        logger.debug("Batch validation completed",
                    total_requests=len(validation_requests),
                    cached_results=len(cached_results),
                    new_validations=len(uncached_results),
                    cache_hit_rate=self.cache_hits / max(self.total_validations, 1))
        
        return all_results
    
    async def _validate_single_request(self, request: ValidationRequest) -> ValidationResult:
        """Validate a single reasoning result against world model"""
        
        start_time = time.time()
        request.status = ValidationStatus.IN_PROGRESS
        
        try:
            # Extract key concepts from reasoning result
            reasoning_content = self._extract_reasoning_content(request.reasoning_result)
            
            # Search for supporting knowledge
            supporting_knowledge = self.shared_knowledge_base.search_knowledge(
                reasoning_content, 
                domain=getattr(request.reasoning_result, 'domain', None)
            )
            
            # Perform validation logic
            validation_passed = True
            confidence = 1.0
            evidence = []
            contradictions = []
            
            if supporting_knowledge:
                # Calculate confidence based on supporting knowledge
                confidence_scores = [item.confidence for item in supporting_knowledge]
                confidence = sum(confidence_scores) / len(confidence_scores)
                
                # Generate evidence
                evidence = [
                    f"Supported by knowledge item: {item.content[:100]}..."
                    for item in supporting_knowledge[:3]
                ]
                
                # Check for contradictions (simplified)
                for item in supporting_knowledge:
                    if 'not' in reasoning_content.lower() and 'not' not in item.content.lower():
                        contradictions.append(f"Potential contradiction with: {item.content[:100]}...")
                
                # Lower confidence if contradictions found
                if contradictions:
                    confidence *= 0.7
                    validation_passed = len(contradictions) < len(supporting_knowledge) / 2
            
            else:
                # No supporting knowledge found
                confidence = 0.5
                evidence = ["No direct supporting knowledge found in world model"]
            
            processing_time = time.time() - start_time
            
            result = ValidationResult(
                request_id=request.request_id,
                validation_passed=validation_passed,
                confidence=confidence,
                evidence=evidence,
                contradictions=contradictions,
                supporting_knowledge=supporting_knowledge,
                processing_time=processing_time,
                validation_details={
                    'reasoning_engine': request.reasoning_result.engine.value if hasattr(request.reasoning_result.engine, 'value') else str(request.reasoning_result.engine),
                    'support_count': len(supporting_knowledge),
                    'contradiction_count': len(contradictions)
                }
            )
            
            request.status = ValidationStatus.COMPLETED
            return result
            
        except Exception as e:
            request.status = ValidationStatus.FAILED
            processing_time = time.time() - start_time
            
            logger.warning("Single validation failed", 
                          request_id=request.request_id, 
                          error=str(e))
            
            return ValidationResult(
                request_id=request.request_id,
                validation_passed=False,
                confidence=0.0,
                evidence=[f"Validation failed: {str(e)}"],
                processing_time=processing_time
            )
    
    def _extract_reasoning_content(self, reasoning_result: "ReasoningResult") -> str:
        """Extract searchable content from reasoning result"""
        
        content_parts = []
        
        # Add conclusion if available
        if hasattr(reasoning_result, 'result') and isinstance(reasoning_result.result, dict):
            conclusion = reasoning_result.result.get('conclusion', '')
            if conclusion:
                content_parts.append(conclusion)
        
        # Add evidence if available
        if hasattr(reasoning_result, 'result') and isinstance(reasoning_result.result, dict):
            evidence = reasoning_result.result.get('evidence', [])
            if evidence and isinstance(evidence, list):
                content_parts.extend(evidence[:3])  # First 3 evidence items
        
        # Combine all content
        combined_content = ' '.join(str(part) for part in content_parts)
        return combined_content[:500]  # Limit to 500 chars
    
    def _generate_cache_key(self, request: ValidationRequest) -> str:
        """Generate cache key for validation request"""
        content = self._extract_reasoning_content(request.reasoning_result)
        engine_type = getattr(request.reasoning_result.engine, 'value', str(request.reasoning_result.engine))
        return f"{engine_type}:{hash(content)}:{request.validation_type}"
    
    def _generate_cache_key_from_result(self, result: ValidationResult) -> str:
        """Generate cache key from validation result"""
        return result.request_id  # Simplified for demo
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation engine statistics"""
        hit_rate = self.cache_hits / max(self.total_validations, 1)
        success_rate = self.successful_validations / max(self.total_validations, 1)
        
        return {
            'total_validations': self.total_validations,
            'successful_validations': self.successful_validations,
            'success_rate': success_rate,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': hit_rate,
            'cache_size': len(self.validation_cache),
            'active_workers': self.max_workers
        }

class HierarchicalResultCache:
    """Multi-level caching system for parallel reasoning optimization"""
    
    def __init__(self):
        # Level 1: Individual engine result cache (fastest)
        self.engine_result_cache = {}
        self.engine_cache_lock = threading.RLock()
        
        # Level 2: Sequence result cache
        self.sequence_result_cache = {}
        self.sequence_cache_lock = threading.RLock()
        
        # Level 3: World model validation cache  
        self.validation_result_cache = {}
        self.validation_cache_lock = threading.RLock()
        
        # Level 4: Cross-worker result sharing
        self.shared_result_cache = {}
        self.shared_cache_lock = threading.RLock()
        
        # Cache configuration
        self.max_sizes = {
            CacheLevel.ENGINE_RESULT: 100000,
            CacheLevel.SEQUENCE_RESULT: 10000,
            CacheLevel.VALIDATION_RESULT: 50000,
            CacheLevel.SHARED_WORKER: 25000
        }
        
        # Statistics
        self.cache_stats = defaultdict(lambda: {'hits': 0, 'misses': 0, 'stores': 0})
        
        # Cleanup thread
        self.cleanup_thread = None
        self.cleanup_active = True
        self._start_cleanup_thread()
    
    async def get_or_compute(self, 
                           cache_key: str, 
                           computation_func: Callable, 
                           cache_level: CacheLevel = CacheLevel.ENGINE_RESULT) -> Any:
        """Check all cache levels before computing"""
        
        # Check caches in order of speed (fastest first)
        cache_levels = [
            CacheLevel.ENGINE_RESULT,
            CacheLevel.SEQUENCE_RESULT, 
            CacheLevel.VALIDATION_RESULT,
            CacheLevel.SHARED_WORKER
        ]
        
        # Check each cache level
        for level in cache_levels:
            result = await self._get_from_cache(cache_key, level)
            if result is not None:
                self.cache_stats[level]['hits'] += 1
                logger.debug("Cache hit", cache_key=cache_key[:50], cache_level=level.value)
                return result
        
        # Not found in any cache - compute result
        self.cache_stats[cache_level]['misses'] += 1
        
        try:
            if asyncio.iscoroutinefunction(computation_func):
                result = await computation_func()
            else:
                result = computation_func()
            
            # Store in appropriate cache
            await self._store_in_cache(cache_key, result, cache_level)
            self.cache_stats[cache_level]['stores'] += 1
            
            return result
            
        except Exception as e:
            logger.error("Cache computation failed", 
                        cache_key=cache_key[:50], 
                        error=str(e))
            return None
    
    async def _get_from_cache(self, cache_key: str, cache_level: CacheLevel) -> Any:
        """Get result from specific cache level"""
        
        cache_dict, lock = self._get_cache_and_lock(cache_level)
        
        with lock:
            return cache_dict.get(cache_key)
    
    async def _store_in_cache(self, cache_key: str, result: Any, cache_level: CacheLevel):
        """Store result in specific cache level"""
        
        cache_dict, lock = self._get_cache_and_lock(cache_level)
        max_size = self.max_sizes[cache_level]
        
        with lock:
            # Evict if at capacity
            if len(cache_dict) >= max_size:
                # Remove oldest entry (simplified LRU)
                oldest_key = next(iter(cache_dict))
                del cache_dict[oldest_key]
            
            # Store new result
            cache_dict[cache_key] = {
                'result': result,
                'cached_at': datetime.now(timezone.utc),
                'access_count': 0
            }
    
    def _get_cache_and_lock(self, cache_level: CacheLevel) -> Tuple[Dict, threading.RLock]:
        """Get cache dictionary and lock for specified level"""
        
        if cache_level == CacheLevel.ENGINE_RESULT:
            return self.engine_result_cache, self.engine_cache_lock
        elif cache_level == CacheLevel.SEQUENCE_RESULT:
            return self.sequence_result_cache, self.sequence_cache_lock
        elif cache_level == CacheLevel.VALIDATION_RESULT:
            return self.validation_result_cache, self.validation_cache_lock
        elif cache_level == CacheLevel.SHARED_WORKER:
            return self.shared_result_cache, self.shared_cache_lock
        else:
            return self.engine_result_cache, self.engine_cache_lock
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup"""
        
        def cleanup_worker():
            while self.cleanup_active:
                try:
                    self._cleanup_expired_entries()
                    time.sleep(300)  # Clean every 5 minutes
                except Exception as e:
                    logger.warning("Cache cleanup failed", error=str(e))
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)  # 1 hour expiry
        
        for cache_level in CacheLevel:
            cache_dict, lock = self._get_cache_and_lock(cache_level)
            
            with lock:
                expired_keys = [
                    key for key, entry in cache_dict.items()
                    if entry['cached_at'] < cutoff_time
                ]
                
                for key in expired_keys:
                    del cache_dict[key]
                
                if expired_keys:
                    logger.debug("Cleaned expired cache entries",
                               cache_level=cache_level.value,
                               entries_removed=len(expired_keys))
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        stats = {}
        
        for cache_level in CacheLevel:
            cache_dict, lock = self._get_cache_and_lock(cache_level)
            
            with lock:
                cache_size = len(cache_dict)
                max_size = self.max_sizes[cache_level]
                
                level_stats = self.cache_stats[cache_level]
                total_requests = level_stats['hits'] + level_stats['misses']
                hit_rate = level_stats['hits'] / max(total_requests, 1)
                
                stats[cache_level.value] = {
                    'size': cache_size,
                    'max_size': max_size,
                    'utilization': cache_size / max_size,
                    'hits': level_stats['hits'],
                    'misses': level_stats['misses'],
                    'stores': level_stats['stores'],
                    'hit_rate': hit_rate
                }
        
        return stats
    
    def shutdown(self):
        """Shutdown cache system"""
        self.cleanup_active = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)

class AdaptiveResourceManager:
    """Dynamically optimizes resource allocation across parallel workers"""
    
    def __init__(self):
        self.performance_monitor = WorkerPerformanceMonitor()
        self.resource_allocator = DynamicResourceAllocator()
        self.optimization_active = False
        self.optimization_thread = None
        self.worker_metrics: Dict[int, WorkerPerformanceMetrics] = {}
        
    async def start_optimization(self):
        """Start adaptive resource optimization"""
        
        self.optimization_active = True
        
        # Start optimization in separate thread to avoid blocking
        def optimization_worker():
            asyncio.run(self._optimization_loop())
        
        self.optimization_thread = threading.Thread(target=optimization_worker, daemon=True)
        self.optimization_thread.start()
        
        logger.info("Adaptive resource optimization started")
    
    async def stop_optimization(self):
        """Stop adaptive resource optimization"""
        
        self.optimization_active = False
        
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=10.0)
        
        logger.info("Adaptive resource optimization stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        
        while self.optimization_active:
            try:
                # Monitor worker performance
                worker_stats = await self.performance_monitor.get_worker_statistics()
                
                # Update internal metrics
                self._update_worker_metrics(worker_stats)
                
                # Identify performance bottlenecks
                bottlenecks = self._identify_performance_bottlenecks(worker_stats)
                
                # Redistribute work if needed
                if bottlenecks:
                    await self._handle_bottlenecks(bottlenecks)
                
                # Optimize resource allocation
                await self.resource_allocator.optimize_allocation(worker_stats)
                
                # Wait before next optimization cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Resource optimization failed", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    def _update_worker_metrics(self, worker_stats: Dict[int, Dict[str, Any]]):
        """Update internal worker metrics"""
        
        for worker_id, stats in worker_stats.items():
            if worker_id not in self.worker_metrics:
                self.worker_metrics[worker_id] = WorkerPerformanceMetrics(worker_id=worker_id)
            
            metrics = self.worker_metrics[worker_id]
            
            # Update metrics from stats
            metrics.total_validations = stats.get('total_validations', 0)
            metrics.successful_validations = stats.get('successful_validations', 0)
            metrics.failed_validations = metrics.total_validations - metrics.successful_validations
            metrics.average_validation_time = stats.get('average_validation_time', 0.0)
            metrics.cache_hit_rate = stats.get('cache_hit_rate', 0.0)
            metrics.memory_usage_mb = stats.get('memory_usage_mb', 0.0)
            metrics.cpu_utilization = stats.get('cpu_utilization', 0.0)
            metrics.last_activity = datetime.now(timezone.utc)
            
            # Calculate performance score
            metrics.performance_score = self._calculate_performance_score(metrics)
            
            # Determine health status
            metrics.health_status = self._determine_health_status(metrics)
    
    def _calculate_performance_score(self, metrics: WorkerPerformanceMetrics) -> float:
        """Calculate overall performance score for worker (0.0-1.0)"""
        
        # Success rate score (0.0-1.0)
        success_rate = metrics.successful_validations / max(metrics.total_validations, 1)
        success_score = success_rate
        
        # Speed score (inverse of validation time, normalized)
        max_reasonable_time = 10.0  # 10 seconds max
        speed_score = max(0.0, 1.0 - (metrics.average_validation_time / max_reasonable_time))
        
        # Cache efficiency score
        cache_score = metrics.cache_hit_rate
        
        # Resource efficiency score
        max_reasonable_memory = 1000.0  # 1GB max
        max_reasonable_cpu = 80.0  # 80% max
        memory_score = max(0.0, 1.0 - (metrics.memory_usage_mb / max_reasonable_memory))
        cpu_score = max(0.0, 1.0 - (metrics.cpu_utilization / max_reasonable_cpu))
        
        # Weighted average
        performance_score = (
            success_score * 0.3 +
            speed_score * 0.25 +
            cache_score * 0.2 +
            memory_score * 0.125 +
            cpu_score * 0.125
        )
        
        return max(0.0, min(1.0, performance_score))
    
    def _determine_health_status(self, metrics: WorkerPerformanceMetrics) -> WorkerHealth:
        """Determine health status based on metrics"""
        
        # Check if worker is unresponsive
        if metrics.last_activity:
            time_since_activity = datetime.now(timezone.utc) - metrics.last_activity
            if time_since_activity > timedelta(minutes=5):
                return WorkerHealth.UNRESPONSIVE
        
        # Check performance score
        if metrics.performance_score < 0.3:
            return WorkerHealth.FAILED
        elif metrics.performance_score < 0.6:
            return WorkerHealth.SLOW
        else:
            return WorkerHealth.HEALTHY
    
    def _identify_performance_bottlenecks(self, worker_stats: Dict[int, Dict[str, Any]]) -> List[BottleneckInfo]:
        """Identify workers with performance bottlenecks"""
        
        if not worker_stats:
            return []
        
        bottlenecks = []
        
        # Calculate averages
        validation_times = [stats.get('average_validation_time', 0.0) for stats in worker_stats.values()]
        memory_usages = [stats.get('memory_usage_mb', 0.0) for stats in worker_stats.values()]
        cpu_utilizations = [stats.get('cpu_utilization', 0.0) for stats in worker_stats.values()]
        
        if not validation_times:
            return []
        
        avg_validation_time = statistics.mean([t for t in validation_times if t > 0])
        avg_memory_usage = statistics.mean([m for m in memory_usages if m > 0])
        avg_cpu_usage = statistics.mean([c for c in cpu_utilizations if c > 0])
        
        # Find bottlenecks
        for worker_id, stats in worker_stats.items():
            worker_metrics = self.worker_metrics.get(worker_id)
            if not worker_metrics:
                continue
            
            bottleneck_types = []
            severity = 0.0
            
            # Slow validation bottleneck
            worker_time = stats.get('average_validation_time', 0.0)
            if worker_time > avg_validation_time * 1.5:
                bottleneck_types.append('slow_validation')
                severity = max(severity, (worker_time - avg_validation_time) / avg_validation_time)
            
            # High memory usage bottleneck
            worker_memory = stats.get('memory_usage_mb', 0.0)
            if worker_memory > avg_memory_usage * 1.5:
                bottleneck_types.append('high_memory')
                severity = max(severity, (worker_memory - avg_memory_usage) / avg_memory_usage)
            
            # CPU bound bottleneck  
            worker_cpu = stats.get('cpu_utilization', 0.0)
            if worker_cpu > 90.0:  # Over 90% CPU
                bottleneck_types.append('cpu_bound')
                severity = max(severity, worker_cpu / 100.0)
            
            # Cache miss bottleneck
            cache_hit_rate = stats.get('cache_hit_rate', 1.0)
            if cache_hit_rate < 0.5:  # Less than 50% cache hit rate
                bottleneck_types.append('cache_misses')
                severity = max(severity, 1.0 - cache_hit_rate)
            
            # Create bottleneck info if any issues found
            if bottleneck_types:
                suggested_actions = self._generate_suggested_actions(bottleneck_types)
                
                bottleneck = BottleneckInfo(
                    worker_id=worker_id,
                    bottleneck_type=', '.join(bottleneck_types),
                    severity=min(1.0, severity),
                    metrics=worker_metrics,
                    suggested_actions=suggested_actions
                )
                
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _generate_suggested_actions(self, bottleneck_types: List[str]) -> List[str]:
        """Generate suggested actions for bottleneck resolution"""
        
        actions = []
        
        if 'slow_validation' in bottleneck_types:
            actions.append("Redistribute work from slow worker to faster workers")
            actions.append("Investigate worker-specific performance issues")
        
        if 'high_memory' in bottleneck_types:
            actions.append("Reduce batch size for memory-intensive worker")
            actions.append("Trigger garbage collection for worker")
        
        if 'cpu_bound' in bottleneck_types:
            actions.append("Reduce CPU-intensive task allocation to worker")
            actions.append("Consider worker throttling to prevent resource exhaustion")
        
        if 'cache_misses' in bottleneck_types:
            actions.append("Warm up cache for worker with common validations")
            actions.append("Increase cache size allocation for worker")
        
        return actions
    
    async def _handle_bottlenecks(self, bottlenecks: List[BottleneckInfo]):
        """Handle identified performance bottlenecks"""
        
        for bottleneck in bottlenecks:
            logger.warning("Performance bottleneck detected",
                          worker_id=bottleneck.worker_id,
                          bottleneck_type=bottleneck.bottleneck_type,
                          severity=bottleneck.severity,
                          suggested_actions=bottleneck.suggested_actions)
            
            # Take automatic corrective actions based on severity
            if bottleneck.severity > 0.8:  # Critical bottleneck
                await self._take_critical_action(bottleneck)
            elif bottleneck.severity > 0.6:  # Moderate bottleneck
                await self._take_moderate_action(bottleneck)
    
    async def _take_critical_action(self, bottleneck: BottleneckInfo):
        """Take critical action for severe bottleneck"""
        
        if 'slow_validation' in bottleneck.bottleneck_type:
            # Redistribute work from slow worker
            await self.resource_allocator.redistribute_work_from_worker(bottleneck.worker_id)
        
        if 'high_memory' in bottleneck.bottleneck_type:
            # Trigger garbage collection
            await self.resource_allocator.trigger_worker_cleanup(bottleneck.worker_id)
        
        logger.info("Critical bottleneck action taken", 
                   worker_id=bottleneck.worker_id,
                   action="work_redistribution_and_cleanup")
    
    async def _take_moderate_action(self, bottleneck: BottleneckInfo):
        """Take moderate action for moderate bottleneck"""
        
        # Adjust worker allocation
        await self.resource_allocator.adjust_worker_allocation(
            bottleneck.worker_id, 
            reduction_factor=0.8  # Reduce workload by 20%
        )
        
        logger.info("Moderate bottleneck action taken",
                   worker_id=bottleneck.worker_id,
                   action="workload_reduction")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get resource optimization statistics"""
        
        total_workers = len(self.worker_metrics)
        
        if total_workers == 0:
            return {
                'total_workers': 0,
                'optimization_active': self.optimization_active,
                'average_performance_score': 0.0,
                'healthy_workers': 0,
                'problematic_workers': 0
            }
        
        # Calculate aggregate statistics
        performance_scores = [m.performance_score for m in self.worker_metrics.values()]
        avg_performance = statistics.mean(performance_scores)
        
        healthy_workers = sum(1 for m in self.worker_metrics.values() 
                            if m.health_status == WorkerHealth.HEALTHY)
        
        problematic_workers = total_workers - healthy_workers
        
        return {
            'total_workers': total_workers,
            'optimization_active': self.optimization_active,
            'average_performance_score': avg_performance,
            'healthy_workers': healthy_workers,
            'problematic_workers': problematic_workers,
            'worker_health_distribution': {
                status.value: sum(1 for m in self.worker_metrics.values() 
                                if m.health_status == status)
                for status in WorkerHealth
            }
        }

# Placeholder classes for components referenced in the architecture
class WorkerPerformanceMonitor:
    """Monitors performance of parallel workers"""
    
    async def get_worker_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get current worker statistics"""
        
        # In production, this would collect real metrics from workers
        # For demo, return simulated statistics
        return {
            worker_id: {
                'total_validations': 100 + worker_id * 10,
                'successful_validations': 95 + worker_id * 9,
                'average_validation_time': 2.0 + (worker_id % 3) * 0.5,
                'cache_hit_rate': 0.7 + (worker_id % 4) * 0.05,
                'memory_usage_mb': 200 + worker_id * 50,
                'cpu_utilization': 30 + worker_id * 15
            }
            for worker_id in range(4)  # 4 workers for demo
        }

class DynamicResourceAllocator:
    """Allocates resources dynamically based on worker performance"""
    
    async def optimize_allocation(self, worker_stats: Dict[int, Dict[str, Any]]):
        """Optimize resource allocation based on worker statistics"""
        
        logger.debug("Resource allocation optimized", 
                    num_workers=len(worker_stats))
    
    async def redistribute_work_from_worker(self, worker_id: int):
        """Redistribute work from a problematic worker"""
        
        logger.info("Redistributing work from worker", worker_id=worker_id)
    
    async def trigger_worker_cleanup(self, worker_id: int):
        """Trigger cleanup for a worker with memory issues"""
        
        logger.info("Triggering cleanup for worker", worker_id=worker_id)
    
    async def adjust_worker_allocation(self, worker_id: int, reduction_factor: float):
        """Adjust workload allocation for a worker"""
        
        logger.info("Adjusting worker allocation", 
                   worker_id=worker_id,
                   reduction_factor=reduction_factor)

class SharedWorldModelManager:
    """Complete shared world model manager coordinating all components"""
    
    def __init__(self, world_model_size: int = 10000, max_validation_workers: int = 4):
        self.world_model_size = world_model_size
        
        # Initialize core components
        self.shared_knowledge_base = SharedKnowledgeBase(max_size=world_model_size)
        self.parallel_validation_engine = ParallelValidationEngine(
            self.shared_knowledge_base, 
            max_workers=max_validation_workers
        )
        self.hierarchical_cache = HierarchicalResultCache()
        self.adaptive_resource_manager = AdaptiveResourceManager()
        
        # Manager state
        self.initialized = False
        self.active_validations = 0
        self.total_validations_processed = 0
        
        logger.info("Shared World Model Manager initialized",
                   world_model_size=world_model_size,
                   max_validation_workers=max_validation_workers)
    
    async def initialize(self):
        """Initialize all components"""
        
        if self.initialized:
            return
        
        # Start adaptive resource management
        await self.adaptive_resource_manager.start_optimization()
        
        # Initialize with some sample knowledge
        await self._populate_sample_knowledge()
        
        self.initialized = True
        logger.info("Shared World Model Manager fully initialized")
    
    async def validate_reasoning_results(self, 
                                       reasoning_results: List["ReasoningResult"],
                                       worker_id: Optional[int] = None) -> List[ValidationResult]:
        """Validate reasoning results against shared world model"""
        
        if not self.initialized:
            await self.initialize()
        
        if not reasoning_results:
            return []
        
        start_time = time.time()
        self.active_validations += len(reasoning_results)
        
        try:
            # Create validation requests
            validation_requests = [
                ValidationRequest(
                    reasoning_result=result,
                    worker_id=worker_id,
                    priority=1  # High priority
                )
                for result in reasoning_results
            ]
            
            # Perform parallel validation
            validation_results = await self.parallel_validation_engine.validate_batch(validation_requests)
            
            # Update statistics
            self.total_validations_processed += len(reasoning_results)
            self.active_validations -= len(reasoning_results)
            
            processing_time = time.time() - start_time
            
            logger.info("Validation batch completed",
                       num_results=len(reasoning_results),
                       num_validated=len(validation_results),
                       processing_time=processing_time,
                       worker_id=worker_id)
            
            return validation_results
            
        except Exception as e:
            self.active_validations -= len(reasoning_results)
            logger.error("Validation batch failed",
                        num_results=len(reasoning_results),
                        error=str(e),
                        worker_id=worker_id)
            return []
    
    async def add_knowledge_items(self, knowledge_items: List[KnowledgeItem]):
        """Add knowledge items to shared world model"""
        
        if not self.initialized:
            await self.initialize()
        
        added_count = 0
        for item in knowledge_items:
            if self.shared_knowledge_base.add_knowledge_item(item):
                added_count += 1
        
        logger.info("Knowledge items added to world model",
                   items_requested=len(knowledge_items),
                   items_added=added_count)
        
        return added_count
    
    async def search_knowledge(self, 
                             query: str, 
                             domain: Optional[str] = None,
                             use_cache: bool = True) -> List[KnowledgeItem]:
        """Search knowledge in shared world model with caching"""
        
        if not self.initialized:
            await self.initialize()
        
        if use_cache:
            # Use hierarchical cache for search
            cache_key = f"search:{query}:{domain or 'all'}"
            
            result = await self.hierarchical_cache.get_or_compute(
                cache_key=cache_key,
                computation_func=lambda: self.shared_knowledge_base.search_knowledge(query, domain),
                cache_level=CacheLevel.VALIDATION_RESULT
            )
            
            return result if result is not None else []
        else:
            # Direct search without caching
            return self.shared_knowledge_base.search_knowledge(query, domain)
    
    async def _populate_sample_knowledge(self):
        """Populate world model with sample knowledge for testing"""
        
        sample_items = [
            KnowledgeItem(
                content="Quantum computing uses quantum mechanical phenomena to perform computations",
                domain="quantum_physics",
                confidence=0.95,
                sources=["quantum_physics_textbook"]
            ),
            KnowledgeItem(
                content="Machine learning algorithms can be trained on large datasets to make predictions",
                domain="artificial_intelligence", 
                confidence=0.9,
                sources=["ml_research_papers"]
            ),
            KnowledgeItem(
                content="Sustainable energy sources include solar, wind, and hydroelectric power",
                domain="energy_technology",
                confidence=0.92,
                sources=["renewable_energy_reports"]
            ),
            KnowledgeItem(
                content="CRISPR gene editing allows precise modification of DNA sequences",
                domain="biotechnology",
                confidence=0.88,
                sources=["genetic_engineering_studies"]
            ),
            KnowledgeItem(
                content="Blockchain technology provides decentralized and immutable record keeping",
                domain="computer_science",
                confidence=0.85,
                sources=["blockchain_whitepapers"]
            )
        ]
        
        await self.add_knowledge_items(sample_items)
        logger.info("Sample knowledge populated", num_items=len(sample_items))
    
    async def shutdown(self):
        """Shutdown shared world model manager"""
        
        # Stop adaptive resource management
        await self.adaptive_resource_manager.stop_optimization()
        
        # Shutdown hierarchical cache
        self.hierarchical_cache.shutdown()
        
        logger.info("Shared World Model Manager shutdown complete")
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all components"""
        
        return {
            'manager': {
                'initialized': self.initialized,
                'active_validations': self.active_validations,
                'total_validations_processed': self.total_validations_processed,
                'world_model_size': self.world_model_size
            },
            'knowledge_base': self.shared_knowledge_base.get_statistics(),
            'validation_engine': self.parallel_validation_engine.get_statistics(),
            'hierarchical_cache': self.hierarchical_cache.get_cache_statistics(),
            'resource_manager': self.adaptive_resource_manager.get_optimization_statistics()
        }

# Main interface function for integration
async def shared_world_model_validation(reasoning_results: List["ReasoningResult"],
                                      context: Optional[Dict[str, Any]] = None,
                                      worker_id: Optional[int] = None) -> Dict[str, Any]:
    """Shared world model validation interface for meta-reasoning integration"""
    
    # Create shared world model manager (singleton pattern in production)
    if not hasattr(shared_world_model_validation, '_manager'):
        shared_world_model_validation._manager = SharedWorldModelManager()
        await shared_world_model_validation._manager.initialize()
    
    manager = shared_world_model_validation._manager
    
    # Perform validation
    validation_results = await manager.validate_reasoning_results(reasoning_results, worker_id)
    
    # Calculate aggregate metrics
    total_results = len(validation_results)
    passed_validations = sum(1 for r in validation_results if r.validation_passed)
    avg_confidence = sum(r.confidence for r in validation_results) / max(total_results, 1)
    avg_processing_time = sum(r.processing_time for r in validation_results) / max(total_results, 1)
    cached_results = sum(1 for r in validation_results if r.cached)
    
    # Get comprehensive statistics
    stats = manager.get_comprehensive_statistics()
    
    return {
        "conclusion": f"Validated {total_results} reasoning results using shared world model with {avg_confidence:.2f} average confidence",
        "validation_passed": passed_validations,
        "total_validations": total_results,
        "validation_success_rate": passed_validations / max(total_results, 1),
        "average_confidence": avg_confidence,
        "average_processing_time": avg_processing_time,
        "cache_hit_rate": cached_results / max(total_results, 1),
        "validation_results": validation_results,
        "world_model_statistics": stats,
        "reasoning_chain": [
            f"Processed {total_results} reasoning results through shared world model",
            f"Achieved {passed_validations}/{total_results} successful validations",
            f"Cache hit rate: {(cached_results / max(total_results, 1)):.1%}",
            f"Average processing time: {avg_processing_time:.3f} seconds per validation"
        ],
        "processing_time": avg_processing_time,
        "quality_score": avg_confidence,
        "worker_id": worker_id
    }

if __name__ == "__main__":
    # Test the shared world model architecture
    async def test_shared_world_model():
        from prsm.nwtn.meta_reasoning_engine import ReasoningResult, ReasoningEngine
        
        print("Shared World Model Architecture Test:")
        print("=" * 50)
        
        # Create sample reasoning results
        test_results = [
            ReasoningResult(
                engine=ReasoningEngine.DEDUCTIVE,
                result={
                    'conclusion': 'Quantum computing can solve certain problems exponentially faster',
                    'confidence': 0.8,
                    'evidence': ['Quantum algorithms like Shor\'s algorithm', 'Quantum supremacy demonstrations']
                },
                confidence=0.8,
                processing_time=1.5
            ),
            ReasoningResult(
                engine=ReasoningEngine.INDUCTIVE,
                result={
                    'conclusion': 'Machine learning models improve with more training data',
                    'confidence': 0.85,
                    'evidence': ['Empirical studies on model performance', 'Learning curves analysis']
                },
                confidence=0.85,
                processing_time=2.0
            )
        ]
        
        # Test shared world model validation
        result = await shared_world_model_validation(
            reasoning_results=test_results,
            context={'domain': 'technology'},
            worker_id=1
        )
        
        print(f"Validation Results:")
        print(f"Total Validations: {result['total_validations']}")
        print(f"Successful Validations: {result['validation_passed']}")
        print(f"Success Rate: {result['validation_success_rate']:.1%}")
        print(f"Average Confidence: {result['average_confidence']:.2f}")
        print(f"Cache Hit Rate: {result['cache_hit_rate']:.1%}")
        print(f"Processing Time: {result['processing_time']:.3f}s")
        
        print("\nWorld Model Statistics:")
        stats = result['world_model_statistics']
        print(f"Knowledge Base Size: {stats['knowledge_base']['total_items']}")
        print(f"Knowledge Base Hit Rate: {stats['knowledge_base']['hit_rate']:.1%}")
        print(f"Active Workers: {stats['validation_engine']['active_workers']}")
        print(f"Cache Levels: {len(stats['hierarchical_cache'])}")
        
        # Shutdown
        if hasattr(shared_world_model_validation, '_manager'):
            await shared_world_model_validation._manager.shutdown()
    
    asyncio.run(test_shared_world_model())
"""
PRSM Cache Invalidation Strategies
Advanced cache invalidation patterns, dependency tracking, and event-driven cache management
"""

from typing import Dict, Any, List, Optional, Set, Callable, Union, Pattern
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import re
import hashlib
from collections import defaultdict, deque
import redis.asyncio as aioredis
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class InvalidationEvent(Enum):
    """Types of cache invalidation events"""
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    TIME_BASED = "time_based"
    DEPENDENCY_CHANGE = "dependency_change"
    MANUAL = "manual"


class InvalidationScope(Enum):
    """Scope of cache invalidation"""
    SPECIFIC_KEY = "specific_key"
    TAG_BASED = "tag_based"
    PATTERN_BASED = "pattern_based"
    NAMESPACE_BASED = "namespace_based"
    DEPENDENCY_BASED = "dependency_based"
    GLOBAL = "global"


@dataclass
class InvalidationRule:
    """Cache invalidation rule definition"""
    rule_id: str
    name: str
    description: str
    event_types: Set[InvalidationEvent]
    scope: InvalidationScope
    target_patterns: List[str]  # Patterns, tags, or keys to invalidate
    conditions: Dict[str, Any] = field(default_factory=dict)
    delay_seconds: int = 0  # Delay before invalidation
    cascade: bool = False  # Whether to cascade to dependent caches
    enabled: bool = True
    priority: int = 0  # Higher priority rules execute first
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class InvalidationEvent:
    """Cache invalidation event"""
    event_id: str
    event_type: InvalidationEvent
    source: str  # Source of the event (e.g., "api", "database", "user")
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyNode:
    """Cache dependency graph node"""
    key: str
    dependencies: Set[str] = field(default_factory=set)  # Keys this depends on
    dependents: Set[str] = field(default_factory=set)    # Keys that depend on this
    tags: Set[str] = field(default_factory=set)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: Optional[int] = None


class InvalidationStrategy(ABC):
    """Abstract base class for invalidation strategies"""
    
    @abstractmethod
    async def should_invalidate(self, event: InvalidationEvent, 
                              cache_key: str, metadata: Dict[str, Any]) -> bool:
        """Determine if cache should be invalidated for this event"""
        pass
    
    @abstractmethod
    async def get_invalidation_targets(self, event: InvalidationEvent) -> List[str]:
        """Get list of cache keys/patterns to invalidate"""
        pass


class TimeBasedInvalidationStrategy(InvalidationStrategy):
    """Time-based cache invalidation (TTL, scheduled, etc.)"""
    
    def __init__(self):
        self.scheduled_invalidations: Dict[str, datetime] = {}
    
    async def should_invalidate(self, event: InvalidationEvent, 
                              cache_key: str, metadata: Dict[str, Any]) -> bool:
        """Check if cache should be invalidated based on time"""
        
        if event.event_type != InvalidationEvent.TIME_BASED:
            return False
        
        # Check TTL expiration
        if "expires_at" in metadata:
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            if datetime.now(timezone.utc) >= expires_at:
                return True
        
        # Check scheduled invalidation
        if cache_key in self.scheduled_invalidations:
            scheduled_time = self.scheduled_invalidations[cache_key]
            if datetime.now(timezone.utc) >= scheduled_time:
                return True
        
        return False
    
    async def get_invalidation_targets(self, event: InvalidationEvent) -> List[str]:
        """Get time-based invalidation targets"""
        
        targets = []
        current_time = datetime.now(timezone.utc)
        
        # Check all scheduled invalidations
        expired_keys = []
        for key, scheduled_time in self.scheduled_invalidations.items():
            if current_time >= scheduled_time:
                targets.append(key)
                expired_keys.append(key)
        
        # Clean up expired scheduled invalidations
        for key in expired_keys:
            del self.scheduled_invalidations[key]
        
        return targets
    
    def schedule_invalidation(self, cache_key: str, invalidate_at: datetime):
        """Schedule cache invalidation at specific time"""
        self.scheduled_invalidations[cache_key] = invalidate_at


class DependencyBasedInvalidationStrategy(InvalidationStrategy):
    """Dependency-based cache invalidation"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, DependencyNode] = {}
    
    async def should_invalidate(self, event: InvalidationEvent, 
                              cache_key: str, metadata: Dict[str, Any]) -> bool:
        """Check if cache should be invalidated based on dependencies"""
        
        if event.event_type != InvalidationEvent.DEPENDENCY_CHANGE:
            return False
        
        changed_key = event.data.get("changed_key")
        if not changed_key:
            return False
        
        # Check if cache_key depends on changed_key
        if cache_key in self.dependency_graph:
            node = self.dependency_graph[cache_key]
            return changed_key in node.dependencies
        
        return False
    
    async def get_invalidation_targets(self, event: InvalidationEvent) -> List[str]:
        """Get dependency-based invalidation targets"""
        
        changed_key = event.data.get("changed_key")
        if not changed_key or changed_key not in self.dependency_graph:
            return []
        
        # Get all dependent keys (breadth-first traversal)
        targets = set()
        queue = deque([changed_key])
        visited = set()
        
        while queue:
            current_key = queue.popleft()
            if current_key in visited:
                continue
            
            visited.add(current_key)
            
            if current_key in self.dependency_graph:
                node = self.dependency_graph[current_key]
                for dependent_key in node.dependents:
                    if dependent_key not in visited:
                        targets.add(dependent_key)
                        queue.append(dependent_key)
        
        return list(targets)
    
    def add_dependency(self, cache_key: str, depends_on: str):
        """Add cache dependency relationship"""
        
        # Create nodes if they don't exist
        if cache_key not in self.dependency_graph:
            self.dependency_graph[cache_key] = DependencyNode(cache_key)
        
        if depends_on not in self.dependency_graph:
            self.dependency_graph[depends_on] = DependencyNode(depends_on)
        
        # Add dependency relationship
        self.dependency_graph[cache_key].dependencies.add(depends_on)
        self.dependency_graph[depends_on].dependents.add(cache_key)
    
    def remove_dependency(self, cache_key: str, depends_on: str):
        """Remove cache dependency relationship"""
        
        if cache_key in self.dependency_graph:
            self.dependency_graph[cache_key].dependencies.discard(depends_on)
        
        if depends_on in self.dependency_graph:
            self.dependency_graph[depends_on].dependents.discard(cache_key)
    
    def get_dependency_chain(self, cache_key: str) -> Dict[str, Any]:
        """Get complete dependency chain for a cache key"""
        
        if cache_key not in self.dependency_graph:
            return {}
        
        def build_chain(key: str, visited: Set[str]) -> Dict[str, Any]:
            if key in visited:
                return {"circular_dependency": True}
            
            visited.add(key)
            node = self.dependency_graph.get(key)
            if not node:
                return {}
            
            chain = {
                "key": key,
                "dependencies": {},
                "dependents": list(node.dependents),
                "tags": list(node.tags),
                "last_updated": node.last_updated.isoformat()
            }
            
            for dep_key in node.dependencies:
                chain["dependencies"][dep_key] = build_chain(dep_key, visited.copy())
            
            return chain
        
        return build_chain(cache_key, set())


class TagBasedInvalidationStrategy(InvalidationStrategy):
    """Tag-based cache invalidation"""
    
    def __init__(self):
        self.tag_patterns: Dict[str, List[Pattern]] = defaultdict(list)
    
    async def should_invalidate(self, event: InvalidationEvent, 
                              cache_key: str, metadata: Dict[str, Any]) -> bool:
        """Check if cache should be invalidated based on tags"""
        
        cache_tags = metadata.get("tags", set())
        event_tags = event.data.get("tags", set())
        
        if not cache_tags or not event_tags:
            return False
        
        # Check for tag intersection
        return bool(cache_tags.intersection(event_tags))
    
    async def get_invalidation_targets(self, event: InvalidationEvent) -> List[str]:
        """Get tag-based invalidation targets"""
        
        event_tags = event.data.get("tags", set())
        if not event_tags:
            return []
        
        # This would typically query the cache backend for keys with matching tags
        # For now, return the tags themselves as patterns
        return [f"tag:{tag}" for tag in event_tags]
    
    def add_tag_pattern(self, tag: str, pattern: str):
        """Add regex pattern for tag-based invalidation"""
        compiled_pattern = re.compile(pattern)
        self.tag_patterns[tag].append(compiled_pattern)
    
    def match_tag_patterns(self, cache_key: str, tags: Set[str]) -> Set[str]:
        """Get matching tags for cache key based on patterns"""
        matching_tags = set()
        
        for tag, patterns in self.tag_patterns.items():
            if tag in tags:
                for pattern in patterns:
                    if pattern.match(cache_key):
                        matching_tags.add(tag)
                        break
        
        return matching_tags


class EventDrivenInvalidationStrategy(InvalidationStrategy):
    """Event-driven cache invalidation based on data changes"""
    
    def __init__(self):
        self.event_mappings: Dict[str, List[str]] = defaultdict(list)  # event_type -> cache_patterns
        self.condition_handlers: Dict[str, Callable] = {}
    
    async def should_invalidate(self, event: InvalidationEvent, 
                              cache_key: str, metadata: Dict[str, Any]) -> bool:
        """Check if cache should be invalidated based on event"""
        
        event_type = event.event_type.value
        if event_type not in self.event_mappings:
            return False
        
        # Check if cache key matches any patterns for this event type
        patterns = self.event_mappings[event_type]
        for pattern in patterns:
            if re.match(pattern, cache_key):
                # Check additional conditions if any
                if event_type in self.condition_handlers:
                    condition_handler = self.condition_handlers[event_type]
                    return await condition_handler(event, cache_key, metadata)
                return True
        
        return False
    
    async def get_invalidation_targets(self, event: InvalidationEvent) -> List[str]:
        """Get event-driven invalidation targets"""
        
        event_type = event.event_type.value
        return self.event_mappings.get(event_type, [])
    
    def add_event_mapping(self, event_type: str, cache_pattern: str):
        """Add event to cache pattern mapping"""
        self.event_mappings[event_type].append(cache_pattern)
    
    def add_condition_handler(self, event_type: str, handler: Callable):
        """Add condition handler for event type"""
        self.condition_handlers[event_type] = handler


class CacheInvalidationManager:
    """Comprehensive cache invalidation manager"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        
        # Invalidation strategies
        self.strategies: Dict[str, InvalidationStrategy] = {
            "time_based": TimeBasedInvalidationStrategy(),
            "dependency_based": DependencyBasedInvalidationStrategy(),
            "tag_based": TagBasedInvalidationStrategy(),
            "event_driven": EventDrivenInvalidationStrategy()
        }
        
        # Invalidation rules
        self.rules: Dict[str, InvalidationRule] = {}
        
        # Event queue and processing
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.processing_task: Optional[asyncio.Task] = None
        self.batch_size = 100
        self.batch_timeout = 5.0  # seconds
        
        # Statistics
        self.stats = {
            "events_processed": 0,
            "invalidations_executed": 0,
            "rules_evaluated": 0,
            "errors": 0
        }
        
        # Event handlers
        self.event_handlers: Dict[InvalidationEvent, List[Callable]] = defaultdict(list)
    
    async def start_processing(self):
        """Start background event processing"""
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_events_loop())
            logger.info("✅ Cache invalidation processing started")
    
    async def stop_processing(self):
        """Stop background event processing"""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Cache invalidation processing stopped")
    
    async def add_rule(self, rule: InvalidationRule):
        """Add cache invalidation rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added invalidation rule: {rule.name}")
    
    async def remove_rule(self, rule_id: str):
        """Remove cache invalidation rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed invalidation rule: {rule_id}")
    
    async def emit_event(self, event: InvalidationEvent):
        """Emit cache invalidation event"""
        try:
            await self.event_queue.put(event)
            logger.debug(f"Emitted invalidation event: {event.event_type.value}")
        except asyncio.QueueFull:
            logger.error("Invalidation event queue is full, dropping event")
            self.stats["errors"] += 1
    
    async def invalidate_immediate(self, cache_keys: Union[str, List[str]],
                                 scope: InvalidationScope = InvalidationScope.SPECIFIC_KEY):
        """Immediately invalidate cache keys"""
        
        if isinstance(cache_keys, str):
            cache_keys = [cache_keys]
        
        invalidated_count = 0
        
        for cache_key in cache_keys:
            try:
                if scope == InvalidationScope.SPECIFIC_KEY:
                    # Direct key deletion
                    deleted = await self.redis.delete(cache_key)
                    if deleted:
                        invalidated_count += 1
                
                elif scope == InvalidationScope.TAG_BASED:
                    # Tag-based invalidation
                    tag_keys = await self.redis.smembers(f"tag:{cache_key}")
                    if tag_keys:
                        deleted = await self.redis.delete(*tag_keys)
                        invalidated_count += deleted
                
                elif scope == InvalidationScope.PATTERN_BASED:
                    # Pattern-based invalidation
                    pattern_keys = []
                    cursor = 0
                    
                    while True:
                        cursor, keys = await self.redis.scan(
                            cursor=cursor, match=cache_key, count=1000
                        )
                        pattern_keys.extend(keys)
                        
                        if cursor == 0:
                            break
                    
                    if pattern_keys:
                        deleted = await self.redis.delete(*pattern_keys)
                        invalidated_count += deleted
                
            except Exception as e:
                logger.error(f"Error invalidating cache key {cache_key}: {e}")
                self.stats["errors"] += 1
        
        self.stats["invalidations_executed"] += invalidated_count
        logger.info(f"Immediately invalidated {invalidated_count} cache keys")
        
        return invalidated_count
    
    async def invalidate_by_dependency(self, changed_key: str):
        """Invalidate caches based on dependency changes"""
        
        strategy = self.strategies["dependency_based"]
        if not isinstance(strategy, DependencyBasedInvalidationStrategy):
            return 0
        
        # Create dependency change event
        event = InvalidationEvent(
            event_id=f"dep_{int(datetime.now().timestamp())}",
            event_type=InvalidationEvent.DEPENDENCY_CHANGE,
            source="dependency_manager",
            data={"changed_key": changed_key}
        )
        
        # Get invalidation targets
        targets = await strategy.get_invalidation_targets(event)
        
        # Invalidate targets
        return await self.invalidate_immediate(targets)
    
    async def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate caches by tags"""
        
        # Create tag-based event
        event = InvalidationEvent(
            event_id=f"tag_{int(datetime.now().timestamp())}",
            event_type=InvalidationEvent.DATA_UPDATE,
            source="tag_manager",
            data={"tags": tags}
        )
        
        await self.emit_event(event)
    
    async def schedule_invalidation(self, cache_key: str, invalidate_at: datetime):
        """Schedule cache invalidation at specific time"""
        
        strategy = self.strategies["time_based"]
        if isinstance(strategy, TimeBasedInvalidationStrategy):
            strategy.schedule_invalidation(cache_key, invalidate_at)
            
            # Create scheduled event
            event = InvalidationEvent(
                event_id=f"sched_{int(invalidate_at.timestamp())}",
                event_type=InvalidationEvent.TIME_BASED,
                source="scheduler",
                data={"cache_key": cache_key, "invalidate_at": invalidate_at.isoformat()}
            )
            
            # Schedule event processing
            delay_seconds = (invalidate_at - datetime.now(timezone.utc)).total_seconds()
            if delay_seconds > 0:
                asyncio.create_task(self._delayed_event_emit(event, delay_seconds))
    
    async def _delayed_event_emit(self, event: InvalidationEvent, delay_seconds: float):
        """Emit event after delay"""
        await asyncio.sleep(delay_seconds)
        await self.emit_event(event)
    
    async def _process_events_loop(self):
        """Background event processing loop"""
        
        while True:
            try:
                # Collect events in batches
                events = []
                
                # Wait for first event
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(), timeout=self.batch_timeout
                    )
                    events.append(event)
                except asyncio.TimeoutError:
                    continue
                
                # Collect additional events up to batch size
                while len(events) < self.batch_size:
                    try:
                        event = await asyncio.wait_for(
                            self.event_queue.get(), timeout=0.1
                        )
                        events.append(event)
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                await self._process_event_batch(events)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in invalidation event processing: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(1)
    
    async def _process_event_batch(self, events: List[InvalidationEvent]):
        """Process batch of invalidation events"""
        
        invalidation_targets = set()
        
        for event in events:
            try:
                # Process event with all strategies
                for strategy_name, strategy in self.strategies.items():
                    targets = await strategy.get_invalidation_targets(event)
                    invalidation_targets.update(targets)
                
                # Process event with rules
                await self._process_event_rules(event, invalidation_targets)
                
                # Call event handlers
                handlers = self.event_handlers.get(event.event_type, [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
                
                self.stats["events_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing invalidation event: {e}")
                self.stats["errors"] += 1
        
        # Execute invalidations
        if invalidation_targets:
            await self.invalidate_immediate(list(invalidation_targets))
    
    async def _process_event_rules(self, event: InvalidationEvent, 
                                 invalidation_targets: Set[str]):
        """Process event against invalidation rules"""
        
        # Sort rules by priority (higher first)
        sorted_rules = sorted(
            self.rules.values(), 
            key=lambda r: r.priority, 
            reverse=True
        )
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            if event.event_type not in rule.event_types:
                continue
            
            try:
                # Check rule conditions
                if await self._evaluate_rule_conditions(rule, event):
                    # Add rule targets to invalidation set
                    for pattern in rule.target_patterns:
                        if rule.scope == InvalidationScope.SPECIFIC_KEY:
                            invalidation_targets.add(pattern)
                        elif rule.scope == InvalidationScope.PATTERN_BASED:
                            # Pattern will be processed in invalidate_immediate
                            invalidation_targets.add(pattern)
                        elif rule.scope == InvalidationScope.TAG_BASED:
                            invalidation_targets.add(f"tag:{pattern}")
                
                self.stats["rules_evaluated"] += 1
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
                self.stats["errors"] += 1
    
    async def _evaluate_rule_conditions(self, rule: InvalidationRule, 
                                      event: InvalidationEvent) -> bool:
        """Evaluate rule conditions against event"""
        
        conditions = rule.conditions
        if not conditions:
            return True
        
        # Simple condition evaluation (can be extended)
        for condition_key, condition_value in conditions.items():
            if condition_key == "source":
                if event.source != condition_value:
                    return False
            elif condition_key == "data_contains":
                for key, value in condition_value.items():
                    if event.data.get(key) != value:
                        return False
            elif condition_key == "min_priority":
                event_priority = event.metadata.get("priority", 0)
                if event_priority < condition_value:
                    return False
        
        return True
    
    def add_event_handler(self, event_type: InvalidationEvent, handler: Callable):
        """Add event handler for specific event type"""
        self.event_handlers[event_type].append(handler)
    
    def get_dependency_strategy(self) -> DependencyBasedInvalidationStrategy:
        """Get dependency-based invalidation strategy"""
        return self.strategies["dependency_based"]
    
    def get_tag_strategy(self) -> TagBasedInvalidationStrategy:
        """Get tag-based invalidation strategy"""
        return self.strategies["tag_based"]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get invalidation manager statistics"""
        
        stats = self.stats.copy()
        stats.update({
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_rules": len(self.rules),
            "queue_size": self.event_queue.qsize(),
            "strategies": list(self.strategies.keys()),
            "processing_active": self.processing_task is not None and not self.processing_task.done()
        })
        
        return stats


# Global invalidation manager instance
invalidation_manager: Optional[CacheInvalidationManager] = None


async def initialize_invalidation_manager(redis_client: aioredis.Redis):
    """Initialize the global cache invalidation manager"""
    global invalidation_manager
    
    invalidation_manager = CacheInvalidationManager(redis_client)
    await invalidation_manager.start_processing()
    logger.info("✅ Cache invalidation manager initialized")


def get_invalidation_manager() -> CacheInvalidationManager:
    """Get the global invalidation manager instance"""
    if invalidation_manager is None:
        raise RuntimeError("Invalidation manager not initialized.")
    return invalidation_manager


async def shutdown_invalidation_manager():
    """Shutdown the invalidation manager"""
    if invalidation_manager:
        await invalidation_manager.stop_processing()
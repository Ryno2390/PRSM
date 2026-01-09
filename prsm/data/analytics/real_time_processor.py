#!/usr/bin/env python3
"""
Real-Time Analytics Processor
=============================

High-performance real-time stream processing for analytics data,
supporting live dashboards and immediate insights.
"""

import asyncio
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator, Union
import json
import threading

from prsm.compute.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events"""
    METRIC_UPDATE = "metric_update"
    SYSTEM_ALERT = "system_alert"
    USER_ACTION = "user_action"
    PERFORMANCE_EVENT = "performance_event"
    CUSTOM_EVENT = "custom_event"


class ProcessingMode(Enum):
    """Stream processing modes"""
    BATCH = "batch"              # Process in batches
    STREAMING = "streaming"      # Process individual events
    MICRO_BATCH = "micro_batch"  # Small batch processing
    WINDOWED = "windowed"        # Time-windowed processing


@dataclass
class StreamEvent:
    """Individual event in the stream"""
    event_id: str
    event_type: StreamEventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "unknown"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        """Create event from dictionary"""
        return cls(
            event_id=data["event_id"],
            event_type=StreamEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            source=data.get("source", "unknown"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class WindowConfiguration:
    """Configuration for windowed processing"""
    window_size: timedelta
    slide_interval: timedelta
    window_type: str = "tumbling"  # tumbling, sliding, session
    max_events_per_window: int = 10000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size_seconds": self.window_size.total_seconds(),
            "slide_interval_seconds": self.slide_interval.total_seconds(),
            "window_type": self.window_type,
            "max_events_per_window": self.max_events_per_window
        }


class StreamBuffer:
    """High-performance circular buffer for streaming events"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        
        # Statistics
        self.total_events = 0
        self.dropped_events = 0
        self.events_by_type = defaultdict(int)
        self.last_event_time = None
    
    def add_event(self, event: StreamEvent):
        """Add event to buffer"""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.dropped_events += 1
            
            self.buffer.append(event)
            self.total_events += 1
            self.events_by_type[event.event_type.value] += 1
            self.last_event_time = event.timestamp
    
    def get_events(self, count: Optional[int] = None,
                   since: Optional[datetime] = None,
                   event_types: Optional[List[StreamEventType]] = None) -> List[StreamEvent]:
        """Get events from buffer with filtering"""
        with self.lock:
            events = list(self.buffer)
            
            # Filter by time
            if since:
                events = [e for e in events if e.timestamp >= since]
            
            # Filter by event type
            if event_types:
                events = [e for e in events if e.event_type in event_types]
            
            # Limit count
            if count and len(events) > count:
                events = events[-count:]
            
            return events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                "current_size": len(self.buffer),
                "max_size": self.max_size,
                "total_events": self.total_events,
                "dropped_events": self.dropped_events,
                "drop_rate": self.dropped_events / max(1, self.total_events),
                "events_by_type": dict(self.events_by_type),
                "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None
            }
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()


class StreamProcessor:
    """Base class for stream processors"""
    
    def __init__(self, processor_id: str, name: str):
        self.processor_id = processor_id
        self.name = name
        self.enabled = True
        
        # Processing statistics
        self.processed_events = 0
        self.processing_errors = 0
        self.avg_processing_time = 0.0
        self.last_processed_time = None
    
    async def process_event(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Process a single event"""
        if not self.enabled:
            return None
        
        start_time = time.time()
        
        try:
            result = await self._process_event_impl(event)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000  # ms
            self._update_processing_stats(processing_time)
            
            return result
            
        except Exception as e:
            self.processing_errors += 1
            logger.error(f"Error processing event in {self.name}: {e}")
            return None
    
    async def _process_event_impl(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Override this method to implement specific processing logic"""
        raise NotImplementedError
    
    def _update_processing_stats(self, processing_time_ms: float):
        """Update processing statistics"""
        self.processed_events += 1
        self.last_processed_time = datetime.now(timezone.utc)
        
        # Update average processing time
        current_avg = self.avg_processing_time
        self.avg_processing_time = (current_avg * (self.processed_events - 1) + processing_time_ms) / self.processed_events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            "processor_id": self.processor_id,
            "name": self.name,
            "enabled": self.enabled,
            "processed_events": self.processed_events,
            "processing_errors": self.processing_errors,
            "error_rate": self.processing_errors / max(1, self.processed_events),
            "avg_processing_time_ms": self.avg_processing_time,
            "last_processed_time": self.last_processed_time.isoformat() if self.last_processed_time else None
        }


class AggregationProcessor(StreamProcessor):
    """Processor for real-time aggregations"""
    
    def __init__(self, processor_id: str, aggregation_fields: List[str],
                 aggregation_functions: List[str] = None):
        super().__init__(processor_id, f"Aggregation Processor ({processor_id})")
        self.aggregation_fields = aggregation_fields
        self.aggregation_functions = aggregation_functions or ["sum", "count", "avg"]
        
        # Aggregation state
        self.aggregations = defaultdict(lambda: defaultdict(list))
        self.last_aggregation_time = datetime.now(timezone.utc)
    
    async def _process_event_impl(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Process event for aggregation"""
        # Extract values for aggregation fields
        for field in self.aggregation_fields:
            if field in event.data:
                value = event.data[field]
                if isinstance(value, (int, float)):
                    self.aggregations[field]["values"].append(value)
                    self.aggregations[field]["timestamps"].append(event.timestamp)
                    
                    # Limit history (keep last 1000 values)
                    if len(self.aggregations[field]["values"]) > 1000:
                        self.aggregations[field]["values"].pop(0)
                        self.aggregations[field]["timestamps"].pop(0)
        
        # Return current aggregations
        return self._calculate_current_aggregations()
    
    def _calculate_current_aggregations(self) -> Dict[str, Any]:
        """Calculate current aggregation values"""
        result = {}
        
        for field in self.aggregation_fields:
            if field in self.aggregations:
                values = self.aggregations[field]["values"]
                if values:
                    field_result = {}
                    
                    if "sum" in self.aggregation_functions:
                        field_result["sum"] = sum(values)
                    if "count" in self.aggregation_functions:
                        field_result["count"] = len(values)
                    if "avg" in self.aggregation_functions:
                        field_result["avg"] = sum(values) / len(values)
                    if "min" in self.aggregation_functions:
                        field_result["min"] = min(values)
                    if "max" in self.aggregation_functions:
                        field_result["max"] = max(values)
                    
                    result[field] = field_result
        
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        return result


class AlertProcessor(StreamProcessor):
    """Processor for real-time alerting"""
    
    def __init__(self, processor_id: str, alert_rules: List[Dict[str, Any]]):
        super().__init__(processor_id, f"Alert Processor ({processor_id})")
        self.alert_rules = alert_rules
        self.triggered_alerts = []
        self.alert_history = deque(maxlen=1000)
    
    async def _process_event_impl(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Process event for alerting"""
        triggered_rules = []
        
        for rule in self.alert_rules:
            if self._evaluate_alert_rule(event, rule):
                alert = {
                    "rule_id": rule.get("id", "unknown"),
                    "rule_name": rule.get("name", "Alert"),
                    "severity": rule.get("severity", "medium"),
                    "message": rule.get("message", "Alert triggered"),
                    "event_id": event.event_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": event.data
                }
                
                triggered_rules.append(alert)
                self.alert_history.append(alert)
        
        if triggered_rules:
            return {
                "alerts": triggered_rules,
                "total_alerts": len(triggered_rules)
            }
        
        return None
    
    def _evaluate_alert_rule(self, event: StreamEvent, rule: Dict[str, Any]) -> bool:
        """Evaluate if an alert rule is triggered"""
        try:
            # Simple rule evaluation (could be extended with complex logic)
            field = rule.get("field")
            operator = rule.get("operator", "gt")
            threshold = rule.get("threshold")
            
            if not field or threshold is None:
                return False
            
            value = event.data.get(field)
            if value is None:
                return False
            
            if operator == "gt":
                return value > threshold
            elif operator == "lt":
                return value < threshold
            elif operator == "eq":
                return value == threshold
            elif operator == "gte":
                return value >= threshold
            elif operator == "lte":
                return value <= threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating alert rule: {e}")
            return False


class FilterProcessor(StreamProcessor):
    """Processor for filtering events"""
    
    def __init__(self, processor_id: str, filter_conditions: List[Dict[str, Any]]):
        super().__init__(processor_id, f"Filter Processor ({processor_id})")
        self.filter_conditions = filter_conditions
        self.filtered_count = 0
        self.passed_count = 0
    
    async def _process_event_impl(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Process event for filtering"""
        if self._should_pass_event(event):
            self.passed_count += 1
            return {"passed": True, "event": event.to_dict()}
        else:
            self.filtered_count += 1
            return None
    
    def _should_pass_event(self, event: StreamEvent) -> bool:
        """Check if event should pass through filters"""
        for condition in self.filter_conditions:
            if not self._evaluate_condition(event, condition):
                return False
        return True
    
    def _evaluate_condition(self, event: StreamEvent, condition: Dict[str, Any]) -> bool:
        """Evaluate a single filter condition"""
        try:
            field = condition.get("field")
            operator = condition.get("operator", "eq")
            value = condition.get("value")
            
            if field == "event_type":
                event_value = event.event_type.value
            elif field == "source":
                event_value = event.source
            elif field in event.data:
                event_value = event.data[field]
            else:
                return True  # Unknown field, pass through
            
            if operator == "eq":
                return event_value == value
            elif operator == "ne":
                return event_value != value
            elif operator == "contains":
                return str(value) in str(event_value)
            elif operator == "in":
                return event_value in value
            elif operator == "gt":
                return event_value > value
            elif operator == "lt":
                return event_value < value
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating filter condition: {e}")
            return True


class RealTimeProcessor:
    """Main real-time analytics processor"""
    
    def __init__(self, buffer_size: int = 100000):
        # Core components
        self.stream_buffer = StreamBuffer(buffer_size)
        self.processors: Dict[str, StreamProcessor] = {}
        self.output_handlers: Dict[str, Callable] = {}
        
        # Processing configuration
        self.processing_mode = ProcessingMode.STREAMING
        self.batch_size = 100
        self.batch_timeout = 1.0  # seconds
        
        # State management
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.event_queue = asyncio.Queue()
        
        # Performance metrics
        self.processing_stats = {
            "total_events_processed": 0,
            "events_per_second": 0.0,
            "avg_processing_latency": 0.0,
            "active_processors": 0,
            "failed_processing_attempts": 0
        }
        
        # Windowed processing
        self.window_config: Optional[WindowConfiguration] = None
        self.windowed_events: Dict[str, List[StreamEvent]] = {}
        
        logger.info("Real-time processor initialized")
    
    def add_processor(self, processor: StreamProcessor):
        """Add a stream processor"""
        self.processors[processor.processor_id] = processor
        self.processing_stats["active_processors"] = len(self.processors)
        logger.info(f"Added processor: {processor.name}")
    
    def remove_processor(self, processor_id: str) -> bool:
        """Remove a stream processor"""
        if processor_id in self.processors:
            del self.processors[processor_id]
            self.processing_stats["active_processors"] = len(self.processors)
            logger.info(f"Removed processor: {processor_id}")
            return True
        return False
    
    def add_output_handler(self, name: str, handler: Callable):
        """Add an output handler for processed results"""
        self.output_handlers[name] = handler
        logger.info(f"Added output handler: {name}")
    
    def configure_windowed_processing(self, config: WindowConfiguration):
        """Configure windowed processing"""
        self.window_config = config
        self.processing_mode = ProcessingMode.WINDOWED
        logger.info(f"Configured windowed processing: {config.window_type} windows of {config.window_size}")
    
    async def start(self):
        """Start the real-time processor"""
        if self.is_running:
            logger.warning("Real-time processor already running")
            return
        
        self.is_running = True
        
        # Start processing task based on mode
        if self.processing_mode == ProcessingMode.STREAMING:
            self.processing_task = asyncio.create_task(self._streaming_processor())
        elif self.processing_mode == ProcessingMode.BATCH:
            self.processing_task = asyncio.create_task(self._batch_processor())
        elif self.processing_mode == ProcessingMode.MICRO_BATCH:
            self.processing_task = asyncio.create_task(self._micro_batch_processor())
        elif self.processing_mode == ProcessingMode.WINDOWED:
            self.processing_task = asyncio.create_task(self._windowed_processor())
        
        logger.info(f"Real-time processor started in {self.processing_mode.value} mode")
    
    async def stop(self):
        """Stop the real-time processor"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time processor stopped")
    
    async def ingest_event(self, event: StreamEvent):
        """Ingest a new event into the stream"""
        # Add to buffer
        self.stream_buffer.add_event(event)
        
        # Add to processing queue
        if self.is_running:
            await self.event_queue.put(event)
    
    async def ingest_events_batch(self, events: List[StreamEvent]):
        """Ingest multiple events efficiently"""
        for event in events:
            self.stream_buffer.add_event(event)
            if self.is_running:
                await self.event_queue.put(event)
    
    async def _streaming_processor(self):
        """Process events in streaming mode"""
        while self.is_running:
            try:
                # Get next event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event through all processors
                await self._process_single_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in streaming processor: {e}")
                self.processing_stats["failed_processing_attempts"] += 1
    
    async def _batch_processor(self):
        """Process events in batch mode"""
        batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Collect events for batch
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=0.1)
                    batch.append(event)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if conditions are met
                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and current_time - last_batch_time >= self.batch_timeout)
                )
                
                if should_process and batch:
                    await self._process_event_batch(batch)
                    batch.clear()
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                self.processing_stats["failed_processing_attempts"] += 1
    
    async def _micro_batch_processor(self):
        """Process events in micro-batch mode"""
        # Similar to batch but with smaller batches and shorter timeouts
        batch = []
        micro_batch_size = min(10, self.batch_size)
        micro_batch_timeout = min(0.1, self.batch_timeout)
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=0.01)
                    batch.append(event)
                except asyncio.TimeoutError:
                    pass
                
                current_time = time.time()
                should_process = (
                    len(batch) >= micro_batch_size or
                    (batch and current_time - last_batch_time >= micro_batch_timeout)
                )
                
                if should_process and batch:
                    await self._process_event_batch(batch)
                    batch.clear()
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in micro-batch processor: {e}")
                self.processing_stats["failed_processing_attempts"] += 1
    
    async def _windowed_processor(self):
        """Process events in windowed mode"""
        if not self.window_config:
            logger.error("Window configuration required for windowed processing")
            return
        
        current_window_start = datetime.now(timezone.utc)
        current_window_events = []
        
        while self.is_running:
            try:
                # Get events with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=0.1)
                    current_window_events.append(event)
                except asyncio.TimeoutError:
                    pass
                
                # Check if window should be processed
                now = datetime.now(timezone.utc)
                window_duration = now - current_window_start
                
                should_process = (
                    window_duration >= self.window_config.window_size or
                    len(current_window_events) >= self.window_config.max_events_per_window
                )
                
                if should_process and current_window_events:
                    # Process window
                    await self._process_window(current_window_events, current_window_start, now)
                    
                    # Start new window
                    if self.window_config.window_type == "tumbling":
                        current_window_start = now
                        current_window_events.clear()
                    elif self.window_config.window_type == "sliding":
                        current_window_start += self.window_config.slide_interval
                        # Keep events that are still in the new window
                        cutoff_time = current_window_start
                        current_window_events = [e for e in current_window_events if e.timestamp >= cutoff_time]
                
            except Exception as e:
                logger.error(f"Error in windowed processor: {e}")
                self.processing_stats["failed_processing_attempts"] += 1
    
    async def _process_single_event(self, event: StreamEvent):
        """Process a single event through all processors"""
        start_time = time.time()
        
        try:
            results = {}
            
            # Process through each processor
            for processor_id, processor in self.processors.items():
                result = await processor.process_event(event)
                if result:
                    results[processor_id] = result
            
            # Send to output handlers
            if results:
                await self._send_to_output_handlers(event, results)
            
            # Update statistics
            self.processing_stats["total_events_processed"] += 1
            processing_time = (time.time() - start_time) * 1000  # ms
            self._update_processing_stats(processing_time)
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            self.processing_stats["failed_processing_attempts"] += 1
    
    async def _process_event_batch(self, events: List[StreamEvent]):
        """Process a batch of events"""
        start_time = time.time()
        
        try:
            batch_results = []
            
            for event in events:
                event_results = {}
                
                # Process through each processor
                for processor_id, processor in self.processors.items():
                    result = await processor.process_event(event)
                    if result:
                        event_results[processor_id] = result
                
                if event_results:
                    batch_results.append({
                        "event": event,
                        "results": event_results
                    })
            
            # Send batch to output handlers
            if batch_results:
                await self._send_batch_to_output_handlers(batch_results)
            
            # Update statistics
            self.processing_stats["total_events_processed"] += len(events)
            processing_time = (time.time() - start_time) * 1000  # ms
            self._update_processing_stats(processing_time)
            
        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
            self.processing_stats["failed_processing_attempts"] += 1
    
    async def _process_window(self, events: List[StreamEvent], 
                            window_start: datetime, window_end: datetime):
        """Process a window of events"""
        start_time = time.time()
        
        try:
            window_results = {
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "event_count": len(events),
                "processor_results": {}
            }
            
            # Process through each processor
            for processor_id, processor in self.processors.items():
                processor_results = []
                
                for event in events:
                    result = await processor.process_event(event)
                    if result:
                        processor_results.append(result)
                
                if processor_results:
                    window_results["processor_results"][processor_id] = processor_results
            
            # Send window results to output handlers
            await self._send_window_to_output_handlers(window_results)
            
            # Update statistics
            self.processing_stats["total_events_processed"] += len(events)
            processing_time = (time.time() - start_time) * 1000  # ms
            self._update_processing_stats(processing_time)
            
        except Exception as e:
            logger.error(f"Error processing window: {e}")
            self.processing_stats["failed_processing_attempts"] += 1
    
    async def _send_to_output_handlers(self, event: StreamEvent, results: Dict[str, Any]):
        """Send single event results to output handlers"""
        for handler_name, handler in self.output_handlers.items():
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event, results)
                else:
                    handler(event, results)
            except Exception as e:
                logger.error(f"Error in output handler {handler_name}: {e}")
    
    async def _send_batch_to_output_handlers(self, batch_results: List[Dict[str, Any]]):
        """Send batch results to output handlers"""
        for handler_name, handler in self.output_handlers.items():
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(batch_results)
                else:
                    handler(batch_results)
            except Exception as e:
                logger.error(f"Error in batch output handler {handler_name}: {e}")
    
    async def _send_window_to_output_handlers(self, window_results: Dict[str, Any]):
        """Send window results to output handlers"""
        for handler_name, handler in self.output_handlers.items():
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(window_results)
                else:
                    handler(window_results)
            except Exception as e:
                logger.error(f"Error in window output handler {handler_name}: {e}")
    
    def _update_processing_stats(self, processing_time_ms: float):
        """Update processing statistics"""
        # Update average processing latency
        total_processed = self.processing_stats["total_events_processed"]
        current_avg = self.processing_stats["avg_processing_latency"]
        self.processing_stats["avg_processing_latency"] = \
            (current_avg * (total_processed - 1) + processing_time_ms) / total_processed
        
        # Calculate events per second (rough estimate)
        # This could be improved with a sliding window calculation
        if total_processed > 0:
            self.processing_stats["events_per_second"] = \
                1000.0 / max(1.0, self.processing_stats["avg_processing_latency"])
    
    def get_event_stream(self, event_types: Optional[List[StreamEventType]] = None,
                        buffer_size: int = 1000) -> AsyncGenerator[StreamEvent, None]:
        """Get async generator for event stream"""
        async def event_generator():
            last_processed = 0
            
            while True:
                # Get new events since last check
                all_events = self.stream_buffer.get_events(event_types=event_types)
                new_events = all_events[last_processed:]
                
                for event in new_events:
                    yield event
                
                last_processed = len(all_events)
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
        
        return event_generator()
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics"""
        processor_stats = {}
        for processor_id, processor in self.processors.items():
            processor_stats[processor_id] = processor.get_stats()
        
        return {
            "processing_stats": self.processing_stats,
            "buffer_stats": self.stream_buffer.get_stats(),
            "processor_stats": processor_stats,
            "configuration": {
                "processing_mode": self.processing_mode.value,
                "batch_size": self.batch_size,
                "batch_timeout": self.batch_timeout,
                "window_config": self.window_config.to_dict() if self.window_config else None
            },
            "status": {
                "is_running": self.is_running,
                "queue_size": self.event_queue.qsize(),
                "active_processors": len(self.processors),
                "output_handlers": len(self.output_handlers)
            }
        }
    
    def clear_buffers(self):
        """Clear all buffers and reset statistics"""
        self.stream_buffer.clear()
        
        # Clear event queue
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Reset statistics
        self.processing_stats = {
            "total_events_processed": 0,
            "events_per_second": 0.0,
            "avg_processing_latency": 0.0,
            "active_processors": len(self.processors),
            "failed_processing_attempts": 0
        }
        
        logger.info("Cleared buffers and reset statistics")


# Export main classes
__all__ = [
    'StreamEventType',
    'ProcessingMode',
    'StreamEvent',
    'WindowConfiguration',
    'StreamBuffer',
    'StreamProcessor',
    'AggregationProcessor',
    'AlertProcessor', 
    'FilterProcessor',
    'RealTimeProcessor'
]
"""
PRSM Distributed Tracing System
Comprehensive distributed tracing with OpenTelemetry integration for microservices observability
"""

from typing import Dict, Any, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import uuid
import time
import logging
import traceback
import contextvars
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
import redis.asyncio as aioredis

# OpenTelemetry imports
try:
    from opentelemetry import trace, context, baggage
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Fallback implementations
    class trace:
        @staticmethod
        def get_tracer(name: str): return NoOpTracer()
    
    class NoOpTracer:
        def start_span(self, name: str, **kwargs): return NoOpSpan()
    
    class NoOpSpan:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def set_attribute(self, key: str, value: Any): pass
        def set_status(self, status): pass
        def record_exception(self, exception): pass

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Span kinds for different operation types"""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class TraceLevel(Enum):
    """Trace sampling levels"""
    OFF = 0
    ERROR_ONLY = 1
    IMPORTANT = 2
    NORMAL = 3
    DEBUG = 4
    VERBOSE = 5


@dataclass
class TraceContext:
    """Trace context information"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    sampling_decision: bool = True
    trace_flags: int = 1


@dataclass
class SpanData:
    """Custom span data for internal tracking"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "ok"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    kind: SpanKind = SpanKind.INTERNAL


@dataclass
class TracingConfig:
    """Tracing system configuration"""
    service_name: str = "prsm-api"
    service_version: str = "1.0.0"
    environment: str = "production"
    
    # Sampling configuration
    sampling_rate: float = 1.0  # 100% sampling by default
    trace_level: TraceLevel = TraceLevel.NORMAL
    
    # Export configuration
    export_to_jaeger: bool = False
    jaeger_endpoint: Optional[str] = None
    export_to_otlp: bool = False
    otlp_endpoint: Optional[str] = None
    export_to_console: bool = False
    
    # Storage configuration
    store_spans_in_redis: bool = True
    span_retention_hours: int = 24
    max_spans_per_trace: int = 1000
    
    # Auto-instrumentation
    auto_instrument_redis: bool = True
    auto_instrument_database: bool = True
    auto_instrument_fastapi: bool = True


class TraceStorage:
    """Redis-backed trace storage for span persistence"""
    
    def __init__(self, redis_client: aioredis.Redis, retention_hours: int = 24):
        self.redis = redis_client
        self.retention_seconds = retention_hours * 3600
        
    async def store_span(self, span_data: SpanData):
        """Store span data in Redis"""
        try:
            span_key = f"span:{span_data.span_id}"
            trace_key = f"trace:{span_data.trace_id}"
            
            # Store individual span
            span_json = {
                "span_id": span_data.span_id,
                "trace_id": span_data.trace_id,
                "parent_span_id": span_data.parent_span_id,
                "operation_name": span_data.operation_name,
                "service_name": span_data.service_name,
                "start_time": span_data.start_time.isoformat(),
                "end_time": span_data.end_time.isoformat() if span_data.end_time else None,
                "duration_ms": span_data.duration_ms,
                "status": span_data.status,
                "tags": span_data.tags,
                "logs": span_data.logs,
                "kind": span_data.kind.value
            }
            
            await self.redis.setex(span_key, self.retention_seconds, json.dumps(span_json))
            
            # Add span to trace
            await self.redis.sadd(trace_key, span_data.span_id)
            await self.redis.expire(trace_key, self.retention_seconds)
            
        except Exception as e:
            logger.error(f"Error storing span: {e}")
    
    async def get_span(self, span_id: str) -> Optional[SpanData]:
        """Retrieve span data"""
        try:
            span_data = await self.redis.get(f"span:{span_id}")
            if span_data:
                data = json.loads(span_data)
                return SpanData(
                    span_id=data["span_id"],
                    trace_id=data["trace_id"],
                    parent_span_id=data.get("parent_span_id"),
                    operation_name=data["operation_name"],
                    service_name=data["service_name"],
                    start_time=datetime.fromisoformat(data["start_time"]),
                    end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
                    duration_ms=data.get("duration_ms"),
                    status=data.get("status", "ok"),
                    tags=data.get("tags", {}),
                    logs=data.get("logs", []),
                    kind=SpanKind(data.get("kind", "internal"))
                )
        except Exception as e:
            logger.error(f"Error retrieving span {span_id}: {e}")
        
        return None
    
    async def get_trace(self, trace_id: str) -> List[SpanData]:
        """Retrieve all spans for a trace"""
        try:
            span_ids = await self.redis.smembers(f"trace:{trace_id}")
            spans = []
            
            for span_id in span_ids:
                span_data = await self.get_span(span_id.decode())
                if span_data:
                    spans.append(span_data)
            
            # Sort by start time
            spans.sort(key=lambda s: s.start_time)
            return spans
            
        except Exception as e:
            logger.error(f"Error retrieving trace {trace_id}: {e}")
            return []


class DistributedTracer:
    """Advanced distributed tracing system"""
    
    def __init__(self, config: TracingConfig, redis_client: Optional[aioredis.Redis] = None):
        self.config = config
        self.redis = redis_client
        
        # Internal state
        self.tracer = None
        self.trace_storage = None
        self.active_spans: Dict[str, SpanData] = {}
        
        # Context variables for async correlation
        self.current_trace_context: contextvars.ContextVar[Optional[TraceContext]] = (
            contextvars.ContextVar('current_trace_context', default=None)
        )
        
        # Metrics
        self.stats = {
            "spans_created": 0,
            "spans_finished": 0,
            "traces_started": 0,
            "traces_completed": 0,
            "sampling_decisions": 0,
            "exports_succeeded": 0,
            "exports_failed": 0
        }
        
        # Initialize tracing
        self.initialize()
    
    def initialize(self):
        """Initialize the tracing system"""
        try:
            if OPENTELEMETRY_AVAILABLE:
                self._setup_opentelemetry()
            else:
                logger.warning("OpenTelemetry not available, using fallback tracer")
                self.tracer = trace.get_tracer(self.config.service_name)
            
            # Setup Redis storage
            if self.redis and self.config.store_spans_in_redis:
                self.trace_storage = TraceStorage(self.redis, self.config.span_retention_hours)
            
            # Setup auto-instrumentation
            self._setup_auto_instrumentation()
            
            logger.info(f"✅ Distributed tracing initialized for service '{self.config.service_name}'")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize tracing: {e}")
            # Fall back to no-op tracer
            self.tracer = trace.get_tracer(self.config.service_name)
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry with configured exporters"""
        
        # Create resource
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.environment
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Setup exporters
        processors = []
        
        if self.config.export_to_console:
            console_exporter = ConsoleSpanExporter()
            processors.append(BatchSpanProcessor(console_exporter))
        
        if self.config.export_to_jaeger and self.config.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_endpoint.split(':')[0],
                agent_port=int(self.config.jaeger_endpoint.split(':')[1]) if ':' in self.config.jaeger_endpoint else 14268
            )
            processors.append(BatchSpanProcessor(jaeger_exporter))
        
        if self.config.export_to_otlp and self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            processors.append(BatchSpanProcessor(otlp_exporter))
        
        # Add processors to tracer provider
        for processor in processors:
            tracer_provider.add_span_processor(processor)
        
        # Get tracer
        self.tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )
    
    def _setup_auto_instrumentation(self):
        """Setup automatic instrumentation for common libraries"""
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        try:
            if self.config.auto_instrument_redis:
                RedisInstrumentor().instrument()
            
            if self.config.auto_instrument_database:
                SQLAlchemyInstrumentor().instrument()
                AsyncPGInstrumentor().instrument()
            
            if self.config.auto_instrument_fastapi:
                # FastAPI instrumentation would be done at app level
                pass
                
        except Exception as e:
            logger.warning(f"Auto-instrumentation setup failed: {e}")
    
    def _should_sample(self, operation_name: str) -> bool:
        """Determine if this trace should be sampled"""
        
        # Always sample errors and important operations
        if self.config.trace_level >= TraceLevel.ERROR_ONLY:
            if "error" in operation_name.lower() or "exception" in operation_name.lower():
                return True
        
        if self.config.trace_level >= TraceLevel.IMPORTANT:
            important_operations = ["auth", "payment", "order", "critical"]
            if any(op in operation_name.lower() for op in important_operations):
                return True
        
        # Sample based on configured rate
        if self.config.trace_level >= TraceLevel.NORMAL:
            import random
            should_sample = random.random() < self.config.sampling_rate
            self.stats["sampling_decisions"] += 1
            return should_sample
        
        return False
    
    @contextmanager
    def start_span(self, 
                   operation_name: str,
                   kind: SpanKind = SpanKind.INTERNAL,
                   child_of: Optional[str] = None,
                   tags: Optional[Dict[str, Any]] = None,
                   start_time: Optional[datetime] = None):
        """Start a new span with context management"""
        
        span_id = str(uuid.uuid4())
        current_context = self.current_trace_context.get()
        
        # Determine trace context
        if child_of:
            # Use specified parent
            parent_span_id = child_of
            trace_id = current_context.trace_id if current_context else str(uuid.uuid4())
        elif current_context:
            # Use current context as parent
            parent_span_id = current_context.span_id
            trace_id = current_context.trace_id
        else:
            # Start new trace
            parent_span_id = None
            trace_id = str(uuid.uuid4())
            self.stats["traces_started"] += 1
        
        # Check sampling decision
        if not self._should_sample(operation_name):
            # Create no-op context
            new_context = TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                sampling_decision=False
            )
            
            token = self.current_trace_context.set(new_context)
            try:
                yield NoOpSpan()
            finally:
                self.current_trace_context.reset(token)
            return
        
        # Create span data
        span_data = SpanData(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.config.service_name,
            start_time=start_time or datetime.now(timezone.utc),
            kind=kind,
            tags=tags or {}
        )
        
        # Set new context
        new_context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            sampling_decision=True
        )
        
        # Start OpenTelemetry span if available
        otel_span = None
        if OPENTELEMETRY_AVAILABLE and self.tracer:
            otel_span = self.tracer.start_span(
                operation_name,
                kind=getattr(trace.SpanKind, kind.name, trace.SpanKind.INTERNAL)
            )
            
            # Set attributes
            if tags:
                for key, value in tags.items():
                    otel_span.set_attribute(key, value)
        
        self.active_spans[span_id] = span_data
        self.stats["spans_created"] += 1
        
        token = self.current_trace_context.set(new_context)
        
        try:
            # Create span wrapper
            span_wrapper = TracingSpan(span_data, otel_span, self)
            yield span_wrapper
            
        except Exception as e:
            # Record exception
            span_data.status = "error"
            span_data.tags["error"] = True
            span_data.tags["error.message"] = str(e)
            span_data.logs.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            })
            
            if otel_span:
                otel_span.record_exception(e)
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
            
            raise
        
        finally:
            # Finish span
            span_data.end_time = datetime.now(timezone.utc)
            span_data.duration_ms = (
                (span_data.end_time - span_data.start_time).total_seconds() * 1000
            )
            
            if otel_span:
                otel_span.end()
            
            # Store span if configured
            if self.trace_storage:
                asyncio.create_task(self.trace_storage.store_span(span_data))
            
            # Update stats
            self.stats["spans_finished"] += 1
            if parent_span_id is None:  # Root span
                self.stats["traces_completed"] += 1
            
            # Cleanup
            self.active_spans.pop(span_id, None)
            self.current_trace_context.reset(token)
    
    @asynccontextmanager
    async def start_async_span(self,
                              operation_name: str,
                              kind: SpanKind = SpanKind.INTERNAL,
                              child_of: Optional[str] = None,
                              tags: Optional[Dict[str, Any]] = None):
        """Start an async span with context management"""
        
        with self.start_span(operation_name, kind, child_of, tags) as span:
            yield span
    
    def get_current_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context"""
        return self.current_trace_context.get()
    
    def inject_trace_context(self) -> Dict[str, str]:
        """Inject trace context for propagation"""
        context_data = {}
        current_context = self.current_trace_context.get()
        
        if current_context and current_context.sampling_decision:
            context_data.update({
                "x-trace-id": current_context.trace_id,
                "x-span-id": current_context.span_id,
                "x-parent-span-id": current_context.parent_span_id or "",
                "x-trace-flags": str(current_context.trace_flags)
            })
            
            # Add baggage
            for key, value in current_context.baggage.items():
                context_data[f"x-baggage-{key}"] = value
        
        # Use OpenTelemetry propagation if available
        if OPENTELEMETRY_AVAILABLE:
            otel_context = {}
            inject(otel_context)
            context_data.update(otel_context)
        
        return context_data
    
    def extract_trace_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extract trace context from headers"""
        
        try:
            # Extract custom headers
            trace_id = headers.get("x-trace-id")
            span_id = headers.get("x-span-id")
            parent_span_id = headers.get("x-parent-span-id")
            trace_flags = int(headers.get("x-trace-flags", "1"))
            
            if trace_id and span_id:
                # Extract baggage
                baggage_data = {}
                for key, value in headers.items():
                    if key.startswith("x-baggage-"):
                        baggage_key = key[10:]  # Remove "x-baggage-" prefix
                        baggage_data[baggage_key] = value
                
                return TraceContext(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id if parent_span_id else None,
                    baggage=baggage_data,
                    trace_flags=trace_flags
                )
        
        except Exception as e:
            logger.debug(f"Error extracting trace context: {e}")
        
        # Try OpenTelemetry extraction
        if OPENTELEMETRY_AVAILABLE:
            try:
                otel_context = extract(headers)
                if otel_context:
                    # Convert to our format
                    span = trace.get_current_span(otel_context)
                    if span and span.is_recording():
                        span_context = span.get_span_context()
                        return TraceContext(
                            trace_id=format(span_context.trace_id, '032x'),
                            span_id=format(span_context.span_id, '016x'),
                            trace_flags=span_context.trace_flags
                        )
            except Exception as e:
                logger.debug(f"Error extracting OpenTelemetry context: {e}")
        
        return None
    
    async def get_trace_analytics(self, trace_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a trace"""
        
        if not self.trace_storage:
            return {"error": "Trace storage not configured"}
        
        spans = await self.trace_storage.get_trace(trace_id)
        if not spans:
            return {"error": f"Trace {trace_id} not found"}
        
        # Calculate trace statistics
        total_duration = 0
        service_durations = defaultdict(float)
        operation_counts = defaultdict(int)
        service_counts = defaultdict(int)
        error_count = 0
        
        for span in spans:
            if span.duration_ms:
                total_duration = max(total_duration, 
                                   span.start_time.timestamp() * 1000 + span.duration_ms)
                service_durations[span.service_name] += span.duration_ms
                
            operation_counts[span.operation_name] += 1
            service_counts[span.service_name] += 1
            
            if span.status == "error":
                error_count += 1
        
        # Calculate critical path
        critical_path = self._calculate_critical_path(spans)
        
        return {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "total_duration_ms": total_duration,
            "error_count": error_count,
            "error_rate": (error_count / len(spans)) * 100 if spans else 0,
            "services": dict(service_counts),
            "service_durations": dict(service_durations),
            "operations": dict(operation_counts),
            "critical_path": critical_path,
            "span_tree": self._build_span_tree(spans)
        }
    
    def _calculate_critical_path(self, spans: List[SpanData]) -> List[Dict[str, Any]]:
        """Calculate the critical path through the trace"""
        
        # Build parent-child relationships
        span_map = {span.span_id: span for span in spans}
        children = defaultdict(list)
        
        for span in spans:
            if span.parent_span_id:
                children[span.parent_span_id].append(span)
        
        # Find root spans
        root_spans = [span for span in spans if not span.parent_span_id]
        
        if not root_spans:
            return []
        
        # Calculate critical path for each root
        def find_longest_path(span: SpanData) -> List[SpanData]:
            if span.span_id not in children:
                return [span]
            
            longest_child_path = []
            max_duration = 0
            
            for child in children[span.span_id]:
                child_path = find_longest_path(child)
                child_duration = sum(s.duration_ms or 0 for s in child_path)
                
                if child_duration > max_duration:
                    max_duration = child_duration
                    longest_child_path = child_path
            
            return [span] + longest_child_path
        
        # Get the longest critical path
        critical_path = []
        max_path_duration = 0
        
        for root in root_spans:
            path = find_longest_path(root)
            path_duration = sum(s.duration_ms or 0 for s in path)
            
            if path_duration > max_path_duration:
                max_path_duration = path_duration
                critical_path = path
        
        return [
            {
                "span_id": span.span_id,
                "operation_name": span.operation_name,
                "service_name": span.service_name,
                "duration_ms": span.duration_ms,
                "start_time": span.start_time.isoformat()
            }
            for span in critical_path
        ]
    
    def _build_span_tree(self, spans: List[SpanData]) -> List[Dict[str, Any]]:
        """Build hierarchical span tree"""
        
        span_map = {span.span_id: span for span in spans}
        children = defaultdict(list)
        
        for span in spans:
            if span.parent_span_id and span.parent_span_id in span_map:
                children[span.parent_span_id].append(span)
        
        def build_tree_node(span: SpanData) -> Dict[str, Any]:
            node = {
                "span_id": span.span_id,
                "operation_name": span.operation_name,
                "service_name": span.service_name,
                "start_time": span.start_time.isoformat(),
                "duration_ms": span.duration_ms,
                "status": span.status,
                "tags": span.tags,
                "children": []
            }
            
            # Add children
            span_children = children.get(span.span_id, [])
            span_children.sort(key=lambda s: s.start_time)
            
            for child in span_children:
                node["children"].append(build_tree_node(child))
            
            return node
        
        # Build tree from root spans
        root_spans = [span for span in spans if not span.parent_span_id]
        root_spans.sort(key=lambda s: s.start_time)
        
        return [build_tree_node(span) for span in root_spans]
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get tracing system statistics"""
        
        return {
            "tracing_enabled": True,
            "service_name": self.config.service_name,
            "sampling_rate": self.config.sampling_rate,
            "trace_level": self.config.trace_level.name,
            "active_spans": len(self.active_spans),
            "statistics": self.stats.copy(),
            "exporters": {
                "jaeger": self.config.export_to_jaeger,
                "otlp": self.config.export_to_otlp,
                "console": self.config.export_to_console,
                "redis": self.config.store_spans_in_redis
            }
        }


class TracingSpan:
    """Wrapper for span operations"""
    
    def __init__(self, span_data: SpanData, otel_span, tracer: DistributedTracer):
        self.span_data = span_data
        self.otel_span = otel_span
        self.tracer = tracer
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the span"""
        self.span_data.tags[key] = value
        if self.otel_span:
            self.otel_span.set_attribute(key, value)
    
    def set_tags(self, tags: Dict[str, Any]):
        """Set multiple tags on the span"""
        for key, value in tags.items():
            self.set_tag(key, value)
    
    def log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.span_data.logs.append(log_entry)
    
    def set_status(self, status: str, message: Optional[str] = None):
        """Set span status"""
        self.span_data.status = status
        if message:
            self.span_data.tags["status.message"] = message
        
        if self.otel_span:
            if status == "error":
                self.otel_span.set_status(Status(StatusCode.ERROR, message or ""))
            else:
                self.otel_span.set_status(Status(StatusCode.OK))
    
    def record_exception(self, exception: Exception):
        """Record an exception in the span"""
        self.set_status("error", str(exception))
        self.set_tag("error", True)
        self.set_tag("error.type", type(exception).__name__)
        self.set_tag("error.message", str(exception))
        
        self.log(
            message=f"Exception: {str(exception)}",
            level="error",
            traceback=traceback.format_exc()
        )
        
        if self.otel_span:
            self.otel_span.record_exception(exception)
    
    @property
    def span_id(self) -> str:
        return self.span_data.span_id
    
    @property
    def trace_id(self) -> str:
        return self.span_data.trace_id


# Global tracer instance
distributed_tracer: Optional[DistributedTracer] = None


def initialize_tracing(config: TracingConfig, redis_client: Optional[aioredis.Redis] = None):
    """Initialize the distributed tracing system"""
    global distributed_tracer
    
    distributed_tracer = DistributedTracer(config, redis_client)
    logger.info("✅ Distributed tracing system initialized")


def get_tracer() -> DistributedTracer:
    """Get the global tracer instance"""
    if distributed_tracer is None:
        raise RuntimeError("Tracing system not initialized")
    return distributed_tracer


# Convenience functions and decorators

def trace_function(operation_name: Optional[str] = None,
                  kind: SpanKind = SpanKind.INTERNAL,
                  tags: Optional[Dict[str, Any]] = None):
    """Decorator to automatically trace function calls"""
    
    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                async with tracer.start_async_span(operation_name, kind, tags=tags) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_tag("result.success", True)
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        raise
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_span(operation_name, kind, tags=tags) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_tag("result.success", True)
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        raise
            
            return sync_wrapper
    
    return decorator


@contextmanager
def start_span(operation_name: str, **kwargs):
    """Context manager for starting spans"""
    tracer = get_tracer()
    with tracer.start_span(operation_name, **kwargs) as span:
        yield span


@asynccontextmanager
async def start_async_span(operation_name: str, **kwargs):
    """Async context manager for starting spans"""
    tracer = get_tracer()
    async with tracer.start_async_span(operation_name, **kwargs) as span:
        yield span


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID"""
    tracer = get_tracer()
    context = tracer.get_current_trace_context()
    return context.trace_id if context else None


def get_current_span_id() -> Optional[str]:
    """Get current span ID"""
    tracer = get_tracer()
    context = tracer.get_current_trace_context()
    return context.span_id if context else None


def inject_trace_headers() -> Dict[str, str]:
    """Get trace headers for HTTP requests"""
    tracer = get_tracer()
    return tracer.inject_trace_context()


def extract_trace_context(headers: Dict[str, str]):
    """Extract trace context from HTTP headers"""
    tracer = get_tracer()
    return tracer.extract_trace_context(headers)
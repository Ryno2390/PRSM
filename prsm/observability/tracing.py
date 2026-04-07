"""
Forge Pipeline Tracing
======================

Lightweight tracing for the forge → dispatch → execute → settle pipeline.
Compatible with OpenTelemetry when available, falls back to in-memory spans.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpanContext:
    """A single trace span."""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_id: str = ""
    operation: str = ""
    service: str = "prsm"
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time <= 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def finish(self, status: str = "ok") -> None:
        self.end_time = time.time()
        self.status = status

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "operation": self.operation,
            "service": self.service,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class ForgeTracer:
    """Traces the forge pipeline lifecycle."""

    def __init__(self):
        self._traces: Dict[str, List[SpanContext]] = {}

    def start_trace(self, operation: str, attributes: Optional[Dict[str, Any]] = None) -> SpanContext:
        trace_id = uuid.uuid4().hex[:32]
        span = SpanContext(
            trace_id=trace_id,
            operation=operation,
            attributes=attributes or {},
        )
        self._traces[trace_id] = [span]
        return span

    def start_span(self, parent: SpanContext, operation: str, attributes: Optional[Dict[str, Any]] = None) -> SpanContext:
        span = SpanContext(
            trace_id=parent.trace_id,
            parent_id=parent.span_id,
            operation=operation,
            attributes=attributes or {},
        )
        if parent.trace_id in self._traces:
            self._traces[parent.trace_id].append(span)
        return span

    def get_trace(self, trace_id: str) -> List[SpanContext]:
        return self._traces.get(trace_id, [])

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent traces as dicts for dashboard display."""
        recent = []
        for trace_id, spans in list(self._traces.items())[-limit:]:
            root = spans[0] if spans else None
            recent.append({
                "trace_id": trace_id,
                "operation": root.operation if root else "?",
                "span_count": len(spans),
                "total_duration_ms": sum(s.duration_ms for s in spans if s.end_time > 0),
                "status": root.status if root else "unknown",
            })
        return recent

    @property
    def trace_count(self) -> int:
        return len(self._traces)

"""
Analytics API
=============

Real-time analytics endpoints for stream event data, aggregations,
and processor health. Backed by the RealTimeProcessor stream pipeline.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import structlog

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from prsm.core.auth import get_current_user
from prsm.core.models import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


def _get_processor():
    """Dependency: get the real-time processor or raise 503."""
    try:
        from prsm.data.analytics.real_time_processor import get_real_time_processor
        return get_real_time_processor()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Analytics processor not available"
        )


@router.get("/status")
async def get_analytics_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get real-time analytics processor health and statistics.
    Returns processor mode, queue depth, buffer fill, and per-processor stats.
    """
    processor = _get_processor()
    return JSONResponse(content=processor.get_processor_stats())


@router.get("/events")
async def get_recent_events(
    limit: int = Query(default=100, ge=1, le=1000),
    since_minutes: int = Query(default=60, ge=1, le=1440),
    event_type: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_user)
):
    """
    Query recent events from the stream buffer.

    - **limit**: Maximum number of events to return (1–1000)
    - **since_minutes**: Only return events from the last N minutes (1–1440)
    - **event_type**: Filter by StreamEventType value (e.g. 'performance_event')
    """
    processor = _get_processor()

    since = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)

    event_type_filter = None
    if event_type:
        from prsm.data.analytics.real_time_processor import StreamEventType
        try:
            event_type_filter = [StreamEventType(event_type)]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event_type '{event_type}'. "
                       f"Valid values: {[e.value for e in StreamEventType]}"
            )

    events = processor.stream_buffer.get_events(
        count=limit,
        since=since,
        event_types=event_type_filter
    )

    return JSONResponse(content={
        "events": [e.to_dict() for e in events],
        "count": len(events),
        "since": since.isoformat(),
        "buffer_stats": processor.stream_buffer.get_stats()
    })


@router.get("/aggregations/{processor_id}")
async def get_aggregations(
    processor_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get current aggregation state for a named AggregationProcessor.

    Built-in processor IDs: **api_performance**
    """
    processor = _get_processor()

    if processor_id not in processor.processors:
        raise HTTPException(
            status_code=404,
            detail=f"Processor '{processor_id}' not found. "
                   f"Available: {list(processor.processors.keys())}"
        )

    target = processor.processors[processor_id]

    # Return processor stats plus aggregation state if it's an AggregationProcessor
    from prsm.data.analytics.real_time_processor import AggregationProcessor
    result = target.get_stats()

    if isinstance(target, AggregationProcessor):
        result["current_aggregations"] = target._calculate_current_aggregations()

    return JSONResponse(content=result)


@router.get("/aggregations")
async def list_aggregations(
    current_user: User = Depends(get_current_user)
):
    """List all active processors and their current stats."""
    processor = _get_processor()

    from prsm.data.analytics.real_time_processor import AggregationProcessor
    result = {}

    for proc_id, proc in processor.processors.items():
        stats = proc.get_stats()
        if isinstance(proc, AggregationProcessor):
            stats["current_aggregations"] = proc._calculate_current_aggregations()
        result[proc_id] = stats

    return JSONResponse(content={
        "processors": result,
        "processing_stats": processor.processing_stats
    })

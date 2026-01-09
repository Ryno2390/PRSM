"""
PRSM Task API Router
Handles distributed task queue and task management endpoints
"""

from typing import Dict, Any

import structlog
from fastapi import APIRouter, HTTPException

from prsm.core.redis_client import get_task_queue

# Initialize router
router = APIRouter()
logger = structlog.get_logger(__name__)

@router.post("/enqueue")
async def enqueue_task(task_request: Dict[str, Any]) -> Dict[str, str]:
    """
    Enqueue task in distributed task queue
    
    📝 TASK DISTRIBUTION:
    Adds tasks to Redis-based distributed queue for processing
    across the PRSM P2P network with priority ordering
    """
    try:
        # Validate required fields
        required_fields = ["queue_name", "task_data"]
        for field in required_fields:
            if field not in task_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        queue_name = task_request["queue_name"]
        task_data = task_request["task_data"]
        priority = task_request.get("priority", 5)  # Default medium priority
        
        # Get task queue connection
        task_queue = get_task_queue()
        if not task_queue:
            raise HTTPException(
                status_code=503,
                detail="Task queue service not available"
            )
        
        # Enqueue task
        task_id = await task_queue.enqueue_task(
            queue_name=queue_name,
            task_data=task_data,
            priority=priority
        )
        
        if task_id:
            logger.info("Task enqueued successfully",
                       task_id=task_id,
                       queue_name=queue_name,
                       priority=priority)
            
            return {
                "success": True,
                "task_id": task_id,
                "queue_name": queue_name,
                "priority": priority,
                "status": "enqueued"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to enqueue task"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to enqueue task", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Task enqueue failed"
        )


@router.get("/status/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get task status from distributed queue
    
    📋 TASK MONITORING:
    Retrieves current status of a task from the distributed queue
    including processing progress and results
    """
    try:
        task_queue = get_task_queue()
        if not task_queue:
            raise HTTPException(
                status_code=503,
                detail="Task queue service not available"
            )
        
        # Get task status
        status = await task_queue.get_task_status(task_id)
        
        if status:
            logger.info("Task status retrieved",
                       task_id=task_id,
                       status=status.get("status", "unknown"))
            
            return {
                "success": True,
                "task_id": task_id,
                "status": status.get("status", "unknown"),
                "progress": status.get("progress", 0),
                "result": status.get("result"),
                "error": status.get("error"),
                "created_at": status.get("created_at"),
                "updated_at": status.get("updated_at")
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Task not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get task status", 
                    task_id=task_id, 
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve task status"
        )


@router.get("/queue/{queue_name}/stats")
async def get_queue_stats(queue_name: str) -> Dict[str, Any]:
    """
    Get queue statistics and health information
    
    📊 QUEUE MONITORING:
    Returns queue metrics including pending tasks, processing rates,
    and worker status for monitoring and scaling decisions
    """
    try:
        task_queue = get_task_queue()
        if not task_queue:
            raise HTTPException(
                status_code=503,
                detail="Task queue service not available"
            )
        
        # Get queue statistics
        stats = await task_queue.get_queue_stats(queue_name)
        
        if stats:
            logger.info("Queue stats retrieved",
                       queue_name=queue_name,
                       pending_tasks=stats.get("pending", 0))
            
            return {
                "success": True,
                "queue_name": queue_name,
                "pending_tasks": stats.get("pending", 0),
                "processing_tasks": stats.get("processing", 0),
                "completed_tasks": stats.get("completed", 0),
                "failed_tasks": stats.get("failed", 0),
                "workers_active": stats.get("workers_active", 0),
                "processing_rate": stats.get("processing_rate", "0 tasks/min"),
                "average_processing_time": stats.get("avg_processing_time", "unknown")
            }
        else:
            # Return empty stats for new/unknown queues
            return {
                "success": True,
                "queue_name": queue_name,
                "pending_tasks": 0,
                "processing_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "workers_active": 0,
                "processing_rate": "0 tasks/min",
                "average_processing_time": "unknown"
            }
        
    except Exception as e:
        logger.error("Failed to get queue stats", 
                    queue_name=queue_name, 
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve queue statistics"
        )
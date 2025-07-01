"""
PRSM Session API Router
Handles session management and lifecycle endpoints
"""

from typing import Dict, Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException

from prsm.core.models import UserInput
from prsm.core.database import SessionQueries
from prsm.core.redis_client import get_session_cache

# Initialize router
router = APIRouter()
logger = structlog.get_logger(__name__)

@router.post("/")
async def create_session(user_input: UserInput) -> Dict[str, Any]:
    """
    Create a new PRSM session with database persistence
    
    🎯 SESSION LIFECYCLE:
    Creates a new session in PostgreSQL with context allocation
    and initializes the reasoning trace for transparency
    """
    try:
        session_id = uuid4()
        
        # Create session in database
        session_data = {
            "session_id": str(session_id),
            "user_id": user_input.user_id,
            "prompt": user_input.prompt,
            "preferences": user_input.preferences,
            "status": "initialized",
            "context_budget": user_input.preferences.get("max_context", 4000),
            "created_at": None,  # Database will set timestamp
            "reasoning_trace": {
                "initialization": {
                    "prompt_received": user_input.prompt[:100] + "..." if len(user_input.prompt) > 100 else user_input.prompt,
                    "preferences": user_input.preferences,
                    "context_allocated": user_input.preferences.get("max_context", 4000)
                }
            }
        }
        
        # Store in database
        db_session = await SessionQueries.create_session(session_data)
        
        if db_session:
            # Also store in Redis cache for fast access
            session_cache = get_session_cache()
            if session_cache:
                await session_cache.store_session(str(session_id), session_data)
            
            logger.info("Session created successfully",
                       session_id=str(session_id),
                       user_id=user_input.user_id)
            
            return {
                "success": True,
                "session_id": str(session_id),
                "session": session_data
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to create session in database"
            )
        
    except Exception as e:
        logger.error("Failed to create session", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to create session"
        )


@router.get("/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    """
    Get session details from database
    
    📋 SESSION RETRIEVAL:
    Fetches complete session information including current status,
    context usage, and reasoning trace from PostgreSQL
    """
    try:
        # Try to get session from Redis cache first for speed
        session_cache = get_session_cache()
        session_data = None
        
        if session_cache:
            session_data = await session_cache.get_session(session_id)
            
        # If not in cache, get from database
        if not session_data:
            session_data = await SessionQueries.get_session(session_id)
            
            # Store in cache for future requests
            if session_data and session_cache:
                await session_cache.store_session(session_id, session_data)
        
        if session_data:
            logger.info("Session retrieved successfully",
                       session_id=session_id,
                       status=session_data.get("status", "unknown"))
            
            return {
                "success": True,
                "session": session_data
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve session", 
                    session_id=session_id, 
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session"
        )
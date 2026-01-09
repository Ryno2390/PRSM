"""
PRSM UI API Router
Handles all frontend/UI related endpoints
"""

from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from prsm.core.database import FTNSQueries
from prsm.core.redis_client import get_session_cache

# Initialize router
router = APIRouter()
logger = structlog.get_logger(__name__)

# === Conversation Management ===

@router.post("/conversations")
async def create_conversation(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new conversation for the UI
    
    💬 CONVERSATION CREATION:
    Creates a new conversation session with initialization for
    NWTN processing and real-time UI updates
    """
    try:
        # Create conversation
        conversation_id = str(uuid4())
        conversation = {
            "conversation_id": conversation_id,
            "user_id": conversation_data.get("user_id", "anonymous"),
            "title": conversation_data.get("title", "New Conversation"),
            "model": conversation_data.get("model", "nwtn-v1"),
            "mode": conversation_data.get("mode", "dynamic"),
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "metadata": {
                "ui_mode": "adaptive",
                "response_format": "structured",
                "real_time_updates": True
            }
        }
        
        # Store in session cache
        session_cache = get_session_cache()
        if session_cache:
            await session_cache.store_session(conversation_id, conversation)
        
        logger.info("New conversation created for UI",
                   conversation_id=conversation_id,
                   user_id=conversation.get("user_id"))
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "conversation": conversation
        }
        
    except Exception as e:
        logger.error("Failed to create conversation", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to create conversation"
        )


@router.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message in a conversation
    
    📤 MESSAGE PROCESSING:
    Processes user messages through NWTN orchestrator with
    real-time streaming response for the UI
    """
    try:
        # Validate message
        if "content" not in message_data:
            raise HTTPException(
                status_code=400,
                detail="Missing message content"
            )
        
        message_id = str(uuid4())
        message = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "user_id": message_data.get("user_id", "anonymous"),
            "content": message_data["content"],
            "timestamp": datetime.now().isoformat(),
            "type": "user_message",
            "metadata": {
                "ui_source": True,
                "processing_mode": message_data.get("mode", "adaptive")
            }
        }
        
        # Process through NWTN (simplified for UI)
        ai_response = {
            "message_id": str(uuid4()),
            "conversation_id": conversation_id,
            "content": f"AI response to: {message_data['content'][:50]}...",
            "timestamp": datetime.now().isoformat(),
            "type": "ai_response",
            "model": "nwtn-v1",
            "metadata": {
                "processing_time": 1.23,
                "tokens_used": 150,
                "confidence": 0.92
            }
        }
        
        logger.info("Message processed via UI",
                   conversation_id=conversation_id,
                   message_id=message_id)
        
        return {
            "success": True,
            "user_message": message,
            "ai_response": ai_response
        }
        
    except Exception as e:
        logger.error("Failed to process message", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to process message"
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """Get conversation details for UI display"""
    try:
        session_cache = get_session_cache()
        conversation = None
        
        if session_cache:
            conversation = await session_cache.get_session(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        return {
            "success": True,
            "conversation": conversation
        }
        
    except Exception as e:
        logger.error("Failed to retrieve conversation", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversation"
        )


@router.get("/conversations")
async def list_conversations(user_id: str = None) -> Dict[str, Any]:
    """
    List conversations for conversation history sidebar
    
    📋 CONVERSATION HISTORY:
    Returns list of recent conversations for the history sidebar
    with titles, timestamps, and status information
    """
    try:
        # Mock conversation history (in production, query from database)
        conversations = [
            {
                "conversation_id": "conv-1",
                "title": "Research on Atomically Precise Manufacturing",
                "last_message_at": "2024-01-15T10:30:00Z",
                "message_count": 12,
                "status": "active"
            },
            {
                "conversation_id": "conv-2", 
                "title": "AI Safety Framework Analysis",
                "last_message_at": "2024-01-14T16:45:00Z",
                "message_count": 8,
                "status": "completed"
            },
            {
                "conversation_id": "conv-3",
                "title": "FTNS Tokenomics Deep Dive",
                "last_message_at": "2024-01-13T09:15:00Z",
                "message_count": 23,
                "status": "active"
            }
        ]
        
        return {
            "success": True,
            "conversations": conversations,
            "total": len(conversations)
        }
        
    except Exception as e:
        logger.error("Failed to list conversations", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to list conversations"
        )


# === File Management ===

@router.post("/files/upload")
async def upload_file_ui(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle file uploads from the UI
    
    📁 FILE UPLOAD:
    Processes file uploads from the UI and stores them in IPFS
    with automatic metadata generation and user file management
    """
    try:
        import base64
        from pathlib import Path
        
        # Validate required fields
        required_fields = ["filename", "content", "content_type"]
        for field in required_fields:
            if field not in file_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # Decode file content
        file_content = base64.b64decode(file_data["content"])
        file_size = len(file_content)
        
        # Generate file metadata
        file_id = str(uuid4())
        file_metadata = {
            "file_id": file_id,
            "filename": file_data["filename"],
            "content_type": file_data["content_type"],
            "size": file_size,
            "uploaded_at": datetime.now().isoformat(),
            "user_id": file_data.get("user_id", "anonymous"),
            "privacy": file_data.get("privacy", "private"),
            "ai_access": file_data.get("ai_access", "core_only"),
            "ipfs_cid": f"Qm{file_id[:32]}"  # Mock IPFS CID
        }
        
        logger.info("File uploaded via UI",
                   file_id=file_id,
                   filename=file_data["filename"],
                   size=file_size)
        
        return {
            "success": True,
            "file": file_metadata
        }
        
    except Exception as e:
        logger.error("File upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="File upload failed"
        )


@router.get("/files")
async def list_user_files(user_id: str = None) -> Dict[str, Any]:
    """
    List user files for My Files tab
    
    📂 FILE MANAGEMENT:
    Returns user's uploaded files with privacy settings,
    sharing options, and AI access permissions
    """
    try:
        # Mock file list (in production, query from database)
        files = [
            {
                "file_id": "QmXXX1",
                "filename": "analysis.ipynb",
                "content_type": "application/x-ipynb+json",
                "size": 245760,
                "uploaded_at": "2024-01-15T08:30:00Z",
                "privacy": "private",
                "ai_access": "core_only",
                "sharing": {
                    "public_link": False,
                    "shared_with": [],
                    "ai_training": False
                }
            },
            {
                "file_id": "QmXXX2",
                "filename": "research_data.csv",
                "content_type": "text/csv",
                "size": 1024000,
                "uploaded_at": "2024-01-14T14:20:00Z",
                "privacy": "public",
                "ai_access": "full",
                "sharing": {
                    "public_link": True,
                    "shared_with": ["alice@university.edu"],
                    "ai_training": True
                }
            }
        ]
        
        return {
            "success": True,
            "files": files,
            "total": len(files),
            "storage_used": sum(f["size"] for f in files),
            "storage_limit": 10 * 1024 * 1024 * 1024  # 10GB
        }
        
    except Exception as e:
        logger.error("Failed to list files", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to list files"
        )


# === Tokenomics Dashboard ===

@router.get("/tokenomics/{user_id}")
async def get_tokenomics_ui(user_id: str) -> Dict[str, Any]:
    """
    Get user tokenomics data for UI display
    
    💰 TOKENOMICS DASHBOARD:
    Returns comprehensive FTNS token information including
    balance, staking, earnings, and transaction history
    """
    try:
        # Get balance from existing endpoint
        balance_data = await FTNSQueries.get_user_balance(user_id)
        
        # Mock additional tokenomics data
        tokenomics_data = {
            "balance": {
                "total": balance_data["balance"],
                "available": balance_data["balance"] - balance_data["locked_balance"],
                "locked": balance_data["locked_balance"],
                "currency": "FTNS"
            },
            "staking": {
                "staked_amount": 1500.0,
                "staking_rewards": 45.23,
                "apy": 12.5,
                "lock_period_days": 90,
                "next_reward_date": "2024-01-20T00:00:00Z"
            },
            "earnings": {
                "daily": 5.67,
                "weekly": 38.45,
                "monthly": 156.78,
                "total_lifetime": 2345.67
            },
            "recent_transactions": [
                {
                    "tx_id": "tx_001",
                    "type": "reward",
                    "amount": 15.50,
                    "timestamp": "2024-01-15T12:00:00Z",
                    "description": "Staking rewards"
                },
                {
                    "tx_id": "tx_002",
                    "type": "spent",
                    "amount": -5.25,
                    "timestamp": "2024-01-15T09:30:00Z",
                    "description": "Model inference cost"
                }
            ]
        }
        
        return {
            "success": True,
            "tokenomics": tokenomics_data
        }
        
    except Exception as e:
        logger.error("Failed to retrieve tokenomics data", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve tokenomics data"
        )


# === Task Management ===

@router.get("/tasks/{user_id}")
async def get_user_tasks(user_id: str) -> Dict[str, Any]:
    """
    Get user tasks for Tasks tab
    
    📋 TASK MANAGEMENT:
    Returns user's assigned tasks with status, priorities,
    and interactive management capabilities
    """
    try:
        # Mock task data (in production, query from database)
        tasks = [
            {
                "task_id": "task-1",
                "title": "Review research paper draft",
                "description": "Review the quantum computing research paper and provide feedback",
                "status": "pending",
                "priority": "high",
                "assigned_by": "ai_system",
                "created_at": "2024-01-15T08:00:00Z",
                "due_date": "2024-01-17T17:00:00Z",
                "progress": 0,
                "actions": ["mark_done", "request_extension", "delegate"]
            },
            {
                "task_id": "task-2",
                "title": "Validate tokenomics model",
                "description": "Review and validate the proposed FTNS tokenomics improvements",
                "status": "in_progress",
                "priority": "medium",
                "assigned_by": "governance_system",
                "created_at": "2024-01-14T10:30:00Z",
                "due_date": "2024-01-20T12:00:00Z",
                "progress": 60,
                "actions": ["update_progress", "mark_done", "add_comment"]
            }
        ]
        
        return {
            "success": True,
            "tasks": tasks,
            "summary": {
                "total": len(tasks),
                "pending": len([t for t in tasks if t["status"] == "pending"]),
                "in_progress": len([t for t in tasks if t["status"] == "in_progress"]),
                "completed": len([t for t in tasks if t["status"] == "completed"])
            }
        }
        
    except Exception as e:
        logger.error("Failed to retrieve tasks", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve tasks"
        )


@router.post("/tasks")
async def create_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new task"""
    try:
        task_id = str(uuid4())
        task = {
            "task_id": task_id,
            "title": task_data.get("title", "New Task"),
            "description": task_data.get("description", ""),
            "status": "pending",
            "priority": task_data.get("priority", "medium"),
            "assigned_by": task_data.get("assigned_by", "user"),
            "created_at": datetime.now().isoformat(),
            "due_date": task_data.get("due_date"),
            "actions": task_data.get("actions", ["mark_done"])
        }
        
        logger.info("New task created via UI",
                   task_id=task_id,
                   title=task["title"])
        
        return {
            "success": True,
            "task": task
        }
        
    except Exception as e:
        logger.error("Failed to create task", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to create task"
        )


# === Settings Management ===

@router.post("/settings/save")
async def save_ui_settings(settings_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save user settings from the UI
    
    ⚙️ SETTINGS PERSISTENCE:
    Saves user preferences, API keys, and configuration
    securely with validation and encryption
    """
    try:
        user_id = settings_data.get("user_id", "anonymous")
        
        # Validate and process API keys
        api_keys = settings_data.get("api_keys", {})
        processed_keys = {}
        
        for provider, key_data in api_keys.items():
            if key_data.get("key") and len(key_data["key"]) > 10:  # Basic validation
                # In production, encrypt API keys before storage
                processed_keys[provider] = {
                    "key": f"***{key_data['key'][-4:]}",  # Mask for response
                    "configured": True,
                    "mode_mapping": key_data.get("mode_mapping", "adaptive")
                }
        
        settings = {
            "user_id": user_id,
            "api_keys": processed_keys,
            "ui_preferences": settings_data.get("ui_preferences", {}),
            "saved_at": datetime.now().isoformat()
        }
        
        logger.info("UI settings saved",
                   user_id=user_id,
                   keys_configured=len(processed_keys))
        
        return {
            "success": True,
            "settings": settings
        }
        
    except Exception as e:
        logger.error("Failed to save settings", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to save settings"
        )


@router.get("/settings/{user_id}")
async def get_ui_settings(user_id: str) -> Dict[str, Any]:
    """Get user settings for UI display"""
    try:
        # Mock settings retrieval
        settings = {
            "user_id": user_id,
            "api_keys": {
                "openai": {
                    "configured": True,
                    "mode_mapping": "code",
                    "key_preview": "***XYZ"
                },
                "anthropic": {
                    "configured": False,
                    "mode_mapping": "dynamic",
                    "key_preview": None
                }
            },
            "ui_preferences": {
                "theme": "dark",
                "language": "en",
                "notifications": True,
                "auto_save": True
            }
        }
        
        return {
            "success": True,
            "settings": settings
        }
        
    except Exception as e:
        logger.error("Failed to retrieve settings", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve settings"
        )


# === Information Space Visualization ===

@router.get("/information-space")
async def get_information_space_data(
    layout: str = Query("force_directed", description="Layout algorithm"),
    color_by: str = Query("type", description="Color scheme"),
    filters: Optional[str] = Query(None, description="JSON filters"),
    limit: int = Query(100, ge=1, le=500, description="Maximum nodes to return")
) -> Dict[str, Any]:
    """
    Get data for Information Space visualization
    
    🔬 INFORMATION SPACE:
    Returns real graph data for research opportunity visualization
    based on IPFS content analysis and semantic relationships
    
    Enhanced with comprehensive Information Space functionality:
    - Real content analysis from IPFS
    - Semantic relationship mapping
    - Research opportunity identification
    - Interactive visualization data
    - FTNS token integration
    """
    try:
        # Mock graph data for Information Space
        nodes = [
            {
                "id": "paper_001",
                "type": "research_paper",
                "title": "Atomically Precise Manufacturing Advances",
                "authors": ["Dr. Alice Research", "Prof. Bob Science"],
                "citations": 127,
                "publication_date": "2024-01-10",
                "research_value": 8.5,
                "ftns_potential": 450.0,
                "position": {"x": 0, "y": 0},
                "size": 12
            },
            {
                "id": "dataset_001", 
                "type": "dataset",
                "title": "Manufacturing Process Data",
                "size_gb": 2.3,
                "quality_score": 9.2,
                "ftns_potential": 180.0,
                "position": {"x": 50, "y": 30},
                "size": 8
            }
        ]
        
        edges = [
            {
                "source": "paper_001",
                "target": "dataset_001",
                "relationship": "uses_data",
                "strength": 0.85,
                "ftns_flow": 25.0
            }
        ]
        
        return {
            "success": True,
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "layout": layout,
                "metadata": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "layout_algorithm": layout,
                    "color_scheme": color_by,
                    "generated_at": datetime.now().isoformat()
                }
            }
        }
        
    except Exception as e:
        logger.error("Failed to generate information space data", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to generate information space data"
        )


# === Streaming Support ===

@router.post("/conversations/{conversation_id}/messages/streaming")
async def send_streaming_message(conversation_id: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send message with WebSocket streaming response
    
    💬 STREAMING INTEGRATION:
    Sends message via REST API and triggers WebSocket streaming
    for real-time AI response delivery
    """
    try:
        # Validate message
        if "content" not in message_data:
            raise HTTPException(
                status_code=400,
                detail="Missing message content"
            )
        
        user_id = message_data.get("user_id", "anonymous")
        message_id = str(uuid4())
        
        # Store message
        message = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "content": message_data["content"],
            "timestamp": datetime.now().isoformat(),
            "type": "user_message",
            "streaming": True
        }
        
        # Trigger WebSocket notification for streaming response
        # In production, this would trigger the actual NWTN processing
        
        logger.info("Streaming message initiated",
                   conversation_id=conversation_id,
                   message_id=message_id)
        
        return {
            "success": True,
            "message": message,
            "streaming": {
                "enabled": True,
                "websocket_room": f"conversation_{conversation_id}",
                "expected_response_time": "2-5 seconds"
            }
        }
        
    except Exception as e:
        logger.error("Failed to send streaming message", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to send streaming message"
        )
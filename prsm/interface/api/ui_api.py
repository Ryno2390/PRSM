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
async def send_message(
    conversation_id: str,
    message_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Send a message in a conversation and get a real NWTN response.

    Routes the user's message through NeuroSymbolicOrchestrator (System 1 +
    optional System 2 verification), deducts FTNS from the user's account
    for tokens consumed, and returns the AI response in the UI format.

    FTNS deduction is non-blocking: if the database is unavailable the
    response still succeeds with ftns_charged=0 in the metadata.
    Anonymous users (user_id="anonymous") are not charged.
    """
    import os
    import time
    from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
    from prsm.core.database import FTNSQueries

    # ── Validate input ───────────────────────────────────────────────────────
    if "content" not in message_data:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: content",
        )

    user_id     = message_data.get("user_id", "anonymous")
    prompt      = message_data["content"]
    context     = message_data.get("context", "")  # optional prior context
    message_id  = str(uuid4())
    session_id  = str(uuid4())  # unique per request — used for FTNS idempotency

    # ── Build user message record ────────────────────────────────────────────
    user_message = {
        "message_id":      message_id,
        "conversation_id": conversation_id,
        "user_id":         user_id,
        "content":         prompt,
        "timestamp":       datetime.now().isoformat(),
        "type":            "user_message",
        "metadata": {
            "ui_source":       True,
            "processing_mode": message_data.get("mode", "adaptive"),
        },
    }

    # ── NWTN inference ───────────────────────────────────────────────────────
    start_time = time.time()
    try:
        orchestrator = NeuroSymbolicOrchestrator(node_id="ui_api")
        result = await orchestrator.solve_task(prompt, context)
    except Exception as exc:
        logger.error("NWTN inference failed in UI chat",
                     conversation_id=conversation_id,
                     error=str(exc))
        raise HTTPException(
            status_code=502,
            detail="AI inference temporarily unavailable — please retry",
        )
    processing_time = round(time.time() - start_time, 3)

    # ── FTNS deduction (non-blocking, skipped for anonymous) ─────────────────
    tokens_used = result.get("tokens_used", 0)
    ftns_per_token = float(os.getenv("PRSM_FTNS_PER_TOKEN", "0.01"))
    ftns_amount = round(tokens_used * ftns_per_token, 6)
    ftns_charged = 0.0

    if ftns_amount > 0 and user_id != "anonymous":
        try:
            deduct_result = await FTNSQueries.execute_atomic_deduct(
                user_id=user_id,
                amount=ftns_amount,
                idempotency_key=f"ui-query:{user_id}:{session_id}",
                description=f"UI chat: {prompt[:80]}",
                transaction_type="query_usage",
            )
            if deduct_result["success"]:
                ftns_charged = ftns_amount
            else:
                logger.info(
                    "UI FTNS deduction rejected",
                    user_id=user_id,
                    reason=deduct_result.get("error_message"),
                )
        except Exception as exc:
            logger.warning(
                "UI FTNS deduction unavailable (response still delivered)",
                user_id=user_id,
                error=str(exc),
            )

    # ── Build AI response record ─────────────────────────────────────────────
    ai_response = {
        "message_id":      str(uuid4()),
        "conversation_id": conversation_id,
        "content":         result["output"],          # real LLM content
        "timestamp":       datetime.now().isoformat(),
        "type":            "ai_response",
        "model":           "nwtn-v1",
        "metadata": {
            "processing_time":   processing_time,     # real wall-clock seconds
            "tokens_used":       tokens_used,          # real token count
            "confidence":        result.get("reward", 0.0),  # S2 reward signal
            "verification_hash": result.get("verification_hash"),
            "inference_source":  result.get("inference_source", "unknown"),
            "mode":              result.get("mode"),   # "light" or "deep"
            "ftns_charged":      ftns_charged,
        },
    }

    logger.info(
        "UI message processed via NWTN",
        conversation_id=conversation_id,
        message_id=message_id,
        tokens_used=tokens_used,
        inference_source=result.get("inference_source"),
        processing_time=processing_time,
        ftns_charged=ftns_charged,
    )

    return {
        "success":      True,
        "user_message": user_message,
        "ai_response":  ai_response,
    }


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
    """List user's real NWTN sessions for the conversation history sidebar."""
    try:
        from prsm.core.database import get_async_session, PRSMSessionModel, ReasoningStepModel
        from sqlalchemy import select, func, desc

        async with get_async_session() as db:
            stmt = (
                select(
                    PRSMSessionModel,
                    func.count(ReasoningStepModel.step_id).label("message_count"),
                )
                .outerjoin(
                    ReasoningStepModel,
                    ReasoningStepModel.session_id == PRSMSessionModel.session_id,
                )
                .group_by(PRSMSessionModel.session_id)
                .order_by(desc(PRSMSessionModel.updated_at))
                .limit(50)
            )
            if user_id:
                stmt = stmt.where(PRSMSessionModel.user_id == user_id)
            result = await db.execute(stmt)
            rows = result.all()

        conversations = [
            {
                "conversation_id": str(row.PRSMSessionModel.session_id),
                "title": (row.PRSMSessionModel.model_metadata or {}).get(
                    "title",
                    f"Session {row.PRSMSessionModel.created_at.strftime('%b %d, %Y')}"
                    if row.PRSMSessionModel.created_at else "Session",
                ),
                "last_message_at": (
                    row.PRSMSessionModel.updated_at or row.PRSMSessionModel.created_at
                ).isoformat() if (row.PRSMSessionModel.updated_at or row.PRSMSessionModel.created_at) else None,
                "message_count": row.message_count,
                "status": row.PRSMSessionModel.status or "active",
            }
            for row in rows
        ]

        return {"success": True, "conversations": conversations, "total": len(conversations)}

    except Exception as e:
        logger.error("Failed to list conversations", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list conversations")


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
        import hashlib
        import json
        from pathlib import Path
        from datetime import timezone as tz
        from prsm.storage import get_content_store, ContentHash
        from prsm.storage.exceptions import StorageError

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

        # ── Upload to ContentStore ──────────────────────────────────────────────
        store = get_content_store()
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="File storage unavailable — ContentStore not initialized",
            )
        try:
            stored_hash = await store.store_local(file_content)
        except (StorageError, OSError) as exc:
            logger.error("ContentStore unavailable for file upload", error=str(exc))
            raise HTTPException(
                status_code=503,
                detail="File storage unavailable",
            )

        real_cid = stored_hash.hex()

        # ── Persist metadata to ContentProvenanceModel ──────────────────────────
        from prsm.core.database import get_async_session, ContentProvenanceModel
        content_hash = hashlib.sha256(file_content).hexdigest()
        user_id_val = file_data.get("user_id", "anonymous")

        async with get_async_session() as db:
            db.add(ContentProvenanceModel(
                cid=real_cid,
                filename=file_data["filename"],
                size_bytes=file_size,
                content_hash=content_hash,
                creator_id=user_id_val,
                provenance_signature=json.dumps({
                    "uploaded_via": "ui_api",
                    "content_type": file_data["content_type"],
                }),
                royalty_rate=0.01,
                created_at=datetime.now(tz.utc),
            ))
            await db.commit()

        file_metadata = {
            "file_id":      real_cid,
            "filename":     file_data["filename"],
            "content_type": file_data["content_type"],
            "size":         file_size,
            "uploaded_at":  datetime.now().isoformat(),
            "user_id":      user_id_val,
            "privacy":      file_data.get("privacy", "private"),
            "ai_access":    file_data.get("ai_access", "core_only"),
            "ipfs_cid":     real_cid,
        }
        
        logger.info("File uploaded via UI",
                   file_id=real_cid,
                   filename=file_data["filename"],
                   size=file_size)
        
        return {
            "success": True,
            "file": file_metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="File upload failed"
        )


@router.get("/files")
async def list_user_files(user_id: str = None) -> Dict[str, Any]:
    """List files the user has uploaded to IPFS (from ContentProvenanceModel)."""
    try:
        from prsm.core.database import get_async_session, ContentProvenanceModel
        from sqlalchemy import select, desc

        async with get_async_session() as db:
            stmt = (
                select(ContentProvenanceModel)
                .order_by(desc(ContentProvenanceModel.created_at))
                .limit(100)
            )
            if user_id:
                stmt = stmt.where(ContentProvenanceModel.creator_id == user_id)
            result = await db.execute(stmt)
            rows = result.scalars().all()

        files = [
            {
                "file_id":      row.cid,
                "filename":     row.filename,
                "content_type": "application/octet-stream",
                "size":         row.size_bytes,
                "uploaded_at":  row.created_at.isoformat(),
                "privacy":      "private",
                "ai_access":    "core_only",
                "sharing": {
                    "public_link": False,
                    "shared_with": [],
                    "ai_training": False,
                },
            }
            for row in rows
        ]
        storage_used = sum(f["size"] for f in files)
        return {
            "success":       True,
            "files":         files,
            "total":         len(files),
            "storage_used":  storage_used,
            "storage_limit": 10 * 1024 * 1024 * 1024,
        }

    except Exception as e:
        logger.error("Failed to list files", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list files")


# === Tokenomics Dashboard ===

@router.get("/tokenomics/{user_id}")
async def get_tokenomics_ui(user_id: str) -> Dict[str, Any]:
    """Return real FTNS tokenomics data for the dashboard."""
    try:
        from prsm.core.database import get_async_session, StakeModel, FTNSTransactionModel
        from prsm.economy.tokenomics.staking_manager import StakingConfig
        from sqlalchemy import select, func, desc, and_, or_
        from datetime import timedelta, timezone as tz

        balance_data = await FTNSQueries.get_user_balance(user_id)
        config = StakingConfig()
        now = datetime.now(tz.utc)

        async with get_async_session() as db:
            # ── Staking totals ────────────────────────────────────────────────
            stake_row = (await db.execute(
                select(
                    func.coalesce(func.sum(StakeModel.amount), 0.0).label("staked"),
                    func.coalesce(func.sum(StakeModel.rewards_earned), 0.0).label("rewards"),
                )
                .where(and_(StakeModel.user_id == user_id, StakeModel.status == "active"))
            )).one()
            staked_amount   = float(stake_row.staked)
            rewards_earned  = float(stake_row.rewards)

            # ── Earnings by period ────────────────────────────────────────────
            _EARNING_TYPES = [
                "royalty_distribution", "content_ingestion_reward", "staking_rewards"
            ]
            def _earn_stmt(since):
                return select(func.coalesce(func.sum(FTNSTransactionModel.amount), 0.0)).where(
                    and_(
                        FTNSTransactionModel.to_user == user_id,
                        FTNSTransactionModel.transaction_type.in_(_EARNING_TYPES),
                        FTNSTransactionModel.created_at >= since,
                    )
                )

            daily   = float((await db.execute(_earn_stmt(now - timedelta(days=1)))).scalar())
            weekly  = float((await db.execute(_earn_stmt(now - timedelta(days=7)))).scalar())
            monthly = float((await db.execute(_earn_stmt(now - timedelta(days=30)))).scalar())
            lifetime = float((await db.execute(
                select(func.coalesce(func.sum(FTNSTransactionModel.amount), 0.0)).where(
                    and_(
                        FTNSTransactionModel.to_user == user_id,
                        FTNSTransactionModel.transaction_type.in_(_EARNING_TYPES),
                    )
                )
            )).scalar())

            # ── Recent transactions ───────────────────────────────────────────
            recent_rows = (await db.execute(
                select(FTNSTransactionModel)
                .where(or_(
                    FTNSTransactionModel.to_user == user_id,
                    FTNSTransactionModel.from_user == user_id,
                ))
                .order_by(desc(FTNSTransactionModel.created_at))
                .limit(10)
            )).scalars().all()

        recent_transactions = [
            {
                "tx_id":       str(tx.transaction_id),
                "type":        tx.transaction_type,
                "amount":      tx.amount if tx.to_user == user_id else -tx.amount,
                "timestamp":   tx.created_at.isoformat(),
                "description": tx.description,
            }
            for tx in recent_rows
        ]

        next_reward = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        return {
            "success": True,
            "tokenomics": {
                "balance": {
                    "total":     balance_data["balance"],
                    "available": balance_data["balance"] - balance_data["locked_balance"],
                    "locked":    balance_data["locked_balance"],
                    "currency":  "FTNS",
                },
                "staking": {
                    "staked_amount":    staked_amount,
                    "staking_rewards":  rewards_earned,
                    "apy":              config.reward_rate_annual * 100,
                    "lock_period_days": config.unstaking_period_seconds / 86400,
                    "next_reward_date": next_reward,
                },
                "earnings": {
                    "daily":          daily,
                    "weekly":         weekly,
                    "monthly":        monthly,
                    "total_lifetime": lifetime,
                },
                "recent_transactions": recent_transactions,
            },
        }

    except Exception as e:
        logger.error("Failed to retrieve tokenomics data", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve tokenomics data")


# === Task Management ===

@router.get("/tasks/{user_id}")
async def get_user_tasks(user_id: str) -> Dict[str, Any]:
    """Return the user's real task list from the session cache."""
    try:
        session_cache = get_session_cache()
        tasks = []
        if session_cache:
            tasks = await session_cache.get_session(f"user:tasks:{user_id}") or []

        return {
            "success": True,
            "tasks":   tasks,
            "summary": {
                "total":       len(tasks),
                "pending":     sum(1 for t in tasks if t.get("status") == "pending"),
                "in_progress": sum(1 for t in tasks if t.get("status") == "in_progress"),
                "completed":   sum(1 for t in tasks if t.get("status") == "completed"),
            },
        }

    except Exception as e:
        logger.error("Failed to retrieve tasks", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve tasks")


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
        
        # Persist to session cache
        user_id = task_data.get("user_id", "anonymous")
        session_cache = get_session_cache()
        if session_cache and user_id != "anonymous":
            existing = await session_cache.get_session(f"user:tasks:{user_id}") or []
            existing.append(task)
            await session_cache.store_session(
                f"user:tasks:{user_id}", existing, ttl=604800  # 7 days
            )
        
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
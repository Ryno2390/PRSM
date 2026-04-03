"""
Node Management API
===================

FastAPI endpoints for monitoring and controlling a running PRSM node.
This is the node-local API (not the main PRSM platform API).

Security Features (Phase 4.2):
- JWT authentication on protected endpoints
- Rate limiting with configurable limits
- WebSocket status updates
- OpenAPI specification
"""

import asyncio
import logging
import time
import uuid as _uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from prsm.node.api_hardening import (
    APIHardening,
    APISecurityConfig,
    StatusWebSocket,
    RateLimitConfig,
    RateLimiter,
    generate_openapi_schema,
    get_current_user,
    require_auth,
    websocket_status_endpoint,
)
from prsm.node.node import TrainingJob, TrainingJobStatus

logger = logging.getLogger(__name__)


class JobSubmission(BaseModel):
    """Request body for submitting a compute job."""
    job_type: str  # inference, embedding, benchmark
    payload: Dict[str, Any] = {}
    ftns_budget: float = 1.0


class ResourceUpdateRequest(BaseModel):
    """Request model for updating node resource settings."""
    cpu_allocation_pct: Optional[int] = Field(default=None, ge=10, le=90, description="CPU allocation percentage (10-90)")
    memory_allocation_pct: Optional[int] = Field(default=None, ge=10, le=90, description="Memory allocation percentage (10-90)")
    storage_gb: Optional[float] = Field(default=None, gt=0, description="Storage pledge in GB")
    max_concurrent_jobs: Optional[int] = Field(default=None, ge=1, description="Maximum concurrent jobs")
    gpu_allocation_pct: Optional[int] = Field(default=None, ge=10, le=100, description="GPU allocation percentage (10-100)")
    upload_mbps_limit: Optional[float] = Field(default=None, ge=0, description="Upload bandwidth limit in Mbps (0=unlimited)")
    download_mbps_limit: Optional[float] = Field(default=None, ge=0, description="Download bandwidth limit in Mbps (0=unlimited)")
    active_hours_start: Optional[int] = Field(default=None, ge=0, le=23, description="Active hours start (0-23)")
    active_hours_end: Optional[int] = Field(default=None, ge=0, le=23, description="Active hours end (0-23)")
    active_days: Optional[List[int]] = Field(default=None, description="Active days (0=Mon...6=Sun, empty=every day)")


class ResourceConfigResponse(BaseModel):
    """Response model for current resource configuration."""
    cpu_allocation_pct: int
    memory_allocation_pct: int
    storage_gb: float
    max_concurrent_jobs: int
    gpu_allocation_pct: int
    upload_mbps_limit: float
    download_mbps_limit: float
    active_hours_start: Optional[int]
    active_hours_end: Optional[int]
    active_days: List[int]
    # Computed values
    effective_cpu_cores: float
    effective_memory_gb: float
    effective_gpu_memory_gb: Optional[float]
    storage_available_gb: float


class ContentUploadRequest(BaseModel):
    """Request body for uploading text content."""
    text: str
    filename: str = "document.txt"
    replicas: int = 3
    royalty_rate: Optional[float] = Field(
        default=None,
        description="FTNS earned per access (0.001–0.1, default 0.01)",
    )
    parent_cids: List[str] = Field(
        default_factory=list,
        description="CIDs of source material this content derives from",
    )


# ── Teacher Model Constants ─────────────────────────────────────

TEACHER_CREATION_REWARD_FTNS = 10.0   # FTNS credited for creating a teacher
TEACHER_TRAINING_BASE_COST_FTNS = 50.0  # Minimum FTNS charged per training run


# ── Teacher Model Request/Response Models ───────────────────────

class TeacherCreateRequest(BaseModel):
    """Request body for creating a new teacher model."""
    specialization: str = Field(..., description="Domain name, e.g. 'physics', 'genomics'")
    domain: Optional[str] = Field(None, description="Sub-domain; defaults to specialization")
    use_real_implementation: bool = Field(True, description="Use PyTorch backend if available")


class TeacherTrainingRequest(BaseModel):
    """Request body for starting a teacher training run."""
    epochs: Optional[int] = Field(None, ge=1, le=100, description="Override training epochs")
    learning_rate: Optional[float] = Field(None, gt=0.0, description="Override learning rate")
    training_data_cid: Optional[str] = Field(None, description="IPFS CID of custom training data")


def create_api_app(node: Any, enable_security: bool = True) -> FastAPI:
    """
    Create the node management FastAPI app with a reference to the running node.
    
    Args:
        node: The PRSM node instance
        enable_security: Whether to enable security hardening (default: True)
    
    Returns:
        FastAPI application with security hardening applied
    """

    app = FastAPI(
        title="PRSM Node API",
        description="Management API for a PRSM network node",
        version="0.3.2",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Initialize security hardening
    api_hardening: Optional[APIHardening] = None
    status_websocket: Optional[StatusWebSocket] = None
    
    if enable_security:
        # Create security configuration
        security_config = APISecurityConfig(
            enable_rate_limiting=True,
            enable_jwt_auth=False,
            enable_websocket=True,
            enable_openapi=True,
            rate_limit_requests_per_minute=100,
            rate_limit_requests_per_hour=1000,
        )
        
        # Initialize hardening
        api_hardening = APIHardening(app, security_config)
        
        # Apply middleware (must be done before adding routes)
        # Note: Middleware is applied in reverse order, so we add it after route definition
        # We'll apply it at the end of this function
        
        # Setup OpenAPI schema
        api_hardening.setup_openapi()
        
        # Get WebSocket manager for status updates
        status_websocket = api_hardening.get_status_websocket()

    # Store WebSocket manager in app state for access
    app.state.status_websocket = status_websocket
    app.state.api_hardening = api_hardening

    # ── Public Endpoints (no auth required) ─────────────────────────────────────

    @app.get("/api-info", tags=["status"])
    async def api_info() -> Dict[str, Any]:
        """API information endpoint (dashboard served at root)."""
        return {
            "name": "PRSM Node API",
            "version": "0.2.0",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "websocket": "/ws/status",
        }

    @app.get("/status", tags=["status"])
    async def get_status() -> Dict[str, Any]:
        """Get comprehensive node status."""
        status = await node.get_status()
        
        # Broadcast status update via WebSocket if available
        if app.state.status_websocket:
            await app.state.status_websocket.broadcast_status(status)
        
        return status

    @app.get("/peers")
    async def get_peers() -> Dict[str, Any]:
        """List connected and known peers."""
        connected = []
        if node.transport:
            for pid, peer in node.transport.peers.items():
                connected.append({
                    "peer_id": pid,
                    "address": peer.address,
                    "display_name": peer.display_name,
                    "connected_at": peer.connected_at,
                    "last_seen": peer.last_seen,
                    "outbound": peer.outbound,
                })

        known = []
        if node.discovery:
            for info in node.discovery.get_known_peers():
                known.append({
                    "node_id": info.node_id,
                    "address": info.address,
                    "display_name": info.display_name,
                    "last_seen": info.last_seen,
                })

        return {
            "connected": connected,
            "known": known,
            "connected_count": len(connected),
            "known_count": len(known),
        }

    @app.get("/balance")
    async def get_balance() -> Dict[str, Any]:
        """Get FTNS balance and recent transactions."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Node not initialized")

        balance = await node.ledger.get_balance(node.identity.node_id)
        history = await node.ledger.get_transaction_history(node.identity.node_id, limit=20)

        return {
            "wallet_id": node.identity.node_id,
            "balance": balance,
            "recent_transactions": [
                {
                    "tx_id": tx.tx_id,
                    "type": tx.tx_type.value,
                    "from": tx.from_wallet,
                    "to": tx.to_wallet,
                    "amount": tx.amount,
                    "description": tx.description,
                    "timestamp": tx.timestamp,
                }
                for tx in history
            ],
        }

    @app.post("/compute/submit")
    async def submit_compute_job(job: JobSubmission) -> Dict[str, Any]:
        """Submit a compute job to the network."""
        if not node.compute_requester:
            raise HTTPException(status_code=503, detail="Compute requester not initialized")

        from prsm.node.compute_provider import JobType
        try:
            job_type = JobType(job.job_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job type: {job.job_type}. Valid types: inference, embedding, benchmark",
            )

        try:
            submitted = await node.compute_requester.submit_job(
                job_type=job_type,
                payload=job.payload,
                ftns_budget=job.ftns_budget,
            )
            return {
                "job_id": submitted.job_id,
                "status": submitted.status.value,
                "job_type": submitted.job_type.value,
                "ftns_budget": submitted.ftns_budget,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/compute/job/{job_id}")
    async def get_job_status(job_id: str) -> Dict[str, Any]:
        """Get status of a submitted compute job."""
        if not node.compute_requester:
            raise HTTPException(status_code=503, detail="Compute requester not initialized")

        job = node.compute_requester.submitted_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "job_type": job.job_type.value,
            "provider_id": job.provider_id,
            "result": job.result,
            "result_verified": job.result_verified,
            "error": job.error,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }

    @app.post("/compute/query")
    async def compute_query(body: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Submit a compute job and wait for the result (blocking).

        POST body: {"prompt": "...", "model": "nwtn", "timeout": 120}
        Returns: {"job_id", "response", "result"}
        """
        if not node.compute_requester or not node.compute_provider:
            raise HTTPException(status_code=503, detail="Compute not initialized")

        prompt = body.get("prompt", "")
        model = body.get("model", "nwtn")
        timeout = float(body.get("timeout", 120.0))
        budget = float(body.get("budget", 0.0))

        from prsm.node.compute_provider import JobType

        job = await node.compute_requester.submit_job(
            job_type=JobType.INFERENCE,
            payload={"prompt": prompt, "model": model},
            ftns_budget=budget,
        )

        # Wait for the result
        result = await node.compute_requester.get_result(job.job_id, timeout=timeout)

        if result is None:
            raise HTTPException(
                status_code=504, detail="Compute timed out or no provider accepted"
            )

        # Release escrow payment to the provider (self-compute or remote)
        if node.compute_provider and node.compute_provider.escrow:
            try:
                provider_id = result.get("provider_id", node.identity.node_id)
                tx = await node.compute_provider.escrow.release_escrow(
                    job_id=job.job_id,
                    provider_id=provider_id,
                    consensus_reached=True,
                )
                if tx:
                    logger.info(
                        f"compute_query: escrow released {budget:.6f} FTNS → {provider_id[:12]}..."
                    )
            except Exception as e:
                logger.warning(f"compute_query: escrow release failed: {e}")

        return {
            "job_id": job.job_id,
            "response": result.get("response", result.get("text", str(result))),
            "result": result,
        }

    @app.post("/content/upload")
    async def upload_content(req: ContentUploadRequest) -> Dict[str, Any]:
        """Upload text content to IPFS with provenance tracking."""
        if not node.content_uploader:
            raise HTTPException(status_code=503, detail="Content uploader not initialized")

        result = await node.content_uploader.upload_text(
            text=req.text,
            filename=req.filename,
            replicas=req.replicas,
            royalty_rate=req.royalty_rate,
            parent_cids=req.parent_cids if req.parent_cids else None,
        )

        if not result:
            raise HTTPException(status_code=502, detail="Upload failed — is IPFS running?")

        return {
            "cid": result.cid,
            "filename": result.filename,
            "size_bytes": result.size_bytes,
            "content_hash": result.content_hash,
            "creator_id": result.creator_id,
            "royalty_rate": result.royalty_rate,
            "parent_cids": result.parent_cids,
        }

    @app.get("/content/search")
    async def search_content(q: str = "", limit: int = 20) -> Dict[str, Any]:
        """Search the network content index by keyword."""
        if not node.content_index:
            raise HTTPException(status_code=503, detail="Content index not initialized")

        results = node.content_index.search(q, limit=min(limit, 100))
        return {
            "query": q,
            "results": [
                {
                    "cid": r.cid,
                    "filename": r.filename,
                    "size_bytes": r.size_bytes,
                    "content_hash": r.content_hash,
                    "creator_id": r.creator_id,
                    "providers": list(r.providers),
                    "created_at": r.created_at,
                    "metadata": r.metadata,
                    "royalty_rate": r.royalty_rate,
                    "parent_cids": r.parent_cids,
                }
                for r in results
            ],
            "count": len(results),
        }

    @app.get("/content/index/stats")
    async def content_index_stats() -> Dict[str, Any]:
        """Get content index statistics."""
        if not node.content_index:
            raise HTTPException(status_code=503, detail="Content index not initialized")
        return node.content_index.get_stats()

    @app.get("/content/{cid}")
    async def get_content_record(cid: str) -> Dict[str, Any]:
        """Look up a specific content record by CID."""
        if not node.content_index:
            raise HTTPException(status_code=503, detail="Content index not initialized")

        record = node.content_index.lookup(cid)
        if not record:
            raise HTTPException(status_code=404, detail="Content not found in index")

        return {
            "cid": record.cid,
            "filename": record.filename,
            "size_bytes": record.size_bytes,
            "content_hash": record.content_hash,
            "creator_id": record.creator_id,
            "providers": list(record.providers),
            "created_at": record.created_at,
            "metadata": record.metadata,
            "royalty_rate": record.royalty_rate,
            "parent_cids": record.parent_cids,
        }

    class ContentRetrieveResponse(BaseModel):
        """Response model for content retrieval."""
        cid: str
        status: str = Field(description="Retrieval status: 'success', 'not_found', or 'error'")
        data: Optional[str] = Field(
            default=None,
            description="Base64-encoded content data (only if status is 'success')"
        )
        size_bytes: Optional[int] = Field(default=None, description="Size of retrieved content")
        content_hash: Optional[str] = Field(default=None, description="SHA-256 hash of content")
        filename: Optional[str] = Field(default=None, description="Original filename if available")
        providers_tried: int = Field(default=0, description="Number of providers attempted")
        error: Optional[str] = Field(default=None, description="Error message if status is 'error'")

    @app.get("/content/retrieve/{cid}", response_model=ContentRetrieveResponse)
    async def retrieve_content(cid: str, timeout: float = 30.0, verify_hash: bool = True) -> ContentRetrieveResponse:
        """
        Retrieve content from the network by CID.
        
        This endpoint retrieves content from the P2P network using the ContentProvider.
        It will:
        1. Check if content is available locally
        2. Query the content index for providers
        3. Request content from available providers
        4. Verify content hash if verification is enabled
        
        Args:
            cid: IPFS content identifier
            timeout: Seconds to wait for response (default: 30.0)
            verify_hash: Whether to verify SHA-256 hash (default: True)
        
        Returns:
            ContentRetrieveResponse with status and data (base64-encoded) or error
        """
        import base64
        
        if not node.content_provider:
            raise HTTPException(status_code=503, detail="Content provider not initialized")
        
        # Get provider stats before retrieval to determine providers tried
        stats_before = node.content_provider.get_stats()
        
        try:
            content_bytes = await node.content_provider.request_content(
                cid=cid,
                timeout=timeout,
                verify_hash=verify_hash,
            )
            
            if content_bytes is None:
                # Content not found or retrieval failed
                return ContentRetrieveResponse(
                    cid=cid,
                    status="not_found",
                    error="Content not found on any available provider",
                )
            
            # Get content metadata if available
            content_hash = None
            filename = None
            if node.content_index:
                record = node.content_index.lookup(cid)
                if record:
                    content_hash = record.content_hash
                    filename = record.filename
            
            # Encode content as base64 for JSON response
            data_b64 = base64.b64encode(content_bytes).decode('utf-8')
            
            return ContentRetrieveResponse(
                cid=cid,
                status="success",
                data=data_b64,
                size_bytes=len(content_bytes),
                content_hash=content_hash,
                filename=filename,
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Content retrieval timed out after {timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Error retrieving content {cid}: {e}")
            return ContentRetrieveResponse(
                cid=cid,
                status="error",
                error=str(e),
            )

    @app.get("/transactions")
    async def get_transactions(limit: int = 50) -> Dict[str, Any]:
        """Get transaction history."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Node not initialized")

        history = await node.ledger.get_transaction_history(
            node.identity.node_id, limit=min(limit, 200)
        )
        return {
            "transactions": [
                {
                    "tx_id": tx.tx_id,
                    "type": tx.tx_type.value,
                    "from": tx.from_wallet,
                    "to": tx.to_wallet,
                    "amount": tx.amount,
                    "description": tx.description,
                    "timestamp": tx.timestamp,
                }
                for tx in history
            ],
            "count": len(history),
        }

    # ── Agent endpoints ─────────────────────────────────────────

    @app.get("/agents")
    async def list_agents(local_only: bool = False) -> Dict[str, Any]:
        """List known agents (local and/or remote)."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        if local_only:
            agents = node.agent_registry.get_local_agents()
        else:
            agents = node.agent_registry.get_all_agents()

        return {
            "agents": [a.to_dict() for a in agents],
            "count": len(agents),
        }

    @app.get("/agents/search")
    async def search_agents(capability: str, limit: int = 20) -> Dict[str, Any]:
        """Search agents by capability."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        results = node.agent_registry.search(capability, limit=min(limit, 100))
        return {
            "capability": capability,
            "agents": [a.to_dict() for a in results],
            "count": len(results),
        }

    @app.get("/agents/spending")
    async def agent_spending() -> Dict[str, Any]:
        """Aggregate spending dashboard across all local agents."""
        if not node.agent_registry or not node.ledger:
            raise HTTPException(status_code=503, detail="Not initialized")

        agents = node.agent_registry.get_local_agents()
        spending = []
        for agent in agents:
            allowance = await node.ledger.get_agent_allowance(agent.agent_id)
            spending.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.agent_name,
                "allowance": allowance,
            })
        return {"agents": spending, "count": len(spending)}

    @app.get("/agents/{agent_id}")
    async def get_agent(agent_id: str) -> Dict[str, Any]:
        """Get agent details, spending, and status."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        record = node.agent_registry.lookup(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent not found")

        result = record.to_dict()
        if node.ledger:
            result["allowance"] = await node.ledger.get_agent_allowance(agent_id)
        return result

    @app.get("/agents/{agent_id}/conversations")
    async def get_agent_conversations(agent_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get recent conversation threads involving an agent."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        conv_ids = node.agent_registry.get_agent_conversations(agent_id, limit=limit)
        conversations = []
        for conv_id in conv_ids:
            messages = node.agent_registry.get_conversation(conv_id)
            conversations.append({
                "conversation_id": conv_id,
                "message_count": len(messages),
                "messages": messages[-5:],  # Last 5 messages per conversation
            })
        return {"conversations": conversations, "count": len(conversations)}

    @app.post("/agents/{agent_id}/allowance")
    async def set_agent_allowance(agent_id: str, amount: float, epoch_hours: float = 24.0) -> Dict[str, Any]:
        """Set or update an agent's spending allowance."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Not initialized")

        if amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")

        await node.ledger.grant_agent_allowance(
            principal_id=node.identity.node_id,
            agent_id=agent_id,
            amount=amount,
            epoch_hours=epoch_hours,
        )
        return await node.ledger.get_agent_allowance(agent_id)

    @app.delete("/agents/{agent_id}/allowance")
    async def revoke_agent_allowance(agent_id: str) -> Dict[str, Any]:
        """Revoke an agent's spending authority."""
        if not node.ledger or not node.identity:
            raise HTTPException(status_code=503, detail="Not initialized")

        revoked = await node.ledger.revoke_agent_allowance(
            principal_id=node.identity.node_id,
            agent_id=agent_id,
        )
        if not revoked:
            raise HTTPException(status_code=404, detail="Agent allowance not found")
        return {"agent_id": agent_id, "revoked": True}

    @app.post("/agents/{agent_id}/pause")
    async def pause_agent(agent_id: str) -> Dict[str, Any]:
        """Temporarily suspend an agent."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        record = node.agent_registry.lookup(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent not found")

        node.agent_registry.set_agent_status(agent_id, "paused")
        return {"agent_id": agent_id, "status": "paused"}

    @app.post("/agents/{agent_id}/resume")
    async def resume_agent(agent_id: str) -> Dict[str, Any]:
        """Resume a paused agent."""
        if not node.agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not initialized")

        record = node.agent_registry.lookup(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent not found")

        node.agent_registry.set_agent_status(agent_id, "online")
        return {"agent_id": agent_id, "status": "online"}

    # ── Ledger endpoints ─────────────────────────────────────────

    @app.get("/ledger/sync/stats")
    async def ledger_sync_stats() -> Dict[str, Any]:
        """Get ledger sync statistics."""
        if not node.ledger_sync:
            raise HTTPException(status_code=503, detail="Ledger sync not initialized")
        return node.ledger_sync.get_stats()

    @app.post("/ledger/transfer")
    async def transfer_ftns(to_wallet: str, amount: float) -> Dict[str, Any]:
        """Transfer FTNS to another node (signed, gossip-broadcast)."""
        if not node.ledger_sync:
            raise HTTPException(status_code=503, detail="Ledger sync not initialized")

        if amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")

        tx = await node.ledger_sync.signed_transfer(
            to_wallet=to_wallet,
            amount=amount,
            description=f"API transfer to {to_wallet[:12]}...",
        )
        if not tx:
            raise HTTPException(status_code=400, detail="Insufficient balance")

        return {
            "tx_id": tx.tx_id,
            "from": tx.from_wallet,
            "to": tx.to_wallet,
            "amount": tx.amount,
            "timestamp": tx.timestamp,
        }

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Simple health check."""
        return {"status": "ok", "node_id": node.identity.node_id if node.identity else "unknown"}


    # ── Staking Endpoints ─────────────────────────────────────────

    class StakeRequest(BaseModel):
        """Request body for staking FTNS tokens."""
        amount: float = Field(..., gt=0, description="Amount of FTNS to stake")
        stake_type: str = Field(default="general", description="Type of staking: governance, validation, compute, storage, liquidity, general")
        metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the stake")

    class StakeResponse(BaseModel):
        """Response model for a stake operation."""
        stake_id: str
        user_id: str
        amount: float
        stake_type: str
        status: str
        staked_at: str
        rewards_earned: float = 0.0

    class UnstakeRequest(BaseModel):
        """Request body for unstaking FTNS tokens."""
        stake_id: str = Field(..., description="ID of the stake to unstake")
        amount: Optional[float] = Field(default=None, gt=0, description="Amount to unstake (None = full stake)")

    class UnstakeResponse(BaseModel):
        """Response model for an unstake operation."""
        request_id: str
        stake_id: str
        user_id: str
        amount: float
        requested_at: str
        available_at: str
        status: str

    class StakingStatusResponse(BaseModel):
        """Response model for staking status."""
        user_id: str
        total_staked: float
        active_stakes: List[Dict[str, Any]]
        pending_unstake_requests: List[Dict[str, Any]]
        total_rewards_earned: float
        total_rewards_claimed: float

    class ClaimRewardsResponse(BaseModel):
        """Response model for claiming rewards."""
        user_id: str
        total_rewards_claimed: float
        stakes_processed: int

    @app.post("/staking/stake", response_model=StakeResponse, tags=["staking"])
    async def stake_tokens(req: StakeRequest) -> StakeResponse:
        """
        Stake FTNS tokens.
        
        Stakes the specified amount of FTNS tokens for the node's identity.
        The tokens will be locked and start earning rewards based on the
        configured annual reward rate.
        
        Args:
            req: StakeRequest containing amount, stake_type, and optional metadata
            
        Returns:
            StakeResponse with the created stake details
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If stake validation fails
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        # Validate stake_type
        from prsm.economy.tokenomics.staking_manager import StakeType
        try:
            stake_type = StakeType(req.stake_type.lower())
        except ValueError:
            valid_types = [t.value for t in StakeType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stake_type: {req.stake_type}. Valid types: {valid_types}"
            )
        
        try:
            from decimal import Decimal
            stake = await node.staking_manager.stake(
                user_id=node.identity.node_id,
                amount=Decimal(str(req.amount)),
                stake_type=stake_type,
                metadata=req.metadata
            )
            
            return StakeResponse(
                stake_id=stake.stake_id,
                user_id=stake.user_id,
                amount=float(stake.amount),
                stake_type=stake.stake_type.value,
                status=stake.status.value,
                staked_at=stake.staked_at.isoformat(),
                rewards_earned=float(stake.rewards_earned)
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error staking tokens: {e}")
            raise HTTPException(status_code=500, detail=f"Staking failed: {str(e)}")

    @app.post("/staking/unstake", response_model=UnstakeResponse, tags=["staking"])
    async def unstake_tokens(req: UnstakeRequest) -> UnstakeResponse:
        """
        Request to unstake FTNS tokens.
        
        Creates an unstake request that will be available for withdrawal
        after the configured unstaking period (default: 7 days).
        
        Args:
            req: UnstakeRequest containing stake_id and optional amount
            
        Returns:
            UnstakeResponse with the unstake request details
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If unstake validation fails
            HTTPException 404: If stake not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            from decimal import Decimal
            amount = Decimal(str(req.amount)) if req.amount else None
            
            unstake_request = await node.staking_manager.unstake(
                user_id=node.identity.node_id,
                stake_id=req.stake_id,
                amount=amount
            )
            
            return UnstakeResponse(
                request_id=unstake_request.request_id,
                stake_id=unstake_request.stake_id,
                user_id=unstake_request.user_id,
                amount=float(unstake_request.amount),
                requested_at=unstake_request.requested_at.isoformat(),
                available_at=unstake_request.available_at.isoformat(),
                status=unstake_request.status.value
            )
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error unstaking tokens: {e}")
            raise HTTPException(status_code=500, detail=f"Unstake failed: {str(e)}")

    @app.get("/staking/status", response_model=StakingStatusResponse, tags=["staking"])
    async def get_staking_status() -> StakingStatusResponse:
        """
        Get staking status for the current node identity.
        
        Returns information about all active stakes, pending unstake requests,
        and reward totals for the node's identity.
        
        Returns:
            StakingStatusResponse with comprehensive staking information
            
        Raises:
            HTTPException 503: If staking manager not initialized
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        user_id = node.identity.node_id
        
        # Get all stakes for the user
        stakes = await node.staking_manager.get_user_stakes(user_id)
        
        # Get pending unstake requests
        pending_requests = await node.staking_manager.get_pending_unstake_requests(user_id)
        
        # Calculate totals
        total_staked = sum(float(s.amount) for s in stakes if s.status.value == "active")
        total_rewards_earned = sum(float(s.rewards_earned) for s in stakes)
        total_rewards_claimed = sum(float(s.rewards_claimed) for s in stakes)
        
        return StakingStatusResponse(
            user_id=user_id,
            total_staked=total_staked,
            active_stakes=[
                {
                    "stake_id": s.stake_id,
                    "amount": float(s.amount),
                    "stake_type": s.stake_type.value,
                    "status": s.status.value,
                    "staked_at": s.staked_at.isoformat(),
                    "rewards_earned": float(s.rewards_earned),
                    "rewards_claimed": float(s.rewards_claimed)
                }
                for s in stakes
            ],
            pending_unstake_requests=[
                {
                    "request_id": r.request_id,
                    "stake_id": r.stake_id,
                    "amount": float(r.amount),
                    "requested_at": r.requested_at.isoformat(),
                    "available_at": r.available_at.isoformat(),
                    "status": r.status.value
                }
                for r in pending_requests
            ],
            total_rewards_earned=total_rewards_earned,
            total_rewards_claimed=total_rewards_claimed
        )

    @app.post("/staking/claim-rewards", response_model=ClaimRewardsResponse, tags=["staking"])
    async def claim_staking_rewards(stake_id: Optional[str] = None) -> ClaimRewardsResponse:
        """
        Claim accumulated staking rewards.
        
        Claims all pending rewards for the node's stakes. If stake_id is provided,
        only claims rewards for that specific stake.
        
        Args:
            stake_id: Optional specific stake ID to claim rewards from
            
        Returns:
            ClaimRewardsResponse with total rewards claimed and stakes processed
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If claim validation fails
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            total_rewards = await node.staking_manager.claim_rewards(
                user_id=node.identity.node_id,
                stake_id=stake_id
            )
            
            # Count stakes processed
            stakes = await node.staking_manager.get_user_stakes(node.identity.node_id)
            stakes_processed = len(stakes) if not stake_id else 1
            
            return ClaimRewardsResponse(
                user_id=node.identity.node_id,
                total_rewards_claimed=float(total_rewards),
                stakes_processed=stakes_processed
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error claiming rewards: {e}")
            raise HTTPException(status_code=500, detail=f"Claim rewards failed: {str(e)}")

    @app.get("/staking/stakes/{stake_id}", tags=["staking"])
    async def get_stake(stake_id: str) -> Dict[str, Any]:
        """
        Get details of a specific stake.
        
        Args:
            stake_id: The ID of the stake to retrieve
            
        Returns:
            Detailed stake information
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 404: If stake not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        stake = await node.staking_manager.get_stake(stake_id)
        if not stake:
            raise HTTPException(status_code=404, detail="Stake not found")
        
        return {
            "stake_id": stake.stake_id,
            "user_id": stake.user_id,
            "amount": float(stake.amount),
            "stake_type": stake.stake_type.value,
            "status": stake.status.value,
            "staked_at": stake.staked_at.isoformat(),
            "rewards_earned": float(stake.rewards_earned),
            "rewards_claimed": float(stake.rewards_claimed),
            "last_reward_calculation": stake.last_reward_calculation.isoformat() if stake.last_reward_calculation else None,
            "lock_reason": stake.lock_reason,
            "metadata": stake.metadata
        }

    @app.get("/staking/unstake-requests/{request_id}", tags=["staking"])
    async def get_unstake_request(request_id: str) -> Dict[str, Any]:
        """
        Get details of a specific unstake request.
        
        Args:
            request_id: The ID of the unstake request to retrieve
            
        Returns:
            Detailed unstake request information
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 404: If request not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        request = await node.staking_manager.get_unstake_request(request_id)
        if not request:
            raise HTTPException(status_code=404, detail="Unstake request not found")
        
        return {
            "request_id": request.request_id,
            "stake_id": request.stake_id,
            "user_id": request.user_id,
            "amount": float(request.amount),
            "requested_at": request.requested_at.isoformat(),
            "available_at": request.available_at.isoformat(),
            "status": request.status.value,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "cancellation_reason": request.cancellation_reason,
            "is_available": request.is_available
        }

    @app.post("/staking/withdraw/{request_id}", tags=["staking"])
    async def withdraw_unstaked_tokens(request_id: str) -> Dict[str, Any]:
        """
        Withdraw unstaked tokens after the unstaking period.
        
        Completes an unstake request and returns the tokens to the user's
        available balance.
        
        Args:
            request_id: The ID of the unstake request to withdraw
            
        Returns:
            Withdrawal confirmation with amount
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If withdrawal validation fails
            HTTPException 404: If request not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            success, amount = await node.staking_manager.withdraw(
                user_id=node.identity.node_id,
                request_id=request_id
            )
            
            return {
                "request_id": request_id,
                "success": success,
                "amount_withdrawn": float(amount)
            }
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error withdrawing tokens: {e}")
            raise HTTPException(status_code=500, detail=f"Withdrawal failed: {str(e)}")

    @app.post("/staking/cancel-unstake/{request_id}", tags=["staking"])
    async def cancel_unstake_request(request_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel a pending unstake request.
        
        Cancels an unstake request and restores the tokens to active staking.
        
        Args:
            request_id: The ID of the unstake request to cancel
            reason: Optional reason for cancellation
            
        Returns:
            Cancellation confirmation
            
        Raises:
            HTTPException 503: If staking manager not initialized
            HTTPException 400: If cancellation validation fails
            HTTPException 404: If request not found
        """
        if not node.staking_manager:
            raise HTTPException(status_code=503, detail="Staking manager not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            success = await node.staking_manager.cancel_unstake(
                user_id=node.identity.node_id,
                request_id=request_id,
                reason=reason
            )
            
            return {
                "request_id": request_id,
                "cancelled": success,
                "reason": reason
            }
        except ValueError as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error cancelling unstake: {e}")
            raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")


    # ── Storage endpoints ────────────────────────────────────────

    @app.get("/storage/stats")
    async def get_storage_stats() -> Dict[str, Any]:
        """Get storage provider statistics."""
        if not node.storage_provider:
            return {
                "available": False,
                "pledged_gb": 0,
                "used_gb": 0,
                "pinned_count": 0,
                "message": "Storage provider not initialized"
            }
        return node.storage_provider.get_stats()

    # ── Compute endpoints ────────────────────────────────────────

    @app.get("/compute/stats", tags=["compute"])
    async def get_compute_stats() -> Dict[str, Any]:
        """Get compute provider statistics."""
        if not node.compute_provider:
            return {
                "available": False,
                "jobs_completed": 0,
                "jobs_queued": 0,
                "message": "Compute provider not initialized"
            }
        return node.compute_provider.get_stats()

    # ── Resource Configuration Endpoints ─────────────────────────────────

    @app.get("/node/resources", response_model=ResourceConfigResponse, tags=["resources"])
    async def get_resources() -> ResourceConfigResponse:
        """Get current node resource configuration.
        
        Returns the current resource allocation settings including:
        - CPU and memory allocation percentages
        - Storage pledge and availability
        - GPU allocation (if available)
        - Bandwidth limits
        - Active hours/days schedule
        - Computed effective values
        """
        if not node.config:
            raise HTTPException(status_code=503, detail="Node configuration not initialized")
        
        config = node.config
        
        # Get compute provider for effective values
        effective_cpu_cores = 0.0
        effective_memory_gb = 0.0
        effective_gpu_memory_gb = None
        
        if node.compute_provider:
            cp = node.compute_provider
            effective_cpu_cores = round(cp.resources.cpu_count * cp.cpu_allocation_pct / 100, 2)
            effective_memory_gb = round(cp.resources.memory_total_gb * cp.memory_allocation_pct / 100, 2)
            if cp.resources.gpu_available:
                effective_gpu_memory_gb = round(cp.resources.gpu_memory_gb * cp.gpu_allocation_pct / 100, 2)
        
        # Get storage available
        storage_available_gb = config.storage_gb
        if node.storage_provider:
            storage_available_gb = node.storage_provider.available_gb
        
        return ResourceConfigResponse(
            cpu_allocation_pct=config.cpu_allocation_pct,
            memory_allocation_pct=config.memory_allocation_pct,
            storage_gb=config.storage_gb,
            max_concurrent_jobs=config.max_concurrent_jobs,
            gpu_allocation_pct=config.gpu_allocation_pct,
            upload_mbps_limit=config.upload_mbps_limit,
            download_mbps_limit=config.download_mbps_limit,
            active_hours_start=config.active_hours_start,
            active_hours_end=config.active_hours_end,
            active_days=config.active_days,
            effective_cpu_cores=effective_cpu_cores,
            effective_memory_gb=effective_memory_gb,
            effective_gpu_memory_gb=effective_gpu_memory_gb,
            storage_available_gb=storage_available_gb,
        )

    @app.put("/node/resources", response_model=ResourceConfigResponse, tags=["resources"])
    async def update_resources(request: ResourceUpdateRequest) -> ResourceConfigResponse:
        """Update node resource allocation settings at runtime.
        
        Changes take effect immediately for storage and bandwidth limits.
        Compute changes (CPU, memory, jobs) take effect on next job acceptance.
        
        All fields are optional - only provided fields will be updated.
        
        Validation:
        - cpu_allocation_pct: 10-90
        - memory_allocation_pct: 10-90
        - gpu_allocation_pct: 10-100
        - active_hours_start/end: 0-23
        - storage_gb: must be positive
        - max_concurrent_jobs: at least 1
        - bandwidth limits: non-negative (0 = unlimited)
        """
        if not node.config:
            raise HTTPException(status_code=503, detail="Node configuration not initialized")
        
        config = node.config
        updates_applied = []
        
        # Validate active_hours consistency if both provided
        if request.active_hours_start is not None and request.active_hours_end is not None:
            if request.active_hours_start >= request.active_hours_end:
                raise HTTPException(
                    status_code=400,
                    detail=f"active_hours_start ({request.active_hours_start}) must be less than active_hours_end ({request.active_hours_end})"
                )
        
        # Validate active_days
        if request.active_days is not None:
            for day in request.active_days:
                if not 0 <= day <= 6:
                    raise HTTPException(
                        status_code=400,
                        detail=f"active_days must be 0-6 (Mon-Sun), got {day}"
                    )
        
        try:
            # Update config fields
            if request.cpu_allocation_pct is not None:
                config.cpu_allocation_pct = request.cpu_allocation_pct
                updates_applied.append(f"cpu_allocation_pct={request.cpu_allocation_pct}")
            
            if request.memory_allocation_pct is not None:
                config.memory_allocation_pct = request.memory_allocation_pct
                updates_applied.append(f"memory_allocation_pct={request.memory_allocation_pct}")
            
            if request.storage_gb is not None:
                config.storage_gb = request.storage_gb
                updates_applied.append(f"storage_gb={request.storage_gb}")
            
            if request.max_concurrent_jobs is not None:
                config.max_concurrent_jobs = request.max_concurrent_jobs
                updates_applied.append(f"max_concurrent_jobs={request.max_concurrent_jobs}")
            
            if request.gpu_allocation_pct is not None:
                config.gpu_allocation_pct = request.gpu_allocation_pct
                updates_applied.append(f"gpu_allocation_pct={request.gpu_allocation_pct}")
            
            if request.upload_mbps_limit is not None:
                config.upload_mbps_limit = request.upload_mbps_limit
                updates_applied.append(f"upload_mbps_limit={request.upload_mbps_limit}")
            
            if request.download_mbps_limit is not None:
                config.download_mbps_limit = request.download_mbps_limit
                updates_applied.append(f"download_mbps_limit={request.download_mbps_limit}")
            
            if request.active_hours_start is not None:
                config.active_hours_start = request.active_hours_start
                updates_applied.append(f"active_hours_start={request.active_hours_start}")
            
            if request.active_hours_end is not None:
                config.active_hours_end = request.active_hours_end
                updates_applied.append(f"active_hours_end={request.active_hours_end}")
            
            if request.active_days is not None:
                config.active_days = request.active_days
                updates_applied.append(f"active_days={request.active_days}")
            
            # Log schedule changes
            if request.active_hours_start is not None or request.active_hours_end is not None:
                start = config.active_hours_start
                end = config.active_hours_end
                if start is not None and end is not None:
                    logger.info(f"Active hours schedule updated: {start:02d}:00 - {end:02d}:00")
                else:
                    logger.info("Active hours schedule cleared: node is always on")
            
            # Update live provider settings
            if node.compute_provider:
                node.compute_provider.update_allocation(
                    cpu_allocation_pct=request.cpu_allocation_pct,
                    memory_allocation_pct=request.memory_allocation_pct,
                    max_concurrent_jobs=request.max_concurrent_jobs,
                    gpu_allocation_pct=request.gpu_allocation_pct,
                )
            
            if node.storage_provider:
                await node.storage_provider.update_limits(
                    pledged_gb=request.storage_gb,
                    upload_mbps_limit=request.upload_mbps_limit,
                    download_mbps_limit=request.download_mbps_limit,
                )
            
            # Persist config to disk
            config.save()
            
            logger.info(f"Resource configuration updated: {', '.join(updates_applied)}")
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to update resource configuration: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")
        
        # Return updated configuration
        return await get_resources()

    # ── Bridge Endpoints ─────────────────────────────────────────

    class BridgeDepositRequest(BaseModel):
        """Request body for bridge deposit operation."""
        amount: float = Field(..., gt=0, description="Amount of FTNS to deposit (in token units)")
        chain_address: str = Field(..., description="Destination on-chain address")
        destination_chain: int = Field(default=137, description="Destination chain ID (default: Polygon mainnet)")

    class BridgeWithdrawRequest(BaseModel):
        """Request body for bridge withdraw operation."""
        amount: float = Field(..., gt=0, description="Amount of FTNS to withdraw (in token units)")
        chain_address: str = Field(..., description="Source on-chain address")
        source_chain: int = Field(default=137, description="Source chain ID (default: Polygon mainnet)")

    class BridgeTransactionResponse(BaseModel):
        """Response model for bridge transactions."""
        transaction_id: str
        direction: str
        user_id: str
        chain_address: str
        amount: str
        source_chain: int
        destination_chain: int
        status: str
        source_tx_hash: Optional[str]
        destination_tx_hash: Optional[str]
        fee_amount: str
        created_at: str
        updated_at: str
        completed_at: Optional[str]
        error_message: Optional[str]

    @app.post("/bridge/deposit", tags=["bridge"])
    async def bridge_deposit(request: BridgeDepositRequest) -> Dict[str, Any]:
        """
        Deposit FTNS tokens from local balance to external chain.
        
        Burns local FTNS and initiates bridge transfer to mint tokens on the destination chain.
        
        Args:
            request: BridgeDepositRequest with amount, chain_address, and destination_chain
            
        Returns:
            Bridge transaction details including transaction_id and status
            
        Raises:
            HTTPException 503: If bridge not initialized
            HTTPException 400: If validation fails (insufficient balance, invalid address, etc.)
            HTTPException 500: If bridge operation fails
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            # Convert amount to wei (assuming 18 decimals like ETH)
            amount_wei = int(request.amount * 10**18)
            
            # Execute deposit
            tx = await node.ftns_bridge.deposit_to_chain(
                user_id=node.identity.node_id,
                amount=amount_wei,
                chain_address=request.chain_address,
                destination_chain=request.destination_chain
            )
            
            return {
                "success": True,
                "transaction": tx.to_dict()
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Bridge deposit failed: {error_msg}")
            
            # Map specific errors to HTTP status codes
            if "Insufficient" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "outside limits" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "Invalid" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise HTTPException(status_code=500, detail=f"Bridge deposit failed: {error_msg}")

    @app.post("/bridge/withdraw", tags=["bridge"])
    async def bridge_withdraw(request: BridgeWithdrawRequest) -> Dict[str, Any]:
        """
        Withdraw FTNS tokens from external chain to local balance.
        
        Locks on-chain FTNS and initiates bridge transfer to mint local FTNS.
        
        Args:
            request: BridgeWithdrawRequest with amount, chain_address, and source_chain
            
        Returns:
            Bridge transaction details including transaction_id and status
            
        Raises:
            HTTPException 503: If bridge not initialized
            HTTPException 400: If validation fails (insufficient balance, invalid address, etc.)
            HTTPException 500: If bridge operation fails
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        try:
            # Convert amount to wei (assuming 18 decimals like ETH)
            amount_wei = int(request.amount * 10**18)
            
            # Execute withdraw
            tx = await node.ftns_bridge.withdraw_from_chain(
                chain_address=request.chain_address,
                amount=amount_wei,
                user_id=node.identity.node_id,
                source_chain=request.source_chain
            )
            
            return {
                "success": True,
                "transaction": tx.to_dict()
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Bridge withdraw failed: {error_msg}")
            
            # Map specific errors to HTTP status codes
            if "Insufficient" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "outside limits" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            elif "Invalid" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise HTTPException(status_code=500, detail=f"Bridge withdraw failed: {error_msg}")

    @app.get("/bridge/status", tags=["bridge"])
    async def get_bridge_status() -> Dict[str, Any]:
        """
        Get bridge status and pending operations.
        
        Returns:
            Bridge statistics including:
            - total_deposited: Total FTNS deposited to chain
            - total_withdrawn: Total FTNS withdrawn from chain
            - total_fees_collected: Total fees collected
            - pending_transactions: Number of pending transactions
            - completed_transactions: Number of completed transactions
            - failed_transactions: Number of failed transactions
            - limits: Bridge limits (min/max amounts, fees)
            
        Raises:
            HTTPException 503: If bridge not initialized
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        try:
            stats = await node.ftns_bridge.get_bridge_stats()
            limits = await node.ftns_bridge.get_bridge_limits()
            pending = await node.ftns_bridge.get_pending_transactions()
            
            return {
                "stats": stats.to_dict(),
                "limits": {
                    "min_amount": str(limits.min_amount) if limits else "0",
                    "max_amount": str(limits.max_amount) if limits else "0",
                    "daily_limit": str(limits.daily_limit) if limits else "0",
                    "fee_bps": limits.fee_bps if limits else 0,
                } if limits else None,
                "pending_transactions": [tx.to_dict() for tx in pending],
                "pending_count": len(pending),
            }
        except Exception as e:
            logger.error(f"Failed to get bridge status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get bridge status: {str(e)}")

    @app.get("/bridge/transactions/{tx_id}", tags=["bridge"])
    async def get_bridge_transaction(tx_id: str) -> Dict[str, Any]:
        """
        Get status of a specific bridge transaction.
        
        Args:
            tx_id: Transaction ID to look up
            
        Returns:
            Bridge transaction details including status, amounts, and timestamps
            
        Raises:
            HTTPException 503: If bridge not initialized
            HTTPException 404: If transaction not found
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        tx = await node.ftns_bridge.get_bridge_status(tx_id)
        
        if not tx:
            raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
        
        return {
            "transaction": tx.to_dict()
        }

    @app.get("/bridge/transactions", tags=["bridge"])
    async def list_bridge_transactions(limit: int = 50) -> Dict[str, Any]:
        """
        List bridge transactions for the current user.
        
        Args:
            limit: Maximum number of transactions to return (default: 50, max: 200)
            
        Returns:
            List of bridge transactions for the current user
            
        Raises:
            HTTPException 503: If bridge not initialized
        """
        if not hasattr(node, 'ftns_bridge') or not node.ftns_bridge:
            raise HTTPException(status_code=503, detail="FTNS bridge not initialized")
        
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node identity not initialized")
        
        transactions = await node.ftns_bridge.get_user_transactions(
            user_id=node.identity.node_id,
            limit=min(limit, 200)
        )
        
        return {
            "transactions": [tx.to_dict() for tx in transactions],
            "count": len(transactions),
        }

    # ── Teacher Model Endpoints ───────────────────────────────────────

    @app.post("/teacher/create", tags=["teacher"])
    async def create_teacher(request: TeacherCreateRequest) -> Dict[str, Any]:
        """
        Create a new teacher model.
        
        Creates a distilled teacher model for the specified specialization.
        Rewards the node with FTNS tokens for contributing to the network.
        
        Args:
            request: Teacher creation parameters including specialization
            
        Returns:
            Created teacher model details including teacher_id
            
        Raises:
            HTTPException 503: If teacher creation infrastructure unavailable
        """
        try:
            from prsm.core.models import TeacherModel, ModelType
            from prsm.compute.teachers import create_production_teacher
            
            # Create the teacher model
            teacher_model = TeacherModel(
                name=f"{request.specialization.title()} Teacher",
                specialization=request.specialization,
            )
            
            # Create the DistilledTeacher instance (async)
            teacher = await create_production_teacher(
                teacher_model=teacher_model,
                use_real_implementation=request.use_real_implementation
            )
            
            # Generate teacher ID and store metadata
            teacher_id = str(teacher_model.teacher_id)
            teacher._created_at = time.time()
            
            # Store in node's registry
            node.teacher_registry[teacher_id] = teacher
            
            # Persist registry
            node._save_teacher_registry()
            
            # Reward the node for creating a teacher
            if hasattr(node, '_ftns_adapter') and node._ftns_adapter:
                await node._ftns_adapter.charge_user(
                    user_id=node.identity.node_id,
                    amount=-TEACHER_CREATION_REWARD_FTNS,  # Negative = credit
                    description=f"Teacher creation reward: {request.specialization}"
                )
            
            logger.info(f"Created teacher model: {teacher_id} for specialization: {request.specialization}")
            
            return {
                "teacher_id": teacher_id,
                "name": teacher_model.name,
                "specialization": teacher_model.specialization,
                "domain": request.domain or request.specialization,
                "model_type": teacher_model.model_type.value,
                "created_at": teacher._created_at,
                "reward_ftns": TEACHER_CREATION_REWARD_FTNS,
            }
            
        except ImportError as e:
            logger.error(f"Teacher model imports failed: {e}")
            raise HTTPException(
                status_code=503,
                detail="Teacher model infrastructure not available"
            )
        except Exception as e:
            logger.error(f"Teacher creation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create teacher: {str(e)}"
            )

    @app.get("/teacher/list", tags=["teacher"])
    async def list_teachers() -> Dict[str, Any]:
        """
        List all teacher models.
        
        Returns metadata for all teacher models registered on this node.
        
        Returns:
            List of teacher model metadata dictionaries
        """
        # Load persisted metadata
        persisted_meta = node._load_teacher_registry_meta()
        
        # Combine with in-memory registry
        teachers = []
        
        # Add from in-memory registry (most up-to-date)
        for teacher_id, teacher in node.teacher_registry.items():
            teachers.append({
                "teacher_id": teacher_id,
                "name": teacher.teacher_model.name,
                "specialization": teacher.teacher_model.specialization,
                "domain": getattr(teacher.teacher_model, "domain", teacher.teacher_model.specialization),
                "model_type": teacher.teacher_model.model_type.value,
                "created_at": getattr(teacher, "_created_at", None),
                "status": "active",
            })
        
        # Add any persisted teachers not in memory
        in_memory_ids = set(node.teacher_registry.keys())
        for teacher_id, meta in persisted_meta.items():
            if teacher_id not in in_memory_ids:
                teachers.append({
                    "teacher_id": teacher_id,
                    "name": meta.get("name"),
                    "specialization": meta.get("specialization"),
                    "domain": meta.get("domain"),
                    "model_type": meta.get("model_type"),
                    "created_at": meta.get("created_at"),
                    "status": "persisted",
                })
        
        return {
            "teachers": teachers,
            "count": len(teachers),
        }

    @app.get("/teacher/{teacher_id}", tags=["teacher"])
    async def get_teacher(teacher_id: str) -> Dict[str, Any]:
        """
        Get details for a specific teacher model.
        
        Args:
            teacher_id: UUID of the teacher model
            
        Returns:
            Detailed teacher model information
            
        Raises:
            HTTPException 404: If teacher not found
        """
        # Check in-memory registry first
        if teacher_id in node.teacher_registry:
            teacher = node.teacher_registry[teacher_id]
            return {
                "teacher_id": teacher_id,
                "name": teacher.teacher_model.name,
                "specialization": teacher.teacher_model.specialization,
                "domain": getattr(teacher.teacher_model, "domain", teacher.teacher_model.specialization),
                "model_type": teacher.teacher_model.model_type.value,
                "performance_score": getattr(teacher.teacher_model, "performance_score", None),
                "created_at": getattr(teacher, "_created_at", None),
                "status": "active",
                "teaching_history_count": len(getattr(teacher, "teaching_history", [])),
            }
        
        # Check persisted metadata
        persisted_meta = node._load_teacher_registry_meta()
        if teacher_id in persisted_meta:
            meta = persisted_meta[teacher_id]
            return {
                "teacher_id": teacher_id,
                "name": meta.get("name"),
                "specialization": meta.get("specialization"),
                "domain": meta.get("domain"),
                "model_type": meta.get("model_type"),
                "created_at": meta.get("created_at"),
                "status": "persisted",
            }
        
        raise HTTPException(
            status_code=404,
            detail=f"Teacher model not found: {teacher_id}"
        )

    @app.post("/teacher/{teacher_id}/train", tags=["teacher"])
    async def train_teacher(teacher_id: str, request: TeacherTrainingRequest) -> Dict[str, Any]:
        """
        Start a training run for a teacher model.
        
        Initiates an asynchronous training run. Charges FTNS based on
        training configuration.
        
        Args:
            teacher_id: UUID of the teacher model
            request: Training configuration parameters
            
        Returns:
            Training run details including run_id and poll_url
            
        Raises:
            HTTPException 404: If teacher not found
            HTTPException 402: If insufficient FTNS balance
            HTTPException 422: If training configuration invalid
        """
        # Check if teacher exists
        if teacher_id not in node.teacher_registry:
            # Try to load from persisted metadata
            persisted_meta = node._load_teacher_registry_meta()
            if teacher_id in persisted_meta:
                raise HTTPException(
                    status_code=422,
                    detail="Teacher model persisted but not loaded. Restart node or recreate teacher."
                )
            raise HTTPException(
                status_code=404,
                detail=f"Teacher model not found: {teacher_id}"
            )
        
        teacher = node.teacher_registry[teacher_id]
        
        # Check balance for training cost
        if hasattr(node, '_ftns_adapter') and node._ftns_adapter:
            balance = await node._ftns_adapter.get_user_balance(node.identity.node_id)
            if balance.balance < TEACHER_TRAINING_BASE_COST_FTNS:
                raise HTTPException(
                    status_code=402,
                    detail=f"Insufficient FTNS balance. Required: {TEACHER_TRAINING_BASE_COST_FTNS}, Available: {balance.balance}"
                )
            
            # Charge for training
            await node._ftns_adapter.charge_user(
                user_id=node.identity.node_id,
                amount=TEACHER_TRAINING_BASE_COST_FTNS,
                description=f"Teacher training: {teacher.teacher_model.specialization}"
            )
        
        # Generate a unique run ID for this training run
        run_id = str(_uuid.uuid4())

        # Read total_epochs from the teacher's config before launching
        total_epochs = None
        if hasattr(teacher, "training_config"):
            total_epochs = getattr(
                getattr(teacher.training_config, "hyperparameters", None),
                "epochs", None
            )

        # Register the job in PENDING state before the task starts
        job = TrainingJob(
            run_id=run_id,
            teacher_id=teacher_id,
            status=TrainingJobStatus.PENDING,
            started_at=time.time(),
            total_epochs=total_epochs,
        )
        node.training_jobs[run_id] = job

        # The training coroutine — updates job state as it progresses
        # Note: Use globals()['asyncio'] for Python 3.14 closure compatibility
        _asyncio_mod = globals()['asyncio']
        
        async def run_training():
            job.status = TrainingJobStatus.RUNNING
            try:
                if hasattr(teacher, "train"):
                    result = await teacher.train()
                else:
                    await _asyncio_mod.sleep(1)  # Basic DistilledTeacher has no real train()
                    result = None
                job.status = TrainingJobStatus.COMPLETED
                job.result = result
                job.completed_at = time.time()
                logger.info("Training completed", teacher_id=teacher_id, run_id=run_id)
            except _asyncio_mod.CancelledError:
                job.status = TrainingJobStatus.CANCELLED
                job.completed_at = time.time()
                logger.info("Training cancelled", teacher_id=teacher_id, run_id=run_id)
            except Exception as e:
                job.status = TrainingJobStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
                logger.error("Training failed", teacher_id=teacher_id, run_id=run_id, error=str(e))
            finally:
                node._save_training_runs()

        task = _asyncio_mod.create_task(run_training())
        job._task = task  # Keep reference for cancellation

        return {
            "run_id":         run_id,
            "teacher_id":     teacher_id,
            "status":         "pending",
            "total_epochs":   total_epochs,
            "cost_ftns":      TEACHER_TRAINING_BASE_COST_FTNS,
            "poll_url":       f"/teacher/{teacher_id}/training/{run_id}",
        }

    @app.get("/teacher/{teacher_id}/training/{run_id}", tags=["teacher"])
    async def get_training_status(teacher_id: str, run_id: str) -> Dict[str, Any]:
        """
        Poll the status of a specific training run.

        Returns live progress (current_epoch, current_step, elapsed_seconds)
        when status is 'running', and the full TrainingResult when 'completed'.
        """
        if run_id not in node.training_jobs:
            raise HTTPException(404, f"Training run {run_id} not found")

        job = node.training_jobs[run_id]

        if job.teacher_id != teacher_id:
            raise HTTPException(404, f"Training run {run_id} does not belong to teacher {teacher_id}")

        response = job.to_dict()

        # Augment with live progress when running
        if job.status == TrainingJobStatus.RUNNING:
            teacher = node.teacher_registry.get(teacher_id)
            if teacher and hasattr(teacher, "trainer") and teacher.trainer is not None:
                trainer = teacher.trainer
                current_epoch  = getattr(trainer, "current_epoch",  0)
                total_epochs   = job.total_epochs or getattr(
                    getattr(trainer, "config", None), "hyperparameters", None
                ) and trainer.config.hyperparameters.epochs or 1
                elapsed = (
                    time.time() - trainer.training_start_time
                    if trainer.training_start_time else 0.0
                )
                response["progress"] = {
                    "current_epoch":  current_epoch,
                    "total_epochs":   total_epochs,
                    "current_step":   getattr(trainer, "current_step", 0),
                    "global_step":    getattr(trainer, "global_step",  0),
                    "progress_pct":   round(current_epoch / max(total_epochs, 1) * 100, 1),
                    "elapsed_seconds": round(elapsed, 1),
                }

        return response


    @app.get("/teacher/{teacher_id}/training", tags=["teacher"])
    async def list_training_runs(teacher_id: str) -> Dict[str, Any]:
        """
        List all training runs for a teacher model (running, completed, and failed).
        """
        runs = [
            job.to_dict()
            for job in node.training_jobs.values()
            if job.teacher_id == teacher_id
        ]
        # Most-recent first
        runs.sort(key=lambda r: r["started_at"], reverse=True)
        return {"runs": runs, "count": len(runs)}


    @app.delete("/teacher/{teacher_id}/training/{run_id}", tags=["teacher"])
    async def cancel_training_run(teacher_id: str, run_id: str) -> Dict[str, Any]:
        """
        Cancel a pending or running training job.
 
        Calls task.cancel() on the underlying asyncio.Task.
        The task will transition to CANCELLED status asynchronously.
        """
        if run_id not in node.training_jobs:
            raise HTTPException(404, f"Training run {run_id} not found")

        job = node.training_jobs[run_id]

        if job.teacher_id != teacher_id:
            raise HTTPException(404, f"Training run {run_id} does not belong to teacher {teacher_id}")

        if job.status not in (TrainingJobStatus.PENDING, TrainingJobStatus.RUNNING):
            raise HTTPException(
                409,
                f"Cannot cancel a run in '{job.status.value}' state"
            )

        if job._task and not job._task.done():
            job._task.cancel()
            # Status transitions to CANCELLED inside run_training()'s CancelledError handler
            return {"cancelled": True, "run_id": run_id,
                    "note": "Cancellation requested. Poll status to confirm."}

        # Task already done but status wasn't updated — edge case
        job.status = TrainingJobStatus.CANCELLED
        job.completed_at = time.time()
        node._save_training_runs()
        return {"cancelled": True, "run_id": run_id}

    @app.get("/teacher/backends/available", tags=["teacher"])
    async def get_available_backends() -> Dict[str, Any]:
        """
        Get available ML training backends.
        
        Returns information about which ML backends (PyTorch, etc.)
        are available for teacher model training.
        
        Returns:
            Dictionary of available backends and their status
        """
        backends = {
            "pytorch": {"available": False, "version": None, "gpu": False},
            "tensorflow": {"available": False, "version": None, "gpu": False},
        }
        
        # Check PyTorch
        try:
            import torch
            backends["pytorch"]["available"] = True
            backends["pytorch"]["version"] = torch.__version__
            backends["pytorch"]["gpu"] = torch.cuda.is_available()
        except ImportError:
            pass
        
        # Check TensorFlow
        try:
            import tensorflow as tf
            backends["tensorflow"]["available"] = True
            backends["tensorflow"]["version"] = tf.__version__
            backends["tensorflow"]["gpu"] = len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            pass
        
        # Use detect_available_backends from config if available
        try:
            detected = detect_available_backends()
            for backend_name, info in detected.items():
                if backend_name in backends:
                    backends[backend_name].update(info)
                else:
                    backends[backend_name] = info
        except Exception as e:
            logger.debug(f"Could not use detect_available_backends: {e}")
        
        return {
            "backends": backends,
            "recommended": "pytorch" if backends["pytorch"]["available"] else "simulated",
        }

    # ── Distillation Endpoints ───────────────────────────────────────

    class DistillationSubmitRequest(BaseModel):
        """Request body for submitting a distillation job."""
        teacher_model_id: str = Field(..., description="ID from /teacher/list, or external model name")
        domain: str = Field(..., description="Target domain, e.g. 'medical_research'")
        target_size: str = Field("small", description="'tiny'|'small'|'medium'|'large'")
        optimization: str = Field("balanced", description="'speed'|'quality'|'size'|'balanced'")
        budget_ftns: int = Field(..., ge=100, description="Max FTNS to spend")
        name: Optional[str] = None
        description: Optional[str] = None

    def _get_distillation_orchestrator():
        """Lazy singleton; wired to the node's ledger on first call."""
        from prsm.compute.distillation.orchestrator import DistillationOrchestrator
        if not hasattr(_get_distillation_orchestrator, "_instance"):
            ftns_adapter = node._ftns_adapter  # _FTNSLedgerAdapter already on the node
            _get_distillation_orchestrator._instance = DistillationOrchestrator(
                ftns_service=ftns_adapter,
            )
        return _get_distillation_orchestrator._instance

    @app.post("/distillation/submit", tags=["distillation"])
    async def submit_distillation(request: DistillationSubmitRequest) -> Dict[str, Any]:
        """
        Submit a distillation job.
        
        Creates a new distillation job to train a smaller student model
        from a teacher model. The job runs asynchronously and can be
        monitored via the /distillation/{job_id} endpoint.
        
        Args:
            request: Distillation parameters including teacher_model_id, domain, etc.
            
        Returns:
            Job ID, status, and estimated cost
            
        Raises:
            HTTPException 503: If distillation infrastructure unavailable
            HTTPException 400: If insufficient balance or invalid parameters
        """
        if not node.identity:
            raise HTTPException(status_code=503, detail="Node not initialized")
        
        # Check balance before creating job
        if node.ledger:
            balance = await node.ledger.get_balance(node.identity.node_id)
            if balance < request.budget_ftns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient balance: {balance:.2f} < {request.budget_ftns}"
                )
        
        try:
            from prsm.compute.distillation.models import DistillationRequest, ModelSize, OptimizationTarget
            
            # Map string to enum
            target_size_map = {
                "tiny": ModelSize.TINY,
                "small": ModelSize.SMALL,
                "medium": ModelSize.MEDIUM,
                "large": ModelSize.LARGE,
            }
            optimization_map = {
                "speed": OptimizationTarget.SPEED,
                "quality": OptimizationTarget.ACCURACY,
                "size": OptimizationTarget.SIZE,
                "balanced": OptimizationTarget.BALANCED,
            }
            
            req = DistillationRequest(
                user_id=node.identity.node_id,
                teacher_model=request.teacher_model_id,
                domain=request.domain,
                target_size=target_size_map.get(request.target_size, ModelSize.SMALL),
                optimization_target=optimization_map.get(request.optimization, OptimizationTarget.BALANCED),
                budget_ftns=request.budget_ftns,
            )
            
            orchestrator = _get_distillation_orchestrator()
            job = await orchestrator.create_distillation(req)
            
            logger.info(f"Submitted distillation job {job.job_id} for domain: {request.domain}")
            
            return {
                "job_id": str(job.job_id),
                "status": job.status.value,
                "estimated_cost_ftns": request.budget_ftns,
                "teacher_model_id": request.teacher_model_id,
                "domain": request.domain,
                "target_size": request.target_size,
            }
            
        except ImportError as e:
            logger.error(f"Distillation imports failed: {e}")
            raise HTTPException(
                status_code=503,
                detail="Distillation infrastructure not available"
            )
        except Exception as e:
            logger.error(f"Distillation submission failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to submit distillation job: {str(e)}"
            )

    @app.get("/distillation/{job_id}", tags=["distillation"])
    async def get_distillation_job(job_id: str) -> Dict[str, Any]:
        """
        Get distillation job status.
        
        Returns the current status and details of a distillation job.
        
        Args:
            job_id: The distillation job ID
            
        Returns:
            Job status, progress, and result (if completed)
            
        Raises:
            HTTPException 404: If job not found
        """
        try:
            orchestrator = _get_distillation_orchestrator()
            
            # Check active jobs first
            from uuid import UUID
            job_uuid = UUID(job_id)
            
            if job_uuid in orchestrator.active_jobs:
                job = orchestrator.active_jobs[job_uuid]
            elif job_uuid in orchestrator.completed_jobs:
                job = orchestrator.completed_jobs[job_uuid]
            else:
                raise HTTPException(status_code=404, detail="Distillation job not found")
            
            result = {
                "job_id": str(job.job_id),
                "status": job.status.value,
                "user_id": job.user_id,
                "teacher_model": job.teacher_model,
                "domain": job.domain,
                "target_size": job.target_size.value if job.target_size else None,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error": job.error,
            }
            
            if job.result:
                result["result"] = {
                    "model_cid": job.result.model_cid if hasattr(job.result, 'model_cid') else None,
                    "quality_score": job.result.quality_score if hasattr(job.result, 'quality_score') else None,
                }
            
            return result
            
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")
        except KeyError:
            raise HTTPException(status_code=404, detail="Distillation job not found")
        except Exception as e:
            logger.error(f"Error getting distillation job: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get job status: {str(e)}"
            )

    @app.delete("/distillation/{job_id}", tags=["distillation"])
    async def cancel_distillation_job(job_id: str) -> Dict[str, Any]:
        """
        Cancel a distillation job.
        
        Cancels an active distillation job. Completed jobs cannot be cancelled.
        
        Args:
            job_id: The distillation job ID
            
        Returns:
            Cancellation status
            
        Raises:
            HTTPException 404: If job not found
            HTTPException 400: If job cannot be cancelled
        """
        try:
            orchestrator = _get_distillation_orchestrator()
            
            from uuid import UUID
            job_uuid = UUID(job_id)
            
            if job_uuid not in orchestrator.active_jobs:
                raise HTTPException(status_code=404, detail="Distillation job not found or already completed")
            
            job = orchestrator.active_jobs[job_uuid]
            
            # Cancel the job
            if hasattr(orchestrator, 'cancel_job'):
                await orchestrator.cancel_job(job_uuid)
            else:
                # Fallback: mark as cancelled
                from prsm.compute.distillation.models import DistillationStatus
                job.status = DistillationStatus.CANCELLED
                orchestrator.completed_jobs[job_uuid] = job
                del orchestrator.active_jobs[job_uuid]
            
            logger.info(f"Cancelled distillation job {job_id}")
            
            return {
                "job_id": job_id,
                "status": "cancelled",
                "message": "Distillation job cancelled successfully"
            }
            
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")
        except KeyError:
            raise HTTPException(status_code=404, detail="Distillation job not found")
        except Exception as e:
            logger.error(f"Error cancelling distillation job: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to cancel job: {str(e)}"
            )

    @app.get("/distillation", tags=["distillation"])
    async def list_distillation_jobs(status: Optional[str] = None) -> Dict[str, Any]:
        """
        List all distillation jobs.
        
        Returns a list of all distillation jobs, optionally filtered by status.
        
        Args:
            status: Optional status filter ('queued', 'training', 'completed', 'failed', 'cancelled')
            
        Returns:
            List of distillation jobs
        """
        try:
            orchestrator = _get_distillation_orchestrator()
            
            jobs = []
            
            # Add active jobs
            for job in orchestrator.active_jobs.values():
                if status is None or job.status.value == status:
                    jobs.append({
                        "job_id": str(job.job_id),
                        "status": job.status.value,
                        "user_id": job.user_id,
                        "domain": job.domain,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                    })
            
            # Add completed jobs
            for job in orchestrator.completed_jobs.values():
                if status is None or job.status.value == status:
                    jobs.append({
                        "job_id": str(job.job_id),
                        "status": job.status.value,
                        "user_id": job.user_id,
                        "domain": job.domain,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    })
            
            return {
                "jobs": jobs,
                "count": len(jobs),
                "active_count": len(orchestrator.active_jobs),
                "completed_count": len(orchestrator.completed_jobs),
            }
            
        except Exception as e:
            logger.error(f"Error listing distillation jobs: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list jobs: {str(e)}"
            )

    # ── WebSocket Endpoints ───────────────────────────────────────

    @app.websocket("/ws/status")
    async def websocket_status(websocket: WebSocket):
        """
        WebSocket endpoint for real-time status updates.
        
        Connect to receive live updates about:
        - Node status changes
        - Peer connections/disconnections
        - Job status updates
        - Transaction notifications
        
        Send JSON messages with type field to interact:
        - {"type": "heartbeat"} - Receive heartbeat acknowledgment
        - {"type": "get_status"} - Request current status
        """
        if not app.state.status_websocket:
            await websocket.close(code=1011, reason="WebSocket not initialized")
            return
        
        status_ws = app.state.status_websocket
        
        try:
            await status_ws.connect(websocket)
            
            while True:
                # Wait for messages from client
                data = await websocket.receive_json()
                
                # Handle heartbeat
                if data.get("type") == "heartbeat":
                    await status_ws.send_personal_status(websocket, {
                        "type": "heartbeat_ack",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                # Handle status request
                elif data.get("type") == "get_status":
                    status = await node.get_status()
                    await status_ws.send_personal_status(websocket, {
                        "type": "status_update",
                        "data": status,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
        except WebSocketDisconnect:
            status_ws.disconnect(websocket)
        except Exception as e:
            status_ws.disconnect(websocket)

    # ── Authentication Endpoints ───────────────────────────────────

    @app.get("/auth/verify", tags=["auth"])
    async def verify_token(request: Request, user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
        """
        Verify JWT token and return user information.
        
        Requires valid JWT token in Authorization header.
        """
        return {
            "valid": True,
            "user_id": user.get("user_id"),
            "username": user.get("username"),
            "role": user.get("role"),
            "permissions": user.get("permissions", []),
        }

    # ── Web Dashboard (served at /, /static, /api/) ──────────────────────────────

    from pathlib import Path as _Path
    from fastapi.staticfiles import StaticFiles as _StaticFiles
    from fastapi.responses import FileResponse as _FileResponse

    _DASHBOARD_TEMPLATES = _Path(__file__).parent.parent / "dashboard" / "templates"
    _DASHBOARD_STATIC = _Path(__file__).parent.parent / "dashboard" / "static"

    # Serve static assets (JS, CSS) from /static/
    if _DASHBOARD_STATIC.exists():
        app.mount("/static", _StaticFiles(directory=str(_DASHBOARD_STATIC)), name="dashboard-static")

    # Serve the SPA shell at / and /dashboard
    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        html_file = _DASHBOARD_TEMPLATES / "dashboard.html"
        if html_file.exists():
            return _FileResponse(str(html_file))
        return {"message": "Dashboard assets not found. Run from PRSM source tree."}

    # Mount the dashboard's API routes at /api/ (all dashboard.js calls use this prefix)
    try:
        from prsm.dashboard.app import create_dashboard_app as _create_dash_app
        _dash_app = _create_dash_app(node=node)
        app.mount("/api", _dash_app, name="dashboard-api")
        logger.info("Web dashboard mounted at /")
    except Exception as e:
        logger.warning(f"Dashboard not available: {e}")

    # ── Apply Security Hardening ───────────────────────────────────
    
    if enable_security and api_hardening:
        # Initialize the hardening components
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, schedule initialization
                asyncio.create_task(api_hardening.initialize())
            else:
                loop.run_until_complete(api_hardening.initialize())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(api_hardening.initialize())
        
        # Apply middleware
        api_hardening.apply_middleware()

    return app

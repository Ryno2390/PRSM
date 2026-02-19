"""
Node Management API
===================

FastAPI endpoints for monitoring and controlling a running PRSM node.
This is the node-local API (not the main PRSM platform API).
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class JobSubmission(BaseModel):
    """Request body for submitting a compute job."""
    job_type: str  # inference, embedding, benchmark
    payload: Dict[str, Any] = {}
    ftns_budget: float = 1.0


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


def create_api_app(node: Any) -> FastAPI:
    """Create the node management FastAPI app with a reference to the running node."""

    app = FastAPI(
        title="PRSM Node API",
        description="Management API for a PRSM network node",
        version="0.1.0",
    )

    @app.get("/status")
    async def get_status() -> Dict[str, Any]:
        """Get comprehensive node status."""
        return await node.get_status()

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

    @app.get("/compute/stats")
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

    return app

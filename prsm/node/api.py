"""
Node Management API
===================

FastAPI endpoints for monitoring and controlling a running PRSM node.
This is the node-local API (not the main PRSM platform API).
"""

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


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
        )

        if not result:
            raise HTTPException(status_code=502, detail="Upload failed — is IPFS running?")

        return {
            "cid": result.cid,
            "filename": result.filename,
            "size_bytes": result.size_bytes,
            "content_hash": result.content_hash,
            "creator_id": result.creator_id,
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

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Simple health check."""
        return {"status": "ok", "node_id": node.identity.node_id if node.identity else "unknown"}

    return app

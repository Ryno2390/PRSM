"""
Compute Requester
=================

Submits compute jobs to the PRSM network and collects results.
Broadcasts job offers via gossip, waits for provider acceptance,
verifies results, and records payments.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prsm.node.compute_provider import JobStatus, JobType
from prsm.node.gossip import (
    GOSSIP_JOB_ACCEPT,
    GOSSIP_JOB_OFFER,
    GOSSIP_JOB_RESULT,
    GOSSIP_PAYMENT_CONFIRM,
    GossipProtocol,
)
from prsm.node.identity import NodeIdentity, verify_signature
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.transport import WebSocketTransport

logger = logging.getLogger(__name__)


@dataclass
class SubmittedJob:
    """Tracks a job submitted to the network."""
    job_id: str
    job_type: JobType
    payload: Dict[str, Any]
    ftns_budget: float
    status: JobStatus = JobStatus.PENDING
    provider_id: Optional[str] = None
    provider_public_key: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    result_verified: bool = False
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    _result_event: asyncio.Event = field(default_factory=asyncio.Event)


class ComputeRequester:
    """Submits compute jobs to the PRSM network.

    Flow:
    1. submit_job() broadcasts a job_offer via gossip
    2. Providers accept (job_accept) — first accept wins
    3. Provider executes and sends job_result
    4. We verify the result signature and record payment
    """

    def __init__(
        self,
        identity: NodeIdentity,
        transport: WebSocketTransport,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        accept_timeout: float = 30.0,
        result_timeout: float = 300.0,
    ):
        self.identity = identity
        self.transport = transport
        self.gossip = gossip
        self.ledger = ledger
        self.accept_timeout = accept_timeout
        self.result_timeout = result_timeout

        self.submitted_jobs: Dict[str, SubmittedJob] = {}
        self._running = False
        self.ledger_sync = None  # Set by node.py after construction

    async def start(self) -> None:
        """Register gossip handlers."""
        self._running = True
        self.gossip.subscribe(GOSSIP_JOB_ACCEPT, self._on_job_accept)
        self.gossip.subscribe(GOSSIP_JOB_RESULT, self._on_job_result)
        logger.info("Compute requester started")

    async def stop(self) -> None:
        self._running = False

    async def submit_job(
        self,
        job_type: JobType,
        payload: Dict[str, Any],
        ftns_budget: float,
    ) -> SubmittedJob:
        """Submit a compute job to the network.

        Returns a SubmittedJob that can be awaited for results via get_result().
        """
        # Check balance
        balance = await self.ledger.get_balance(self.identity.node_id)
        if balance < ftns_budget:
            raise ValueError(f"Insufficient FTNS balance: {balance:.2f} < {ftns_budget:.2f}")

        job = SubmittedJob(
            job_id=uuid.uuid4().hex,
            job_type=job_type,
            payload=payload,
            ftns_budget=ftns_budget,
        )
        self.submitted_jobs[job.job_id] = job

        # Broadcast job offer
        await self.gossip.publish(GOSSIP_JOB_OFFER, {
            "job_id": job.job_id,
            "job_type": job_type.value,
            "requester_id": self.identity.node_id,
            "payload": payload,
            "ftns_budget": ftns_budget,
        })

        logger.info(f"Submitted job {job.job_id[:8]} ({job_type.value}), budget: {ftns_budget} FTNS")
        return job

    async def get_result(self, job_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for a job result. Returns the result dict or None on timeout."""
        job = self.submitted_jobs.get(job_id)
        if not job:
            return None

        timeout = timeout or self.result_timeout
        try:
            await asyncio.wait_for(job._result_event.wait(), timeout=timeout)
            return job.result
        except asyncio.TimeoutError:
            job.status = JobStatus.FAILED
            job.error = "Timed out waiting for result"
            return None

    # ── Gossip handlers ──────────────────────────────────────────

    async def _on_job_accept(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle a provider accepting our job."""
        job_id = data.get("job_id", "")
        job = self.submitted_jobs.get(job_id)
        if not job:
            return

        # First accept wins
        if job.status != JobStatus.PENDING:
            return

        job.status = JobStatus.ACCEPTED
        job.provider_id = data.get("provider_id", origin)
        job.provider_public_key = data.get("public_key", "")
        logger.info(f"Job {job_id[:8]} accepted by provider {job.provider_id[:8]}")

    async def _on_job_result(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle a job result from a provider."""
        job_id = data.get("job_id", "")
        job = self.submitted_jobs.get(job_id)
        if not job:
            return

        provider_id = data.get("provider_id", origin)
        status = data.get("status", "")

        if status == "failed":
            job.status = JobStatus.FAILED
            job.error = data.get("error", "Unknown error")
            job.completed_at = time.time()
            job._result_event.set()
            logger.warning(f"Job {job_id[:8]} failed: {job.error}")
            return

        # Verify result signature if we have the provider's public key
        result = data.get("result", {})
        signature = data.get("signature", "")
        pub_key = data.get("public_key", job.provider_public_key or "")

        verified = False
        if pub_key and signature:
            result_bytes = json.dumps(result, sort_keys=True).encode()
            verified = verify_signature(pub_key, result_bytes, signature)

        job.status = JobStatus.COMPLETED
        job.result = result
        job.result_verified = verified
        job.completed_at = time.time()

        # Record payment
        try:
            if provider_id != self.identity.node_id:
                tx = await self.ledger.transfer(
                    from_wallet=self.identity.node_id,
                    to_wallet=provider_id,
                    amount=job.ftns_budget,
                    tx_type=TransactionType.COMPUTE_PAYMENT,
                    description=f"Payment for job {job_id[:8]}",
                )

                # Broadcast payment via ledger sync
                if self.ledger_sync:
                    try:
                        await self.ledger_sync.broadcast_transaction(tx)
                    except Exception:
                        pass

                # Confirm payment on network
                await self.gossip.publish(GOSSIP_PAYMENT_CONFIRM, {
                    "job_id": job_id,
                    "requester_id": self.identity.node_id,
                    "provider_id": provider_id,
                    "amount": job.ftns_budget,
                })
        except ValueError as e:
            logger.error(f"Payment failed for job {job_id[:8]}: {e}")

        job._result_event.set()

        logger.info(
            f"Job {job_id[:8]} completed by {provider_id[:8]}, "
            f"verified={verified}, paid {job.ftns_budget} FTNS"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return requester statistics."""
        statuses = {}
        for job in self.submitted_jobs.values():
            statuses[job.status.value] = statuses.get(job.status.value, 0) + 1

        total_spent = sum(
            j.ftns_budget for j in self.submitted_jobs.values()
            if j.status == JobStatus.COMPLETED
        )

        return {
            "total_submitted": len(self.submitted_jobs),
            "by_status": statuses,
            "total_ftns_spent": total_spent,
        }

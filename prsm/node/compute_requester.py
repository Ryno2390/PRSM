"""
Compute Requester
=================

Submits compute jobs to the PRSM network and collects results.
Broadcasts job offers via gossip, waits for provider acceptance,
verifies results, and records payments.

Supports capability-based smart routing to target capable peers first.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from prsm.node.compute_provider import JobStatus, JobType
from prsm.node.gossip import (
    GOSSIP_JOB_ACCEPT,
    GOSSIP_JOB_CANCEL,
    GOSSIP_JOB_CONFIRM,
    GOSSIP_JOB_OFFER,
    GOSSIP_JOB_RESULT,
    GOSSIP_PAYMENT_CONFIRM,
    GossipProtocol,
)
from prsm.node.identity import NodeIdentity, verify_signature
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.transport import WebSocketTransport
from prsm.node.payment_escrow import PaymentEscrow

if TYPE_CHECKING:
    from prsm.node.discovery import PeerDiscovery

logger = logging.getLogger(__name__)


# Mapping of job types to required capabilities
JOB_TYPE_CAPABILITIES = {
    JobType.INFERENCE: "inference",
    JobType.EMBEDDING: "embedding",
    JobType.BENCHMARK: "benchmark",
    JobType.TRAINING: "training",
}

# Mapping of job types to preferred backends
JOB_TYPE_PREFERRED_BACKENDS = {
    JobType.INFERENCE: ["anthropic", "openai", "local"],
    JobType.EMBEDDING: ["openai", "local"],
    JobType.BENCHMARK: ["local", "anthropic", "openai"],
    JobType.TRAINING: ["local", "anthropic", "openai"],  # Training: local first for large workloads
}


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
        discovery: Optional["PeerDiscovery"] = None,
        accept_timeout: float = 30.0,
        result_timeout: float = 300.0,
        smart_routing: bool = True,
    ):
        self.identity = identity
        self.transport = transport
        self.gossip = gossip
        self.ledger = ledger
        self.discovery = discovery
        self.accept_timeout = accept_timeout
        self.result_timeout = result_timeout
        self.smart_routing = smart_routing

        self.submitted_jobs: Dict[str, SubmittedJob] = {}
        self._running = False
        self.ledger_sync = None  # Set by node.py after construction
        self.escrow: Optional["PaymentEscrow"] = None  # Set by node.py after construction

    async def start(self) -> None:
        """Register gossip handlers."""
        self._running = True
        self.gossip.subscribe(GOSSIP_JOB_ACCEPT, self._on_job_accept)
        self.gossip.subscribe(GOSSIP_JOB_RESULT, self._on_job_result)
        logger.info("Compute requester started")

    async def stop(self) -> None:
        self._running = False

    def set_discovery(self, discovery: "PeerDiscovery") -> None:
        """Set the peer discovery instance for smart routing."""
        self.discovery = discovery

    def _get_capable_peers(self, job_type: JobType) -> List[str]:
        """Find peers capable of handling a specific job type.

        Args:
            job_type: The type of job to find capable peers for.

        Returns:
            List of peer IDs that have the required capabilities.
        """
        if not self.discovery:
            return []

        # Get required capability for job type
        required_capability = JOB_TYPE_CAPABILITIES.get(job_type)
        if not required_capability:
            logger.debug(f"No capability mapping for job type {job_type.value}")
            return []

        # Find peers with the required capability
        capable_peers = self.discovery.find_peers_with_capability(required_capability)

        # Optionally filter by preferred backends
        preferred_backends = JOB_TYPE_PREFERRED_BACKENDS.get(job_type, [])
        if preferred_backends:
            backend_peers = []
            for backend in preferred_backends:
                backend_peers.extend(self.discovery.find_peers_with_backend(backend))
            # Prefer peers that have both capability and preferred backend
            backend_peer_ids = {p.node_id for p in backend_peers}
            capable_peers = [p for p in capable_peers if p.node_id in backend_peer_ids] or capable_peers

        return [p.node_id for p in capable_peers]

    async def submit_job(
        self,
        job_type: JobType,
        payload: Dict[str, Any],
        ftns_budget: float = 0.0,
        target_peers: Optional[List[str]] = None,
        use_escrow: bool = True,
        job_id: Optional[str] = None,
    ) -> SubmittedJob:
        """Submit a compute job to the network, optionally with escrow.

        Args:
            job_id: Optional pre-assigned job ID (e.g. from API escrow).
                    If None, a new UUID is generated.
        """
        if ftns_budget > 0:
            balance = await self.ledger.get_balance(self.identity.node_id)
            if balance < ftns_budget:
                raise ValueError(f"Insufficient FTNS balance: {balance:.2f} < {ftns_budget:.2f}")

        job = SubmittedJob(
            job_id=job_id or uuid.uuid4().hex,
            job_type=job_type,
            payload=payload,
            ftns_budget=ftns_budget,
        )
        self.submitted_jobs[job.job_id] = job

        # Create escrow for the job (Phase 1: on-chain FTNS economy)
        if use_escrow and self.escrow is not None and ftns_budget > 0:
            escrow_entry = await self.escrow.create_escrow(
                job_id=job.job_id,
                amount=ftns_budget,
                requester_id=self.identity.node_id,
            )
            if escrow_entry:
                job.escrow_id = escrow_entry.escrow_id
                logger.info(
                    f"Escrow created for {job.job_id[:8]}: "
                    f"{ftns_budget:.6f} FTNS locked"
                )
            else:
                logger.warning(
                    f"Escrow creation failed for {job.job_id[:8]} — "
                    f"proceeding without escrow"
                )
        elif ftns_budget > 0 and self.escrow is None:
            logger.debug(
                f"No escrow available for {job.job_id[:8]} — "
                f"running without on-chain escrow"
            )

        # Determine target peers for smart routing
        routing_targets = target_peers
        routing_mode = "broadcast"

        if self.smart_routing and self.discovery and target_peers is None:
            capable_peers = self._get_capable_peers(job_type)
            if capable_peers:
                routing_targets = capable_peers
                routing_mode = "targeted"
                logger.info(
                    f"Smart routing: targeting {len(capable_peers)} "
                    f"capable peers for {job_type.value}"
                )
            else:
                logger.info(
                    f"No capable peers found for {job_type.value}, falling back to broadcast"
                )

        # Broadcast job offer
        job_offer = {
            "job_id": job.job_id,
            "job_type": job_type.value,
            "requester_id": self.identity.node_id,
            "payload": payload,
            "ftns_budget": ftns_budget,
        }

        # Add target peers if smart routing
        if routing_targets:
            job_offer["target_peers"] = routing_targets

        await self.gossip.publish(GOSSIP_JOB_OFFER, job_offer)

        logger.info(
            f"Submitted job {job.job_id[:8]} ({job_type.value}), "
            f"budget: {ftns_budget} FTNS, routing: {routing_mode}"
        )
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
        """Handle a provider accepting our job.

        First-accept-wins: the first provider to accept is confirmed via
        GOSSIP_JOB_CONFIRM.  Other providers that accepted should see the
        confirm message (with a different provider_id) and drop the job.
        """
        job_id = data.get("job_id", "")
        job = self.submitted_jobs.get(job_id)
        if not job:
            return

        # First accept wins — ignore subsequent accepts
        if job.status != JobStatus.PENDING:
            return

        provider_id = data.get("provider_id", origin)
        job.status = JobStatus.ACCEPTED
        job.provider_id = provider_id
        job.provider_public_key = data.get("public_key", "")

        # Confirm this provider so it starts execution;
        # other providers that accepted will see the confirm and drop the job.
        await self.gossip.publish(GOSSIP_JOB_CONFIRM, {
            "job_id": job_id,
            "provider_id": provider_id,
            "requester_id": self.identity.node_id,
        })

        logger.info(f"Job {job_id[:8]} confirmed provider {provider_id[:8]}")

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

        # Verify result signature — required for payment
        result = data.get("result", {})
        signature = data.get("signature", "")
        pub_key = data.get("public_key", job.provider_public_key or "")

        verified = False
        if pub_key and signature:
            result_bytes = json.dumps(result, sort_keys=True).encode()
            verified = verify_signature(pub_key, result_bytes, signature)

        if not verified and provider_id != self.identity.node_id:
            # Reject unsigned/unverified results from remote providers.
            # Self-compute results (same node) are trusted without signature.
            logger.warning(
                f"Job {job_id[:8]}: rejecting unverified result from "
                f"{provider_id[:8]} (missing or invalid signature)"
            )
            return

        job.status = JobStatus.COMPLETED
        job.result = result
        job.result_verified = verified
        job.completed_at = time.time()

        # Record payment — only for remote providers.
        # Self-compute payment is handled by the API escrow release.
        try:
            if provider_id != self.identity.node_id and job.ftns_budget > 0:
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

    async def submit_training_job(
        self,
        teacher_model_id: str,
        domain: str,
        target_size: str = "small",
        budget_ftns: float = 200.0,
    ) -> Optional[str]:
        """Broadcast a training job to capable P2P nodes. Returns job_id or None.
        
        Args:
            teacher_model_id: ID of the teacher model to distill from
            domain: Target domain for the distilled model
            target_size: Target model size ('tiny', 'small', 'medium', 'large')
            budget_ftns: Maximum FTNS to spend on this training job
            
        Returns:
            job_id if job was submitted to peers, None if no capable peers found
            (caller should fall back to local execution)
        """
        if not self.discovery:
            logger.info("No discovery service; cannot find training-capable peers")
            return None
            
        capable_peers = self.discovery.find_peers_with_capability("training")
        if not capable_peers:
            logger.info("No training-capable peers found; running locally")
            return None  # Caller should fall back to local execution

        job_id = uuid.uuid4().hex
        await self.gossip.publish(GOSSIP_JOB_OFFER, {
            "job_id": job_id,
            "job_type": JobType.TRAINING.value,
            "requester_id": self.identity.node_id,
            "payload": {
                "teacher_model_id": teacher_model_id,
                "domain": domain,
                "target_size": target_size,
                "budget_ftns": budget_ftns,
            },
            "ftns_budget": budget_ftns,
        })
        
        # Track the submitted job
        job = SubmittedJob(
            job_id=job_id,
            job_type=JobType.TRAINING,
            payload={
                "teacher_model_id": teacher_model_id,
                "domain": domain,
                "target_size": target_size,
            },
            ftns_budget=budget_ftns,
        )
        self.submitted_jobs[job_id] = job
        
        logger.info(
            f"Submitted training job {job_id[:8]} to {len(capable_peers)} capable peers, "
            f"budget: {budget_ftns} FTNS"
        )
        return job_id

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

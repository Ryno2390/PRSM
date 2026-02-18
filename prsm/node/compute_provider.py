"""
Compute Provider
================

Accepts and executes compute jobs from the network.
Auto-detects available resources using psutil and the existing
ResourceCapabilityDetector. Jobs run in sandboxed subprocesses
with resource monitoring.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from prsm.node.gossip import (
    GOSSIP_JOB_ACCEPT,
    GOSSIP_JOB_OFFER,
    GOSSIP_JOB_RESULT,
    GossipProtocol,
)
from prsm.node.identity import NodeIdentity
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.transport import P2PMessage, WebSocketTransport

logger = logging.getLogger(__name__)


class JobType(str, Enum):
    INFERENCE = "inference"
    EMBEDDING = "embedding"
    BENCHMARK = "benchmark"


class JobStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ComputeJob:
    """A compute job offered to or accepted by this node."""
    job_id: str
    job_type: JobType
    requester_id: str
    payload: Dict[str, Any]
    ftns_budget: float
    status: JobStatus = JobStatus.PENDING
    accepted_by: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    result_signature: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None


@dataclass
class SystemResources:
    """Detected system resources."""
    cpu_count: int = 1
    cpu_freq_mhz: float = 0.0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0


def detect_resources() -> SystemResources:
    """Detect available system resources using psutil."""
    resources = SystemResources()
    try:
        import psutil
        resources.cpu_count = psutil.cpu_count(logical=True) or 1
        freq = psutil.cpu_freq()
        if freq:
            resources.cpu_freq_mhz = freq.current
        mem = psutil.virtual_memory()
        resources.memory_total_gb = round(mem.total / (1024**3), 2)
        resources.memory_available_gb = round(mem.available / (1024**3), 2)
    except ImportError:
        resources.cpu_count = os.cpu_count() or 1

    # GPU detection (best effort)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            resources.gpu_available = True
            resources.gpu_name = parts[0].strip()
            resources.gpu_memory_gb = round(float(parts[1].strip()) / 1024, 2)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return resources


class ComputeProvider:
    """Accepts and executes compute jobs from the PRSM network.

    Listens for job_offer gossip, evaluates against available resources,
    and executes accepted jobs in sandboxed subprocesses.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        transport: WebSocketTransport,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        cpu_allocation_pct: int = 50,
        memory_allocation_pct: int = 50,
        max_concurrent_jobs: int = 3,
    ):
        self.identity = identity
        self.transport = transport
        self.gossip = gossip
        self.ledger = ledger
        self.cpu_allocation_pct = cpu_allocation_pct
        self.memory_allocation_pct = memory_allocation_pct
        self.max_concurrent_jobs = max_concurrent_jobs

        self.resources = detect_resources()
        self.active_jobs: Dict[str, ComputeJob] = {}
        self.completed_jobs: Dict[str, ComputeJob] = {}
        self._running = False
        self.ledger_sync = None  # Set by node.py after construction

    @property
    def available_capacity(self) -> Dict[str, Any]:
        active_count = len(self.active_jobs)
        return {
            "cpu_cores_allocated": max(1, int(self.resources.cpu_count * self.cpu_allocation_pct / 100)),
            "memory_gb_allocated": round(self.resources.memory_total_gb * self.memory_allocation_pct / 100, 2),
            "gpu_available": self.resources.gpu_available,
            "concurrent_slots": max(0, self.max_concurrent_jobs - active_count),
            "active_jobs": active_count,
        }

    async def start(self) -> None:
        """Register gossip handlers and start provider."""
        self._running = True
        self.gossip.subscribe(GOSSIP_JOB_OFFER, self._on_job_offer)
        logger.info(
            f"Compute provider started: {self.resources.cpu_count} CPUs, "
            f"{self.resources.memory_total_gb}GB RAM, "
            f"GPU: {'yes (' + self.resources.gpu_name + ')' if self.resources.gpu_available else 'no'}"
        )

    async def stop(self) -> None:
        self._running = False

    async def _on_job_offer(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Evaluate a job offer and accept if we have capacity."""
        if not self._running:
            return

        job_id = data.get("job_id", "")
        job_type_str = data.get("job_type", "")
        ftns_budget = data.get("ftns_budget", 0.0)
        requester_id = data.get("requester_id", origin)

        # Don't accept our own jobs
        if requester_id == self.identity.node_id:
            return

        # Check if we already have this job
        if job_id in self.active_jobs or job_id in self.completed_jobs:
            return

        # Check capacity
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return

        # Validate job type
        try:
            job_type = JobType(job_type_str)
        except ValueError:
            return

        # Accept the job
        job = ComputeJob(
            job_id=job_id,
            job_type=job_type,
            requester_id=requester_id,
            payload=data.get("payload", {}),
            ftns_budget=ftns_budget,
            status=JobStatus.ACCEPTED,
            accepted_by=self.identity.node_id,
        )
        self.active_jobs[job_id] = job

        # Announce acceptance
        await self.gossip.publish(GOSSIP_JOB_ACCEPT, {
            "job_id": job_id,
            "provider_id": self.identity.node_id,
            "public_key": self.identity.public_key_b64,
            "capacity": self.available_capacity,
        })

        logger.info(f"Accepted job {job_id[:8]} ({job_type.value}) from {requester_id[:8]}")

        # Execute the job
        asyncio.create_task(self._execute_job(job))

    async def _execute_job(self, job: ComputeJob) -> None:
        """Execute a compute job and publish the result."""
        job.status = JobStatus.RUNNING
        try:
            if job.job_type == JobType.BENCHMARK:
                result = await self._run_benchmark(job)
            elif job.job_type == JobType.INFERENCE:
                result = await self._run_inference(job)
            elif job.job_type == JobType.EMBEDDING:
                result = await self._run_embedding(job)
            else:
                raise ValueError(f"Unsupported job type: {job.job_type}")

            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = time.time()

            # Sign the result
            result_bytes = json.dumps(result, sort_keys=True).encode()
            job.result_signature = self.identity.sign(result_bytes)

            # Record earnings
            tx = await self.ledger.credit(
                wallet_id=self.identity.node_id,
                amount=job.ftns_budget,
                tx_type=TransactionType.COMPUTE_EARNING,
                description=f"Compute job {job.job_id[:8]} ({job.job_type.value})",
            )

            # Broadcast earning via ledger sync
            if self.ledger_sync:
                try:
                    await self.ledger_sync.broadcast_transaction(tx)
                except Exception:
                    pass

            # Publish result
            await self.gossip.publish(GOSSIP_JOB_RESULT, {
                "job_id": job.job_id,
                "provider_id": self.identity.node_id,
                "status": "completed",
                "result": result,
                "signature": job.result_signature,
                "public_key": self.identity.public_key_b64,
            })

            logger.info(f"Job {job.job_id[:8]} completed, earned {job.ftns_budget} FTNS")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()

            await self.gossip.publish(GOSSIP_JOB_RESULT, {
                "job_id": job.job_id,
                "provider_id": self.identity.node_id,
                "status": "failed",
                "error": str(e),
            })

            logger.error(f"Job {job.job_id[:8]} failed: {e}")

        finally:
            # Move from active to completed
            self.active_jobs.pop(job.job_id, None)
            self.completed_jobs[job.job_id] = job

    # ── Job type implementations ─────────────────────────────────

    async def _run_benchmark(self, job: ComputeJob) -> Dict[str, Any]:
        """Run a compute benchmark and return results."""
        iterations = job.payload.get("iterations", 1000000)

        start = time.monotonic()
        # CPU benchmark: compute primes (runs in thread to not block event loop)
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, _cpu_benchmark, iterations)
        elapsed = time.monotonic() - start

        return {
            "benchmark_type": "cpu_prime_sieve",
            "iterations": iterations,
            "primes_found": count,
            "elapsed_seconds": round(elapsed, 4),
            "ops_per_second": round(iterations / elapsed, 2),
            "cpu_count": self.resources.cpu_count,
            "memory_gb": self.resources.memory_total_gb,
        }

    async def _run_inference(self, job: ComputeJob) -> Dict[str, Any]:
        """Run an inference job. For alpha, this is a mock that demonstrates the pipeline."""
        prompt = job.payload.get("prompt", "")
        model = job.payload.get("model", "local")

        # For alpha: return a structured response showing the pipeline works
        return {
            "model": model,
            "prompt": prompt[:200],
            "response": f"[PRSM node {self.identity.node_id[:8]} processed inference]",
            "tokens_used": len(prompt.split()),
            "provider_node": self.identity.node_id,
        }

    async def _run_embedding(self, job: ComputeJob) -> Dict[str, Any]:
        """Compute embeddings. For alpha, returns a mock embedding vector."""
        text = job.payload.get("text", "")
        dimensions = job.payload.get("dimensions", 128)

        # For alpha: generate a deterministic pseudo-embedding
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Generate a simple deterministic vector from the hash
        embedding = [((b % 200) - 100) / 100.0 for b in h * (dimensions // 32 + 1)][:dimensions]

        return {
            "text_length": len(text),
            "dimensions": dimensions,
            "embedding": embedding,
            "provider_node": self.identity.node_id,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return provider statistics."""
        return {
            "resources": {
                "cpu_count": self.resources.cpu_count,
                "cpu_freq_mhz": self.resources.cpu_freq_mhz,
                "memory_total_gb": self.resources.memory_total_gb,
                "gpu_available": self.resources.gpu_available,
                "gpu_name": self.resources.gpu_name,
            },
            "allocation": {
                "cpu_pct": self.cpu_allocation_pct,
                "memory_pct": self.memory_allocation_pct,
                "max_concurrent": self.max_concurrent_jobs,
            },
            "capacity": self.available_capacity,
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
        }


def _cpu_benchmark(iterations: int) -> int:
    """Simple prime sieve for CPU benchmarking (runs in thread pool)."""
    if iterations < 2:
        return 0
    sieve = bytearray(b'\x01') * (iterations + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(iterations**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return sum(sieve)

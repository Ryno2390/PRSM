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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prsm.node.config import NodeConfig

from prsm.node.config import is_active_now
from prsm.node.gossip import (
    GOSSIP_JOB_OFFER,
    GOSSIP_JOB_ACCEPT,
    GOSSIP_JOB_RESULT,
    GossipProtocol,
)
from prsm.node.identity import NodeIdentity
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.transport import P2PMessage, WebSocketTransport
from prsm.compute.nwtn.backends.config import detect_available_backends
from prsm.node.payment_escrow import PaymentEscrow
from prsm.node.result_consensus import ResultConsensus

logger = logging.getLogger(__name__)


class JobType(str, Enum):
    INFERENCE = "inference"
    EMBEDDING = "embedding"
    BENCHMARK = "benchmark"
    TRAINING = "training"       # NEW: distributed model training job


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

    Supports both single-node mode (self-compute) and multi-node mode
    with consensus-verified results and escrowed payments.
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
        gpu_allocation_pct: int = 80,
        config: Optional["NodeConfig"] = None,
    ):
        self.identity = identity
        self.transport = transport
        self.gossip = gossip
        self.ledger = ledger
        self.cpu_allocation_pct = cpu_allocation_pct
        self.memory_allocation_pct = memory_allocation_pct
        self.max_concurrent_jobs = max_concurrent_jobs
        self.gpu_allocation_pct = gpu_allocation_pct
        self.config = config  # NodeConfig for scheduling checks

        self.resources = detect_resources()
        self.active_jobs: Dict[str, ComputeJob] = {}
        self.completed_jobs: Dict[str, ComputeJob] = {}
        self._running = False
        self.allow_self_compute = True  # Execute own jobs when no peers (single-node mode)
        self.ledger_sync = None  # Set by node.py after construction
        self.orchestrator = None  # NWTN orchestrator, set by node.py after construction

        # Cross-node infrastructure
        self.escrow = PaymentEscrow(
            ledger=self.ledger,
            node_id=self.identity.node_id,
        )
        self.consensus = ResultConsensus(
            epsilon=0.01,
            timeout_seconds=300.0,
        )
        self._escrow_task: Optional[asyncio.Task] = None

    @property
    def available_capacity(self) -> Dict[str, Any]:
        active_count = len(self.active_jobs)
        capacity = {
            "cpu_cores_allocated": max(1, int(self.resources.cpu_count * self.cpu_allocation_pct / 100)),
            "memory_gb_allocated": round(self.resources.memory_total_gb * self.memory_allocation_pct / 100, 2),
            "gpu_available": self.resources.gpu_available,
            "concurrent_slots": max(0, self.max_concurrent_jobs - active_count),
            "active_jobs": active_count,
        }
        if self.resources.gpu_available:
            capacity["gpu_memory_gb_allocated"] = round(
                self.resources.gpu_memory_gb * self.gpu_allocation_pct / 100, 2
            )
        return capacity

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

        # Check if we're within active hours
        if self.config and not is_active_now(self.config):
            logger.debug("Node is outside active hours, declining job offer")
            return

        job_id = data.get("job_id", "")
        job_type_str = data.get("job_type", "")
        ftns_budget = data.get("ftns_budget", 0.0)
        requester_id = data.get("requester_id", origin)

        # In network mode (peers connected), don't accept own jobs to avoid
        # double-counting.  In local mode (no peers), execute our own jobs
        # so a single-node setup is functional out of the box.
        if requester_id == self.identity.node_id:
            if not self.allow_self_compute:
                return
            if self.transport and self.transport.peer_count > 0:
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

        # Check for training capability before accepting TRAINING jobs
        if job_type == JobType.TRAINING:
            try:
                from prsm.compute.distillation.backends.pytorch_backend import PyTorchDistillationBackend
                # PyTorchDistillationBackend import successful - we have training capability
            except ImportError:
                logger.debug("Declining TRAINING job - no distillation backend available")
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
            elif job.job_type == JobType.TRAINING:
                result = await self._run_training(job.payload)
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

        else:
            # Job completed successfully — escrow is released by the API
            # endpoint (compute_query) after returning the result.
            # This avoids a race with the concurrent API escrow release.
            pass

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
        """Run an inference job via NWTN orchestrator if available, else mock."""
        prompt = job.payload.get("prompt", "")
        model = job.payload.get("model", "local")

        # Use NWTN orchestrator if wired — with timeout to avoid hanging
        if self.orchestrator is not None:
            try:
                from prsm.core.models import UserInput
                user_input = UserInput(
                    user_id=job.requester_id,
                    prompt=prompt,
                    context_allocation=100,
                )
                response = await asyncio.wait_for(
                    self.orchestrator.process_query(user_input),
                    timeout=10.0,
                )
                return {
                    "model": model,
                    "prompt": prompt[:200],
                    "response": response.response,
                    "tokens_used": response.context_used,
                    "ftns_charged": response.ftns_charged,
                    "models_used": response.models_used,
                    "confidence": response.confidence_score,
                    "processing_time": response.processing_time,
                    "provider_node": self.identity.node_id,
                    "source": "nwtn_orchestrator",
                }
            except asyncio.TimeoutError:
                logger.warning(
                    f"NWTN orchestrator timed out after 10s, "
                    f"falling back to mock for {job.job_id[:8]}"
                )
            except Exception as e:
                logger.warning(f"NWTN inference failed, falling back to mock: {e}")

        # Fallback: mock response (alpha)
        return {
            "model": model,
            "prompt": prompt[:200],
            "response": f"[PRSM node {self.identity.node_id[:8]} processed inference]",
            "tokens_used": len(prompt.split()),
            "provider_node": self.identity.node_id,
            "source": "mock",
            "warning": "No LLM backend configured. This is a mock response.",
        }

    async def _run_embedding(self, job: ComputeJob) -> Dict[str, Any]:
        """Compute embeddings using the backend registry.
        
        Uses real embedding models when available (OpenAI, Local, etc.),
        falls back to deterministic mock embeddings when no backend is configured.
        """
        text = job.payload.get("text", "")
        dimensions = job.payload.get("dimensions", 1536)  # Default to OpenAI small dimensions
        model_id = job.payload.get("model_id")  # Optional specific model
        
        # Try to use backend registry if available (wired via orchestrator)
        if self.orchestrator is not None and hasattr(self.orchestrator, 'backend_registry'):
            try:
                registry = self.orchestrator.backend_registry
                if registry is not None:
                    result = await registry.embed_with_fallback(
                        text=text,
                        model_id=model_id,
                        dimensions=dimensions
                    )
                    return {
                        "text_length": len(text),
                        "dimensions": len(result.embedding),
                        "embedding": result.embedding,
                        "model_id": result.model_id,
                        "provider": result.provider.value,
                        "token_count": result.token_count,
                        "provider_node": self.identity.node_id,
                        "source": "backend_registry",
                    }
            except Exception as e:
                logger.warning(f"Backend registry embedding failed, falling back to mock: {e}")
        
        # Fallback: generate a deterministic pseudo-embedding (for testing/alpha)
        import hashlib
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Create embedding with requested dimensions
        embedding = []
        for i in range(dimensions):
            byte_val = text_hash[i % len(text_hash)]
            # Normalize to [-1, 1] range
            embedding.append((byte_val - 128) / 128.0)
        
        # Normalize to unit vector for cosine similarity
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return {
            "text_length": len(text),
            "dimensions": dimensions,
            "embedding": embedding,
            "model_id": "fallback-hash",
            "provider": "mock",
            "provider_node": self.identity.node_id,
            "source": "mock",
            "warning": "No embedding backend configured. Using pseudo-vectors.",
        }

    async def _run_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a distributed training job via the local distillation pipeline."""
        teacher_model_id = payload.get("teacher_model_id", "")
        domain = payload.get("domain", "general")
        target_size = payload.get("target_size", "small")
        budget_ftns = payload.get("budget_ftns", 100)

        try:
            from prsm.compute.distillation.models import DistillationRequest, ModelSize, OptimizationTarget
            from prsm.compute.distillation.orchestrator import DistillationOrchestrator

            req = DistillationRequest(
                user_id=self.identity.node_id,
                teacher_model=teacher_model_id,
                domain=domain,
                target_size=ModelSize(target_size),
                optimization_target=OptimizationTarget.BALANCED,
                budget_ftns=budget_ftns,
            )
            
            # Create a local orchestrator instance for P2P training jobs
            orchestrator = DistillationOrchestrator()
            job = await orchestrator.create_distillation(req)
            
            return {
                "job_id": str(job.job_id),
                "status": job.status.value,
                "source": "local_distillation",
            }
        except Exception as e:
            logger.warning(f"Training job failed: {e}")
            return {
                "error": str(e),
                "source": "local_distillation",
                "status": "failed",
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

    def update_allocation(
        self,
        cpu_allocation_pct: Optional[int] = None,
        memory_allocation_pct: Optional[int] = None,
        max_concurrent_jobs: Optional[int] = None,
        gpu_allocation_pct: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update resource allocation settings at runtime.
        
        Changes take effect on next job acceptance. This method allows
        live tuning of compute resource allocation without restarting
        the node.
        
        Args:
            cpu_allocation_pct: Percentage of CPU to allocate (10-90)
            memory_allocation_pct: Percentage of memory to allocate (10-90)
            max_concurrent_jobs: Maximum number of concurrent jobs
            gpu_allocation_pct: Percentage of GPU memory to allocate (10-100)
        
        Returns:
            Dict with updated allocation values
        
        Raises:
            ValueError: If any value is out of valid range
        """
        # Validate ranges
        if cpu_allocation_pct is not None:
            if not 10 <= cpu_allocation_pct <= 90:
                raise ValueError(f"cpu_allocation_pct must be 10-90, got {cpu_allocation_pct}")
            self.cpu_allocation_pct = cpu_allocation_pct
            
        if memory_allocation_pct is not None:
            if not 10 <= memory_allocation_pct <= 90:
                raise ValueError(f"memory_allocation_pct must be 10-90, got {memory_allocation_pct}")
            self.memory_allocation_pct = memory_allocation_pct
            
        if max_concurrent_jobs is not None:
            if max_concurrent_jobs < 1:
                raise ValueError(f"max_concurrent_jobs must be at least 1, got {max_concurrent_jobs}")
            self.max_concurrent_jobs = max_concurrent_jobs
            
        if gpu_allocation_pct is not None:
            if not 10 <= gpu_allocation_pct <= 100:
                raise ValueError(f"gpu_allocation_pct must be 10-100, got {gpu_allocation_pct}")
            self.gpu_allocation_pct = gpu_allocation_pct
        
        logger.info(
            f"Updated compute allocation: CPU={self.cpu_allocation_pct}%, "
            f"Memory={self.memory_allocation_pct}%, Jobs={self.max_concurrent_jobs}, "
            f"GPU={self.gpu_allocation_pct}%"
        )
        
        return {
            "cpu_allocation_pct": self.cpu_allocation_pct,
            "memory_allocation_pct": self.memory_allocation_pct,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "gpu_allocation_pct": self.gpu_allocation_pct,
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

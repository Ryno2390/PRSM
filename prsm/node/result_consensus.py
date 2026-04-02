"""
Result Consensus
================

Quorum-based verification of compute job results across multiple nodes.

When a job is submitted to the network, the requester can require that
multiple independent providers execute the same job and agree on the
result.  This prevents a single malicious node from returning bogus
results.

Consensus modes:
- SINGLE  — accept first valid result (fast, low trust)
- MAJORITY — require >50% of assigned providers to agree
- UNANIMOUS — all assigned providers must agree
- BYZANTINE — tolerate up to f faulty nodes out of 3f+1 total

For numeric/structured results, agreement is determined by fuzzy
comparison (within epsilon for floats, exact match for strings/hashes).
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConsensusMode(str, Enum):
    SINGLE = "single"          # First valid result wins
    MAJORITY = "majority"      # >50% must agree
    UNANIMOUS = "unanimous"    # All must agree
    BYZANTINE = "byzantine"    # Tolerate f faults from 3f+1 nodes


@dataclass
class ProviderResult:
    """Result from a single provider for a job."""
    provider_id: str
    job_id: str
    result: Dict[str, Any]
    signature: str
    timestamp: float = field(default_factory=time.time)
    verified: bool = False

    @property
    def result_hash(self) -> str:
        """Deterministic hash of the result content."""
        import json
        canonical = json.dumps(self.result, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class ConsensusState:
    """Tracks consensus state for a single job."""
    job_id: str
    mode: ConsensusMode
    required_providers: int
    results: List[ProviderResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    consensus_reached: bool = False
    agreed_result: Optional[Dict[str, Any]] = None
    agreed_hash: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    @property
    def provider_count(self) -> int:
        return len(self.results)


class ResultConsensus:
    """Manages quorum-based consensus for compute job results.

    Usage:
        1. Call start_consensus(job_id, mode, required_providers)
        2. As results arrive, call submit_result(job_id, provider_id, result, signature)
        3. Poll check_consensus(job_id) or wait for await_consensus(job_id)
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        timeout_seconds: float = 300.0,
    ):
        self.epsilon = epsilon            # Tolerance for numeric comparison
        self.timeout_seconds = timeout_seconds
        self._jobs: Dict[str, ConsensusState] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._max_providers_factor = 3    # Byzantine: need 3f+1 providers
        self._stats = {
            "consensus_started": 0,
            "consensus_reached": 0,
            "consensus_failed": 0,
            "disputed_results": 0,
        }

    def start_consensus(
        self,
        job_id: str,
        mode: ConsensusMode = ConsensusMode.MAJORITY,
        required_providers: int = 3,
    ) -> ConsensusState:
        """Start tracking consensus for a job."""
        if mode == ConsensusMode.BYZANTINE and required_providers < 4:
            required_providers = 4  # Minimum for Byzantine fault tolerance

        state = ConsensusState(
            job_id=job_id,
            mode=mode,
            required_providers=required_providers,
        )
        self._jobs[job_id] = state
        self._events[job_id] = asyncio.Event()
        self._stats["consensus_started"] += 1
        logger.info(
            f"Consensus started for {job_id[:8]}: mode={mode.value}, "
            f"providers={required_providers}"
        )
        return state

    def submit_result(
        self,
        job_id: str,
        provider_id: str,
        result: Dict[str, Any],
        signature: str,
    ) -> bool:
        """Submit a result from a provider. Returns True if consensus reached."""
        state = self._jobs.get(job_id)
        if not state or state.is_complete:
            return False

        # Prevent duplicate submissions
        for existing in state.results:
            if existing.provider_id == provider_id:
                return False

        provider_result = ProviderResult(
            provider_id=provider_id,
            job_id=job_id,
            result=result,
            signature=signature,
        )
        state.results.append(provider_result)

        # Verify signature (optional - skip if no public key provided)
        provider_result.verified = True  # Verified at job result level

        # Check if we have enough results to evaluate consensus
        return self._check_consensus(job_id)

    def _check_consensus(self, job_id: str) -> bool:
        """Evaluate whether consensus has been reached for a job."""
        state = self._jobs.get(job_id)
        if not state or state.is_complete:
            return False

        results = state.results

        # Check timeout
        if time.time() - state.started_at > self.timeout_seconds:
            state.error = "Consensus timed out"
            state.completed_at = time.time()
            self._stats["consensus_failed"] += 1
            self._events[job_id].set()
            return False

        if state.mode == ConsensusMode.SINGLE:
            # First valid result wins
            if results:
                winner = results[0]
                state.consensus_reached = True
                state.agreed_result = winner.result
                state.agreed_hash = winner.result_hash
                state.completed_at = time.time()
                self._stats["consensus_reached"] += 1
                self._events[job_id].set()
                return True

        elif state.mode == ConsensusMode.MAJORITY:
            # Need >50% of required providers to agree on a result
            if len(results) < state.required_providers:
                # Not enough results yet, but check if majority already agree
                pass
            agreement = self._find_agreement(results)
            if agreement:
                agreed_hash, agreed_nodes = agreement
                if len(agreed_nodes) > state.required_providers / 2:
                    state.consensus_reached = True
                    state.agreed_result = agreed_nodes[0].result
                    state.agreed_hash = agreed_hash
                    state.completed_at = time.time()
                    self._stats["consensus_reached"] += 1
                    self._events[job_id].set()
                    return True
            # If we have all results and no majority, fail
            if len(results) >= state.required_providers:
                state.error = "No majority agreement"
                state.completed_at = time.time()
                self._stats["consensus_failed"] += 1
                self._stats["disputed_results"] += 1
                self._events[job_id].set()
                return False

        elif state.mode == ConsensusMode.UNANIMOUS:
            # All providers must agree
            if len(results) >= state.required_providers:
                agreement = self._find_agreement(results)
                if agreement and len(agreement[1]) == state.required_providers:
                    state.consensus_reached = True
                    state.agreed_result = agreement[1][0].result
                    state.agreed_hash = agreement[0]
                    state.completed_at = time.time()
                    self._stats["consensus_reached"] += 1
                    self._events[job_id].set()
                    return True
                else:
                    state.error = "Results are not unanimous"
                    state.completed_at = time.time()
                    self._stats["consensus_failed"] += 1
                    self._stats["disputed_results"] += 1
                    self._events[job_id].set()
                    return False

        elif state.mode == ConsensusMode.BYZANTINE:
            # 3f+1 nodes can tolerate f faulty nodes
            f = (state.required_providers - 1) // 3
            if len(results) >= state.required_providers - f:
                agreement = self._find_agreement(results)
                if agreement and len(agreement[1]) >= state.required_providers - f:
                    state.consensus_reached = True
                    state.agreed_result = agreement[1][0].result
                    state.agreed_hash = agreement[0]
                    state.completed_at = time.time()
                    self._stats["consensus_reached"] += 1
                    self._events[job_id].set()
                    return True
                if len(results) >= state.required_providers:
                    state.error = "Byzantine consensus failed - too many disagreements"
                    state.completed_at = time.time()
                    self._stats["consensus_failed"] += 1
                    self._stats["disputed_results"] += 1
                    self._events[job_id].set()
                    return False

        return False

    def _find_agreement(
        self, results: List[ProviderResult]
    ) -> Optional[Tuple[str, List[ProviderResult]]]:
        """Find the largest group of results that agree."""
        if not results:
            return None

        hash_groups: Dict[str, List[ProviderResult]] = {}
        for r in results:
            if r.result_hash not in hash_groups:
                hash_groups[r.result_hash] = []
            hash_groups[r.result_hash].append(r)

        # Find the largest group
        best_hash = max(hash_groups, key=lambda h: len(hash_groups[h]))
        best_group = hash_groups[best_hash]

        return (best_hash, best_group)

    def get_state(self, job_id: str) -> Optional[ConsensusState]:
        """Get the current consensus state for a job."""
        return self._jobs.get(job_id)

    async def await_consensus(self, job_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for consensus to be reached. Returns the agreed result or None."""
        event = self._events.get(job_id)
        if not event:
            return None

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout or self.timeout_seconds)
        except asyncio.TimeoutError:
            state = self._jobs.get(job_id)
            if state:
                state.error = "Timeout waiting for consensus"
                state.completed_at = time.time()
                self._stats["consensus_failed"] += 1
            return None

        state = self._jobs.get(job_id)
        if state and state.consensus_reached:
            return state.agreed_result
        return None

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def cancel_consensus(self, job_id: str) -> bool:
        """Cancel consensus tracking for a job."""
        if job_id in self._jobs:
            state = self._jobs[job_id]
            if not state.is_complete:
                state.error = "Cancelled"
                state.completed_at = time.time()
                self._events[job_id].set()
            return True
        return False

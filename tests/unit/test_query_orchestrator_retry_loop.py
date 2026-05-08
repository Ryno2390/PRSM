"""QueryOrchestrator retry-loop shell — A4 closure.

Per `docs/2026-05-07-aggregator-selector-threat-model.md` §A4:
- Mitigation 1: bounded selection retries (`MAX_AGG_RETRIES = 3`)
- Mitigation 2: `record_preemption` on dropped aggregator
- Mitigation 3: escrow refund on no-aggregator-after-retries

This module wraps `QueryOrchestrator.dispatch_query` with retry logic
that consumes the typed exceptions sub-modules raise:

  - `AggregationCommitMismatchError` → slash aggregator + retry
  - `PartialResultIntegrityError`    → record preemption + retry
  - `InsufficientCandidatesError`    → no retry (no aggregator to retry against)
  - `ValueError` (programming error) → no retry
  - any other exception              → no retry (fail-open is wrong here)

After `policy.max_retries` failures, raises `QueryRetriesExhaustedError`
and triggers an escrow refund.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pytest

from prsm.compute.agents.instruction_set import (
    AgentInstruction,
    AgentOp,
    InstructionManifest,
)
from prsm.compute.query_orchestrator import (
    AggregatedResult,
    AggregationCommit,
    AggregationCommitMismatchError,
    InsufficientCandidatesError,
    PartialResult,
    PartialResultIntegrityError,
    QueryOrchestrator,
    StakedNode,
)
from prsm.compute.query_orchestrator.retry_loop import (
    EscrowClient,
    PreemptionRecorder,
    QueryRetriesExhaustedError,
    RetryPolicy,
    SlashClient,
    dispatch_with_retries,
)


# ──────────────────────────────────────────────────────────────────────
# Stubs
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _StubOrchestrator:
    """Test-only stand-in for QueryOrchestrator. Returns canned results
    or raises canned exceptions on each call."""
    behaviors: list  # list of (AggregatedResult|Exception)
    calls: int = 0

    async def dispatch_query(self, **kwargs) -> AggregatedResult:
        idx = self.calls
        self.calls += 1
        if idx >= len(self.behaviors):
            raise RuntimeError("stub exhausted — test specified too few behaviors")
        b = self.behaviors[idx]
        if isinstance(b, Exception):
            raise b
        return b


@dataclass
class _StubSlashClient:
    slashed: list = field(default_factory=list)

    def slash_aggregator(
        self, pubkey_hash: bytes, batch_id: str, reason: str
    ) -> None:
        self.slashed.append((pubkey_hash, batch_id, reason))


@dataclass
class _StubPreemption:
    recorded: list = field(default_factory=list)

    def record_preemption(self, provider_id: str) -> None:
        self.recorded.append(provider_id)


@dataclass
class _StubEscrow:
    refunded: list = field(default_factory=list)

    async def refund(self, query_id: bytes, prompter: str):
        self.refunded.append((query_id, prompter))
        return None


def _make_result(query_id: bytes = b"q" * 32) -> AggregatedResult:
    return AggregatedResult(
        query_id=query_id,
        payload=b"ok",
        aggregator_node_id="agg-1",
        contributing_shards=("prsm:s-0",),
    )


def _commit_mismatch() -> AggregationCommitMismatchError:
    return AggregationCommitMismatchError("test: commit mismatch")


def _partial_integrity() -> PartialResultIntegrityError:
    return PartialResultIntegrityError("test: dp_noise marker missing")


# ──────────────────────────────────────────────────────────────────────
# Happy path — single attempt
# ──────────────────────────────────────────────────────────────────────


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_single_attempt_no_retry(self):
        orch = _StubOrchestrator(behaviors=[_make_result()])
        slash, preempt, escrow = _StubSlashClient(), _StubPreemption(), _StubEscrow()
        result = await dispatch_with_retries(
            orchestrator=orch,
            query="x", prompter_node_id="p", query_id=b"q" * 32,
            slash_client=slash, preemption_recorder=preempt,
            escrow_client=escrow,
        )
        assert isinstance(result, AggregatedResult)
        assert orch.calls == 1
        assert slash.slashed == []
        assert preempt.recorded == []
        assert escrow.refunded == []


# ──────────────────────────────────────────────────────────────────────
# A9 commit-mismatch → slash + retry
# ──────────────────────────────────────────────────────────────────────


class TestCommitMismatchRetry:
    @pytest.mark.asyncio
    async def test_one_mismatch_then_success(self):
        orch = _StubOrchestrator(
            behaviors=[_commit_mismatch(), _make_result()],
        )
        slash, preempt, escrow = _StubSlashClient(), _StubPreemption(), _StubEscrow()
        result = await dispatch_with_retries(
            orchestrator=orch,
            query="x", prompter_node_id="p", query_id=b"q" * 32,
            slash_client=slash, preemption_recorder=preempt,
            escrow_client=escrow,
        )
        assert isinstance(result, AggregatedResult)
        assert orch.calls == 2
        assert len(slash.slashed) == 1
        # No escrow refund — succeeded on retry.
        assert escrow.refunded == []

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_refunds_escrow(self):
        orch = _StubOrchestrator(behaviors=[_commit_mismatch()] * 5)
        slash, preempt, escrow = _StubSlashClient(), _StubPreemption(), _StubEscrow()
        with pytest.raises(QueryRetriesExhaustedError):
            await dispatch_with_retries(
                orchestrator=orch,
                query="x", prompter_node_id="p", query_id=b"q" * 32,
                slash_client=slash, preemption_recorder=preempt,
                escrow_client=escrow,
                policy=RetryPolicy(max_retries=3),
            )
        # Initial attempt + 3 retries = 4 calls.
        assert orch.calls == 4
        # Each commit-mismatch slashed.
        assert len(slash.slashed) == 4
        # Escrow refunded once after exhaustion.
        assert len(escrow.refunded) == 1


# ──────────────────────────────────────────────────────────────────────
# A5 partial-integrity → preemption + retry
# ──────────────────────────────────────────────────────────────────────


class TestPartialIntegrityRetry:
    @pytest.mark.asyncio
    async def test_one_integrity_then_success(self):
        orch = _StubOrchestrator(
            behaviors=[_partial_integrity(), _make_result()],
        )
        slash, preempt, escrow = _StubSlashClient(), _StubPreemption(), _StubEscrow()
        result = await dispatch_with_retries(
            orchestrator=orch,
            query="x", prompter_node_id="p", query_id=b"q" * 32,
            slash_client=slash, preemption_recorder=preempt,
            escrow_client=escrow,
        )
        assert isinstance(result, AggregatedResult)
        assert orch.calls == 2
        # Source-agent preemption recorded; aggregator NOT slashed
        # (the failure was upstream of the aggregator).
        assert len(preempt.recorded) == 1
        assert slash.slashed == []
        assert escrow.refunded == []


# ──────────────────────────────────────────────────────────────────────
# Non-retryable errors propagate without retry
# ──────────────────────────────────────────────────────────────────────


class TestNonRetryablePropagation:
    @pytest.mark.asyncio
    async def test_insufficient_candidates_not_retried(self):
        orch = _StubOrchestrator(
            behaviors=[InsufficientCandidatesError("no eligible aggregators")],
        )
        slash, preempt, escrow = _StubSlashClient(), _StubPreemption(), _StubEscrow()
        with pytest.raises(InsufficientCandidatesError):
            await dispatch_with_retries(
                orchestrator=orch,
                query="x", prompter_node_id="p", query_id=b"q" * 32,
                slash_client=slash, preemption_recorder=preempt,
                escrow_client=escrow,
            )
        assert orch.calls == 1
        # Escrow STILL refunded — prompter paid for nothing.
        assert len(escrow.refunded) == 1

    @pytest.mark.asyncio
    async def test_value_error_not_retried(self):
        orch = _StubOrchestrator(behaviors=[ValueError("query is empty")])
        slash, preempt, escrow = _StubSlashClient(), _StubPreemption(), _StubEscrow()
        with pytest.raises(ValueError):
            await dispatch_with_retries(
                orchestrator=orch,
                query="", prompter_node_id="p", query_id=b"q" * 32,
                slash_client=slash, preemption_recorder=preempt,
                escrow_client=escrow,
            )
        assert orch.calls == 1
        # ValueError = caller bug; no escrow refund (the caller can
        # decide what to do).
        assert escrow.refunded == []


# ──────────────────────────────────────────────────────────────────────
# Mixed retry path — commit + integrity + success
# ──────────────────────────────────────────────────────────────────────


class TestMixedRetries:
    @pytest.mark.asyncio
    async def test_commit_then_integrity_then_success(self):
        orch = _StubOrchestrator(
            behaviors=[_commit_mismatch(), _partial_integrity(), _make_result()],
        )
        slash, preempt, escrow = _StubSlashClient(), _StubPreemption(), _StubEscrow()
        result = await dispatch_with_retries(
            orchestrator=orch,
            query="x", prompter_node_id="p", query_id=b"q" * 32,
            slash_client=slash, preemption_recorder=preempt,
            escrow_client=escrow,
            policy=RetryPolicy(max_retries=3),
        )
        assert isinstance(result, AggregatedResult)
        assert orch.calls == 3
        assert len(slash.slashed) == 1
        assert len(preempt.recorded) == 1
        assert escrow.refunded == []


# ──────────────────────────────────────────────────────────────────────
# Policy contract pin
# ──────────────────────────────────────────────────────────────────────


class TestPolicyContractPin:
    """Pin the RetryPolicy shape so the future Foundation-council
    ratification (governance parameter `MAX_AGG_RETRIES`) knows where
    to land."""

    def test_default_max_retries_is_three(self):
        # Per threat-model §"Open governance questions" item 3.
        assert RetryPolicy().max_retries == 3

    def test_max_retries_is_overridable(self):
        p = RetryPolicy(max_retries=10)
        assert p.max_retries == 10

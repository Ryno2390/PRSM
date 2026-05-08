"""QueryOrchestrator swarm_runner — full query round-trip orchestration.

Closes the data-query path: takes the decomposer's manifest, the
shard_finder's candidates, and the aggregator_selector's chosen
aggregator, and produces a final `AggregatedResult` for the prompter.

What the runner owns:
  - Per-shard agent fan-out via a pluggable `SwarmDispatcher` Protocol
    (production wires `prsm/compute/agents/dispatcher.py::AgentDispatcher`)
  - DP-noise enforcement on partials BEFORE they reach the aggregator
    (threat-model A5 mitigation 2 — partials must be DP-noised by the
    source agent; the runner verifies the marker is set)
  - Aggregator handoff via a pluggable `AggregatorClient` Protocol
  - A9 commit-verify: the aggregator's pre-commit hash MUST match
    the plaintext it delivers — mismatch raises and the orchestrator
    routes to slash via ReputationTracker.record_slash

What the runner does NOT own:
  - Aggregator selection (aggregator_selector.py)
  - Shard discovery (shard_finder.py)
  - Query decomposition (decomposer.py)
  - Settlement payments (settlement registry)
  - Escrow refunds on failure (orchestrator's retry loop)

Threat-model coverage: A5 (DP noise) + A9 (commit-verify) anchor
into the runner. A1/A2/A6/A7/A8/A10 land at aggregator_selector.
A3/A4 land at the orchestrator's retry-loop shell.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

import pytest

from prsm.compute.agents.instruction_set import (
    AgentInstruction,
    AgentOp,
    InstructionManifest,
)
from prsm.compute.query_orchestrator import (
    AggregationCommit,
    AggregationCommitMismatchError,
    AggregatedResult,
    PartialResult,
    PartialResultIntegrityError,
    ShardCandidate,
    StakedNode,
    run_swarm,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _agg_node(*, node_id: str = "agg-1", has_tee: bool = True) -> StakedNode:
    return StakedNode(
        node_id=node_id,
        pubkey_hash=hashlib.sha256(node_id.encode()).digest(),
        stake_amount_ftns=1_000,
        tier="T2",
        has_tee=has_tee,
        reputation_score=1.0,
    )


def _manifest() -> InstructionManifest:
    return InstructionManifest(
        query="count records",
        instructions=[AgentInstruction(op=AgentOp.COUNT)],
    )


def _shards(n: int = 3) -> list[ShardCandidate]:
    return [
        ShardCandidate(
            cid=f"prsm:shard-{i}",
            similarity=0.9 - i * 0.01,
            creator_id=f"creator-{i}",
            holder_node_ids=(f"node-{i}",),
        )
        for i in range(n)
    ]


@dataclass
class _StubDispatcher:
    """Test-only SwarmDispatcher. Returns canned partials."""
    canned_partials: list[PartialResult]
    fan_out_calls: int = 0

    async def fan_out(
        self,
        manifest: InstructionManifest,
        shards,
    ) -> list[PartialResult]:
        self.fan_out_calls += 1
        return self.canned_partials


@dataclass
class _StubAggregatorClient:
    """Test-only AggregatorClient. Returns canned plaintext + commit."""
    plaintext: bytes
    commit_digest: bytes | None = None  # None means use sha256(plaintext)
    received_partials: list = None  # type: ignore

    def __post_init__(self):
        self.received_partials = []

    async def aggregate(
        self,
        aggregator: StakedNode,
        manifest: InstructionManifest,
        partials,
        query_id: bytes,
    ) -> tuple[bytes, AggregationCommit]:
        self.received_partials = list(partials)
        digest = self.commit_digest if self.commit_digest is not None else (
            hashlib.sha256(self.plaintext).digest()
        )
        commit = AggregationCommit(
            query_id=query_id,
            aggregator_pubkey_hash=aggregator.pubkey_hash,
            result_digest=digest,
        )
        return self.plaintext, commit


def _make_partial(*, shard_cid: str, payload: bytes = b"x", dp_applied: bool = True) -> PartialResult:
    return PartialResult(
        shard_cid=shard_cid,
        payload=payload,
        agent_signature=b"\x00" * 64,
        creator_id="some-creator",
        dp_noise_applied=dp_applied,
    )


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_full_round_trip(self):
        partials = [_make_partial(shard_cid=f"prsm:shard-{i}") for i in range(3)]
        plaintext = b"final aggregated answer"
        dispatcher = _StubDispatcher(canned_partials=partials)
        agg_client = _StubAggregatorClient(plaintext=plaintext)

        result = await run_swarm(
            manifest=_manifest(),
            shards=_shards(3),
            aggregator=_agg_node(),
            dispatcher=dispatcher,
            aggregator_client=agg_client,
            query_id=b"q" * 32,
        )

        assert isinstance(result, AggregatedResult)
        assert result.payload == plaintext
        assert result.aggregator_node_id == "agg-1"
        assert dispatcher.fan_out_calls == 1
        # Aggregator received our partials.
        assert len(agg_client.received_partials) == 3

    @pytest.mark.asyncio
    async def test_contributing_shards_recorded(self):
        partials = [_make_partial(shard_cid=f"prsm:shard-{i}") for i in range(2)]
        agg_client = _StubAggregatorClient(plaintext=b"ok")
        result = await run_swarm(
            manifest=_manifest(),
            shards=_shards(2),
            aggregator=_agg_node(),
            dispatcher=_StubDispatcher(canned_partials=partials),
            aggregator_client=agg_client,
            query_id=b"q" * 32,
        )
        assert set(result.contributing_shards) == {"prsm:shard-0", "prsm:shard-1"}


# ──────────────────────────────────────────────────────────────────────
# A5 — DP-noise enforcement
# ──────────────────────────────────────────────────────────────────────


class TestA5DpNoiseEnforcement:
    """Threat-model A5 mitigation 2: per-shard partials MUST already
    be DP-noised by the source agent before reaching the aggregator.
    The runner verifies the marker — refuses to forward un-noised
    partials to the aggregator."""

    @pytest.mark.asyncio
    async def test_un_noised_partial_raises(self):
        partials = [
            _make_partial(shard_cid="prsm:shard-0", dp_applied=True),
            _make_partial(shard_cid="prsm:shard-1", dp_applied=False),  # bad
        ]
        with pytest.raises(PartialResultIntegrityError, match="dp_noise"):
            await run_swarm(
                manifest=_manifest(),
                shards=_shards(2),
                aggregator=_agg_node(),
                dispatcher=_StubDispatcher(canned_partials=partials),
                aggregator_client=_StubAggregatorClient(plaintext=b"ok"),
                query_id=b"q" * 32,
            )

    @pytest.mark.asyncio
    async def test_all_noised_partials_pass(self):
        partials = [_make_partial(shard_cid=f"x-{i}", dp_applied=True) for i in range(3)]
        result = await run_swarm(
            manifest=_manifest(),
            shards=_shards(3),
            aggregator=_agg_node(),
            dispatcher=_StubDispatcher(canned_partials=partials),
            aggregator_client=_StubAggregatorClient(plaintext=b"ok"),
            query_id=b"q" * 32,
        )
        assert result.payload == b"ok"


# ──────────────────────────────────────────────────────────────────────
# A9 — commit-verify
# ──────────────────────────────────────────────────────────────────────


class TestA9CommitVerify:
    """Threat-model A9: the aggregator's pre-commit hash MUST match
    the plaintext they finally deliver. Mismatch routes to slash
    upstream — the runner just raises so the orchestrator can
    consume the exception."""

    @pytest.mark.asyncio
    async def test_mismatched_commit_raises(self):
        partials = [_make_partial(shard_cid=f"x-{i}") for i in range(2)]
        # Aggregator delivers `plaintext` but commits to a different digest.
        wrong_digest = hashlib.sha256(b"different-result").digest()
        agg_client = _StubAggregatorClient(
            plaintext=b"actual-result",
            commit_digest=wrong_digest,
        )
        with pytest.raises(AggregationCommitMismatchError):
            await run_swarm(
                manifest=_manifest(),
                shards=_shards(2),
                aggregator=_agg_node(),
                dispatcher=_StubDispatcher(canned_partials=partials),
                aggregator_client=agg_client,
                query_id=b"q" * 32,
            )

    @pytest.mark.asyncio
    async def test_commit_query_id_mismatch_raises(self):
        # Aggregator commits to a different query_id than we sent.
        # Forensic-traceability invariant.
        partials = [_make_partial(shard_cid="x-0")]
        agg_client = _StubAggregatorClient(plaintext=b"r")
        # Override aggregate() to return a commit with the wrong query_id.
        original = agg_client.aggregate

        async def bad_aggregate(aggregator, manifest, partials, query_id):
            plaintext, commit = await original(aggregator, manifest, partials, query_id)
            # Substitute a different query_id on the commit.
            tampered = AggregationCommit(
                query_id=b"\xff" * 32,
                aggregator_pubkey_hash=commit.aggregator_pubkey_hash,
                result_digest=commit.result_digest,
            )
            return plaintext, tampered

        agg_client.aggregate = bad_aggregate  # type: ignore
        with pytest.raises(AggregationCommitMismatchError, match="query_id"):
            await run_swarm(
                manifest=_manifest(),
                shards=_shards(1),
                aggregator=_agg_node(),
                dispatcher=_StubDispatcher(canned_partials=partials),
                aggregator_client=agg_client,
                query_id=b"q" * 32,
            )


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    @pytest.mark.asyncio
    async def test_empty_shards_raises(self):
        with pytest.raises(ValueError, match="shards"):
            await run_swarm(
                manifest=_manifest(),
                shards=[],
                aggregator=_agg_node(),
                dispatcher=_StubDispatcher(canned_partials=[]),
                aggregator_client=_StubAggregatorClient(plaintext=b"x"),
                query_id=b"q" * 32,
            )

    @pytest.mark.asyncio
    async def test_no_partials_returned_raises(self):
        # Dispatcher returned empty — nothing to aggregate.
        with pytest.raises(ValueError, match="partials"):
            await run_swarm(
                manifest=_manifest(),
                shards=_shards(2),
                aggregator=_agg_node(),
                dispatcher=_StubDispatcher(canned_partials=[]),
                aggregator_client=_StubAggregatorClient(plaintext=b"x"),
                query_id=b"q" * 32,
            )

    @pytest.mark.asyncio
    async def test_query_id_must_be_thirty_two_bytes(self):
        partials = [_make_partial(shard_cid="x-0")]
        with pytest.raises(ValueError, match="query_id"):
            await run_swarm(
                manifest=_manifest(),
                shards=_shards(1),
                aggregator=_agg_node(),
                dispatcher=_StubDispatcher(canned_partials=partials),
                aggregator_client=_StubAggregatorClient(plaintext=b"x"),
                query_id=b"too-short",
            )

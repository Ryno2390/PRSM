"""QueryOrchestrator composition — ties the 4 core sub-modules together.

End-to-end integration at the orchestrator level. Each call to
`QueryOrchestrator.dispatch_query` runs:

    decompose_query  →  manifest
    find_relevant_shards  →  shards
    select_aggregator  →  aggregator
    run_swarm  →  AggregatedResult

Every threat-model invariant the sub-modules enforce flows through
unchanged. This file just verifies the wiring chain and that the
orchestrator's `dispatch_query` honours each sub-module's error
contract end-to-end.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Sequence

import pytest

from prsm.compute.agents.instruction_set import InstructionManifest
from prsm.compute.query_orchestrator import (
    AggregatedResult,
    AggregationCommit,
    AggregationCommitMismatchError,
    DecomposerOutputError,
    InsufficientCandidatesError,
    PartialResult,
    PartialResultIntegrityError,
    QueryOrchestrator,
    ShardCandidate,
    StakedNode,
)


# ──────────────────────────────────────────────────────────────────────
# Stubs
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _StubIndex:
    canned: list[tuple[str, float, str]]

    def find_top_k(self, query: str, k: int):
        return self.canned[:k]


@dataclass
class _StubDispatcher:
    canned_partials: list[PartialResult]

    async def fan_out(self, manifest, shards):
        return self.canned_partials


@dataclass
class _StubAggClient:
    plaintext: bytes
    commit_digest: bytes | None = None

    async def aggregate(self, aggregator, manifest, partials, query_id):
        digest = self.commit_digest or hashlib.sha256(self.plaintext).digest()
        commit = AggregationCommit(
            query_id=query_id,
            aggregator_pubkey_hash=aggregator.pubkey_hash,
            result_digest=digest,
        )
        return self.plaintext, commit


def _make_node(*, node_id: str, has_tee: bool = True) -> StakedNode:
    return StakedNode(
        node_id=node_id,
        pubkey_hash=hashlib.sha256(node_id.encode()).digest(),
        stake_amount_ftns=1_000,
        tier="T2",
        has_tee=has_tee,
        reputation_score=1.0,
    )


def _make_partial(cid: str, *, dp_applied: bool = True) -> PartialResult:
    return PartialResult(
        shard_cid=cid,
        payload=b"x",
        agent_signature=b"\x00" * 64,
        creator_id="c",
        dp_noise_applied=dp_applied,
    )


def _build(
    *,
    semantic_canned: list | None = None,
    partials: list | None = None,
    plaintext: bytes = b"r",
    candidate_nodes: list | None = None,
    commit_digest: bytes | None = None,
) -> QueryOrchestrator:
    """Construct an orchestrator wired to test stubs."""
    if semantic_canned is None:
        semantic_canned = [("prsm:s-0", 0.9, "c0"), ("prsm:s-1", 0.85, "c1")]
    if partials is None:
        partials = [_make_partial(c) for c, _, _ in semantic_canned]
    if candidate_nodes is None:
        candidate_nodes = [_make_node(node_id=f"node-{i}") for i in range(3)]
    return QueryOrchestrator(
        semantic_index=_StubIndex(canned=semantic_canned),
        dispatcher=_StubDispatcher(canned_partials=partials),
        aggregator_client=_StubAggClient(
            plaintext=plaintext, commit_digest=commit_digest
        ),
        candidate_pool_provider=lambda: tuple(candidate_nodes),
        beacon_provider=lambda: hashlib.sha256(b"daily-beacon").digest(),
    )


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_full_compose_round_trip(self):
        orch = _build()
        result = await orch.dispatch_query(
            query="count records",
            prompter_node_id="prompter-x",
            query_id=b"q" * 32,
        )
        assert isinstance(result, AggregatedResult)
        assert result.payload == b"r"
        assert result.aggregator_node_id.startswith("node-")
        assert len(result.contributing_shards) >= 1


# ──────────────────────────────────────────────────────────────────────
# Error propagation per sub-module contract
# ──────────────────────────────────────────────────────────────────────


class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_empty_query_raises_value_error(self):
        orch = _build()
        with pytest.raises(ValueError, match="empty"):
            await orch.dispatch_query(
                query="", prompter_node_id="x", query_id=b"q" * 32
            )

    @pytest.mark.asyncio
    async def test_no_candidate_aggregators_raises(self):
        # Candidate pool = only the prompter → fail-closed (A2).
        only_prompter = _make_node(node_id="alice")
        orch = _build(candidate_nodes=[only_prompter])
        with pytest.raises(InsufficientCandidatesError):
            await orch.dispatch_query(
                query="count records",
                prompter_node_id="alice",
                query_id=b"q" * 32,
            )

    @pytest.mark.asyncio
    async def test_no_relevant_shards_raises(self):
        # Semantic index returns nothing matching threshold.
        orch = _build(
            semantic_canned=[("prsm:s-0", 0.05, "c0")],  # below threshold
        )
        with pytest.raises(ValueError, match="shards|partials"):
            await orch.dispatch_query(
                query="anything", prompter_node_id="x", query_id=b"q" * 32
            )

    @pytest.mark.asyncio
    async def test_dp_noise_violation_propagates(self):
        # One partial missing the dp_noise marker — A5 must surface
        # all the way up to the caller.
        orch = _build(
            partials=[
                _make_partial("prsm:s-0", dp_applied=True),
                _make_partial("prsm:s-1", dp_applied=False),  # bad
            ],
        )
        with pytest.raises(PartialResultIntegrityError):
            await orch.dispatch_query(
                query="count records",
                prompter_node_id="x",
                query_id=b"q" * 32,
            )

    @pytest.mark.asyncio
    async def test_aggregator_commit_mismatch_propagates(self):
        # Aggregator sends a commit that doesn't match the plaintext.
        orch = _build(
            plaintext=b"actual",
            commit_digest=hashlib.sha256(b"different").digest(),
        )
        with pytest.raises(AggregationCommitMismatchError):
            await orch.dispatch_query(
                query="count records",
                prompter_node_id="x",
                query_id=b"q" * 32,
            )


# ──────────────────────────────────────────────────────────────────────
# Wiring contract pin
# ──────────────────────────────────────────────────────────────────────


class TestWiringContractPin:
    """Pin the constructor arg shape so the future node.py wiring task
    knows exactly what it must provide. If a future change renames or
    reshapes any of these, this test blows."""

    def test_constructor_args(self):
        orch = _build()
        # All 5 wiring slots present.
        assert orch.semantic_index is not None
        assert orch.dispatcher is not None
        assert orch.aggregator_client is not None
        assert callable(orch.candidate_pool_provider)
        assert callable(orch.beacon_provider)

    @pytest.mark.asyncio
    async def test_dispatch_query_signature(self):
        # Pin the dispatch_query call shape — kwargs only, three
        # required: query (str), prompter_node_id (str), query_id (32 bytes).
        orch = _build()
        # Missing query_id should error.
        with pytest.raises(TypeError):
            await orch.dispatch_query(query="x", prompter_node_id="p")  # type: ignore

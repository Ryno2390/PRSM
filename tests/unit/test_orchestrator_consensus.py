"""Unit tests for Phase 7.1 Task 5 — MarketplaceOrchestrator consensus routing.

These verify the orchestrator's routing logic between the single-provider
(Phase 7) and k-of-n consensus (Phase 7.1) paths, top-k selection rules,
and minority-receipt queuing behavior. MultiShardDispatcher is mocked —
its own unit tests live in test_multi_dispatcher.py.
"""
from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from prsm.compute.model_sharding.models import ModelShard
from prsm.compute.multi_dispatcher import (
    ConsensusFailedError,
    ConsensusShardReceipt,
    InsufficientResponsesError,
)
from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.marketplace.errors import NoEligibleProvidersError
from prsm.marketplace.filter import EligibilityFilter
from prsm.marketplace.listing import sign_listing
from prsm.marketplace.orchestrator import MarketplaceOrchestrator
from prsm.marketplace.policy import DispatchPolicy
from prsm.marketplace.reputation import ReputationTracker
from prsm.node.identity import generate_node_identity


# ── Helpers ──────────────────────────────────────────────────────────


def _make_shard(index: int = 0) -> ModelShard:
    tensor = np.array([[1.0, 2.0]], dtype=np.float64)
    tb = tensor.tobytes()
    return ModelShard(
        shard_id=f"shard-{index}",
        model_id="m",
        shard_index=index,
        total_shards=1,
        tensor_data=tb,
        tensor_shape=tensor.shape,
        size_bytes=len(tb),
        checksum=hashlib.sha256(tb).hexdigest(),
    )


def _make_listing(**overrides):
    identity = overrides.pop("identity", None) or generate_node_identity(
        display_name="p",
    )
    kwargs = dict(
        capacity_shards_per_sec=10.0,
        max_shard_bytes=10 * 1024 * 1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.05,
        tee_capable=False,
        stake_tier="premium",
        ttl_seconds=300,
    )
    kwargs.update(overrides)
    return sign_listing(identity=identity, **kwargs)


def _make_consensus_receipt(
    job_id: str,
    shard_index: int,
    majority_ids: list[str],
    minority_ids: list[str],
    *,
    agreed_hash: str = "HASH_OK",
    agreed_output: np.ndarray = None,
) -> ConsensusShardReceipt:
    if agreed_output is None:
        agreed_output = np.array([42.0], dtype=np.float64)

    def _receipt(pid: str, h: str) -> ShardExecutionReceipt:
        return ShardExecutionReceipt(
            job_id=job_id,
            shard_index=shard_index,
            provider_id=pid,
            provider_pubkey_b64="PUBKEY",
            output_hash=h,
            executed_at_unix=1_700_000_000,
            signature="SIG",
        )

    return ConsensusShardReceipt(
        job_id=job_id,
        shard_index=shard_index,
        consensus_mode="majority",
        k=len(majority_ids) + len(minority_ids),
        responded=len(majority_ids) + len(minority_ids),
        agreed_output_hash=agreed_hash,
        agreed_output=agreed_output,
        majority_receipts=[_receipt(pid, agreed_hash) for pid in majority_ids],
        minority_receipts=[_receipt(pid, "HASH_CHEAT") for pid in minority_ids],
        consensus_reached_unix=1_700_000_100,
    )


def _make_consensus_orchestrator(listings):
    """Build an orchestrator wired with a mocked MultiShardDispatcher."""
    identity = generate_node_identity(display_name="requester")
    directory = MagicMock()
    directory.list_active_providers = MagicMock(return_value=listings)
    directory.get_listing = MagicMock(
        side_effect=lambda pid, **kw: next(
            (l for l in listings if l.provider_id == pid), None
        )
    )
    reputation = ReputationTracker()
    eligibility = EligibilityFilter(reputation_tracker=reputation)

    price_negotiator = MagicMock()
    price_negotiator.request_quote = AsyncMock()

    remote_dispatcher = MagicMock()
    remote_dispatcher.dispatch = AsyncMock()

    multi_dispatcher = MagicMock()
    multi_dispatcher.dispatch_with_consensus = AsyncMock()

    orchestrator = MarketplaceOrchestrator(
        identity=identity,
        directory=directory,
        eligibility_filter=eligibility,
        reputation=reputation,
        price_negotiator=price_negotiator,
        remote_dispatcher=remote_dispatcher,
        multi_dispatcher=multi_dispatcher,
    )
    return orchestrator, multi_dispatcher, reputation


def _run(coro):
    return asyncio.run(coro)


# ── Routing guard ────────────────────────────────────────────────────


def test_consensus_mode_none_preserves_phase7_single_path():
    """Default policy → single-provider dispatch; MultiDispatcher never
    touched. Phase 7 behavior byte-for-byte preserved."""
    listing = _make_listing()
    orch, multi, _ = _make_consensus_orchestrator([listing])

    # Wire the single dispatcher so the existing Phase 7 path completes.
    from prsm.marketplace.price_handshake import PriceQuote
    orch.price_negotiator.request_quote = AsyncMock(return_value=PriceQuote(
        request_id="q", listing_id=listing.listing_id, shard_index=0,
        quoted_price_ftns=0.05, quote_expires_unix=9999999999,
        provider_id=listing.provider_id,
        provider_pubkey_b64=listing.provider_pubkey_b64,
        signature="sig",
    ))
    orch.remote_dispatcher.dispatch = AsyncMock(
        return_value=np.array([1.0], dtype=np.float64),
    )

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-phase7.1", policy=DispatchPolicy(),  # consensus_mode=None
    ))

    multi.dispatch_with_consensus.assert_not_called()
    orch.remote_dispatcher.dispatch.assert_awaited_once()


def test_consensus_mode_single_also_preserves_single_path():
    """Explicit 'single' mode — also skips MultiDispatcher."""
    listing = _make_listing()
    orch, multi, _ = _make_consensus_orchestrator([listing])

    from prsm.marketplace.price_handshake import PriceQuote
    orch.price_negotiator.request_quote = AsyncMock(return_value=PriceQuote(
        request_id="q", listing_id=listing.listing_id, shard_index=0,
        quoted_price_ftns=0.05, quote_expires_unix=9999999999,
        provider_id=listing.provider_id,
        provider_pubkey_b64=listing.provider_pubkey_b64,
        signature="sig",
    ))
    orch.remote_dispatcher.dispatch = AsyncMock(
        return_value=np.array([1.0], dtype=np.float64),
    )

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="j", policy=DispatchPolicy(consensus_mode="single"),
    ))
    multi.dispatch_with_consensus.assert_not_called()


# ── Consensus happy path ─────────────────────────────────────────────


def test_consensus_majority_all_agree_returns_output():
    """3 providers, all agree → returns agreed_output from receipt,
    all three providers get success reputation, no minority queued."""
    listings = [_make_listing() for _ in range(3)]
    orch, multi, reputation = _make_consensus_orchestrator(listings)

    expected = np.array([42.0, 42.0], dtype=np.float64)
    multi.dispatch_with_consensus.return_value = _make_consensus_receipt(
        job_id="job-ok", shard_index=0,
        majority_ids=[l.provider_id for l in listings],
        minority_ids=[],
        agreed_output=expected,
    )

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-ok",
        policy=DispatchPolicy(
            consensus_mode="majority", consensus_k=3,
        ),
    ))

    np.testing.assert_array_equal(result, expected)
    multi.dispatch_with_consensus.assert_awaited_once()
    # Every majority provider recorded a success.
    for l in listings:
        rep = reputation.get_reputation(l.provider_id)
        assert rep is not None
        assert len(rep.successful_dispatches) == 1
        assert len(rep.failed_dispatches) == 0
    # No challenges queued.
    assert orch.consensus_minority_queue == []


def test_consensus_minority_detected_queues_challenges():
    """2-of-3 agree, 1 minority → minority gets failure rep + entry in
    consensus_minority_queue for the downstream challenge submitter."""
    listings = [_make_listing() for _ in range(3)]
    orch, multi, reputation = _make_consensus_orchestrator(listings)

    majority = [l.provider_id for l in listings[:2]]
    minority = [listings[2].provider_id]
    multi.dispatch_with_consensus.return_value = _make_consensus_receipt(
        job_id="job-cheat", shard_index=0,
        majority_ids=majority, minority_ids=minority,
    )

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-cheat",
        policy=DispatchPolicy(
            consensus_mode="majority", consensus_k=3,
        ),
    ))

    # Majority got success; minority got failure.
    for pid in majority:
        rep = reputation.get_reputation(pid)
        assert len(rep.successful_dispatches) == 1
    minority_rep = reputation.get_reputation(minority[0])
    assert len(minority_rep.failed_dispatches) == 1
    assert len(minority_rep.successful_dispatches) == 0

    # Challenge queued with full data the submitter needs.
    assert len(orch.consensus_minority_queue) == 1
    q = orch.consensus_minority_queue[0]
    assert q["job_id"] == "job-cheat"
    assert q["shard_index"] == 0
    assert len(q["minority_receipts"]) == 1
    assert q["minority_receipts"][0].provider_id == minority[0]
    assert len(q["majority_receipts"]) == 2


# ── Insufficient pool ────────────────────────────────────────────────


def test_consensus_insufficient_eligible_pool_raises():
    """k=3 but only 2 listings post-filter → NoEligibleProvidersError
    (don't silently degrade to k=2; the requester asked for k=3)."""
    listings = [_make_listing() for _ in range(2)]
    orch, multi, _ = _make_consensus_orchestrator(listings)

    with pytest.raises(NoEligibleProvidersError, match="k=3"):
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job", policy=DispatchPolicy(
                consensus_mode="majority", consensus_k=3,
            ),
        ))
    multi.dispatch_with_consensus.assert_not_called()


# ── Multi-dispatch error surface ─────────────────────────────────────


def test_insufficient_responses_error_maps_to_no_eligible():
    """MultiDispatcher raises InsufficientResponsesError → orchestrator
    fails the shard with a reputational hit on every selected provider."""
    listings = [_make_listing() for _ in range(3)]
    orch, multi, reputation = _make_consensus_orchestrator(listings)
    multi.dispatch_with_consensus.side_effect = InsufficientResponsesError(
        responded=1, k=3, threshold=2,
    )

    with pytest.raises(NoEligibleProvidersError, match="InsufficientResponses"):
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job", policy=DispatchPolicy(
                consensus_mode="majority", consensus_k=3,
            ),
        ))
    # All three selected providers get a failure on the blanket mapping.
    for l in listings:
        rep = reputation.get_reputation(l.provider_id)
        assert len(rep.failed_dispatches) == 1


def test_consensus_failed_error_maps_to_no_eligible():
    """All 3 responded with 3 different hashes → ConsensusFailedError.
    Same blanket reputation hit as InsufficientResponses for MVP."""
    listings = [_make_listing() for _ in range(3)]
    orch, multi, reputation = _make_consensus_orchestrator(listings)
    multi.dispatch_with_consensus.side_effect = ConsensusFailedError(
        responded=3, k=3, unique_hashes=3,
    )

    with pytest.raises(NoEligibleProvidersError, match="ConsensusFailed"):
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job", policy=DispatchPolicy(
                consensus_mode="majority", consensus_k=3,
            ),
        ))


# ── Top-k selection ──────────────────────────────────────────────────


def test_select_top_k_ranks_by_tier_and_price():
    """Higher tier + lower price → higher score. Given identical prices,
    tier breaks the tie."""
    # All else equal, premium ranks above standard ranks above open.
    listings = [
        _make_listing(stake_tier="open"),
        _make_listing(stake_tier="premium"),
        _make_listing(stake_tier="standard"),
        _make_listing(stake_tier="critical"),
    ]
    picked = MarketplaceOrchestrator._select_top_k(listings, k=2)
    tiers = [l.stake_tier for l in picked]
    assert tiers == ["critical", "premium"]


def test_select_top_k_rewards_lower_price_at_same_tier():
    """Same tier, different prices → cheaper wins."""
    listings = [
        _make_listing(stake_tier="premium", price_per_shard_ftns=0.10),
        _make_listing(stake_tier="premium", price_per_shard_ftns=0.05),
        _make_listing(stake_tier="premium", price_per_shard_ftns=0.08),
    ]
    picked = MarketplaceOrchestrator._select_top_k(listings, k=1)
    assert picked[0].price_per_shard_ftns == 0.05


def test_select_top_k_lexicographic_tiebreak():
    """Identical tier + identical price → break by provider_id asc.
    Deterministic: the requester can replay selection for audit."""
    # Use fixed identities so provider_ids are stable.
    ids = [generate_node_identity(display_name=f"p{i}") for i in range(3)]
    listings = [
        _make_listing(identity=ids[0], stake_tier="premium",
                      price_per_shard_ftns=0.05),
        _make_listing(identity=ids[1], stake_tier="premium",
                      price_per_shard_ftns=0.05),
        _make_listing(identity=ids[2], stake_tier="premium",
                      price_per_shard_ftns=0.05),
    ]
    sorted_ids = sorted(l.provider_id for l in listings)
    picked = MarketplaceOrchestrator._select_top_k(listings, k=2)
    assert [l.provider_id for l in picked] == sorted_ids[:2]


def test_select_top_k_passes_exact_ids_to_multi_dispatcher():
    """MultiDispatcher must receive node_ids in the exact top-k order
    the selector returned — no re-selection inside."""
    listings = [
        _make_listing(stake_tier="critical", price_per_shard_ftns=0.01),
        _make_listing(stake_tier="standard", price_per_shard_ftns=0.01),
        _make_listing(stake_tier="premium", price_per_shard_ftns=0.01),
    ]
    orch, multi, _ = _make_consensus_orchestrator(listings)
    multi.dispatch_with_consensus.return_value = _make_consensus_receipt(
        job_id="j", shard_index=0,
        majority_ids=[l.provider_id for l in sorted(
            listings,
            key=lambda l: {"open": 1, "standard": 2, "premium": 3, "critical": 4}[l.stake_tier],
            reverse=True,
        )[:3]],
        minority_ids=[],
    )

    _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="j", policy=DispatchPolicy(
            consensus_mode="majority", consensus_k=3,
        ),
    ))

    call_kwargs = multi.dispatch_with_consensus.await_args.kwargs
    node_ids = call_kwargs["node_ids"]
    assert len(node_ids) == 3
    # Top ranked first — critical, then premium, then standard (when
    # prices are equal, higher tier wins).
    actual_tiers = [
        next(l.stake_tier for l in listings if l.provider_id == nid)
        for nid in node_ids
    ]
    assert actual_tiers == ["critical", "premium", "standard"]


# ── Error surface ────────────────────────────────────────────────────


def test_consensus_requested_without_multi_dispatcher_raises():
    """Policy requests consensus but no multi_dispatcher wired → surface
    the misconfiguration rather than silently falling back to single."""
    listings = [_make_listing() for _ in range(3)]
    identity = generate_node_identity(display_name="requester")
    directory = MagicMock()
    directory.list_active_providers = MagicMock(return_value=listings)
    directory.get_listing = MagicMock(
        side_effect=lambda pid, **kw: next(
            (l for l in listings if l.provider_id == pid), None
        )
    )
    reputation = ReputationTracker()
    eligibility = EligibilityFilter(reputation_tracker=reputation)

    price_negotiator = MagicMock()
    price_negotiator.request_quote = AsyncMock()
    remote_dispatcher = MagicMock()
    remote_dispatcher.dispatch = AsyncMock()

    orch = MarketplaceOrchestrator(
        identity=identity,
        directory=directory,
        eligibility_filter=eligibility,
        reputation=reputation,
        price_negotiator=price_negotiator,
        remote_dispatcher=remote_dispatcher,
        # multi_dispatcher intentionally omitted
    )

    with pytest.raises(RuntimeError, match="multi_dispatcher"):
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="j", policy=DispatchPolicy(
                consensus_mode="majority", consensus_k=3,
            ),
        ))

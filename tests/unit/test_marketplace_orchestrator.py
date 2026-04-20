"""Unit tests for MarketplaceOrchestrator.

Phase 3 Task 7. Unit-scoped — uses mocks for directory, dispatcher,
price negotiator, reputation. The full wiring is covered by the
integration test in tests/integration/test_phase3_marketplace_e2e.py.
"""
from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from prsm.compute.model_sharding.models import ModelShard, PipelineStakeTier
from prsm.compute.remote_dispatcher import (
    MissingAttestationError,
    ShardDispatchError,
    ShardPreemptedError,
)
from prsm.marketplace.errors import NoEligibleProvidersError
from prsm.marketplace.filter import EligibilityFilter
from prsm.marketplace.listing import sign_listing
from prsm.marketplace.orchestrator import MarketplaceOrchestrator
from prsm.marketplace.policy import DispatchPolicy
from prsm.marketplace.price_handshake import PriceQuote, PriceQuoteRejected
from prsm.marketplace.reputation import ReputationTracker
from prsm.node.identity import generate_node_identity


def _make_shard(index: int = 0) -> ModelShard:
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tensor_bytes = tensor.tobytes()
    return ModelShard(
        shard_id=f"shard-{index}",
        model_id="m-test",
        shard_index=index,
        total_shards=1,
        tensor_data=tensor_bytes,
        tensor_shape=tensor.shape,
        size_bytes=len(tensor_bytes),
        checksum=hashlib.sha256(tensor_bytes).hexdigest(),
    )


def _make_listing(**overrides):
    identity = overrides.pop("identity", None) or generate_node_identity(
        display_name="p"
    )
    kwargs = dict(
        capacity_shards_per_sec=10.0,
        max_shard_bytes=10 * 1024 * 1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.05,
        tee_capable=False,
        stake_tier="standard",
        ttl_seconds=300,
    )
    kwargs.update(overrides)
    return sign_listing(identity=identity, **kwargs)


def _make_orchestrator(listings, *, verify_tee=False):
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

    orchestrator = MarketplaceOrchestrator(
        identity=identity,
        directory=directory,
        eligibility_filter=eligibility,
        reputation=reputation,
        price_negotiator=price_negotiator,
        remote_dispatcher=remote_dispatcher,
    )
    return (
        orchestrator, price_negotiator, remote_dispatcher, reputation,
    )


def _make_quote(listing, price: float = 0.05) -> PriceQuote:
    return PriceQuote(
        request_id="q-1",
        listing_id=listing.listing_id,
        shard_index=0,
        quoted_price_ftns=price,
        quote_expires_unix=9999999999,
        provider_id=listing.provider_id,
        provider_pubkey_b64=listing.provider_pubkey_b64,
        signature="sig",
    )


def _run(coro):
    return asyncio.run(coro)


def test_orchestrator_happy_path_single_shard():
    listing = _make_listing()
    orch, quoter, dispatcher, reputation = _make_orchestrator([listing])

    expected_output = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.return_value = expected_output

    shard = _make_shard(0)
    input_tensor = np.array([1.0, 1.0], dtype=np.float64)
    policy = DispatchPolicy()

    result = _run(orch.orchestrate_sharded_inference(
        shards=[shard], input_tensor=input_tensor,
        job_id="job-1", policy=policy,
    ))

    np.testing.assert_array_equal(result, expected_output)
    quoter.request_quote.assert_awaited_once()
    dispatcher.dispatch.assert_awaited_once()
    # Reputation recorded success with some latency.
    rep = reputation.get_reputation(listing.provider_id)
    assert rep is not None
    assert len(rep.successful_dispatches) == 1
    assert len(rep.latency_samples_ms) == 1


def test_orchestrator_no_eligible_providers_raises():
    """When filter excludes every listing, raise NoEligibleProvidersError
    without any network traffic."""
    # Listing too expensive for the policy.
    listing = _make_listing(price_per_shard_ftns=10.0)
    orch, quoter, dispatcher, _ = _make_orchestrator([listing])

    shard = _make_shard(0)
    input_tensor = np.array([1.0, 1.0], dtype=np.float64)
    policy = DispatchPolicy(max_price_per_shard_ftns=0.10)

    with pytest.raises(NoEligibleProvidersError):
        _run(orch.orchestrate_sharded_inference(
            shards=[shard], input_tensor=input_tensor,
            job_id="job-1", policy=policy,
        ))

    quoter.request_quote.assert_not_called()
    dispatcher.dispatch.assert_not_called()


def test_orchestrator_empty_directory_raises():
    orch, _, _, _ = _make_orchestrator([])

    shard = _make_shard(0)
    input_tensor = np.array([1.0, 1.0], dtype=np.float64)

    with pytest.raises(NoEligibleProvidersError):
        _run(orch.orchestrate_sharded_inference(
            shards=[shard], input_tensor=input_tensor,
            job_id="job-1", policy=DispatchPolicy(),
        ))


def test_orchestrator_quote_rejection_tries_next_provider():
    """First provider rejects the quote; orchestrator moves to the second."""
    listing_bad = _make_listing()
    listing_good = _make_listing()
    orch, quoter, dispatcher, reputation = _make_orchestrator(
        [listing_bad, listing_good]
    )

    expected = np.array([42.0], dtype=np.float64)

    # First call rejects, second accepts.
    quoter.request_quote.side_effect = [
        PriceQuoteRejected(
            request_id="q-1", listing_id=listing_bad.listing_id,
            reason="overloaded",
        ),
        _make_quote(listing_good),
    ]
    dispatcher.dispatch.return_value = expected

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    np.testing.assert_array_equal(result, expected)
    assert quoter.request_quote.await_count == 2
    assert dispatcher.dispatch.await_count == 1
    # Good provider got the success.
    _, kwargs = dispatcher.dispatch.await_args_list[0].args, dispatcher.dispatch.await_args_list[0].kwargs
    assert kwargs["node_id"] == listing_good.provider_id


def test_orchestrator_preemption_reroutes_to_fresh_pool():
    """Preempted provider → record_preemption (no reputation penalty),
    try next. Honest-work failure stays off the score denominator."""
    listing_preempted = _make_listing()
    listing_ok = _make_listing()
    orch, quoter, dispatcher, reputation = _make_orchestrator(
        [listing_preempted, listing_ok]
    )

    quoter.request_quote.side_effect = [
        _make_quote(listing_preempted),
        _make_quote(listing_ok),
    ]
    dispatcher.dispatch.side_effect = [
        ShardPreemptedError(
            shard_index=0, node_id=listing_preempted.provider_id,
            reason="spot_eviction",
        ),
        np.array([1.0], dtype=np.float64),
    ]

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    np.testing.assert_array_equal(result, np.array([1.0]))
    # Preempted provider: preemption counted, NOT a failure.
    rep_preempted = reputation.get_reputation(listing_preempted.provider_id)
    assert len(rep_preempted.preempted_dispatches) == 1
    assert len(rep_preempted.failed_dispatches) == 0
    # OK provider: success recorded.
    rep_ok = reputation.get_reputation(listing_ok.provider_id)
    assert len(rep_ok.successful_dispatches) == 1


def test_orchestrator_missing_attestation_penalizes_and_retries():
    """When policy.require_tee=True and a provider's receipt omits
    tee_attestation, the dispatcher raises MissingAttestationError.
    Orchestrator records failure (liar penalty) and moves on."""
    identity_bad = generate_node_identity(display_name="liar")
    identity_ok = generate_node_identity(display_name="honest")
    listing_liar = _make_listing(identity=identity_bad, tee_capable=True)
    listing_honest = _make_listing(identity=identity_ok, tee_capable=True)
    orch, quoter, dispatcher, reputation = _make_orchestrator(
        [listing_liar, listing_honest]
    )

    quoter.request_quote.side_effect = [
        _make_quote(listing_liar),
        _make_quote(listing_honest),
    ]
    dispatcher.dispatch.side_effect = [
        MissingAttestationError("no tee quote"),
        np.array([1.0], dtype=np.float64),
    ]

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1",
        policy=DispatchPolicy(require_tee=True),
    ))

    np.testing.assert_array_equal(result, np.array([1.0]))
    # Liar: failure recorded.
    rep_liar = reputation.get_reputation(listing_liar.provider_id)
    assert len(rep_liar.failed_dispatches) == 1
    assert len(rep_liar.successful_dispatches) == 0


def test_orchestrator_shard_dispatch_error_propagates():
    """Malicious/unrecoverable ShardDispatchError from the dispatcher
    escalates immediately — we don't silently retry on a different
    provider because the error isn't provider-specific (could be
    verification failure on a signed receipt, which is a protocol-
    level problem)."""
    listing = _make_listing()
    orch, quoter, dispatcher, reputation = _make_orchestrator([listing])

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.side_effect = ShardDispatchError("bad receipt")

    with pytest.raises(ShardDispatchError):
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job-1", policy=DispatchPolicy(),
        ))

    rep = reputation.get_reputation(listing.provider_id)
    assert len(rep.failed_dispatches) == 1


def test_orchestrator_exhausted_pool_raises_no_eligible():
    """All providers in the eligible pool rejected or preempted →
    NoEligibleProvidersError with a summary of the last failure."""
    listing_a = _make_listing()
    listing_b = _make_listing()
    orch, quoter, dispatcher, _ = _make_orchestrator([listing_a, listing_b])

    quoter.request_quote.return_value = PriceQuoteRejected(
        request_id="q", listing_id="x", reason="overloaded",
    )

    with pytest.raises(NoEligibleProvidersError) as excinfo:
        _run(orch.orchestrate_sharded_inference(
            shards=[_make_shard(0)],
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            job_id="job-1", policy=DispatchPolicy(),
        ))

    assert "exhausted" in str(excinfo.value).lower()
    dispatcher.dispatch.assert_not_called()


def test_orchestrator_multiple_shards_concatenates_in_order():
    """Two shards → two dispatches → concatenate axis=0 in shard index
    order."""
    listing = _make_listing()
    orch, quoter, dispatcher, _ = _make_orchestrator([listing])

    quoter.request_quote.return_value = _make_quote(listing)
    dispatcher.dispatch.side_effect = [
        np.array([1.0, 2.0], dtype=np.float64),  # shard 0
        np.array([3.0, 4.0], dtype=np.float64),  # shard 1
    ]

    result = _run(orch.orchestrate_sharded_inference(
        shards=[_make_shard(0), _make_shard(1)],
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        job_id="job-1", policy=DispatchPolicy(),
    ))

    np.testing.assert_array_equal(
        result, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )

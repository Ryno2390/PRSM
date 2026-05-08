"""MarketplaceCandidatePoolProvider — bridges existing primitives
(MarketplaceDirectory + ReputationTracker) to the QueryOrchestrator's
`candidate_pool_provider` callable contract.

Per the QueryOrchestrator wiring-readiness assessment, the
QueryOrchestrator class needs a `Callable[[], tuple[StakedNode, ...]]`
that returns the current T2+ stake pool snapshot. This module wires
that callable from the existing marketplace gossip directory.

v1 limitations (documented inline in the impl):
  - stake_amount_ftns is derived from the tier label, not from a
    per-listing on-chain stake_of read (avoiding N RPC calls per
    selection)
  - has_tee defaults to False — listings don't carry that marker yet;
    Tier C queries will find no candidates until ProviderListing is
    extended (separate follow-on)
"""
from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass

import pytest

from prsm.compute.query_orchestrator import StakedNode
from prsm.compute.query_orchestrator.marketplace_candidate_pool_provider import (
    DEFAULT_STAKE_PER_TIER,
    MarketplaceCandidatePoolProvider,
)
from prsm.marketplace.listing import ProviderListing
from prsm.marketplace.reputation import ReputationTracker


# ──────────────────────────────────────────────────────────────────────
# Stubs
# ──────────────────────────────────────────────────────────────────────


def _make_listing(
    *,
    provider_id: str,
    pubkey_b64: str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    stake_tier: str = "T2",
    tee_capable: bool = False,
) -> ProviderListing:
    """Build a ProviderListing for tests."""
    return ProviderListing(
        listing_id="listing-x",
        provider_id=provider_id,
        provider_pubkey_b64=pubkey_b64,
        capacity_shards_per_sec=1.0,
        max_shard_bytes=1024 * 1024,
        supported_dtypes=["int64"],
        price_per_shard_ftns=0.001,
        tee_capable=tee_capable,
        stake_tier=stake_tier,
        advertised_at_unix=int(time.time()),
        ttl_seconds=300,
        signature="AAAA",
    )


@dataclass
class _StubDirectory:
    canned_listings: list

    def list_active_providers(self, at_unix=None):
        return list(self.canned_listings)


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_returns_tuple_of_staked_nodes(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="provider-1", stake_tier="T2"),
            _make_listing(provider_id="provider-2", stake_tier="T3"),
        ])
        reputation = ReputationTracker()
        provider = MarketplaceCandidatePoolProvider(
            directory=directory,
            reputation=reputation,
        )
        pool = provider()
        assert isinstance(pool, tuple)
        assert len(pool) == 2
        for node in pool:
            assert isinstance(node, StakedNode)

    def test_t1_filtered_out(self):
        # T1 is below the aggregator-eligibility threshold (Vision §6).
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="t1-only", stake_tier="T1"),
            _make_listing(provider_id="t2-eligible", stake_tier="T2"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory,
            reputation=ReputationTracker(),
        )
        pool = provider()
        ids = {n.node_id for n in pool}
        assert "t2-eligible" in ids
        assert "t1-only" not in ids

    def test_unknown_tier_filtered_out(self):
        # Defensive: a malformed listing shouldn't sneak past the gate.
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="bad", stake_tier="T99"),
            _make_listing(provider_id="good", stake_tier="T4"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory,
            reputation=ReputationTracker(),
        )
        pool = provider()
        assert {n.node_id for n in pool} == {"good"}


# ──────────────────────────────────────────────────────────────────────
# Field threading
# ──────────────────────────────────────────────────────────────────────


class TestFieldThreading:
    def test_stake_amount_derived_from_tier(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id=f"p-{tier}", stake_tier=tier)
            for tier in ["T2", "T3", "T4"]
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=ReputationTracker(),
        )
        pool = provider()
        per_id = {n.node_id: n for n in pool}
        # T4 should outweigh T3 should outweigh T2 — derived from tier.
        assert (
            per_id["p-T2"].stake_amount_ftns
            < per_id["p-T3"].stake_amount_ftns
            < per_id["p-T4"].stake_amount_ftns
        )

    def test_pubkey_hash_is_sha256_of_decoded_pubkey(self):
        # 32 zero bytes encoded as base64 — easy to verify.
        zero_pub_b64 = base64.b64encode(b"\x00" * 32).decode("ascii")
        directory = _StubDirectory(canned_listings=[
            _make_listing(
                provider_id="p-1", pubkey_b64=zero_pub_b64, stake_tier="T2",
            ),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=ReputationTracker(),
        )
        [node] = provider()
        assert node.pubkey_hash == hashlib.sha256(b"\x00" * 32).digest()
        assert len(node.pubkey_hash) == 32

    def test_reputation_score_threaded(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="rep-test", stake_tier="T2"),
        ])
        reputation = ReputationTracker()
        # Push a slash so score drops.
        reputation.record_slash(
            provider_id="rep-test",
            batch_id="batch-1",
            slash_amount_wei=1,
            reason="DOUBLE_SPEND",
        )
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=reputation,
        )
        [node] = provider()
        # Slashed → score below NEUTRAL_SCORE.
        assert node.reputation_score < ReputationTracker.NEUTRAL_SCORE

    def test_unknown_provider_gets_neutral_reputation(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="never-tracked", stake_tier="T2"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=ReputationTracker(),
        )
        [node] = provider()
        assert node.reputation_score == ReputationTracker.NEUTRAL_SCORE


# ──────────────────────────────────────────────────────────────────────
# Stake-per-tier override
# ──────────────────────────────────────────────────────────────────────


class TestStakePerTierOverride:
    def test_default_table_has_t2_t3_t4(self):
        # Pin the default table — Foundation council can ratify
        # different values via constructor override.
        assert "T2" in DEFAULT_STAKE_PER_TIER
        assert "T3" in DEFAULT_STAKE_PER_TIER
        assert "T4" in DEFAULT_STAKE_PER_TIER
        assert (
            DEFAULT_STAKE_PER_TIER["T2"]
            < DEFAULT_STAKE_PER_TIER["T3"]
            < DEFAULT_STAKE_PER_TIER["T4"]
        )

    def test_override_threads_through(self):
        directory = _StubDirectory(canned_listings=[
            _make_listing(provider_id="p", stake_tier="T2"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory,
            reputation=ReputationTracker(),
            stake_amount_per_tier={"T2": 999, "T3": 1000, "T4": 1001},
        )
        [node] = provider()
        assert node.stake_amount_ftns == 999


# ──────────────────────────────────────────────────────────────────────
# Empty pool / degenerate
# ──────────────────────────────────────────────────────────────────────


class TestDegenerate:
    def test_empty_directory_returns_empty_tuple(self):
        provider = MarketplaceCandidatePoolProvider(
            directory=_StubDirectory(canned_listings=[]),
            reputation=ReputationTracker(),
        )
        pool = provider()
        assert pool == ()

    def test_malformed_pubkey_skipped_silently(self):
        # Invalid base64 should not blow the whole call — just drop
        # the offending listing.
        directory = _StubDirectory(canned_listings=[
            _make_listing(
                provider_id="malformed",
                pubkey_b64="!!!not-base64!!!",
                stake_tier="T2",
            ),
            _make_listing(provider_id="good", stake_tier="T2"),
        ])
        provider = MarketplaceCandidatePoolProvider(
            directory=directory, reputation=ReputationTracker(),
        )
        pool = provider()
        assert {n.node_id for n in pool} == {"good"}


# ──────────────────────────────────────────────────────────────────────
# Callable contract pin
# ──────────────────────────────────────────────────────────────────────


class TestCallableContract:
    """The QueryOrchestrator class accepts
    `candidate_pool_provider: Callable[[], tuple[StakedNode, ...]]`.
    Pin that the adapter satisfies it."""

    def test_provider_is_callable(self):
        provider = MarketplaceCandidatePoolProvider(
            directory=_StubDirectory(canned_listings=[]),
            reputation=ReputationTracker(),
        )
        assert callable(provider)

    def test_provider_called_with_no_args_returns_tuple(self):
        provider = MarketplaceCandidatePoolProvider(
            directory=_StubDirectory(canned_listings=[]),
            reputation=ReputationTracker(),
        )
        result = provider()
        assert isinstance(result, tuple)

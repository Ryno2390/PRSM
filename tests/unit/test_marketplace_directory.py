"""Unit tests for MarketplaceDirectory.

Phase 3 Task 2. Verifies:
  - Valid listing is ingested and queryable.
  - Malformed dict is dropped silently.
  - Bad-signature listing is dropped silently.
  - Mismatched provider_id (attacker attempting identity spoof) is dropped.
  - Newer listing for same provider_id replaces older.
  - Older listing for same provider_id is ignored (anti-replay).
  - Expired listings are evicted on query.
  - get_listing returns None for expired or unknown providers.
  - size() reflects only active listings.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from prsm.marketplace.directory import MarketplaceDirectory
from prsm.marketplace.listing import ProviderListing, sign_listing
from prsm.node.gossip import GOSSIP_MARKETPLACE_LISTING
from prsm.node.identity import generate_node_identity


def _make_gossip():
    """Gossip stub that just records the handler and lets us fire events."""
    gossip = MagicMock()
    gossip._handler = None

    def subscribe(topic, handler):
        assert topic == GOSSIP_MARKETPLACE_LISTING
        gossip._handler = handler

    gossip.subscribe = subscribe
    return gossip


async def _fire(gossip, data, origin: str = "test-origin"):
    await gossip._handler(GOSSIP_MARKETPLACE_LISTING, data, origin)


def _make_listing(identity, **overrides):
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


def _run(coro):
    return asyncio.run(coro)


def test_directory_ingests_valid_listing():
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)
    identity = generate_node_identity(display_name="p1")

    listing = _make_listing(identity)
    _run(_fire(gossip, listing.to_dict()))

    assert directory.size() == 1
    got = directory.get_listing(identity.node_id)
    assert got == listing


def test_directory_drops_malformed_data():
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)

    # Missing required field.
    _run(_fire(gossip, {"listing_id": "foo", "provider_id": "bar"}))

    assert directory.size() == 0


def test_directory_drops_bad_signature():
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)
    identity = generate_node_identity(display_name="p1")
    listing = _make_listing(identity)

    # Tamper the signature post-signing.
    tampered = ProviderListing(
        **{**listing.to_dict(), "signature": "A" * 80}
    )
    _run(_fire(gossip, tampered.to_dict()))

    assert directory.size() == 0


def test_directory_drops_mismatched_provider_id():
    """Attacker claims a victim's node_id while carrying own pubkey."""
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)
    attacker = generate_node_identity(display_name="attacker")
    victim = generate_node_identity(display_name="victim")

    listing = _make_listing(attacker)
    forged = ProviderListing(
        **{**listing.to_dict(), "provider_id": victim.node_id}
    )
    _run(_fire(gossip, forged.to_dict()))

    assert directory.size() == 0


def test_directory_replaces_older_listing_for_same_provider():
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)
    identity = generate_node_identity(display_name="p1")

    now = int(time.time())
    old = _make_listing(identity, price_per_shard_ftns=0.10)
    # Re-sign from scratch to set an earlier timestamp (sign_listing
    # uses int(time.time()) by default unless we override).
    old = sign_listing(
        identity=identity, capacity_shards_per_sec=10.0,
        max_shard_bytes=1024, supported_dtypes=["float64"],
        price_per_shard_ftns=0.10, tee_capable=False,
        stake_tier="standard", ttl_seconds=300,
        advertised_at_unix=now - 10,
    )
    new = sign_listing(
        identity=identity, capacity_shards_per_sec=10.0,
        max_shard_bytes=1024, supported_dtypes=["float64"],
        price_per_shard_ftns=0.05, tee_capable=False,
        stake_tier="standard", ttl_seconds=300,
        advertised_at_unix=now,
    )

    _run(_fire(gossip, old.to_dict()))
    _run(_fire(gossip, new.to_dict()))

    got = directory.get_listing(identity.node_id)
    assert got is not None
    assert got.price_per_shard_ftns == 0.05
    assert got.advertised_at_unix == now


def test_directory_ignores_older_listing_for_same_provider():
    """Anti-replay: if an attacker tries to re-broadcast an old listing
    with attacker-favorable terms (e.g., low-price loss-leader), the
    directory MUST NOT downgrade from the current newer entry."""
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)
    identity = generate_node_identity(display_name="p1")

    now = int(time.time())
    old_cheap = sign_listing(
        identity=identity, capacity_shards_per_sec=10.0,
        max_shard_bytes=1024, supported_dtypes=["float64"],
        price_per_shard_ftns=0.001, tee_capable=False,  # suspiciously cheap
        stake_tier="standard", ttl_seconds=300,
        advertised_at_unix=now - 100,
    )
    new_normal = sign_listing(
        identity=identity, capacity_shards_per_sec=10.0,
        max_shard_bytes=1024, supported_dtypes=["float64"],
        price_per_shard_ftns=0.05, tee_capable=False,
        stake_tier="standard", ttl_seconds=300,
        advertised_at_unix=now,
    )

    # Legitimate broadcast first.
    _run(_fire(gossip, new_normal.to_dict()))
    # Replay attack: old cheap listing arrives second.
    _run(_fire(gossip, old_cheap.to_dict()))

    got = directory.get_listing(identity.node_id)
    assert got is not None
    assert got.price_per_shard_ftns == 0.05
    # Still the newer one.


def test_directory_evicts_expired_on_list():
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)
    identity = generate_node_identity(display_name="p1")

    now = int(time.time())
    listing = sign_listing(
        identity=identity, capacity_shards_per_sec=1.0,
        max_shard_bytes=1024, supported_dtypes=["float64"],
        price_per_shard_ftns=0.01, tee_capable=False,
        stake_tier="open", ttl_seconds=10,
        advertised_at_unix=now,
    )
    _run(_fire(gossip, listing.to_dict()))
    assert directory.size(at_unix=now) == 1

    # Jump forward past ttl.
    assert directory.size(at_unix=now + 100) == 0
    assert directory.list_active_providers(at_unix=now + 100) == []


def test_directory_get_returns_none_for_expired():
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)
    identity = generate_node_identity(display_name="p1")

    now = int(time.time())
    listing = sign_listing(
        identity=identity, capacity_shards_per_sec=1.0,
        max_shard_bytes=1024, supported_dtypes=["float64"],
        price_per_shard_ftns=0.01, tee_capable=False,
        stake_tier="open", ttl_seconds=10,
        advertised_at_unix=now,
    )
    _run(_fire(gossip, listing.to_dict()))

    assert directory.get_listing(identity.node_id, at_unix=now + 100) is None


def test_directory_get_returns_none_for_unknown():
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)

    assert directory.get_listing("unknown-node-id") is None


def test_directory_multiple_providers():
    gossip = _make_gossip()
    directory = MarketplaceDirectory(gossip=gossip)

    ids = [generate_node_identity(display_name=f"p{i}") for i in range(3)]
    for i, ident in enumerate(ids):
        listing = _make_listing(ident, price_per_shard_ftns=0.01 * (i + 1))
        _run(_fire(gossip, listing.to_dict()))

    assert directory.size() == 3
    active = directory.list_active_providers()
    prices = sorted(l.price_per_shard_ftns for l in active)
    assert prices == [0.01, 0.02, 0.03]

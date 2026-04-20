"""Unit tests for ProviderListing + sign/verify.

Phase 3 Task 1. Exercises:
  - sign/verify roundtrip (happy path)
  - tamper detection (field mutation breaks signature)
  - forged provider_id (claim A's identity with B's pubkey)
  - dict-based serialization roundtrip
  - ttl expiry
  - sanity-check rejection (negative ttl, empty dtypes)
"""
from __future__ import annotations

import time

from prsm.marketplace.listing import (
    ProviderListing,
    build_listing_signing_payload,
    sign_listing,
    verify_listing,
)
from prsm.node.identity import generate_node_identity


def _fresh():
    return generate_node_identity(display_name="test-provider")


def _make_valid_listing(identity, **overrides):
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


def test_listing_signing_roundtrip():
    identity = _fresh()
    listing = _make_valid_listing(identity)
    assert verify_listing(listing) is True
    assert listing.provider_id == identity.node_id
    assert listing.provider_pubkey_b64 == identity.public_key_b64


def test_listing_tamper_detection_price():
    """Flipping price after signing breaks the signature."""
    identity = _fresh()
    listing = _make_valid_listing(identity)
    tampered = ProviderListing(
        **{**listing.to_dict(), "price_per_shard_ftns": 0.01}
    )
    assert verify_listing(tampered) is False


def test_listing_tamper_detection_capacity():
    identity = _fresh()
    listing = _make_valid_listing(identity)
    tampered = ProviderListing(
        **{**listing.to_dict(), "capacity_shards_per_sec": 99999.0}
    )
    assert verify_listing(tampered) is False


def test_listing_provider_id_must_match_pubkey():
    """Closes the 'claim provider A's node_id while carrying B's pubkey'
    attack at the listing layer — same guard Phase 2 receipts use."""
    victim = _fresh()
    attacker = _fresh()
    assert victim.node_id != attacker.node_id

    listing = _make_valid_listing(attacker)
    forged = ProviderListing(
        **{**listing.to_dict(), "provider_id": victim.node_id}
    )
    assert verify_listing(forged) is False


def test_listing_roundtrip_serialization():
    identity = _fresh()
    listing = _make_valid_listing(identity, stake_tier="premium", tee_capable=True)
    as_dict = listing.to_dict()
    restored = ProviderListing.from_dict(as_dict)
    assert restored == listing
    assert verify_listing(restored) is True


def test_listing_is_expired():
    identity = _fresh()
    now = int(time.time())
    listing = sign_listing(
        identity=identity,
        capacity_shards_per_sec=1.0,
        max_shard_bytes=1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.01,
        tee_capable=False,
        stake_tier="open",
        ttl_seconds=1,
        advertised_at_unix=now,
    )
    assert listing.is_expired(at_unix=now + 2) is True
    assert listing.is_expired(at_unix=now) is False


def test_listing_rejects_empty_dtypes():
    """A listing with no supported_dtypes can never be selected —
    verify drops it at ingestion so it never clutters the directory."""
    identity = _fresh()
    # sign_listing happily signs anything, so we construct by hand.
    payload = build_listing_signing_payload(
        listing_id="test", provider_id=identity.node_id,
        capacity_shards_per_sec=1.0, max_shard_bytes=1024,
        price_per_shard_ftns=0.01, tee_capable=False,
        stake_tier="open", advertised_at_unix=int(time.time()),
        ttl_seconds=60,
    )
    sig = identity.sign(payload)
    listing = ProviderListing(
        listing_id="test",
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        capacity_shards_per_sec=1.0,
        max_shard_bytes=1024,
        supported_dtypes=[],  # empty
        price_per_shard_ftns=0.01,
        tee_capable=False,
        stake_tier="open",
        advertised_at_unix=int(time.time()),
        ttl_seconds=60,
        signature=sig,
    )
    assert verify_listing(listing) is False


def test_listing_rejects_negative_ttl():
    identity = _fresh()
    payload = build_listing_signing_payload(
        listing_id="neg", provider_id=identity.node_id,
        capacity_shards_per_sec=1.0, max_shard_bytes=1024,
        price_per_shard_ftns=0.01, tee_capable=False,
        stake_tier="open", advertised_at_unix=int(time.time()),
        ttl_seconds=-1,
    )
    sig = identity.sign(payload)
    listing = ProviderListing(
        listing_id="neg", provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        capacity_shards_per_sec=1.0, max_shard_bytes=1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.01, tee_capable=False,
        stake_tier="open", advertised_at_unix=int(time.time()),
        ttl_seconds=-1, signature=sig,
    )
    assert verify_listing(listing) is False

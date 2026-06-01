"""Sprint 909 — content-access royalty gossip authentication + rate clamp.

The money-path review confirmed a CRITICAL: _on_content_access (subscriber
of GOSSIP_CONTENT_ACCESS) minted FTNS off UNSIGNED, unauthenticated gossip
with no signature check, and Case 2 used an attacker-supplied royalty_rate
from the payload. It was masked only by a cid-vs-content_id publisher/
subscriber key mismatch (the honest path was dead).

sp909 fixes the cluster atomically (the masking made partial fixes
dangerous): publishers sign the event (sign_content_access_event) and emit
content_id; the subscriber ed25519-verifies the signature against the
origin's key before any credit, clamps the Case-2 rate to the node's OWN
stored royalty_rate, and keys the replay nonce on (origin, nonce).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.content_uploader import (
    ContentUploader,
    UploadedContent,
    SOURCE_CREATOR_SHARE,
)
from prsm.node.content_provider import sign_content_access_event
from prsm.node.local_ledger import LocalLedger
from prsm.node.identity import generate_node_identity


async def _uploader(node_id="creator_node"):
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    await ledger.create_wallet(node_id, "creator")
    identity = MagicMock()
    identity.node_id = node_id
    up = ContentUploader(
        identity=identity, gossip=AsyncMock(), ledger=ledger, transport=AsyncMock(),
    )
    up._platform_royalty_transfer = AsyncMock()
    up.uploaded_content = {"parentA": UploadedContent(
        content_id="parentA", filename="p.bin", size_bytes=10,
        content_hash="h", creator_id=node_id, royalty_rate=0.04,
    )}
    return up, ledger


def _raw_event(royalty_rate=0.04, nonce="evt-1"):
    return {
        "content_id": "deriv1",
        "accessor_id": "acc",
        "creator_id": "someone_else",
        "royalty_rate": royalty_rate,
        "parent_cids": ["parentA"],
        "timestamp": 123.0,
        "access_nonce": nonce,
    }


_OWN_RATE_CREDIT = 0.04 * SOURCE_CREATOR_SHARE  # 1 of 1 parents, own rate


@pytest.mark.asyncio
async def test_unsigned_event_credits_nothing():
    up, ledger = await _uploader()
    await up._on_content_access("content_access", _raw_event(), "origin_x")
    assert await ledger.get_balance("creator_node") == 0.0


@pytest.mark.asyncio
async def test_bad_signature_credits_nothing():
    up, ledger = await _uploader()
    origin = generate_node_identity()
    ev = sign_content_access_event(_raw_event(), origin)
    ev["royalty_rate"] = 99.0  # tamper AFTER signing -> signature invalid
    await up._on_content_access("content_access", ev, origin.node_id)
    assert await ledger.get_balance("creator_node") == 0.0


@pytest.mark.asyncio
async def test_forged_origin_key_credits_nothing():
    """A signature from key A but claiming origin_public_key B must fail."""
    up, ledger = await _uploader()
    a = generate_node_identity()
    b = generate_node_identity()
    ev = sign_content_access_event(_raw_event(), a)
    ev["origin_public_key"] = b.public_key_b64  # mismatched key
    await up._on_content_access("content_access", ev, a.node_id)
    assert await ledger.get_balance("creator_node") == 0.0


@pytest.mark.asyncio
async def test_valid_signature_credits_at_clamped_own_rate():
    up, ledger = await _uploader()
    origin = generate_node_identity()
    # Attacker inflates royalty_rate to 10.0 in the (validly-signed) event.
    ev = sign_content_access_event(_raw_event(royalty_rate=10.0), origin)
    await up._on_content_access("content_access", ev, origin.node_id)
    bal = await ledger.get_balance("creator_node")
    # Credited at the node's OWN 0.04 rate, NOT the gossiped 10.0.
    assert bal == pytest.approx(_OWN_RATE_CREDIT)
    assert bal < 10.0 * SOURCE_CREATOR_SHARE


@pytest.mark.asyncio
async def test_signed_publish_roundtrips_and_reconciles_key():
    up, ledger = await _uploader()
    origin = generate_node_identity()
    # Publisher-shaped payload uses 'cid'; the signer reconciles to content_id.
    ev = sign_content_access_event({
        "cid": "deriv1", "accessor_id": "acc", "creator_id": "x",
        "royalty_rate": 0.04, "parent_cids": ["parentA"],
        "timestamp": 1.0, "access_nonce": "n1",
    }, origin)
    assert ev["content_id"] == "deriv1"  # key reconciled in the signed payload
    await up._on_content_access("content_access", ev, origin.node_id)
    assert await ledger.get_balance("creator_node") == pytest.approx(_OWN_RATE_CREDIT)


@pytest.mark.asyncio
async def test_replay_still_credits_once_after_auth():
    up, ledger = await _uploader()
    origin = generate_node_identity()
    ev = sign_content_access_event(_raw_event(nonce="n-rep"), origin)
    await up._on_content_access("content_access", ev, origin.node_id)
    await up._on_content_access("content_access", ev, origin.node_id)  # replay
    assert await ledger.get_balance("creator_node") == pytest.approx(_OWN_RATE_CREDIT)

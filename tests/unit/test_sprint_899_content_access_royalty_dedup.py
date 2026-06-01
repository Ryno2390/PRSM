"""Sprint 899 — content-access royalty crediting is replay-safe.

Same bug class as sp898, found extending that thread through the gossip
handlers. `content_uploader._on_content_access` (subscriber of
GOSSIP_CONTENT_ACCESS) credits royalties when this node is the content
creator (Case 1, via record_access) or a source creator of a derivative
(Case 2, a direct ledger.credit) — with NO idempotency guard at all
(not even a racy check).

The gossip layer's `_handle_gossip` invokes subscriber callbacks
unconditionally (no dedup before dispatch); transport-level nonce dedup
only catches the SAME gossip envelope. A node re-broadcasting the same
content-access event (fresh envelope), or a digest-response replay, or
just multi-path gossip, re-fires `_on_content_access` → royalties are
credited AGAIN on every delivery. Counterfeit FTNS minted from
redundant access events.

sp899 makes the credit idempotent: the GOSSIP_CONTENT_ACCESS payload
now carries a unique `access_nonce`, and `_on_content_access`
atomically CLAIMS it on the local ledger (the sp898 record_nonce
primitive) before crediting — exactly once per access event per node.
Legacy events without an explicit nonce fall back to a derived stable
key (cid:accessor:timestamp) so they still dedup.
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

# sp909 — content-access events must now be SIGNED to pass _on_content_access
# authentication. A single origin identity signs all events in this module.
_ORIGIN = generate_node_identity()


async def _uploader_with_real_ledger(node_id="creator_node"):
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    await ledger.create_wallet(node_id, "creator")
    identity = MagicMock()
    identity.node_id = node_id
    up = ContentUploader(
        identity=identity,
        gossip=AsyncMock(),
        ledger=ledger,
        transport=AsyncMock(),
    )
    # Isolate the royalty-credit dedup from the platform-transfer leg.
    up._platform_royalty_transfer = AsyncMock()
    return up, ledger


def _own_parent(node_id, cid="parentA"):
    return UploadedContent(
        content_id=cid,
        filename="parent.bin",
        size_bytes=10,
        content_hash="hash",
        creator_id=node_id,
        royalty_rate=0.04,
    )


def _access_event(nonce="evt-1", *, include_nonce=True):
    ev = {
        "content_id": "deriv1",
        "accessor_id": "accessor",
        "creator_id": "someone_else",
        "royalty_rate": 0.04,
        "parent_cids": ["parentA"],
        "timestamp": 123.0,
    }
    if include_nonce:
        ev["access_nonce"] = nonce
    # sp909 — sign so the event passes _on_content_access authentication.
    return sign_content_access_event(ev, _ORIGIN)


# source_royalty = royalty_rate * SOURCE_CREATOR_SHARE * (mine/total)
_EXPECTED = 0.04 * SOURCE_CREATOR_SHARE  # 1 of 1 parents → full share


# ── The bug: replayed access event double-credits royalty ────

@pytest.mark.asyncio
async def test_replayed_access_event_credits_royalty_once():
    up, ledger = await _uploader_with_real_ledger()
    up.uploaded_content = {"parentA": _own_parent("creator_node")}

    data = _access_event()
    await up._on_content_access("content_access", data, _ORIGIN.node_id)
    await up._on_content_access("content_access", data, _ORIGIN.node_id)  # replay

    bal = await ledger.get_balance("creator_node")
    assert bal == pytest.approx(_EXPECTED), (
        f"double-credit: {bal} != {_EXPECTED}"
    )


@pytest.mark.asyncio
async def test_many_replays_credit_once():
    up, ledger = await _uploader_with_real_ledger()
    up.uploaded_content = {"parentA": _own_parent("creator_node")}
    data = _access_event()
    for _ in range(6):
        await up._on_content_access("content_access", data, _ORIGIN.node_id)
    assert await ledger.get_balance("creator_node") == pytest.approx(
        _EXPECTED,
    )


# ── Distinct access events each credit (no over-dedup) ───────

@pytest.mark.asyncio
async def test_distinct_access_events_each_credit():
    up, ledger = await _uploader_with_real_ledger()
    up.uploaded_content = {"parentA": _own_parent("creator_node")}

    await up._on_content_access(
        "content_access", _access_event("evt-1"), _ORIGIN.node_id,
    )
    await up._on_content_access(
        "content_access", _access_event("evt-2"), _ORIGIN.node_id,
    )
    bal = await ledger.get_balance("creator_node")
    assert bal == pytest.approx(2 * _EXPECTED)


# ── Legacy events (no explicit nonce) still dedup ────────────

@pytest.mark.asyncio
async def test_legacy_event_without_nonce_dedups_on_derived_key():
    up, ledger = await _uploader_with_real_ledger()
    up.uploaded_content = {"parentA": _own_parent("creator_node")}

    legacy = _access_event(include_nonce=False)  # pre-sp899 shape, signed
    await up._on_content_access("content_access", legacy, _ORIGIN.node_id)
    await up._on_content_access("content_access", legacy, _ORIGIN.node_id)
    assert await ledger.get_balance("creator_node") == pytest.approx(
        _EXPECTED,
    )

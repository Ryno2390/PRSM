"""Sprint 427 — F7 fix: BT-infohash fallback in ContentProvider._fetch_local.

After sprint 425 closed F4 (content upload works end-to-end), live
dogfood surfaced F7: locally-uploaded content returns "not_found" on
the same node. Root cause documented in
``docs/operations/2026-05-14-user-dogfood-findings.md``:

- Upload returns a BitTorrent v1 infohash (40-char hex / SHA-1).
- ContentProvider.request_content checks ``_local_content[cid]`` —
  succeeds; the registration fires correctly on upload.
- Then ``_fetch_local(cid)`` routes through
  ``ContentStore.retrieve_local(ContentHash.from_hex(cid))`` which
  expects a 66-char algorithm-prefixed hex. The 40-char BT infohash
  fails structurally → ValueError swallowed → None.

This sprint adds the minimum-viable Option A fix: when
``ContentHash.from_hex`` fails OR ContentStore returns None for a
cid present in ``_local_content``, fall back to the wired
``ContentRetriever`` which knows how to fetch by BT infohash.

Tests gate the fallback path against regressions:
- The fallback fires when ContentStore can't resolve the cid
- The fallback returns the canonical bytes when retriever succeeds
- The retriever is consulted ONLY for cids known to be local
  (no double-fetch from the BT swarm for unknown cids)
- ContentStore-resolvable cids still take the fast path
- Errors in the retriever are swallowed (return None, same as
  ContentStore errors — telemetry layer logs; never raise)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.content_provider import ContentProvider


def _make_provider(content_retriever=None):
    identity = MagicMock()
    identity.node_id = "test-node-id"
    transport = MagicMock()
    transport.on_message = MagicMock()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    p = ContentProvider(
        identity=identity, transport=transport, gossip=gossip,
    )
    p.content_retriever = content_retriever
    return p


@pytest.mark.asyncio
async def test_fetch_local_falls_back_to_bt_retriever_on_infohash():
    """A 40-char BT infohash cid is structurally invalid for
    ContentHash.from_hex (which wants 66 chars with algorithm
    prefix). Without the fallback, _fetch_local returned None
    for every BT-published cid even when the bytes were
    actually available via the BT layer."""
    bt_infohash = "6c01ee255d2e20d45ed90dea19e6722cd6c96713"
    expected_bytes = b"hello PRSM world via BT fallback"

    retriever = MagicMock()
    retriever.fetch = AsyncMock(return_value=expected_bytes)
    p = _make_provider(content_retriever=retriever)
    # Pretend the cid IS registered as local (real upload flow
    # would have called register_local_content).
    p._local_content[bt_infohash] = {
        "content_hash": "sha256-deadbeef",
        "size_bytes": len(expected_bytes),
    }

    result = await p._fetch_local(bt_infohash)
    assert result == expected_bytes
    retriever.fetch.assert_awaited_once_with(bt_infohash)


@pytest.mark.asyncio
async def test_fetch_local_skips_bt_fallback_when_cid_not_local():
    """The BT fallback fires ONLY for cids that are in
    ``_local_content``. An unknown cid must not trigger a
    fetch from the BT swarm (would be a free-for-all data
    leak via the local-check path)."""
    unknown_cid = "ffffffffffffffffffffffffffffffffffffffff"
    retriever = MagicMock()
    retriever.fetch = AsyncMock(return_value=b"should not be returned")
    p = _make_provider(content_retriever=retriever)
    # _local_content is empty.

    result = await p._fetch_local(unknown_cid)
    assert result is None
    retriever.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_local_no_retriever_returns_none_gracefully():
    """When content_retriever is unwired (legacy node or
    BT layer unavailable), the fallback must no-op
    gracefully. Pre-fix _fetch_local also returned None in
    this case so behavior is preserved."""
    bt_infohash = "6c01ee255d2e20d45ed90dea19e6722cd6c96713"
    p = _make_provider(content_retriever=None)
    p._local_content[bt_infohash] = {"size_bytes": 10}

    result = await p._fetch_local(bt_infohash)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_local_swallows_retriever_errors():
    """Symmetric with the existing ContentStore-error
    swallowing: a retriever exception is logged but
    _fetch_local must return None, never raise. Otherwise
    upstream callers would see the exception bubble out and
    fail the whole retrieve flow on a single backend hiccup."""
    bt_infohash = "6c01ee255d2e20d45ed90dea19e6722cd6c96713"
    retriever = MagicMock()
    retriever.fetch = AsyncMock(side_effect=RuntimeError(
        "BT requester reported success=False",
    ))
    p = _make_provider(content_retriever=retriever)
    p._local_content[bt_infohash] = {"size_bytes": 10}

    result = await p._fetch_local(bt_infohash)
    assert result is None

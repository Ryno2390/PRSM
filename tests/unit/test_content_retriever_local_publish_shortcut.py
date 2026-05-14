"""Sprint 428 — F8 fix: ContentRetriever short-circuits to locally-published bytes.

After sprint 427 shipped the F7 Option A shim (BT-infohash fallback
in ContentProvider._fetch_local), live verification surfaced F8:
bt_provider and bt_requester use separate libtorrent sessions, so
the shim engages but the BT swarm fetch fails ("Torrent not found")
because the locally-seeded torrent isn't visible to the requester.

This sprint adds F8's Option A' fix: ContentRetriever.fetch checks
the wired ContentPublisher first. If the publisher knows the
infohash (i.e., we published it locally), return its staged bytes
directly without involving the BT swarm at all. Symmetric pattern
with sprint 427's ContentProvider shim.

Scope: Tier A only. Tier A is the default publish mode and
produces the BT-infohash cids that block F7/F8. Tier B/C local
short-circuit can be added later if the same friction surfaces for
encrypted publishes.

Tests pin:
- Local shortcut fires for known-published infohash
- Local shortcut returns canonical bytes (same as published data)
- Local shortcut absent → fall through to BT requester
- Local publisher unwired → fall through to BT requester
- Tier B/C (multi-file staged dir) skips shortcut (deferred)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.content_publisher import ContentRetriever


def _make_retriever(tmp_path, *, with_publisher=None, bt_result=None):
    requester = MagicMock()
    requester.request_content = AsyncMock(return_value=bt_result)
    retriever = ContentRetriever(
        bt_requester=requester,
        cache_dir=tmp_path / "cache",
    )
    retriever.content_publisher = with_publisher
    return retriever, requester


class _FakeTierAPublisher:
    """Mimics the subset of ContentPublisher needed for the local-
    shortcut path: knows which infohashes it published and where the
    Tier A staged file lives."""

    def __init__(self):
        self._published: dict[str, Path] = {}

    def register(self, infohash: str, path: Path):
        self._published[infohash] = path

    def local_publish_path(self, infohash: str):
        return self._published.get(infohash)


@pytest.mark.asyncio
async def test_local_shortcut_returns_staged_bytes(tmp_path):
    """When the wired publisher reports it published this
    infohash locally + the staged path is a Tier A single
    file, ContentRetriever.fetch returns the staged bytes
    directly. No BT round-trip — closes F8 single-node
    self-fetch."""
    data = b"hello PRSM world via local-publish shortcut"
    infohash = "57f2d3ac5442df7ac500b87b38b0ecfa78d76124"
    staged = tmp_path / hashlib.sha256(data).hexdigest()
    staged.write_bytes(data)

    publisher = _FakeTierAPublisher()
    publisher.register(infohash, staged)
    retriever, requester = _make_retriever(
        tmp_path, with_publisher=publisher,
    )

    result = await retriever.fetch(infohash)
    assert result == data
    # Critically, the BT requester was NEVER called — local shortcut
    # short-circuits before the swarm probe.
    requester.request_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_shortcut_skipped_for_unknown_infohash(tmp_path):
    """Unknown infohashes (not locally-published) must fall
    through to the BT requester. The shortcut must not hide
    real remote-fetch behavior for legitimate cross-node
    requests."""
    publisher = _FakeTierAPublisher()  # empty
    bt_result = MagicMock()
    bt_result.success = False
    bt_result.error = "stub: not found in swarm"
    retriever, requester = _make_retriever(
        tmp_path, with_publisher=publisher, bt_result=bt_result,
    )

    with pytest.raises(RuntimeError, match="not found in swarm"):
        await retriever.fetch("ffffffffffffffffffffffffffffffffffffffff")
    requester.request_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_shortcut_no_publisher_falls_through(tmp_path):
    """When no publisher is wired (legacy retrievers, BT-only
    nodes), the shortcut must no-op gracefully and the BT
    request path remains the canonical fetch route."""
    bt_result = MagicMock()
    bt_result.success = False
    bt_result.error = "stub"
    retriever, requester = _make_retriever(
        tmp_path, with_publisher=None, bt_result=bt_result,
    )

    with pytest.raises(RuntimeError):
        await retriever.fetch(
            "57f2d3ac5442df7ac500b87b38b0ecfa78d76124",
        )
    requester.request_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_shortcut_skips_when_staged_path_missing(tmp_path):
    """If publisher claims to have the infohash but the
    staged file is gone (cleanup race, disk error), fall
    through to BT — don't lie about local availability."""
    publisher = _FakeTierAPublisher()
    publisher.register(
        "57f2d3ac5442df7ac500b87b38b0ecfa78d76124",
        tmp_path / "does-not-exist",
    )
    bt_result = MagicMock()
    bt_result.success = False
    bt_result.error = "stub"
    retriever, requester = _make_retriever(
        tmp_path, with_publisher=publisher, bt_result=bt_result,
    )

    with pytest.raises(RuntimeError):
        await retriever.fetch(
            "57f2d3ac5442df7ac500b87b38b0ecfa78d76124",
        )
    requester.request_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_shortcut_skips_when_staged_path_is_directory(tmp_path):
    """Tier B/C publishes produce a multi-file staging dir,
    not a single file. The local shortcut is Tier A only —
    if the staged path is a directory, fall through to BT
    rather than trying to read a dir as bytes (would either
    OSError or silently return wrong content)."""
    publisher = _FakeTierAPublisher()
    tier_bc_dir = tmp_path / "tier-bc-staged-root"
    tier_bc_dir.mkdir()
    (tier_bc_dir / "manifest.bin").write_bytes(b"encrypted")
    publisher.register(
        "57f2d3ac5442df7ac500b87b38b0ecfa78d76124",
        tier_bc_dir,
    )
    bt_result = MagicMock()
    bt_result.success = False
    bt_result.error = "stub"
    retriever, requester = _make_retriever(
        tmp_path, with_publisher=publisher, bt_result=bt_result,
    )

    with pytest.raises(RuntimeError):
        await retriever.fetch(
            "57f2d3ac5442df7ac500b87b38b0ecfa78d76124",
        )
    requester.request_content.assert_awaited_once()

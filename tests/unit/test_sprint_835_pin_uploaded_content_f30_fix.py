"""Sprint 835 — F30 fix: pin endpoint handles uploaded-content
CID (torrent infohash) without false 404.

Sprint 834 added inline /content/{cid}/pin wired to
StorageProvider.pin_content. Live dogfood surfaced F30: after
`prsm storage upload`, the returned CID does NOT resolve via
StorageProvider.exists_local even though the upload succeeded.

Two distinct problems composed:

1. Identifier-space mismatch: the operator-facing CID is the
   BitTorrent infohash (40-char SHA-1) produced by
   ContentPublisher. StorageProvider.pin_content called
   ContentHash.from_hex(cid), which expects algo-prefixed
   SHA-256 hex (66 chars total). Length AND format mismatch.

2. Manifest-cache scope: even with the correct ContentHash,
   ContentStore.exists_local checks `self._manifest_cache` —
   populated only for sharded-download content (sprint 263).
   Tier A uploads stage the raw file but DON'T register a
   manifest entry. So even fixing #1 wouldn't make exists_local
   return True for content this node uploaded itself.

Sprint 835 fix: when the pin endpoint receives a CID that
matches `content_uploader.uploaded_content[cid]`, we KNOW the
content is locally available (we put it there). Short-circuit
the StorageProvider.pin_content path and write
PinnedContent(cid, size_bytes) directly into
sp.pinned_content[cid] from the UploadedContent entry. The
rest of the StorageProvider machinery (challenge scheduling,
size accounting) reads from the same dict.

Content this node didn't upload itself (retrieved from peers)
still goes through the original pin_content path so the
exists_local manifest gate fires correctly there.

Live-attested 2026-05-24 — full storage CLI round-trip:
  $ prsm storage upload payload.txt → CID: 0e9048b6...
  $ prsm storage pin 0e9048b6... → ✅ pinned, 22 bytes
  $ prsm storage pins → table with the pinned entry
Pre-835 the pin step always 404'd.

Pin tests:
- Pin works when CID is in uploaded_content (sprint 835 path)
- Pin records the entry in sp.pinned_content[cid]
- Pin falls back to pin_content() when CID is NOT in
  uploaded_content
- Fallback path returns 404 when pin_content returns False
- 503 still surfaces when storage_provider is None
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_node():
    """Minimal node with content_uploader + storage_provider."""
    node = MagicMock()
    # storage_provider with empty pinned_content + async
    # pin_content returning False by default.
    sp = MagicMock()
    sp.pinned_content = {}
    sp.pin_content = AsyncMock(return_value=False)
    node.storage_provider = sp
    # content_uploader with an empty uploaded_content map.
    cu = MagicMock()
    cu.uploaded_content = {}
    node.content_uploader = cu
    return node


@pytest.mark.asyncio
async def test_pin_works_for_uploaded_content(mock_node):
    """The F30 fix: a CID this node uploaded passes pin without
    hitting exists_local. Records directly in pinned_content."""
    from fastapi import FastAPI
    from prsm.node.api import create_api_app

    # Pre-populate uploaded_content with the operator-facing CID.
    entry = MagicMock()
    entry.content_hash = "deadbeef" * 8  # 64-char SHA-256 hex
    entry.size_bytes = 42
    mock_node.content_uploader.uploaded_content["abc123"] = entry

    app: FastAPI = create_api_app(mock_node, enable_security=False)

    # Find the pin route + call it directly (avoids httpx mocks).
    pin_route = None
    for route in app.routes:
        if getattr(route, "path", "") == "/content/{cid}/pin":
            pin_route = route
            break
    assert pin_route is not None

    result = await pin_route.endpoint(cid="abc123")
    assert result["pinned"] is True
    assert result["cid"] == "abc123"
    assert result["size_bytes"] == 42

    # And the pinned_content dict was populated.
    assert "abc123" in mock_node.storage_provider.pinned_content
    pinned = mock_node.storage_provider.pinned_content["abc123"]
    assert pinned.cid == "abc123"
    assert pinned.size_bytes == 42

    # CRUCIAL: sp.pin_content was NOT called — sprint 835 short-
    # circuits this path entirely for uploaded content.
    mock_node.storage_provider.pin_content.assert_not_called()


@pytest.mark.asyncio
async def test_pin_falls_back_for_non_uploaded_content(mock_node):
    """When the CID isn't in uploaded_content, sprint 835 falls
    back to the legacy StorageProvider.pin_content path
    (handles retrieved-from-peers content)."""
    from prsm.node.api import create_api_app
    from fastapi import HTTPException

    mock_node.storage_provider.pin_content.return_value = False

    app = create_api_app(mock_node, enable_security=False)
    pin_route = None
    for route in app.routes:
        if getattr(route, "path", "") == "/content/{cid}/pin":
            pin_route = route
            break

    # Not in uploaded_content → falls to pin_content → returns
    # False → endpoint raises 404.
    with pytest.raises(HTTPException) as exc_info:
        await pin_route.endpoint(cid="not_uploaded_here")
    assert exc_info.value.status_code == 404
    assert "not present locally" in exc_info.value.detail

    # The fallback path DID call pin_content.
    mock_node.storage_provider.pin_content.assert_called_once_with(
        "not_uploaded_here",
    )


@pytest.mark.asyncio
async def test_pin_503_when_storage_provider_unwired(mock_node):
    """Regression: sprint 834's 503 path still fires when
    storage_provider isn't on the node at all."""
    from prsm.node.api import create_api_app
    from fastapi import HTTPException

    mock_node.storage_provider = None  # unwired

    app = create_api_app(mock_node, enable_security=False)
    pin_route = None
    for route in app.routes:
        if getattr(route, "path", "") == "/content/{cid}/pin":
            pin_route = route
            break

    with pytest.raises(HTTPException) as exc_info:
        await pin_route.endpoint(cid="abc")
    assert exc_info.value.status_code == 503
    assert "Storage provider not initialized" in exc_info.value.detail

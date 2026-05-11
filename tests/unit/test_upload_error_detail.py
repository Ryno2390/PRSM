"""Sprint 180 — /content/upload surfaces underlying exception in
the 502 detail instead of a generic "content store unavailable?".

Sprint 179 dogfood revealed multiple libtorrent API-drift bugs
(settings_pack removed, add_dht_router moved, get_torrent_info
removed). Each one surfaced as the same opaque 502:
  "Upload failed — content store unavailable?"

Without seeing the underlying log, operators can't tell whether
the failure is:
  - libtorrent API mismatch (operator-fixable: pin a working
    libtorrent version or rebuild)
  - VPN blocking P2P traffic (operator-fixable: disable VPN or
    use a P2P-friendly plan)
  - Disk full / permission error
  - Genuine code bug

Sprint 180 surfaces the exception class + message in the 502
detail and distinguishes None-return from raised-exception:

  - upload_text() raises: 502 with
    "Upload failed: ConnectionRefusedError: ..."
  - upload_text() returns None: 502 with explanation listing
    the common None-return causes
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_with_uploader(uploader_behavior):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.content_uploader = MagicMock()
    node.content_uploader.content_publisher = MagicMock()  # wired
    node.content_uploader.upload_text = uploader_behavior
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_502_detail_includes_exception_class_and_message():
    """Sprint 180 — upload_text raising surfaces type + message
    in the 502 body."""
    node = _node_with_uploader(
        AsyncMock(side_effect=ConnectionRefusedError("port 6881 blocked")),
    )
    resp = _client(node).post("/content/upload", json={"text": "hi"})
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "ConnectionRefusedError" in detail
    assert "port 6881 blocked" in detail


def test_502_detail_lists_common_causes_when_none_returned():
    """Sprint 180 — upload_text returning None gets an explanatory
    detail listing the common reasons (unwired publisher, swallowed
    publish exception, BT layer crash)."""
    node = _node_with_uploader(AsyncMock(return_value=None))
    resp = _client(node).post("/content/upload", json={"text": "hi"})
    assert resp.status_code == 502
    detail = resp.json()["detail"].lower()
    assert "returned none" in detail
    # Mentions at least one of the canonical None-return causes.
    assert any(
        cue in detail for cue in [
            "publisher",
            "publish",
            "bittorrent",
        ]
    )


def test_502_detail_distinguishes_exception_vs_none():
    """Sprint 180 invariant — the two failure modes are
    distinguishable from the detail text alone."""
    raise_resp = _client(
        _node_with_uploader(
            AsyncMock(side_effect=RuntimeError("disk full")),
        ),
    ).post("/content/upload", json={"text": "hi"})
    none_resp = _client(
        _node_with_uploader(AsyncMock(return_value=None)),
    ).post("/content/upload", json={"text": "hi"})

    raise_detail = raise_resp.json()["detail"]
    none_detail = none_resp.json()["detail"]
    assert raise_detail != none_detail
    # Raise path mentions the exception class explicitly.
    assert "RuntimeError" in raise_detail
    # None path explicitly says "returned None".
    assert "returned None" in none_detail

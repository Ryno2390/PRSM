"""Sprint 291 — /content/upload fingerprint hook.

When a content upload succeeds, the response is decorated
with fingerprint dedup signal:
  - duplicate_of_creator: original creator's address if this
                          fingerprint was already registered
                          to someone else; None otherwise.
  - canonical_creator:    the authoritative creator address
                          for this fingerprint (== request
                          creator on first upload, == prior
                          uploader on duplicate).

The bytes upload still proceeds — operators may want the
content cached for serving even if it's a duplicate.
Royalty routing to the canonical creator is enforced
elsewhere (sprint 248 royalty distribution arc).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from prsm.marketplace.content_fingerprint_registry import (
    ContentFingerprintRegistry,
)
from prsm.node.api import create_api_app


class _FakeUploader:
    """Returns a deterministic UploadedContent-shaped result."""

    def __init__(self, content_hash="sha256-abc",
                 creator_id="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"):
        self.content_hash = content_hash
        self.creator_id = creator_id
        # Sprint-160 upload handler requires non-None
        # content_publisher attribute (libtorrent layer
        # check).
        self.content_publisher = MagicMock()

    async def upload_text(self, **kwargs):
        result = MagicMock()
        result.cid = "bafy-xyz"
        result.filename = kwargs.get("filename", "f.bin")
        result.size_bytes = 1024
        result.content_hash = self.content_hash
        # Whichever creator was provided in the request is
        # what the uploader returns on its result. Dedup is a
        # separate layer.
        result.creator_id = (
            kwargs.get("creator_eth_address")
            or self.creator_id
        )
        result.royalty_rate = (
            kwargs.get("royalty_rate") or 0.1
        )
        result.parent_cids = []
        return result


def _client(
    fingerprint_registry=None,
    uploader=None,
    content_hash="sha256-abc",
):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._content_fingerprint_registry = fingerprint_registry
    node.content_uploader = (
        uploader or _FakeUploader(content_hash=content_hash)
    )
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── First upload registers fingerprint ───────────────────


def test_first_upload_registers_fingerprint():
    reg = ContentFingerprintRegistry()
    resp = _client(reg).post(
        "/content/upload",
        json={
            "text": "hello world",
            "filename": "f.txt",
            "creator_eth_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # Registered Alice as canonical
    assert reg.canonical_creator("sha256-abc") == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    # Response signals it's not a duplicate
    assert body["duplicate_of_creator"] is None
    assert body["canonical_creator"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


# ── Duplicate upload surfaces canonical creator ──────────


def test_duplicate_upload_surfaces_canonical():
    reg = ContentFingerprintRegistry()
    # Alice uploads first
    reg.register("sha256-abc", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # Bob tries to re-upload same content
    resp = _client(reg).post(
        "/content/upload",
        json={
            "text": "hello world",
            "filename": "f.txt",
            "creator_eth_address": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # Upload bytes succeed (we still cache for serving)
    assert body["cid"] == "bafy-xyz"
    # Dedup signal: bob's upload is recognized as duplicate
    assert body["duplicate_of_creator"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert body["canonical_creator"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    # The duplicate attempt was counted
    entry = reg.get_entry("sha256-abc")
    assert entry.duplicate_attempt_count == 1


# ── Same creator re-upload not flagged as duplicate ──────


def test_same_creator_re_upload_not_duplicate():
    reg = ContentFingerprintRegistry()
    reg.register("sha256-abc", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    resp = _client(reg).post(
        "/content/upload",
        json={
            "text": "hello world",
            "filename": "f.txt",
            "creator_eth_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        },
    )
    body = resp.json()
    assert body["duplicate_of_creator"] is None
    assert body["canonical_creator"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    # No duplicate attempts logged
    entry = reg.get_entry("sha256-abc")
    assert entry.duplicate_attempt_count == 0


# ── Registry unwired: hook gracefully no-ops ─────────────


def test_upload_works_when_registry_unwired():
    """Backwards-compat: pre-sprint-291 callers don't have
    the registry wired. Upload still succeeds; dedup fields
    surface as None."""
    resp = _client(None).post(
        "/content/upload",
        json={
            "text": "hello world",
            "filename": "f.txt",
            "creator_eth_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["duplicate_of_creator"] is None
    assert body.get("canonical_creator") is None


# ── No creator_eth_address: skip dedup ───────────────────


def test_upload_without_creator_skips_dedup():
    """Anonymous/operator-internal uploads don't carry
    creator_eth_address. Skip dedup tracking entirely."""
    reg = ContentFingerprintRegistry()
    resp = _client(reg).post(
        "/content/upload",
        json={"text": "hello world", "filename": "f.txt"},
    )
    assert resp.status_code == 200
    # Registry should NOT have learned anything from this
    # anonymous upload
    assert reg.count() == 0


# ── Registry exception doesn't break upload ──────────────


def test_registry_exception_does_not_break_upload():
    class BoomRegistry:
        def register(self, **kwargs):
            raise RuntimeError("registry exploded")
    resp = _client(BoomRegistry()).post(
        "/content/upload",
        json={
            "text": "hello world",
            "filename": "f.txt",
            "creator_eth_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cid"] == "bafy-xyz"


# ── Inspection endpoints ─────────────────────────────────


def test_get_fingerprint_503_when_unwired():
    resp = _client(None).get(
        "/marketplace/fingerprint/sha256-abc",
    )
    assert resp.status_code == 503


def test_get_fingerprint_404_unknown():
    reg = ContentFingerprintRegistry()
    resp = _client(reg).get(
        "/marketplace/fingerprint/sha256-unknown",
    )
    assert resp.status_code == 404


def test_get_fingerprint_happy_path():
    reg = ContentFingerprintRegistry()
    reg.register("sha256-abc", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    resp = _client(reg).get(
        "/marketplace/fingerprint/sha256-abc",
    )
    body = resp.json()
    assert body["content_hash"] == "sha256-abc"
    assert body["canonical_creator"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert body["duplicate_attempt_count"] == 0


def test_list_fingerprints_503_when_unwired():
    resp = _client(None).get("/marketplace/fingerprint")
    assert resp.status_code == 503


def test_list_fingerprints_empty():
    reg = ContentFingerprintRegistry()
    resp = _client(reg).get("/marketplace/fingerprint")
    body = resp.json()
    assert body["fingerprints"] == []
    assert body["count"] == 0


def test_list_fingerprints_newest_first():
    reg = ContentFingerprintRegistry()
    reg.register("sha256-a", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", timestamp=100.0)
    reg.register("sha256-b", "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", timestamp=200.0)
    resp = _client(reg).get("/marketplace/fingerprint")
    body = resp.json()
    hashes = [f["content_hash"] for f in body["fingerprints"]]
    assert hashes == ["sha256-b", "sha256-a"]


def test_list_fingerprints_invalid_limit():
    reg = ContentFingerprintRegistry()
    resp = _client(reg).get(
        "/marketplace/fingerprint?limit=0",
    )
    assert resp.status_code == 422


# ── MCP tool ─────────────────────────────────────────────


import pytest
from unittest.mock import patch
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_content_fingerprint,
)


def test_mcp_tool_registered():
    assert "prsm_content_fingerprint" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_content_fingerprint({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_content_fingerprint(
        {"action": "explode"},
    )
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_lookup_requires_content_hash():
    r = await handle_prsm_content_fingerprint(
        {"action": "lookup"},
    )
    assert "content_hash" in r


@pytest.mark.asyncio
async def test_mcp_lookup_happy_path():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "content_hash": "sha256-abc",
            "canonical_creator": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "first_seen_unix": 100,
            "duplicate_attempt_count": 3,
        }),
    ) as mock_call:
        r = await handle_prsm_content_fingerprint({
            "action": "lookup",
            "content_hash": "sha256-abc",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/marketplace/fingerprint/sha256-abc"
    assert "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in r
    assert "3" in r  # duplicate attempts


@pytest.mark.asyncio
async def test_mcp_lookup_unknown_returns_helpful():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "detail": "no fingerprint for sha256-unknown",
        }),
    ):
        r = await handle_prsm_content_fingerprint({
            "action": "lookup",
            "content_hash": "sha256-unknown",
        })
    assert "unknown" in r.lower() or "no fingerprint" in r.lower()


@pytest.mark.asyncio
async def test_mcp_list_renders_rows():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "fingerprints": [{
                "content_hash": "sha256-x",
                "canonical_creator": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "first_seen_unix": 100,
                "duplicate_attempt_count": 2,
            }],
            "count": 1, "limit": 100,
        }),
    ):
        r = await handle_prsm_content_fingerprint(
            {"action": "list"},
        )
    assert "sha256-x" in r
    # MCP truncates to first 16 chars for compact rendering
    assert "0xaaaaaaaaaaaaaa" in r
    # Duplicate-attempt warning marker present
    assert "2 dup" in r.lower() or "dup" in r.lower()

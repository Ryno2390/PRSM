"""Sprint 258 — allocation_mode threaded through audit ring +
endpoint + MCP wrapper.

Post-hoc audit can now distinguish a 50/50 uniform-2-shard split
from a coincidentally-equal rate_weighted outcome.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.royalty_dispatch_log import (
    RoyaltyDispatchEntry, RoyaltyDispatchRing,
)
from prsm.mcp_server import handle_prsm_royalty_dispatch_history


def test_entry_round_trip_carries_allocation_mode():
    e = RoyaltyDispatchEntry(
        timestamp=100.0,
        job_id="j1",
        cid="c1",
        status="sent",
        tx_hash="0xtx",
        gross_wei=42,
        allocation_mode="rate_weighted",
    )
    d = e.to_dict()
    assert d["allocation_mode"] == "rate_weighted"


def test_entry_defaults_none_for_backcompat():
    e = RoyaltyDispatchEntry(
        timestamp=100.0,
        job_id="j1",
        cid="c1",
        status="sent",
        tx_hash="0xtx",
        gross_wei=42,
    )
    assert e.allocation_mode is None
    assert e.to_dict()["allocation_mode"] is None


def test_ring_persists_and_loads_allocation_mode(tmp_path):
    r1 = RoyaltyDispatchRing(persist_dir=tmp_path)
    r1.append(
        job_id="j1", cid="c1", status="sent",
        tx_hash="0xtx", gross_wei=42, timestamp=100.0,
        allocation_mode="rate_weighted",
    )
    r2 = RoyaltyDispatchRing(persist_dir=tmp_path)
    recent = r2.recent(limit=10)
    assert recent[0].allocation_mode == "rate_weighted"


def test_ring_filter_by_allocation_mode():
    r = RoyaltyDispatchRing()
    r.append(
        job_id="j1", cid="c1", status="sent",
        tx_hash="0x1", gross_wei=1,
        allocation_mode="uniform",
    )
    r.append(
        job_id="j2", cid="c2", status="sent",
        tx_hash="0x2", gross_wei=1,
        allocation_mode="rate_weighted",
    )
    r.append(
        job_id="j3", cid="c3", status="sent",
        tx_hash="0x3", gross_wei=1,
        allocation_mode="uniform",
    )
    only_uniform = r.recent(limit=10, allocation_mode="uniform")
    assert [e.job_id for e in only_uniform] == ["j3", "j1"]
    only_weighted = r.recent(
        limit=10, allocation_mode="rate_weighted",
    )
    assert [e.job_id for e in only_weighted] == ["j2"]


def test_pre_258_persisted_entries_load_with_none():
    """Pre-258 disk files have no allocation_mode key. Load
    cleanly with the field None."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        from pathlib import Path
        p = Path(tmp_path)
        legacy = {
            "timestamp": 100.0,
            "job_id": "j-old",
            "cid": "c-old",
            "status": "sent",
            "tx_hash": "0xold",
            "gross_wei": 100,
        }
        (p / "legacy.json").write_text(json.dumps(legacy))
        ring = RoyaltyDispatchRing(persist_dir=p)
        e = ring.recent(limit=10)[0]
        assert e.job_id == "j-old"
        assert e.allocation_mode is None


# Endpoint passthrough


def _client(ring=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._royalty_dispatch_ring = ring
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_endpoint_allocation_mode_filter():
    r = RoyaltyDispatchRing()
    r.append(
        job_id="j1", cid="c1", status="sent",
        tx_hash="0x1", gross_wei=1, allocation_mode="uniform",
    )
    r.append(
        job_id="j2", cid="c2", status="sent",
        tx_hash="0x2", gross_wei=1,
        allocation_mode="rate_weighted",
    )
    resp = _client(r).get(
        "/admin/royalty-dispatch-history?allocation_mode=rate_weighted",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["job_id"] == "j2"
    assert (
        body["entries"][0]["allocation_mode"] == "rate_weighted"
    )


# MCP wrapper passthrough


@pytest.mark.asyncio
async def test_mcp_passes_allocation_mode_to_query():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "entries": [], "total": 0,
            "offset": 0, "limit": 20,
        }),
    ) as mock_call:
        await handle_prsm_royalty_dispatch_history({
            "allocation_mode": "rate_weighted",
        })
    args, _ = mock_call.await_args
    assert "allocation_mode=rate_weighted" in args[1]


@pytest.mark.asyncio
async def test_mcp_renders_allocation_mode_inline():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "entries": [{
                "timestamp": 100.0,
                "job_id": "j1",
                "cid": "c1",
                "status": "sent",
                "tx_hash": "0xtx",
                "gross_wei": 42,
                "allocation_mode": "rate_weighted",
            }],
            "total": 1, "offset": 0, "limit": 20,
        }),
    ):
        result = await handle_prsm_royalty_dispatch_history({})
    assert "mode=rate_weighted" in result

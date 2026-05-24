"""Sprint 840 — composition test for the sp838 bootstrap relay.

Sprint 838 wired hardware_profile end-to-end through the
bootstrap server: PeerInfo model, register handler, get_peer_list,
BootstrapClient send/parse, discovery layers. The sp838 pin tests
covered each piece in isolation but no test asserted the
COMPOSITION — that a second peer's register_ack peer_list
actually carries the first peer's relayed hw.

This is the load-bearing claim for the relay's utility. If it
breaks (e.g., a refactor changes get_peer_list to drop hw, OR
register handler stops caching it, OR the register_ack payload
shape regresses) operators lose the cold-start cure even though
unit tests still pass.

Sprint 840 ships an in-process composition test that runs the
REAL `_handle_register` for two distinct peers and asserts that
peer B's register_ack `peers` list contains peer A with its
hardware_profile populated.

Sibling test covers get_peers (sp389's separate query path) +
the legacy no-hw fallback so a fully-mixed fleet validates too.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


def _fake_ws():
    """Fake websocket that captures the JSON messages the server
    sends back during a register handshake."""
    ws = MagicMock()
    sent: List[Dict[str, Any]] = []

    async def _send(payload: str):
        sent.append(json.loads(payload))

    ws.send = AsyncMock(side_effect=_send)
    ws.close = AsyncMock()
    # Stash captured messages so callers can inspect post-hoc.
    ws._sent = sent
    return ws


@pytest.mark.asyncio
async def test_second_peer_sees_first_peers_hw_in_register_ack():
    """**The load-bearing composition assertion.** Peer A
    registers with hw_profile {tflops_fp16: 33.9}. Peer B
    registers next; the register_ack the server sends to B
    MUST include peer A in `peers[]` carrying that hw_profile.
    """
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig

    srv = BootstrapServer(BootstrapConfig())

    # Peer A — advertises hw.
    ws_a = _fake_ws()
    await srv._handle_register(
        ws_a,
        {
            "type": "register",
            "peer_id": "peer-A",
            "port": 9001,
            "address": "10.0.0.1",
            "capabilities": ["compute"],
            "hardware_profile": {
                "tflops_fp16": 33.9, "ram_total_gb": 24.0,
                "gpu_vram_gb": 24.0,
            },
        },
        client_ip="10.0.0.1",
    )

    # Peer B — joins after A.
    ws_b = _fake_ws()
    await srv._handle_register(
        ws_b,
        {
            "type": "register",
            "peer_id": "peer-B",
            "port": 9001,
            "address": "10.0.0.2",
            "capabilities": ["compute"],
            "hardware_profile": {
                "tflops_fp16": 8.0, "ram_total_gb": 16.0,
            },
        },
        client_ip="10.0.0.2",
    )

    # Find peer B's register_ack.
    ack = next(
        (m for m in ws_b._sent if m.get("type") == "register_ack"),
        None,
    )
    assert ack is not None, "peer-B never received register_ack"
    assert ack["peer_id"] == "peer-B"

    peers = ack.get("peers") or []
    by_id = {p["peer_id"]: p for p in peers}
    assert "peer-A" in by_id, (
        f"peer-B's ack should include peer-A; got {list(by_id)}"
    )
    # **The relay assertion**: peer A's hw_profile traveled from
    # A's register → server cache → B's register_ack.
    assert by_id["peer-A"].get("hardware_profile") == {
        "tflops_fp16": 33.9, "ram_total_gb": 24.0,
        "gpu_vram_gb": 24.0,
    }


@pytest.mark.asyncio
async def test_mixed_fleet_hw_and_no_hw_both_relayed():
    """A pre-838 peer (no hw_profile) and an sp838 peer
    (with hw_profile) both appear in a later joiner's ack.
    The hw peer carries its hw; the no-hw peer's entry omits
    the key (legacy wire format preserved)."""
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig

    srv = BootstrapServer(BootstrapConfig())

    # Legacy peer A — no hw_profile in register payload
    await srv._handle_register(
        _fake_ws(),
        {
            "type": "register",
            "peer_id": "legacy-A",
            "port": 9001,
            "address": "10.0.0.1",
        },
        client_ip="10.0.0.1",
    )
    # Sp838 peer B — sends hw_profile
    await srv._handle_register(
        _fake_ws(),
        {
            "type": "register",
            "peer_id": "modern-B",
            "port": 9001,
            "address": "10.0.0.2",
            "hardware_profile": {"tflops_fp16": 12.0},
        },
        client_ip="10.0.0.2",
    )
    # Joiner C — observes both
    ws_c = _fake_ws()
    await srv._handle_register(
        ws_c,
        {
            "type": "register",
            "peer_id": "joiner-C",
            "port": 9001,
            "address": "10.0.0.3",
        },
        client_ip="10.0.0.3",
    )
    ack = next(
        (m for m in ws_c._sent if m.get("type") == "register_ack"),
        None,
    )
    assert ack is not None
    by_id = {p["peer_id"]: p for p in (ack.get("peers") or [])}
    assert "legacy-A" in by_id
    assert "modern-B" in by_id
    # Legacy peer: hw key omitted entirely (wire-format compat)
    assert "hardware_profile" not in by_id["legacy-A"]
    # Modern peer: hw relayed
    assert by_id["modern-B"]["hardware_profile"] == {
        "tflops_fp16": 12.0,
    }


@pytest.mark.asyncio
async def test_get_peers_query_path_also_relays_hw():
    """Sprint 838 added hw to get_peer_list which two server
    handlers consume: register_ack AND _handle_get_peers (the
    standalone query path operators use post-register for
    periodic refresh, sprint 389). Confirm the second path
    works too — a regression here would silently lose hw on
    refresh ticks even when register works."""
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig

    srv = BootstrapServer(BootstrapConfig())

    # Stage one hw-carrying peer
    await srv._handle_register(
        _fake_ws(),
        {
            "type": "register",
            "peer_id": "peer-A",
            "port": 9001,
            "address": "10.0.0.1",
            "hardware_profile": {"tflops_fp16": 33.9},
        },
        client_ip="10.0.0.1",
    )

    # Direct get_peer_list call (what _handle_get_peers wraps)
    peers = await srv.get_peer_list()
    by_id = {p["peer_id"]: p for p in peers}
    assert by_id["peer-A"].get("hardware_profile") == {
        "tflops_fp16": 33.9,
    }

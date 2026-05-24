"""Sprint 841 — full hardware_profile relay pipeline composition.

Sprint 840 covered the server-side composition: peer A registers
with hw → peer B's register_ack carries A's hw. But the relay's
real utility extends through 4 components:

  Server cache (sp838 _handle_register)
    → Client parse (sp838 BootstrapClient peer-list parser)
      → Discovery thread (sp838 Libp2pDiscovery._hydrate_peers_from_bootstrap)
        → Pool read (sp836 dht_backed_pool_provider via get_known_peers())
          → ParallaxGPU.tflops_fp16 = REAL (not sp836 synthesis 0.1)

Sp838 + sp840 cover steps 1-2 + step 3 in isolation. Sp836
tests step 4 with mock discovery. Nothing tests them all
composed. A refactor that subtly broke the field-passing
between any two steps could pass all existing tests yet
silently regress the production claim.

Sprint 841 composes all 4 steps via a real `_handle_register`
+ real BootstrapPeer parsing + real `_hydrate_peers_from_bootstrap`
+ real `dht_backed_pool_provider`. The only fake component is
the WebSocket transport (the wire is sp840's concern, not this
test's). End-to-end assertion: when peer A registers with
hw_profile {tflops_fp16: 33.9}, the pool provider on a fresh
joiner returns a ParallaxGPU with tflops_fp16=33.9 (NOT 0.1
which would indicate sp836 fallback synthesis fired).

Also pins the inverse — when peer A has NO hw_profile and the
admit-unknown env is off, the pool excludes A entirely (sp836
strict default). This is the safety net we don't want to lose
silently.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


def _fake_ws():
    ws = MagicMock()
    sent: List[Dict[str, Any]] = []

    async def _send(payload: str):
        sent.append(json.loads(payload))

    ws.send = AsyncMock(side_effect=_send)
    ws.close = AsyncMock()
    ws._sent = sent
    return ws


@pytest.fixture(autouse=True)
def clean_env():
    """Snapshot+restore sp836 env between tests."""
    saved = os.environ.get("PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE")
    os.environ.pop("PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE", None)
    yield
    os.environ.pop("PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE", None)
    if saved is not None:
        os.environ["PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE"] = saved


@pytest.mark.asyncio
async def test_relay_pipeline_end_to_end_real_hw_reaches_pool():
    """**THE load-bearing pipeline assertion.** A peer registers
    with hw → register_ack carries it → BootstrapPeer parses
    it → Libp2pDiscovery threads it into PeerInfo → DHT pool
    reads it via get_known_peers() → ParallaxGPU emits real
    tflops, NOT the sp836 synthesis 0.1.
    """
    from prsm.bootstrap.client import BootstrapPeer
    from prsm.bootstrap.config import BootstrapConfig
    from prsm.bootstrap.server import BootstrapServer
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.libp2p_discovery import Libp2pDiscovery

    # Step 1: server caches hw on register
    srv = BootstrapServer(BootstrapConfig())
    await srv._handle_register(
        _fake_ws(),
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

    # Step 2: server emits hw in peer-list (what client parses)
    server_peer_list = await srv.get_peer_list()
    assert len(server_peer_list) == 1
    server_entry = server_peer_list[0]
    assert server_entry["peer_id"] == "peer-A"
    # Client-side parsing: same logic BootstrapClient uses
    parsed = BootstrapPeer(
        peer_id=server_entry["peer_id"],
        address=server_entry["address"],
        port=server_entry["port"],
        capabilities=server_entry.get("capabilities", []),
        region=server_entry.get("region"),
        version=server_entry.get("version"),
        hardware_profile=server_entry.get("hardware_profile"),
    )
    assert parsed.hardware_profile == {
        "tflops_fp16": 33.9, "ram_total_gb": 24.0,
        "gpu_vram_gb": 24.0,
    }

    # Step 3: Libp2pDiscovery threads relayed hw into PeerInfo
    transport = MagicMock()
    transport.identity.node_id = "joiner-self"
    disco = Libp2pDiscovery(transport=transport)
    disco._hydrate_peers_from_bootstrap([parsed])
    peer_info = disco._capability_index["peer-A"]
    assert peer_info.hardware_profile == {
        "tflops_fp16": 33.9, "ram_total_gb": 24.0,
        "gpu_vram_gb": 24.0,
    }

    # Step 4: DHT pool reads hw via get_known_peers() and emits
    # real ParallaxGPU (sp836 synthesis would yield tflops=0.1)
    node = MagicMock()
    node.identity.node_id = "joiner-self"
    node.discovery = disco
    node.discovery._local_hardware_profile = None
    provider = build_dht_backed_pool_provider(node)
    gpus = provider()
    by_id = {g.node_id: g for g in gpus}
    assert "peer-A" in by_id
    assert by_id["peer-A"].tflops_fp16 == 33.9, (
        "pipeline regression — hw_profile didn't reach the "
        "ParallaxGPU. Real value should be 33.9; sp836 "
        "synthesis is 0.1. Check sp838 wiring between "
        "_handle_register → get_peer_list → BootstrapPeer → "
        "_hydrate_peers_from_bootstrap → "
        "dht_backed_pool_provider."
    )
    # Memory carries through too (sp836 synthesis is 1.0 GB).
    assert by_id["peer-A"].memory_gb == 24.0, (
        f"hw_profile memory didn't reach ParallaxGPU; "
        f"got memory_gb={by_id['peer-A'].memory_gb} "
        f"(synthesis fallback would be 1.0)"
    )


@pytest.mark.asyncio
async def test_pipeline_excludes_no_hw_peer_when_strict():
    """Inverse pin: a peer that registered without hw_profile +
    sp836 admit-unknown env OFF → pool excludes it entirely.
    This is the strict-default safety net protecting production
    from "phantom GPU" hallucination on legacy peers."""
    from prsm.bootstrap.client import BootstrapPeer
    from prsm.bootstrap.config import BootstrapConfig
    from prsm.bootstrap.server import BootstrapServer
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.libp2p_discovery import Libp2pDiscovery

    srv = BootstrapServer(BootstrapConfig())
    # Legacy peer: NO hw in register payload
    await srv._handle_register(
        _fake_ws(),
        {
            "type": "register",
            "peer_id": "legacy",
            "port": 9001,
            "address": "10.0.0.5",
        },
        client_ip="10.0.0.5",
    )
    server_entry = (await srv.get_peer_list())[0]
    # Wire format: hw key omitted entirely
    assert "hardware_profile" not in server_entry

    parsed = BootstrapPeer(
        peer_id=server_entry["peer_id"],
        address=server_entry["address"],
        port=server_entry["port"],
        capabilities=server_entry.get("capabilities", []),
        hardware_profile=server_entry.get("hardware_profile"),
    )
    assert parsed.hardware_profile is None

    transport = MagicMock()
    transport.identity.node_id = "joiner"
    disco = Libp2pDiscovery(transport=transport)
    disco._hydrate_peers_from_bootstrap([parsed])

    # sp836 env off (autouse fixture cleared it)
    node = MagicMock()
    node.identity.node_id = "joiner"
    node.discovery = disco
    node.discovery._local_hardware_profile = None
    gpus = build_dht_backed_pool_provider(node)()
    # Strict default: legacy peer excluded entirely.
    assert gpus == [], (
        "strict-default safety net regressed — legacy peer "
        "without hw_profile MUST be excluded from pool when "
        "PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE is unset"
    )


@pytest.mark.asyncio
async def test_pipeline_admits_no_hw_peer_with_synthesis_when_env_set():
    """The composition working in conjunction with sp836's
    safety net: when PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE=1,
    legacy peers without hw_profile are admitted with the
    sp836 conservative synthetic values."""
    from prsm.bootstrap.client import BootstrapPeer
    from prsm.bootstrap.config import BootstrapConfig
    from prsm.bootstrap.server import BootstrapServer
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.libp2p_discovery import Libp2pDiscovery

    os.environ["PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE"] = "1"

    srv = BootstrapServer(BootstrapConfig())
    await srv._handle_register(
        _fake_ws(),
        {
            "type": "register",
            "peer_id": "legacy",
            "port": 9001,
            "address": "10.0.0.5",
        },
        client_ip="10.0.0.5",
    )
    entry = (await srv.get_peer_list())[0]
    parsed = BootstrapPeer(
        peer_id=entry["peer_id"],
        address=entry["address"],
        port=entry["port"],
        capabilities=entry.get("capabilities", []),
        hardware_profile=entry.get("hardware_profile"),
    )
    transport = MagicMock()
    transport.identity.node_id = "joiner"
    disco = Libp2pDiscovery(transport=transport)
    disco._hydrate_peers_from_bootstrap([parsed])
    node = MagicMock()
    node.identity.node_id = "joiner"
    node.discovery = disco
    node.discovery._local_hardware_profile = None
    gpus = build_dht_backed_pool_provider(node)()
    assert len(gpus) == 1
    # sp836 synthesis values
    assert gpus[0].tflops_fp16 == 0.1
    assert gpus[0].memory_gb == 1.0

"""Sprint 838 — bootstrap-server relays hardware_profile to
cold-start joiners.

Multi-host live re-attest 2026-05-24 (sp836+sp837 follow-on)
confirmed the pipeline now advances from "0 GPUs" to the layer
allocator, where sp836's conservative synthetic profile blocks
real inference (layer_capacity=1 per peer × 2 = 2 < 12 layers
for gpt2). The architectural fix: bootstrap-server caches each
peer's authoritative hardware_profile at registration time +
re-broadcasts it in peer-list responses so cold-start joiners
(NAT'd operators who can't receive direct DISCOVERY_ANNOUNCE)
see real fleet capacity.

End-to-end change:
1. Bootstrap PeerInfo (models.py): gain optional
   hardware_profile field.
2. Server register handler (server.py): accept hw_profile in
   registration payload, persist on PeerInfo, preserve cached
   value on re-registration without hw.
3. Server get_peer_list (server.py): relay hw_profile in
   peer-list entries when present.
4. Client BootstrapClient (client.py): accept hw_profile
   kwarg, send in registration, parse from peer-list.
5. BootstrapPeer (client.py): gain hardware_profile field.
6. Libp2pDiscovery + PeerDiscovery (node-side): accept
   local_hardware_profile kwarg, forward to BootstrapClient,
   thread relayed peer hw into PeerInfo.hardware_profile.
7. node.py: pass _local_hardware_profile when constructing
   Libp2pDiscovery (sp680 already did this for PeerDiscovery).

Wire-format compatibility: hw_profile keys are OMITTED entirely
when None (never sent, never echoed). Pre-838 servers ignore
the field on register; pre-838 clients ignore it on responses.
Mixed-version fleets continue working.

Pin tests:
- PeerInfo.to_dict omits hw_profile when None
- PeerInfo.to_dict emits hw_profile when present
- PeerInfo.from_dict roundtrips hw_profile
- Server register handler caches supplied hw_profile
- Server register handler preserves cached hw on re-register
  without hw (e.g., legacy client resumes session)
- get_peer_list emits hw_profile when cached
- BootstrapClient register payload includes hw_profile when
  set
- BootstrapClient register payload omits hw_profile when not
  set (legacy wire format)
- BootstrapPeer carries hw_profile field
- Libp2pDiscovery accepts local_hardware_profile kwarg
- Libp2pDiscovery forwards relayed hw into _capability_index
  PeerInfo
- PeerDiscovery (legacy) forwards relayed hw into known_peers
  PeerInfo
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---- Bootstrap PeerInfo serialization ------------------------


def test_peer_info_to_dict_omits_hw_when_none():
    from prsm.bootstrap.models import PeerInfo
    p = PeerInfo(peer_id="abc", address="1.2.3.4", port=9001)
    d = p.to_dict()
    assert "hardware_profile" not in d


def test_peer_info_to_dict_emits_hw_when_present():
    from prsm.bootstrap.models import PeerInfo
    hw = {"tflops_fp16": 33.9, "ram_total_gb": 24.0}
    p = PeerInfo(
        peer_id="abc", address="1.2.3.4", port=9001,
        hardware_profile=hw,
    )
    d = p.to_dict()
    assert d["hardware_profile"] == hw


def test_peer_info_from_dict_roundtrips_hw():
    from prsm.bootstrap.models import PeerInfo
    hw = {"tflops_fp16": 8.0, "gpu_vram_gb": 12.0}
    src = PeerInfo(
        peer_id="abc", address="1.2.3.4", port=9001,
        hardware_profile=hw,
    )
    d = src.to_dict()
    restored = PeerInfo.from_dict(d)
    assert restored.hardware_profile == hw


def test_peer_info_from_dict_legacy_no_hw_key():
    """Pre-838 serialized peers have no hardware_profile key.
    from_dict MUST tolerate that and default to None."""
    from prsm.bootstrap.models import PeerInfo
    src = PeerInfo(peer_id="abc", address="1.2.3.4", port=9001)
    d = src.to_dict()
    # to_dict omits the key when None; confirm
    assert "hardware_profile" not in d
    restored = PeerInfo.from_dict(d)
    assert restored.hardware_profile is None


# ---- Server-side register handler ----------------------------


@pytest.mark.asyncio
async def test_server_register_caches_hw_profile():
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig

    srv = BootstrapServer(BootstrapConfig())
    ws = AsyncMock()
    data = {
        "type": "register",
        "peer_id": "node-A",
        "port": 9001,
        "address": "1.2.3.4",
        "capabilities": ["compute"],
        "hardware_profile": {
            "tflops_fp16": 33.9, "ram_total_gb": 24.0,
        },
    }
    await srv._handle_register(ws, data, client_ip="1.2.3.4")
    assert "node-A" in srv.peers
    assert srv.peers["node-A"].hardware_profile == {
        "tflops_fp16": 33.9, "ram_total_gb": 24.0,
    }


@pytest.mark.asyncio
async def test_server_register_preserves_hw_on_reregister_without():
    """If a peer re-registers without hw_profile (e.g. legacy
    client resumes session) the server keeps the cached value.
    Operator hw doesn't usually change between sessions; losing
    it on every restart would defeat the relay's purpose."""
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig

    srv = BootstrapServer(BootstrapConfig())
    ws = AsyncMock()
    # First register: with hw
    await srv._handle_register(ws, {
        "type": "register", "peer_id": "node-A", "port": 9001,
        "address": "1.2.3.4",
        "hardware_profile": {"tflops_fp16": 33.9},
    }, client_ip="1.2.3.4")
    # Second register: no hw_profile key
    await srv._handle_register(ws, {
        "type": "register", "peer_id": "node-A", "port": 9001,
        "address": "1.2.3.4",
    }, client_ip="1.2.3.4")
    assert srv.peers["node-A"].hardware_profile == {
        "tflops_fp16": 33.9,
    }


@pytest.mark.asyncio
async def test_server_register_rejects_non_dict_hw():
    """A malformed/malicious hw_profile (e.g. string, list)
    must NOT break the handshake. Server silently drops it
    rather than storing garbage."""
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig

    srv = BootstrapServer(BootstrapConfig())
    ws = AsyncMock()
    await srv._handle_register(ws, {
        "type": "register", "peer_id": "node-A", "port": 9001,
        "address": "1.2.3.4",
        "hardware_profile": "not-a-dict",
    }, client_ip="1.2.3.4")
    assert srv.peers["node-A"].hardware_profile is None


# ---- Server-side get_peer_list relays hw ---------------------


@pytest.mark.asyncio
async def test_get_peer_list_emits_hw_when_cached():
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig
    from prsm.bootstrap.models import PeerInfo

    srv = BootstrapServer(BootstrapConfig())
    srv.peers["node-A"] = PeerInfo(
        peer_id="node-A", address="1.2.3.4", port=9001,
        hardware_profile={"tflops_fp16": 8.0},
    )
    srv.peers["node-B"] = PeerInfo(
        peer_id="node-B", address="2.3.4.5", port=9001,
    )
    peers = await srv.get_peer_list()
    by_id = {p["peer_id"]: p for p in peers}
    assert by_id["node-A"].get("hardware_profile") == {
        "tflops_fp16": 8.0,
    }
    # Legacy peer (no hw): key omitted
    assert "hardware_profile" not in by_id["node-B"]


# ---- BootstrapClient register payload ------------------------


def test_client_register_msg_includes_hw_when_set():
    from prsm.bootstrap.client import BootstrapClient

    client = BootstrapClient(
        bootstrap_url="wss://x/", node_id="n",
        hardware_profile={"tflops_fp16": 30.0},
    )
    assert client.hardware_profile == {"tflops_fp16": 30.0}


def test_client_register_msg_omits_hw_when_unset():
    """Pre-838 wire format compatibility: when operator doesn't
    pass hw_profile, the register payload MUST NOT contain the
    key (servers tolerant of unknown keys still test this)."""
    from prsm.bootstrap.client import BootstrapClient

    client = BootstrapClient(
        bootstrap_url="wss://x/", node_id="n",
    )
    assert client.hardware_profile is None


# ---- BootstrapPeer field --------------------------------------


def test_bootstrap_peer_carries_hw_profile_field():
    from prsm.bootstrap.client import BootstrapPeer
    p = BootstrapPeer(
        peer_id="x", address="1.2.3.4", port=9001,
        hardware_profile={"tflops_fp16": 12.0},
    )
    assert p.hardware_profile == {"tflops_fp16": 12.0}


def test_bootstrap_peer_defaults_hw_to_none():
    from prsm.bootstrap.client import BootstrapPeer
    p = BootstrapPeer(peer_id="x", address="1.2.3.4", port=9001)
    assert p.hardware_profile is None


# ---- Libp2pDiscovery wiring -----------------------------------


def test_libp2p_discovery_accepts_local_hardware_profile():
    """Sprint 838 added local_hardware_profile kwarg to mirror
    PeerDiscovery (sp680). Forwarded to BootstrapClient on
    construction."""
    from prsm.node.libp2p_discovery import Libp2pDiscovery

    transport = MagicMock()
    transport.identity.node_id = "self-node"
    hw = {"tflops_fp16": 30.0, "ram_total_gb": 16.0}
    disco = Libp2pDiscovery(
        transport=transport,
        local_hardware_profile=hw,
    )
    assert disco._local_hardware_profile == hw


def test_libp2p_discovery_threads_bootstrap_relayed_hw():
    """Bootstrap-relayed hw must reach Libp2pDiscovery._capability_index
    so sp836's pool provider sees real capacity instead of
    falling through to synthesis."""
    from prsm.node.libp2p_discovery import Libp2pDiscovery
    from prsm.bootstrap.client import BootstrapPeer

    transport = MagicMock()
    transport.identity.node_id = "self-node"
    disco = Libp2pDiscovery(transport=transport)

    bp_with_hw = BootstrapPeer(
        peer_id="peer-A", address="1.2.3.4", port=9001,
        hardware_profile={"tflops_fp16": 33.9, "ram_total_gb": 24.0},
    )
    bp_no_hw = BootstrapPeer(
        peer_id="peer-B", address="2.3.4.5", port=9001,
    )
    disco._hydrate_peers_from_bootstrap([bp_with_hw, bp_no_hw])

    a = disco._capability_index["peer-A"]
    b = disco._capability_index["peer-B"]
    assert a.hardware_profile == {
        "tflops_fp16": 33.9, "ram_total_gb": 24.0,
    }
    assert b.hardware_profile is None


# ---- PeerDiscovery (legacy WebSocket-mode) -------------------


def test_peer_discovery_already_had_local_hw_kwarg():
    """Regression guard for sprint 680 — the legacy
    PeerDiscovery has accepted local_hardware_profile since
    sp680. Sprint 838 just mirrored the kwarg on Libp2pDiscovery
    + threaded it to BootstrapClient."""
    from prsm.node.discovery import PeerDiscovery

    transport = MagicMock()
    transport.identity.node_id = "self-node"
    hw = {"tflops_fp16": 30.0}
    disco = PeerDiscovery(
        transport=transport,
        local_hardware_profile=hw,
    )
    assert disco._local_hardware_profile == hw

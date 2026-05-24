"""Sprint 836 — F31 fix: DHT-backed pool reads via get_known_peers()
and admits unknown-hardware peers under conservative defaults.

Multi-host re-attest 2026-05-24 surfaced F31: a fresh Mac
operator joining the live mainnet fleet (bootstrap-us +
SFO + bootstrap-us-operator) saw 2 known peers via /peers but
the DHT-backed pool reported gpu_count=0. Chain inference
dispatch returned "GPU pool is empty" despite the fleet being
provably alive.

Two composed problems diagnosed:

1. API contract mismatch: PRSM has two discovery
   implementations — Libp2pDiscovery (libp2p transport,
   default) and PeerDiscovery (WebSocket transport, legacy
   fallback). Only PeerDiscovery exposes a `known_peers` dict
   attribute. Sp682's dht_backed_pool_provider read
   `discovery.known_peers` which silently returned {} on
   Libp2pDiscovery → pool always saw 0 peers under libp2p.
   Both implementations expose `get_known_peers()` method;
   reading via that works on both.

2. Cold-start hardware_profile gap: bootstrap-server doesn't
   propagate hardware_profile in its peer-list responses.
   Peers only advertise hw via direct DISCOVERY_ANNOUNCE,
   which never reaches a NAT'd inbound-blocked joiner.
   Even with #1 fixed, peers had hardware_profile=None →
   _hw_dict_to_parallax_gpu returned None → pool empty.

Sprint 836 fix:
- Switch peer lookup from `.known_peers` attribute to
  `.get_known_peers()` method (works on both backends)
- Add PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE env-gated opt-in
  that synthesizes a conservative {tflops_fp16: 0.1,
  ram_total_gb: 1.0, gpu_vram_gb: 0.0,
  memory_bandwidth_gbps: 25.0} profile for hw=None peers.
- Production default: env unset = strict (legacy behavior;
  real hardware required for staking/tier policy)
- Dogfood + multi-host tests: env=1 admits cold-start peers
  under conservative defaults

Live-attested 2026-05-24 against the live mainnet fleet:
  Mac with PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE=1
  /admin/parallax/pool/snapshot → gpu_count=2
  Both SFO + bootstrap-us-operator admitted with
  tier_attestation=tier-none, conservative hw defaults.
Pre-836: same fleet, same env minus the flag → gpu_count=0.

Pin tests:
- Provider uses get_known_peers() not .known_peers attr
- Libp2pDiscovery (which has NO known_peers attribute) returns
  peers via the method
- hardware_profile=None peers excluded by default (strict)
- hardware_profile=None peers admitted with synthetic profile
  when PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE=1
- Real hardware_profile takes precedence over synthesis
- Env var registered in sp696 parallax-readiness CLI
"""
from __future__ import annotations

import inspect
import os
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env():
    """Snapshot+restore the admit env so tests that set it
    don't leak to subsequent tests in any file."""
    saved = os.environ.get("PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE")
    os.environ.pop("PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE", None)
    yield
    # Always drop whatever the test left, then restore the
    # pre-test value if there was one.
    os.environ.pop("PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE", None)
    if saved is not None:
        os.environ["PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE"] = saved


def _peer_info(node_id: str, hw=None):
    """Minimal PeerInfo-shaped object."""
    p = MagicMock()
    p.node_id = node_id
    p.hardware_profile = hw
    return p


def _make_node(self_id="self_node", peer_infos=None):
    """Mock node where discovery exposes get_known_peers()."""
    node = MagicMock()
    node.identity.node_id = self_id
    node.discovery = MagicMock()
    node.discovery._local_hardware_profile = None
    # No .known_peers attribute — mimics Libp2pDiscovery.
    if hasattr(node.discovery, "known_peers"):
        del node.discovery.known_peers
    node.discovery.get_known_peers.return_value = peer_infos or []
    return node


# ---- API contract: reads via get_known_peers() ---------------


def test_provider_uses_get_known_peers_method():
    """Sprint 836 part 1: pool provider MUST call
    get_known_peers() — both Libp2pDiscovery and PeerDiscovery
    expose it. The legacy .known_peers attribute path is
    Libp2p-incompatible."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    node = _make_node()
    provider = build_dht_backed_pool_provider(node)
    provider()
    node.discovery.get_known_peers.assert_called()


def test_provider_works_when_known_peers_attr_missing():
    """Regression guard: a discovery that has get_known_peers()
    but NO `known_peers` attribute (Libp2pDiscovery) must NOT
    crash. Pre-836 the getattr default {} silently returned
    no peers; we need a meaningful result."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    peers = [
        _peer_info(
            "peer-a",
            hw={"tflops_fp16": 30.0, "ram_total_gb": 16.0},
        ),
    ]
    node = _make_node(peer_infos=peers)
    # Confirm setup matches Libp2pDiscovery shape
    assert not hasattr(node.discovery, "known_peers")
    provider = build_dht_backed_pool_provider(node)
    result = provider()
    assert len(result) == 1
    assert result[0].node_id == "peer-a"


# ---- hardware_profile=None handling --------------------------


def test_hw_none_excluded_by_default_strict():
    """Strict default: peers with hardware_profile=None are
    silently excluded (legacy sp682 behavior preserved)."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    peers = [_peer_info("peer-a", hw=None)]
    node = _make_node(peer_infos=peers)
    # No env set
    provider = build_dht_backed_pool_provider(node)
    result = provider()
    assert result == []


def test_hw_none_admitted_when_env_set():
    """When PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE=1, peers with
    hardware_profile=None get a synthetic conservative
    profile and are admitted to the pool."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    os.environ["PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE"] = "1"
    peers = [_peer_info("peer-a", hw=None)]
    node = _make_node(peer_infos=peers)
    provider = build_dht_backed_pool_provider(node)
    result = provider()
    assert len(result) == 1
    assert result[0].node_id == "peer-a"
    # Conservative synthetic defaults
    assert result[0].tflops_fp16 == 0.1
    assert result[0].memory_gb == 1.0


@pytest.mark.parametrize("flag_value", ["1", "true", "yes", "TRUE", "Yes"])
def test_admit_env_accepts_truthy_variations(flag_value):
    """The env-gate matches the broader project convention
    (1/true/yes case-insensitive)."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    os.environ["PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE"] = flag_value
    peers = [_peer_info("peer-a", hw=None)]
    node = _make_node(peer_infos=peers)
    provider = build_dht_backed_pool_provider(node)
    assert len(provider()) == 1


@pytest.mark.parametrize("flag_value", ["0", "false", "no", "", "off"])
def test_admit_env_rejects_falsy_variations(flag_value):
    """Falsy values keep the strict default."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    os.environ["PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE"] = flag_value
    peers = [_peer_info("peer-a", hw=None)]
    node = _make_node(peer_infos=peers)
    provider = build_dht_backed_pool_provider(node)
    assert provider() == []


def test_real_hardware_takes_precedence():
    """When a peer DOES advertise hardware_profile, the real
    values are used — synthesis only fires on None."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    os.environ["PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE"] = "1"
    peers = [
        _peer_info(
            "real-peer",
            hw={"tflops_fp16": 33.9, "ram_total_gb": 24.0,
                "gpu_vram_gb": 24.0},
        ),
    ]
    node = _make_node(peer_infos=peers)
    provider = build_dht_backed_pool_provider(node)
    result = provider()
    assert len(result) == 1
    # Real hardware values — NOT synthesis (0.1, 1.0)
    assert result[0].tflops_fp16 == 33.9
    assert result[0].memory_gb == 24.0


def test_mix_of_real_and_unknown_when_env_set():
    """A pool with some peers advertising hw + some not: both
    admitted under env-on. Real hw uses real values; unknown
    hw uses synthesis."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    os.environ["PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE"] = "1"
    peers = [
        _peer_info(
            "real",
            hw={"tflops_fp16": 33.9, "ram_total_gb": 24.0},
        ),
        _peer_info("unknown", hw=None),
    ]
    node = _make_node(peer_infos=peers)
    provider = build_dht_backed_pool_provider(node)
    result = provider()
    assert len(result) == 2
    by_id = {g.node_id: g for g in result}
    assert by_id["real"].tflops_fp16 == 33.9
    assert by_id["unknown"].tflops_fp16 == 0.1


# ---- get_known_peers raise is fail-soft ----------------------


def test_get_known_peers_raising_returns_empty():
    """A discovery raising from get_known_peers() must not
    crash the provider — return empty pool (own node only)."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    node = _make_node()
    node.discovery.get_known_peers.side_effect = RuntimeError("boom")
    provider = build_dht_backed_pool_provider(node)
    # Should not raise
    result = provider()
    assert result == []


# ---- Env var registered in parallax-readiness ----------------


def test_admit_unknown_hardware_env_in_readiness_registry():
    """The sp696 readiness CLI must surface the new env so
    operators can see it via `prsm node parallax-readiness`."""
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    names = [t[0] for t in _PARALLAX_ENV_REGISTRY]
    assert "PRSM_PARALLAX_ADMIT_UNKNOWN_HARDWARE" in names

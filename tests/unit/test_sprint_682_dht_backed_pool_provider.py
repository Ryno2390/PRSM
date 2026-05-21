"""Sprint 682 — DHT-backed GpuPoolProvider.

Translates sprint 680's hardware_profile-bearing peers (plus self)
into the `Sequence[ParallaxGPU]` the ParallaxScheduledExecutor
needs. Replaces the sprint-558 static-empty pool when the operator
opts in via `PRSM_PARALLAX_GPU_POOL_KIND=dht-backed`.

Conservative defaults fill ParallaxGPU fields not present in
HardwareProfile (memory_bandwidth_gbps, layer_capacity, region).
Sprint 683 will layer on-chain stake reads on top of the
stake_amount=0 placeholder.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_provider_returns_callable():
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    node = MagicMock()
    node.identity.node_id = "selfid"
    node.discovery = MagicMock()
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {}
    provider = build_dht_backed_pool_provider(node)
    assert callable(provider)


def test_provider_includes_self_when_local_profile_present():
    """Self's own hardware shows up as a ParallaxGPU — critical
    so a single-node operator can run inference at all."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = {
        "tflops_fp16": 4.6,
        "gpu_vram_gb": 0.0,
        "ram_total_gb": 16.0,
        "gpu_name": "Apple M4",
        "gpu_api": "metal",
    }
    node.discovery.known_peers = {}
    provider = build_dht_backed_pool_provider(node)
    gpus = list(provider())
    assert len(gpus) == 1
    assert gpus[0].node_id == "a" * 32
    assert gpus[0].tflops_fp16 == 4.6
    assert gpus[0].memory_gb == 16.0  # falls back to ram_total_gb


def test_provider_excludes_self_when_local_profile_missing():
    """No local profile → don't fabricate one for self either.
    Peer simply doesn't participate as a stage host."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {}
    provider = build_dht_backed_pool_provider(node)
    assert list(provider()) == []


def test_provider_includes_advertising_peers():
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {
        "b" * 32: PeerInfo(
            node_id="b" * 32,
            address="1.2.3.4:9001",
            hardware_profile={
                "tflops_fp16": 9.5,
                "gpu_vram_gb": 24.0,
                "ram_total_gb": 64.0,
                "gpu_name": "RTX 4090",
                "gpu_api": "cuda",
            },
        ),
    }
    provider = build_dht_backed_pool_provider(node)
    gpus = list(provider())
    assert len(gpus) == 1
    assert gpus[0].node_id == "b" * 32
    assert gpus[0].tflops_fp16 == 9.5
    # gpu_vram_gb takes precedence over ram_total_gb
    assert gpus[0].memory_gb == 24.0
    assert gpus[0].gpu_name == "RTX 4090"


def test_provider_excludes_legacy_peers_without_hardware_profile():
    """Pre-680 peers (hardware_profile=None) → silently excluded
    from the pool, never throw."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {
        "legacy": PeerInfo(node_id="legacy", address="1.2.3.4:9001"),
    }
    provider = build_dht_backed_pool_provider(node)
    assert list(provider()) == []


def test_provider_excludes_peers_with_zero_tflops():
    """A peer whose advertised tflops_fp16=0 + tflops_fp32=0 can't
    construct a valid ParallaxGPU — exclude rather than crash."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {
        "zero": PeerInfo(
            node_id="zero",
            address="1.2.3.4:9001",
            hardware_profile={
                "tflops_fp16": 0.0,
                "tflops_fp32": 0.0,
                "ram_total_gb": 16.0,
            },
        ),
    }
    provider = build_dht_backed_pool_provider(node)
    assert list(provider()) == []


def test_provider_falls_back_to_tflops_fp32_when_fp16_missing():
    """Some profilers only emit tflops_fp32. Derive fp16 ≈ fp32 * 2
    (standard hardware ratio) when fp16 is missing/zero."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {
        "fp32only": PeerInfo(
            node_id="fp32only",
            address="1.2.3.4:9001",
            hardware_profile={
                "tflops_fp32": 4.0,
                "ram_total_gb": 16.0,
            },
        ),
    }
    provider = build_dht_backed_pool_provider(node)
    gpus = list(provider())
    assert len(gpus) == 1
    assert gpus[0].tflops_fp16 == pytest.approx(8.0)


def test_provider_uses_region_env_var(monkeypatch):
    """PRSM_PARALLAX_REGION overrides the default region. Critical
    so operators in different DC regions get their pools
    partitioned per Parallax §3.2."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    monkeypatch.setenv("PRSM_PARALLAX_REGION", "us-east-1")
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = {
        "tflops_fp16": 4.6, "ram_total_gb": 16.0,
    }
    node.discovery.known_peers = {}
    provider = build_dht_backed_pool_provider(node)
    gpus = list(provider())
    assert gpus[0].region == "us-east-1"


def test_provider_defaults_stake_to_zero():
    """Sprint 682 placeholder: stake_amount=0 for everyone. Sprint
    683 will read the StakeBond contract per peer. Pin this so a
    future refactor doesn't silently flip the default to a
    spoofable peer-claim."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = {
        "tflops_fp16": 4.6, "ram_total_gb": 16.0,
    }
    node.discovery.known_peers = {}
    provider = build_dht_backed_pool_provider(node)
    gpus = list(provider())
    assert gpus[0].stake_amount == 0


def test_provider_defaults_tier_attestation_to_none():
    """No TEE-attestation in sprint 682 → all entries marked
    TIER_ATTESTATION_NONE. Adapter B (per-request TEE gate) is
    where real attestations get enforced; the pool itself doesn't
    discriminate."""
    from prsm.compute.parallax_scheduling.prsm_types import (
        TIER_ATTESTATION_NONE,
    )
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = {
        "tflops_fp16": 4.6, "ram_total_gb": 16.0,
    }
    node.discovery.known_peers = {}
    provider = build_dht_backed_pool_provider(node)
    gpus = list(provider())
    assert gpus[0].tier_attestation == TIER_ATTESTATION_NONE


def test_provider_skips_peer_with_invalid_hardware_dict():
    """Peer advertises a hardware_profile that's not a dict (could
    happen if a malicious or buggy peer crafts a list) → skip,
    never raise."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {
        "bad": PeerInfo(
            node_id="bad", address="x:1",
            hardware_profile=["not", "a", "dict"],
        ),
    }
    provider = build_dht_backed_pool_provider(node)
    assert list(provider()) == []


def test_pool_kind_dht_backed_recognized():
    """The inference_wiring._KNOWN_GPU_POOL_KINDS tuple must
    include 'dht-backed' so operators can opt in via env."""
    from prsm.node.inference_wiring import _KNOWN_GPU_POOL_KINDS
    assert "dht-backed" in _KNOWN_GPU_POOL_KINDS

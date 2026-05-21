"""Sprint 695 — PRSM_PARALLAX_MEMORY_GB_OVERRIDE forces multi-stage.

Sprint 688's deferred goal was multi-stage allocation across
NYC+SFO through the real ParallaxScheduledExecutor (sprints
677-679 used the chain-exec-ping bypass). Live-attest showed
the upstream Parallax allocator chose 1 stage even with both
peers visible — gpt2 fits on either 1.92GB droplet via the
water-filling memory estimate.

Sprint 695 adds operator override so the advertised memory_gb
can be artificially constrained, forcing the allocator to split
layers across peers. Pure-additive: env unset preserves current
behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_memory_override_pins_advertised_value(monkeypatch):
    """PRSM_PARALLAX_MEMORY_GB_OVERRIDE=0.3 → every ParallaxGPU
    reports memory_gb=0.3 regardless of hardware_profile."""
    monkeypatch.setenv("PRSM_PARALLAX_MEMORY_GB_OVERRIDE", "0.3")
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
    assert len(gpus) == 1
    assert gpus[0].memory_gb == 0.3


def test_memory_override_falls_back_on_garbage(monkeypatch):
    """Non-float override → keeps the heuristic value, doesn't crash."""
    monkeypatch.setenv("PRSM_PARALLAX_MEMORY_GB_OVERRIDE", "garbage")
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
    assert gpus[0].memory_gb == 16.0  # heuristic value preserved


def test_memory_override_excludes_when_non_positive(monkeypatch):
    """Override <= 0 → exclude GPU. Defensive — operator probably
    typo'd but we shouldn't construct ParallaxGPU with invalid
    memory (positive-only validator would raise)."""
    monkeypatch.setenv("PRSM_PARALLAX_MEMORY_GB_OVERRIDE", "0")
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
    assert list(provider()) == []


def test_memory_override_applies_to_remote_peers(monkeypatch):
    """Override applies to ALL peers (self + remote) so the
    allocator sees a consistent constrained pool."""
    monkeypatch.setenv("PRSM_PARALLAX_MEMORY_GB_OVERRIDE", "0.3")
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = {
        "tflops_fp16": 4.6, "ram_total_gb": 16.0,
    }
    node.discovery.known_peers = {
        "b" * 32: PeerInfo(
            node_id="b" * 32, address="1.2.3.4:9001",
            hardware_profile={
                "tflops_fp16": 8.0, "gpu_vram_gb": 24.0,
            },
        ),
    }
    provider = build_dht_backed_pool_provider(node)
    gpus = list(provider())
    assert len(gpus) == 2
    assert all(g.memory_gb == 0.3 for g in gpus)


def test_memory_override_unset_preserves_heuristic(monkeypatch):
    """No env set → heuristic memory value preserved (sprint 682
    behavior). Pure-additive contract."""
    monkeypatch.delenv("PRSM_PARALLAX_MEMORY_GB_OVERRIDE", raising=False)
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
    assert gpus[0].memory_gb == 16.0

"""Sprint 843 — producer-side hw overrides propagate via sp838 relay.

Multi-host live re-attest 2026-05-25 (post-sp838 fleet deploy)
exposed that the three hw override envs (TFLOPS_FP16, MEMORY_GB,
LAYER_CAPACITY) were consumer-side only: setting them on a
droplet only affected what THAT droplet saw in ITS pool view,
not what a remote consumer saw for that droplet via the sp838
hw_profile relay. Mac probe showed 1.92GB raw memory + cap=1
for both droplets despite each droplet's local env having
LAYER_CAPACITY_OVERRIDE=3.

Sprint 843 fix splits across two modules:

  Producer (prsm/node/hardware_profile_loader.py):
    new _merge_hardware_overrides(data) injects env-resolved
    overrides into the dict as explicit fields:
      tflops_fp16_override / memory_gb_override /
      layer_capacity_override
    Called from all 3 return paths (env-pin file, cache, fresh
    compute) so every advertised profile carries the operator's
    intent.

  Consumer (prsm/node/dht_backed_pool_provider.py):
    _hw_dict_to_parallax_gpu reads per-peer override fields
    FIRST. Consumer-side env becomes a coarse fallback for
    peers without explicit overrides.

Pre-843 wire-format compat: missing/invalid env → field
omitted; relayed dict shape unchanged.

Pin tests:
- Producer writes layer_capacity_override when env set
- Producer writes tflops_fp16_override when env set
- Producer writes memory_gb_override when env set
- Producer rejects non-positive / unparseable env values
- All 3 fields absent when no env set (wire-format compat)
- Consumer honors per-peer layer_capacity_override
- Consumer honors per-peer tflops_fp16_override
- Consumer honors per-peer memory_gb_override
- Consumer per-peer field wins over consumer-side env
- Consumer falls back to env when per-peer field absent
- Consumer ignores invalid per-peer override types
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


_OVERRIDE_ENVS = (
    "PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE",
    "PRSM_PARALLAX_MEMORY_GB_OVERRIDE",
    "PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE",
)


@pytest.fixture(autouse=True)
def clean_env():
    """Snapshot+restore the three sp843 env vars between tests."""
    saved = {k: os.environ.get(k) for k in _OVERRIDE_ENVS}
    for k in _OVERRIDE_ENVS:
        os.environ.pop(k, None)
    yield
    for k in _OVERRIDE_ENVS:
        os.environ.pop(k, None)
        if saved[k] is not None:
            os.environ[k] = saved[k]


# ============================================================
# Producer (hardware_profile_loader._merge_hardware_overrides)
# ============================================================


def test_producer_writes_layer_capacity_override():
    from prsm.node.hardware_profile_loader import (
        _merge_hardware_overrides,
    )
    os.environ["PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE"] = "3"
    data: Dict[str, Any] = {"ram_total_gb": 1.92}
    _merge_hardware_overrides(data)
    assert data["layer_capacity_override"] == 3
    assert isinstance(data["layer_capacity_override"], int)


def test_producer_writes_tflops_override():
    from prsm.node.hardware_profile_loader import (
        _merge_hardware_overrides,
    )
    os.environ["PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE"] = "30.0"
    data: Dict[str, Any] = {}
    _merge_hardware_overrides(data)
    assert data["tflops_fp16_override"] == 30.0


def test_producer_writes_memory_override():
    from prsm.node.hardware_profile_loader import (
        _merge_hardware_overrides,
    )
    os.environ["PRSM_PARALLAX_MEMORY_GB_OVERRIDE"] = "0.8"
    data: Dict[str, Any] = {}
    _merge_hardware_overrides(data)
    assert data["memory_gb_override"] == 0.8


def test_producer_omits_fields_when_no_env():
    """Wire-format compat: pre-843 dict shape preserved when no
    overrides set."""
    from prsm.node.hardware_profile_loader import (
        _merge_hardware_overrides,
    )
    data: Dict[str, Any] = {"ram_total_gb": 1.92}
    _merge_hardware_overrides(data)
    assert "layer_capacity_override" not in data
    assert "tflops_fp16_override" not in data
    assert "memory_gb_override" not in data


@pytest.mark.parametrize("raw", ["0", "-1", "not-a-num", "", " "])
def test_producer_rejects_invalid_layer_capacity(raw):
    from prsm.node.hardware_profile_loader import (
        _merge_hardware_overrides,
    )
    os.environ["PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE"] = raw
    data: Dict[str, Any] = {}
    _merge_hardware_overrides(data)
    assert "layer_capacity_override" not in data


@pytest.mark.parametrize("raw", ["0", "-0.5", "bad", ""])
def test_producer_rejects_invalid_memory(raw):
    from prsm.node.hardware_profile_loader import (
        _merge_hardware_overrides,
    )
    os.environ["PRSM_PARALLAX_MEMORY_GB_OVERRIDE"] = raw
    data: Dict[str, Any] = {}
    _merge_hardware_overrides(data)
    assert "memory_gb_override" not in data


# ============================================================
# Consumer (dht_backed_pool_provider._hw_dict_to_parallax_gpu)
# ============================================================


def test_consumer_honors_peer_layer_capacity_override():
    """The load-bearing assertion: producer's
    layer_capacity_override field reaches consumer's
    ParallaxGPU.layer_capacity unchanged."""
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    hw = {
        "tflops_fp16": 0.1,
        "ram_total_gb": 1.92,
        "layer_capacity_override": 3,
    }
    gpu = _hw_dict_to_parallax_gpu("peer-A", hw, "default")
    assert gpu is not None
    assert gpu.layer_capacity == 3


def test_consumer_honors_peer_tflops_override():
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    hw = {
        "tflops_fp16": 0.07,
        "ram_total_gb": 1.92,
        "tflops_fp16_override": 30.0,
    }
    gpu = _hw_dict_to_parallax_gpu("peer-A", hw, "default")
    assert gpu is not None
    assert gpu.tflops_fp16 == 30.0


def test_consumer_honors_peer_memory_override():
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    hw = {
        "tflops_fp16": 0.1,
        "ram_total_gb": 1.92,
        "memory_gb_override": 0.8,
    }
    gpu = _hw_dict_to_parallax_gpu("peer-A", hw, "default")
    assert gpu is not None
    assert gpu.memory_gb == 0.8


def test_per_peer_field_wins_over_consumer_env():
    """Consumer with LAYER_CAPACITY=5 in its OWN env sees a
    peer advertising override=3 → uses 3 (per-peer wins).
    Closes the sp838 multi-host gap."""
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    os.environ["PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE"] = "5"
    hw = {
        "tflops_fp16": 0.1,
        "ram_total_gb": 1.92,
        "layer_capacity_override": 3,
    }
    gpu = _hw_dict_to_parallax_gpu("peer-A", hw, "default")
    assert gpu is not None
    assert gpu.layer_capacity == 3


def test_consumer_env_falls_back_when_no_peer_override():
    """When per-peer field absent + consumer env set,
    env applies (legacy sp686 behavior preserved)."""
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    os.environ["PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE"] = "12"
    hw = {
        "tflops_fp16": 0.1,
        "ram_total_gb": 1.92,
    }
    gpu = _hw_dict_to_parallax_gpu("peer-A", hw, "default")
    assert gpu is not None
    assert gpu.layer_capacity == 12


def test_memory_heuristic_fallback_when_no_override_or_env():
    """When neither per-peer nor consumer env set, fall back to
    the legacy memory_gb/2 heuristic."""
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    hw = {
        "tflops_fp16": 0.1,
        "ram_total_gb": 10.0,
    }
    gpu = _hw_dict_to_parallax_gpu("peer-A", hw, "default")
    assert gpu is not None
    # 10.0 / 2.0 = 5
    assert gpu.layer_capacity == 5


@pytest.mark.parametrize("bad_val", ["3", 3.5, -1, 0, None, [3]])
def test_consumer_ignores_invalid_layer_cap_override(bad_val):
    """Wire data is untrusted — non-positive-int per-peer
    override field is ignored, falling back to memory math."""
    from prsm.node.dht_backed_pool_provider import (
        _hw_dict_to_parallax_gpu,
    )
    hw = {
        "tflops_fp16": 0.1,
        "ram_total_gb": 10.0,
        "layer_capacity_override": bad_val,
    }
    gpu = _hw_dict_to_parallax_gpu("peer-A", hw, "default")
    assert gpu is not None
    # Memory heuristic applies (5), not the bad value
    assert gpu.layer_capacity == 5

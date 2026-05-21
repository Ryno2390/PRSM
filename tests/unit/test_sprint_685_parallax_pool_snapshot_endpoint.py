"""Sprint 685 — /admin/parallax/pool/snapshot endpoint.

Live-attest surface for the DHT-backed pool. GET returns the
current pool as the provider would see it for the next inference
request. Operators use this to confirm peer discovery + hardware
advertisement is working without launching a full inference.

Schema (200 OK):
  {
    pool_kind: "dht-backed" | "static-empty" | null,
    gpu_count: int,
    gpus: [
      {
        node_id, region, layer_capacity, stake_amount,
        tier_attestation, tflops_fp16, memory_gb,
        memory_bandwidth_gbps, gpu_name, device, num_gpus,
      }, ...
    ],
  }

503 when no parallax executor is wired (operator hasn't opted in).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app_with_node(node):
    """Boot a minimal FastAPI app with just the snapshot endpoint
    registered against the given fake node object."""
    from prsm.node.api import register_parallax_pool_snapshot_endpoint
    app = FastAPI()
    register_parallax_pool_snapshot_endpoint(app, node)
    return TestClient(app)


def test_snapshot_returns_503_when_no_executor():
    """Pre-opt-in node (no inference_executor) → 503 with
    actionable message."""
    node = MagicMock()
    node.inference_executor = None
    client = _build_app_with_node(node)
    resp = client.get("/admin/parallax/pool/snapshot")
    assert resp.status_code == 503
    assert "PRSM_PARALLAX" in resp.json()["detail"]


def test_snapshot_returns_503_when_executor_missing_pool_provider():
    """Executor present but doesn't expose _pool_provider (some
    legacy executor sitting in inference_executor slot) → 503."""
    node = MagicMock()
    node.inference_executor = MagicMock(spec=[])  # no _pool_provider
    client = _build_app_with_node(node)
    resp = client.get("/admin/parallax/pool/snapshot")
    assert resp.status_code == 503


def test_snapshot_returns_empty_pool():
    """Provider returns [] → 200 with gpu_count=0 + empty gpus
    list. Distinguish from 503 (no executor) — this is "executor
    wired, no peers advertise yet"."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    discovery_node = MagicMock()
    discovery_node.identity.node_id = "a" * 32
    discovery_node.discovery._local_hardware_profile = None
    discovery_node.discovery.known_peers = {}
    provider = build_dht_backed_pool_provider(discovery_node)
    node = MagicMock()
    node.inference_executor = MagicMock()
    node.inference_executor._pool_provider = provider
    client = _build_app_with_node(node)
    resp = client.get("/admin/parallax/pool/snapshot")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["gpu_count"] == 0
    assert payload["gpus"] == []


def test_snapshot_returns_self_plus_peer():
    """Multi-node fleet: self + 1 advertising peer → 200 with
    both serialized."""
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo

    discovery_node = MagicMock()
    discovery_node.identity.node_id = "a" * 32
    discovery_node.discovery._local_hardware_profile = {
        "tflops_fp16": 4.6, "ram_total_gb": 16.0,
        "gpu_name": "Apple M4", "gpu_api": "metal",
    }
    discovery_node.discovery.known_peers = {
        "peerB": PeerInfo(
            node_id="peerB", address="2.2.2.2:9001",
            hardware_profile={
                "tflops_fp16": 8.0, "gpu_vram_gb": 12.0,
                "gpu_name": "RTX 3060", "gpu_api": "cuda",
            },
        ),
    }
    provider = build_dht_backed_pool_provider(discovery_node)

    node = MagicMock()
    node.inference_executor = MagicMock()
    node.inference_executor._pool_provider = provider
    client = _build_app_with_node(node)
    resp = client.get("/admin/parallax/pool/snapshot")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["gpu_count"] == 2
    ids = {g["node_id"] for g in payload["gpus"]}
    assert ids == {"a" * 32, "peerB"}
    # Schema check
    for g in payload["gpus"]:
        for k in (
            "node_id", "region", "layer_capacity", "stake_amount",
            "tier_attestation", "tflops_fp16", "memory_gb",
            "memory_bandwidth_gbps", "gpu_name", "device", "num_gpus",
        ):
            assert k in g


def test_snapshot_handles_provider_exception():
    """Provider raises → 500 with the error message, NOT a crash."""
    def _bad_provider():
        raise RuntimeError("provider blew up")
    node = MagicMock()
    node.inference_executor = MagicMock()
    node.inference_executor._pool_provider = _bad_provider
    client = _build_app_with_node(node)
    resp = client.get("/admin/parallax/pool/snapshot")
    assert resp.status_code == 500
    assert "provider blew up" in resp.json()["detail"]


def test_snapshot_reports_pool_kind_from_env(monkeypatch):
    """The pool_kind field echoes PRSM_PARALLAX_GPU_POOL_KIND so
    operators can verify the right kind was actually picked up by
    the daemon (vs falling through to default)."""
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_KIND", "dht-backed")
    node = MagicMock()
    node.inference_executor = MagicMock()
    node.inference_executor._pool_provider = lambda: []
    client = _build_app_with_node(node)
    resp = client.get("/admin/parallax/pool/snapshot")
    assert resp.status_code == 200
    assert resp.json()["pool_kind"] == "dht-backed"

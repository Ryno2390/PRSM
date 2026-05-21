"""Sprint 684 — end-to-end wiring smoke test for the DHT-backed pool.

Confirms that `build_parallax_executor_or_none` accepts the new
`PRSM_PARALLAX_GPU_POOL_KIND=dht-backed` value, constructs the
provider via `build_dht_backed_pool_provider`, and the provider
correctly enumerates self + known peers when invoked.

Composes sprints 680/681/682/683 without faking any layer. The
only fake is the node container (transport/identity); the
discovery + provider + reader paths are real.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_dht_pool_kind_recognized_in_build_parallax_executor(
    tmp_path, monkeypatch,
):
    """Setting PRSM_PARALLAX_GPU_POOL_KIND=dht-backed routes
    `build_parallax_executor_or_none` through the new provider
    factory (not the static-empty one). The executor itself may
    still return None for unrelated reasons (catalog/trust), but
    the unrecognized-kind branch is bypassed."""
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "schema_version": "v1",
        "models": {
            "gpt2": {
                "model_name": "gpt2", "num_layers": 12,
                "hidden_dim": 768, "num_attention_heads": 12,
                "num_kv_heads": 12, "vocab_size": 50257,
                "head_size": 64, "intermediate_dim": 3072,
            },
        },
    }))
    monkeypatch.setenv("PRSM_PARALLAX_MODEL_CATALOG_FILE", str(catalog))
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "mock")
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_KIND", "dht-backed")
    monkeypatch.delenv("PRSM_STAKE_BOND_ADDRESS", raising=False)

    from prsm.node.inference_wiring import build_parallax_executor_or_none
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery = MagicMock()
    node.discovery._local_hardware_profile = {
        "tflops_fp16": 4.6, "ram_total_gb": 16.0,
    }
    node.discovery.known_peers = {}

    executor = build_parallax_executor_or_none(node)
    assert executor is not None  # construction succeeded


def test_dht_pool_provider_observes_live_discovery_changes(monkeypatch):
    """The provider closure must read `known_peers` at EACH call,
    so peers joining/leaving show up without rebuilding the
    provider. Defends against a future refactor that snapshots
    known_peers at provider-build time."""
    monkeypatch.delenv("PRSM_STAKE_BOND_ADDRESS", raising=False)
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo

    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {}

    provider = build_dht_backed_pool_provider(node)
    assert list(provider()) == []

    node.discovery.known_peers["peerA"] = PeerInfo(
        node_id="peerA", address="1.2.3.4:9001",
        hardware_profile={"tflops_fp16": 4.6, "ram_total_gb": 16.0},
    )
    gpus = list(provider())
    assert len(gpus) == 1
    assert gpus[0].node_id == "peerA"


def test_dht_pool_provider_self_plus_two_peers(monkeypatch):
    """Multi-node fleet shape: self + 2 advertising peers + 1
    legacy peer → provider returns 3 (self + 2 advertisers,
    legacy excluded)."""
    monkeypatch.delenv("PRSM_STAKE_BOND_ADDRESS", raising=False)
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
        "peerB": PeerInfo(
            node_id="peerB", address="2.2.2.2:9001",
            hardware_profile={"tflops_fp16": 8.0, "gpu_vram_gb": 12.0},
        ),
        "peerC": PeerInfo(
            node_id="peerC", address="3.3.3.3:9001",
            hardware_profile={"tflops_fp16": 12.0, "gpu_vram_gb": 24.0},
        ),
        "legacy": PeerInfo(
            node_id="legacy", address="4.4.4.4:9001",
            hardware_profile=None,
        ),
    }
    provider = build_dht_backed_pool_provider(node)
    gpus = list(provider())
    assert {g.node_id for g in gpus} == {"a" * 32, "peerB", "peerC"}

"""Sprint 558 — production wiring path for ParallaxScheduledExecutor.

First sprint in the arc that takes /compute/inference from "mock
only" (sprint 438) to "real ParallaxScheduledExecutor". Sprint 546
wired sprint-419's ActivationDPAware decorator into the chain-RPC
factory, but no production caller invokes the factory — `/compute/
inference` still falls through to MockInferenceExecutor or 503.

Sprint 558 closes the loop at the OPT-IN PATH:

  PRSM_INFERENCE_EXECUTOR=parallax  → daemon calls
      build_parallax_executor_or_none(node)

The builder reads operator-supplied component config from env vars
(model catalog file, trust-stack kind, gpu-pool kind) and returns
a configured ParallaxScheduledExecutor when all required pieces are
present. Each missing piece logs an actionable warning + returns
None — daemon still boots, /compute/inference still returns the
existing 503 with the operator-readable detail surfaced by sprint
438. No new failure modes for existing operators.

Future sprints (559/560/561) wire each component's real
implementation. This sprint locks the WIRING CONTRACT so the
component knobs are operationally discoverable.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest


# ── env-var contract ──────────────────────────────────────


def test_returns_none_when_no_env_vars_set(monkeypatch):
    """Default state: no PRSM_PARALLAX_* env vars → returns None.
    No log noise; this is the "operator never opted in" path."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    for k in (
        "PRSM_PARALLAX_MODEL_CATALOG_FILE",
        "PRSM_PARALLAX_TRUST_STACK_KIND",
        "PRSM_PARALLAX_GPU_POOL_KIND",
    ):
        monkeypatch.delenv(k, raising=False)
    node = MagicMock()
    node.identity = MagicMock(node_id="stub")
    assert build_parallax_executor_or_none(node) is None


def test_returns_none_when_catalog_file_missing(
    monkeypatch, caplog,
):
    """All component-kinds set but catalog file doesn't exist on
    disk → log actionable warning + return None."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE",
        "/nonexistent/path/catalog.json",
    )
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "mock")
    monkeypatch.setenv("PRSM_PARALLAX_GPU_POOL_KIND", "static-empty")
    node = MagicMock()
    node.identity = MagicMock(node_id="stub")

    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(node)
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert "PRSM_PARALLAX_MODEL_CATALOG_FILE" in log_text, (
        "warning must name the missing env var"
    )


def test_returns_none_when_trust_stack_kind_unrecognized(
    monkeypatch, tmp_path, caplog,
):
    """An unknown trust-stack kind → log + return None. Defensive
    against typos in operator config."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{}")
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE", str(catalog_path),
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_TRUST_STACK_KIND", "totally-bogus",
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_GPU_POOL_KIND", "static-empty",
    )
    node = MagicMock()
    node.identity = MagicMock(node_id="stub")

    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(node)
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert (
        "PRSM_PARALLAX_TRUST_STACK_KIND" in log_text
        or "trust" in log_text.lower()
    )


def test_returns_none_when_gpu_pool_kind_unrecognized(
    monkeypatch, tmp_path, caplog,
):
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{}")
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE", str(catalog_path),
    )
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "mock")
    monkeypatch.setenv(
        "PRSM_PARALLAX_GPU_POOL_KIND", "also-bogus",
    )
    node = MagicMock()
    node.identity = MagicMock(node_id="stub")

    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(node)
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert (
        "PRSM_PARALLAX_GPU_POOL_KIND" in log_text
        or "gpu" in log_text.lower()
        or "pool" in log_text.lower()
    )


# ── happy path ────────────────────────────────────────────


def test_returns_executor_when_all_components_present(
    monkeypatch, tmp_path,
):
    """All env vars set + catalog file valid → returns a
    ParallaxScheduledExecutor whose supported_models reflects the
    catalog. This is the smoke test that the wiring CONSTRUCTS;
    actual cross-host inference still gates on multi-host bench."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )

    # Sprint 559: v1 schema with top-level schema_version + models.
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "schema_version": "v1",
        "models": {
            "test-model": {
                "model_name": "test-model",
                "mlx_model_name": "test-model",
                "head_size": 64,
                "hidden_dim": 128,
                "intermediate_dim": 256,
                "num_attention_heads": 4,
                "num_kv_heads": 4,
                "vocab_size": 1000,
                "num_layers": 4,
            },
        },
    }))
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE", str(catalog_path),
    )
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "mock")
    monkeypatch.setenv(
        "PRSM_PARALLAX_GPU_POOL_KIND", "static-empty",
    )
    from prsm.node.identity import generate_node_identity
    node = MagicMock()
    node.identity = generate_node_identity("test-settler")

    result = build_parallax_executor_or_none(node)
    assert isinstance(result, ParallaxScheduledExecutor), (
        f"expected ParallaxScheduledExecutor, got {type(result)}"
    )
    assert "test-model" in result.supported_models()


def test_returns_none_when_catalog_json_malformed(
    monkeypatch, tmp_path, caplog,
):
    """Catalog file exists but isn't valid JSON → log + None."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not valid json {{{")
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE", str(bad_path),
    )
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "mock")
    monkeypatch.setenv(
        "PRSM_PARALLAX_GPU_POOL_KIND", "static-empty",
    )
    node = MagicMock()
    node.identity = MagicMock(node_id="stub")

    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(node)
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert "json" in log_text.lower() or "parse" in log_text.lower()


# ── node.py integration ───────────────────────────────────


def test_node_imports_inference_wiring_module():
    """The new wiring module must be importable from node.py path —
    sprint 559+ assumes this import works."""
    import prsm.node.inference_wiring as mod
    assert hasattr(mod, "build_parallax_executor_or_none")
    assert callable(mod.build_parallax_executor_or_none)

"""Sprint 609 (Phase 2F-4) — env selector recognizes layer_stage kind.

Extends sprint 604's _build_stage_executor_from_env. Three new
test surfaces:
  - PRSM_PARALLAX_STAGE_EXECUTOR_KIND=layer_stage + node arg + all
    other env config → factory call → returns
    LayerStageServerStageExecutor
  - layer_stage kind with missing deps → falls back to stub + warns
  - layer_stage kind with node=None → falls back to stub + warns
  - PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=identity → IdentityLayerSliceRunner
  - PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=bogus → identity + warns
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


def test_runner_selector_default_identity():
    from prsm.node.chain_executor_adapters import (
        _build_layer_slice_runner_from_env, IdentityLayerSliceRunner,
    )
    os.environ.pop("PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND", None)
    runner = _build_layer_slice_runner_from_env()
    assert isinstance(runner, IdentityLayerSliceRunner)


def test_runner_selector_unknown_falls_back_to_identity(caplog):
    from prsm.node.chain_executor_adapters import (
        _build_layer_slice_runner_from_env, IdentityLayerSliceRunner,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND": "huggingface"},
        clear=False,
    ):
        with caplog.at_level("WARNING"):
            runner = _build_layer_slice_runner_from_env()
    assert isinstance(runner, IdentityLayerSliceRunner)
    assert any("huggingface" in r.getMessage() for r in caplog.records)


def test_stage_selector_layer_stage_no_node_falls_back(caplog):
    from prsm.node.chain_executor_adapters import (
        _build_stage_executor_from_env,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_STAGE_EXECUTOR_KIND": "layer_stage"},
        clear=False,
    ):
        with caplog.at_level("WARNING"):
            exe = _build_stage_executor_from_env(node=None)
    # Falls back to stub
    from prsm.node.chain_executor_adapters import build_stub_stage_executor
    stub = build_stub_stage_executor()
    assert type(exe).__name__ == type(stub).__name__
    assert any("node arg" in r.getMessage() for r in caplog.records)


def test_stage_selector_layer_stage_missing_anchor_falls_back(caplog):
    from prsm.node.chain_executor_adapters import (
        _build_stage_executor_from_env, build_stub_stage_executor,
    )
    node = MagicMock()
    node.identity = MagicMock(node_id="self")
    with patch.dict(
        os.environ,
        {
            "PRSM_PARALLAX_STAGE_EXECUTOR_KIND": "layer_stage",
            "PRSM_MODEL_REGISTRY_ROOT": "/tmp/registry",
            # Sprint 629: pin sepolia so networks.py mainnet default
            # doesn't fill in the anchor and defeat this test.
            "PRSM_NETWORK": "sepolia",
        },
        clear=False,
    ):
        os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
        with caplog.at_level("WARNING"):
            exe = _build_stage_executor_from_env(node=node)
    stub = build_stub_stage_executor()
    assert type(exe).__name__ == type(stub).__name__
    # Warns about the factory raise
    assert any(
        "build_layer_stage_server_executor" in r.getMessage()
        or "anchor" in r.getMessage().lower()
        for r in caplog.records
    )


def test_stage_selector_layer_stage_happy_path(tmp_path):
    """All env set + node valid + anchor module mocked →
    layer_stage returns a LayerStageServerStageExecutor.
    """
    from prsm.node.chain_executor_adapters import (
        _build_stage_executor_from_env,
        LayerStageServerStageExecutor,
    )
    node = MagicMock()
    node.identity = MagicMock(node_id="self")

    fake_anchor_mod = MagicMock()
    fake_anchor_mod.PublisherKeyAnchorClient = MagicMock(
        return_value=MagicMock(lookup=MagicMock(return_value=None)),
    )
    fake_layer_server = MagicMock()
    fake_layer_server.handle = MagicMock(return_value=b"x")
    fake_server_mod = MagicMock()
    fake_server_mod.LayerStageServer = MagicMock(
        return_value=fake_layer_server,
    )

    with patch.dict(
        os.environ,
        {
            "PRSM_PARALLAX_STAGE_EXECUTOR_KIND": "layer_stage",
            "PRSM_MODEL_REGISTRY_ROOT": str(tmp_path),
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xdead",
        },
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {
                "prsm.security.publisher_key_anchor.client": fake_anchor_mod,
                "prsm.compute.chain_rpc.server": fake_server_mod,
            },
        ):
            exe = _build_stage_executor_from_env(node=node)

    assert isinstance(exe, LayerStageServerStageExecutor)

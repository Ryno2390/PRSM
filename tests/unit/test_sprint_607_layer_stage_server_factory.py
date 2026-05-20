"""Sprint 607 (Phase 2F-2) — build_layer_stage_server_executor factory.

Constructs a fully-configured LayerStageServer from operator config
+ wraps it in sprint-606's adapter so it's ready to use as the
node's _chain_stage_executor.

Required pieces:
  node.identity                       — NodeIdentity for response signing
  FilesystemModelRegistry(root, anchor)  — from PRSM_MODEL_REGISTRY_ROOT env
  SoftwareTEERuntime                   — zero-arg default
  anchor                               — sprint 580 _build_anchor_or_none
  runner                               — operator-supplied (Phase 2F-3+)

Tests:
  - Factory exists + has the right signature
  - Raises StageExecutionError when PRSM_MODEL_REGISTRY_ROOT unset
  - Raises StageExecutionError when anchor missing
  - Raises StageExecutionError when runner is None
  - Happy path: returns a LayerStageServerStageExecutor
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


def test_factory_function_exists():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "build_layer_stage_server_executor")


def test_factory_raises_when_registry_root_unset():
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor, StageExecutionError,
    )
    node = MagicMock()
    node.identity = MagicMock(node_id="self")
    runner = MagicMock()
    runner.run_layer_range = MagicMock()  # sufficient for Protocol shape
    os.environ.pop("PRSM_MODEL_REGISTRY_ROOT", None)
    os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
    with pytest.raises(StageExecutionError, match="MODEL_REGISTRY_ROOT"):
        build_layer_stage_server_executor(node=node, runner=runner)


def test_factory_raises_when_anchor_unset(tmp_path):
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor, StageExecutionError,
    )
    node = MagicMock()
    node.identity = MagicMock(node_id="self")
    runner = MagicMock()
    runner.run_layer_range = MagicMock()
    with patch.dict(
        os.environ,
        {"PRSM_MODEL_REGISTRY_ROOT": str(tmp_path)},
        clear=False,
    ):
        os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
        with pytest.raises(StageExecutionError, match="anchor"):
            build_layer_stage_server_executor(node=node, runner=runner)


def test_factory_raises_when_runner_is_none(tmp_path):
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor, StageExecutionError,
    )
    node = MagicMock()
    node.identity = MagicMock(node_id="self")

    # Mock the anchor module so anchor IS available — then the
    # runner-None check is what raises.
    fake_anchor_mod = MagicMock()
    fake_anchor_mod.PublisherKeyAnchorClient = MagicMock(
        return_value=MagicMock(lookup=MagicMock(return_value=None)),
    )

    with patch.dict(
        os.environ,
        {
            "PRSM_MODEL_REGISTRY_ROOT": str(tmp_path),
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xdead",
        },
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {"prsm.security.publisher_key_anchor.client": fake_anchor_mod},
        ):
            with pytest.raises(StageExecutionError, match="runner"):
                build_layer_stage_server_executor(node=node, runner=None)


def test_factory_returns_layer_stage_server_executor(tmp_path):
    """Happy path: env set + runner supplied → factory returns
    a LayerStageServerStageExecutor instance.
    """
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
        LayerStageServerStageExecutor,
    )
    node = MagicMock()
    node.identity = MagicMock(node_id="self")
    runner = MagicMock()
    runner.run_layer_range = MagicMock()  # Protocol-shape pass

    # Mock the heavy LayerStageServer constructor + anchor module
    fake_layer_server = MagicMock()
    fake_layer_server.handle = MagicMock(return_value=b"x")
    fake_anchor_mod = MagicMock()
    fake_anchor_mod.PublisherKeyAnchorClient = MagicMock(
        return_value=MagicMock(lookup=MagicMock(return_value=None)),
    )
    fake_server_mod = MagicMock()
    fake_server_mod.LayerStageServer = MagicMock(
        return_value=fake_layer_server,
    )

    with patch.dict(
        os.environ,
        {
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
            exe = build_layer_stage_server_executor(
                node=node, runner=runner,
            )

    assert isinstance(exe, LayerStageServerStageExecutor)
    # LayerStageServer was called once
    fake_server_mod.LayerStageServer.assert_called_once()

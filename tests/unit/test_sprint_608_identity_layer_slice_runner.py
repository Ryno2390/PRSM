"""Sprint 608 (Phase 2F-3) — IdentityLayerSliceRunner.

The smallest possible LayerSliceRunner: returns the input activation
UNCHANGED. Lets the FULL real-model wire (LayerStageServer parses
the RunLayerSliceRequest, verifies upstream tokens, signs the
response) be exercised end-to-end without needing actual model
weights loaded.

Analogue to Phase 2E-3's EchoStageExecutor but at the runner layer
(deeper in the stack — LayerStageServer.handle still runs full
parse + sign + validate paths).

Phase 2F-4+ ships real HuggingFace / model-framework runners.
"""
from __future__ import annotations

import numpy as np


def test_module_exposes_identity_runner():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "IdentityLayerSliceRunner")


def test_runner_has_run_layer_range_method():
    """LayerSliceRunner Protocol requires .run_layer_range."""
    from prsm.node.chain_executor_adapters import IdentityLayerSliceRunner
    runner = IdentityLayerSliceRunner()
    assert hasattr(runner, "run_layer_range")
    assert callable(runner.run_layer_range)


def test_runner_returns_input_activation_unchanged():
    """The identity runner echoes the activation through unchanged."""
    from prsm.node.chain_executor_adapters import IdentityLayerSliceRunner
    from prsm.compute.tee.models import TEEType

    runner = IdentityLayerSliceRunner()
    activation = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = runner.run_layer_range(
        model=None,
        layer_range=(0, 4),
        activation=activation,
        privacy_tier=None,
        is_final_stage=False,
    )

    assert np.array_equal(result.output, activation)
    assert result.duration_seconds >= 0
    assert isinstance(result.tee_attestation, bytes)
    assert result.tee_type == TEEType.SOFTWARE
    assert result.epsilon_spent == 0.0


def test_runner_2d_activation():
    """Multi-dim activation (e.g., [batch, hidden_dim]) passes
    through cleanly.
    """
    from prsm.node.chain_executor_adapters import IdentityLayerSliceRunner

    runner = IdentityLayerSliceRunner()
    activation = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    result = runner.run_layer_range(
        model=None, layer_range=(0, 1),
        activation=activation, privacy_tier=None,
        is_final_stage=False,
    )
    assert np.array_equal(result.output, activation)
    assert result.output.shape == (2, 3, 4)


def test_runner_returns_copy_not_alias():
    """Defense-in-depth: the runner must not return the input
    array by-reference. Caller mutations to .output must NOT
    affect the source.
    """
    from prsm.node.chain_executor_adapters import IdentityLayerSliceRunner

    runner = IdentityLayerSliceRunner()
    activation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = runner.run_layer_range(
        model=None, layer_range=(0, 1),
        activation=activation, privacy_tier=None,
        is_final_stage=False,
    )
    # Mutating output must not bleed into source
    result.output[0] = 999.0
    assert activation[0] == 1.0, (
        "IdentityLayerSliceRunner must return a COPY, not alias the "
        "input array — defense against caller mutations"
    )

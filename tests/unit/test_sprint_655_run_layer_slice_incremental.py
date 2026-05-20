"""Sprint 655 — KV-cache arc piece 2: _run_layer_slice_incremental
delegator skeleton.

Sprint 654 documented the gate rejecting non-PREFILL on the inline
path. Sprint 655 adds the SERVER-side delegator method that will
route INCREMENTAL requests to the runner once sprint 656 wires
HuggingFaceLayerSliceRunner.run_layer_range_incremental.

This sprint's deliverable: the method exists, returns NOT_IMPLEMENTED
gracefully against runners that don't yet support INCREMENTAL,
preserves the same TIMEOUT/INTERNAL_ERROR mapping as the PREFILL
delegator. Pure structural plumbing.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from prsm.compute.chain_rpc.protocol import (
    DecodeMode, PrivacyLevel, ContentTier, HandoffToken,
    RunLayerSliceRequest,
)
from prsm.compute.chain_rpc.server import LayerStageServer
from prsm.compute.tee.models import TEEType
from prsm.node.identity import generate_node_identity


def _make_request(*, request_id="r0"):
    settler = generate_node_identity()
    deadline = 9999999999.0
    token = HandoffToken.sign(
        identity=settler, request_id=request_id,
        chain_stage_index=0, chain_total_stages=1,
        deadline_unix=deadline,
    )
    return RunLayerSliceRequest(
        request_id=request_id, model_id="test-model",
        layer_range=(0, 1),
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        activation_blob=np.zeros((1, 1, 4), dtype=np.float32).tobytes(),
        activation_shape=(1, 1, 4),
        activation_dtype="float32",
        upstream_token=token, deadline_unix=deadline,
    )


def _make_server_with_runner(runner):
    return LayerStageServer(
        identity=generate_node_identity(),
        registry=MagicMock(get=MagicMock()),
        runner=runner,
        tee_runtime=MagicMock(),
        anchor=MagicMock(),
    )


def test_method_exists_on_server():
    """Sprint 655 — the delegator method is added to LayerStageServer."""
    server = _make_server_with_runner(MagicMock(run_layer_range=MagicMock()))
    assert hasattr(server, "_run_layer_slice_incremental"), (
        "sprint 655 must add _run_layer_slice_incremental method"
    )


def test_returns_NOT_IMPLEMENTED_when_runner_lacks_method():
    """If runner doesn't implement run_layer_range_incremental,
    delegator returns a clean (StageErrorCode, message) tuple.
    """
    runner = MagicMock(spec=["run_layer_range"])
    # No run_layer_range_incremental attribute
    server = _make_server_with_runner(runner)
    request = _make_request()
    activation = np.zeros((1, 1, 4), dtype=np.float32)
    result = server._run_layer_slice_incremental(
        request, MagicMock(), activation, prev_kv_state=None,
    )
    assert isinstance(result, tuple)
    code, message = result
    assert "incremental" in message.lower()
    assert "sprint 656" in message.lower()


def test_delegates_to_runner_when_method_present():
    """When runner DOES implement run_layer_range_incremental,
    delegator calls it with the canonical kwargs.
    """
    runner_result = MagicMock()  # opaque result
    runner = MagicMock()
    runner.run_layer_range_incremental = MagicMock(
        return_value=runner_result,
    )
    server = _make_server_with_runner(runner)
    request = _make_request()
    activation = np.zeros((1, 1, 4), dtype=np.float32)
    sentinel_kv = object()
    result = server._run_layer_slice_incremental(
        request, MagicMock(), activation, prev_kv_state=sentinel_kv,
    )
    assert result is runner_result
    runner.run_layer_range_incremental.assert_called_once()
    call_kwargs = runner.run_layer_range_incremental.call_args.kwargs
    # Canonical kwargs match _run_layer_slice + prev_kv_state
    assert call_kwargs["activation"] is activation
    assert call_kwargs["prev_kv_state"] is sentinel_kv
    assert call_kwargs["layer_range"] == request.layer_range
    assert call_kwargs["privacy_tier"] == request.privacy_tier


def test_timeout_error_mapped_to_StageErrorCode():
    """Runner raises TimeoutError → (TIMEOUT, message) tuple."""
    from prsm.compute.chain_rpc.protocol import StageErrorCode
    runner = MagicMock()
    runner.run_layer_range_incremental = MagicMock(
        side_effect=TimeoutError("forward exceeded budget"),
    )
    server = _make_server_with_runner(runner)
    request = _make_request()
    activation = np.zeros((1, 1, 4), dtype=np.float32)
    result = server._run_layer_slice_incremental(
        request, MagicMock(), activation, prev_kv_state=None,
    )
    code, msg = result
    assert code == StageErrorCode.TIMEOUT
    assert "budget" in msg


def test_generic_exception_mapped_to_INTERNAL_ERROR():
    """Runner raises something else → (INTERNAL_ERROR, ...) tuple."""
    from prsm.compute.chain_rpc.protocol import StageErrorCode
    runner = MagicMock()
    runner.run_layer_range_incremental = MagicMock(
        side_effect=RuntimeError("kv tensor mismatch"),
    )
    server = _make_server_with_runner(runner)
    request = _make_request()
    activation = np.zeros((1, 1, 4), dtype=np.float32)
    result = server._run_layer_slice_incremental(
        request, MagicMock(), activation, prev_kv_state=None,
    )
    code, msg = result
    assert code == StageErrorCode.INTERNAL_ERROR
    assert "RuntimeError" in msg

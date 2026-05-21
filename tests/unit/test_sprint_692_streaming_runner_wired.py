"""Sprint 692 F41 fix — streaming_runner construction in
build_layer_stage_server_executor.

Sprint 691 closed F40 (token_stream_send_message adapter) but
live-attest of /compute/inference/stream surfaced F41: the
LayerStageServer was constructed without a streaming_runner
(default None), so handle_token_stream returned
"INTERNAL_ERROR: server has no streaming_runner configured".

Sprint 692 wires SyntheticStreamingRunner as the streaming_runner
when PRSM_PARALLAX_HF_MODEL_ID is set. SyntheticStreamingRunner
wraps the existing unary runner: runs full forward, decodes
activation → text via build_hf_output_decoder, chunks into
streaming frames. NOT real token-by-token autoregressive decode
(that's AutoregressiveStreamingRunner + sprint 693's prompt
registry territory) but it exposes the SSE protocol end-to-end
so operators can validate the streaming wire path.
"""
from __future__ import annotations

import inspect

import pytest


def test_streaming_runner_constructed_in_source():
    """Pin: build_layer_stage_server_executor must construct a
    streaming_runner and pass it to LayerStageServer."""
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "SyntheticStreamingRunner" in src, (
        "build_layer_stage_server_executor must construct a "
        "SyntheticStreamingRunner (sprint 692 F41)"
    )
    assert "streaming_runner=streaming_runner" in src, (
        "LayerStageServer must be constructed with the "
        "streaming_runner= kwarg"
    )


def test_streaming_runner_skipped_when_no_hf_model_id(monkeypatch):
    """No PRSM_PARALLAX_HF_MODEL_ID → streaming_runner stays None
    (server's clear 'no streaming_runner configured' error
    surfaces; better than silent garbage)."""
    monkeypatch.delenv("PRSM_PARALLAX_HF_MODEL_ID", raising=False)
    # Source-level guard: gating must be present
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "PRSM_PARALLAX_HF_MODEL_ID" in src
    assert "streaming_runner = None" in src


def test_synthetic_streaming_runner_exists_in_streaming_module():
    """Sanity — the module sprint 692 depends on exists."""
    from prsm.compute.inference.streaming_runner import (
        SyntheticStreamingRunner,
    )
    assert SyntheticStreamingRunner is not None


def test_output_decoder_reused_for_synthetic_runner():
    """Sprint 692 reuses sprint 616's build_hf_output_decoder for
    the synthetic runner. Pin to ensure we don't accidentally
    introduce a parallel-decoder implementation that diverges."""
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "build_hf_output_decoder" in src, (
        "sprint 692 must reuse sprint 616's build_hf_output_decoder "
        "for the streaming runner's output decoder (no parallel impl)"
    )

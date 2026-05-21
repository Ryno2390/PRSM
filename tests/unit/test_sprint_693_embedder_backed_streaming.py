"""Sprint 693 F42 bypass — EmbedderBackedStreamingRunner.

Sprint 692 wired SyntheticStreamingRunner which works but doesn't
produce real per-token autoregressive output (it chunks one
forward pass's text). Real autoregressive streaming via
AutoregressiveStreamingRunner needs a `prompt_provider` callable
that returns the prompt text — but PRSM's pipeline embeds the
prompt into activation BEFORE dispatch, so server-side
reconstruction requires a per-request_id registry (F42 — multi-
sprint design problem).

Sprint 693 sidesteps F42 entirely with
EmbedderBackedStreamingRunner: uses HF's
`model.generate(inputs_embeds=activation, ...)` path. The
activation IS the initial embedding fed into the model's
transformer blocks, so HF's autoregressive loop runs from there
without needing the original text.

`PRSM_PARALLAX_STREAMING_RUNNER_KIND=embedder_backed` (the new
default) enables this; `=synthetic` falls back to sprint-692
text-chunking behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def test_runner_constructor_requires_generate_method():
    """Model without generate() → clear error."""
    from prsm.compute.inference.autoregressive_runner import (
        EmbedderBackedStreamingRunner,
    )
    from prsm.compute.tee.models import TEEType
    model = MagicMock(spec=[])  # no .generate
    with pytest.raises(RuntimeError, match="generate"):
        EmbedderBackedStreamingRunner(
            model=model,
            tokenizer=MagicMock(),
            tee_attestation=b"att",
            tee_type=TEEType.SOFTWARE,
        )


def test_runner_constructor_requires_tokenizer_decode():
    """Tokenizer without decode() → clear error."""
    from prsm.compute.inference.autoregressive_runner import (
        EmbedderBackedStreamingRunner,
    )
    from prsm.compute.tee.models import TEEType
    tokenizer = MagicMock(spec=[])  # no .decode
    with pytest.raises(RuntimeError, match="decode"):
        EmbedderBackedStreamingRunner(
            model=MagicMock(),
            tokenizer=tokenizer,
            tee_attestation=b"att",
            tee_type=TEEType.SOFTWARE,
        )


def test_runner_constructor_requires_bytes_attestation():
    from prsm.compute.inference.autoregressive_runner import (
        EmbedderBackedStreamingRunner,
    )
    from prsm.compute.tee.models import TEEType
    with pytest.raises(RuntimeError, match="tee_attestation"):
        EmbedderBackedStreamingRunner(
            model=MagicMock(),
            tokenizer=MagicMock(),
            tee_attestation="not-bytes",  # type: ignore[arg-type]
            tee_type=TEEType.SOFTWARE,
        )


def test_non_tail_yields_single_error_chunk():
    """is_final_stage=False → single terminal chunk with
    finish_reason=error (per tail-only contract)."""
    from prsm.compute.inference.autoregressive_runner import (
        EmbedderBackedStreamingRunner,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    runner = EmbedderBackedStreamingRunner(
        model=MagicMock(),
        tokenizer=MagicMock(),
        tee_attestation=b"att",
        tee_type=TEEType.SOFTWARE,
    )
    chunks = list(runner.run_layer_slice_streaming(
        model=MagicMock(),
        layer_range=(0, 12),
        activation=np.zeros((1, 5, 768), dtype=np.float32),
        privacy_tier=PrivacyLevel.NONE,
        is_final_stage=False,
    ))
    assert len(chunks) == 1
    assert chunks[0].finish_reason == "error"


def test_chain_executor_adapters_constructs_embedder_backed_by_default():
    """Pin: build_layer_stage_server_executor must default to
    embedder_backed kind (sprint 693)."""
    import inspect
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "EmbedderBackedStreamingRunner" in src, (
        "sprint 693 must construct EmbedderBackedStreamingRunner"
    )
    assert "embedder_backed" in src, (
        "default streaming kind must be embedder_backed"
    )


def test_chain_executor_adapters_supports_synthetic_kind_override():
    """Pin: operators can opt back into sprint-692 synthetic
    chunking via PRSM_PARALLAX_STREAMING_RUNNER_KIND=synthetic."""
    import inspect
    from prsm.node.chain_executor_adapters import (
        build_layer_stage_server_executor,
    )
    src = inspect.getsource(build_layer_stage_server_executor)
    assert "synthetic" in src
    assert "SyntheticStreamingRunner" in src

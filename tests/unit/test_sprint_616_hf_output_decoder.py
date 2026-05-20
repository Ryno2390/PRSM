"""Sprint 616 (Phase 2F-5g) — build_hf_output_decoder factory + wire.

Symmetric inverse of sprint-614's prompt_encoder. The
RpcChainExecutor factory takes an optional
``output_decoder: Callable[[np.ndarray], str]`` that converts the
chain tail's final activation (logits, after sprint-613 LM head)
into a string. Default identity-decoder is useless for actual
generation.

Sprint 616 ships:
  - build_hf_output_decoder(model_id, device) → closure
  - Closure lazy-loads HF AutoTokenizer (cached)
  - On call: logits → argmax over vocab → token_id →
    tokenizer.decode(token_id) → str
  - Source-grep tests for env-wire into _build_chain_executor
    (PRSM_PARALLAX_OUTPUT_DECODER_KIND=huggingface)

Tokenizer-only load is lighter than sprint 614's encoder (no
embedding lookup needed), so this path scales better to large
models.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _install_fake_tokenizer():
    fake_tokenizer = MagicMock()
    fake_tokenizer.decode = MagicMock(return_value=" hello")

    fake_tf = MagicMock()
    fake_tf.AutoTokenizer = MagicMock()
    fake_tf.AutoTokenizer.from_pretrained = MagicMock(
        return_value=fake_tokenizer,
    )

    return fake_tf, fake_tokenizer


def test_factory_function_exists():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "build_hf_output_decoder")


def test_factory_returns_callable():
    from prsm.node.chain_executor_adapters import build_hf_output_decoder
    fake_tf, _ = _install_fake_tokenizer()
    with patch.dict(sys.modules, {"transformers": fake_tf}):
        decoder = build_hf_output_decoder(model_id="fake/m")
    assert callable(decoder)


def test_decoder_argmax_decodes_token_id():
    """Logits with shape [B, S, V] → argmax over V at last position
    → tokenizer.decode([token_id]) → str.
    """
    from prsm.node.chain_executor_adapters import build_hf_output_decoder
    fake_tf, fake_tok = _install_fake_tokenizer()

    # Logits where vocab idx 42 is highest at last position
    logits = np.zeros((1, 3, 100), dtype=np.float32)
    logits[0, -1, 42] = 999.0

    with patch.dict(sys.modules, {"transformers": fake_tf}):
        decoder = build_hf_output_decoder(model_id="fake/m")
        result = decoder(logits)

    assert isinstance(result, str)
    # decode received [42] (the argmax token)
    fake_tok.decode.assert_called_once()
    args, _ = fake_tok.decode.call_args
    assert list(args[0]) == [42]


def test_decoder_lazy_loads_tokenizer():
    from prsm.node.chain_executor_adapters import build_hf_output_decoder
    fake_tf, _ = _install_fake_tokenizer()
    with patch.dict(sys.modules, {"transformers": fake_tf}):
        decoder = build_hf_output_decoder(model_id="fake/m")
        fake_tf.AutoTokenizer.from_pretrained.assert_not_called()
        decoder(np.zeros((1, 1, 10), dtype=np.float32))
    fake_tf.AutoTokenizer.from_pretrained.assert_called_once_with("fake/m")


def test_decoder_caches_loaded_tokenizer():
    from prsm.node.chain_executor_adapters import build_hf_output_decoder
    fake_tf, _ = _install_fake_tokenizer()
    with patch.dict(sys.modules, {"transformers": fake_tf}):
        decoder = build_hf_output_decoder(model_id="fake/m")
        for _ in range(3):
            decoder(np.zeros((1, 1, 10), dtype=np.float32))
    assert fake_tf.AutoTokenizer.from_pretrained.call_count == 1


def test_decoder_raises_stage_execution_error_on_load_failure():
    from prsm.node.chain_executor_adapters import (
        build_hf_output_decoder, StageExecutionError,
    )
    fake_tf = MagicMock()
    fake_tf.AutoTokenizer = MagicMock()
    fake_tf.AutoTokenizer.from_pretrained = MagicMock(
        side_effect=RuntimeError("tokenizer 404"),
    )
    with patch.dict(sys.modules, {"transformers": fake_tf}):
        decoder = build_hf_output_decoder(model_id="missing/m")
        with pytest.raises(StageExecutionError, match="404"):
            decoder(np.zeros((1, 1, 10), dtype=np.float32))


def test_decoder_handles_2d_logits():
    """[S, V] (no batch) → argmax at last position."""
    from prsm.node.chain_executor_adapters import build_hf_output_decoder
    fake_tf, fake_tok = _install_fake_tokenizer()
    logits = np.zeros((3, 100), dtype=np.float32)
    logits[-1, 7] = 999.0
    with patch.dict(sys.modules, {"transformers": fake_tf}):
        decoder = build_hf_output_decoder(model_id="fake/m")
        decoder(logits)
    args, _ = fake_tok.decode.call_args
    assert list(args[0]) == [7]


# ── env-wire into _build_chain_executor ──────────────────────────


def test_chain_executor_imports_build_hf_output_decoder():
    import inspect
    from prsm.node.inference_wiring import _build_chain_executor
    src = inspect.getsource(_build_chain_executor)
    assert "build_hf_output_decoder" in src


def test_chain_executor_reads_output_decoder_env_var():
    import inspect
    from prsm.node.inference_wiring import _build_chain_executor
    src = inspect.getsource(_build_chain_executor)
    assert "PRSM_PARALLAX_OUTPUT_DECODER_KIND" in src

"""Sprint 612 (Phase 2F-5c) — polymorphic HF layer extraction.

HuggingFace transformer architectures expose their layer lists at
different attribute paths:
  LLaMA / Mistral / Qwen   → model.model.layers (Phase 2F-5b)
  GPT-2 / Falcon           → model.transformer.h
  GPT-NeoX                 → model.gpt_neox.layers

Sprint 612 introduces ``_resolve_layers(hf_model)`` that tries these
paths in order + returns the first that resolves. Raises
StageExecutionError listing all attempted paths on miss.

Phase 2F-5b only supported LLaMA-style; this extends to the common
causal-LM architectures.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _llama_style_model():
    m = MagicMock()
    m.model = MagicMock()
    m.model.layers = [MagicMock() for _ in range(4)]
    return m


def _gpt2_style_model():
    m = MagicMock()
    # NO model.model attribute (so first path misses)
    m.model = None
    m.transformer = MagicMock()
    m.transformer.h = [MagicMock() for _ in range(4)]
    return m


def _gpt_neox_style_model():
    m = MagicMock()
    m.model = None
    m.transformer = None
    m.gpt_neox = MagicMock()
    m.gpt_neox.layers = [MagicMock() for _ in range(4)]
    return m


def test_helper_exposed():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "_resolve_hf_layers")


def test_llama_style_resolves_via_model_layers():
    from prsm.node.chain_executor_adapters import _resolve_hf_layers
    model = _llama_style_model()
    layers = _resolve_hf_layers(model)
    assert layers is model.model.layers


def test_gpt2_style_resolves_via_transformer_h():
    from prsm.node.chain_executor_adapters import _resolve_hf_layers
    model = _gpt2_style_model()
    layers = _resolve_hf_layers(model)
    assert layers is model.transformer.h


def test_gpt_neox_style_resolves_via_gpt_neox_layers():
    from prsm.node.chain_executor_adapters import _resolve_hf_layers
    model = _gpt_neox_style_model()
    layers = _resolve_hf_layers(model)
    assert layers is model.gpt_neox.layers


def test_unknown_architecture_raises_with_paths_attempted():
    """Model without any of the known paths → StageExecutionError
    listing all attempted paths so operator triages."""
    from prsm.node.chain_executor_adapters import (
        _resolve_hf_layers, StageExecutionError,
    )
    bare = MagicMock(spec=[])  # no attrs at all
    with pytest.raises(StageExecutionError) as exc_info:
        _resolve_hf_layers(bare)
    msg = str(exc_info.value)
    assert ".model.layers" in msg
    assert ".transformer.h" in msg
    assert ".gpt_neox.layers" in msg


def test_run_layer_range_uses_helper_for_gpt2_model():
    """Integration: a GPT-2-style mocked model goes through the
    forward pass via transformer.h.
    """
    import sys
    from unittest.mock import patch
    import numpy as np

    fake_layer = MagicMock(return_value=(MagicMock(),))
    fake_model = _gpt2_style_model()
    fake_model.transformer.h = [fake_layer for _ in range(8)]
    fake_model.eval = MagicMock(return_value=fake_model)
    fake_model.to = MagicMock(return_value=fake_model)

    fake_tf = MagicMock()
    fake_tf.AutoModelForCausalLM = MagicMock()
    fake_tf.AutoModelForCausalLM.from_pretrained = MagicMock(
        return_value=fake_model,
    )

    fake_torch = MagicMock()
    fake_torch.float32 = "f32"
    fake_torch.no_grad = MagicMock()
    fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)
    fake_torch.arange = MagicMock(return_value=MagicMock())

    def _from_numpy(arr):
        t = MagicMock()
        t.to = MagicMock(return_value=t)
        t.dim = MagicMock(return_value=arr.ndim)
        t.unsqueeze = MagicMock(return_value=t)
        t.shape = arr.shape if arr.ndim == 3 else (1,) + arr.shape
        t.squeeze = MagicMock(return_value=t)
        cpu_mock = MagicMock()
        cpu_mock.numpy = MagicMock(return_value=arr)
        t.cpu = MagicMock(return_value=cpu_mock)
        return t
    fake_torch.from_numpy = MagicMock(side_effect=_from_numpy)

    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner

    with patch.dict(sys.modules, {"transformers": fake_tf, "torch": fake_torch}):
        runner = HuggingFaceLayerSliceRunner(model_id="gpt2")
        runner.run_layer_range(
            model=None, layer_range=(1, 3),
            activation=np.zeros((1, 2, 4), dtype=np.float32),
            privacy_tier=None, is_final_stage=False,
        )

    # transformer.h was iterated (2 layers in range)
    assert fake_layer.call_count == 2

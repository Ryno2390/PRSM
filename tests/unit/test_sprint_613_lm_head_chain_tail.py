"""Sprint 613 (Phase 2F-5d) — LM head at chain tail.

When ``is_final_stage=True``, the HF runner must apply the model's
LM head to the post-layers hidden state, producing logits as the
output activation. Polymorphic resolution:
  model.lm_head      LLaMA / Mistral / GPT-2 / Falcon
  model.embed_out    GPT-NeoX / Pythia

Tests:
  - _resolve_hf_lm_head finds .lm_head when present
  - _resolve_hf_lm_head finds .embed_out fallback
  - Raises StageExecutionError when neither is present
  - is_final_stage=True invokes lm_head on output of last layer
  - is_final_stage=False does NOT invoke lm_head
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_resolve_lm_head_finds_lm_head():
    from prsm.node.chain_executor_adapters import _resolve_hf_lm_head
    model = MagicMock()
    model.lm_head = MagicMock(name="LMHeadLayer")
    model.embed_out = None
    head = _resolve_hf_lm_head(model)
    assert head is model.lm_head


def test_resolve_lm_head_falls_back_to_embed_out():
    from prsm.node.chain_executor_adapters import _resolve_hf_lm_head
    model = MagicMock()
    model.lm_head = None
    model.embed_out = MagicMock(name="EmbedOutLayer")
    head = _resolve_hf_lm_head(model)
    assert head is model.embed_out


def test_resolve_lm_head_raises_when_neither_present():
    from prsm.node.chain_executor_adapters import (
        _resolve_hf_lm_head, StageExecutionError,
    )
    model = MagicMock(spec=[])
    with pytest.raises(StageExecutionError, match="lm_head|embed_out"):
        _resolve_hf_lm_head(model)


def _make_fake_hf_with_lm_head():
    fake_layer = MagicMock(return_value=(MagicMock(),))
    fake_lm_head = MagicMock()
    fake_lm_head_output = MagicMock()
    fake_lm_head_output.dim = MagicMock(return_value=3)
    fake_lm_head_output.squeeze = MagicMock(return_value=fake_lm_head_output)
    fake_lm_head_output.cpu = MagicMock()
    fake_lm_head_output.cpu.return_value.numpy = MagicMock(
        return_value=np.zeros((1, 2, 100), dtype=np.float32),
    )
    fake_lm_head.return_value = fake_lm_head_output

    fake_model = MagicMock()
    fake_model.model = MagicMock()
    fake_model.model.layers = [fake_layer for _ in range(4)]
    fake_model.lm_head = fake_lm_head
    fake_model.embed_out = None
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

    return fake_tf, fake_torch, fake_model, fake_lm_head


def test_final_stage_true_invokes_lm_head():
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner

    fake_tf, fake_torch, fake_model, fake_lm_head = _make_fake_hf_with_lm_head()
    with patch.dict(sys.modules, {
        "transformers": fake_tf, "torch": fake_torch,
    }):
        runner = HuggingFaceLayerSliceRunner(model_id="m")
        activation = np.zeros((1, 2, 16), dtype=np.float32)
        runner.run_layer_range(
            model=None, layer_range=(0, 4),
            activation=activation, privacy_tier=None,
            is_final_stage=True,
        )
    fake_lm_head.assert_called_once()


def test_final_stage_false_does_NOT_invoke_lm_head():
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner

    fake_tf, fake_torch, fake_model, fake_lm_head = _make_fake_hf_with_lm_head()
    with patch.dict(sys.modules, {
        "transformers": fake_tf, "torch": fake_torch,
    }):
        runner = HuggingFaceLayerSliceRunner(model_id="m")
        activation = np.zeros((1, 2, 16), dtype=np.float32)
        runner.run_layer_range(
            model=None, layer_range=(0, 4),
            activation=activation, privacy_tier=None,
            is_final_stage=False,
        )
    fake_lm_head.assert_not_called()

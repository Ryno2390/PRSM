"""Sprint 610 (Phase 2F-5a) — HuggingFaceLayerSliceRunner skeleton.

First real-model LayerSliceRunner shape. Phase 2F-5a ships:
  - Class signature + constructor accepting model_id + device
  - Lazy transformers import in run_layer_range (no import-time cost
    when daemon is on identity / stub runner)
  - run_layer_range raises with structured Phase-2F-5b hint
  - Conforms to LayerSliceRunner Protocol shape

Phase 2F-5b: real model loading from HuggingFace hub or local
checkpoint + forward pass through assigned layers.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_module_exposes_huggingface_runner():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "HuggingFaceLayerSliceRunner")


def test_constructor_accepts_model_id():
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner
    runner = HuggingFaceLayerSliceRunner(model_id="meta-llama/Llama-3.2-1B")
    assert runner.model_id == "meta-llama/Llama-3.2-1B"


def test_constructor_accepts_device():
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner
    runner = HuggingFaceLayerSliceRunner(
        model_id="test/model", device="cpu",
    )
    assert runner.device == "cpu"


def test_constructor_default_device_is_cpu():
    """CPU default keeps the runner construction safe on
    machines without GPUs.
    """
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner
    runner = HuggingFaceLayerSliceRunner(model_id="test/model")
    assert runner.device == "cpu"


def test_runner_has_run_layer_range_method():
    """Conforms to LayerSliceRunner Protocol shape."""
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner
    runner = HuggingFaceLayerSliceRunner(model_id="test/model")
    assert hasattr(runner, "run_layer_range")
    assert callable(runner.run_layer_range)


def test_run_layer_range_raises_phase_2f5b_hint():
    """Phase 2F-5a is skeleton only — invocation must raise with
    actionable Phase-2F-5b pending message so operators trying to
    use the runner see what's missing.
    """
    from prsm.node.chain_executor_adapters import (
        HuggingFaceLayerSliceRunner, StageExecutionError,
    )
    runner = HuggingFaceLayerSliceRunner(model_id="test/model")
    with pytest.raises(StageExecutionError) as exc_info:
        runner.run_layer_range(
            model=None,
            layer_range=(0, 1),
            activation=np.zeros(3, dtype=np.float32),
            privacy_tier=None,
            is_final_stage=False,
        )
    msg = str(exc_info.value).lower()
    assert "phase 2f-5b" in msg or "huggingface" in msg

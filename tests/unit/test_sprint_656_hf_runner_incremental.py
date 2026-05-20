"""Sprint 656 (KV-cache arc piece 3) — HuggingFaceLayerSliceRunner
gains `run_layer_range_incremental` with DynamicCache."""
from __future__ import annotations

import os
import pytest


os.environ.setdefault("HF_HUB_OFFLINE", "1")


def test_method_exists_on_runner():
    from prsm.node.chain_executor_adapters import (
        HuggingFaceLayerSliceRunner,
    )
    assert hasattr(
        HuggingFaceLayerSliceRunner, "run_layer_range_incremental",
    )


def test_runner_returns_tuple_of_result_and_cache():
    import numpy as np
    from prsm.node.chain_executor_adapters import (
        HuggingFaceLayerSliceRunner,
    )

    runner = HuggingFaceLayerSliceRunner(
        model_id="gpt2", device="cpu",
    )
    try:
        runner._ensure_model_loaded()
    except Exception as exc:
        pytest.skip(f"gpt2 not available offline: {exc}")
    from transformers.cache_utils import DynamicCache

    rng = np.random.default_rng(7)
    activation = rng.standard_normal((1, 2, 768)).astype(np.float32) * 0.1

    result = runner.run_layer_range_incremental(
        model=None, layer_range=(0, 12),
        activation=activation,
        privacy_tier=None, is_final_stage=False,
        prev_kv_state=None,
    )
    assert isinstance(result, tuple) and len(result) == 2
    layer_result, cache = result
    assert hasattr(layer_result, "output")
    assert isinstance(cache, DynamicCache)
    # 2 positions were forwarded → cache should hold 2 positions
    assert cache.get_seq_length() == 2


def test_hot_cache_grows_seq_len_by_one():
    import numpy as np
    from prsm.node.chain_executor_adapters import (
        HuggingFaceLayerSliceRunner,
    )

    runner = HuggingFaceLayerSliceRunner(
        model_id="gpt2", device="cpu",
    )
    try:
        runner._ensure_model_loaded()
    except Exception as exc:
        pytest.skip(f"gpt2 not available offline: {exc}")

    rng = np.random.default_rng(11)
    cold_act = rng.standard_normal((1, 2, 768)).astype(np.float32) * 0.1
    _, cold_cache = runner.run_layer_range_incremental(
        model=None, layer_range=(0, 12),
        activation=cold_act,
        privacy_tier=None, is_final_stage=False,
        prev_kv_state=None,
    )
    assert cold_cache.get_seq_length() == 2

    hot_act = rng.standard_normal((1, 1, 768)).astype(np.float32) * 0.1
    _, hot_cache = runner.run_layer_range_incremental(
        model=None, layer_range=(0, 12),
        activation=hot_act,
        privacy_tier=None, is_final_stage=False,
        prev_kv_state=cold_cache,
    )
    assert hot_cache.get_seq_length() == 3


def test_final_stage_applies_ln_f_and_lm_head():
    import numpy as np
    from prsm.node.chain_executor_adapters import (
        HuggingFaceLayerSliceRunner,
    )

    runner = HuggingFaceLayerSliceRunner(
        model_id="gpt2", device="cpu",
    )
    try:
        runner._ensure_model_loaded()
    except Exception as exc:
        pytest.skip(f"gpt2 not available offline: {exc}")

    rng = np.random.default_rng(13)
    activation = rng.standard_normal((1, 2, 768)).astype(np.float32) * 0.1
    result, _ = runner.run_layer_range_incremental(
        model=None, layer_range=(0, 12),
        activation=activation,
        privacy_tier=None, is_final_stage=True,
        prev_kv_state=None,
    )
    assert result.output.shape == (1, 2, 50257)

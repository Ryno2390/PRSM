"""Sprint 615 (Phase 2F-5f) — prompt_encoder wired into _build_chain_executor.

Source-grep tests for the env-var wiring inside _build_chain_executor's
rpc branch. Full integration is exercised by deploying with the env
set + invoking chain_executor.execute_chain.
"""
from __future__ import annotations

import inspect


def test_chain_executor_imports_build_hf_prompt_encoder():
    """The rpc branch must import + use the sprint-614 factory."""
    from prsm.node.inference_wiring import _build_chain_executor
    src = inspect.getsource(_build_chain_executor)
    assert "build_hf_prompt_encoder" in src, (
        "Sprint 615: _build_chain_executor rpc branch must call "
        "build_hf_prompt_encoder when PRSM_PARALLAX_PROMPT_ENCODER_KIND=huggingface"
    )


def test_chain_executor_reads_env_var():
    from prsm.node.inference_wiring import _build_chain_executor
    src = inspect.getsource(_build_chain_executor)
    assert "PRSM_PARALLAX_PROMPT_ENCODER_KIND" in src
    assert "huggingface" in src


def test_chain_executor_warns_when_hf_kind_set_but_model_id_missing():
    """When PROMPT_ENCODER_KIND=huggingface but HF_MODEL_ID unset,
    the rpc branch warns + falls back to the factory's default
    encoder.
    """
    from prsm.node.inference_wiring import _build_chain_executor
    src = inspect.getsource(_build_chain_executor)
    # Source must contain the fallback warning
    assert "PRSM_PARALLAX_HF_MODEL_ID unset" in src or "HF_MODEL_ID" in src
    assert "utf8 default" in src.lower() or "default encoder" in src.lower()

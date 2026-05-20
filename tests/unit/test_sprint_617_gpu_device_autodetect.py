"""Sprint 617 (Phase 2F-5h) — GPU device auto-detect + safe fallback.

The HF runner / encoder / decoder factories all take a ``device``
arg defaulting to "cpu". Sprint 617 adds a centralized
``_resolve_hf_device(requested)`` helper that:

  - "cpu" requested → return "cpu" (no question)
  - "cuda" requested + torch.cuda.is_available() → return "cuda"
  - "cuda" requested + NOT available → log WARNING + return "cpu"
  - "auto" or "" → "cuda" if available else "cpu"

Same helper is invoked by all three factory paths (runner / encoder /
decoder) so operators flipping PRSM_PARALLAX_HF_DEVICE=auto get
sane defaults across the entire pipeline.

Tests use a mock torch.cuda module so CI doesn't actually need GPU.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _install_fake_torch(cuda_available: bool):
    fake_torch = MagicMock()
    fake_torch.cuda = MagicMock()
    fake_torch.cuda.is_available = MagicMock(return_value=cuda_available)
    return fake_torch


def test_helper_exposed():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "_resolve_hf_device")


def test_explicit_cpu_returned_unchanged():
    from prsm.node.chain_executor_adapters import _resolve_hf_device
    # Even with cuda available, "cpu" request honored
    with patch.dict(sys.modules, {"torch": _install_fake_torch(True)}):
        assert _resolve_hf_device("cpu") == "cpu"


def test_cuda_when_available_returned_unchanged():
    from prsm.node.chain_executor_adapters import _resolve_hf_device
    with patch.dict(sys.modules, {"torch": _install_fake_torch(True)}):
        assert _resolve_hf_device("cuda") == "cuda"


def test_cuda_when_unavailable_falls_back_to_cpu_with_warning(caplog):
    from prsm.node.chain_executor_adapters import _resolve_hf_device
    with patch.dict(sys.modules, {"torch": _install_fake_torch(False)}):
        with caplog.at_level("WARNING"):
            result = _resolve_hf_device("cuda")
    assert result == "cpu"
    assert any(
        "cuda" in r.getMessage().lower()
        and "fall" in r.getMessage().lower()
        for r in caplog.records
    )


def test_auto_picks_cuda_when_available():
    from prsm.node.chain_executor_adapters import _resolve_hf_device
    with patch.dict(sys.modules, {"torch": _install_fake_torch(True)}):
        assert _resolve_hf_device("auto") == "cuda"


def test_auto_falls_back_to_cpu_when_no_cuda():
    from prsm.node.chain_executor_adapters import _resolve_hf_device
    with patch.dict(sys.modules, {"torch": _install_fake_torch(False)}):
        assert _resolve_hf_device("auto") == "cpu"


def test_empty_string_treated_as_auto():
    """PRSM_PARALLAX_HF_DEVICE unset (caller passes "") → auto-detect."""
    from prsm.node.chain_executor_adapters import _resolve_hf_device
    with patch.dict(sys.modules, {"torch": _install_fake_torch(True)}):
        assert _resolve_hf_device("") == "cuda"
    with patch.dict(sys.modules, {"torch": _install_fake_torch(False)}):
        assert _resolve_hf_device("") == "cpu"


def test_torch_not_installed_returns_cpu():
    """No torch available → can't claim cuda; return cpu safely."""
    from prsm.node.chain_executor_adapters import _resolve_hf_device
    # Patch torch to raise ImportError on attribute access
    bad_torch = MagicMock()
    bad_torch.cuda = MagicMock()
    bad_torch.cuda.is_available = MagicMock(
        side_effect=ImportError("torch broken"),
    )
    with patch.dict(sys.modules, {"torch": bad_torch}):
        assert _resolve_hf_device("auto") == "cpu"


def test_runner_uses_resolve_hf_device():
    """HuggingFaceLayerSliceRunner constructor must funnel through
    _resolve_hf_device so the auto/fallback logic applies.
    """
    import inspect
    from prsm.node.chain_executor_adapters import HuggingFaceLayerSliceRunner
    src = inspect.getsource(HuggingFaceLayerSliceRunner)
    assert "_resolve_hf_device" in src


def test_prompt_encoder_uses_resolve_hf_device():
    import inspect
    from prsm.node.chain_executor_adapters import build_hf_prompt_encoder
    src = inspect.getsource(build_hf_prompt_encoder)
    assert "_resolve_hf_device" in src

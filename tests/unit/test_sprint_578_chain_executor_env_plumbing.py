"""Sprint 578 — chain_executor env-driven plumbing.

Sprint 558 wired _StubChainExecutor with this note in the source:
  "The real chain executor comes from sprint 546's
   make_rpc_chain_executor factory in a future sprint."

Sprint 578 = Phase 1 plumbing, mirror of sprints 576/577.
New _build_chain_executor(node) helper switches on
PRSM_PARALLAX_CHAIN_EXECUTOR_KIND:
  unset / "stub" → _StubChainExecutor (current default; raises
                    on execute_chain so caller must check pool
                    first — sprint 558 invariant).
  "rpc"          → Phase 2 wires real via make_rpc_chain_executor
                    using node.identity/transport/anchor. Phase 1
                    logs WARNING + falls back to stub.
  anything else  → WARNING + stub fallback.

build_parallax_executor_or_none calls helper instead of hardcoding.
Phase 2 (real RPC chain executor) becomes additive in the helper.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


def test_default_kind_stub_returns_stub_executor():
    """Unset env → _StubChainExecutor."""
    from prsm.node.inference_wiring import (
        _build_chain_executor,
        _StubChainExecutor,
    )
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PRSM_PARALLAX_CHAIN_EXECUTOR_KIND", None)
        exe = _build_chain_executor(MagicMock())
    assert isinstance(exe, _StubChainExecutor)


def test_explicit_stub_kind_is_alias():
    from prsm.node.inference_wiring import (
        _build_chain_executor,
        _StubChainExecutor,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_CHAIN_EXECUTOR_KIND": "stub"},
        clear=False,
    ):
        exe = _build_chain_executor(MagicMock())
    assert isinstance(exe, _StubChainExecutor)


def test_rpc_kind_falls_back_to_stub_with_warning(caplog):
    """Phase 2 will wire real RPC; Phase 1 must not crash if
    operator sets rpc early — fall back to stub + warn.
    """
    from prsm.node.inference_wiring import (
        _build_chain_executor,
        _StubChainExecutor,
    )
    with patch.dict(
        os.environ,
        {
            "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND": "rpc",
            # Sprint 629: pin sepolia (no anchor default) so the
            # rpc-fallback path is exercised. Default network is
            # mainnet which now has a baked-in anchor.
            "PRSM_NETWORK": "sepolia",
        },
        clear=False,
    ):
        os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
        with caplog.at_level("WARNING"):
            exe = _build_chain_executor(MagicMock())
    assert isinstance(exe, _StubChainExecutor)
    assert any(
        "phase 2" in r.getMessage().lower()
        or "rpc" in r.getMessage().lower()
        or "make_rpc_chain_executor" in r.getMessage()
        for r in caplog.records
    )


def test_unknown_kind_falls_back_to_stub(caplog):
    from prsm.node.inference_wiring import (
        _build_chain_executor,
        _StubChainExecutor,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_CHAIN_EXECUTOR_KIND": "bogus_xyz"},
        clear=False,
    ):
        with caplog.at_level("WARNING"):
            exe = _build_chain_executor(MagicMock())
    assert isinstance(exe, _StubChainExecutor)
    assert any(
        "bogus_xyz" in r.getMessage() or "unknown" in r.getMessage().lower()
        for r in caplog.records
    )


def test_parallax_executor_builder_uses_helper():
    """build_parallax_executor_or_none must call the helper so
    PRSM_PARALLAX_CHAIN_EXECUTOR_KIND is honored. Phase 2 wires
    the real chain executor via the same hook.
    """
    import inspect
    from prsm.node.inference_wiring import build_parallax_executor_or_none
    src = inspect.getsource(build_parallax_executor_or_none)
    assert "_build_chain_executor" in src, (
        "build_parallax_executor_or_none must call the helper"
    )

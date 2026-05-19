"""Sprint 577 — consensus_hook submitter env-driven plumbing.

Sprint 562 left consensus_hook with a hardcoded _LoggingChallengeSubmitter:

  consensus_hook=ConsensusMismatchHook(
      submitter=_LoggingChallengeSubmitter(),
      sample_rate=0.0,
  ),

The deferral note said: "on-chain dispatch via Phase 7.1x
ConsensusChallengeSubmitter deferred pending translation layer
(ChallengeRecord → on-chain ABI shape)".

Sprint 577 = Phase 1 plumbing, mirror of sprint 576's pattern.
New `_build_consensus_submitter()` switches on
PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND:
  unset / "logging" → _LoggingChallengeSubmitter (current default)
  "onchain"          → Phase 2 will return real OnChainChallengeSubmitter;
                       Phase 1 logs WARNING + falls back to logging.
  anything else      → WARNING + logging fallback.

Trust-stack constructors call the helper rather than hardcoding —
Phase 2 (real on-chain dispatch) becomes a non-churn additive
change to the helper.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


def test_default_kind_logging_returns_logging_submitter():
    """Unset env → _LoggingChallengeSubmitter."""
    from prsm.node.inference_wiring import (
        _build_consensus_submitter,
        _LoggingChallengeSubmitter,
    )
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND", None)
        sub = _build_consensus_submitter()
    assert isinstance(sub, _LoggingChallengeSubmitter)


def test_explicit_logging_kind_is_explicit_alias():
    from prsm.node.inference_wiring import (
        _build_consensus_submitter,
        _LoggingChallengeSubmitter,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND": "logging"},
        clear=False,
    ):
        sub = _build_consensus_submitter()
    assert isinstance(sub, _LoggingChallengeSubmitter)


def test_onchain_kind_falls_back_to_logging_with_warning(caplog):
    """Phase 2 will implement on-chain; Phase 1 must NOT crash if
    an operator sets `onchain` early — fall back to logging + warn.
    """
    from prsm.node.inference_wiring import (
        _build_consensus_submitter,
        _LoggingChallengeSubmitter,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND": "onchain"},
        clear=False,
    ):
        with caplog.at_level("WARNING"):
            sub = _build_consensus_submitter()
    assert isinstance(sub, _LoggingChallengeSubmitter)
    # Warning about Phase 2 deferral
    assert any(
        "phase 2" in r.getMessage().lower()
        or "onchain" in r.getMessage().lower()
        or "translation layer" in r.getMessage().lower()
        for r in caplog.records
    )


def test_unknown_kind_falls_back_to_logging(caplog):
    from prsm.node.inference_wiring import (
        _build_consensus_submitter,
        _LoggingChallengeSubmitter,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND": "bogus_kind_xyz"},
        clear=False,
    ):
        with caplog.at_level("WARNING"):
            sub = _build_consensus_submitter()
    assert isinstance(sub, _LoggingChallengeSubmitter)
    assert any(
        "bogus_kind_xyz" in r.getMessage() or "unknown" in r.getMessage().lower()
        for r in caplog.records
    )


def test_production_trust_stack_uses_helper():
    """Production trust-stack constructor must call the helper
    so future kinds activate without churning that constructor.
    """
    import inspect
    from prsm.node.inference_wiring import (
        _build_production_trust_stack_or_none,
    )
    src = inspect.getsource(_build_production_trust_stack_or_none)
    assert "_build_consensus_submitter" in src, (
        "_build_production_trust_stack_or_none must call the helper "
        "so PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND env is honored"
    )


def test_mock_trust_stack_uses_helper():
    """Mock trust-stack constructor likewise. Keeps both paths in
    sync so operators switching trust_stack_kind don't see a
    submitter behavior gap.
    """
    import inspect
    from prsm.node.inference_wiring import _build_mock_trust_stack
    src = inspect.getsource(_build_mock_trust_stack)
    assert "_build_consensus_submitter" in src, (
        "_build_mock_trust_stack must also call the helper"
    )

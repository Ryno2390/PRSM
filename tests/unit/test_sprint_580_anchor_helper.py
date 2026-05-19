"""Sprint 580 — _build_anchor_or_none module helper.

Pre-580 the PublisherKeyAnchorClient construction lived inside
``_build_production_trust_stack_or_none``. That coupling makes the
chain_executor Phase 2 wiring (sprint 581+) awkward — the chain
executor's factory needs ``anchor`` as a required arg, but it would
have to either:
  (a) reach into the trust-stack constructor's internals, or
  (b) duplicate the anchor-construction code.

Sprint 580 extracts a module-level ``_build_anchor_or_none()``
helper. Same env-driven semantics as the inline version:
  - PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS unset → return None
  - construction fails  → log + return None
  - construction succeeds → return PublisherKeyAnchorClient

Both ``_build_production_trust_stack_or_none`` (existing caller)
and ``_build_chain_executor`` (Phase 2) call this helper. Pure
refactor; no behavior change.
"""
from __future__ import annotations

import os
from unittest.mock import patch


def test_anchor_helper_returns_none_when_env_unset():
    """No env → None (no daemon failure, just absent anchor)."""
    from prsm.node.inference_wiring import _build_anchor_or_none
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
        anchor = _build_anchor_or_none()
    assert anchor is None


def test_anchor_helper_returns_client_on_success():
    """Env set + construction succeeds → client instance.

    Mock the import so we don't touch real RPC.
    """
    from unittest.mock import MagicMock
    fake_client = MagicMock()
    fake_module = MagicMock()
    fake_module.PublisherKeyAnchorClient = MagicMock(
        return_value=fake_client,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xdeadbeef"},
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {"prsm.security.publisher_key_anchor.client": fake_module},
        ):
            from prsm.node.inference_wiring import _build_anchor_or_none
            anchor = _build_anchor_or_none()
    assert anchor is fake_client


def test_anchor_helper_returns_none_on_construction_failure(caplog):
    """Env set but constructor raises → log warning + return None."""
    from unittest.mock import MagicMock
    fake_module = MagicMock()
    fake_module.PublisherKeyAnchorClient = MagicMock(
        side_effect=RuntimeError("rpc failure"),
    )
    with patch.dict(
        os.environ,
        {"PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xdeadbeef"},
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {"prsm.security.publisher_key_anchor.client": fake_module},
        ):
            from prsm.node.inference_wiring import _build_anchor_or_none
            with caplog.at_level("WARNING"):
                anchor = _build_anchor_or_none()
    assert anchor is None
    # Warning surfaces the failure mode
    assert any(
        "rpc failure" in r.getMessage() or "construction failed" in r.getMessage()
        for r in caplog.records
    )


def test_production_trust_stack_uses_anchor_helper():
    """Existing caller must route through the helper after sprint 580."""
    import inspect
    from prsm.node.inference_wiring import (
        _build_production_trust_stack_or_none,
    )
    src = inspect.getsource(_build_production_trust_stack_or_none)
    assert "_build_anchor_or_none" in src, (
        "_build_production_trust_stack_or_none must call helper "
        "after sprint-580 refactor"
    )

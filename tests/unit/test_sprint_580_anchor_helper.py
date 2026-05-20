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
    """No env AND no networks.py default → None.

    Sprint 629 changed the semantics: env unset on a network WITH a
    networks.py default (Base mainnet) now returns the deployed-anchor
    client. To assert the absent-anchor branch, this test pins
    PRSM_NETWORK=sepolia (no default → None is correct).
    """
    from prsm.node.inference_wiring import _build_anchor_or_none
    with patch.dict(os.environ, {"PRSM_NETWORK": "sepolia"}, clear=False):
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


# --------------------------------------------------------------------------
# Sprint 629 — networks.py fallback
# --------------------------------------------------------------------------


def test_anchor_helper_uses_networks_py_default_on_mainnet():
    """Sprint 629 fix: env unset on Base mainnet → use networks.py default.

    Pre-629 the helper read PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS env
    directly; if unset, it returned None even though
    `prsm.config.networks` ships a baked-in default for Base mainnet
    (Phase 3.x.3 deploy 0xd811ad9986f44f404b0fd992168a7cc76206df03 at
    block 46248175). Result: any operator on a fresh install who set
    PRSM_PARALLAX_TRUST_STACK_KIND=production but didn't manually
    pin the anchor env silently fell back to the stub anchor —
    production trust-stack disabled without a visible error.

    Sprint 629 routes the helper through `resolve_endpoints()` so the
    networks.py fallback is honored. Operators get the deployed anchor
    by default; explicit env still overrides via the existing
    `_override("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", ...)` plumbing.
    """
    from unittest.mock import MagicMock
    fake_client = MagicMock()
    fake_module = MagicMock()
    fake_module.PublisherKeyAnchorClient = MagicMock(
        return_value=fake_client,
    )
    with patch.dict(
        os.environ,
        {"PRSM_NETWORK": "mainnet"},
        clear=False,
    ):
        # Critical: env unset for the per-field override
        os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
        with patch.dict(
            "sys.modules",
            {"prsm.security.publisher_key_anchor.client": fake_module},
        ):
            from prsm.node.inference_wiring import _build_anchor_or_none
            anchor = _build_anchor_or_none()
    # Pre-629 behavior: anchor is None (bug).
    # Post-629 behavior: networks.py default kicks in → real client.
    assert anchor is fake_client, (
        "_build_anchor_or_none must honor networks.py default for the "
        "configured network when env is unset (sprint 629 fix)"
    )
    # Confirm the constructor was called with the Base mainnet default
    # address (sprint 621 deploy).
    call_kwargs = fake_module.PublisherKeyAnchorClient.call_args.kwargs
    assert call_kwargs.get("contract_address", "").lower() == (
        "0xd811ad9986f44f404b0fd992168a7cc76206df03"
    )


def test_anchor_helper_returns_none_when_network_has_no_default():
    """No env + no networks.py default → still None (sepolia case).

    Sprint 629 must not silently fabricate an anchor for networks
    that don't have one yet. Base Sepolia's networks.py entry has
    publisher_key_anchor=None — the helper must return None there
    so production-trust-stack callers correctly diagnose "not
    deployed yet."
    """
    with patch.dict(
        os.environ,
        {"PRSM_NETWORK": "sepolia"},
        clear=False,
    ):
        os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
        from prsm.node.inference_wiring import _build_anchor_or_none
        anchor = _build_anchor_or_none()
    assert anchor is None

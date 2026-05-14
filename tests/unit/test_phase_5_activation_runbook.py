"""Sprint 421 — Phase 5 activation runbook pin.

The runbook at `docs/operations/phase-5-fiat-surface-
activation-runbook.md` is the operator-facing step-by-
step for activating the Phase 5 fiat surface when
commission gates clear (Coinbase CDP keys, KYC vendor,
Aerodrome pool seed, Foundation funding).

This pin keeps the runbook tied to the canonical env-var
set in `prsm/economy/web3/fiat_surface_health.py` — if
the codebase adds new required env vars or renames
existing ones, the runbook tests fire to keep the doc
current.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = (
    REPO_ROOT / "docs" / "operations"
    / "phase-5-fiat-surface-activation-runbook.md"
)


def _read():
    return RUNBOOK.read_text()


# ── Presence + structure ─────────────────────────────────


def test_runbook_exists():
    assert RUNBOOK.is_file(), (
        f"Phase 5 activation runbook missing at {RUNBOOK}"
    )


def test_runbook_has_pre_flight_section():
    text = _read()
    assert "Pre-flight" in text
    # Names all 5 pre-flight gates
    for gate in (
        "Foundation funding",
        "Coinbase CDP",
        "KYC vendor",
        "Aerodrome",
        "Base mainnet RPC",
    ):
        assert gate in text, f"pre-flight gate missing: {gate}"


def test_runbook_has_rollback_path():
    text = _read()
    assert "Rollback" in text
    # Non-destructive rollback specifically
    assert "non-destructive" in text.lower()


def test_runbook_documents_smoke_tests():
    text = _read()
    # Names each canonical surface
    for surface in (
        "Aerodrome quote",
        "Coinbase WaaS",
        "KYC vendor handshake",
        "Onramp full path",
    ):
        assert surface in text, f"smoke test missing: {surface}"


# ── Canonical env-var pins ───────────────────────────────


def test_runbook_documents_kyc_env_vars():
    """The sprint-282/283 KYC env vars must be in the
    runbook — if `fiat_surface_health.py` or `kyc_client.py`
    add new ones, this test fires to update the doc."""
    text = _read()
    for var in (
        "KYC_VENDOR",
        "KYC_VENDOR_API_KEY",
        "PRSM_KYC_STORE_DIR",
        "PRSM_FIAT_COMPLIANCE_LOG_DIR",
    ):
        assert var in text, f"KYC env var missing: {var}"


def test_runbook_documents_coinbase_env_vars():
    text = _read()
    for var in (
        "COINBASE_CDP_API_KEY_NAME",
        "COINBASE_CDP_API_KEY_PRIVATE",
        "COINBASE_CDP_PAYMASTER_ENDPOINT",
        "COINBASE_CDP_PAYMASTER_API_KEY",
        "PRSM_PAYMASTER_POLICY_ID",
    ):
        assert var in text, f"Coinbase env var missing: {var}"


def test_runbook_documents_aerodrome_env_vars():
    text = _read()
    for var in (
        "BASE_RPC_URL",
        "AERODROME_USDC_FTNS_POOL_ADDRESS",
    ):
        assert var in text, f"Aerodrome env var missing: {var}"


def test_runbook_documents_per_vendor_webhook_secrets():
    """Sprint 283's webhook signature verification requires
    a vendor-specific secret. The runbook must name all 3
    in `_vendor_secret_var()`."""
    text = _read()
    for var in (
        "PERSONA_WEBHOOK_SECRET",
        "ONFIDO_WEBHOOK_TOKEN",
        "PLAID_WEBHOOK_SECRET",
    ):
        assert var in text, f"webhook-secret var missing: {var}"


def test_runbook_warns_against_bypass_vars_in_prod():
    """Sprint 285's health check fires ERROR on these in
    a commissioned env. Runbook must NAME them + WARN."""
    text = _read()
    for var in (
        "PRSM_FIAT_HEALTH_CHECK_BYPASS",
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED",
    ):
        assert var in text
    # And the runbook must specifically warn against
    # production use
    assert (
        "do not set" in text.lower()
        or "test/dev environments only" in text.lower()
    )


# ── Source-truth-parity with fiat_surface_health.py ──────


def test_runbook_matches_vendor_secret_var_dispatch():
    """Sprint 283's `_vendor_secret_var()` maps each
    vendor name to its webhook-secret env var. The runbook
    must reference the canonical dispatch. If the dispatch
    is expanded (e.g., new vendor added), this test fires."""
    from prsm.economy.web3.fiat_surface_health import (
        _vendor_secret_var,
    )
    text = _read()
    for vendor in ("persona", "onfido", "plaid"):
        secret_var = _vendor_secret_var(vendor)
        assert secret_var, (
            f"vendor {vendor} has no secret_var in dispatch — "
            "either the test is stale or the source dropped support"
        )
        assert secret_var in text, (
            f"runbook missing {vendor}'s canonical "
            f"webhook secret {secret_var}"
        )


def test_runbook_documents_check_fiat_surface_health_invocation():
    """The runbook MUST show operators how to invoke the
    sprint-285 health check programmatically — this is
    the canonical pre/post-activation verification step."""
    text = _read()
    assert "check_fiat_surface_health" in text

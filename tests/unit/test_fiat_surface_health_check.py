"""Sprint 286 — fiat-surface startup health-check.

Closes a real correctness gap: sprints 283/284 built the
webhook security primitives, but they're opt-in via env-var
presence. Without this check an operator can commission KYC
(KYC_VENDOR + KYC_VENDOR_API_KEY set) and forget to set
PERSONA_WEBHOOK_SECRET, in which case sprint-283's
pass-through default lets any HTTP caller flip status to
VERIFIED.

Health-check enumerates known dangerous combinations and
returns Findings (severity + cause + remediation). Severity
levels:
  ERROR    — combination unsafe in production; refuse to
              start by default
  WARN     — combination probably unsafe; log loudly but
              proceed
  INFO     — observation only (e.g., not commissioned yet)

Operator override: PRSM_FIAT_HEALTH_CHECK_BYPASS=1 forces
proceed even on ERROR (dev/staging escape hatch). Mirrors the
PRSM_KYC_WEBHOOK_VERIFY_DISABLED pattern.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.fiat_surface_health import (
    Finding,
    FindingSeverity,
    check_fiat_surface_health,
)


# ── Empty environment: all uncommissioned ────────────────


def test_empty_env_no_errors():
    """Nothing commissioned → no ERROR findings; possibly
    INFO/WARN that surfaces are unwired."""
    findings = check_fiat_surface_health(env={})
    errors = [f for f in findings if f.severity == FindingSeverity.ERROR]
    assert errors == []


# ── KYC commissioned but no webhook secret ───────────────


def test_kyc_commissioned_persona_secret_missing():
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        # PERSONA_WEBHOOK_SECRET deliberately absent
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings if f.severity == FindingSeverity.ERROR
    ]
    assert len(errors) == 1
    assert "persona_webhook_secret" in errors[0].cause.lower()
    assert "verified" in errors[0].remediation.lower() \
        or "PERSONA_WEBHOOK_SECRET" in errors[0].remediation


def test_kyc_commissioned_onfido_secret_missing():
    env = {
        "KYC_VENDOR": "onfido",
        "KYC_VENDOR_API_KEY": "k",
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings if f.severity == FindingSeverity.ERROR
    ]
    assert len(errors) == 1
    assert "onfido_webhook_token" in errors[0].cause.lower()


def test_kyc_commissioned_with_correct_secret_no_error():
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        "PERSONA_WEBHOOK_SECRET": "wh_secret",
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings if f.severity == FindingSeverity.ERROR
    ]
    assert errors == []


def test_kyc_commissioned_wrong_vendor_secret_does_not_satisfy():
    """KYC_VENDOR=persona but only ONFIDO_WEBHOOK_TOKEN is set
    — that secret protects the wrong path; Persona webhook
    still pass-through-vulnerable."""
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        "ONFIDO_WEBHOOK_TOKEN": "wrong_path",
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings if f.severity == FindingSeverity.ERROR
    ]
    assert len(errors) == 1
    assert "persona" in errors[0].cause.lower()


# ── Compliance ring missing in commissioned env ──────────


def test_kyc_commissioned_compliance_dir_unset():
    """Commission state with no PRSM_FIAT_COMPLIANCE_LOG_DIR
    means audit data lives in-memory only. For regulators
    expecting 5-7yr retention this is a high-severity
    misconfig — WARN (not ERROR) because compliance is a
    legal obligation, not a security one. Operator may
    accept the risk in pre-production."""
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        "PERSONA_WEBHOOK_SECRET": "wh_secret",
        # PRSM_FIAT_COMPLIANCE_LOG_DIR deliberately absent
    }
    findings = check_fiat_surface_health(env=env)
    warns = [
        f for f in findings if f.severity == FindingSeverity.WARN
    ]
    compliance_warns = [
        f for f in warns if "compliance" in f.cause.lower()
    ]
    assert len(compliance_warns) == 1


def test_compliance_dir_set_no_warn():
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        "PERSONA_WEBHOOK_SECRET": "wh_secret",
        "PRSM_FIAT_COMPLIANCE_LOG_DIR": "/var/lib/prsm/compliance",
    }
    findings = check_fiat_surface_health(env=env)
    compliance_warns = [
        f for f in findings if "compliance" in f.cause.lower()
    ]
    assert compliance_warns == []


# ── Jurisdiction missing ─────────────────────────────────


def test_kyc_commissioned_no_jurisdiction():
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        "PERSONA_WEBHOOK_SECRET": "wh_secret",
        "PRSM_FIAT_COMPLIANCE_LOG_DIR": "/tmp/x",
    }
    findings = check_fiat_surface_health(env=env)
    warns = [
        f for f in findings
        if f.severity == FindingSeverity.WARN
        and "jurisdiction" in f.cause.lower()
    ]
    assert len(warns) == 1


# ── Webhook verify disabled is an ERROR in prod ──────────


def test_verify_disabled_with_commissioned_is_error():
    """PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1 + commissioned KYC
    = explicit production foot-gun. ERROR severity."""
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        "PERSONA_WEBHOOK_SECRET": "wh_secret",
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED": "1",
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings
        if f.severity == FindingSeverity.ERROR
        and "verify_disabled" in f.cause.lower()
    ]
    assert len(errors) == 1


def test_verify_disabled_uncommissioned_is_only_info():
    """In dev/staging with no commission, the disable flag is
    expected; INFO not ERROR."""
    env = {
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED": "1",
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings if f.severity == FindingSeverity.ERROR
    ]
    assert errors == []


# ── Coinbase commission consistency ──────────────────────


def test_coinbase_partial_commission_is_error():
    """COINBASE_CDP_API_KEY_NAME without _PRIVATE means real
    API calls will fail — partially-configured WaaS is a
    deploy-time foot-gun."""
    env = {
        "COINBASE_CDP_API_KEY_NAME": "name",
        # PRIVATE missing
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings
        if f.severity == FindingSeverity.ERROR
        and "coinbase_cdp" in f.cause.lower()
    ]
    assert len(errors) == 1


def test_paymaster_partial_commission_is_error():
    env = {
        "COINBASE_CDP_PAYMASTER_ENDPOINT": "https://x",
        # API_KEY missing
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings
        if f.severity == FindingSeverity.ERROR
        and "paymaster" in f.cause.lower()
    ]
    assert len(errors) == 1


def test_coinbase_both_keys_set_no_error():
    env = {
        "COINBASE_CDP_API_KEY_NAME": "name",
        "COINBASE_CDP_API_KEY_PRIVATE": "priv",
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings
        if f.severity == FindingSeverity.ERROR
        and "coinbase_cdp" in f.cause.lower()
    ]
    assert errors == []


# ── Aerodrome partial config ─────────────────────────────


def test_aerodrome_pool_without_rpc_url_warn():
    """Pool address set but no BASE_RPC_URL — calls will
    fail. Operator misconfig. WARN since Aerodrome quoter is
    read-only (no security implication)."""
    env = {
        "AERODROME_USDC_FTNS_POOL_ADDRESS": "0xpool",
    }
    findings = check_fiat_surface_health(env=env)
    warns = [
        f for f in findings
        if f.severity == FindingSeverity.WARN
        and "aerodrome" in f.cause.lower()
    ]
    assert len(warns) == 1


# ── Multiple errors stack ────────────────────────────────


def test_multiple_findings_returned():
    """Worst-case combo: KYC commissioned, no webhook secret,
    no compliance dir, verify-disabled. All findings surface."""
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED": "1",
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings if f.severity == FindingSeverity.ERROR
    ]
    # At least webhook_secret missing + verify_disabled
    assert len(errors) >= 2


# ── Bypass flag suppresses errors ────────────────────────


def test_bypass_flag_demotes_errors_to_info():
    """PRSM_FIAT_HEALTH_CHECK_BYPASS=1 — for dev/staging
    where the operator KNOWS they're misconfigured. Findings
    are reported but downgraded to INFO so caller can
    proceed."""
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        "PRSM_FIAT_HEALTH_CHECK_BYPASS": "1",
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings if f.severity == FindingSeverity.ERROR
    ]
    assert errors == []
    infos = [
        f for f in findings if f.severity == FindingSeverity.INFO
    ]
    # The original ERROR finding is now an INFO with the
    # bypass marker.
    bypassed = [f for f in infos if "bypass" in f.cause.lower()]
    assert len(bypassed) >= 1


def test_bypass_flag_must_be_truthy():
    env = {
        "KYC_VENDOR": "persona",
        "KYC_VENDOR_API_KEY": "k",
        "PRSM_FIAT_HEALTH_CHECK_BYPASS": "0",
    }
    findings = check_fiat_surface_health(env=env)
    errors = [
        f for f in findings if f.severity == FindingSeverity.ERROR
    ]
    # '0' does NOT bypass
    assert len(errors) >= 1


# ── Finding round-trip ───────────────────────────────────


def test_finding_to_dict():
    f = Finding(
        severity=FindingSeverity.ERROR,
        cause="kyc_commissioned_persona_webhook_secret_missing",
        remediation="Set PERSONA_WEBHOOK_SECRET to your "
                    "Persona webhook secret.",
    )
    d = f.to_dict()
    assert d["severity"] == "ERROR"
    assert d["cause"] == (
        "kyc_commissioned_persona_webhook_secret_missing"
    )

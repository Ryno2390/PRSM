"""Sprint 286 — fiat-surface startup health-check.

Closes the dangerous-default-combo correctness gap left by
sprints 283/284. Webhook signature verification is opt-in by
env-var presence: an operator who commissions KYC but
forgets to set ``PERSONA_WEBHOOK_SECRET`` silently restores
the pre-sprint-283 pass-through behavior — any HTTP caller
can flip status to VERIFIED.

This module enumerates known dangerous combinations and
returns a list of Findings the caller (Node startup, MCP
status tool, /health endpoint) can surface to the operator.
Findings have three severities:

  ERROR   combination unsafe in production; refuse to start
          by default
  WARN    combination probably unsafe; log loudly but
          proceed
  INFO    observation only (e.g., not commissioned yet,
          bypass flag set)

Operator escape hatch: ``PRSM_FIAT_HEALTH_CHECK_BYPASS=1``
demotes all ERROR findings to INFO. Mirrors the
``PRSM_KYC_WEBHOOK_VERIFY_DISABLED`` pattern from sprint 283.

Per Vision §14 "Crypto-UX adoption barrier" mitigation: the
operator UX is as much about preventing self-foot-guns as
end-user simplicity. Loud actionable findings are the
operator-side equivalent of "no seed phrases."
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class FindingSeverity(str, Enum):
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"


@dataclass
class Finding:
    severity: FindingSeverity
    cause: str  # short machine-readable identifier
    remediation: str  # human-readable fix instructions

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Coerce Enum → string for JSON serialization
        d["severity"] = self.severity.value
        return d


_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _kyc_commissioned(env: Mapping[str, str]) -> bool:
    return bool(
        (env.get("KYC_VENDOR") or "").strip()
        and (env.get("KYC_VENDOR_API_KEY") or "").strip()
    )


def _vendor_secret_var(vendor: str) -> str:
    return {
        "persona": "PERSONA_WEBHOOK_SECRET",
        "onfido": "ONFIDO_WEBHOOK_TOKEN",
        "plaid": "PLAID_WEBHOOK_SECRET",
    }.get(vendor, "")


def check_fiat_surface_health(
    env: Mapping[str, str],
) -> List[Finding]:
    """Enumerate dangerous-combination findings for the
    current env. Returns a list (possibly empty)."""
    findings: List[Finding] = []
    bypass = _truthy(env.get("PRSM_FIAT_HEALTH_CHECK_BYPASS"))

    # ── KYC commissioned without webhook secret ──────────
    if _kyc_commissioned(env):
        vendor = (env.get("KYC_VENDOR") or "").strip().lower()
        secret_var = _vendor_secret_var(vendor)
        if secret_var and not (
            env.get(secret_var) or ""
        ).strip():
            findings.append(Finding(
                severity=FindingSeverity.ERROR,
                cause=(
                    f"kyc_commissioned_"
                    f"{secret_var.lower()}_missing"
                ),
                remediation=(
                    f"KYC_VENDOR={vendor} is commissioned but "
                    f"{secret_var} is unset. Without it, the "
                    f"sprint-283 webhook signature check "
                    f"falls back to pass-through and any "
                    f"caller can flip KYC status to VERIFIED. "
                    f"Set {secret_var} to the webhook secret "
                    f"issued by your vendor."
                ),
            ))

    # ── verify-disabled in commissioned env ──────────────
    if (
        _kyc_commissioned(env)
        and _truthy(env.get("PRSM_KYC_WEBHOOK_VERIFY_DISABLED"))
    ):
        findings.append(Finding(
            severity=FindingSeverity.ERROR,
            cause=(
                "kyc_commissioned_webhook_"
                "verify_disabled"
            ),
            remediation=(
                "PRSM_KYC_WEBHOOK_VERIFY_DISABLED is set in a "
                "commissioned KYC environment — webhook "
                "signature + replay defenses are off. Unset "
                "the env var in production."
            ),
        ))

    # ── Compliance ring persistence (audit retention) ────
    if _kyc_commissioned(env):
        if not (
            env.get("PRSM_FIAT_COMPLIANCE_LOG_DIR") or ""
        ).strip():
            findings.append(Finding(
                severity=FindingSeverity.WARN,
                cause=(
                    "fiat_compliance_log_dir_unset"
                ),
                remediation=(
                    "PRSM_FIAT_COMPLIANCE_LOG_DIR is unset in "
                    "a commissioned KYC environment — audit "
                    "events live in-memory only. Regulators "
                    "(AUSTRAC, FinCEN, IRS) expect 5-7yr "
                    "retention. Set to a persistent directory."
                ),
            ))

    # ── Jurisdiction tag ─────────────────────────────────
    if _kyc_commissioned(env):
        if not (
            env.get("PRSM_OPERATOR_JURISDICTION") or ""
        ).strip():
            findings.append(Finding(
                severity=FindingSeverity.WARN,
                cause="operator_jurisdiction_unset",
                remediation=(
                    "PRSM_OPERATOR_JURISDICTION is unset — "
                    "fiat compliance entries will lack "
                    "jurisdiction labels, complicating audit "
                    "filtering. Set to e.g. 'US-CA', 'EU-DE'."
                ),
            ))

    # ── Coinbase CDP partial commission ──────────────────
    cdp_name = (env.get("COINBASE_CDP_API_KEY_NAME") or "").strip()
    cdp_priv = (
        env.get("COINBASE_CDP_API_KEY_PRIVATE") or ""
    ).strip()
    if bool(cdp_name) != bool(cdp_priv):
        findings.append(Finding(
            severity=FindingSeverity.ERROR,
            cause="coinbase_cdp_partial_commission",
            remediation=(
                "COINBASE_CDP_API_KEY_NAME and "
                "COINBASE_CDP_API_KEY_PRIVATE must be set "
                "together. Currently one is set and the "
                "other isn't — WaaS API calls will fail."
            ),
        ))

    # ── Paymaster partial commission ─────────────────────
    pm_endpoint = (
        env.get("COINBASE_CDP_PAYMASTER_ENDPOINT") or ""
    ).strip()
    pm_key = (
        env.get("COINBASE_CDP_PAYMASTER_API_KEY") or ""
    ).strip()
    if bool(pm_endpoint) != bool(pm_key):
        findings.append(Finding(
            severity=FindingSeverity.ERROR,
            cause="coinbase_paymaster_partial_commission",
            remediation=(
                "COINBASE_CDP_PAYMASTER_ENDPOINT and "
                "COINBASE_CDP_PAYMASTER_API_KEY must be set "
                "together — gasless transfer calls will fail."
            ),
        ))

    # ── Aerodrome pool without RPC URL ───────────────────
    aero_pool = (
        env.get("AERODROME_USDC_FTNS_POOL_ADDRESS") or ""
    ).strip()
    base_rpc = (env.get("BASE_RPC_URL") or "").strip()
    if aero_pool and not base_rpc:
        findings.append(Finding(
            severity=FindingSeverity.WARN,
            cause="aerodrome_pool_without_rpc_url",
            remediation=(
                "AERODROME_USDC_FTNS_POOL_ADDRESS is set but "
                "BASE_RPC_URL is unset — pool-state RPC calls "
                "will fail. Set BASE_RPC_URL to a Base "
                "mainnet JSON-RPC endpoint."
            ),
        ))

    # ── Apply bypass: demote ERRORs to INFO ──────────────
    if bypass and findings:
        demoted: List[Finding] = []
        for f in findings:
            if f.severity == FindingSeverity.ERROR:
                demoted.append(Finding(
                    severity=FindingSeverity.INFO,
                    cause=f"bypassed:{f.cause}",
                    remediation=(
                        f"PRSM_FIAT_HEALTH_CHECK_BYPASS=1 "
                        f"demoted: {f.remediation}"
                    ),
                ))
            else:
                demoted.append(f)
        return demoted

    return findings

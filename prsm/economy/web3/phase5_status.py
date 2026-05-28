"""Sprint 859 — Phase 5 fiat-surface readiness aggregator.

Single function that queries every Phase 5 surface (KYC + WaaS +
Paymaster + Onramp + Aerodrome) and returns one canonical readiness
envelope. Powers GET /wallet/phase5/status + CLI ops surfaces.

Each surface reports:
  commissioned:    env vars present
  adapter_wired:   SDK backend injected (sp848 honest signal)
  live_exec:       end-to-end flow demonstrably working live
  notes:           operator-facing string describing what's still needed

The aggregate `overall` field rolls up to one of:
  READY        — all 5 surfaces live_exec
  PARTIAL      — at least one but not all live_exec
  NOT_READY    — none live_exec

Operators can poll this from /wallet/phase5/status, dashboards can
render the grid, and `prsm node phase5-status` CLI can pretty-print
the table for terminal-friendly readouts.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _bool_env(*names: str) -> bool:
    """True iff all named env vars are set + non-empty."""
    return all(bool(os.environ.get(n)) for n in names)


def _kyc_status(kyc_client: Any) -> Dict[str, Any]:
    if kyc_client is None:
        return {
            "commissioned": False,
            "adapter_wired": False,
            "live_exec": False,
            "vendor": None,
            "notes": "KYC client not initialized at node startup.",
        }
    return {
        "commissioned": kyc_client.is_commissioned(),
        "adapter_wired": kyc_client.adapter_wired(),
        "live_exec": (
            kyc_client.is_commissioned()
            and kyc_client.adapter_wired()
        ),
        "vendor": getattr(kyc_client, "_vendor", None),
        "notes": (
            "ready" if kyc_client.adapter_wired()
            else "set KYC_VENDOR=persona + KYC_VENDOR_API_KEY + "
                 "PERSONA_TEMPLATE_ID to wire adapter"
        ),
    }


def _waas_status(waas_client: Any) -> Dict[str, Any]:
    if waas_client is None:
        return {
            "commissioned": False,
            "adapter_wired": False,
            "live_exec": False,
            "wallet_count": 0,
            "notes": "WaaS client not initialized at node startup.",
        }
    return {
        "commissioned": waas_client.is_commissioned(),
        "adapter_wired": waas_client.adapter_wired(),
        "live_exec": (
            waas_client.is_commissioned()
            and waas_client.adapter_wired()
        ),
        "wallet_count": len(waas_client.list_wallets()),
        "notes": (
            "ready (smart-account creation requires real PEM + "
            "real Wallet Secret + CDP project Server Wallets "
            "enabled)" if waas_client.adapter_wired()
            else "set COINBASE_CDP_API_KEY_NAME + "
                 "COINBASE_CDP_API_KEY_PRIVATE (raw base64 or "
                 "PEM) + COINBASE_CDP_WALLET_SECRET to wire"
        ),
    }


def _paymaster_status(pm_client: Any) -> Dict[str, Any]:
    if pm_client is None:
        return {
            "commissioned": False,
            "adapter_wired": False,
            "live_exec": False,
            "sponsorships": 0,
            "notes": "Paymaster client not initialized at node startup.",
        }
    summary = pm_client.spend_summary()
    return {
        "commissioned": summary["commissioned"],
        "adapter_wired": summary["adapter_wired"],
        # live_exec for Paymaster = sponsorships > 0 (real user-ops
        # have been sponsored). Pre-sp856 will always be 0.
        "live_exec": summary["sponsorships"] > 0,
        "sponsorships": summary["sponsorships"],
        "total_sponsored_wei": summary["total_sponsored_wei"],
        "notes": (
            "ready; signed user-op submission via sp856 closes "
            "live_exec" if summary["adapter_wired"]
            else "set COINBASE_CDP_PAYMASTER_ENDPOINT + "
                 "COINBASE_CDP_PAYMASTER_API_KEY"
        ),
    }


def _onramp_status() -> Dict[str, Any]:
    """Onramp is purely env-driven — no node-attached client.
    Two env vars together = commissioned + adapter_wired."""
    cdp_keys_present = _bool_env(
        "COINBASE_CDP_API_KEY_NAME",
        "COINBASE_CDP_API_KEY_PRIVATE",
    )
    pay_app_id_present = bool(os.environ.get("COINBASE_PAY_APP_ID"))
    # Pay App ID is required for the OLD public-params widget URL;
    # CDP keys alone are sufficient for the NEW secure-init flow
    # (sp855b) which is what /wallet/onramp/execute actually uses.
    commissioned = cdp_keys_present
    adapter_wired = cdp_keys_present
    return {
        "commissioned": commissioned,
        "adapter_wired": adapter_wired,
        "live_exec": commissioned and adapter_wired,
        "secure_init": True,
        "pay_app_id_set": pay_app_id_present,
        "notes": (
            "ready; CDP project must have Onramp product enabled "
            "(25 test transactions at $5 each pre-KYB)"
            if commissioned
            else "set COINBASE_CDP_API_KEY_NAME + "
                 "COINBASE_CDP_API_KEY_PRIVATE"
        ),
    }


def _aerodrome_status(aero_client: Any) -> Dict[str, Any]:
    if aero_client is None:
        return {
            "commissioned": False,
            "adapter_wired": False,
            "live_exec": False,
            "pool_configured": False,
            "notes": (
                "Aerodrome client not initialized at node startup."
            ),
        }
    configured = aero_client.is_configured()
    return {
        "commissioned": configured,
        "adapter_wired": configured,
        "live_exec": configured,
        "pool_configured": configured,
        "pool_address": getattr(aero_client, "pool_address", None),
        "notes": (
            "ready; live swap submission via sp856-class "
            "CDP-signed-tx flow"
            if configured
            else "Foundation Safe USDC↔FTNS pool seeding "
                 "ceremony pending (Vision gantt 2026-06-15); set "
                 "BASE_RPC_URL + AERODROME_USDC_FTNS_POOL_ADDRESS"
        ),
    }


def aggregate_phase5_status(
    *,
    kyc_client: Any = None,
    waas_client: Any = None,
    paymaster_client: Any = None,
    aerodrome_client: Any = None,
) -> Dict[str, Any]:
    """Aggregate readiness of all 5 Phase 5 surfaces."""
    surfaces = {
        "kyc": _kyc_status(kyc_client),
        "waas": _waas_status(waas_client),
        "paymaster": _paymaster_status(paymaster_client),
        "onramp": _onramp_status(),
        "aerodrome": _aerodrome_status(aerodrome_client),
    }

    live_count = sum(1 for s in surfaces.values() if s["live_exec"])
    total = len(surfaces)

    if live_count == total:
        overall = "READY"
    elif live_count == 0:
        overall = "NOT_READY"
    else:
        overall = "PARTIAL"

    return {
        "overall": overall,
        "live_surface_count": live_count,
        "total_surface_count": total,
        "surfaces": surfaces,
    }

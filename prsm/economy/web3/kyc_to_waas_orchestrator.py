"""Sprint 858 — KYC→WaaS auto-provision orchestrator.

Closes the user-onboarding UX seam surfaced during sp852's E2E
walk: today an operator must call /wallet/kyc/initiate, wait for
Persona webhook to flip the record VERIFIED, THEN separately call
/wallet/waas/provision. Sp858 wires the auto-flow:

  Persona webhook → status=VERIFIED → auto-provision WaaS wallet →
  user immediately has a usable Base address for onramp.

The orchestrator is fail-soft on purpose:
  - If WaaS client isn't initialized: log + skip (sp848-style
    honest-signal pattern — webhook still returns 200 to Persona
    so they don't keep retrying)
  - If WaaS provision raises: log + skip (operator sees the
    exception in logs + can manually retry; KYC record stays
    VERIFIED)
  - Idempotent: WaaS.provision_wallet returns existing PROVISIONED
    record for the same user_id (sp851 invariant)

Only the VERIFIED transition triggers; REJECTED / EXPIRED / PENDING
do not (no point provisioning a wallet for a rejected user).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Statuses that should trigger auto-provision. Conservative: only
# the explicit "verified by vendor" terminal state.
_AUTO_PROVISION_STATUSES = {"VERIFIED"}


def maybe_auto_provision_waas(
    *,
    waas_client: Any,
    user_id: str,
    email: str,
    new_status: str,
    old_status: Optional[str] = None,
) -> Optional[Any]:
    """Conditional WaaS provision triggered by KYC status change.

    Returns the provisioned WaasWalletRecord on successful trigger,
    or None when no provision was attempted (wrong status, no WaaS
    client, etc.).

    Logs but does NOT raise on provision failure — webhook handlers
    must keep returning 200 to the vendor.
    """
    if new_status not in _AUTO_PROVISION_STATUSES:
        logger.debug(
            "sp858: KYC %s → status=%s — not a trigger status, "
            "skipping auto-provision",
            user_id, new_status,
        )
        return None
    if old_status == new_status:
        # No actual transition — e.g., Persona re-firing the same
        # event. Idempotent on the WaaS side anyway, but skipping
        # is cheaper.
        logger.debug(
            "sp858: KYC %s already at VERIFIED — auto-provision "
            "is idempotent, skipping the redundant call",
            user_id,
        )
        return None
    if waas_client is None:
        logger.info(
            "sp858: KYC %s VERIFIED but WaaS client not "
            "initialized — auto-provision skipped (operator can "
            "manually provision later)",
            user_id,
        )
        return None
    if not email:
        logger.warning(
            "sp858: KYC %s VERIFIED but no email on record — "
            "WaaS.provision_wallet requires email; skipping",
            user_id,
        )
        return None
    try:
        record = waas_client.provision_wallet(user_id, email)
        logger.info(
            "sp858: auto-provisioned WaaS wallet for user_id=%s "
            "after KYC VERIFIED — status=%s address=%s",
            user_id, record.status,
            record.address or "(pending)",
        )
        return record
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "sp858: WaaS auto-provision failed for user_id=%s "
            "(KYC stays VERIFIED; operator can retry manually): "
            "%s", user_id, exc,
        )
        return None

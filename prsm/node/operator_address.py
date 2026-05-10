"""Operator on-chain address resolution.

Used by /admin/earnings-summary heartbeat lookup. Resolution
order:
  1. PRSM_OPERATOR_ADDRESS — explicit override (wins always)
  2. FTNS_WALLET_PRIVATE_KEY — derive via eth_account
  3. None — earnings-summary heartbeat stream degrades

Fail-soft on derivation errors: bad PK shouldn't crash node
startup. Operators can verify resolution via /admin/earnings-
summary.operator_address.
"""
from __future__ import annotations

import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)


def resolve_operator_address() -> Optional[str]:
    """Resolve operator on-chain address from env."""
    explicit = os.environ.get("PRSM_OPERATOR_ADDRESS", "").strip()
    if explicit:
        return explicit

    pk = os.environ.get("FTNS_WALLET_PRIVATE_KEY", "").strip()
    if not pk:
        return None

    if not pk.startswith("0x"):
        pk = "0x" + pk

    try:
        from eth_account import Account
        account = Account.from_key(pk)
        return account.address
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to derive operator address from "
            "FTNS_WALLET_PRIVATE_KEY: %s. Set "
            "PRSM_OPERATOR_ADDRESS explicitly to bypass.",
            exc,
        )
        return None

"""Sprint 793 — production wire-up of WalletApiServices.

Sprint 792 shipped the /devices/earnings endpoint with a
`ReceiptLookup` Protocol on WalletApiServices. The default
`_NoOpReceiptLookup` returns empty so the endpoint doesn't
500 on unwired deployments — but production daemons need a
real adapter backed by `node._receipt_store` for the endpoint
to actually return per-device earnings.

This module ships:
  _ReceiptStoreAdapter — implements ReceiptLookup over the
    node's ReceiptStore. In-memory filter by settler_node_id.
  wire_wallet_api_services(node) — daemon-startup helper that
    builds + registers WalletApiServices via wallet_api.set_services.
    Idempotent (resets first, then sets), fail-soft (wraps in
    try/except so daemon doesn't crash on this peripheral surface).

PRSMNode.start calls wire_wallet_api_services(self) so the
endpoint just works on a running daemon. No env vars or operator
action required — the wire-up is automatic.

Honest scope:
- Adapter scans up to 1000 most-recent receipts from the store's
  in-memory cache (existing ReceiptStore.list contract). Operators
  with very large receipt histories may need to migrate to a
  persistent indexed store — but that's a follow-on (most
  operators have << 1000 receipts in any practical window).
- The binding service still uses InMemoryWalletBindingStore by
  default. A production daemon with multi-process workers needs
  to swap to SqliteWalletBindingStore (or Postgres) so bindings
  survive a worker restart. Sprint 794 candidate.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Dict, Optional


logger = logging.getLogger(__name__)


# Cap the in-memory scan window. ReceiptStore.list enforces an
# upper bound of 1000 itself; we use that explicitly so the
# adapter's behavior is operator-readable from the source.
_MAX_RECEIPTS_SCAN = 1000


class _ReceiptStoreAdapter:
    """Sprint 793 — ReceiptLookup over PRSMNode._receipt_store.

    Implements the sprint-792 Protocol. Scans the store's most-
    recent receipts and returns only those whose settler_node_id
    matches one of the requested node_ids.
    """

    def __init__(self, *, receipt_store: Optional[Any]) -> None:
        self._receipt_store = receipt_store

    def list_receipts_for_node_ids(
        self, node_ids: Iterable[str],
    ) -> List[Dict[str, Any]]:
        if self._receipt_store is None:
            return []
        # Materialize early — empty node_ids returns empty
        # regardless of store contents (defense against unbounded
        # scans being mistaken for "give me all receipts").
        ids = list(node_ids)
        if not ids:
            return []
        try:
            receipts = self._receipt_store.list(
                offset=0,
                limit=_MAX_RECEIPTS_SCAN,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "_ReceiptStoreAdapter scan failed: %s", exc,
            )
            return []
        allowed = set(ids)
        return [
            r for r in receipts
            if isinstance(r, dict)
            and r.get("settler_node_id") in allowed
        ]


def wire_wallet_api_services(node: Any) -> None:
    """Sprint 793 — register a production WalletApiServices on
    `wallet_api`.

    Idempotent: resets the existing services slot first, then
    set_services. Safe to call from daemon restart paths.

    Fail-soft: any exception during wiring is logged + swallowed.
    The /devices/earnings endpoint then falls back to the
    `_NoOpReceiptLookup` default — operators see empty earnings
    instead of a 500.
    """
    try:
        from prsm.interface.api import wallet_api
        from prsm.interface.onboarding.wallet_binding import (
            InMemoryWalletBindingStore,
            WalletBindingService,
        )
        receipt_store = getattr(node, "_receipt_store", None)
        adapter = _ReceiptStoreAdapter(receipt_store=receipt_store)

        wallet_api.reset_services_for_tests()
        services = wallet_api.WalletApiServices(
            settings=wallet_api.WalletApiSettings(
                expected_domain="prsm-network.com",
                expected_chain_id=8453,
            ),
            nonce_store=wallet_api.InMemoryNonceStore(),
            binding_service=WalletBindingService(
                InMemoryWalletBindingStore(),
            ),
            price_source=wallet_api.StaticPriceSource(
                price_usd=0,  # type: ignore[arg-type]
            ),
            balance_lookup=wallet_api._ZeroBalanceLookup(),
            receipt_lookup=adapter,
        )
        wallet_api.set_services(services)
        logger.info(
            "Sprint 793 — WalletApiServices wired with "
            "_ReceiptStoreAdapter (receipt_store=%s)",
            "present" if receipt_store is not None else "None",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 793 — wire_wallet_api_services failed (the "
            "/devices/earnings endpoint will fall back to the "
            "no-op default): %s", exc,
        )

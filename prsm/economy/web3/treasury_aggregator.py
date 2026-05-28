"""Sprint 864 — Phase 5 treasury aggregator.

Rolls up all WaaS wallet balances into a single PRSM treasury view:
total USDC + total FTNS + total native ETH across every PROVISIONED
wallet, with per-wallet breakdown. Powers GET /wallet/treasury and
the forthcoming ``prsm node treasury`` CLI surface.

Composable on top of sp862's WalletBalanceReader — each provisioned
wallet hit via the same JSON-RPC path. Operators with N wallets
incur N×3 RPC calls + 1 eth_blockNumber; for the dogfood-scale
fleet (single-digit wallets) that's sub-second total.

Skips wallets without an address (PENDING_COMMISSION records that
haven't been provisioned yet). Returns:

  {
    overall: {
      total_usdc, total_usdc_units,
      total_ftns, total_ftns_units,
      total_native_eth, total_native_eth_wei,
      wallet_count_total,         # all known wallets
      wallet_count_with_address,  # excluded from balance reads if 0
      wallet_count_funded,        # any balance > 0 across the 3 assets
      block_number,               # max block across reads
      rpc_url,
    },
    wallets: [
      {user_id, address, wallet_id, status, balances:{...}},
      ...
    ],
  }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def aggregate_treasury(
    *,
    waas_client: Any,
    balance_reader: Any,
    max_wallets: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate live Base mainnet balances across all WaaS wallets.

    ``max_wallets`` caps the per-call RPC load — set when the
    fleet grows beyond a few hundred wallets. None = unbounded.
    """
    if waas_client is None:
        return {
            "overall": {
                "total_usdc": 0.0, "total_usdc_units": 0,
                "total_ftns": 0.0, "total_ftns_units": 0,
                "total_native_eth": 0.0, "total_native_eth_wei": 0,
                "wallet_count_total": 0,
                "wallet_count_with_address": 0,
                "wallet_count_funded": 0,
                "block_number": 0,
                "rpc_url": None,
            },
            "wallets": [],
            "note": "WaaS client not initialized.",
        }

    records = waas_client.list_wallets()
    total = len(records)
    with_address = [r for r in records if r.address]
    if max_wallets is not None:
        with_address = with_address[:max_wallets]

    sum_usdc_units = 0
    sum_ftns_units = 0
    sum_wei = 0
    max_block = 0
    rpc_url: Optional[str] = None
    funded = 0
    wallets_out: List[Dict[str, Any]] = []

    for rec in with_address:
        try:
            bal = balance_reader.get_balances(rec.address)
        except Exception as exc:  # noqa: BLE001
            # Don't let one bad RPC read kill the whole treasury
            # view — log + continue with a placeholder.
            logger.warning(
                "treasury_aggregator: balance read failed for "
                "user_id=%s address=%s: %s",
                rec.user_id, rec.address, exc,
            )
            wallets_out.append({
                "user_id": rec.user_id,
                "wallet_id": rec.wallet_id,
                "address": rec.address,
                "status": rec.status,
                "balances": None,
                "error": str(exc),
            })
            continue
        sum_usdc_units += bal.usdc_units
        sum_ftns_units += bal.ftns_units
        sum_wei += bal.native_eth_wei
        max_block = max(max_block, bal.block_number)
        rpc_url = bal.rpc_url
        if (
            bal.usdc_units > 0
            or bal.ftns_units > 0
            or bal.native_eth_wei > 0
        ):
            funded += 1
        wallets_out.append({
            "user_id": rec.user_id,
            "wallet_id": rec.wallet_id,
            "address": rec.address,
            "status": rec.status,
            "balances": bal.to_dict(),
        })

    return {
        "overall": {
            "total_usdc": sum_usdc_units / (10 ** 6),
            "total_usdc_units": sum_usdc_units,
            "total_ftns": sum_ftns_units / (10 ** 18),
            "total_ftns_units": sum_ftns_units,
            "total_native_eth": sum_wei / (10 ** 18),
            "total_native_eth_wei": sum_wei,
            "wallet_count_total": total,
            "wallet_count_with_address": len(with_address),
            "wallet_count_funded": funded,
            "block_number": max_block,
            "rpc_url": rpc_url,
        },
        "wallets": wallets_out,
    }

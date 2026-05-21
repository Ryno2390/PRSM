"""Sprint 683 — on-chain stake reads for the DHT-backed pool provider.

Wraps StakeManagerClient.stake_of() with:
  - graceful degradation (returns 0 on any chain error)
  - 60s TTL cache (the pool provider can be hit per-request, the
    chain RPC cannot — cached reads keep the path cheap)
  - lazy contract construction (no chain dependency until first
    call that actually needs it)

Configuration via env:
  PRSM_STAKE_BOND_ADDRESS — checksum address of StakeBond on Base
  PRSM_BASE_RPC_URL       — RPC endpoint (defaults to mainnet Base)

Used by sprint 682's `build_dht_backed_pool_provider` via the
optional `stake_reader=` kwarg.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


_DEFAULT_TTL_SECONDS = 60.0
_DEFAULT_RPC = "https://mainnet.base.org"


class OnChainStakeReader:
    """Read provider stake amounts from the StakeBond contract."""

    def __init__(
        self,
        client_factory: Optional[Callable[[], Any]] = None,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client_factory = client_factory
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._cache: Dict[str, Tuple[int, float]] = {}
        self._client: Any = None
        self._client_construction_failed = False

    # ── Public ────────────────────────────────────────────────

    def stake_amount_for(self, operator_address: Optional[str]) -> int:
        if not operator_address:
            return 0

        contract_addr = os.environ.get(
            "PRSM_STAKE_BOND_ADDRESS", "",
        ).strip()
        if not contract_addr:
            return 0

        cached = self._cache.get(operator_address)
        if cached is not None:
            amount, ts = cached
            if self._clock() - ts < self._ttl_seconds:
                return amount

        client = self._get_client()
        if client is None:
            return 0

        try:
            record = client.stake_of(operator_address)
            amount = int(getattr(record, "amount_wei", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "stake_of(%s) raised: %s — returning 0",
                operator_address[:10], exc,
            )
            return 0

        self._cache[operator_address] = (amount, self._clock())
        return amount

    # ── Internal ──────────────────────────────────────────────

    def _get_client(self) -> Optional[Any]:
        if self._client is not None:
            return self._client
        if self._client_construction_failed:
            return None
        if self._client_factory is not None:
            try:
                self._client = self._client_factory()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "stake-reader client_factory raised: %s", exc,
                )
                self._client_construction_failed = True
                return None
            return self._client

        try:
            from prsm.economy.web3.stake_manager import StakeManagerClient
            rpc = os.environ.get("PRSM_BASE_RPC_URL", "").strip() or _DEFAULT_RPC
            contract_addr = os.environ.get(
                "PRSM_STAKE_BOND_ADDRESS", "",
            ).strip()
            self._client = StakeManagerClient(
                rpc_url=rpc, contract_address=contract_addr,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "StakeManagerClient construction failed: %s", exc,
            )
            self._client_construction_failed = True
            return None
        return self._client

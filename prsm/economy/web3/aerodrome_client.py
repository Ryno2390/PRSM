"""Sprint 279 — Aerodrome read-only pool quoter.

Aerodrome is a Velodrome-fork DEX on Base; PRSM's Phase 5 swap
layer uses an Aerodrome USDC-FTNS pool to bridge fiat (via
Coinbase CDP) and FTNS. This adapter provides operators with
on-demand pool-state visibility + swap quoting BEFORE the
Vision-gantt-2026-06-15 pool-seeding ceremony, so any operator
running their node can see the seeding status from their MCP.

Scope this sprint: read-only. No on-chain writes. The adapter
calls `getReserves()` / `token0()` / `token1()` / `stable()` /
`totalSupply()` on the pool contract and computes swap quotes
locally via the volatile-pool constant-product invariant.
Stable-pool stableswap curve is out of scope for v1.

Backend abstraction: production wires a real
``web3.contract`` instance; tests use a fake backend. This
keeps the test suite hermetic AND lets the future paymaster /
real-swap follow-on sprint plug the same client in without
contract-coupling test debt.

Operator env:
  - BASE_RPC_URL                       — JSON-RPC URL
  - AERODROME_USDC_FTNS_POOL_ADDRESS   — pool address (post-
                                          seeding ceremony)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class AerodromeQuoteError(Exception):
    """Raised when a quote cannot be produced (unknown token,
    stable pool, etc.). Distinct from RPC/backend errors which
    surface as None returns from get_pool_state + quote_swap."""


class _AerodromeBackend(Protocol):
    """Production backend wraps ``web3.contract``. Tests use a
    fake. Method returns canonical pool-state dict."""

    def get_pool_state(
        self, pool_address: str,
    ) -> Dict[str, Any]: ...


@dataclass
class AerodromePoolState:
    pool_address: str
    token0: str
    token1: str
    reserve0: int
    reserve1: int
    stable: bool
    fee_bps: int
    total_supply: int
    block_number: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AerodromeQuote:
    amount_in: int
    token_in: str
    token_out: str
    amount_out: int
    price_impact_bps: int  # basis points (10000 = 100%)
    route: str  # "aerodrome"
    fee_bps: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AerodromeClient:
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        pool_address: Optional[str] = None,
        *,
        backend: Optional[_AerodromeBackend] = None,
    ) -> None:
        self._rpc_url = rpc_url
        self._pool_address = pool_address
        self._backend = backend

    @classmethod
    def from_env(
        cls, *, backend: Optional[_AerodromeBackend] = None,
    ) -> "AerodromeClient":
        rpc_url = os.environ.get("BASE_RPC_URL") or None
        pool_address = (
            os.environ.get("AERODROME_USDC_FTNS_POOL_ADDRESS")
            or None
        )
        return cls(
            rpc_url=rpc_url, pool_address=pool_address,
            backend=backend,
        )

    def is_configured(self) -> bool:
        """True iff both RPC URL and pool address are present.

        The seeding ceremony populates the pool address env
        var, so this flips from False to True at the moment
        operators paste the deployed-pool address into config.
        """
        return bool(self._rpc_url and self._pool_address)

    @property
    def pool_address(self) -> Optional[str]:
        return self._pool_address

    def get_pool_state(
        self, pool_address: Optional[str] = None,
    ) -> Optional[AerodromePoolState]:
        """Return current pool state or None when unconfigured
        / backend raises. Fail-soft on RPC errors so operator
        UX doesn't break when Base RPC has a hiccup."""
        target = pool_address or self._pool_address
        if not target or not self._rpc_url:
            return None
        if self._backend is None:
            # No backend = nothing to call. Caller is expected
            # to wire a backend (or use from_env in production
            # which will populate one once the real Web3
            # backend lands).
            return None
        try:
            payload = self._backend.get_pool_state(target)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AerodromeClient: get_pool_state raised "
                "for %s: %s", target, exc,
            )
            return None
        return AerodromePoolState(
            pool_address=payload.get("pool_address", target),
            token0=payload.get("token0", ""),
            token1=payload.get("token1", ""),
            reserve0=int(payload.get("reserve0", 0)),
            reserve1=int(payload.get("reserve1", 0)),
            stable=bool(payload.get("stable", False)),
            fee_bps=int(payload.get("fee_bps", 30)),
            total_supply=int(payload.get("total_supply", 0)),
            block_number=int(payload.get("block_number", 0)),
        )

    def quote_swap(
        self,
        amount_in: int,
        token_in: str,
        pool_address: Optional[str] = None,
    ) -> Optional[AerodromeQuote]:
        """Quote an exact-amount-in swap via the pool's
        constant-product invariant.

        Returns None when unconfigured (no pool address /
        no RPC); raises AerodromeQuoteError when configured
        but the request is malformed (unknown token; stable
        pool); raises ValueError on bad inputs.
        """
        if not isinstance(amount_in, int) and not isinstance(
            amount_in, float,
        ):
            raise ValueError("amount_in must be numeric")
        if amount_in <= 0:
            raise ValueError(
                f"amount_in must be > 0, got {amount_in}"
            )
        if not token_in:
            raise ValueError("token_in is required")

        state = self.get_pool_state(pool_address)
        if state is None:
            return None
        if state.stable:
            raise AerodromeQuoteError(
                "Stable-pool swap math (stableswap invariant) "
                "not implemented in v1; volatile pools only."
            )

        # Identify direction.
        if token_in == state.token0:
            reserve_in = state.reserve0
            reserve_out = state.reserve1
            token_out = state.token1
        elif token_in == state.token1:
            reserve_in = state.reserve1
            reserve_out = state.reserve0
            token_out = state.token0
        else:
            raise AerodromeQuoteError(
                f"token_in={token_in!r} not in pool "
                f"(token0={state.token0!r}, "
                f"token1={state.token1!r})"
            )

        fee_bps = state.fee_bps
        # Uniswap-V2 constant-product with fee applied to
        # amount_in: out = (in * (10000-fee) * R_out) /
        # (10000 * R_in + in * (10000-fee))
        amount_in_with_fee = amount_in * (10_000 - fee_bps)
        numerator = amount_in_with_fee * reserve_out
        denominator = 10_000 * reserve_in + amount_in_with_fee
        if denominator == 0:
            raise AerodromeQuoteError(
                "denominator=0; pool reserves likely empty"
            )
        amount_out = numerator // denominator

        # Price impact = pure curve slippage (fee shown
        # separately via fee_bps). The constant-product math
        # gives a clean closed form: impact =
        # amount_in / (R_in + amount_in). Tiny trade in deep
        # pool → near zero; large fraction of reserves →
        # large impact.
        if reserve_in > 0 and amount_in > 0:
            slippage = amount_in / (reserve_in + amount_in)
            price_impact_bps = int(round(slippage * 10_000))
        else:
            price_impact_bps = 0

        return AerodromeQuote(
            amount_in=int(amount_in),
            token_in=token_in,
            token_out=token_out,
            amount_out=int(amount_out),
            price_impact_bps=price_impact_bps,
            route="aerodrome",
            fee_bps=fee_bps,
        )

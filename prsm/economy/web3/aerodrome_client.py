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


# Aerodrome Pool function selectors (keccak4 of the signature).
_SEL_TOKEN0 = "0x0dfe1681"        # token0()
_SEL_TOKEN1 = "0xd21220a7"        # token1()
_SEL_GET_RESERVES = "0x0902f1ac"  # getReserves() -> (r0, r1, ts)
_SEL_STABLE = "0x22be3de1"        # stable() -> bool
_SEL_TOTAL_SUPPLY = "0x18160ddd"  # totalSupply()

_RPC_TIMEOUT_SECONDS = 15.0


def _decode_uint256(result: Optional[str]) -> int:
    if not result or result == "0x":
        return 0
    return int(result, 16)


def _decode_address(result: Optional[str]) -> str:
    """Last 20 bytes of a 32-byte word → 0x-prefixed address."""
    if not result:
        return ""
    h = result[2:] if result.startswith("0x") else result
    return "0x" + h[-40:]


class AerodromeRpcBackend:
    """Sprint 902 — real Base-RPC backend for pool reads.

    Implements ``get_pool_state`` via JSON-RPC ``eth_call`` against the
    Aerodrome Pool contract (token0/token1/getReserves/stable/
    totalSupply) + ``eth_blockNumber``. Mirrors sp862's
    WalletBalanceReader RPC pattern (httpx, 15s timeout). Read-only;
    raises on transport/RPC error so AerodromeClient.get_pool_state can
    fail-soft to None. ``fee_bps`` defaults to the Aerodrome volatile
    convention (30) — exact fee is factory-governed and not load-bearing
    for the constant-product quote.
    """

    def __init__(
        self,
        rpc_url: str,
        *,
        client: Any = None,
        fee_bps: int = 30,
    ) -> None:
        self._rpc_url = rpc_url
        self._fee_bps = fee_bps
        if client is None:
            import httpx
            self._client = httpx.Client(timeout=_RPC_TIMEOUT_SECONDS)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        self._rpc_id = 0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _rpc_call(self, method: str, params: list) -> Any:
        self._rpc_id += 1
        resp = self._client.post(
            self._rpc_url,
            json={
                "jsonrpc": "2.0", "id": self._rpc_id,
                "method": method, "params": params,
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(
                f"Base RPC {method} error: {payload['error']!r}"
            )
        return payload.get("result")

    def _call(self, pool_address: str, selector: str) -> Optional[str]:
        return self._rpc_call("eth_call", [
            {"to": pool_address, "data": selector}, "latest",
        ])

    def get_pool_state(self, pool_address: str) -> Dict[str, Any]:
        token0 = _decode_address(self._call(pool_address, _SEL_TOKEN0))
        token1 = _decode_address(self._call(pool_address, _SEL_TOKEN1))
        reserves_raw = self._call(pool_address, _SEL_GET_RESERVES) or ""
        h = reserves_raw[2:] if reserves_raw.startswith("0x") else reserves_raw
        reserve0 = int(h[0:64], 16) if len(h) >= 64 else 0
        reserve1 = int(h[64:128], 16) if len(h) >= 128 else 0
        stable = _decode_uint256(
            self._call(pool_address, _SEL_STABLE),
        ) != 0
        total_supply = _decode_uint256(
            self._call(pool_address, _SEL_TOTAL_SUPPLY),
        )
        block_number = _decode_uint256(
            self._rpc_call("eth_blockNumber", []),
        )
        return {
            "pool_address": pool_address,
            "token0": token0,
            "token1": token1,
            "reserve0": reserve0,
            "reserve1": reserve1,
            "stable": stable,
            "fee_bps": self._fee_bps,
            "total_supply": total_supply,
            "block_number": block_number,
        }


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
        # Sprint 902 — wire the real Base-RPC backend when an RPC URL
        # is configured (and the caller didn't inject one), so live
        # pool reads work the moment the seed ceremony populates the
        # pool address. Pre-902 from_env left backend=None, so
        # get_pool_state always returned None even on a live node.
        if backend is None and rpc_url:
            backend = AerodromeRpcBackend(rpc_url=rpc_url)
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

        # Identify direction. Sprint 902 — compare addresses
        # case-insensitively: eth_call returns pool token addresses
        # lowercase, while callers (the onramp→swap orchestrator) pass
        # the EIP-55 checksummed constant, so `==` would spuriously
        # report "token_in not in pool" on every live quote.
        _ti = token_in.lower()
        if _ti == state.token0.lower():
            reserve_in = state.reserve0
            reserve_out = state.reserve1
            token_out = state.token1
        elif _ti == state.token1.lower():
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

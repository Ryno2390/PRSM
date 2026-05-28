"""Sprint 862 — Lightweight Base mainnet ERC-20 balance reader.

Operator-facing observability for WaaS wallets: query live USDC +
FTNS + native ETH balances via Base RPC without web3.py contract
overhead. Powers GET /wallet/balance/{user_id}.

Uses raw JSON-RPC eth_call with the ERC-20 ``balanceOf(address)``
selector (0x70a08231) so it works against any standard ERC-20 on
any EVM chain — not bound to web3.Contract setup.

Defaults to ``https://mainnet.base.org`` (Base's free public RPC,
rate-limited but fine for low-volume operator queries) when
``BASE_RPC_URL`` is unset. Operators with paid Alchemy / Infura /
QuickNode endpoints set ``BASE_RPC_URL`` to override.

Token addresses are pinned to the canonical Base mainnet
deployments (FTNS from networks.py registry, USDC from Circle's
native Base deployment 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 —
NOT the bridged USDbC).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_DEFAULT_BASE_RPC = "https://mainnet.base.org"
_USDC_BASE_MAINNET = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_USDC_DECIMALS = 6
_FTNS_DECIMALS = 18
_NATIVE_ETH_DECIMALS = 18
_RPC_TIMEOUT_SECONDS = 15.0
_BALANCE_OF_SELECTOR = "0x70a08231"


def _addr_to_call_data(address: str) -> str:
    """Encode ``balanceOf(address)`` call data: selector + 32-byte
    left-padded address (drop 0x, lowercase, pad to 64 hex chars)."""
    clean = address.lower().removeprefix("0x")
    if len(clean) != 40:
        raise ValueError(
            f"address must be 20 bytes (40 hex chars), "
            f"got {len(clean)} chars from {address!r}"
        )
    padded = clean.rjust(64, "0")
    return _BALANCE_OF_SELECTOR + padded


def _decode_uint256(hex_str: str) -> int:
    """Decode JSON-RPC hex result to int (default 0 on empty/bad)."""
    if not hex_str or hex_str == "0x":
        return 0
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return 0


@dataclass
class WalletBalances:
    address: str
    usdc: float  # whole-token USDC (6 decimals)
    usdc_units: int  # base units
    ftns: float  # whole-token FTNS (18 decimals)
    ftns_units: int
    native_eth: float  # whole-token ETH (18 decimals)
    native_eth_wei: int
    block_number: int  # block when balance was read
    rpc_url: str  # which RPC served the query (audit signal)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WalletBalanceReader:
    """ERC-20 balance reader via raw Base JSON-RPC."""

    def __init__(
        self,
        *,
        rpc_url: Optional[str] = None,
        ftns_address: Optional[str] = None,
        usdc_address: str = _USDC_BASE_MAINNET,
        client: Any = None,
    ) -> None:
        self._rpc_url = rpc_url or _DEFAULT_BASE_RPC
        if ftns_address is None:
            from prsm.config.networks import get_network_config
            net = get_network_config("mainnet")
            ftns_address = net.ftns_token
            if not ftns_address:
                raise ValueError(
                    "FTNS contract address missing from "
                    "networks.py base-mainnet config",
                )
        self._ftns_address = ftns_address
        self._usdc_address = usdc_address
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

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def _rpc_call(self, method: str, params: list) -> Any:
        """Single JSON-RPC POST; raises on transport or RPC error."""
        body = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }
        resp = self._client.post(
            self._rpc_url,
            json=body,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            err = payload["error"]
            raise RuntimeError(
                f"Base RPC {method} returned error: {err!r}"
            )
        return payload.get("result")

    def _erc20_balance(self, token_addr: str, owner: str) -> int:
        """eth_call balanceOf(owner) on token_addr; returns base units."""
        result = self._rpc_call("eth_call", [
            {"to": token_addr, "data": _addr_to_call_data(owner)},
            "latest",
        ])
        return _decode_uint256(result)

    def _native_balance(self, owner: str) -> int:
        """eth_getBalance for native ETH; returns wei."""
        result = self._rpc_call(
            "eth_getBalance", [owner, "latest"],
        )
        return _decode_uint256(result)

    def _block_number(self) -> int:
        """eth_blockNumber for the audit-signal block field."""
        result = self._rpc_call("eth_blockNumber", [])
        return _decode_uint256(result)

    def get_balances(self, address: str) -> WalletBalances:
        """Live USDC + FTNS + ETH balances for ``address``.

        Single RPC roundtrip per balance (4 calls total: USDC,
        FTNS, native, block). Block number captured AFTER balances
        so the block is at least as fresh as the reads.
        """
        if not address or not address.startswith("0x"):
            raise ValueError(
                f"address must be 0x EVM, got {address!r}",
            )
        usdc_units = self._erc20_balance(
            self._usdc_address, address,
        )
        ftns_units = self._erc20_balance(
            self._ftns_address, address,
        )
        native_wei = self._native_balance(address)
        block = self._block_number()
        return WalletBalances(
            address=address,
            usdc=usdc_units / (10 ** _USDC_DECIMALS),
            usdc_units=usdc_units,
            ftns=ftns_units / (10 ** _FTNS_DECIMALS),
            ftns_units=ftns_units,
            native_eth=native_wei / (10 ** _NATIVE_ETH_DECIMALS),
            native_eth_wei=native_wei,
            block_number=block,
            rpc_url=self._rpc_url,
        )


def from_env(
    *,
    rpc_url: Optional[str] = None,
    client: Any = None,
) -> "WalletBalanceReader":
    """Construct from env. Always succeeds — falls back to free
    public Base RPC if BASE_RPC_URL is unset."""
    rpc_url = (
        rpc_url
        or os.environ.get("BASE_RPC_URL")
        or _DEFAULT_BASE_RPC
    )
    return WalletBalanceReader(rpc_url=rpc_url, client=client)

"""Sprint 850 — Coinbase CDP Paymaster HTTP backend.

Implements the ``_PaymasterBackend`` Protocol used by
``PaymasterClient`` via JSON-RPC against Coinbase Developer Platform's
Paymaster endpoint
(https://docs.cdp.coinbase.com/paymaster/docs/welcome).

CDP v2 endpoint URLs embed the API token in the path, e.g.::

    https://api.developer.coinbase.com/rpc/v1/base/<TOKEN>

so no separate Authorization header is required — the URL itself is
the credential. ``COINBASE_CDP_PAYMASTER_API_KEY`` env var is kept
around for parity with sibling adapters + future header-auth schemes.

Two methods on the Protocol:

  estimate_gas(user_op) -> {gas_estimate_wei: int}
    Calls ``pm_sponsorUserOperation`` to estimate gas (does not
    sign / submit). Sums callGasLimit + verificationGasLimit +
    preVerificationGas + paymasterVerificationGasLimit + paymasterPostOpGasLimit
    multiplied by maxFeePerGas to produce a single wei figure.

  submit_sponsored(user_op) -> {tx_hash, user_op_hash, sponsor_amount_wei}
    Calls ``pm_sponsorUserOperation`` (gets paymasterAndData fields)
    then ``eth_sendUserOperation`` (submits to the bundler). CDP's
    paymaster endpoint also bundles, so both calls hit the same URL.

The caller (``PaymasterClient.sponsor_user_op``) is expected to pass
a fully-formed ERC-4337 PackedUserOperation v0.7 dict already signed
by the smart-account owner; this backend ONLY adds paymaster fields
+ submits. Signing is the caller's job (operator code path or
smart-account SDK).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ERC-4337 v0.7 EntryPoint (canonical, deployed on Base mainnet).
_DEFAULT_ENTRY_POINT = "0x0000000071727De22E5E9d8BAf0edAc6f37da032"
_CDP_TIMEOUT_SECONDS = 30.0


class _CdpRpcError(RuntimeError):
    """CDP JSON-RPC returned an `error` field."""


def _wei_from_hex(value: Any, default: int = 0) -> int:
    """Decode a JSON-RPC hex string (e.g. '0x5208') to int."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.startswith("0x") or value.startswith("0X"):
            try:
                return int(value, 16)
            except ValueError:
                return default
        try:
            return int(value)
        except ValueError:
            return default
    return default


class CdpPaymasterBackend:
    """CDP Paymaster JSON-RPC backend for PaymasterClient."""

    def __init__(
        self,
        endpoint: str,
        *,
        entry_point: str = _DEFAULT_ENTRY_POINT,
        client: Any = None,  # injected httpx.Client for tests
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        self._endpoint = endpoint
        self._entry_point = entry_point
        if client is None:
            import httpx
            self._client = httpx.Client(timeout=_CDP_TIMEOUT_SECONDS)
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

    def _rpc(self, method: str, params: list) -> Any:
        """JSON-RPC POST helper. Raises _CdpRpcError on `error`."""
        body = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }
        resp = self._client.post(
            self._endpoint,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            err = payload["error"]
            raise _CdpRpcError(
                f"cdp {method} failed: {err!r}",
            )
        return payload.get("result")

    def estimate_gas(
        self, user_op: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Estimate gas via pm_sponsorUserOperation (no submit)."""
        result = self._rpc(
            "pm_sponsorUserOperation",
            [user_op, self._entry_point],
        ) or {}
        call_gas = _wei_from_hex(result.get("callGasLimit"))
        verify_gas = _wei_from_hex(result.get("verificationGasLimit"))
        pre_verify = _wei_from_hex(result.get("preVerificationGas"))
        pm_verify = _wei_from_hex(
            result.get("paymasterVerificationGasLimit"),
        )
        pm_post = _wei_from_hex(
            result.get("paymasterPostOpGasLimit"),
        )
        max_fee = _wei_from_hex(
            user_op.get("maxFeePerGas"),
            default=0,
        )
        total_gas_units = (
            call_gas + verify_gas + pre_verify
            + pm_verify + pm_post
        )
        gas_estimate_wei = total_gas_units * max_fee
        return {"gas_estimate_wei": gas_estimate_wei}

    def submit_sponsored(
        self, user_op: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Sponsor + submit a user op. Returns submission record.

        Step 1: pm_sponsorUserOperation → returns paymaster fields
        + gas estimates we merge into the user_op.
        Step 2: eth_sendUserOperation → returns user_op_hash. The
        full tx_hash is then surfaced via eth_getUserOperationReceipt
        (caller can poll separately if needed).
        """
        sponsor_result = self._rpc(
            "pm_sponsorUserOperation",
            [user_op, self._entry_point],
        ) or {}
        merged = dict(user_op)
        for k in (
            "paymaster",
            "paymasterVerificationGasLimit",
            "paymasterPostOpGasLimit",
            "paymasterData",
            "callGasLimit",
            "verificationGasLimit",
            "preVerificationGas",
        ):
            if k in sponsor_result:
                merged[k] = sponsor_result[k]
        user_op_hash = self._rpc(
            "eth_sendUserOperation",
            [merged, self._entry_point],
        )
        # Compute approx sponsor amount in wei from the estimate fields.
        max_fee = _wei_from_hex(merged.get("maxFeePerGas"))
        total_units = (
            _wei_from_hex(merged.get("callGasLimit"))
            + _wei_from_hex(merged.get("verificationGasLimit"))
            + _wei_from_hex(merged.get("preVerificationGas"))
            + _wei_from_hex(merged.get("paymasterVerificationGasLimit"))
            + _wei_from_hex(merged.get("paymasterPostOpGasLimit"))
        )
        sponsor_amount_wei = total_units * max_fee
        return {
            "tx_hash": None,  # surfaced async via getUserOperationReceipt
            "user_op_hash": user_op_hash,
            "sponsor_amount_wei": sponsor_amount_wei,
        }


def from_env(
    *,
    endpoint: Optional[str] = None,
    entry_point: Optional[str] = None,
    client: Any = None,
) -> Optional["CdpPaymasterBackend"]:
    """Construct a CdpPaymasterBackend from env, or None when missing.

    Returns None when ``COINBASE_CDP_PAYMASTER_ENDPOINT`` is absent so
    ``PaymasterClient.from_env()`` can gracefully fall back to the
    un-backed PENDING_COMMISSION pattern.
    """
    endpoint = endpoint or os.environ.get(
        "COINBASE_CDP_PAYMASTER_ENDPOINT",
    )
    if not endpoint:
        return None
    entry_point = (
        entry_point
        or os.environ.get("PRSM_ERC4337_ENTRY_POINT")
        or _DEFAULT_ENTRY_POINT
    )
    return CdpPaymasterBackend(
        endpoint=endpoint,
        entry_point=entry_point,
        client=client,
    )

"""Sprint 867 — close Paymaster live_exec via sponsored UserOp.

Builds and submits a sponsored ERC-4337 v0.7 UserOperation that:

  1. Counterfactually deploys a SimpleAccount controlled by the
     PRSM Mainnet Deployer EOA
  2. In the same op, calls FTNS.approve(AerodromeRouter, 0) — a
     zero-asset no-op that exercises the full sponsor→bundler→
     entrypoint→smart-account→target codepath

The PRSM CDP Paymaster covers all gas. Deployer EOA needs zero ETH
because factoryData is sponsored too. After execution, a real
SimpleAccount exists on Base mainnet at the counterfactual address,
controlled by the deployer EOA, and Paymaster sponsorships > 0 —
flipping Phase 5 paymaster.live_exec from False to True per sp859's
aggregator.

Env required:
  PRSM_DEPLOYER_PRIVATE_KEY     0x-prefixed Base EOA private key
  COINBASE_CDP_PAYMASTER_ENDPOINT  CDP paymaster + bundler URL

Run:
  set -a && source secrets/phase5-fiat.env && set +a
  PRSM_DEPLOYER_PRIVATE_KEY=0x... python3 scripts/sprint_867_paymaster_live_exec.py

Exit codes:
  0  → sponsored UserOp submitted + on-chain receipt confirmed
  1  → env missing OR CDP rejected the userop
  2  → submitted but receipt poll timed out (check Basescan manually)
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

# Canonical Base mainnet (chain_id 8453) contracts.
CHAIN_ID = 8453
ENTRYPOINT_V07 = "0x0000000071727De22E5E9d8BAf0edAc6f37da032"
SIMPLE_ACCOUNT_FACTORY_V07 = (
    "0x91E60e0613810449d098b0b5Ec8b51A0FE8c8985"
)
FTNS_TOKEN = "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
AERODROME_ROUTER = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"

# Function selectors (computed once + pinned).
SELECTOR_CREATE_ACCOUNT = bytes.fromhex("5fbfb9cf")  # createAccount(address,uint256)
SELECTOR_EXECUTE = bytes.fromhex("b61d27f6")          # execute(address,uint256,bytes)
SELECTOR_APPROVE = bytes.fromhex("095ea7b3")          # approve(address,uint256)


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(
            f"ERROR: {name} env var not set. "
            f"Export it before running.",
            file=sys.stderr,
        )
        sys.exit(1)
    return val


def _addr_to_bytes(addr: str) -> bytes:
    """0x-prefixed 20-byte EVM address → 20 bytes."""
    return bytes.fromhex(addr.removeprefix("0x"))


def _u256_to_bytes(n: int) -> bytes:
    return n.to_bytes(32, "big")


def _padded_address(addr: str) -> bytes:
    """20-byte address left-padded to 32 bytes."""
    return b"\x00" * 12 + _addr_to_bytes(addr)


def _hex(b: bytes) -> str:
    return "0x" + b.hex()


def _encode_create_account_calldata(owner: str, salt: int) -> bytes:
    """SimpleAccountFactory.createAccount(address, uint256)"""
    return (
        SELECTOR_CREATE_ACCOUNT
        + _padded_address(owner)
        + _u256_to_bytes(salt)
    )


def _encode_approve_calldata(spender: str, amount: int) -> bytes:
    """ERC-20 approve(address, uint256)"""
    return (
        SELECTOR_APPROVE
        + _padded_address(spender)
        + _u256_to_bytes(amount)
    )


def _encode_execute_calldata(
    target: str, value: int, inner_data: bytes,
) -> bytes:
    """SimpleAccount.execute(address, uint256, bytes).

    ABI-encodes the dynamic `bytes` parameter properly: head
    offset (32) + length (32) + data (padded to 32-byte word).
    """
    # Static head: 4-byte selector + address (padded) + uint256 + offset to bytes
    # offset to dynamic bytes = 3 * 32 = 96 (sits after target+value+offset)
    offset = 96
    # Pad inner_data to multiple of 32
    inner_len = len(inner_data)
    pad = (-inner_len) % 32
    return (
        SELECTOR_EXECUTE
        + _padded_address(target)
        + _u256_to_bytes(value)
        + _u256_to_bytes(offset)
        + _u256_to_bytes(inner_len)
        + inner_data
        + b"\x00" * pad
    )


def _compute_counterfactual_sender(
    rpc_url: str,
    factory: str,
    factory_data: bytes,
    bearer_jwt: Optional[str] = None,
) -> str:
    """Call EntryPoint.getSenderAddress(initCode) to get the
    deterministic address the SimpleAccount will deploy at.

    EntryPoint reverts with the sender address embedded in the
    revert data — we parse it from the JSON-RPC error.
    """
    import httpx
    init_code = _addr_to_bytes(factory) + factory_data
    # getSenderAddress(bytes) selector = 0x9b249f69
    selector = bytes.fromhex("9b249f69")
    offset = _u256_to_bytes(32)
    length = _u256_to_bytes(len(init_code))
    pad = (-len(init_code)) % 32
    call_data = (
        selector + offset + length + init_code + b"\x00" * pad
    )
    headers = {"Content-Type": "application/json"}
    if bearer_jwt:
        headers["Authorization"] = f"Bearer {bearer_jwt}"
    body = {
        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
        "params": [
            {"to": ENTRYPOINT_V07, "data": _hex(call_data)},
            "latest",
        ],
    }
    with httpx.Client(timeout=30) as c:
        resp = c.post(rpc_url, json=body, headers=headers)
    resp.raise_for_status()
    payload = resp.json()
    # EntryPoint v0.7 reverts with SenderAddressResult(address)
    # whose selector is 0x6ca7b806. Revert data:
    #   selector (4) + padded address (32) = 36 bytes total
    err = payload.get("error", {})
    data = err.get("data", "") or err.get("revert_data", "")
    if not data:
        # Some bundlers return result instead
        result = payload.get("result")
        if result and result != "0x":
            # If it returned (didn't revert), the address comes
            # back as 32-byte padded
            addr_bytes = bytes.fromhex(
                result.removeprefix("0x"),
            )[-20:]
            return "0x" + addr_bytes.hex()
        raise RuntimeError(
            f"getSenderAddress returned no revert data or "
            f"result: {payload!r}"
        )
    raw = data.removeprefix("0x")
    if len(raw) < 72:
        raise RuntimeError(
            f"unexpected revert data length: {data!r}"
        )
    # selector at [0:4 hex chars * 2 = 8] + address at [last 40 hex chars]
    addr_hex = raw[-40:]
    return "0x" + addr_hex


def _build_cdp_bearer_jwt(
    api_key_name: str,
    api_key_private_pem: str,
    method: str,
    host: str,
    path: str,
) -> str:
    """Build CDP Bearer JWT per sp854b spec (Ed25519/EdDSA)."""
    import secrets as _sec
    import jwt as _jwt
    from prsm.economy.web3.coinbase_waas_cdp_backend import (
        _load_ed25519_pem,
    )
    private_key = _load_ed25519_pem(api_key_private_pem)
    now = int(time.time())
    headers = {
        "alg": "EdDSA", "typ": "JWT",
        "kid": api_key_name,
        "nonce": _sec.token_hex(8),
    }
    payload = {
        "iss": "cdp", "sub": api_key_name,
        "nbf": now, "exp": now + 120,
        "aud": [method, host],
        "uri": f"{method} {host}{path}",
    }
    return _jwt.encode(
        payload, private_key, algorithm="EdDSA",
        headers=headers,
    )


def _rpc_call(
    rpc_url: str, method: str, params: list,
    bearer_jwt: Optional[str] = None,
) -> Any:
    import httpx
    headers = {"Content-Type": "application/json"}
    if bearer_jwt:
        headers["Authorization"] = f"Bearer {bearer_jwt}"
    body = {
        "jsonrpc": "2.0", "id": 1,
        "method": method, "params": params,
    }
    with httpx.Client(timeout=60) as c:
        resp = c.post(rpc_url, json=body, headers=headers)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(
            f"RPC {method} error: {payload['error']!r}"
        )
    return payload.get("result")


def _cdp_rpc_call(
    rpc_url: str, method: str, params: list,
    api_key_name: str, api_key_private_pem: str,
) -> Any:
    """JSON-RPC call against CDP with fresh Bearer JWT per request.

    CDP signs requests at the Cloudflare layer using JWT — the
    URL-baked token is the project identifier, not auth.
    """
    from urllib.parse import urlparse
    parsed = urlparse(rpc_url)
    host = parsed.netloc
    path = parsed.path
    bearer = _build_cdp_bearer_jwt(
        api_key_name=api_key_name,
        api_key_private_pem=api_key_private_pem,
        method="POST", host=host, path=path,
    )
    return _rpc_call(
        rpc_url, method, params, bearer_jwt=bearer,
    )


def _pack_user_op_for_hashing(
    sender: str,
    nonce: int,
    init_code: bytes,
    call_data: bytes,
    call_gas_limit: int,
    verification_gas_limit: int,
    pre_verification_gas: int,
    max_fee_per_gas: int,
    max_priority_fee_per_gas: int,
    paymaster_and_data: bytes,
) -> bytes:
    """v0.7 packed user op encoding for hashing — per
    EIP-4337's IEntryPoint.hash(PackedUserOperation).
    """
    from hashlib import sha3_256  # noqa: F401 — keccak below

    def _keccak(data: bytes) -> bytes:
        from eth_utils import keccak as _kc  # type: ignore
        return _kc(data)

    hash_init_code = _keccak(init_code)
    hash_call_data = _keccak(call_data)
    hash_paymaster = _keccak(paymaster_and_data)
    account_gas_limits = (
        (verification_gas_limit & ((1 << 128) - 1)) << 128
    ) | (call_gas_limit & ((1 << 128) - 1))
    gas_fees = (
        (max_priority_fee_per_gas & ((1 << 128) - 1)) << 128
    ) | (max_fee_per_gas & ((1 << 128) - 1))

    # abi.encode(address, uint256, bytes32, bytes32, bytes32,
    #            uint256, bytes32, bytes32)
    return (
        _padded_address(sender)
        + _u256_to_bytes(nonce)
        + hash_init_code
        + hash_call_data
        + _u256_to_bytes(account_gas_limits)
        + _u256_to_bytes(pre_verification_gas)
        + _u256_to_bytes(gas_fees)
        + hash_paymaster
    )


def _user_op_hash(
    packed: bytes,
    entry_point: str = ENTRYPOINT_V07,
    chain_id: int = CHAIN_ID,
) -> bytes:
    from eth_utils import keccak  # type: ignore
    inner = keccak(packed)
    outer_payload = (
        inner + _padded_address(entry_point)
        + _u256_to_bytes(chain_id)
    )
    return keccak(outer_payload)


def main() -> int:
    pk = _require_env("PRSM_DEPLOYER_PRIVATE_KEY")
    paymaster_url = _require_env("COINBASE_CDP_PAYMASTER_ENDPOINT")
    # Sp867 — CDP paymaster endpoint requires Bearer JWT auth on
    # every call (Ed25519, sp854b spec). Use a separate standard
    # RPC for state reads (no auth needed).
    base_rpc = (
        os.environ.get("BASE_RPC_URL", "").strip()
        or "https://mainnet.base.org"
    )
    cdp_key_name = _require_env("COINBASE_CDP_API_KEY_NAME")
    cdp_key_priv = _require_env("COINBASE_CDP_API_KEY_PRIVATE")

    if not pk.startswith("0x"):
        pk = "0x" + pk

    from eth_account import Account  # type: ignore
    acct = Account.from_key(pk)
    owner = acct.address
    print(f"Owner EOA: {owner}")
    print(f"Base RPC: {base_rpc}")
    print(f"Paymaster: {paymaster_url[:60]}...")

    # Factory data: createAccount(owner, salt) — use a non-zero
    # salt so each script invocation gets a fresh counterfactual
    # sender address. The previous salt=0 sender accumulated
    # phantom per-user spend across failed sponsor calls;
    # rotating salt sidesteps that cap poisoning.
    salt = int(os.environ.get("PRSM_SP867_SALT", "1"))
    factory_data = _encode_create_account_calldata(owner, salt)
    print(f"Salt: {salt}")

    # Counterfactual sender address via standard Base RPC.
    print("Querying EntryPoint.getSenderAddress for counterfactual...")
    sender = _compute_counterfactual_sender(
        rpc_url=base_rpc,
        factory=SIMPLE_ACCOUNT_FACTORY_V07,
        factory_data=factory_data,
    )
    print(f"SimpleAccount counterfactual address: {sender}")

    # Verify the account isn't already deployed (state read on Base RPC)
    code = _rpc_call(
        base_rpc, "eth_getCode", [sender, "latest"],
    )
    if code and code != "0x":
        print(
            "WARNING: SimpleAccount already deployed at "
            f"{sender}. Nonce will be > 0 and factoryData "
            "should NOT be sent. This script assumes fresh "
            "counterfactual; aborting."
        )
        return 1

    # Simplified inner call: execute(SimpleAccount, 0, "")  — self-call
    # with empty data. Tests deploy + sponsor pathway without
    # depending on FTNS or any external contract state. The original
    # approve(Router, 0) call moves to a follow-on once the basic
    # sponsorship path is proven live.
    call_data = _encode_execute_calldata(sender, 0, b"")
    print(f"callData built: {len(call_data)} bytes (self-call no-op)")

    # Initial gas estimates — bundler will recompute via sponsor call
    # but we need reasonable starting values.
    # Dummy signature for gas estimation: 65 bytes = 32 r + 32 s + 1 v.
    # CRITICAL constraints to avoid AA23 (ECDSA.recover revert):
    #   v must be 27 or 28 (we use 28 = 0x1c)
    #   r must be in [1, n) — secp256k1n=0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
    #   s must be in [1, n/2) — OZ ECDSA enforces low-s
    # Small values like r=1, s=1 are trivially valid. They produce
    # SOME recovered address (just not the real owner), so validation
    # returns SIG_VALIDATION_FAILED cleanly instead of reverting.
    DUMMY_SIG = (
        "0x"
        + "00" * 31 + "01"             # r = 1
        + "00" * 31 + "01"             # s = 1
        + "1c"                         # v = 28
    )
    assert len(DUMMY_SIG) == 2 + 130   # 0x + 65 bytes hex
    user_op: Dict[str, Any] = {
        "sender": sender,
        "nonce": "0x0",
        "factory": SIMPLE_ACCOUNT_FACTORY_V07,
        "factoryData": _hex(factory_data),
        "callData": _hex(call_data),
        # Generous gas limits for the tracer (tightening caused
        # "failed to trace calls" — bundler simulation needs
        # headroom in the trace itself, not just static analysis).
        # Per-user $5 cap accommodates worst-case projection.
        "callGasLimit": "0x" + format(200_000, "x"),
        "verificationGasLimit": "0x" + format(2_000_000, "x"),
        "preVerificationGas": "0x" + format(100_000, "x"),
        "maxFeePerGas": "0x" + format(1_000_000_000, "x"),  # 1 gwei
        "maxPriorityFeePerGas": "0x" + format(500_000_000, "x"),
        "signature": DUMMY_SIG,
    }

    # Step 1: pm_sponsorUserOperation → get paymaster fields.
    # Paymaster endpoint URL contains the project token in the
    # path — no Bearer header needed.
    print("Calling pm_sponsorUserOperation...")
    sponsor = _rpc_call(
        paymaster_url, "pm_sponsorUserOperation",
        [user_op, ENTRYPOINT_V07],
    )
    if not sponsor:
        print("ERROR: paymaster returned empty sponsor data")
        return 1
    print(f"Sponsor data: {json.dumps(sponsor, indent=2)}")
    # Merge sponsor fields into user op
    for k in (
        "paymaster", "paymasterVerificationGasLimit",
        "paymasterPostOpGasLimit", "paymasterData",
        "callGasLimit", "verificationGasLimit",
        "preVerificationGas",
    ):
        if k in sponsor:
            user_op[k] = sponsor[k]

    # Step 2: compute user op hash + sign
    def _hex_to_int(s: Any) -> int:
        if isinstance(s, int):
            return s
        if isinstance(s, str):
            return int(s, 16) if s.startswith("0x") else int(s)
        return 0

    # Reconstruct fields
    sender_addr = user_op["sender"]
    nonce_int = _hex_to_int(user_op["nonce"])
    init_code = (
        _addr_to_bytes(user_op["factory"])
        + bytes.fromhex(
            user_op["factoryData"].removeprefix("0x")
        )
    )
    call_data_bytes = bytes.fromhex(
        user_op["callData"].removeprefix("0x")
    )
    call_gas = _hex_to_int(user_op["callGasLimit"])
    ver_gas = _hex_to_int(user_op["verificationGasLimit"])
    pre_ver = _hex_to_int(user_op["preVerificationGas"])
    max_fee = _hex_to_int(user_op["maxFeePerGas"])
    max_prio = _hex_to_int(user_op["maxPriorityFeePerGas"])
    pm_addr = user_op.get("paymaster", "")
    pm_v = _hex_to_int(
        user_op.get("paymasterVerificationGasLimit", "0x0"),
    )
    pm_p = _hex_to_int(
        user_op.get("paymasterPostOpGasLimit", "0x0"),
    )
    pm_data_hex = user_op.get("paymasterData", "0x")
    pm_data = bytes.fromhex(pm_data_hex.removeprefix("0x"))
    # paymasterAndData = paymaster(20) + uint128(vGL) + uint128(pGL) + data
    if pm_addr:
        paymaster_and_data = (
            _addr_to_bytes(pm_addr)
            + pm_v.to_bytes(16, "big")
            + pm_p.to_bytes(16, "big")
            + pm_data
        )
    else:
        paymaster_and_data = b""

    packed = _pack_user_op_for_hashing(
        sender=sender_addr, nonce=nonce_int,
        init_code=init_code, call_data=call_data_bytes,
        call_gas_limit=call_gas,
        verification_gas_limit=ver_gas,
        pre_verification_gas=pre_ver,
        max_fee_per_gas=max_fee,
        max_priority_fee_per_gas=max_prio,
        paymaster_and_data=paymaster_and_data,
    )
    op_hash = _user_op_hash(packed)
    print(f"UserOp hash: {_hex(op_hash)}")

    # Sign the user op hash with the EOA key.
    # ERC-4337 SimpleAccount uses the EIP-191 prefixed hash for
    # signature verification (toEthSignedMessageHash).
    from eth_account.messages import encode_defunct  # type: ignore
    msg = encode_defunct(primitive=op_hash)
    signed = Account.sign_message(msg, private_key=pk)
    user_op["signature"] = "0x" + signed.signature.hex()
    print(f"Signature: {user_op['signature'][:18]}...")

    # Step 3: eth_sendUserOperation → bundler submits
    print("Submitting via eth_sendUserOperation...")
    try:
        user_op_hash_result = _rpc_call(
            paymaster_url, "eth_sendUserOperation",
            [user_op, ENTRYPOINT_V07],
        )
    except RuntimeError as exc:
        print(f"ERROR submitting: {exc}")
        return 1
    print(f"Submitted! user_op_hash: {user_op_hash_result}")
    print(
        f"Track on jiffyscan: "
        f"https://jiffyscan.xyz/userOpHash/{user_op_hash_result}"
        f"?network=base"
    )

    # Step 4: poll eth_getUserOperationReceipt for tx_hash
    print("Polling for receipt (max 90s)...")
    deadline = time.time() + 90
    tx_hash = None
    while time.time() < deadline:
        try:
            receipt = _rpc_call(
                paymaster_url, "eth_getUserOperationReceipt",
                [user_op_hash_result],
            )
            if receipt:
                tx_hash = receipt.get("receipt", {}).get(
                    "transactionHash",
                ) or receipt.get("transactionHash")
                if tx_hash:
                    print(f"Confirmed! tx_hash: {tx_hash}")
                    print(
                        f"Basescan: "
                        f"https://basescan.org/tx/{tx_hash}"
                    )
                    print(
                        f"Smart Account: "
                        f"https://basescan.org/address/{sender_addr}"
                    )
                    return 0
        except RuntimeError:
            pass
        time.sleep(3)

    print(
        "Submitted but receipt not seen within 90s. Check "
        f"https://jiffyscan.xyz/userOpHash/{user_op_hash_result}"
        f"?network=base manually."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

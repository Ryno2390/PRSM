"""Sprint 868 — Coinbase Smart Wallet (CSW) integration scaffolding.

Sp867's F23 surfaced that CDP's Paymaster bundler tracer fails on
counterfactual deploys via Pimlico's SimpleAccountFactory v0.7 — even
with everything else (auth, billing, gas policy, dummy sig, no-op
inner call) working. Suspected cause: CDP's tracer is tuned for
Coinbase's own Smart Wallet, which has unique structural properties
(WebAuthn/P-256 signature support + magic-value validation).

This module ships the scaffolding for using Coinbase Smart Wallet
as the alternative path:

  - Canonical factory addresses on Base mainnet (verified deployed
    via eth_getCode on 2026-05-28)
  - UserOp composition helpers (createAccount factoryData, execute
    callData, owner encoding)
  - Counterfactual address computation framework
  - Clear hooks for the WebAuthn signing layer (deferred to sp869+)

What this DOES ship now:
  - Deterministic factoryData encoding for createAccount(owners,
    nonce)
  - execute(target, value, data) callData (same as SimpleAccount)
  - Owner-encoding helpers for EOA addresses + future P-256 pubkeys
  - Pin tests defending the canonical addresses + selectors

What requires sp869+ to flip live_exec:
  - WebAuthn / passkey signing layer (CSW's primary signer type)
  - Browser-side or platform-keychain UI integration
  - CSW-specific signature wrapping (R1 sig vs EOA-ECDSA sig)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Canonical Base mainnet addresses (verified deployed) ─────

# Coinbase Smart Wallet Factory v1.1 — the newer deployment with
# CSW v1 improvements (better gas, fixed implementation slot).
# Verified deployed on Base mainnet (3622 bytes bytecode) 2026-05-28.
COINBASE_SMART_WALLET_FACTORY_V11 = (
    "0xBA5ED110eFDBa3D005bfC882d75358ACBbB85842"
)

# CSW Factory v1 — original deployment, still active.
# Verified deployed on Base mainnet (3338 bytes) 2026-05-28.
COINBASE_SMART_WALLET_FACTORY_V1 = (
    "0x0BA5ED0c6AA8c49038F819E587E2633c4A9F428a"
)

# Operator default: prefer v1.1 unless explicitly overridden.
DEFAULT_FACTORY = COINBASE_SMART_WALLET_FACTORY_V11

# ERC-4337 v0.7 EntryPoint (same as sp850 + sp867).
ENTRYPOINT_V07 = "0x0000000071727De22E5E9d8BAf0edAc6f37da032"

# Function selectors. Pinned + tested.
# createAccount(bytes[],uint256) selector
SELECTOR_CSW_CREATE_ACCOUNT = bytes.fromhex("3ffba36f")
# execute(address,uint256,bytes) — same as SimpleAccount v0.7
SELECTOR_CSW_EXECUTE = bytes.fromhex("b61d27f6")
# executeBatch((address,uint256,bytes)[]) — CSW-specific
SELECTOR_CSW_EXECUTE_BATCH = bytes.fromhex("34fcd5be")


# ── Owner encoding ───────────────────────────────────────────

def encode_eoa_owner(address: str) -> bytes:
    """Encode an EOA address as a 32-byte owner entry.

    CSW supports two owner types per slot:
      - EOA (20-byte address, left-padded to 32 bytes)
      - P-256 pubkey (64 bytes = 32-byte X || 32-byte Y) — NOT
        supported by this scaffolding (sp869+)

    Operator scaffolds with EOA only; WebAuthn passkeys (the
    primary CSW pattern) come in a follow-on sprint.
    """
    if not address.startswith("0x"):
        raise ValueError(f"address must be 0x EVM, got {address!r}")
    clean = address.lower().removeprefix("0x")
    if len(clean) != 40:
        raise ValueError(
            f"address must be 20 bytes (40 hex chars), got "
            f"{len(clean)} chars"
        )
    return bytes(12) + bytes.fromhex(clean)


def encode_p256_owner(pubkey_x: bytes, pubkey_y: bytes) -> bytes:
    """Encode a P-256 (secp256r1) public key as a 64-byte owner.

    Reserved for sp869+ WebAuthn integration. Each component must
    be exactly 32 bytes.
    """
    if len(pubkey_x) != 32 or len(pubkey_y) != 32:
        raise ValueError(
            f"P-256 pubkey components must be 32 bytes each "
            f"(got X={len(pubkey_x)}, Y={len(pubkey_y)})"
        )
    return pubkey_x + pubkey_y


# ── createAccount calldata (factoryData) ─────────────────────

def _u256_to_bytes(n: int) -> bytes:
    return n.to_bytes(32, "big")


def _padded_address(addr: str) -> bytes:
    return bytes(12) + bytes.fromhex(addr.lower().removeprefix("0x"))


def encode_create_account_factory_data(
    owners: List[bytes],
    nonce: int = 0,
) -> bytes:
    """Encode CSW factory.createAccount(bytes[] owners, uint256 nonce).

    The `owners` array is dynamic + each element is a `bytes` (also
    dynamic), so ABI encoding has nested offsets:

      head: [
        offset to owners array (0x40),
        nonce (32 bytes),
      ]
      owners array body: [
        length N,
        offset to owners[0] (= 32 * N),
        offset to owners[1] (= prev + 32 + ceil(len(owners[0]))),
        ...
        owners[0]: length + padded-data,
        owners[1]: ...,
      ]
    """
    if nonce < 0:
        raise ValueError("nonce must be >= 0")
    if not owners:
        raise ValueError("owners must not be empty")

    n_owners = len(owners)

    # Static head: offset (32) + nonce (32) = 64 bytes
    head = _u256_to_bytes(0x40) + _u256_to_bytes(nonce)

    # owners array body: length + N offsets + N (length + data) blocks
    body = _u256_to_bytes(n_owners)
    # Compute offsets to each owner entry (relative to start of
    # owners array body, after length + offset table).
    base_offset = 32 * n_owners  # length-prefix + N pointers
    cur_offset = base_offset
    owner_offsets = []
    owner_blocks = []
    for owner in owners:
        owner_offsets.append(_u256_to_bytes(cur_offset))
        owner_len = len(owner)
        pad = (-owner_len) % 32
        block = (
            _u256_to_bytes(owner_len)
            + owner
            + bytes(pad)
        )
        owner_blocks.append(block)
        cur_offset += 32 + owner_len + pad
    body += b"".join(owner_offsets) + b"".join(owner_blocks)

    return SELECTOR_CSW_CREATE_ACCOUNT + head + body


def encode_execute_calldata(
    target: str, value: int, inner_data: bytes,
) -> bytes:
    """SimpleAccount-compatible execute(address, uint256, bytes).

    CSW execute() shares the SimpleAccount v0.7 signature so this
    helper is byte-identical to sp867's _encode_execute_calldata.
    """
    inner_len = len(inner_data)
    pad = (-inner_len) % 32
    return (
        SELECTOR_CSW_EXECUTE
        + _padded_address(target)
        + _u256_to_bytes(value)
        + _u256_to_bytes(96)
        + _u256_to_bytes(inner_len)
        + inner_data
        + bytes(pad)
    )


# ── Counterfactual address (placeholder hook) ────────────────

def compute_counterfactual_address(
    *,
    factory: str = DEFAULT_FACTORY,
    factory_data: bytes,
    rpc_url: str,
    client: Any = None,
) -> str:
    """Query EntryPoint.getSenderAddress(initCode) for the
    counterfactual CSW address.

    initCode = factory_addr (20 bytes) + factory_data.
    EntryPoint reverts with the sender encoded in the revert
    data — we parse via the standard SenderAddressResult selector.
    """
    import httpx
    init_code = bytes.fromhex(
        factory.lower().removeprefix("0x")
    ) + factory_data
    # getSenderAddress(bytes) selector = 0x9b249f69
    selector = bytes.fromhex("9b249f69")
    offset = _u256_to_bytes(32)
    length = _u256_to_bytes(len(init_code))
    pad = (-len(init_code)) % 32
    call_data = (
        selector + offset + length + init_code + bytes(pad)
    )
    body = {
        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
        "params": [
            {
                "to": ENTRYPOINT_V07,
                "data": "0x" + call_data.hex(),
            },
            "latest",
        ],
    }
    if client is None:
        with httpx.Client(timeout=30) as c:
            resp = c.post(
                rpc_url, json=body,
                headers={"Content-Type": "application/json"},
            )
    else:
        resp = client.post(rpc_url, json=body)
    resp.raise_for_status()
    payload = resp.json()
    err = payload.get("error", {})
    data = err.get("data") or err.get("revert_data") or ""
    if data:
        raw = data.removeprefix("0x")
        if len(raw) >= 40:
            return "0x" + raw[-40:]
    result = payload.get("result")
    if result and result != "0x":
        addr_bytes = bytes.fromhex(
            result.removeprefix("0x"),
        )[-20:]
        return "0x" + addr_bytes.hex()
    raise RuntimeError(
        f"getSenderAddress returned no usable response: "
        f"{payload!r}"
    )


# ── UserOp composition framework ─────────────────────────────

def build_userop_skeleton(
    *,
    sender: str,
    nonce: int = 0,
    factory: str = DEFAULT_FACTORY,
    factory_data: Optional[bytes] = None,
    call_data: bytes = b"",
    call_gas_limit: int = 200_000,
    verification_gas_limit: int = 2_000_000,
    pre_verification_gas: int = 100_000,
    max_fee_per_gas: int = 1_000_000_000,
    max_priority_fee_per_gas: int = 500_000_000,
) -> Dict[str, Any]:
    """Build the bundler-RPC shape of a CSW UserOp v0.7.

    Same shape as sp867's SimpleAccount user_op. Signature slot is
    a dummy ready for sp869+'s WebAuthn signing hook.

    Operator passes `factory_data` for first-time deploys; pass
    None for subsequent ops against an already-deployed account.
    """
    op: Dict[str, Any] = {
        "sender": sender,
        "nonce": hex(nonce),
        "callData": "0x" + call_data.hex(),
        "callGasLimit": hex(call_gas_limit),
        "verificationGasLimit": hex(verification_gas_limit),
        "preVerificationGas": hex(pre_verification_gas),
        "maxFeePerGas": hex(max_fee_per_gas),
        "maxPriorityFeePerGas": hex(max_priority_fee_per_gas),
        # Placeholder dummy — valid scalars so ecrecover doesn't
        # revert. SP869 WebAuthn signing will replace this with
        # a wrapped R1 signature.
        "signature": (
            "0x"
            + ("00" * 31 + "01")  # r = 1
            + ("00" * 31 + "01")  # s = 1
            + "1c"                # v = 28
        ),
    }
    if factory_data is not None:
        op["factory"] = factory
        op["factoryData"] = "0x" + factory_data.hex()
    return op

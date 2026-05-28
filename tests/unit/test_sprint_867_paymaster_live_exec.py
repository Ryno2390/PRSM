"""Sprint 867 — sponsored UserOp encoding pin tests.

Defends the deterministic ABI encoding + EIP-4337 v0.7 hashing
that produces the byte-identical UserOp the bundler validates
server-side. Any drift = bundler rejects with "invalid signature".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to sys.path so the sprint script imports
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "scripts"),
)

from sprint_867_paymaster_live_exec import (  # noqa: E402
    AERODROME_ROUTER,
    CHAIN_ID,
    ENTRYPOINT_V07,
    FTNS_TOKEN,
    SELECTOR_APPROVE,
    SELECTOR_CREATE_ACCOUNT,
    SELECTOR_EXECUTE,
    SIMPLE_ACCOUNT_FACTORY_V07,
    _addr_to_bytes,
    _encode_approve_calldata,
    _encode_create_account_calldata,
    _encode_execute_calldata,
    _pack_user_op_for_hashing,
    _padded_address,
    _u256_to_bytes,
    _user_op_hash,
)


# ── Canonical address sanity ─────────────────────────────────

def test_entrypoint_v07_canonical_address():
    """Verifies the pinned EntryPoint matches Coinbase + Pimlico
    + eth-infinitism canonical v0.7 deployment."""
    assert ENTRYPOINT_V07 == (
        "0x0000000071727De22E5E9d8BAf0edAc6f37da032"
    )


def test_chain_id_base_mainnet():
    assert CHAIN_ID == 8453  # Base mainnet


def test_ftns_canonical_address_pinned():
    """FTNS on Base mainnet — must match networks.py base config."""
    assert FTNS_TOKEN == (
        "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
    )


def test_aerodrome_router_canonical_address_pinned():
    """Same router pinned in sp855's swap envelope — they must
    match or the sp867 sponsored approve targets a different
    Aerodrome version than sp855's swap submission would later
    use."""
    assert AERODROME_ROUTER == (
        "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
    )


# ── Function selectors ───────────────────────────────────────

def test_selector_create_account():
    """Pin: createAccount(address,uint256) → 0x5fbfb9cf."""
    assert SELECTOR_CREATE_ACCOUNT.hex() == "5fbfb9cf"


def test_selector_execute():
    """Pin: execute(address,uint256,bytes) → 0xb61d27f6."""
    assert SELECTOR_EXECUTE.hex() == "b61d27f6"


def test_selector_approve():
    """Pin: approve(address,uint256) → 0x095ea7b3."""
    assert SELECTOR_APPROVE.hex() == "095ea7b3"


# ── Primitive encoding ───────────────────────────────────────

def test_addr_to_bytes_20_bytes():
    b = _addr_to_bytes("0x" + "ab" * 20)
    assert len(b) == 20
    assert b == b"\xab" * 20


def test_padded_address_32_bytes():
    p = _padded_address("0x" + "ab" * 20)
    assert len(p) == 32
    assert p == b"\x00" * 12 + b"\xab" * 20


def test_u256_to_bytes_zero():
    assert _u256_to_bytes(0) == b"\x00" * 32


def test_u256_to_bytes_max_uint128():
    """Max uint128 = (1<<128)-1 packed into uint256."""
    val = (1 << 128) - 1
    encoded = _u256_to_bytes(val)
    assert len(encoded) == 32
    # Low 16 bytes = ff*16; high 16 bytes = 00
    assert encoded[:16] == b"\x00" * 16
    assert encoded[16:] == b"\xff" * 16


# ── createAccount calldata ───────────────────────────────────

def test_create_account_calldata_shape():
    owner = "0x" + "ab" * 20
    cd = _encode_create_account_calldata(owner, 0)
    # selector (4) + address (32) + salt (32) = 68 bytes
    assert len(cd) == 68
    assert cd[:4] == SELECTOR_CREATE_ACCOUNT
    # owner padded to 32
    assert cd[4:36] == b"\x00" * 12 + b"\xab" * 20
    # salt = 0
    assert cd[36:68] == b"\x00" * 32


def test_create_account_with_nonzero_salt():
    owner = "0x" + "cd" * 20
    cd = _encode_create_account_calldata(owner, 42)
    # salt encoded as big-endian uint256
    salt = int.from_bytes(cd[36:68], "big")
    assert salt == 42


# ── approve calldata ─────────────────────────────────────────

def test_approve_calldata_shape():
    cd = _encode_approve_calldata(AERODROME_ROUTER, 0)
    assert len(cd) == 68  # 4 + 32 + 32
    assert cd[:4] == SELECTOR_APPROVE


def test_approve_calldata_amount_zero():
    """Sp867's specific case: spender = AerodromeRouter, amount=0
    (the zero-asset no-op exercise of the sponsorship codepath)."""
    cd = _encode_approve_calldata(AERODROME_ROUTER, 0)
    amount = int.from_bytes(cd[36:68], "big")
    assert amount == 0


def test_approve_calldata_amount_max():
    cd = _encode_approve_calldata(
        AERODROME_ROUTER, 2**256 - 1,
    )
    amount = int.from_bytes(cd[36:68], "big")
    assert amount == 2**256 - 1


# ── execute calldata ─────────────────────────────────────────

def test_execute_calldata_pads_inner_bytes():
    """SimpleAccount.execute encodes inner bytes ABI-style:
    head (selector + target + value + offset) + length + data
    padded to 32-byte multiple."""
    inner = b"\x01\x02\x03"  # 3 bytes
    cd = _encode_execute_calldata(
        FTNS_TOKEN, 0, inner,
    )
    # Selector (4) + target (32) + value (32) + offset (32) +
    # length (32) + inner (padded to 32) = 164
    assert len(cd) == 164
    assert cd[:4] == SELECTOR_EXECUTE
    # Offset to bytes = 96 (3 * 32 after selector)
    offset = int.from_bytes(cd[68:100], "big")
    assert offset == 96
    # Length = 3
    length = int.from_bytes(cd[100:132], "big")
    assert length == 3
    # Inner data + 29 bytes pad
    assert cd[132:135] == inner
    assert cd[135:164] == b"\x00" * 29


def test_execute_calldata_with_aligned_inner_bytes():
    """If inner is already 32-byte aligned, no extra padding."""
    inner = b"\xaa" * 32
    cd = _encode_execute_calldata(FTNS_TOKEN, 0, inner)
    assert len(cd) == 4 + 32 + 32 + 32 + 32 + 32  # = 164


def test_execute_calldata_real_approve_nested():
    """Full real-world case: execute(FTNS, 0, approve(Router, 0))
    — the actual sp867 callData."""
    inner = _encode_approve_calldata(AERODROME_ROUTER, 0)
    assert len(inner) == 68
    cd = _encode_execute_calldata(FTNS_TOKEN, 0, inner)
    # 4 + 32 + 32 + 32 + 32 + 68 + 28 (pad to 96 = 3*32) = 228
    assert len(cd) == 228
    # FTNS target encoded
    assert cd[4:36] == _padded_address(FTNS_TOKEN)
    # value = 0
    assert cd[36:68] == b"\x00" * 32
    # Offset = 96
    assert int.from_bytes(cd[68:100], "big") == 96
    # Length = 68
    assert int.from_bytes(cd[100:132], "big") == 68
    # Inner approve calldata
    assert cd[132:200] == inner


# ── Packed UserOp hashing (EIP-4337 v0.7) ────────────────────

def test_pack_user_op_known_size():
    """The packed structure for hashing is 8 fields × 32 bytes
    each = 256 bytes total (3 hashes + 5 padded primitives)."""
    packed = _pack_user_op_for_hashing(
        sender="0x" + "11" * 20,
        nonce=0,
        init_code=b"",
        call_data=b"",
        call_gas_limit=200_000,
        verification_gas_limit=500_000,
        pre_verification_gas=100_000,
        max_fee_per_gas=2_000_000_000,
        max_priority_fee_per_gas=1_000_000_000,
        paymaster_and_data=b"",
    )
    assert len(packed) == 256


def test_pack_user_op_account_gas_limits_packing():
    """accountGasLimits packs verificationGasLimit (high 128 bits)
    | callGasLimit (low 128 bits) into a single uint256."""
    packed = _pack_user_op_for_hashing(
        sender="0x" + "00" * 20, nonce=0,
        init_code=b"", call_data=b"",
        call_gas_limit=0xAAAA,
        verification_gas_limit=0xBBBB,
        pre_verification_gas=0,
        max_fee_per_gas=0, max_priority_fee_per_gas=0,
        paymaster_and_data=b"",
    )
    # accountGasLimits at offset 4*32 = 128
    account_gas = int.from_bytes(packed[128:160], "big")
    # High 128 bits = verification (0xBBBB), low 128 bits = call (0xAAAA)
    assert (account_gas >> 128) == 0xBBBB
    assert (account_gas & ((1 << 128) - 1)) == 0xAAAA


def test_pack_user_op_gas_fees_packing():
    """gasFees packs maxPriorityFeePerGas (high) | maxFeePerGas
    (low) — both are uint128 in PackedUserOperation."""
    packed = _pack_user_op_for_hashing(
        sender="0x" + "00" * 20, nonce=0,
        init_code=b"", call_data=b"",
        call_gas_limit=0, verification_gas_limit=0,
        pre_verification_gas=0,
        max_fee_per_gas=0xCCCC,
        max_priority_fee_per_gas=0xDDDD,
        paymaster_and_data=b"",
    )
    # gasFees at offset 6*32 = 192
    gas_fees = int.from_bytes(packed[192:224], "big")
    assert (gas_fees >> 128) == 0xDDDD  # priority high
    assert (gas_fees & ((1 << 128) - 1)) == 0xCCCC  # max low


def test_user_op_hash_includes_entry_point_and_chain():
    """userOpHash = keccak256(packedHash + entryPoint + chainId).
    Different chain_id MUST produce different hash."""
    packed = _pack_user_op_for_hashing(
        sender="0x" + "11" * 20, nonce=0,
        init_code=b"", call_data=b"",
        call_gas_limit=100_000,
        verification_gas_limit=100_000,
        pre_verification_gas=50_000,
        max_fee_per_gas=1_000_000_000,
        max_priority_fee_per_gas=1_000_000_000,
        paymaster_and_data=b"",
    )
    h_base = _user_op_hash(
        packed, entry_point=ENTRYPOINT_V07, chain_id=8453,
    )
    h_mainnet = _user_op_hash(
        packed, entry_point=ENTRYPOINT_V07, chain_id=1,
    )
    assert h_base != h_mainnet
    assert len(h_base) == 32


def test_user_op_hash_deterministic():
    """Same packed input → same hash. Catches accidental
    nondeterminism (e.g., importing time-dependent salt)."""
    packed = _pack_user_op_for_hashing(
        sender="0x" + "22" * 20, nonce=42,
        init_code=b"\xab\xcd",
        call_data=b"\xef",
        call_gas_limit=100_000,
        verification_gas_limit=100_000,
        pre_verification_gas=50_000,
        max_fee_per_gas=1_000_000_000,
        max_priority_fee_per_gas=1_000_000_000,
        paymaster_and_data=b"\xff" * 50,
    )
    h1 = _user_op_hash(packed)
    h2 = _user_op_hash(packed)
    assert h1 == h2

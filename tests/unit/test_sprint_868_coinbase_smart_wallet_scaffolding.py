"""Sprint 868 — Coinbase Smart Wallet scaffolding pin tests."""
from __future__ import annotations

import pytest

from prsm.economy.web3.coinbase_smart_wallet import (
    COINBASE_SMART_WALLET_FACTORY_V1,
    COINBASE_SMART_WALLET_FACTORY_V11,
    DEFAULT_FACTORY,
    ENTRYPOINT_V07,
    SELECTOR_CSW_CREATE_ACCOUNT,
    SELECTOR_CSW_EXECUTE,
    SELECTOR_CSW_EXECUTE_BATCH,
    build_userop_skeleton,
    encode_create_account_factory_data,
    encode_eoa_owner,
    encode_execute_calldata,
    encode_p256_owner,
)


# ── Canonical addresses ──────────────────────────────────────

def test_factory_v11_address_pinned():
    """Verified deployed on Base mainnet via eth_getCode probe
    2026-05-28 (3622 bytes bytecode)."""
    assert COINBASE_SMART_WALLET_FACTORY_V11 == (
        "0xBA5ED110eFDBa3D005bfC882d75358ACBbB85842"
    )


def test_factory_v1_address_pinned():
    assert COINBASE_SMART_WALLET_FACTORY_V1 == (
        "0x0BA5ED0c6AA8c49038F819E587E2633c4A9F428a"
    )


def test_default_factory_is_v11():
    """v1.1 is the newer + recommended deployment. Operators can
    opt into v1 if needed but default is v1.1."""
    assert DEFAULT_FACTORY == COINBASE_SMART_WALLET_FACTORY_V11


def test_entrypoint_matches_sp867():
    """CSW + SimpleAccount both target EntryPoint v0.7 — must
    match across both code paths to keep the bundler happy."""
    assert ENTRYPOINT_V07 == (
        "0x0000000071727De22E5E9d8BAf0edAc6f37da032"
    )


# ── Function selectors ───────────────────────────────────────

def test_selector_create_account_pinned():
    """keccak256('createAccount(bytes[],uint256)')[0:4] = 0x3ffba36f.
    Verified via eth_utils.keccak in development."""
    assert SELECTOR_CSW_CREATE_ACCOUNT.hex() == "3ffba36f"


def test_selector_execute_matches_simple_account():
    """CSW execute(address,uint256,bytes) is byte-identical to
    SimpleAccount v0.7 execute — pinned same value."""
    assert SELECTOR_CSW_EXECUTE.hex() == "b61d27f6"


def test_selector_execute_batch_pinned():
    assert SELECTOR_CSW_EXECUTE_BATCH.hex() == "34fcd5be"


# ── Owner encoding ───────────────────────────────────────────

def test_encode_eoa_owner_pads_to_32_bytes():
    """20-byte address left-padded to 32 bytes."""
    addr = "0x" + "ab" * 20
    encoded = encode_eoa_owner(addr)
    assert len(encoded) == 32
    assert encoded == bytes(12) + bytes.fromhex("ab" * 20)


def test_encode_eoa_owner_lowercases():
    """Checksum-insensitive encoding — owner identity is the
    20 raw bytes."""
    upper = encode_eoa_owner("0x" + "AB" * 20)
    lower = encode_eoa_owner("0x" + "ab" * 20)
    assert upper == lower


def test_encode_eoa_owner_rejects_bad_address():
    with pytest.raises(ValueError):
        encode_eoa_owner("0xabc")
    with pytest.raises(ValueError):
        encode_eoa_owner("not-0x" + "ab" * 20)


def test_encode_p256_owner_64_bytes():
    """P-256 pubkey is 32-byte X || 32-byte Y = 64 bytes total."""
    x = b"\x01" * 32
    y = b"\x02" * 32
    encoded = encode_p256_owner(x, y)
    assert len(encoded) == 64
    assert encoded[:32] == x
    assert encoded[32:] == y


def test_encode_p256_owner_rejects_wrong_length():
    with pytest.raises(ValueError):
        encode_p256_owner(b"\x01" * 31, b"\x02" * 32)
    with pytest.raises(ValueError):
        encode_p256_owner(b"\x01" * 32, b"\x02" * 33)


# ── createAccount factoryData encoding ───────────────────────

def test_create_account_factory_data_single_eoa_owner():
    """One EOA owner + nonce=0. Verifies the ABI head structure
    + array body offsets."""
    owner = encode_eoa_owner("0x" + "11" * 20)
    fd = encode_create_account_factory_data([owner], nonce=0)
    # Selector (4) + offset (32) + nonce (32) + arr length (32) +
    # arr offset (32) + owner length (32) + owner (32) = 196 bytes
    assert fd[:4] == SELECTOR_CSW_CREATE_ACCOUNT
    # Head: offset to owners (always 0x40)
    assert int.from_bytes(fd[4:36], "big") == 0x40
    # Head: nonce
    assert int.from_bytes(fd[36:68], "big") == 0
    # Owners array length
    assert int.from_bytes(fd[68:100], "big") == 1
    # Offset to owners[0]
    assert int.from_bytes(fd[100:132], "big") == 32  # 32 * 1
    # owners[0] length
    assert int.from_bytes(fd[132:164], "big") == 32
    # owners[0] data
    assert fd[164:196] == owner


def test_create_account_factory_data_multi_owner():
    o1 = encode_eoa_owner("0x" + "11" * 20)
    o2 = encode_eoa_owner("0x" + "22" * 20)
    fd = encode_create_account_factory_data([o1, o2], nonce=0)
    # owners length = 2
    assert int.from_bytes(fd[68:100], "big") == 2
    # owners[0] offset = 64 (after length + 2 ptrs)
    assert int.from_bytes(fd[100:132], "big") == 64
    # owners[1] offset = 64 + 32 + 32 = 128
    assert int.from_bytes(fd[132:164], "big") == 128


def test_create_account_factory_data_nonzero_nonce():
    owner = encode_eoa_owner("0x" + "11" * 20)
    fd = encode_create_account_factory_data([owner], nonce=42)
    assert int.from_bytes(fd[36:68], "big") == 42


def test_create_account_factory_data_rejects_empty_owners():
    with pytest.raises(ValueError):
        encode_create_account_factory_data([], nonce=0)


def test_create_account_factory_data_rejects_negative_nonce():
    owner = encode_eoa_owner("0x" + "11" * 20)
    with pytest.raises(ValueError):
        encode_create_account_factory_data([owner], nonce=-1)


def test_create_account_with_p256_owner_64_byte_block():
    """A P-256 owner is 64 bytes — different from EOA's 32. The
    encoder must handle variable-length owners cleanly."""
    p256_owner = encode_p256_owner(b"\xaa" * 32, b"\xbb" * 32)
    fd = encode_create_account_factory_data(
        [p256_owner], nonce=0,
    )
    # owners[0] length = 64
    assert int.from_bytes(fd[132:164], "big") == 64
    # owners[0] data: 32 bytes of 0xaa + 32 bytes of 0xbb
    assert fd[164:228] == p256_owner


# ── execute calldata ─────────────────────────────────────────

def test_execute_calldata_matches_sp867_shape():
    """CSW execute is byte-identical to SimpleAccount execute.
    Same selector + same ABI encoding (target + value + offset +
    length + data + pad)."""
    inner = bytes.fromhex("095ea7b3" + "00" * 64)  # approve(0, 0)
    cd = encode_execute_calldata(
        "0x" + "ff" * 20, 0, inner,
    )
    assert cd[:4] == SELECTOR_CSW_EXECUTE
    # offset to bytes = 96
    assert int.from_bytes(cd[68:100], "big") == 96
    # length of inner
    assert int.from_bytes(cd[100:132], "big") == 68


# ── UserOp skeleton ──────────────────────────────────────────

def test_userop_skeleton_has_canonical_fields_no_factory():
    """Subsequent-tx UserOp (account already deployed) — no
    factory / factoryData fields."""
    op = build_userop_skeleton(
        sender="0x" + "11" * 20, nonce=5,
        call_data=b"",
    )
    assert op["sender"] == "0x" + "11" * 20
    assert op["nonce"] == "0x5"
    assert "factory" not in op
    assert "factoryData" not in op
    assert op["callData"] == "0x"


def test_userop_skeleton_with_factory_includes_factory_fields():
    """First-time deploy UserOp — factory + factoryData populated."""
    fd = b"\xde\xad\xbe\xef"
    op = build_userop_skeleton(
        sender="0x" + "11" * 20,
        factory_data=fd,
    )
    assert op["factory"] == DEFAULT_FACTORY
    assert op["factoryData"] == "0xdeadbeef"


def test_userop_skeleton_dummy_sig_has_valid_ecdsa_shape():
    """Sp867 lesson: dummy sig with v=0 reverts ecrecover (AA23).
    Use v=28 + valid r/s so signature recovery returns SOME
    address, validation returns SIG_VALIDATION_FAILED cleanly."""
    op = build_userop_skeleton(sender="0x" + "11" * 20)
    sig = op["signature"].removeprefix("0x")
    assert len(sig) == 130  # 32 r + 32 s + 1 v
    # v = 28 (0x1c)
    assert sig[-2:] == "1c"
    # r and s are small valid scalars
    r = int(sig[:64], 16)
    s = int(sig[64:128], 16)
    secp256k1_n = (
        0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
    )
    assert 1 <= r < secp256k1_n
    assert 1 <= s < secp256k1_n // 2  # low-s canonical

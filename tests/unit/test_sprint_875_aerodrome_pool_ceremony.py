"""Sprint 875 — Aerodrome pool ceremony helper pin tests."""
from __future__ import annotations

import json

import pytest

from prsm.economy.web3.aerodrome_pool_ceremony import (
    AERODROME_POOL_FACTORY_MAINNET,
    AERODROME_ROUTER_V2_MAINNET,
    FTNS_BASE_MAINNET,
    FTNS_BASE_SEPOLIA,
    MAINNET_CONFIG,
    SELECTOR_AERODROME_ADD_LIQUIDITY,
    SELECTOR_ERC20_APPROVE,
    SEPOLIA_CONFIG,
    USDC_BASE_MAINNET,
    USDC_BASE_SEPOLIA,
    build_ceremony_batch,
    build_runbook_markdown,
    encode_add_liquidity_calldata,
    encode_approve_calldata,
)


# ── Canonical addresses pinned ───────────────────────────────

def test_mainnet_addresses_match_sp855():
    """Drift here = ceremony hits wrong contracts. Must match
    sp855's pinned addresses byte-for-byte."""
    assert USDC_BASE_MAINNET == (
        "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    )
    assert FTNS_BASE_MAINNET == (
        "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
    )
    assert AERODROME_ROUTER_V2_MAINNET == (
        "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
    )
    assert AERODROME_POOL_FACTORY_MAINNET == (
        "0x420DD381b31aEf6683db6B902084cB0FFECe40Da"
    )


def test_sepolia_addresses_pinned():
    """Sepolia USDC = Circle's testnet deployment. FTNS = T1
    deploy from networks.py."""
    assert USDC_BASE_SEPOLIA == (
        "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
    )
    assert FTNS_BASE_SEPOLIA == (
        "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
    )


def test_mainnet_config_chain_id():
    assert MAINNET_CONFIG.chain_id == 8453
    assert MAINNET_CONFIG.name == "mainnet"


def test_sepolia_config_chain_id():
    assert SEPOLIA_CONFIG.chain_id == 84532
    assert SEPOLIA_CONFIG.name == "sepolia"


# ── Selectors ────────────────────────────────────────────────

def test_erc20_approve_selector():
    """keccak256('approve(address,uint256)')[:4] = 0x095ea7b3."""
    assert SELECTOR_ERC20_APPROVE.hex() == "095ea7b3"


def test_aerodrome_add_liquidity_selector():
    """keccak256('addLiquidity(address,address,bool,uint256,
    uint256,uint256,uint256,address,uint256)')[:4] = 0xe8e33700"""
    assert SELECTOR_AERODROME_ADD_LIQUIDITY.hex() == "e8e33700"


# ── approve calldata ────────────────────────────────────────

def test_approve_calldata_canonical_shape():
    cd = encode_approve_calldata(
        AERODROME_ROUTER_V2_MAINNET, 1_000_000,
    )
    assert cd[:4] == SELECTOR_ERC20_APPROVE
    assert len(cd) == 4 + 32 + 32  # selector + addr + amount
    # Spender padded
    assert cd[4:36][-20:] == bytes.fromhex(
        AERODROME_ROUTER_V2_MAINNET.lower().removeprefix("0x"),
    )
    # Amount
    assert int.from_bytes(cd[36:68], "big") == 1_000_000


# ── addLiquidity calldata ────────────────────────────────────

def test_add_liquidity_calldata_full_shape():
    """All 9 params encoded in canonical order."""
    cd = encode_add_liquidity_calldata(
        token_a="0x" + "11" * 20,
        token_b="0x" + "22" * 20,
        stable=False,
        amount_a_desired=1000,
        amount_b_desired=2000,
        amount_a_min=950,
        amount_b_min=1900,
        to="0x" + "33" * 20,
        deadline=1_800_000_000,
    )
    assert cd[:4] == SELECTOR_AERODROME_ADD_LIQUIDITY
    # 4 + 9 * 32 = 292 bytes (booleans pad to 32 too)
    assert len(cd) == 4 + 9 * 32


def test_add_liquidity_stable_flag_encoding():
    """Stable=true encodes as 1; stable=false encodes as 0."""
    cd_volatile = encode_add_liquidity_calldata(
        token_a="0x" + "11" * 20, token_b="0x" + "22" * 20,
        stable=False,
        amount_a_desired=1, amount_b_desired=1,
        amount_a_min=1, amount_b_min=1,
        to="0x" + "33" * 20, deadline=1,
    )
    cd_stable = encode_add_liquidity_calldata(
        token_a="0x" + "11" * 20, token_b="0x" + "22" * 20,
        stable=True,
        amount_a_desired=1, amount_b_desired=1,
        amount_a_min=1, amount_b_min=1,
        to="0x" + "33" * 20, deadline=1,
    )
    # Stable flag at offset 4 + 32*2 = 68, value padded
    assert int.from_bytes(cd_volatile[68:100], "big") == 0
    assert int.from_bytes(cd_stable[68:100], "big") == 1


# ── Ceremony batch ────────────────────────────────────────────

def test_batch_three_transactions():
    """USDC.approve + FTNS.approve + Router.addLiquidity."""
    batch = build_ceremony_batch(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=50_000 * 10**6,  # 50k USDC
        seed_ftns_units=50_000 * 10**18,  # 50k FTNS
    )
    assert len(batch["transactions"]) == 3


def test_batch_tx1_targets_usdc():
    batch = build_ceremony_batch(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=1_000_000,
        seed_ftns_units=10**18,
    )
    assert batch["transactions"][0]["to"] == USDC_BASE_MAINNET
    assert batch["transactions"][0]["data"].startswith("0x095ea7b3")


def test_batch_tx2_targets_ftns():
    batch = build_ceremony_batch(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=1_000_000,
        seed_ftns_units=10**18,
    )
    assert batch["transactions"][1]["to"] == FTNS_BASE_MAINNET
    assert batch["transactions"][1]["data"].startswith("0x095ea7b3")


def test_batch_tx3_targets_router():
    batch = build_ceremony_batch(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=1_000_000,
        seed_ftns_units=10**18,
    )
    assert batch["transactions"][2]["to"] == (
        AERODROME_ROUTER_V2_MAINNET
    )
    assert batch["transactions"][2]["data"].startswith("0xe8e33700")


def test_batch_chain_id_matches_network():
    mainnet = build_ceremony_batch(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=1, seed_ftns_units=1,
    )
    assert mainnet["chainId"] == "8453"
    sepolia = build_ceremony_batch(
        network=SEPOLIA_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=1, seed_ftns_units=1,
    )
    assert sepolia["chainId"] == "84532"


def test_batch_meta_includes_seed_details():
    """Meta description must include seed amounts + slippage so
    co-signers see the ceremony intent in the Safe UI."""
    batch = build_ceremony_batch(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=50_000 * 10**6,
        seed_ftns_units=50_000 * 10**18,
        slippage_bps=200,
    )
    desc = batch["meta"]["description"]
    assert "50000" in desc
    assert "slippage_bps=200" in desc
    assert MAINNET_CONFIG.name in batch["meta"]["name"].lower()


def test_batch_rejects_zero_seed_amount():
    with pytest.raises(ValueError):
        build_ceremony_batch(
            network=MAINNET_CONFIG,
            seeder_safe="0x" + "ab" * 20,
            seed_usdc_units=0, seed_ftns_units=10**18,
        )
    with pytest.raises(ValueError):
        build_ceremony_batch(
            network=MAINNET_CONFIG,
            seeder_safe="0x" + "ab" * 20,
            seed_usdc_units=10**6, seed_ftns_units=0,
        )


def test_batch_rejects_out_of_range_slippage():
    with pytest.raises(ValueError):
        build_ceremony_batch(
            network=MAINNET_CONFIG,
            seeder_safe="0x" + "ab" * 20,
            seed_usdc_units=10**6, seed_ftns_units=10**18,
            slippage_bps=-1,
        )
    with pytest.raises(ValueError):
        build_ceremony_batch(
            network=MAINNET_CONFIG,
            seeder_safe="0x" + "ab" * 20,
            seed_usdc_units=10**6, seed_ftns_units=10**18,
            slippage_bps=10_001,
        )


def test_batch_slippage_math_in_addliquidity():
    """amount_a_min = amount_a_desired × (1 - slippage_bps/10000).
    The Aerodrome Router rejects with INSUFFICIENT_AMOUNT if
    realized < min, so this defines the operator's tolerance for
    price moves between sign + execute."""
    batch = build_ceremony_batch(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=10**6, seed_ftns_units=10**18,
        slippage_bps=500,  # 5%
    )
    tx3_data = bytes.fromhex(
        batch["transactions"][2]["data"].removeprefix("0x"),
    )
    # addLiquidity layout: selector(4) + tokenA(32) + tokenB(32) +
    # stable(32) + aDesired(32) + bDesired(32) + aMin(32) + bMin(32) +
    # to(32) + deadline(32). So aMin starts at 4 + 5*32 = 164.
    a_min = int.from_bytes(tx3_data[164:196], "big")
    b_min = int.from_bytes(tx3_data[196:228], "big")
    # Determine which of USDC/FTNS is token_a (sorted by address)
    if int(USDC_BASE_MAINNET, 16) < int(FTNS_BASE_MAINNET, 16):
        a_desired, b_desired = 10**6, 10**18
    else:
        a_desired, b_desired = 10**18, 10**6
    assert a_min == a_desired * (10_000 - 500) // 10_000
    assert b_min == b_desired * (10_000 - 500) // 10_000


def test_batch_json_serializable():
    """Operator must be able to json.dumps() + paste into Safe UI."""
    batch = build_ceremony_batch(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=10**6, seed_ftns_units=10**18,
    )
    serialized = json.dumps(batch, indent=2)
    reloaded = json.loads(serialized)
    assert reloaded["version"] == "1.0"
    assert len(reloaded["transactions"]) == 3


# ── Runbook markdown ─────────────────────────────────────────

def test_runbook_includes_canonical_addresses():
    """Runbook is read by hardware-wallet co-signers BEFORE
    signing — every address they need to verify must appear
    verbatim."""
    runbook = build_runbook_markdown(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=50_000 * 10**6,
        seed_ftns_units=50_000 * 10**18,
    )
    assert USDC_BASE_MAINNET in runbook
    assert FTNS_BASE_MAINNET in runbook
    assert AERODROME_ROUTER_V2_MAINNET in runbook
    assert AERODROME_POOL_FACTORY_MAINNET in runbook


def test_runbook_includes_opening_price():
    """The opening market price is the load-bearing economic
    decision. The runbook must surface it explicitly so co-signers
    see what the ceremony will set."""
    runbook = build_runbook_markdown(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=50_000 * 10**6,   # 50k USDC
        seed_ftns_units=5_000 * 10**18,   # 5k FTNS → $10/FTNS
    )
    assert "$10.000000 per FTNS" in runbook


def test_runbook_includes_sepolia_rehearsal_pointer():
    """Mainnet runbook MUST point at Sepolia rehearsal as the
    risk-reduction step. The whole point of sp875."""
    runbook = build_runbook_markdown(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=10**6, seed_ftns_units=10**18,
    )
    assert "Sepolia" in runbook
    assert "Rehearsal" in runbook or "rehearsal" in runbook


def test_runbook_describes_3_transactions():
    runbook = build_runbook_markdown(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=10**6, seed_ftns_units=10**18,
    )
    assert "Tx 1" in runbook
    assert "Tx 2" in runbook
    assert "Tx 3" in runbook


def test_runbook_includes_post_ceremony_env_wiring():
    """After ceremony lands, operator must set
    AERODROME_USDC_FTNS_POOL_ADDRESS. Runbook surfaces this."""
    runbook = build_runbook_markdown(
        network=MAINNET_CONFIG,
        seeder_safe="0x" + "ab" * 20,
        seed_usdc_units=10**6, seed_ftns_units=10**18,
    )
    assert "AERODROME_USDC_FTNS_POOL_ADDRESS" in runbook

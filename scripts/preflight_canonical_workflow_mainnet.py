#!/usr/bin/env python3
"""Mainnet pre-flight checker for the canonical-workflow trace.

Read-only. Sends ZERO transactions. Safe to run any number of times.

Verifies the demo-payer wallet has enough Base mainnet ETH for gas
and enough FTNS to cover the trace's 10-FTNS gross before the user
commits to the actual exercise script (which DOES send transactions).

Run AFTER the Path A operational gates have cleared:
  1. Foundation Safe 2-of-3 multisig disbursement of 100 FTNS to the
     demo payer wallet has confirmed on-chain.
  2. Base mainnet ETH funded to the demo payer wallet (~0.005 ETH or
     more).

Usage:

    source ~/.prsm/mainnet-payer.env
    PYTHONPATH=. .venv/bin/python3.14 \\
        scripts/preflight_canonical_workflow_mainnet.py

Exit codes:
    0  all checks pass — safe to run the trace script
    2  required env var missing
    3  ETH balance too low
    4  FTNS balance too low
    5  contract responsiveness check failed
    6  network mismatch (chain id is not Base mainnet 8453)
"""
from __future__ import annotations

import os
import sys

from eth_account import Account
from web3 import Web3


# Defaults match Path A from the runbook. Override via env if needed.
EXPECTED_CHAIN_ID = int(os.environ.get("PRSM_CHAIN_ID", "8453"))
FTNS_TOKEN = os.environ.get(
    "PRSM_FTNS_TOKEN", "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
)
PROVENANCE_REGISTRY = os.environ.get(
    "PRSM_PROVENANCE_REGISTRY",
    "0xdF470BFa9eF310B196801D5105468515d0069915",
)
ROYALTY_DISTRIBUTOR = os.environ.get(
    "PRSM_ROYALTY_DISTRIBUTOR",
    "0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2",
)
EXPLORER = os.environ.get("PRSM_EXPLORER", "https://basescan.org")

# Trace parameters — must match the exercise script.
GROSS_FTNS = 10
ETH_GAS_BUDGET = 0.005  # ETH


def _required(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        print(f"❌ FAIL: env var {name} is required", file=sys.stderr)
        sys.exit(2)
    return v


def main() -> int:
    rpc_url = _required("BASE_SEPOLIA_RPC_URL")  # script reuses this name
    private_key = _required("PRIVATE_KEY")

    web3 = Web3(Web3.HTTPProvider(rpc_url))
    payer = Account.from_key(private_key).address

    print(f"\n=== Mainnet pre-flight ===")
    print(f"RPC:                    {rpc_url}")
    print(f"Demo payer:             {payer}")
    print(f"FTNS token:             {FTNS_TOKEN}")
    print(f"ProvenanceRegistry v1:  {PROVENANCE_REGISTRY}")
    print(f"RoyaltyDistributor:     {ROYALTY_DISTRIBUTOR}")
    print(f"Explorer:               {EXPLORER}")
    print()

    # ── Check 1: chain id ────────────────────────────────────────────
    chain_id = web3.eth.chain_id
    if chain_id != EXPECTED_CHAIN_ID:
        print(
            f"❌ FAIL: chain id mismatch — RPC reports {chain_id}, "
            f"expected {EXPECTED_CHAIN_ID} (Base mainnet)",
            file=sys.stderr,
        )
        return 6
    print(f"✅ chain id 8453 (Base mainnet)")

    # ── Check 2: ETH balance ─────────────────────────────────────────
    eth_wei = web3.eth.get_balance(payer)
    eth = float(Web3.from_wei(eth_wei, "ether"))
    eth_status = "✅" if eth >= ETH_GAS_BUDGET else "❌"
    print(
        f"{eth_status} ETH balance: {eth:.6f} "
        f"(need ≥ {ETH_GAS_BUDGET} for ~3 txs)",
    )
    if eth < ETH_GAS_BUDGET:
        print(
            f"\n   To fund: transfer ≥ 0.005 Base mainnet ETH to {payer}",
            file=sys.stderr,
        )
        return 3

    # ── Check 3: FTNS balance ────────────────────────────────────────
    balance_calldata = (
        Web3.keccak(text="balanceOf(address)")[:4]
        + b"\x00" * 12
        + bytes.fromhex(payer[2:])
    )
    raw = web3.eth.call({
        "to": Web3.to_checksum_address(FTNS_TOKEN),
        "data": balance_calldata,
    })
    ftns_wei = int.from_bytes(raw, "big")
    ftns = ftns_wei / 10**18
    ftns_required = GROSS_FTNS
    ftns_status = "✅" if ftns >= ftns_required else "❌"
    print(
        f"{ftns_status} FTNS balance: {ftns:.4f} "
        f"(need ≥ {ftns_required} for one 10-FTNS gross trace)",
    )
    if ftns < ftns_required:
        print(
            f"\n   To fund: Foundation Safe (2-of-3 multisig at "
            f"0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791) sends "
            f"≥ {ftns_required} FTNS to {payer} via the FTNS contract "
            f"at {FTNS_TOKEN}",
            file=sys.stderr,
        )
        return 4

    # ── Check 4: contract responsiveness (no state mutation) ─────────
    # ProvenanceRegistry — read MAX_ROYALTY_RATE_BPS public constant
    # via call to the function selector. If this returns the expected
    # 9800, the contract is alive at the address.
    max_bps_selector = Web3.keccak(text="MAX_ROYALTY_RATE_BPS()")[:4]
    try:
        raw = web3.eth.call({
            "to": Web3.to_checksum_address(PROVENANCE_REGISTRY),
            "data": max_bps_selector,
        })
        max_bps = int.from_bytes(raw, "big")
        if max_bps != 9800:
            print(
                f"❌ FAIL: ProvenanceRegistry MAX_ROYALTY_RATE_BPS returned "
                f"{max_bps}, expected 9800",
                file=sys.stderr,
            )
            return 5
    except Exception as exc:  # noqa: BLE001
        print(
            f"❌ FAIL: ProvenanceRegistry not responsive at "
            f"{PROVENANCE_REGISTRY}: {exc}",
            file=sys.stderr,
        )
        return 5
    print(
        f"✅ ProvenanceRegistry responsive (MAX_ROYALTY_RATE_BPS=9800)",
    )

    # RoyaltyDistributor — read networkTreasury() public address.
    treasury_selector = Web3.keccak(text="networkTreasury()")[:4]
    try:
        raw = web3.eth.call({
            "to": Web3.to_checksum_address(ROYALTY_DISTRIBUTOR),
            "data": treasury_selector,
        })
        treasury = Web3.to_checksum_address("0x" + raw[-20:].hex())
    except Exception as exc:  # noqa: BLE001
        print(
            f"❌ FAIL: RoyaltyDistributor not responsive at "
            f"{ROYALTY_DISTRIBUTOR}: {exc}",
            file=sys.stderr,
        )
        return 5
    print(f"✅ RoyaltyDistributor.networkTreasury() = {treasury}")
    if treasury.lower() != "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791".lower():
        print(
            f"⚠️  WARNING: networkTreasury is NOT the documented "
            f"Foundation Safe (0x91b0...5791). Investigate before "
            f"proceeding.",
            file=sys.stderr,
        )

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"✅ ALL PRE-FLIGHT CHECKS PASSED")
    print(f"{'=' * 60}")
    print(f"Demo payer:             {payer}")
    print(f"  ETH balance:          {eth:.6f}")
    print(f"  FTNS balance:         {ftns:.4f}")
    print(f"Network treasury:       {treasury}")
    print()
    print(f"Safe to run the trace:")
    print(f"  PYTHONPATH=. .venv/bin/python3.14 \\")
    print(f"      scripts/exercise_canonical_workflow_base_sepolia.py")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

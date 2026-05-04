#!/usr/bin/env python3
"""
Generate the Foundation Safe transaction that grants MINTER_ROLE on FTNSToken
to the deployed EmissionController.

Per docs/2026-04-30-post-audit-deploy-ceremony-runbook.md §4.1, this is the
load-bearing post-audit-bundle-handoff governance tx. Without MINTER_ROLE,
EmissionController cannot mint and the entire emission economy is dead.

The runbook §4.1 instructs the operator to manually enter parameters in
Safe Transaction Builder. This script automates that prep:

1. Computes keccak256("MINTER_ROLE") (already known but proven from source)
2. Encodes the grantRole(role, account) calldata
3. Prints all parameters needed for Safe UI Transaction Builder
4. Optionally writes a Safe Transaction Builder import JSON
5. With --verify mode, queries on-chain hasRole post-execution

Modes:

  prep — print Safe-UI parameters + write import JSON (default)
  verify — query on-chain hasRole(MINTER_ROLE, emission) and report

Usage:

  # PREP mode — generate the tx parameters before Safe signing
  export FTNS_TOKEN_ADDRESS=0x5276a3756C85f2E9e46f6D34386167a209aa16e5
  export EMISSION_CONTROLLER_ADDRESS=0x...    # from phase8-emission-base-*.json
  python3 scripts/safe-tx-grant-minter-role.py prep

  # VERIFY mode — confirm role granted post-Safe-execution
  python3 scripts/safe-tx-grant-minter-role.py verify

Default env values:
  FTNS_TOKEN_ADDRESS         = 0x5276a3756C85f2E9e46f6D34386167a209aa16e5 (canonical)
  BASE_RPC_URL               = https://mainnet.base.org
  CHAIN_ID                   = 8453
  EMISSION_CONTROLLER_ADDRESS — required, no default (varies per Phase 8 deploy)
  SAFE_ADDRESS               = 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 (Foundation Safe)

Exit codes:
  0 = prep succeeded / verify shows role IS granted
  1 = verify shows role is NOT granted
  2 = invalid input (missing env, malformed address)
  3 = RPC error
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from eth_utils import keccak, to_bytes, to_checksum_address
from web3 import Web3


# ── Constants ───────────────────────────────────────────────────────────

# keccak256("MINTER_ROLE") per FTNSTokenSimple.sol
# Verified by source: bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
MINTER_ROLE_HASH = "0x" + keccak(text="MINTER_ROLE").hex()

# Function selector for AccessControl.grantRole(bytes32,address)
GRANT_ROLE_SELECTOR = "0x" + keccak(text="grantRole(bytes32,address)").hex()[:8]

# Function selector for AccessControl.hasRole(bytes32,address) view
HAS_ROLE_SELECTOR = "0x" + keccak(text="hasRole(bytes32,address)").hex()[:8]

DEFAULT_FTNS = "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
DEFAULT_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
DEFAULT_RPC = "https://mainnet.base.org"
DEFAULT_CHAIN_ID = 8453


def _encode_grant_role_calldata(role_hash: str, account_addr: str) -> str:
    """
    Encode grantRole(bytes32,address) calldata.

    Layout: selector (4 bytes) || role (32 bytes) || account (32 bytes left-padded)
    """
    role_bytes = to_bytes(hexstr=role_hash)
    if len(role_bytes) != 32:
        raise ValueError(f"role hash must be 32 bytes, got {len(role_bytes)}")
    account_padded = bytes(12) + to_bytes(hexstr=account_addr)
    if len(account_padded) != 32:
        raise ValueError(f"padded account must be 32 bytes, got {len(account_padded)}")
    selector_bytes = to_bytes(hexstr=GRANT_ROLE_SELECTOR)
    calldata = selector_bytes + role_bytes + account_padded
    return "0x" + calldata.hex()


def _resolve_addresses() -> dict:
    """Resolve env vars with sensible defaults; validate format."""
    ftns = os.environ.get("FTNS_TOKEN_ADDRESS", DEFAULT_FTNS).strip()
    emission = os.environ.get("EMISSION_CONTROLLER_ADDRESS", "").strip()
    safe = os.environ.get("SAFE_ADDRESS", DEFAULT_SAFE).strip()

    if not emission:
        print(
            "ERROR: EMISSION_CONTROLLER_ADDRESS env var is required.",
            file=sys.stderr,
        )
        print(
            "       After Phase 8 emission-stack deploys, the address is in",
            file=sys.stderr,
        )
        print(
            "       contracts/deployments/phase8-emission-base-*.json under",
            file=sys.stderr,
        )
        print(
            "       contracts.EmissionController.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    for label, addr in [("FTNS", ftns), ("EmissionController", emission), ("Safe", safe)]:
        if not Web3.is_address(addr):
            print(f"ERROR: {label} address {addr!r} is not a valid Ethereum address", file=sys.stderr)
            raise SystemExit(2)

    return {
        "ftns": to_checksum_address(ftns),
        "emission": to_checksum_address(emission),
        "safe": to_checksum_address(safe),
    }


def _connect_rpc() -> Web3:
    rpc_url = os.environ.get("BASE_RPC_URL", DEFAULT_RPC)
    chain_id = int(os.environ.get("CHAIN_ID", str(DEFAULT_CHAIN_ID)))
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"ERROR: cannot reach RPC at {rpc_url}", file=sys.stderr)
        raise SystemExit(3)
    rpc_chain = w3.eth.chain_id
    if rpc_chain != chain_id:
        print(
            f"ERROR: RPC chainId={rpc_chain} != expected {chain_id}. "
            f"BASE_RPC_URL likely points at the wrong network.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    return w3


def cmd_prep() -> int:
    """Generate Safe-UI parameters + Safe Transaction Builder JSON."""
    addrs = _resolve_addresses()
    calldata = _encode_grant_role_calldata(MINTER_ROLE_HASH, addrs["emission"])

    print("=" * 70)
    print("Foundation Safe transaction — grant MINTER_ROLE to EmissionController")
    print("=" * 70)
    print()
    print("Per post-audit ceremony runbook §4.1. This is the load-bearing")
    print("governance tx — without it EmissionController cannot mint.")
    print()
    print("─── Constants (from FTNSTokenSimple.sol source) ─────────────────────")
    print(f"  MINTER_ROLE hash:    {MINTER_ROLE_HASH}")
    print(f"  grantRole selector:  {GRANT_ROLE_SELECTOR}")
    print()
    print("─── Resolved addresses ──────────────────────────────────────────────")
    print(f"  FTNSToken (target):  {addrs['ftns']}")
    print(f"  EmissionController:  {addrs['emission']}  (will receive MINTER_ROLE)")
    print(f"  Foundation Safe:     {addrs['safe']}  (sender — must hold DEFAULT_ADMIN_ROLE)")
    print()
    print("─── Safe Transaction Builder UI parameters ─────────────────────────")
    print("  Open https://app.safe.global → your Foundation Safe")
    print("  → New transaction → Transaction Builder → Add new transaction")
    print()
    print(f"  To address:           {addrs['ftns']}")
    print(f"  ETH value:            0")
    print(f"  ABI fragment:         function grantRole(bytes32 role, address account)")
    print(f"  ↳ role:               {MINTER_ROLE_HASH}")
    print(f"  ↳ account:            {addrs['emission']}")
    print()
    print("─── Or, raw calldata mode in Safe UI ────────────────────────────────")
    print(f"  To:                   {addrs['ftns']}")
    print(f"  Value:                0")
    print(f"  Data:                 {calldata}")
    print(f"  Operation:            CALL (NOT DelegateCall)")
    print()

    # Write Safe Transaction Builder import JSON
    out_path = Path("/tmp") / f"safe-tx-grant-minter-{addrs['emission'][:10]}.json"
    safe_tx_builder_json = {
        "version": "1.0",
        "chainId": str(DEFAULT_CHAIN_ID),
        "createdAt": int(__import__("time").time() * 1000),
        "meta": {
            "name": "Grant MINTER_ROLE on FTNSToken to EmissionController",
            "description": (
                "Post-audit-bundle-handoff governance tx per "
                "docs/2026-04-30-post-audit-deploy-ceremony-runbook.md §4.1. "
                "Required before EmissionController can mint."
            ),
            "txBuilderVersion": "1.16.5",
            "createdFromSafeAddress": addrs["safe"],
            "createdFromOwnerAddress": "",
            "checksum": "",
        },
        "transactions": [
            {
                "to": addrs["ftns"],
                "value": "0",
                "data": calldata,
                "contractMethod": {
                    "inputs": [
                        {"internalType": "bytes32", "name": "role", "type": "bytes32"},
                        {"internalType": "address", "name": "account", "type": "address"},
                    ],
                    "name": "grantRole",
                    "payable": False,
                },
                "contractInputsValues": {
                    "role": MINTER_ROLE_HASH,
                    "account": addrs["emission"],
                },
            }
        ],
    }
    out_path.write_text(json.dumps(safe_tx_builder_json, indent=2))
    print(f"─── Safe Transaction Builder import JSON ────────────────────────────")
    print(f"  Written to: {out_path}")
    print(f"  In Safe UI → Transaction Builder → click '...' → Drag and drop")
    print(f"  this JSON file to import the tx.")
    print()
    print("─── Pre-execution sanity (run before signing) ──────────────────────")
    print("  Confirm Foundation Safe holds DEFAULT_ADMIN_ROLE on FTNS:")
    print(f"  cast call {addrs['ftns']} \\")
    print(f"    \"hasRole(bytes32,address)(bool)\" \\")
    print(f"    0x0000000000000000000000000000000000000000000000000000000000000000 \\")
    print(f"    {addrs['safe']} \\")
    print(f"    --rpc-url $BASE_RPC_URL")
    print(f"  # expect: true")
    print()
    print("─── After Safe execution, verify ───────────────────────────────────")
    print("  python3 scripts/safe-tx-grant-minter-role.py verify")
    print()
    print("=" * 70)
    return 0


def cmd_verify() -> int:
    """Query on-chain hasRole(MINTER_ROLE, EmissionController) and report."""
    addrs = _resolve_addresses()
    w3 = _connect_rpc()

    # Encode hasRole(MINTER_ROLE_HASH, emission) calldata
    role_bytes = to_bytes(hexstr=MINTER_ROLE_HASH)
    account_padded = bytes(12) + to_bytes(hexstr=addrs["emission"])
    selector_bytes = to_bytes(hexstr=HAS_ROLE_SELECTOR)
    calldata = "0x" + (selector_bytes + role_bytes + account_padded).hex()

    print(f"Verifying MINTER_ROLE grant on Base mainnet...")
    print(f"  FTNSToken:           {addrs['ftns']}")
    print(f"  EmissionController:  {addrs['emission']}")
    print(f"  MINTER_ROLE hash:    {MINTER_ROLE_HASH}")
    print()

    try:
        result = w3.eth.call({"to": addrs["ftns"], "data": calldata})
    except Exception as e:
        print(f"ERROR: hasRole call failed: {e}", file=sys.stderr)
        return 3

    # Decode bool: result is 32 bytes; non-zero = true
    has_role = int.from_bytes(result, "big") != 0

    if has_role:
        print(f"✅ EmissionController HAS MINTER_ROLE on FTNSToken")
        print(f"   The emission economy is wired correctly.")
        return 0
    print(f"❌ EmissionController does NOT have MINTER_ROLE on FTNSToken")
    print()
    print("   Possible causes:")
    print(f"   1. The Safe tx hasn't been signed/executed yet — go to")
    print(f"      https://app.safe.global → your Foundation Safe and check")
    print(f"      pending transactions.")
    print(f"   2. The Safe tx executed but to the wrong target — re-run prep")
    print(f"      and confirm addresses match.")
    print(f"   3. EmissionController not yet deployed — check Phase 8 deploy")
    print(f"      manifest at contracts/deployments/phase8-emission-base-*.json")
    return 1


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("prep", "verify"):
        print(__doc__, file=sys.stderr)
        print("\nERROR: must specify 'prep' or 'verify' as first arg", file=sys.stderr)
        return 2

    if sys.argv[1] == "prep":
        return cmd_prep()
    return cmd_verify()


if __name__ == "__main__":
    sys.exit(main())

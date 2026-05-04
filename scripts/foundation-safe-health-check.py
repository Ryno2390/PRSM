#!/usr/bin/env python3
"""
Foundation Safe + on-chain treasury layer health check.

Single-command snapshot of the live state on Base mainnet:

  - Foundation Safe: owners + threshold + balance + recent tx history
  - ProvenanceRegistry: bytecode present
  - RoyaltyDistributor: immutable wiring (ftns/registry/networkTreasury/fee)
  - FTNSToken: symbol + total supply + Foundation Safe's role bits

Useful for:
  - Daily ops: confirm nothing has drifted
  - Auditor demo: zero-friction proof the live treasury is wired correctly
  - Pre-governance-tx sanity: confirm Safe still holds DEFAULT_ADMIN_ROLE
    on FTNS before signing role-grant txs

Usage:
  python3 scripts/foundation-safe-health-check.py
  # optional: BASE_RPC_URL=<your_alchemy_url>  for faster + more reliable reads

Exit codes:
  0 = all healthy
  1 = at least one anomaly (script prints which)
  2 = RPC unreachable
"""
from __future__ import annotations

import os
import sys
import time

from eth_utils import keccak, to_bytes
from web3 import Web3


# ── Pinned addresses (from project memory + 2026-05-04 ceremony) ───────

FOUNDATION_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
PROVENANCE_REGISTRY = "0xdF470BFa9eF310B196801D5105468515d0069915"
ROYALTY_DISTRIBUTOR = "0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2"
FTNS_TOKEN = "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"

# Expected Safe owners (from project memory reference_multisig_addresses_location.md)
EXPECTED_OWNERS = {
    "0x7e824c2c247ff2ab6740887df3580b7fd915e1ba": "Ledger",
    "0x0d39032a95e2a1c6d1cd0725553667dbd88b9f18": "Trezor",
    "0x1623ceba9afc6050e2c2c38ee371f6ec50f7023c": "OneKey",
}
EXPECTED_THRESHOLD = 2

# Expected immutable wiring on RoyaltyDistributor
EXPECTED_NETWORK_FEE_BPS = 200  # 2.00%

DEFAULT_RPC = "https://mainnet.base.org"
DEFAULT_CHAIN_ID = 8453


# ── Function selectors ─────────────────────────────────────────────────

def _sel(sig: str) -> str:
    """4-byte function selector for a signature like 'getOwners()'."""
    return "0x" + keccak(text=sig).hex()[:8]


# Safe (Gnosis Safe v1.4.x)
SAFE_GET_OWNERS = _sel("getOwners()")           # returns address[]
SAFE_GET_THRESHOLD = _sel("getThreshold()")     # returns uint256
SAFE_NONCE = _sel("nonce()")                    # returns uint256

# RoyaltyDistributor immutables
RD_FTNS = _sel("ftns()")
RD_REGISTRY = _sel("registry()")
RD_NETWORK_TREASURY = _sel("networkTreasury()")
RD_NETWORK_FEE_BPS = _sel("NETWORK_FEE_BPS()")

# FTNSTokenSimple
FTNS_SYMBOL = _sel("symbol()")
FTNS_TOTAL_SUPPLY = _sel("totalSupply()")
FTNS_HAS_ROLE = _sel("hasRole(bytes32,address)")

# AccessControl roles (FTNSTokenSimple constants)
DEFAULT_ADMIN_ROLE = "0x" + ("00" * 32)
MINTER_ROLE = "0x" + keccak(text="MINTER_ROLE").hex()
PAUSER_ROLE = "0x" + keccak(text="PAUSER_ROLE").hex()
BURNER_ROLE = "0x" + keccak(text="BURNER_ROLE").hex()


# ── Output formatting ──────────────────────────────────────────────────

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def section(title: str) -> None:
    print(f"\n{BOLD}── {title} {'─' * (66 - len(title))}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}·{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> int:
    rpc_url = os.environ.get("BASE_RPC_URL", DEFAULT_RPC)
    expected_chain = int(os.environ.get("CHAIN_ID", str(DEFAULT_CHAIN_ID)))

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"{RED}ERROR: cannot reach RPC at {rpc_url}{RESET}", file=sys.stderr)
        return 2

    rpc_chain = w3.eth.chain_id
    if rpc_chain != expected_chain:
        print(
            f"{RED}ERROR: RPC chainId={rpc_chain} != expected {expected_chain}. "
            f"BASE_RPC_URL points at the wrong network.{RESET}",
            file=sys.stderr,
        )
        return 2

    print(f"{BOLD}PRSM Foundation Safe + Treasury Layer Health Check{RESET}")
    print(f"  RPC: {rpc_url}")
    print(f"  Chain: {rpc_chain} ({BOLD}Base mainnet{RESET})")
    print(f"  Block: {w3.eth.block_number:,}")

    anomalies = 0

    # Free public RPC throttles; small sleep between calls keeps us under the
    # rate limit. Premium RPC URLs (Alchemy/Infura) don't need this but the
    # cost is negligible.
    rpc_throttle = 0.0 if "alchemy" in rpc_url or "infura" in rpc_url else 0.15

    def call(target: str, data: str) -> bytes | None:
        if rpc_throttle:
            time.sleep(rpc_throttle)
        for attempt in range(3):
            try:
                return w3.eth.call({"to": Web3.to_checksum_address(target), "data": data})
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(1.0 * (attempt + 1))  # back off 1s, 2s
                    continue
                fail(f"RPC call failed at {target}: {e}")
                nonlocal anomalies
                anomalies += 1
                return None
        return None

    def decode_address(blob: bytes) -> str:
        return Web3.to_checksum_address("0x" + blob[-20:].hex())

    def decode_uint(blob: bytes) -> int:
        return int.from_bytes(blob, "big")

    def decode_bool(blob: bytes) -> bool:
        return decode_uint(blob) != 0

    def decode_string(blob: bytes) -> str:
        # ABI string decode: offset (32) | length (32) | data
        if len(blob) < 64:
            return ""
        length = int.from_bytes(blob[32:64], "big")
        return blob[64:64 + length].decode("utf-8", errors="replace")

    # ── Safe ────────────────────────────────────────────────────────────
    section("Foundation Safe")
    print(f"  Address: {FOUNDATION_SAFE}")

    safe_bal = w3.eth.get_balance(FOUNDATION_SAFE)
    print(f"  Balance: {safe_bal} wei = {safe_bal/1e18:.6f} ETH")

    code = w3.eth.get_code(FOUNDATION_SAFE)
    if code == b"" or code == b"\x00":
        fail(f"NO BYTECODE at Foundation Safe address — Safe contract missing")
        anomalies += 1
        return 1
    ok(f"bytecode: {len(code)} bytes (Safe contract present)")

    # getOwners()
    blob = call(FOUNDATION_SAFE, SAFE_GET_OWNERS)
    if blob is not None:
        # ABI: array offset (32) | length (32) | addresses (20-byte left-padded * length)
        length = int.from_bytes(blob[32:64], "big")
        owners = [decode_address(blob[64 + 32 * i: 64 + 32 * (i + 1)]) for i in range(length)]
        if len(owners) != len(EXPECTED_OWNERS):
            fail(f"owner count: {len(owners)}, expected {len(EXPECTED_OWNERS)}")
            anomalies += 1
        else:
            ok(f"owner count: {len(owners)}")
        for owner in owners:
            label = EXPECTED_OWNERS.get(owner.lower(), "UNKNOWN")
            if label == "UNKNOWN":
                fail(f"owner {owner} is NOT a known device — investigate immediately")
                anomalies += 1
            else:
                ok(f"owner {owner} → {label}")

    # getThreshold()
    blob = call(FOUNDATION_SAFE, SAFE_GET_THRESHOLD)
    if blob is not None:
        threshold = decode_uint(blob)
        if threshold != EXPECTED_THRESHOLD:
            fail(f"threshold: {threshold}, expected {EXPECTED_THRESHOLD}")
            anomalies += 1
        else:
            ok(f"threshold: {threshold} of 3")

    # nonce() — informational only; tells us how many txs have been executed
    blob = call(FOUNDATION_SAFE, SAFE_NONCE)
    if blob is not None:
        nonce = decode_uint(blob)
        info(f"executed-tx count (nonce): {nonce}")

    # ── ProvenanceRegistry ──────────────────────────────────────────────
    section("ProvenanceRegistry (Phase 1.3 Task 8)")
    print(f"  Address: {PROVENANCE_REGISTRY}")
    code = w3.eth.get_code(PROVENANCE_REGISTRY)
    if not code:
        fail("NO BYTECODE — ProvenanceRegistry missing")
        anomalies += 1
    else:
        ok(f"bytecode: {len(code)} bytes")

    # ── RoyaltyDistributor ──────────────────────────────────────────────
    section("RoyaltyDistributor (Phase 1.3 Task 8)")
    print(f"  Address: {ROYALTY_DISTRIBUTOR}")
    code = w3.eth.get_code(ROYALTY_DISTRIBUTOR)
    if not code:
        fail("NO BYTECODE — RoyaltyDistributor missing")
        anomalies += 1
    else:
        ok(f"bytecode: {len(code)} bytes")

    # immutable getters
    checks = [
        (RD_FTNS, "ftns()", FTNS_TOKEN, "address"),
        (RD_REGISTRY, "registry()", PROVENANCE_REGISTRY, "address"),
        (RD_NETWORK_TREASURY, "networkTreasury()", FOUNDATION_SAFE, "address"),
        (RD_NETWORK_FEE_BPS, "NETWORK_FEE_BPS()", EXPECTED_NETWORK_FEE_BPS, "uint"),
    ]
    for selector, name, expected, kind in checks:
        blob = call(ROYALTY_DISTRIBUTOR, selector)
        if blob is None:
            continue
        if kind == "address":
            actual = decode_address(blob)
            if actual.lower() == expected.lower():
                ok(f"{name}: {actual}")
            else:
                fail(f"{name}: on-chain={actual}, expected={expected}")
                anomalies += 1
        else:
            actual = decode_uint(blob)
            if actual == expected:
                ok(f"{name}: {actual} ({actual/100:.2f}%)")
            else:
                fail(f"{name}: on-chain={actual}, expected={expected}")
                anomalies += 1

    # ── FTNSToken ──────────────────────────────────────────────────────
    section("FTNSToken (canonical)")
    print(f"  Address: {FTNS_TOKEN}")

    blob = call(FTNS_TOKEN, FTNS_SYMBOL)
    if blob is not None:
        symbol = decode_string(blob)
        if symbol == "FTNS":
            ok(f"symbol(): {symbol}")
        else:
            fail(f"symbol(): {symbol!r}, expected FTNS")
            anomalies += 1

    blob = call(FTNS_TOKEN, FTNS_TOTAL_SUPPLY)
    if blob is not None:
        supply = decode_uint(blob)
        info(f"totalSupply(): {supply / 1e18:,.0f} FTNS ({supply} wei)")

    # Foundation Safe role bits on FTNS — all informational at the
    # Phase 1.3 Task 8 milestone. FTNS DEFAULT_ADMIN_ROLE handoff is a
    # separate decision per project memory project_production_ftns_state_*
    # ("admin = hot key being loaded to hardware 2026-05-01; existing-mode
    # role-handoff still pending for T-0").
    #
    # When that handoff happens, re-run this health-check and DEFAULT_ADMIN_ROLE
    # SHOULD show True. Until then, all four roles correctly show False on
    # the Foundation Safe.
    print()
    print(f"  {DIM}Foundation Safe role bits on FTNS (informational pre-handoff):{RESET}")
    role_checks = [
        (DEFAULT_ADMIN_ROLE, "DEFAULT_ADMIN_ROLE", "true after FTNS admin handoff (separate ceremony)"),
        (MINTER_ROLE, "MINTER_ROLE", "false expected — granted to EmissionController per §4.1"),
        (PAUSER_ROLE, "PAUSER_ROLE", "true if Safe handles emergency pause post-handoff"),
        (BURNER_ROLE, "BURNER_ROLE", "true if Safe handles emergency burn post-handoff"),
    ]
    safe_padded = bytes(12) + to_bytes(hexstr=FOUNDATION_SAFE)
    for role_hash, role_name, semantics in role_checks:
        role_bytes = to_bytes(hexstr=role_hash)
        selector_bytes = to_bytes(hexstr=FTNS_HAS_ROLE)
        calldata = "0x" + (selector_bytes + role_bytes + safe_padded).hex()
        blob = call(FTNS_TOKEN, calldata)
        if blob is None:
            continue
        has = decode_bool(blob)
        # All informational here — correctness depends on stage.
        # Operators should run with FTNS_HANDOFF_DONE=1 after the handoff
        # ceremony to flip DEFAULT_ADMIN_ROLE to a hard assertion.
        ftns_handoff_done = os.environ.get("FTNS_HANDOFF_DONE", "0") == "1"
        if role_name == "DEFAULT_ADMIN_ROLE" and ftns_handoff_done:
            if has:
                ok(f"{role_name}: {has}  ({semantics})")
            else:
                fail(f"{role_name}: {has}  (FTNS_HANDOFF_DONE=1 set; admin role required but absent)")
                anomalies += 1
        else:
            print(f"  {DIM}·{RESET} {role_name}: {has}  ({DIM}{semantics}{RESET})")

    # ── Final ──────────────────────────────────────────────────────────
    print()
    if anomalies == 0:
        print(f"{GREEN}{BOLD}✅ All healthy. Foundation treasury layer state matches expectations.{RESET}")
        return 0
    print(f"{RED}{BOLD}❌ {anomalies} anomal{'y' if anomalies == 1 else 'ies'} detected. Investigate above.{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

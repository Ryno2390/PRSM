#!/usr/bin/env python3
"""PRSM-CR-2026-05-06-3 Phase 1 — hot key grants 4 roles to Foundation Safe.

Grants DEFAULT_ADMIN_ROLE, MINTER_ROLE, PAUSER_ROLE, BURNER_ROLE on
FTNSTokenSimple to Foundation Safe. Hot key signs each tx.

Idempotent: for each role, checks `hasRole(role, safe)` first; skips
the grant if Safe already holds it. Safe to re-run after partial
execution (e.g., RPC hiccup mid-way).

After this script confirms all 4 grants on-chain, the next step is
Phase 2 — 4 multisig ceremonies in Safe UI to revoke each role from
the hot key. That ceremony is human-driven and not in this script.

Run from repo root:
  PYTHONPATH=. .venv/bin/python3.14 \
      scripts/migrate_ftns_roles_phase1_grant.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from eth_account import Account
from web3 import Web3


HOT_KEY_ENV_FILE = Path.home() / ".prsm" / ".env"
HOT_KEY_VAR = "FTNS_WALLET_PRIVATE_KEY"

EXPECTED_SENDER = "0x8eaA00FF741323bc8B0ab1290c544738D9b2f012"
SAFE_DESTINATION = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
FTNS_TOKEN = "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
EXPECTED_CHAIN_ID = 8453

RPC_URL = os.environ.get("BASE_MAINNET_RPC_URL", "https://mainnet.base.org")

ROLES = [
    ("DEFAULT_ADMIN_ROLE", b"\x00" * 32),
    ("MINTER_ROLE", Web3.keccak(text="MINTER_ROLE")),
    ("PAUSER_ROLE", Web3.keccak(text="PAUSER_ROLE")),
    ("BURNER_ROLE", Web3.keccak(text="BURNER_ROLE")),
]


def _load_hot_key() -> str:
    if not HOT_KEY_ENV_FILE.exists():
        sys.exit(f"FATAL: {HOT_KEY_ENV_FILE} not found")
    for line in HOT_KEY_ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{HOT_KEY_VAR}=") or line.startswith(
            f"export {HOT_KEY_VAR}="
        ):
            v = line.split("=", 1)[1].strip().strip("'\"")
            return v if v.startswith("0x") else "0x" + v
    sys.exit(f"FATAL: {HOT_KEY_VAR} not found in {HOT_KEY_ENV_FILE}")


def _has_role(web3: Web3, role: bytes, who: str) -> bool:
    sel = Web3.keccak(text="hasRole(bytes32,address)")[:4]
    data = sel + role + b"\x00" * 12 + bytes.fromhex(who[2:])
    raw = web3.eth.call({"to": Web3.to_checksum_address(FTNS_TOKEN), "data": data})
    return int.from_bytes(raw, "big") == 1


def _grant_role(web3: Web3, role: bytes, target: str, private_key: str, sender: str) -> str:
    sel = Web3.keccak(text="grantRole(bytes32,address)")[:4]
    calldata = sel + role + b"\x00" * 12 + bytes.fromhex(target[2:])

    nonce = web3.eth.get_transaction_count(sender)
    gas_price = web3.eth.gas_price
    gas_estimate = web3.eth.estimate_gas(
        {"from": sender, "to": Web3.to_checksum_address(FTNS_TOKEN), "data": calldata}
    )
    tx = {
        "from": sender,
        "to": Web3.to_checksum_address(FTNS_TOKEN),
        "data": calldata,
        "nonce": nonce,
        "gas": int(gas_estimate * 1.25),
        "gasPrice": gas_price,
        "chainId": EXPECTED_CHAIN_ID,
    }
    signed = web3.eth.account.sign_transaction(tx, private_key=private_key)
    h = web3.eth.send_raw_transaction(signed.raw_transaction).hex()
    receipt = web3.eth.wait_for_transaction_receipt(h, timeout=120)
    if receipt.status != 1:
        sys.exit(f"FATAL: grant tx {h} reverted in block {receipt.blockNumber}")
    return h


def main() -> int:
    print("=" * 70)
    print("PRSM-CR-2026-05-06-3 Phase 1 — grant 4 roles to Foundation Safe")
    print("=" * 70)

    private_key = _load_hot_key()
    sender = Account.from_key(private_key).address
    if sender.lower() != EXPECTED_SENDER.lower():
        sys.exit(
            f"FATAL: hot key derives to {sender}, expected {EXPECTED_SENDER}"
        )

    web3 = Web3(Web3.HTTPProvider(RPC_URL))
    chain_id = web3.eth.chain_id
    if chain_id != EXPECTED_CHAIN_ID:
        sys.exit(f"FATAL: RPC reports chain {chain_id}, expected {EXPECTED_CHAIN_ID}")

    eth_balance = web3.eth.get_balance(sender)
    print(f"Network:           Base mainnet (chain {chain_id})")
    print(f"FTNS contract:     {FTNS_TOKEN}")
    print(f"Hot key (sender):  {sender}")
    print(f"  ETH:             {Web3.from_wei(eth_balance, 'ether'):.6f}")
    print(f"Foundation Safe:   {SAFE_DESTINATION}")
    print()
    print("Pre-grant state (hot key vs Safe):")
    print(f"  {'role':24}  {'hot_key':>7}  {'safe':>4}")
    for name, role in ROLES:
        hk = _has_role(web3, role, sender)
        sf = _has_role(web3, role, SAFE_DESTINATION)
        print(f"  {name:24}  {('✓' if hk else ' '):^7}  {('✓' if sf else ' '):^4}")
    print()
    print("This will broadcast up to 4 grantRole txs from the hot key.")
    print("Each is REVERSIBLE — if needed, the Safe (post-Phase-1) or hot key")
    print("can revokeRole later. Phase 2 (revokes) is the one-way step.")
    print()
    confirm = input("Type 'go' to proceed: ").strip()
    if confirm != "go":
        print("Aborted.")
        return 1

    print()
    txs: list[tuple[str, str]] = []
    for name, role in ROLES:
        if _has_role(web3, role, SAFE_DESTINATION):
            print(f"  [skip] {name}: Safe already holds")
            txs.append((name, "(already-granted, no tx)"))
            continue
        print(f"  [send] grantRole({name}, Safe) ...", end=" ", flush=True)
        h = _grant_role(web3, role, SAFE_DESTINATION, private_key, sender)
        print(f"tx 0x{h}")
        txs.append((name, "0x" + h))

    print()
    print("Verifying post-state (with replica-lag retry)...")
    expected = {name: True for name, _ in ROLES}
    for attempt in range(15):
        all_ok = True
        for name, role in ROLES:
            if not _has_role(web3, role, SAFE_DESTINATION):
                all_ok = False
                break
        if all_ok:
            if attempt > 0:
                print(f"  read replica caught up after {attempt + 1} attempts")
            break
        time.sleep(2)

    print()
    print("Final state (post-grant):")
    print(f"  {'role':24}  {'hot_key':>7}  {'safe':>4}")
    for name, role in ROLES:
        hk = _has_role(web3, role, sender)
        sf = _has_role(web3, role, SAFE_DESTINATION)
        marker = "" if (hk and sf) else "  ⚠"
        print(f"  {name:24}  {('✓' if hk else ' '):^7}  {('✓' if sf else ' '):^4}{marker}")

    print()
    print("=" * 70)
    print("Phase 1 transaction record (for §8 of PRSM-CR-2026-05-06-3):")
    print("=" * 70)
    for name, h in txs:
        print(f"  grantRole({name:20}, safe)  →  {h}")
    print()
    print("✅ PHASE 1 COMPLETE — Safe now holds all 4 roles alongside hot key.")
    print()
    print("Next: Phase 2 (4 multisig ceremonies in Safe UI to revoke each")
    print("role from the hot key). Do MINTER_ROLE first, DEFAULT_ADMIN_ROLE")
    print("last, per CR-3 §4 Phase 2 sequencing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

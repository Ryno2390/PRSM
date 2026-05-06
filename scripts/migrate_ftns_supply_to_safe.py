#!/usr/bin/env python3
"""One-shot: move 100M FTNS from hot deployer EOA to Foundation Safe.

Closes the security gap where the full FTNS supply sits on the hot
deployer key (~/.prsm/.env). Does NOT touch DEFAULT_ADMIN_ROLE — that
migration is its own ratified ceremony (separate PRSM-CR-*).

Pre-conditions verified at runtime:
  - Sender address == 0x8eaA00FF741323bc8B0ab1290c544738D9b2f012
  - Sender FTNS balance == 100_000_000 * 10**18 (exactly)
  - Destination == 0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791 (Safe)
  - Chain id == 8453 (Base mainnet)
  - User types "yes" at the confirmation prompt

If any check fails, exits without sending. The script does not modify
anything until the user explicitly confirms.

Run from repo root:
  PYTHONPATH=. .venv/bin/python3.14 scripts/migrate_ftns_supply_to_safe.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from eth_account import Account
from web3 import Web3


HOT_KEY_ENV_FILE = Path.home() / ".prsm" / ".env"
HOT_KEY_VAR = "FTNS_WALLET_PRIVATE_KEY"

EXPECTED_SENDER = "0x8eaA00FF741323bc8B0ab1290c544738D9b2f012"
SAFE_DESTINATION = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
FTNS_TOKEN = "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
EXPECTED_CHAIN_ID = 8453
EXPECTED_AMOUNT_WEI = 100_000_000 * 10**18

RPC_URL = os.environ.get("BASE_MAINNET_RPC_URL", "https://mainnet.base.org")


def _load_hot_key() -> str:
    """Read FTNS_WALLET_PRIVATE_KEY from ~/.prsm/.env without sourcing."""
    if not HOT_KEY_ENV_FILE.exists():
        sys.exit(f"FATAL: {HOT_KEY_ENV_FILE} not found")
    for line in HOT_KEY_ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{HOT_KEY_VAR}=") or line.startswith(
            f"export {HOT_KEY_VAR}="
        ):
            return line.split("=", 1)[1].strip().strip("'\"")
    sys.exit(f"FATAL: {HOT_KEY_VAR} not found in {HOT_KEY_ENV_FILE}")


def _balance_of(web3: Web3, token: str, who: str) -> int:
    sel = Web3.keccak(text="balanceOf(address)")[:4]
    data = sel + b"\x00" * 12 + bytes.fromhex(who[2:])
    raw = web3.eth.call(
        {"to": Web3.to_checksum_address(token), "data": data}
    )
    return int.from_bytes(raw, "big")


def main() -> int:
    print("=" * 70)
    print("FTNS SUPPLY MIGRATION — hot deployer EOA → Foundation Safe")
    print("=" * 70)

    private_key = _load_hot_key()
    sender = Account.from_key(private_key).address

    # ── Pre-flight checks ───────────────────────────────────────────────
    if sender.lower() != EXPECTED_SENDER.lower():
        sys.exit(
            f"FATAL: hot key derives to {sender}, expected "
            f"{EXPECTED_SENDER}. Wrong env var or wrong file."
        )

    web3 = Web3(Web3.HTTPProvider(RPC_URL))
    chain_id = web3.eth.chain_id
    if chain_id != EXPECTED_CHAIN_ID:
        sys.exit(
            f"FATAL: RPC reports chain {chain_id}, expected "
            f"{EXPECTED_CHAIN_ID} (Base mainnet)"
        )

    sender_ftns = _balance_of(web3, FTNS_TOKEN, sender)
    if sender_ftns != EXPECTED_AMOUNT_WEI:
        sys.exit(
            f"FATAL: sender holds {sender_ftns / 10**18:,.4f} FTNS, "
            f"expected exactly {EXPECTED_AMOUNT_WEI / 10**18:,.0f}. "
            "Either the supply was already moved or some external "
            "action has occurred. ABORTING — investigate before retrying."
        )

    safe_ftns_before = _balance_of(web3, FTNS_TOKEN, SAFE_DESTINATION)
    sender_eth = web3.eth.get_balance(sender)

    print()
    print(f"Network:           Base mainnet (chain {chain_id})")
    print(f"FTNS contract:     {FTNS_TOKEN}")
    print(f"Sender:            {sender}")
    print(f"  FTNS balance:    {sender_ftns / 10**18:,.0f}")
    print(
        f"  ETH balance:     {Web3.from_wei(sender_eth, 'ether'):.6f}"
    )
    print(f"Destination:       {SAFE_DESTINATION}  (Foundation Safe)")
    print(f"  FTNS pre-tx:     {safe_ftns_before / 10**18:,.4f}")
    print(f"Amount to send:    {EXPECTED_AMOUNT_WEI / 10**18:,.0f} FTNS")
    print()
    print("After this tx:  sender = 0 FTNS, Safe = 100,000,000 FTNS")
    print("This is IRREVERSIBLE.")
    print()

    confirm = input("Type 'yes' (lowercase, no quotes) to broadcast: ").strip()
    if confirm != "yes":
        print("Aborted by user. No transaction sent.")
        return 1

    # ── Build, sign, broadcast ──────────────────────────────────────────
    transfer_sel = Web3.keccak(text="transfer(address,uint256)")[:4]
    calldata = (
        transfer_sel
        + b"\x00" * 12
        + bytes.fromhex(SAFE_DESTINATION[2:])
        + EXPECTED_AMOUNT_WEI.to_bytes(32, "big")
    )

    nonce = web3.eth.get_transaction_count(sender)
    gas_price = web3.eth.gas_price

    # Estimate gas with a 25% buffer
    gas_estimate = web3.eth.estimate_gas(
        {
            "from": sender,
            "to": Web3.to_checksum_address(FTNS_TOKEN),
            "data": calldata,
        }
    )
    gas_limit = int(gas_estimate * 1.25)

    tx = {
        "from": sender,
        "to": Web3.to_checksum_address(FTNS_TOKEN),
        "data": calldata,
        "nonce": nonce,
        "gas": gas_limit,
        "gasPrice": gas_price,
        "chainId": chain_id,
    }

    print(f"\nGas estimate:    {gas_estimate} (limit set to {gas_limit})")
    print(f"Gas price:       {gas_price / 1e9:.4f} gwei")
    print(
        f"Est. tx cost:    "
        f"{Web3.from_wei(gas_limit * gas_price, 'ether'):.8f} ETH"
    )

    signed = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"\nBroadcast:       {tx_hash.hex()}")
    print(f"Basescan:        https://basescan.org/tx/{tx_hash.hex()}")
    print("\nWaiting for receipt (timeout 120s)...")

    receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status != 1:
        sys.exit(
            f"FATAL: tx reverted (status=0). Block "
            f"{receipt.blockNumber}. Investigate before retrying."
        )

    print(f"\n✅ Confirmed in block {receipt.blockNumber}")
    print(f"   Gas used: {receipt.gasUsed}")

    # ── Post-state verification ─────────────────────────────────────────
    # Public RPC providers (Alchemy load balancer, mainnet.base.org) often
    # have read replicas that lag the writer for a few seconds after a
    # successful tx. Poll up to 15× with 2s spacing before declaring the
    # post-state wrong.
    import time as _time

    expected_safe = EXPECTED_AMOUNT_WEI + safe_ftns_before
    sender_after = safe_after = None
    for attempt in range(15):
        sender_after = _balance_of(web3, FTNS_TOKEN, sender)
        safe_after = _balance_of(web3, FTNS_TOKEN, SAFE_DESTINATION)
        total = sender_after + safe_after
        if sender_after == 0 and safe_after == expected_safe:
            break
        # If total ever exceeds supply, that's stale replica — keep polling.
        if attempt < 14:
            _time.sleep(2)

    print()
    print("Post-tx state:")
    print(f"  Sender FTNS: {sender_after / 10**18:,.4f} (was 100M)")
    print(f"  Safe FTNS:   {safe_after / 10**18:,.4f}")

    if sender_after != 0 or safe_after != expected_safe:
        sys.exit(
            "FATAL: post-state still inconsistent after 30s of polling. "
            "Tx confirmed but balances do not match expected. Investigate "
            "manually before retrying. The tx itself may have succeeded "
            "and the read replica is permanently broken — check Basescan "
            "first."
        )

    print()
    print("=" * 70)
    print("✅ MIGRATION COMPLETE — 100M FTNS now in Foundation Safe")
    print("=" * 70)
    print()
    print("Next: run the multisig disbursement ceremony (Safe UI) to send")
    print(
        "100 FTNS from Safe to demo payer "
        "0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

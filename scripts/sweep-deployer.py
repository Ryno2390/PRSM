#!/usr/bin/env python3
"""
Sweep leftover ETH from a disposable deployer key back to a recovery address.

Used post-deploy ceremony per Multi-Sig Action Plan §6 hygiene step. Replaces
the action-plan paste-the-key-as-Python-literal one-liner — that pattern
caused the 2026-05-04 security incident (private key visible in chat as part
of a syntax-error log message). This script reads the key from the env var
only, never as a CLI argument or Python literal.

Run from a shell that has PRIVATE_KEY exported. The shell's history must
NOT have captured the export — if you `unset HISTFILE` before generating
the key per action-plan §5, this is true.

Usage:

    export PRIVATE_KEY="0x..."     # already set if running post-deploy
    export RECOVERY_ADDR="0x..."   # where to send the leftover ETH
    export BASE_RPC_URL="..."      # optional; defaults to mainnet.base.org
    python3 scripts/sweep-deployer.py

Output:
    DEPLOYER address: 0x...
    Balance:          ... wei = ... ETH
    ...
    Sweep tx hash:    0x...
    Block: ...   Status: SUCCESS
    Deployer left:    ... wei  (dust within safety buffer)

Rationale for fixed safety buffer instead of estimated fee:

Base is an OP Stack rollup. Total tx cost = L2 execution gas + L1 data fee.
`w3.eth.gas_price` returns ONLY the L2 component; the L1 portion is not
exposed and varies. Sweeping `bal - 21000*gas_price` consistently fails
with `insufficient funds` because the L1 fee tips the balance over.

Fix: reserve a fixed safety buffer (5e12 wei ≈ $0.012). Leaves dust
abandoned at the deployer — acceptable trade-off for guaranteed success.
"""
from __future__ import annotations

import os
import sys

from eth_account import Account
from web3 import Web3


# Reserve a fixed buffer to absorb L1 data fee jitter on Base. Larger than
# typical L1 fee (~6e9 wei) by 3 orders of magnitude. 5e12 wei ≈ $0.012 at
# $2.5K ETH — acceptable to abandon at the dust address.
SAFETY_BUFFER_WEI = 5 * 10**12


def main() -> int:
    private_key = os.environ.get("PRIVATE_KEY", "").strip()
    if not private_key:
        print("ERROR: PRIVATE_KEY env var is not set", file=sys.stderr)
        print(
            "       Run this script from the deployer shell where you "
            "generated the key.",
            file=sys.stderr,
        )
        return 2

    if not private_key.startswith("0x"):
        # eth_account.Account.create() returns hex without 0x in some
        # versions. Operator may have hit lesson L1 already; auto-fix here.
        private_key = "0x" + private_key
        print(
            "[sweep-deployer] note: PRIVATE_KEY missing 0x prefix; "
            "prepending automatically.",
            file=sys.stderr,
        )

    # Validate format defensively. eth_account will also raise but we
    # want a clear error before even touching web3.
    if len(private_key) != 66 or any(
        c not in "0123456789abcdefABCDEFx" for c in private_key
    ):
        print(
            f"ERROR: PRIVATE_KEY format invalid (len={len(private_key)}, "
            f"expected 66 chars = '0x' + 64 hex)",
            file=sys.stderr,
        )
        return 2

    recovery_addr = os.environ.get("RECOVERY_ADDR", "").strip()
    if not recovery_addr:
        print("ERROR: RECOVERY_ADDR env var is not set", file=sys.stderr)
        print(
            "       Set to the address that should receive the swept ETH "
            "(e.g. your Ledger Base address).",
            file=sys.stderr,
        )
        return 2
    if not Web3.is_address(recovery_addr):
        print(
            f"ERROR: RECOVERY_ADDR={recovery_addr} is not a valid address",
            file=sys.stderr,
        )
        return 2
    # Normalize to checksummed form for downstream use.
    recovery_addr = Web3.to_checksum_address(recovery_addr)

    rpc_url = os.environ.get("BASE_RPC_URL", "https://mainnet.base.org")
    chain_id = int(os.environ.get("CHAIN_ID", "8453"))

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(
            f"ERROR: cannot reach RPC at {rpc_url}",
            file=sys.stderr,
        )
        return 3

    rpc_chain_id = w3.eth.chain_id
    if rpc_chain_id != chain_id:
        print(
            f"ERROR: RPC reports chainId={rpc_chain_id}, expected {chain_id}. "
            f"BASE_RPC_URL likely points at the wrong network.",
            file=sys.stderr,
        )
        return 3

    acct = Account.from_key(private_key)
    print(f"DEPLOYER address: {acct.address}")
    print(f"RECOVERY address: {recovery_addr}")
    print(f"RPC:              {rpc_url}")
    print(f"Chain id:         {rpc_chain_id}")

    if acct.address.lower() == recovery_addr.lower():
        print(
            "ERROR: DEPLOYER == RECOVERY. Refusing to send funds back to "
            "self (would just burn gas).",
            file=sys.stderr,
        )
        return 4

    bal = w3.eth.get_balance(acct.address)
    print(f"Balance:          {bal} wei = {bal/1e18:.8f} ETH")

    if bal == 0:
        print("Balance is zero — nothing to sweep.")
        return 0

    # 2x L2 gas price for safety; node baseFee can shift between read and send.
    gas_price = int(w3.eth.gas_price * 2)
    print(f"Gas price (2x):   {gas_price} wei")
    print(
        f"Buffer reserved:  {SAFETY_BUFFER_WEI} wei "
        f"= {SAFETY_BUFFER_WEI/1e18:.8f} ETH (covers L1 data fee on Base)"
    )

    send_amt = bal - SAFETY_BUFFER_WEI
    if send_amt <= 0:
        print(
            f"ERROR: balance {bal} <= safety buffer {SAFETY_BUFFER_WEI}; "
            f"too low to sweep safely.",
            file=sys.stderr,
        )
        return 5
    print(f"Sending:          {send_amt} wei = {send_amt/1e18:.8f} ETH")

    tx = {
        "to": recovery_addr,
        "value": send_amt,
        "gas": 21000,
        "gasPrice": gas_price,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "chainId": chain_id,
    }
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"Sweep tx hash:    {tx_hash.hex()}")
    print(f"Basescan:         https://basescan.org/tx/{tx_hash.hex()}")

    print("Waiting for receipt (timeout 120s)...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    print(f"Block:    {receipt.blockNumber}")
    status = "SUCCESS" if receipt.status == 1 else "FAILED"
    print(f"Status:   {status}")
    print(f"Gas used: {receipt.gasUsed}")

    final_bal = w3.eth.get_balance(acct.address)
    print(
        f"Deployer left: {final_bal} wei "
        f"= {final_bal/1e18:.8f} ETH (dust within safety buffer)"
    )
    recovery_bal = w3.eth.get_balance(recovery_addr)
    print(
        f"Recovery now:  {recovery_bal} wei = {recovery_bal/1e18:.8f} ETH"
    )

    if status != "SUCCESS":
        print("ERROR: sweep tx reverted. Investigate before closing terminal.", file=sys.stderr)
        return 6

    print("\n✅ Sweep complete. You can now close this terminal:")
    print("    exit")
    print(
        "(The PRIVATE_KEY env var dies with the shell. Combined with "
        "`unset HISTFILE` from action-plan §5, the key has no persistent "
        "footprint.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

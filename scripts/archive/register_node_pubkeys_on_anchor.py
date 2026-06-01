"""Sprint 623 — register Mac + droplet pubkeys on the live PublisherKeyAnchor.

Required: PRSM_DEPLOYER_PRIVATE_KEY env var (0x-prefixed hex)
         of an EOA funded with a few cents of Base ETH for gas.

Sends two TXs sequentially:
  1. register(Mac's Ed25519 pubkey)
  2. register(Droplet's Ed25519 pubkey)

After both confirm, verifies via anchor.lookup() that each
node_id maps to the registered pubkey.

Gas: ~50k per register call. Total cost at Base base fees
(~0.01 gwei) is roughly $0.01.

Usage:
  PRSM_DEPLOYER_PRIVATE_KEY=0x... python3 scripts/register_node_pubkeys_on_anchor.py
"""
from __future__ import annotations

import os
import sys
import time


ANCHOR = "0xd811ad9986f44f404b0fd992168a7cc76206df03"
RPC = os.environ.get("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
CHAIN_ID = 8453

MAC_NODE_ID = "cdefb8e5214c72731aeb3fbe6833fc6f"
MAC_PUBKEY = bytes.fromhex(
    "20a0a2f1c2b76a4e019f6a48736bdc765f146d3d85bf43e0e25daf0c6e638956"
)
DROPLET_NODE_ID = "484f003c895ee02ac7ed01e570a6a51f"
DROPLET_PUBKEY = bytes.fromhex(
    "bce6d2ec5fe2dcda42ed9916352c06011bc7c0b741cc1b4dbb0abbaba7106494"
)


def main() -> int:
    pk = (os.environ.get("PRSM_DEPLOYER_PRIVATE_KEY", "") or "").strip()
    if not pk:
        print(
            "ERROR: PRSM_DEPLOYER_PRIVATE_KEY env not set. "
            "Export your funded Base EOA private key (0x-prefixed)."
        )
        return 1
    if not pk.startswith("0x"):
        pk = "0x" + pk

    from web3 import Web3
    from eth_account import Account

    w3 = Web3(Web3.HTTPProvider(RPC))
    if not w3.is_connected():
        print(f"ERROR: cannot connect to {RPC}")
        return 1

    acct = Account.from_key(pk)
    sender = acct.address
    print(f"Sender: {sender}")
    balance_wei = w3.eth.get_balance(sender)
    balance_eth = balance_wei / 10**18
    print(f"Balance: {balance_eth:.6f} ETH")
    if balance_wei < 10**14:  # 0.0001 ETH safety floor
        print("ERROR: insufficient balance for gas; need ~0.0001 ETH.")
        return 1

    # Contract ABI (minimal — register + lookup)
    abi = [
        {
            "type": "function",
            "name": "register",
            "stateMutability": "nonpayable",
            "inputs": [{"name": "publicKey", "type": "bytes"}],
            "outputs": [],
        },
        {
            "type": "function",
            "name": "lookup",
            "stateMutability": "view",
            "inputs": [{"name": "nodeId", "type": "bytes16"}],
            "outputs": [{"name": "", "type": "bytes"}],
        },
    ]
    anchor = w3.eth.contract(
        address=Web3.to_checksum_address(ANCHOR), abi=abi,
    )

    nonce = w3.eth.get_transaction_count(sender)

    def send_register(label: str, pubkey: bytes, nonce_val: int) -> str:
        print(f"\n=== Submitting register() for {label} ===")
        # Pre-check: lookup current state
        node_id_bytes = bytes.fromhex(
            {"Mac": MAC_NODE_ID, "Droplet": DROPLET_NODE_ID}[label]
        )
        existing = anchor.functions.lookup(node_id_bytes).call()
        if existing and len(existing) > 0:
            if existing == pubkey:
                print(f"  Already registered with this pubkey — skipping.")
                return "skipped"
            print(
                f"  WARNING: lookup returned different pubkey ({existing.hex()}). "
                f"This shouldn't happen for fresh node_ids. Aborting."
            )
            return "aborted"

        tx = anchor.functions.register(pubkey).build_transaction({
            "from": sender,
            "nonce": nonce_val,
            "chainId": CHAIN_ID,
            "gas": 100_000,
            "maxFeePerGas": w3.to_wei("0.1", "gwei"),
            "maxPriorityFeePerGas": w3.to_wei("0.05", "gwei"),
        })
        signed = acct.sign_transaction(tx)
        # web3 v6: raw_transaction; v5: rawTransaction
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction"
        )
        tx_hash = w3.eth.send_raw_transaction(raw)
        print(f"  TX hash: 0x{tx_hash.hex()}")
        print(f"  Waiting for confirmation...")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        status = "SUCCESS" if receipt.status == 1 else "REVERTED"
        print(f"  {status} — block {receipt.blockNumber}, gasUsed {receipt.gasUsed}")
        return f"0x{tx_hash.hex()}"

    mac_result = send_register("Mac", MAC_PUBKEY, nonce)
    if mac_result not in ("skipped", "aborted"):
        nonce += 1
    droplet_result = send_register("Droplet", DROPLET_PUBKEY, nonce)

    # Post-verify
    print("\n=== Post-registration lookup ===")
    mac_lookup = anchor.functions.lookup(bytes.fromhex(MAC_NODE_ID)).call()
    droplet_lookup = anchor.functions.lookup(
        bytes.fromhex(DROPLET_NODE_ID),
    ).call()
    print(f"  Mac     ({MAC_NODE_ID}): "
          f"{'✓' if mac_lookup == MAC_PUBKEY else '✗'} "
          f"{mac_lookup.hex() if mac_lookup else '(empty)'}")
    print(f"  Droplet ({DROPLET_NODE_ID}): "
          f"{'✓' if droplet_lookup == DROPLET_PUBKEY else '✗'} "
          f"{droplet_lookup.hex() if droplet_lookup else '(empty)'}")

    return 0 if (mac_lookup == MAC_PUBKEY and droplet_lookup == DROPLET_PUBKEY) else 1


if __name__ == "__main__":
    sys.exit(main())

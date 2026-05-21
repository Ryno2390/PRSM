"""Sprint 674 — register a Lambda Labs Cloud GPU operator's pubkey on
the live PublisherKeyAnchor.

Required env vars:
  PRSM_DEPLOYER_PRIVATE_KEY  0x-prefixed Base EOA private key
                             (funded with a few cents of ETH for gas)
  LAMBDA_NODE_ID             node_id from the Lambda instance's
                             ~/.prsm/identity.json (32 hex chars)
  LAMBDA_PUBKEY_B64          base64 pubkey from the same identity.json

Sends ONE register(LAMBDA_PUBKEY) TX to the anchor at
0xd811ad9986f44f404b0fd992168a7cc76206df03. Gas: ~50k → ~$0.01 at
Base base fees.

After confirmation, verifies via anchor.lookup(LAMBDA_NODE_ID) that
the registration succeeded.

Pairs with docs/operations/lambda-gpu-operator-deploy.md.

Sprint 674 follow-on to sprint 623's register_node_pubkeys_on_anchor.py —
that one was hardcoded for Mac + droplet; this one is parameterized
for any new operator node.

NEVER COMMIT THE DEPLOYER PRIVATE KEY. Export via env at run time.
"""
from __future__ import annotations

import base64
import os
import sys
import time


ANCHOR = "0xd811ad9986f44f404b0fd992168a7cc76206df03"
RPC = os.environ.get("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
CHAIN_ID = 8453


def main() -> int:
    pk = (os.environ.get("PRSM_DEPLOYER_PRIVATE_KEY", "") or "").strip()
    if not pk:
        print(
            "ERROR: PRSM_DEPLOYER_PRIVATE_KEY env not set. "
            "Export your funded Base EOA private key (0x-prefixed).",
            file=sys.stderr,
        )
        return 1
    if not pk.startswith("0x"):
        pk = "0x" + pk

    lambda_node_id = (
        os.environ.get("LAMBDA_NODE_ID", "") or ""
    ).strip().lower()
    if len(lambda_node_id) != 32 or not all(
        c in "0123456789abcdef" for c in lambda_node_id
    ):
        print(
            f"ERROR: LAMBDA_NODE_ID must be 32 lowercase hex chars; "
            f"got {lambda_node_id!r}",
            file=sys.stderr,
        )
        return 1

    lambda_pubkey_b64 = (
        os.environ.get("LAMBDA_PUBKEY_B64", "") or ""
    ).strip()
    if not lambda_pubkey_b64:
        print(
            "ERROR: LAMBDA_PUBKEY_B64 env not set. Copy from the "
            "Lambda instance's ~/.prsm/identity.json `public_key_b64` field.",
            file=sys.stderr,
        )
        return 1
    try:
        pubkey_bytes = base64.b64decode(lambda_pubkey_b64)
    except Exception as exc:
        print(f"ERROR: LAMBDA_PUBKEY_B64 decode failed: {exc}", file=sys.stderr)
        return 1
    if len(pubkey_bytes) != 32:
        print(
            f"ERROR: pubkey must decode to 32 bytes (Ed25519); "
            f"got {len(pubkey_bytes)} bytes",
            file=sys.stderr,
        )
        return 1

    from web3 import Web3
    from eth_account import Account

    w3 = Web3(Web3.HTTPProvider(RPC))
    if not w3.is_connected():
        print(f"ERROR: Web3 not connected to {RPC}", file=sys.stderr)
        return 1

    deployer = Account.from_key(pk)
    print(f"Deployer EOA: {deployer.address}")
    balance = w3.eth.get_balance(deployer.address)
    print(f"Balance: {Web3.from_wei(balance, 'ether'):.6f} ETH")
    if balance < Web3.to_wei(0.0001, "ether"):
        print(
            "ERROR: Deployer balance < 0.0001 ETH — won't cover gas. "
            "Top up before re-running.",
            file=sys.stderr,
        )
        return 1

    # Check whether already registered
    from prsm.security.publisher_key_anchor.client import (
        PublisherKeyAnchorClient,
    )
    anchor = PublisherKeyAnchorClient(
        contract_address=ANCHOR, rpc_url=RPC,
    )
    existing = anchor.lookup(lambda_node_id)
    if existing:
        print(
            f"⚠ node_id {lambda_node_id} already has pubkey "
            f"{existing[:16]}... on anchor. Skipping registration.",
        )
        return 0

    # Build + send register() TX
    # The PublisherKeyAnchor's register signature takes (node_id_bytes, pubkey_bytes)
    # — see prsm/security/publisher_key_anchor/contract.py
    # Sprint 675 fix — actual contract signature is
    # `register(bytes publicKey)`; node_id is derived on-chain via
    # `bytes16(sha256(publicKey))`. Original draft used a phantom
    # `register(bytes16, bytes32)` which reverts at dispatch.
    register_abi = [{
        "name": "register",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "publicKey", "type": "bytes"}],
        "outputs": [],
    }]
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(ANCHOR),
        abi=register_abi,
    )
    import hashlib as _hashlib
    expected_node_id = _hashlib.sha256(pubkey_bytes).digest()[:16].hex()
    if expected_node_id != lambda_node_id:
        print(
            f"ERROR: supplied LAMBDA_NODE_ID {lambda_node_id} doesn't "
            f"match contract-derived sha256(pubkey)[:16] = "
            f"{expected_node_id}.",
            file=sys.stderr,
        )
        return 1

    nonce = w3.eth.get_transaction_count(deployer.address)
    gas_price = w3.eth.gas_price
    tx = contract.functions.register(pubkey_bytes).build_transaction({
        "from": deployer.address,
        "nonce": nonce,
        "gas": 150000,  # sprint 676 fix — was 80k, OOG'd live 2026-05-21
        "gasPrice": gas_price,
        "chainId": CHAIN_ID,
    })
    signed = deployer.sign_transaction(tx)
    raw_tx = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw_tx)
    print(f"TX sent: {tx_hash.hex()}")
    print("Waiting for confirmation...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status != 1:
        print(f"✗ TX reverted. Receipt: {dict(receipt)}", file=sys.stderr)
        return 1
    print(
        f"✓ Confirmed in block {receipt.blockNumber} "
        f"(gas used: {receipt.gasUsed})"
    )

    # Verify via lookup
    time.sleep(2)
    looked_up = anchor.lookup(lambda_node_id)
    if not looked_up:
        print(
            "✗ anchor.lookup returned empty after confirmation — "
            "RPC indexing delay?",
            file=sys.stderr,
        )
        return 1
    if looked_up != lambda_pubkey_b64:
        print(
            f"✗ Looked-up pubkey doesn't match registered:\n"
            f"  expected: {lambda_pubkey_b64}\n"
            f"  got:      {looked_up}",
            file=sys.stderr,
        )
        return 1
    print(
        f"🎯 anchor.lookup({lambda_node_id}) = {looked_up[:16]}... ✓"
    )
    print()
    print(f"Lambda operator {lambda_node_id} is now registered on the live")
    print(f"PublisherKeyAnchor + can be a stage in multi-host inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

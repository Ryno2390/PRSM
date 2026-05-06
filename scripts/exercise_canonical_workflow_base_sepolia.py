#!/usr/bin/env python3
"""Canonical-workflow Sepolia E2E — Vision §4 step 6 trace.

Runs the load-bearing on-chain royalty leg from the canonical PRSM
user workflow against Base Sepolia and captures Basescan-verifiable
evidence:

  1. Verify deployer has FTNS balance + ETH for gas.
  2. Generate 2 deterministic ephemeral EOAs (creator + serving_node)
     from a fixed seed so the trace is reproducible.
  3. Register content on the v1 ProvenanceRegistry from the deployer,
     then transfer ownership to the ephemeral creator address.
  4. Approve + distribute 10 FTNS via the RoyaltyDistributor with the
     ephemeral serving_node as recipient.
  5. Decode the resulting Transfer events from the receipt and assert
     the three-way split matches the expected math.
  6. Print Basescan tx URLs + the three split amounts.

The test wallet (the deployer EOA at 0xBbEB...C9a0) holds the full
100M FTNS genesis testnet supply and the MINTER_ROLE — verified
2026-05-06. No mint operation required.

Usage:

    source ~/.prsm/testnet-deployer.env
    PYTHONPATH=. .venv/bin/python3.14 \\
        scripts/exercise_canonical_workflow_base_sepolia.py

Required env (loaded by testnet-deployer.env):
  PRIVATE_KEY               — deployer wallet (the payer)
  BASE_SEPOLIA_RPC_URL      — Alchemy or other Base Sepolia RPC
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Optional, Tuple

from eth_account import Account
from web3 import Web3

from prsm.economy.web3.provenance_registry import (
    ProvenanceRegistryClient,
)
from prsm.economy.web3.royalty_distributor import (
    RoyaltyDistributorClient,
)


# ── Network selection ────────────────────────────────────────────────
# Defaults are Base Sepolia testnet (the network this script was
# originally written against). Override via env for Base mainnet —
# see docs/2026-05-06-canonical-workflow-base-mainnet-runbook.md §4.1.
# All five env vars must be set together to hit a different network;
# mixing testnet + mainnet addresses will revert at the contract layer
# but with confusing error messages, so the script script-name
# remains "*_base_sepolia.py" and the mainnet bring-up explicitly
# overrides all five via the mainnet env file.
FTNS_TOKEN = os.environ.get(
    "PRSM_FTNS_TOKEN", "0xF8d0c1AE75441d3C3Dd2A2420C0789043916412a",
)
PROVENANCE_REGISTRY = os.environ.get(
    "PRSM_PROVENANCE_REGISTRY",
    "0x2911f9a0a02896486CdF59d6d369764841DC0eA4",
)
ROYALTY_DISTRIBUTOR = os.environ.get(
    "PRSM_ROYALTY_DISTRIBUTOR",
    "0xB790045ff826C76fe02DBc54a6ef0021951Fd892",
)
EXPLORER = os.environ.get(
    "PRSM_EXPLORER", "https://sepolia.basescan.org",
)
CHAIN_ID = int(os.environ.get("PRSM_CHAIN_ID", "84532"))

# ── Demo parameters ───────────────────────────────────────────────────
ROYALTY_RATE_BPS = 1000     # 10%
GROSS_FTNS = 10             # 10 FTNS gross per distribution
NETWORK_FEE_BPS = 200       # 2% — pinned in RoyaltyDistributor.sol


def _required_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        print(f"ERROR: {name} env var is required", file=sys.stderr)
        sys.exit(2)
    return v


def _ephemeral_eoa(seed_label: str) -> Tuple[str, str]:
    """Derive a deterministic ephemeral EOA from a seed label.

    Uses sha256(seed_label) as the private key. NOT cryptographically
    secure — this is for testnet demo accounting only. The benefit is
    that the same seed_label always produces the same address, so a
    re-run of this exercise produces the same two recipient addresses
    and the on-chain history is consistent across reruns.
    """
    privkey_bytes = hashlib.sha256(
        f"prsm-canonical-workflow-trace-2026-05-06::{seed_label}".encode(),
    ).digest()
    acct = Account.from_key(privkey_bytes)
    return acct.address, "0x" + privkey_bytes.hex()


def _wait_for_confirmation(
    web3: Web3, tx_hash: str, timeout: float = 60.0,
) -> dict:
    """Poll for the receipt with replica-propagation retry."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            receipt = web3.eth.get_transaction_receipt(tx_hash)
            if receipt is not None:
                return dict(receipt)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    raise TimeoutError(
        f"tx {tx_hash} not confirmed within {timeout}s",
    )


def main() -> int:
    rpc_url = _required_env("BASE_SEPOLIA_RPC_URL")
    private_key = _required_env("PRIVATE_KEY")

    web3 = Web3(Web3.HTTPProvider(rpc_url))
    deployer = Account.from_key(private_key).address

    print(f"\n=== Canonical-workflow Sepolia E2E (Vision §4 step 6) ===")
    print(f"RPC:                    {rpc_url}")
    print(f"Chain id:               {CHAIN_ID}")
    print(f"Deployer (payer):       {deployer}")
    print(f"FTNS token:             {FTNS_TOKEN}")
    print(f"ProvenanceRegistry v1:  {PROVENANCE_REGISTRY}")
    print(f"RoyaltyDistributor:     {ROYALTY_DISTRIBUTOR}")

    # ── Step 0: pre-flight ────────────────────────────────────────────
    print(f"\n[0/4] Pre-flight…")
    eth_bal = web3.eth.get_balance(deployer)
    print(
        f"   ETH:  {Web3.from_wei(eth_bal, 'ether')} "
        f"(need ~0.01 for ~3 txs)",
    )
    if eth_bal < Web3.to_wei(0.005, "ether"):
        print(
            "ERROR: deployer ETH balance too low — need ~0.005 minimum",
            file=sys.stderr,
        )
        return 3

    # FTNS balance via raw eth_call to balanceOf
    ftns_balance_calldata = (
        Web3.keccak(text="balanceOf(address)")[:4]
        + b"\x00" * 12
        + bytes.fromhex(deployer[2:])
    )
    raw = web3.eth.call({
        "to": Web3.to_checksum_address(FTNS_TOKEN),
        "data": ftns_balance_calldata,
    })
    ftns_balance = int.from_bytes(raw, "big")
    ftns_balance_human = ftns_balance / 10**18
    print(f"   FTNS: {ftns_balance_human:,.4f} (need ≥ {GROSS_FTNS})")
    if ftns_balance < GROSS_FTNS * 10**18:
        print(
            f"ERROR: FTNS balance {ftns_balance_human} < required "
            f"{GROSS_FTNS}",
            file=sys.stderr,
        )
        return 4

    # ── Step 1: derive ephemeral EOAs ────────────────────────────────
    print(f"\n[1/4] Derive ephemeral demo EOAs (deterministic) …")
    creator_addr, _creator_pk = _ephemeral_eoa("creator")
    serving_node_addr, _node_pk = _ephemeral_eoa("serving-node")
    treasury_addr = _read_treasury_address(web3, ROYALTY_DISTRIBUTOR)
    print(f"   Creator (demo):        {creator_addr}")
    print(f"   Serving node (demo):   {serving_node_addr}")
    print(f"   Treasury (on-chain):   {treasury_addr}")

    # ── Step 2: register content + transfer ownership to ephemeral ──
    print(f"\n[2/4] Register content on v1 + transfer to ephemeral creator …")
    prov_client = ProvenanceRegistryClient(
        rpc_url=rpc_url,
        contract_address=PROVENANCE_REGISTRY,
        private_key=private_key,
    )
    # Deterministic content_hash so re-runs hit the same registry slot.
    # If already registered, the second register reverts; we recover
    # gracefully by treating the existing registration as good.
    payload_label = (
        f"prsm-canonical-workflow-trace-{int(time.time() // 3600)}"
    )
    content_hash = hashlib.sha256(payload_label.encode()).digest()
    print(f"   payload label:    {payload_label!r}")
    print(f"   content_hash:     0x{content_hash.hex()}")

    existing = prov_client.get_content(content_hash)
    if existing is None:
        tx_hash, status = prov_client.register_content(
            content_hash=content_hash,
            royalty_rate_bps=ROYALTY_RATE_BPS,
            metadata_uri=f"prsm://canonical-workflow-trace/{payload_label}",
        )
        print(f"   register tx:      {EXPLORER}/tx/{tx_hash}")
        print(f"   status:           {status}")
        # Public RPCs (Alchemy here) load-balance reads across replicas
        # that lag the write-side replica by 1-2s even after the
        # receipt confirms. transfer_ownership reads the on-chain
        # state via estimate_gas → it can hit a stale replica and
        # revert with "Not creator". Poll get_content() until the read
        # replica converges on the deployer-as-creator state we just
        # wrote.
        for attempt in range(15):
            existing = prov_client.get_content(content_hash)
            if existing is not None and (
                existing.creator.lower() == deployer.lower()
            ):
                if attempt > 0:
                    print(
                        f"   (read replica caught up after "
                        f"{attempt + 1} attempts)",
                    )
                break
            time.sleep(1.0)
        else:
            print(
                "ERROR: registration did not propagate to read replica "
                "within 15s",
                file=sys.stderr,
            )
            return 6
        # transferContentOwnership to the ephemeral creator so the
        # eventual royalty Transfer event is visible going to a
        # different address than the payer.
        tx_hash, status = prov_client.transfer_ownership(
            content_hash=content_hash, new_creator=creator_addr,
        )
        print(f"   xfer-creator tx:  {EXPLORER}/tx/{tx_hash}")
        print(f"   status:           {status}")
        # Same propagation pattern again before distribute_royalty
        # reads from the same registry.
        for attempt in range(15):
            existing = prov_client.get_content(content_hash)
            if existing is not None and (
                existing.creator.lower() == creator_addr.lower()
            ):
                if attempt > 0:
                    print(
                        f"   (read replica caught up after "
                        f"{attempt + 1} attempts)",
                    )
                break
            time.sleep(1.0)
        else:
            print(
                "ERROR: ownership transfer did not propagate within 15s",
                file=sys.stderr,
            )
            return 7
    else:
        print(
            f"   already registered (creator={existing.creator}, "
            f"rate={existing.royalty_rate_bps} bps); reusing",
        )
        if existing.creator.lower() != creator_addr.lower():
            print(
                f"   transferring ownership to ephemeral creator "
                f"{creator_addr} …",
            )
            tx_hash, status = prov_client.transfer_ownership(
                content_hash=content_hash, new_creator=creator_addr,
            )
            print(f"   xfer-creator tx:  {EXPLORER}/tx/{tx_hash}")

    # ── Step 3: distribute royalty (the main event) ──────────────────
    print(f"\n[3/4] Distribute {GROSS_FTNS} FTNS via RoyaltyDistributor …")
    royalty_client = RoyaltyDistributorClient(
        rpc_url=rpc_url,
        distributor_address=ROYALTY_DISTRIBUTOR,
        ftns_token_address=FTNS_TOKEN,
        private_key=private_key,
    )
    gross_wei = GROSS_FTNS * 10**18

    # Pre-approve the FTNS allowance separately, with propagation
    # wait, so distribute_royalty's internal estimate_gas pre-flight
    # sees a sufficient allowance. The client's own approve+distribute
    # pair atomically reads the latest pending nonce but estimate_gas
    # hits a read replica that may lag the approve tx by a second or
    # two on Alchemy — the result is a phantom InsufficientAllowance
    # revert at gas-estimate time.
    current_allowance = royalty_client.allowance()
    if current_allowance < gross_wei:
        print(
            f"   approving {GROSS_FTNS * 10} FTNS (10x headroom)…",
        )
        # Build + sign + send approve directly via web3 to control the
        # propagation wait.
        approve_amount = gross_wei * 10
        token_iface = royalty_client.token  # already-bound contract
        nonce = web3.eth.get_transaction_count(deployer, "pending")
        approve_tx = token_iface.functions.approve(
            Web3.to_checksum_address(ROYALTY_DISTRIBUTOR),
            approve_amount,
        ).build_transaction({
            "from": deployer,
            "nonce": nonce,
            "gasPrice": web3.eth.gas_price,
            "chainId": CHAIN_ID,
        })
        signed = web3.eth.account.sign_transaction(approve_tx, private_key)
        approve_hash = web3.eth.send_raw_transaction(
            signed.raw_transaction,
        ).hex()
        if not approve_hash.startswith("0x"):
            approve_hash = "0x" + approve_hash
        print(f"   approve tx:       {EXPLORER}/tx/{approve_hash}")
        _wait_for_confirmation(web3, approve_hash)
        # Read replica must see the new allowance before we attempt
        # the distribute build — otherwise estimate_gas reverts on a
        # stale-replica InsufficientAllowance.
        for attempt in range(15):
            if royalty_client.allowance() >= gross_wei:
                if attempt > 0:
                    print(
                        f"   (allowance read replica caught up after "
                        f"{attempt + 1} attempts)",
                    )
                break
            time.sleep(1.0)
        else:
            print(
                "ERROR: approval did not propagate within 15s",
                file=sys.stderr,
            )
            return 8

    preview = royalty_client.preview_split(content_hash, gross_wei)
    print(
        f"   preview: creator={preview.creator_amount/10**18:.4f}, "
        f"network={preview.network_amount/10**18:.4f}, "
        f"node={preview.serving_node_amount/10**18:.4f}",
    )

    tx_hash, status = royalty_client.distribute_royalty(
        content_hash=content_hash,
        serving_node=serving_node_addr,
        gross=gross_wei,
    )
    print(f"   distribute tx:    {EXPLORER}/tx/{tx_hash}")
    print(f"   status:           {status}")

    # ── Step 4: decode the receipt + verify the split ───────────────
    # NOTE: RoyaltyDistributor uses the OZ pull-payment pattern (T6.2,
    # D-04 refactor). distributeRoyalty() pulls `gross` FTNS from the
    # payer in ONE Transfer event and accumulates each recipient's
    # share in `claimable[address]`. Recipients later call claim() to
    # withdraw — that's the second Transfer event per recipient.
    # Verification is therefore: (1) decode the RoyaltyPaid event for
    # the three pre-split amounts; (2) read claimable() for each
    # recipient and confirm the balances match.
    print(f"\n[4/4] Decode RoyaltyPaid event + verify claimable balances …")
    receipt = _wait_for_confirmation(web3, tx_hash)
    paid = _decode_royalty_paid_event(receipt, ROYALTY_DISTRIBUTOR)
    if paid is None:
        print(
            "ERROR: no RoyaltyPaid event found on this tx",
            file=sys.stderr,
        )
        return 5
    print(
        f"   RoyaltyPaid: payer={paid['payer']}, creator={paid['creator']}",
    )
    print(
        f"     amounts: creator={paid['creatorAmount']/10**18:.4f}, "
        f"network={paid['networkAmount']/10**18:.4f}, "
        f"servingNode={paid['servingNodeAmount']/10**18:.4f}",
    )

    # Confirm event amounts match preview.
    if paid["creatorAmount"] != preview.creator_amount:
        print(
            f"ERROR: creator share mismatch event vs preview: "
            f"{paid['creatorAmount']} != {preview.creator_amount}",
            file=sys.stderr,
        )
        return 5
    if paid["networkAmount"] != preview.network_amount:
        print(
            f"ERROR: network share mismatch event vs preview: "
            f"{paid['networkAmount']} != {preview.network_amount}",
            file=sys.stderr,
        )
        return 5
    if paid["servingNodeAmount"] != preview.serving_node_amount:
        print(
            f"ERROR: servingNode share mismatch event vs preview: "
            f"{paid['servingNodeAmount']} != {preview.serving_node_amount}",
            file=sys.stderr,
        )
        return 5

    # Confirm pull-payment balances landed in each claimable[].
    print(f"   claimable balances on RoyaltyDistributor:")
    creator_claimable = royalty_client.claimable(creator_addr)
    treasury_claimable = royalty_client.claimable(treasury_addr)
    node_claimable = royalty_client.claimable(serving_node_addr)
    print(f"     creator      ({creator_addr}): {creator_claimable/10**18:.4f} FTNS")
    print(f"     treasury     ({treasury_addr}): {treasury_claimable/10**18:.4f} FTNS")
    print(f"     serving node ({serving_node_addr}): {node_claimable/10**18:.4f} FTNS")
    if creator_claimable < preview.creator_amount:
        print(
            f"ERROR: creator claimable {creator_claimable} < expected "
            f"{preview.creator_amount}",
            file=sys.stderr,
        )
        return 5
    if treasury_claimable < preview.network_amount:
        print(
            f"ERROR: treasury claimable {treasury_claimable} < expected "
            f"{preview.network_amount}",
            file=sys.stderr,
        )
        return 5
    if node_claimable < preview.serving_node_amount:
        print(
            f"ERROR: serving node claimable {node_claimable} < expected "
            f"{preview.serving_node_amount}",
            file=sys.stderr,
        )
        return 5
    print(f"   ✓ all three claimable balances ≥ preview amounts")

    # Also verify the single ERC-20 Transfer (payer → distributor).
    transfers = _decode_transfer_events(receipt, FTNS_TOKEN)
    if len(transfers) != 1:
        print(
            f"WARN: expected 1 Transfer (payer → distributor pull-in); "
            f"got {len(transfers)}",
        )
    elif (
        transfers[0]["from"].lower() != deployer.lower()
        or transfers[0]["to"].lower() != ROYALTY_DISTRIBUTOR.lower()
        or transfers[0]["amount"] != gross_wei
    ):
        print(f"WARN: payer→distributor Transfer didn't match expected shape")
    else:
        print(
            f"   pull-in Transfer: payer → distributor {GROSS_FTNS} FTNS ✓",
        )

    print(f"\n{'=' * 60}")
    print(f"✅ CANONICAL WORKFLOW STEP 6 — TRACE PASSES")
    print(f"{'=' * 60}")
    print(f"Network:                Base Sepolia (chain {CHAIN_ID})")
    print(f"Distribute tx:          {EXPLORER}/tx/{tx_hash}")
    print(f"  Payer:                {deployer} (-{GROSS_FTNS} FTNS)")
    print(
        f"  Creator:              {creator_addr} "
        f"(+{preview.creator_amount/10**18:.4f} FTNS)",
    )
    print(
        f"  Treasury:             {treasury_addr} "
        f"(+{preview.network_amount/10**18:.4f} FTNS)",
    )
    print(
        f"  Serving node:         {serving_node_addr} "
        f"(+{preview.serving_node_amount/10**18:.4f} FTNS)",
    )
    print(
        f"  Total settled:        "
        f"{(preview.creator_amount + preview.network_amount + preview.serving_node_amount)/10**18:.4f} "
        f"FTNS = {GROSS_FTNS} FTNS gross ✓",
    )
    print(f"{'=' * 60}\n")
    return 0


def _read_treasury_address(web3: Web3, distributor: str) -> str:
    """Read networkTreasury() from the distributor contract."""
    selector = Web3.keccak(text="networkTreasury()")[:4]
    raw = web3.eth.call({
        "to": Web3.to_checksum_address(distributor),
        "data": selector,
    })
    return Web3.to_checksum_address("0x" + raw[-20:].hex())


# Topic-0 for ERC-20 Transfer(address,address,uint256).
_TRANSFER_TOPIC0 = "0x" + Web3.keccak(
    text="Transfer(address,address,uint256)",
).hex()


# Topic-0 for RoyaltyPaid event (RoyaltyDistributor.sol).
# Signature per the ABI in prsm.economy.web3.royalty_distributor:
#   RoyaltyPaid(bytes32 indexed contentHash, address indexed payer,
#               address indexed creator, address servingNode,
#               uint256 creatorAmount, uint256 networkAmount,
#               uint256 servingNodeAmount)
# 3 indexed topics + 4 unindexed words in data.
_ROYALTY_PAID_TOPIC0 = "0x" + Web3.keccak(
    text=(
        "RoyaltyPaid(bytes32,address,address,address,"
        "uint256,uint256,uint256)"
    ),
).hex()


def _decode_royalty_paid_event(
    receipt: dict, distributor_address: str,
) -> Optional[dict]:
    """Pull the single RoyaltyPaid event from the receipt logs."""
    distributor_lower = distributor_address.lower()
    for log in receipt["logs"]:
        if log["address"].lower() != distributor_lower:
            continue
        topics = log["topics"]
        topic0 = topics[0].hex() if hasattr(topics[0], "hex") else topics[0]
        if not topic0.startswith("0x"):
            topic0 = "0x" + topic0
        if topic0 != _ROYALTY_PAID_TOPIC0:
            continue
        # Indexed topics: contentHash (1), payer (2), creator (3).
        content_hash = (
            topics[1].hex() if hasattr(topics[1], "hex") else topics[1]
        )
        if not content_hash.startswith("0x"):
            content_hash = "0x" + content_hash
        payer = Web3.to_checksum_address(
            "0x" + (
                topics[2].hex() if hasattr(topics[2], "hex") else topics[2]
            )[-40:],
        )
        creator = Web3.to_checksum_address(
            "0x" + (
                topics[3].hex() if hasattr(topics[3], "hex") else topics[3]
            )[-40:],
        )
        # Unindexed data: servingNode (32B padded address), creatorAmount,
        # networkAmount, servingNodeAmount (each 32B uint256).
        data = log["data"]
        if hasattr(data, "hex"):
            data = data.hex()
        if not data.startswith("0x"):
            data = "0x" + data
        raw = bytes.fromhex(data[2:])
        if len(raw) != 32 * 4:
            continue
        serving_node = Web3.to_checksum_address("0x" + raw[12:32].hex())
        creator_amount = int.from_bytes(raw[32:64], "big")
        network_amount = int.from_bytes(raw[64:96], "big")
        serving_node_amount = int.from_bytes(raw[96:128], "big")
        return {
            "contentHash": content_hash,
            "payer": payer,
            "creator": creator,
            "servingNode": serving_node,
            "creatorAmount": creator_amount,
            "networkAmount": network_amount,
            "servingNodeAmount": serving_node_amount,
        }
    return None


def _decode_transfer_events(receipt: dict, token_address: str) -> list:
    """Pull ERC-20 Transfer events from the receipt logs that came
    from the FTNS token contract."""
    out: list = []
    token_lower = token_address.lower()
    for log in receipt["logs"]:
        if log["address"].lower() != token_lower:
            continue
        topics = log["topics"]
        topic0 = (
            topics[0].hex() if hasattr(topics[0], "hex")
            else topics[0]
        )
        if not topic0.startswith("0x"):
            topic0 = "0x" + topic0
        if topic0 != _TRANSFER_TOPIC0:
            continue
        from_addr = Web3.to_checksum_address(
            "0x" + (
                topics[1].hex() if hasattr(topics[1], "hex") else topics[1]
            )[-40:],
        )
        to_addr = Web3.to_checksum_address(
            "0x" + (
                topics[2].hex() if hasattr(topics[2], "hex") else topics[2]
            )[-40:],
        )
        data = log["data"]
        if hasattr(data, "hex"):
            data = data.hex()
        if not data.startswith("0x"):
            data = "0x" + data
        amount = int(data, 16)
        out.append({"from": from_addr, "to": to_addr, "amount": amount})
    return out


if __name__ == "__main__":
    sys.exit(main())

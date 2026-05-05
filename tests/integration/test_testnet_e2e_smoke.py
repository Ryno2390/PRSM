"""T8 (2026-05-05) — E2E smoke test against deployed Base Sepolia contracts.

Validates the full data-contribution loop end-to-end on the LIVE testnet:
  1. Read FTNS balance for the deployer (sanity check that the chain is
     reachable + contracts deployed)
  2. Compute a content_hash for a synthetic file
  3. Register it on-chain via ProvenanceRegistry.registerContent
  4. Verify is_registered() returns True
  5. Read claimable balance for the deployer (should be 0 — nobody paid yet)

This is a NETWORK test that hits the actual deployed Base Sepolia stack.
Skipped by default in CI; run explicitly:

    pytest tests/integration/test_testnet_e2e_smoke.py -m testnet -v

Pre-reqs:
  - PRIVATE_KEY env var (with 0x prefix, 66 chars total)
  - BASE_SEPOLIA_RPC_URL env var (or default https://sepolia.base.org)
  - The signer must have:
      * ≥ 0.001 Base Sepolia ETH (gas for registerContent)
      * (Optional) some testnet-FTNS for full payment-loop testing

What this DOESN'T test (yet — separate test for full loop):
  - The pay-for-access flow (ContentEconomy.process_content_access)
  - Royalty distribution from another address
  - Claim withdrawal (depends on royalties having been credited)

Those require a 2-node setup which is out of scope for a single-process
smoke test. Tracked as a follow-up.
"""
from __future__ import annotations

import hashlib
import os
import time

import pytest

# Skip everything in this file unless the testnet marker is selected.
pytestmark = pytest.mark.testnet


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def network_cfg():
    """Load the testnet network config; skip module if addresses missing."""
    from prsm.config.networks import get_network_config
    cfg = get_network_config("testnet")
    if not cfg.is_complete():
        pytest.skip("testnet network config incomplete — run T1 deploy first")
    return cfg


@pytest.fixture(scope="module")
def signer():
    """Load the deployer's signing key from env."""
    pk = os.environ.get("PRIVATE_KEY") or os.environ.get("FTNS_WALLET_PRIVATE_KEY")
    if not pk or not pk.startswith("0x") or len(pk) != 66:
        pytest.skip("PRIVATE_KEY env var not set or malformed")
    from eth_account import Account
    return Account.from_key(pk)


@pytest.fixture(scope="module")
def rpc_url():
    return os.environ.get("BASE_SEPOLIA_RPC_URL", "https://sepolia.base.org")


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


def test_chain_is_reachable(rpc_url, network_cfg):
    """Sanity: can we read chainId from the configured RPC?"""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    chain_id = w3.eth.chain_id
    assert chain_id == network_cfg.chain_id, (
        f"chainId mismatch: rpc returned {chain_id}, "
        f"expected {network_cfg.chain_id}"
    )


def test_ftns_balance_readable(rpc_url, network_cfg, signer):
    """Sanity: can we read the deployer's FTNS balance?"""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    erc20_abi = [{
        "inputs": [{"name": "owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    }]
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(network_cfg.ftns_token),
        abi=erc20_abi)
    balance = contract.functions.balanceOf(signer.address).call()
    # Deployer should have a non-zero balance (genesis 100M was minted to it)
    assert balance > 0, (
        f"deployer {signer.address} has 0 FTNS — did you run T1 deploy?"
    )


def test_provenance_register_and_verify(
    rpc_url, network_cfg, signer
):
    """Round-trip: register a unique content hash on-chain, verify it lands."""
    from prsm.economy.web3.provenance_registry import (
        ProvenanceRegistryClient, compute_content_hash,
    )

    client = ProvenanceRegistryClient(
        rpc_url=rpc_url,
        contract_address=network_cfg.provenance_registry,
        private_key=signer.key.hex(),
    )

    # Use a unique synthetic content (timestamp-based) so the test is
    # idempotent across reruns — never collides with a previously
    # registered hash.
    synthetic = f"T8 smoke {time.time_ns()}".encode()
    content_hash = compute_content_hash(signer.address, synthetic)
    assert len(content_hash) == 32

    # Pre-condition: should NOT be registered yet (unique hash)
    assert not client.is_registered(content_hash), (
        "synthetic hash unexpectedly already registered — clock collision?"
    )

    # Register on-chain
    tx_hash, status = client.register_content(
        content_hash=content_hash,
        royalty_rate_bps=500,  # 5%
        metadata_uri="ipfs://QmSmokeTest",
    )
    assert tx_hash and tx_hash.startswith("0x")

    # Post-condition: should now be registered. Poll with short retries
    # to absorb RPC state-read propagation lag (Alchemy returns stale
    # reads for a few seconds after tx confirmation).
    registered = False
    for _ in range(8):
        if client.is_registered(content_hash):
            registered = True
            break
        time.sleep(2)
    assert registered, (
        f"content_hash registration didn't take effect after 16s "
        f"(tx {tx_hash})"
    )


def test_claimable_readable(rpc_url, network_cfg, signer):
    """Read RoyaltyDistributor.claimable[deployer] — should not throw."""
    from prsm.economy.web3.royalty_distributor import RoyaltyDistributorClient

    client = RoyaltyDistributorClient(
        rpc_url=rpc_url,
        distributor_address=network_cfg.royalty_distributor,
        ftns_token_address=network_cfg.ftns_token,
        private_key=None,  # read-only
    )
    claimable = client.claimable(signer.address)
    # No assertion on value — could be 0 or anything depending on payment
    # history. Just verify the call works (i.e., contract is reachable
    # and the new ABI we added in T6.2 is wired correctly).
    assert isinstance(claimable, int)
    assert claimable >= 0


def test_content_uploader_register_smoke(
    rpc_url, network_cfg, signer, tmp_path
):
    """Exercise ContentUploader._register_on_chain against the real chain.

    Doesn't go through full upload() (which requires IPFS). Just calls
    the helper directly with a unique hash + synthetic CID.
    """
    from prsm.node.content_uploader import ContentUploader
    from prsm.economy.web3.provenance_registry import (
        ProvenanceRegistryClient, compute_content_hash,
    )
    from unittest.mock import MagicMock

    client = ProvenanceRegistryClient(
        rpc_url=rpc_url,
        contract_address=network_cfg.provenance_registry,
        private_key=signer.key.hex(),
    )
    uploader = ContentUploader(
        identity=MagicMock(),
        gossip=MagicMock(),
        ledger=MagicMock(),
        creator_address=signer.address,
        provenance_client=client,
    )
    # Unique synthetic content
    synthetic = f"T8 ContentUploader smoke {time.time_ns()}".encode()
    h = compute_content_hash(signer.address, synthetic)
    h_hex = "0x" + h.hex()

    tx_hash = uploader._register_on_chain(
        provenance_hash_hex=h_hex,
        royalty_rate=0.05,
        cid=f"QmSmoke{int(time.time())}",
    )
    assert tx_hash and tx_hash.startswith("0x"), (
        f"on-chain registration failed (likely insufficient gas at "
        f"{signer.address}? returned: {tx_hash!r})"
    )
    # Idempotent re-call should return None (already-registered branch)
    tx_hash_2 = uploader._register_on_chain(
        provenance_hash_hex=h_hex,
        royalty_rate=0.05,
        cid="QmSmokeDup",
    )
    assert tx_hash_2 is None, (
        f"second call should detect already-registered; got tx {tx_hash_2}"
    )

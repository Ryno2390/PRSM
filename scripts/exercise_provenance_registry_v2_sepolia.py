#!/usr/bin/env python3
"""PRSM-PROV-1 Item 7 (T7.7) — Live exercise of ProvenanceRegistryV2
against a deployed (Sepolia or local-Hardhat) contract.

Drives the dispute round-trip end-to-end against a real chain:
  1. Compute an embedding commitment from a (model_id, dim, vector).
  2. Register a content_hash with that commitment.
  3. Read the on-chain commitment back via verifyEmbeddingCommitment.
  4. Run a positive dispute: same vector → contract returns true.
  5. Run a negative dispute: different vector → contract returns false.
  6. Run the anti-zero-forgery dispute: byte-hash-only content →
     submitting zero claim returns false (the load-bearing safety
     property).

Use this AFTER deploying with
``contracts/scripts/deploy-provenance-registry-v2.js`` to confirm
the deployed contract matches the V2Client's expectations and the
Hardhat-tested invariants survive a real-chain round trip.

Required env vars:
  PRSM_PROVENANCE_REGISTRY_V2_ADDRESS  -  deployed contract address
  PRSM_RPC_URL                         -  RPC endpoint (Sepolia /
                                          local Hardhat)
  PRSM_PRIVATE_KEY                     -  signing key (must hold
                                          enough native ETH for gas)

Usage:
  # Local Hardhat (after `npx hardhat node` in another terminal +
  # running deploy-provenance-registry-v2.js --network localhost):
  PRSM_PROVENANCE_REGISTRY_V2_ADDRESS=0x... \
  PRSM_RPC_URL=http://127.0.0.1:8545 \
  PRSM_PRIVATE_KEY=0x... \
      python scripts/exercise_provenance_registry_v2_sepolia.py

  # Sepolia:
  PRSM_PROVENANCE_REGISTRY_V2_ADDRESS=0x... \
  PRSM_RPC_URL=https://sepolia.infura.io/v3/... \
  PRSM_PRIVATE_KEY=0x... \
      python scripts/exercise_provenance_registry_v2_sepolia.py

Exits 0 on success, non-zero with a diagnostic on any assertion miss.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Optional

from prsm.economy.web3.provenance_registry_v2 import (
    ProvenanceRegistryV2Client,
    ZERO_BYTES32,
    compute_embedding_commitment,
    compute_kind_tag,
)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        print(f"ERROR: {name} env var is required", file=sys.stderr)
        sys.exit(2)
    return value


def _content_hash_for(label: str) -> bytes:
    """A keccak-shaped 32-byte hash for the test fixtures. Production
    callers pass content_hash from ContentUploader's CID-derived
    sha256; this script just needs deterministic 32-byte values."""
    return hashlib.sha256(label.encode("utf-8")).digest()


def _round_trip(
    client: ProvenanceRegistryV2Client,
    *,
    label: str,
    model_id: str,
    dim: int,
    vector_bytes: bytes,
    royalty_rate_bps: int = 100,
    metadata_uri: str = "ipfs://placeholder",
    fingerprint_kind: Optional[bytes] = None,
) -> bytes:
    """Register `label` content with an embedding commitment derived
    from (model_id, dim, vector_bytes). Returns the content_hash."""
    content_hash = _content_hash_for(label)
    commitment = compute_embedding_commitment(
        model_id=model_id, dim=dim, vector_bytes=vector_bytes,
    )
    kind = fingerprint_kind if fingerprint_kind is not None \
        else compute_kind_tag("text-vector")

    print(f"  → registerContent({label[:16]!r}…)")
    tx_hash, status = client.register_content_v2(
        content_hash=content_hash,
        royalty_rate_bps=royalty_rate_bps,
        metadata_uri=metadata_uri,
        embedding_commitment=commitment,
        fingerprint_kind=kind,
    )
    print(f"    tx_hash: {tx_hash} status: {status}")

    # Read back to confirm the on-chain commitment matches.
    onchain = client.get_content(content_hash)
    if onchain is None:
        print(
            f"ERROR: get_content({label!r}) returned None after register",
            file=sys.stderr,
        )
        sys.exit(3)
    if bytes(onchain.embedding_commitment) != commitment:
        print(
            f"ERROR: on-chain commitment mismatch for {label!r}: "
            f"on-chain={onchain.embedding_commitment.hex()} "
            f"expected={commitment.hex()}",
            file=sys.stderr,
        )
        sys.exit(3)
    print(f"    on-chain embedding_commitment: {commitment.hex()[:16]}…")
    return content_hash


def main() -> int:
    rpc_url = _required_env("PRSM_RPC_URL")
    contract_address = _required_env("PRSM_PROVENANCE_REGISTRY_V2_ADDRESS")
    private_key = _required_env("PRSM_PRIVATE_KEY")

    print(f"\n=== ProvenanceRegistryV2 live exercise ===")
    print(f"RPC:      {rpc_url}")
    print(f"Contract: {contract_address}")

    client = ProvenanceRegistryV2Client(
        rpc_url=rpc_url,
        contract_address=contract_address,
        private_key=private_key,
    )
    print(f"Signer:   {client.address}")

    chain_id = client.web3.eth.chain_id
    print(f"Chain id: {chain_id}")

    # Use a per-run label suffix so reruns against the same chain
    # don't collide on "Already registered". The contract is
    # idempotent per content_hash; making the hash unique lets the
    # script be re-run without manual cleanup.
    run_id = str(int(time.time()))

    # ── Test 1: positive dispute round-trip ────────────────────────
    print(f"\n[1/3] Positive dispute round-trip…")
    model_id = "openai/text-embedding-3-small"
    dim = 8
    correct_vector = b"".join(
        i.to_bytes(4, "little") for i in range(dim)
    )
    label = f"content-positive-{run_id}"
    content_hash = _round_trip(
        client,
        label=label,
        model_id=model_id,
        dim=dim,
        vector_bytes=correct_vector,
    )
    # Same vector should win the dispute.
    if not client.dispute_provenance(
        content_hash=content_hash,
        model_id=model_id, dim=dim, vector_bytes=correct_vector,
    ):
        print("ERROR: positive dispute returned False", file=sys.stderr)
        return 4
    print("    ✓ same-vector dispute returned True (matches on-chain)")

    # ── Test 2: negative dispute (different vector) ────────────────
    print(f"\n[2/3] Negative dispute (substituted vector)…")
    wrong_vector = b"".join(
        (i + 999).to_bytes(4, "little") for i in range(dim)
    )
    if client.dispute_provenance(
        content_hash=content_hash,
        model_id=model_id, dim=dim, vector_bytes=wrong_vector,
    ):
        print(
            "ERROR: negative dispute returned True for substituted "
            "vector — V2 commitment binding is broken",
            file=sys.stderr,
        )
        return 5
    print("    ✓ different-vector dispute returned False")

    # ── Test 3: anti-zero-forgery against legacy byte-hash-only ────
    print(f"\n[3/3] Anti-zero-forgery against byte-hash-only content…")
    legacy_label = f"content-byte-hash-only-{run_id}"
    legacy_hash = _content_hash_for(legacy_label)
    print(f"  → registerContent legacy ({legacy_label[:24]}…)")
    tx_hash, status = client.register_content_v2(
        content_hash=legacy_hash,
        royalty_rate_bps=100,
        metadata_uri="ipfs://legacy",
        embedding_commitment=ZERO_BYTES32,
        fingerprint_kind=ZERO_BYTES32,
    )
    print(f"    tx_hash: {tx_hash} status: {status}")
    # Verify directly against the contract: claimed=zero, on-chain=zero,
    # but the anti-zero-forgery short-circuit must return false anyway.
    is_match = client.verify_embedding_commitment(
        content_hash=legacy_hash, claimed=ZERO_BYTES32,
    )
    if is_match:
        print(
            "ERROR: verifyEmbeddingCommitment returned True for "
            "(zero, zero) — anti-zero-forgery guard is BROKEN",
            file=sys.stderr,
        )
        return 6
    print("    ✓ verify(zero, zero) returned False (guard active)")

    print(f"\n{'=' * 60}")
    print("✅ ALL CHECKS PASSED")
    print(f"{'=' * 60}")
    print(f"Contract:           {contract_address}")
    print(f"Chain id:           {chain_id}")
    print(f"Round-trip hashes:")
    print(f"  positive content: 0x{content_hash.hex()}")
    print(f"  legacy content:   0x{legacy_hash.hex()}")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

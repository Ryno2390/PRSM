"""Unit tests for ShardExecutionReceipt + VerificationStrategy.

Phase 2 Task 2. Exercises the signed-receipt round-trip:
create → sign → verify — with fresh Ed25519 NodeIdentity instances.
"""
from __future__ import annotations

import asyncio
import hashlib

from prsm.compute.shard_receipt import (
    ReceiptOnlyVerification,
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)
from prsm.node.identity import generate_node_identity


def _fresh_identity():
    return generate_node_identity(display_name="shard-receipt-test")


def _run(coro):
    return asyncio.run(coro)


def test_receipt_verification_happy_path():
    """A receipt signed by a known identity verifies correctly."""
    identity = _fresh_identity()

    job_id = "job-happy-path"
    shard_index = 0
    output_bytes = b"deterministic output payload"
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id=job_id,
        shard_index=shard_index,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    signature = identity.sign(payload)

    receipt = ShardExecutionReceipt(
        job_id=job_id,
        shard_index=shard_index,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=output_hash,
        executed_at_unix=executed_at,
        signature=signature,
    )

    verifier = ReceiptOnlyVerification()
    assert _run(verifier.verify(receipt.to_dict(), output_bytes)) is True


def test_receipt_verification_bad_signature():
    """A tampered signature fails verification."""
    identity = _fresh_identity()

    output_bytes = b"original output"
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id="job-bad-sig",
        shard_index=0,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    good_sig = identity.sign(payload)
    tampered_sig = good_sig[:-4] + ("AAAA" if good_sig[-4:] != "AAAA" else "BBBB")

    receipt = ShardExecutionReceipt(
        job_id="job-bad-sig",
        shard_index=0,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=output_hash,
        executed_at_unix=executed_at,
        signature=tampered_sig,
    )

    verifier = ReceiptOnlyVerification()
    assert _run(verifier.verify(receipt.to_dict(), output_bytes)) is False


def test_receipt_verification_output_hash_mismatch():
    """A receipt whose output_hash doesn't match the actual bytes fails verification."""
    identity = _fresh_identity()

    wrong_bytes = b"attacker tried to substitute this"
    declared_hash = hashlib.sha256(b"original honest output").hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id="job-hash-mismatch",
        shard_index=0,
        output_hash=declared_hash,
        executed_at_unix=executed_at,
    )
    signature = identity.sign(payload)

    receipt = ShardExecutionReceipt(
        job_id="job-hash-mismatch",
        shard_index=0,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=declared_hash,
        executed_at_unix=executed_at,
        signature=signature,
    )

    verifier = ReceiptOnlyVerification()
    assert _run(verifier.verify(receipt.to_dict(), wrong_bytes)) is False


def test_receipt_verification_wrong_pubkey():
    """A signature valid over the correct payload but signed by a
    different key (attacker-controlled) fails verification."""
    victim = _fresh_identity()
    attacker = _fresh_identity()
    assert victim.public_key_b64 != attacker.public_key_b64

    output_bytes = b"honest output"
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id="job-wrong-key",
        shard_index=0,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    attacker_sig = attacker.sign(payload)

    receipt = ShardExecutionReceipt(
        job_id="job-wrong-key",
        shard_index=0,
        provider_id=victim.node_id,
        provider_pubkey_b64=victim.public_key_b64,
        output_hash=output_hash,
        executed_at_unix=executed_at,
        signature=attacker_sig,
    )

    verifier = ReceiptOnlyVerification()
    assert _run(verifier.verify(receipt.to_dict(), output_bytes)) is False

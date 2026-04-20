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


def test_receipt_verification_claimed_id_mismatches_pubkey():
    """Codex P2: receipt claims provider A's node_id but carries B's
    pubkey. Pre-fix this verified True (self-authenticating); now the
    node_id-binding check rejects it before the signature check."""
    victim = _fresh_identity()
    attacker = _fresh_identity()
    assert victim.node_id != attacker.node_id

    output_bytes = b"output"
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id="job-attack",
        shard_index=0,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    attacker_sig = attacker.sign(payload)

    receipt = ShardExecutionReceipt(
        job_id="job-attack",
        shard_index=0,
        provider_id=victim.node_id,
        provider_pubkey_b64=attacker.public_key_b64,
        output_hash=output_hash,
        executed_at_unix=executed_at,
        signature=attacker_sig,
    )

    verifier = ReceiptOnlyVerification()
    assert _run(verifier.verify(receipt.to_dict(), output_bytes)) is False


def test_receipt_verification_expected_provider_mismatch():
    """Codex P2: dispatcher supplies expected_provider_id; receipt claims
    a different provider. Verification must reject even if signature
    would otherwise verify."""
    sender = _fresh_identity()
    intended = _fresh_identity()
    assert sender.node_id != intended.node_id

    output_bytes = b"output"
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id="job-expected",
        shard_index=0,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    signature = sender.sign(payload)

    receipt = ShardExecutionReceipt(
        job_id="job-expected",
        shard_index=0,
        provider_id=sender.node_id,
        provider_pubkey_b64=sender.public_key_b64,
        output_hash=output_hash,
        executed_at_unix=executed_at,
        signature=signature,
    )

    verifier = ReceiptOnlyVerification()
    assert _run(
        verifier.verify(
            receipt.to_dict(), output_bytes,
            expected_provider_id=intended.node_id,
        )
    ) is False


def test_receipt_tee_attestation_schema_roundtrip():
    """Phase 2.1 Line Item C: tee_attestation is a reserved optional
    dict on ShardExecutionReceipt. Absence and presence both roundtrip
    cleanly via to_dict / from_dict. Verification logic lands later —
    this test only guards the wire format."""
    identity = _fresh_identity()
    output_hash = hashlib.sha256(b"x").hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload("job-tee", 0, output_hash, executed_at)
    signature = identity.sign(payload)

    receipt_none = ShardExecutionReceipt(
        job_id="job-tee", shard_index=0,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=output_hash, executed_at_unix=executed_at,
        signature=signature,
    )
    d_none = receipt_none.to_dict()
    assert d_none["tee_attestation"] is None
    assert ShardExecutionReceipt.from_dict(d_none).tee_attestation is None

    quote = {
        "quote_type": "intel_sgx_dcap",
        "quote_b64": "BASE64_QUOTE_BYTES",
        "binding": {
            "input_hash": hashlib.sha256(b"input").hexdigest(),
            "output_hash": output_hash,
            "shard_id": "job-tee:0",
            "nonce": "b1c7e2fa",
        },
    }
    receipt_tee = ShardExecutionReceipt(
        job_id="job-tee", shard_index=0,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=output_hash, executed_at_unix=executed_at,
        signature=signature,
        tee_attestation=quote,
    )
    d_tee = receipt_tee.to_dict()
    assert d_tee["tee_attestation"] == quote
    assert ShardExecutionReceipt.from_dict(d_tee).tee_attestation == quote

    # Existing verifier ignores the field (verification deferred to
    # Phase 2.1 followup). Both receipts still pass receipt-only checks.
    verifier = ReceiptOnlyVerification()
    assert _run(verifier.verify(d_none, b"x")) is True
    assert _run(verifier.verify(d_tee, b"x")) is True


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

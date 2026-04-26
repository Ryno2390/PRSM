"""
Inference receipt signing and verification.

Phase 3.x.1 Task 2 — wires :func:`InferenceReceipt.signing_payload` to the
existing :class:`prsm.node.identity.NodeIdentity` Ed25519 keypair so that
every inference can produce a verifiable signed receipt.

Verification path: callers can verify ``InferenceReceipt`` independently of
PRSM by holding only the settler node's public key. Anyone with the
public key can call :func:`verify_receipt` against a serialized receipt and
confirm:

1. The signature was produced by the holder of the matching private key
2. None of the receipt fields have been altered since signing

Combined with TEE attestation verification (separate, against the platform
vendor's attestation service), this converts MCP inference from "trust
the provider" to "verifiable inference" per `PRSM_Vision.md` §7.

Reuses ``NodeIdentity.sign``/``verify`` rather than introducing parallel
crypto. Receipts use the same Ed25519 keypair the node uses for everything
else; one identity, one trust anchor.
"""

from __future__ import annotations

import base64
import dataclasses
from typing import Optional

from prsm.compute.inference.models import InferenceReceipt
from prsm.node.identity import NodeIdentity, verify_signature


def sign_receipt(
    receipt: InferenceReceipt, identity: NodeIdentity
) -> InferenceReceipt:
    """Return a copy of ``receipt`` with ``settler_signature`` populated.

    The original receipt is unchanged (frozen dataclass). The new receipt
    has:
    - ``settler_node_id`` set to ``identity.node_id``
    - ``settler_signature`` set to the Ed25519 signature of
      ``signing_payload()``

    Note: ``signing_payload()`` already excludes the signature itself
    (verified by Task 1 ``test_signing_payload_excludes_signature``), so we
    can safely populate ``settler_node_id`` first, then sign, then
    populate ``settler_signature`` — no circular reference issue.
    """
    # Step 1: ensure settler_node_id is set BEFORE we compute the signing
    # payload, since signing_payload includes settler_node_id (so callers
    # can verify which node signed).
    intermediate = dataclasses.replace(
        receipt,
        settler_node_id=identity.node_id,
    )

    # Step 2: sign the canonical bytes
    payload = intermediate.signing_payload()
    signature_b64 = identity.sign(payload)
    signature_bytes = base64.b64decode(signature_b64)

    # Step 3: produce final receipt with both fields populated
    return dataclasses.replace(
        intermediate,
        settler_signature=signature_bytes,
    )


def verify_receipt(
    receipt: InferenceReceipt,
    public_key_b64: Optional[str] = None,
    identity: Optional[NodeIdentity] = None,
) -> bool:
    """Verify ``receipt.settler_signature`` against ``signing_payload()``.

    Provide one of:
    - ``public_key_b64`` — base64-encoded Ed25519 public key (matches the
      settler's published key)
    - ``identity`` — a :class:`NodeIdentity` instance whose public key
      matches the signer

    Returns False if:
    - Signature is empty (receipt was never signed)
    - Signature does not match the canonical signing payload
    - Public key does not correspond to the signer
    - Any cryptographic verification step throws

    Tampering with any field included in ``signing_payload()`` invalidates
    the signature — verified by Task 2 tampering tests.
    """
    if not receipt.settler_signature:
        return False
    if public_key_b64 is None and identity is None:
        return False

    payload = receipt.signing_payload()
    signature_b64 = base64.b64encode(receipt.settler_signature).decode()

    if identity is not None:
        return identity.verify(payload, signature_b64)
    # public_key_b64 is not None here per the early-return above
    assert public_key_b64 is not None  # for type-checkers
    return verify_signature(public_key_b64, payload, signature_b64)


def is_signed(receipt: InferenceReceipt) -> bool:
    """Return True iff ``receipt`` carries a non-empty settler signature.

    Note: this is NOT a verification check — it only tests that the field
    is populated. Use :func:`verify_receipt` to validate the signature
    cryptographically.
    """
    return bool(receipt.settler_signature) and bool(receipt.settler_node_id)

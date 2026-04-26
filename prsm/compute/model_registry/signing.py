"""
Model manifest signing and verification.

Phase 3.x.2 Task 2 — wires :func:`ModelManifest.signing_payload` to the
existing :class:`prsm.node.identity.NodeIdentity` Ed25519 keypair so that
every published model carries a publisher-verifiable manifest.

Verification path: callers can verify ``ModelManifest`` independently of
PRSM by holding only the publisher's public key. Anyone with the public
key can call :func:`verify_manifest` against a serialized manifest and
confirm:

1. The signature was produced by the holder of the matching private key
2. None of the manifest fields have been altered since signing
3. (Combined with the registry's per-shard sha256 verification on
   ``get()``) the actual shard bytes match what the publisher signed.

Same crypto trust-anchor as ``InferenceReceipt`` (Phase 3.x.1 Task 2):
``NodeIdentity.sign`` / ``verify`` — one keypair per node, used for
identity, receipts, and now manifests.
"""

from __future__ import annotations

import base64
import dataclasses
from typing import Optional

from prsm.compute.model_registry.models import ModelManifest
from prsm.node.identity import NodeIdentity, verify_signature


def sign_manifest(
    manifest: ModelManifest, identity: NodeIdentity
) -> ModelManifest:
    """Return a copy of ``manifest`` with ``publisher_signature`` populated.

    The original manifest is unchanged (frozen dataclass). The returned
    manifest has:
    - ``publisher_node_id`` set to ``identity.node_id``
    - ``publisher_signature`` set to the Ed25519 signature of
      ``signing_payload()``

    The signing payload excludes ``publisher_signature`` itself
    (Phase 3.x.2 Task 1 ``test_payload_excludes_signature``), so the
    two-step replace is safe: set ``publisher_node_id`` first, compute
    the canonical bytes, then set the signature.
    """
    # Step 1: stamp publisher_node_id BEFORE computing the signing
    # payload (publisher_node_id is part of the payload).
    intermediate = dataclasses.replace(
        manifest,
        publisher_node_id=identity.node_id,
    )

    # Step 2: sign the canonical bytes
    payload = intermediate.signing_payload()
    signature_b64 = identity.sign(payload)
    signature_bytes = base64.b64decode(signature_b64)

    # Step 3: produce the signed manifest
    return dataclasses.replace(
        intermediate,
        publisher_signature=signature_bytes,
    )


def verify_manifest(
    manifest: ModelManifest,
    public_key_b64: Optional[str] = None,
    identity: Optional[NodeIdentity] = None,
) -> bool:
    """Verify ``manifest.publisher_signature`` against ``signing_payload()``.

    Provide one of:
    - ``public_key_b64`` — base64-encoded Ed25519 public key matching
      the publisher's published key
    - ``identity`` — a :class:`NodeIdentity` instance whose public key
      matches the publisher

    Returns False (does NOT raise) if:
    - Signature is empty (manifest was never signed)
    - Neither verification credential supplied
    - Signature doesn't match the canonical signing payload
    - Public key doesn't correspond to the signer
    - Any underlying crypto step raises

    Tampering with any field in ``signing_payload()`` invalidates the
    signature — verified exhaustively by the Task 2 tampering tests.
    """
    if not manifest.publisher_signature:
        return False
    if public_key_b64 is None and identity is None:
        return False

    payload = manifest.signing_payload()
    signature_b64 = base64.b64encode(manifest.publisher_signature).decode()

    try:
        if identity is not None:
            return identity.verify(payload, signature_b64)
        # public_key_b64 is not None per the early-return above
        assert public_key_b64 is not None  # for type-checkers
        return verify_signature(public_key_b64, payload, signature_b64)
    except Exception:
        # NodeIdentity.verify / verify_signature should already return
        # False on malformed input, but a defensive catch here mirrors
        # the receipt verifier's "fail closed, never raise" contract.
        return False


def is_signed(manifest: ModelManifest) -> bool:
    """Return True iff ``manifest`` carries a non-empty publisher signature.

    NOT a cryptographic check — only confirms the field is populated.
    Use :func:`verify_manifest` to validate the signature.
    """
    return bool(manifest.publisher_signature) and bool(manifest.publisher_node_id)

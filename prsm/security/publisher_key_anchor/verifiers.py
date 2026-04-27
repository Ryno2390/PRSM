"""
Phase 3.x.3 Task 4 — anchor-backed verifier wrappers.

Three 3-line wrappers that compose ``PublisherKeyAnchorClient.lookup``
with the existing per-artifact verifier:

- ``verify_manifest_with_anchor(manifest, anchor)`` — wraps Phase 3.x.2
  ``verify_manifest``; resolves ``manifest.publisher_node_id`` via anchor.
- ``verify_entry_with_anchor(entry, anchor)`` — wraps Phase 3.x.4
  ``verify_entry``; resolves ``entry.node_id`` via anchor.
- ``verify_receipt_with_anchor(receipt, anchor)`` — wraps Phase 3.x.1
  ``verify_receipt``; resolves ``receipt.settler_node_id`` via anchor.

Each wrapper returns ``False`` (does NOT raise) on:
- Empty signature on the artifact (the artifact was never signed).
- Anchor lookup returns ``None`` (publisher not registered on-chain).
- Underlying signature verification fails (tampered artifact OR
  publisher's on-chain key has been rotated post-signing).

Anchor RPC failures DO propagate as ``AnchorRPCError`` — those are
transport-level operational issues, not "verification failed."
Callers that need to retry vs. reject can distinguish the two.

The wrappers are 3 lines each by design. Logic-wise they're a
trivial composition; their value is type-checked correctness +
the cross-artifact replay protection inherited from the underlying
verifiers (each artifact type uses a distinct domain separator in
its signing payload — see Phase 3.x.4 Task 2 cross-artifact tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prsm.compute.inference.models import InferenceReceipt
from prsm.compute.inference.receipt import verify_receipt
from prsm.compute.model_registry.models import ModelManifest
from prsm.compute.model_registry.signing import verify_manifest
from prsm.security.privacy_budget_persistence.models import PrivacyBudgetEntry
from prsm.security.privacy_budget_persistence.signing import verify_entry

if TYPE_CHECKING:
    from prsm.security.publisher_key_anchor.client import (
        PublisherKeyAnchorClient,
    )


def verify_manifest_with_anchor(
    manifest: ModelManifest,
    anchor: "PublisherKeyAnchorClient",
) -> bool:
    """Anchor-resolved verification of a Phase 3.x.2 ``ModelManifest``.

    Returns False on any of: unregistered publisher, signature mismatch,
    tampered manifest field. Anchor RPC failures propagate as
    ``AnchorRPCError``.

    .. warning::
        **Anchor key rotation invalidates pre-rotation signatures.**
        If the publisher's key is rotated via ``adminOverride`` (Phase
        3.x.3 emergency revocation path), every manifest signed under
        the OLD key will fail verification against the NEW anchored
        key. This is by design — that's the whole point of revocation
        — but downstream callers caching "verified manifest" results
        MUST invalidate those caches after a known anchor rotation.
        Use ``anchor.invalidate(node_id)`` to evict the lookup cache.
    """
    pubkey = anchor.lookup(manifest.publisher_node_id)
    if pubkey is None:
        return False
    return verify_manifest(manifest, public_key_b64=pubkey)


def verify_entry_with_anchor(
    entry: PrivacyBudgetEntry,
    anchor: "PublisherKeyAnchorClient",
) -> bool:
    """Anchor-resolved verification of a Phase 3.x.4 ``PrivacyBudgetEntry``.

    Returns False on any of: unregistered publisher, signature mismatch,
    tampered entry field, broken chain link (the entry's signing
    payload includes ``prev_entry_hash``). Anchor RPC failures
    propagate as ``AnchorRPCError``.

    .. warning::
        **Anchor key rotation invalidates pre-rotation signatures.**
        Same caveat as ``verify_manifest_with_anchor`` — see that
        wrapper's docstring for the cache-invalidation guidance.
    """
    # Defense in depth: PrivacyBudgetEntry.node_id is bare hex by
    # construction (no 0x prefix), but a future refactor might change
    # that. Strip explicitly so the wrapper stays robust against the
    # normalization that PublisherKeyAnchorClient.lookup also applies.
    node_id = entry.node_id
    if isinstance(node_id, str) and node_id.startswith("0x"):
        node_id = node_id[2:]
    pubkey = anchor.lookup(node_id)
    if pubkey is None:
        return False
    return verify_entry(entry, public_key_b64=pubkey)


def verify_receipt_with_anchor(
    receipt: InferenceReceipt,
    anchor: "PublisherKeyAnchorClient",
) -> bool:
    """Anchor-resolved verification of a Phase 3.x.1 ``InferenceReceipt``.

    Returns False on any of: unregistered settler, signature mismatch,
    tampered receipt field. Anchor RPC failures propagate as
    ``AnchorRPCError``.

    Note: an inference receipt's signing identity is the SETTLING node
    (``settler_node_id``), not the model publisher. Both must be
    independently anchored if cross-node verification is needed for
    both artifact types.

    .. warning::
        **Anchor key rotation invalidates pre-rotation signatures.**
        Same caveat as ``verify_manifest_with_anchor`` — see that
        wrapper's docstring for the cache-invalidation guidance.
    """
    pubkey = anchor.lookup(receipt.settler_node_id)
    if pubkey is None:
        return False
    return verify_receipt(receipt, public_key_b64=pubkey)

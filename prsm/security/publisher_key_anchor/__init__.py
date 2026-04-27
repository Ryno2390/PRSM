"""
PRSM publisher-key anchor — on-chain registry of `node_id → public_key`.

Phase 3.x.3 closes the cross-node trust-boundary caveat shared by
Phase 3.x.2 (model registry) and Phase 3.x.4 (privacy-budget journal).
Verifiers resolve a publisher's public key from the on-chain anchor
contract instead of trusting a local sidecar file.

Trust upgrade:
- Operator swaps `node.pubkey` sidecar locally → defeated by anchor lookup
- Adversary tries to register a victim's `node_id` → defeated by the
  contract's `sha256(pubkey)` derivation; adversary can only register
  pubkeys they own
- Compromised publisher key → multisig `adminOverride` emergency path

See ``docs/2026-04-27-phase3.x.3-publisher-key-anchor-design-plan.md``.
"""

from prsm.security.publisher_key_anchor.client import (
    PUBLISHER_KEY_ANCHOR_ABI,
    PublisherKeyAnchorClient,
)
from prsm.security.publisher_key_anchor.exceptions import (
    AnchorRPCError,
    PublisherAlreadyRegisteredError,
    PublisherKeyAnchorError,
    PublisherNotRegisteredError,
)

__all__ = [
    # Client (Task 3)
    "PublisherKeyAnchorClient",
    "PUBLISHER_KEY_ANCHOR_ABI",
    # Exceptions
    "PublisherKeyAnchorError",
    "PublisherAlreadyRegisteredError",
    "PublisherNotRegisteredError",
    "AnchorRPCError",
]

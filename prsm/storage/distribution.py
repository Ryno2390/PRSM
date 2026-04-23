"""
Distribution Manager — shard placement, ContentDescriptor management, and signing.

Handles the assignment of content shards to network peers with constraints
for ASN diversity, owner exclusion, and key-shard separation.  Also provides
Ed25519-based descriptor signing/verification and CRDT-style conflict resolution.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from prsm.storage.exceptions import PlacementError
from prsm.storage.models import ContentDescriptor, ContentHash, ReplicationPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ShardPlacement result
# ---------------------------------------------------------------------------

@dataclass
class ShardPlacement:
    """Result of shard placement computation."""

    shard_assignments: Dict[str, List[str]]  # shard_hash hex -> [node_ids]
    key_share_holders: List[str]
    contract_key_share_holders: List[str]
    degraded_constraints: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DistributionManager
# ---------------------------------------------------------------------------

class DistributionManager:
    """
    Orchestrates shard placement across the peer network and manages
    ContentDescriptor lifecycle (creation, signing, verification, conflict
    resolution).
    """

    MIN_NETWORK_NODES = 3

    def __init__(self, node_id, discovery, transport, blob_store, key_manager):
        self._node_id = node_id
        self._discovery = discovery
        self._transport = transport
        self._blob_store = blob_store
        self._key_manager = key_manager

    # ------------------------------------------------------------------
    # Shard placement
    # ------------------------------------------------------------------

    def _compute_shard_placement(
        self,
        shard_hashes: List[str],
        replication_factor: int,
        owner_node_id: str,
        peer_asn_map: Dict[str, str],
    ) -> ShardPlacement:
        """Compute placement with constraints.

        1. Owner exclusion (never relaxed)
        2. ASN diversity (relaxed first in degraded mode)
        3. Key-shard separation (relaxed only as last resort before owner escrow)

        Raises :class:`PlacementError` when fewer than *MIN_NETWORK_NODES*
        non-owner peers are available.
        """
        # Step 1: filter out the owner
        eligible = [nid for nid in peer_asn_map if nid != owner_node_id]

        # Step 2: check minimum
        if len(eligible) < self.MIN_NETWORK_NODES:
            raise PlacementError(
                reason=(
                    f"Only {len(eligible)} non-owner peer(s) available, "
                    f"need at least {self.MIN_NETWORK_NODES}"
                ),
                min_nodes_needed=self.MIN_NETWORK_NODES,
            )

        degraded: List[str] = []

        # Step 3: group by ASN
        asn_groups: Dict[str, List[str]] = defaultdict(list)
        for nid in eligible:
            asn_groups[peer_asn_map[nid]].append(nid)

        # Step 4: assign shards
        shard_assignments: Dict[str, List[str]] = {}
        all_shard_nodes: Set[str] = set()

        # Track usage for round-robin within ASN groups
        asn_cursor: Dict[str, int] = {asn: 0 for asn in asn_groups}
        asn_list = list(asn_groups.keys())

        for shard_hash in shard_hashes:
            assigned: List[str] = []
            used_asns: Set[str] = set()

            # First pass: pick from distinct ASN groups
            # Rotate starting ASN per shard for balance
            for asn in asn_list:
                if len(assigned) >= replication_factor:
                    break
                if asn in used_asns:
                    continue
                group = asn_groups[asn]
                cursor = asn_cursor[asn]
                node = group[cursor % len(group)]
                assigned.append(node)
                used_asns.add(asn)
                asn_cursor[asn] = cursor + 1

            # Second pass: if not enough distinct ASNs, fill from any node
            if len(assigned) < replication_factor:
                if "asn_relaxed" not in degraded:
                    degraded.append("asn_relaxed")
                    logger.warning(
                        "ASN diversity constraint relaxed — not enough distinct ASNs"
                    )
                for nid in eligible:
                    if len(assigned) >= replication_factor:
                        break
                    if nid not in assigned:
                        assigned.append(nid)

            shard_assignments[shard_hash] = assigned
            all_shard_nodes.update(assigned)

            # Rotate ASN list for next shard to spread load
            if asn_list:
                asn_list = asn_list[1:] + asn_list[:1]

        # Step 5: key share holders — prefer nodes that hold zero shards
        non_shard_nodes = [nid for nid in eligible if nid not in all_shard_nodes]
        if non_shard_nodes:
            key_share_holders = non_shard_nodes
        else:
            # Relax key-shard separation
            degraded.append("key_shard_separation_relaxed")
            logger.warning(
                "Key-shard separation constraint relaxed — not enough non-shard nodes"
            )
            key_share_holders = eligible[:]

        # Step 6: contract key share holders — same pool as key share holders
        contract_key_share_holders = key_share_holders[:]

        return ShardPlacement(
            shard_assignments=shard_assignments,
            key_share_holders=key_share_holders,
            contract_key_share_holders=contract_key_share_holders,
            degraded_constraints=degraded,
        )

    # ------------------------------------------------------------------
    # Descriptor creation
    # ------------------------------------------------------------------

    def _create_descriptor_stub(
        self,
        content_hash: ContentHash,
        owner_node_id: str,
        visibility: str,
        replication_policy: ReplicationPolicy,
        contract_pubkey: bytes,
    ) -> ContentDescriptor:
        """Create an unsigned descriptor with epoch=1, version=1."""
        now = time.time()
        return ContentDescriptor(
            content_hash=content_hash,
            manifest_holders=[],
            key_share_holders=[],
            contract_key_share_holders=[],
            shard_map={},
            replication_policy=replication_policy,
            visibility=visibility,
            epoch=1,
            version=1,
            owner_node_id=owner_node_id,
            contract_pubkey=contract_pubkey,
            signature=b"",
            signer_type="",
            created_at=now,
            updated_at=now,
        )

    # ------------------------------------------------------------------
    # Descriptor signing helpers
    # ------------------------------------------------------------------

    def _descriptor_signing_data(self, descriptor: ContentDescriptor) -> bytes:
        """Deterministic JSON bytes for signing.

        Uses sorted keys and sorted lists to guarantee consistent ordering.
        """
        rp = descriptor.replication_policy
        data = {
            "content_hash": descriptor.content_hash.hex(),
            "contract_key_share_holders": sorted(descriptor.contract_key_share_holders),
            "contract_pubkey": descriptor.contract_pubkey.hex(),
            "epoch": descriptor.epoch,
            "key_share_holders": sorted(descriptor.key_share_holders),
            "manifest_holders": sorted(descriptor.manifest_holders),
            "owner_node_id": descriptor.owner_node_id,
            "replication_policy": {
                "degraded_constraints": sorted(rp.degraded_constraints),
                "key_shard_separation": rp.key_shard_separation,
                "min_asn_diversity": rp.min_asn_diversity,
                "owner_excluded": rp.owner_excluded,
                "replication_factor": rp.replication_factor,
            },
            "shard_map": {
                k: sorted(v) for k, v in sorted(descriptor.shard_map.items())
            },
            "version": descriptor.version,
            "visibility": descriptor.visibility,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

    def _sign_descriptor(
        self,
        descriptor: ContentDescriptor,
        private_key: Ed25519PrivateKey,
        signer_type: str,
    ) -> ContentDescriptor:
        """Sign *descriptor* with an Ed25519 private key."""
        signing_data = self._descriptor_signing_data(descriptor)
        signature = private_key.sign(signing_data)
        descriptor.signature = signature
        descriptor.signer_type = signer_type
        return descriptor

    def _verify_descriptor_signature(
        self,
        descriptor: ContentDescriptor,
        public_key: Ed25519PublicKey,
    ) -> bool:
        """Verify descriptor signature; returns False on failure."""
        signing_data = self._descriptor_signing_data(descriptor)
        try:
            public_key.verify(descriptor.signature, signing_data)
            return True
        except InvalidSignature:
            return False

    # ------------------------------------------------------------------
    # Contract update validation
    # ------------------------------------------------------------------

    def _validate_contract_update(
        self,
        base: ContentDescriptor,
        updated: ContentDescriptor,
    ) -> bool:
        """Validate that a contract-signed update only touches allowed fields.

        Allowed to change: shard_map, manifest_holders, key_share_holders,
            contract_key_share_holders, updated_at, version
        Must NOT change: owner_node_id, epoch, visibility,
            replication_policy (any field), contract_pubkey
        """
        if base.owner_node_id != updated.owner_node_id:
            return False
        if base.epoch != updated.epoch:
            return False
        if base.visibility != updated.visibility:
            return False
        bp = base.replication_policy
        up = updated.replication_policy
        if (
            bp.replication_factor != up.replication_factor
            or bp.min_asn_diversity != up.min_asn_diversity
            or bp.owner_excluded != up.owner_excluded
            or bp.key_shard_separation != up.key_shard_separation
        ):
            return False
        if base.contract_pubkey != updated.contract_pubkey:
            return False
        return True

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def _resolve_conflict(
        self,
        a: ContentDescriptor,
        b: ContentDescriptor,
    ) -> ContentDescriptor:
        """Highest ``(epoch, version)`` wins."""
        if (a.epoch, a.version) >= (b.epoch, b.version):
            return a
        return b

"""Sprint 246 — resolver: contributing_shards → per-shard
royalty targets.

Chips at the on-chain content-access royalty leg foundation
flagged in Vision §13: settlement-side callers need a clean way
to go from a swarm query's ``contributing_shards: List[str]``
to the (content_hash, creator_eth_address, royalty_rate)
tuples required by ``RoyaltyDistributorClient.distribute_
royalty()``.

This module exposes ``resolve_content_royalty_targets()`` —
pure function, no side effects, no on-chain dispatch. Each
caller (forge settlement, inference settlement, batch
reconciliation) shares the same skip/keep semantics:
  - Missing record → skip (creator can't be identified)
  - Missing/empty creator_eth_address → skip (can't dispatch
    on-chain without a destination; v1 pre-sprint-243 uploads
    all land here)
  - Lookup exception → skip (defensive against transient
    ContentIndex failures)
  - Skipped count surfaced for operator telemetry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentRoyaltyTarget:
    """A single content-access royalty target — one per
    contributing shard that has the metadata required for
    on-chain dispatch."""

    shard_cid: str
    content_hash: str
    creator_eth_address: str
    royalty_rate: float


def resolve_content_royalty_targets(
    *,
    contributing_shards: List[str],
    content_index: Any,
) -> Tuple[List[ContentRoyaltyTarget], int]:
    """Look up each shard's record and build royalty targets.

    Returns ``(targets, skipped_count)``. A shard is skipped if
    the ContentIndex lookup misses, raises, or the record lacks
    a ``creator_eth_address``.
    """
    targets: List[ContentRoyaltyTarget] = []
    skipped = 0
    for cid in contributing_shards:
        try:
            record = content_index.lookup(cid)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "content_royalty_resolver: lookup raised for %s "
                "(%s); skipping", cid[:12], exc,
            )
            skipped += 1
            continue
        if record is None:
            skipped += 1
            continue
        eth = getattr(record, "creator_eth_address", None)
        if not eth:
            skipped += 1
            continue
        targets.append(ContentRoyaltyTarget(
            shard_cid=cid,
            content_hash=getattr(record, "content_hash", ""),
            creator_eth_address=eth,
            royalty_rate=float(
                getattr(record, "royalty_rate", 0.0)
            ),
        ))
    return targets, skipped

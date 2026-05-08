"""Bridges existing marketplace primitives to the QueryOrchestrator's
`candidate_pool_provider` callable contract.

The QueryOrchestrator class accepts
`candidate_pool_provider: Callable[[], tuple[StakedNode, ...]]` —
each query's selector pulls a fresh T2+ pool snapshot via this
callable. This module supplies an implementation backed by the
existing `MarketplaceDirectory` (gossip-aggregated active listings)
+ `ReputationTracker` (per-provider score + slash history).

Architectural choice (v1): derive `stake_amount_ftns` from the
listing's `stake_tier` label rather than running an on-chain
`stake_of(provider)` per listing every selection. The latter is
N RPC calls per pool query — too expensive for production paths.
The tier label is already verified at marketplace-listing-ingestion
time (signature-bound + tier-eligibility-checked at the slasher).
Foundation council can ratify different tier→stake mappings via
the constructor override.

v1 limitations (separately followed-on):
  - Real per-listing stake_of read for fine-grained collusion
    weighting deferred — see threat-model A1 §"Open governance
    questions" item 4. v1 derives stake from tier label only.

Per `docs/2026-05-08-query-orchestrator-wiring-readiness.md`
"candidate_pool_provider" — Blocker 4 supporting piece for B7.
"""
from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from prsm.compute.query_orchestrator.aggregator_selector import StakedNode
from prsm.marketplace.reputation import ReputationTracker

logger = logging.getLogger(__name__)


# Default tier→stake mapping. Numbers are nominal FTNS amounts —
# they bound the selector's stake-weighted draw. Foundation council
# ratification target (per threat-model A1 governance question 4 +
# the Phase 7 staking-tier policy). Override via constructor kwarg.
DEFAULT_STAKE_PER_TIER: Mapping[str, int] = {
    "T2": 1_000,
    "T3": 5_000,
    "T4": 25_000,
}


# Tiers eligible for aggregator selection per Vision §6 (T2+).
ELIGIBLE_TIERS = frozenset({"T2", "T3", "T4"})


@dataclass
class MarketplaceCandidatePoolProvider:
    """Callable adapter: marketplace directory + reputation →
    `tuple[StakedNode, ...]` snapshot.

    Production wiring: in `node.py`, this is constructed once at
    startup with the node's `MarketplaceDirectory` instance and the
    `ReputationTracker` from the marketplace orchestrator. The
    QueryOrchestrator pulls a fresh snapshot per query via __call__.
    """

    directory: Any  # MarketplaceDirectory — duck-typed for testability
    reputation: ReputationTracker
    stake_amount_per_tier: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_STAKE_PER_TIER),
    )

    def __call__(self) -> Tuple[StakedNode, ...]:
        """Return current T2+ pool snapshot.

        Filters silently:
          - Tier not in `ELIGIBLE_TIERS` (T1 and unknown tiers dropped)
          - Malformed pubkey_b64 (logged at debug; listing dropped)

        Empty directory or fully-filtered result → empty tuple.
        Caller (orchestrator) handles the empty case via
        `InsufficientCandidatesError` from `select_aggregator`.
        """
        listings = self.directory.list_active_providers()
        out: list[StakedNode] = []
        for listing in listings:
            if listing.stake_tier not in ELIGIBLE_TIERS:
                continue
            pubkey_hash = self._compute_pubkey_hash(
                listing.provider_pubkey_b64,
                provider_id=listing.provider_id,
            )
            if pubkey_hash is None:
                continue
            stake_amount = self.stake_amount_per_tier.get(
                listing.stake_tier, 0,
            )
            out.append(StakedNode(
                node_id=listing.provider_id,
                pubkey_hash=pubkey_hash,
                stake_amount_ftns=stake_amount,
                tier=listing.stake_tier,
                # ProviderListing.tee_capable is the gossip-advertised
                # marker. Tier C queries (requires_tee=True at the
                # selector) filter on this; the listing's signature
                # binds it so a relay can't forge tee_capable=True.
                has_tee=bool(listing.tee_capable),
                reputation_score=self.reputation.score_for(
                    listing.provider_id,
                ),
            ))
        return tuple(out)

    @staticmethod
    def _compute_pubkey_hash(
        pubkey_b64: str, *, provider_id: str,
    ) -> bytes | None:
        """SHA-256 over the decoded pubkey bytes — matches the
        threat-model A8 `pubkey_hash` identity binding used by
        `select_aggregator`. Returns None on malformed base64
        (caller drops the listing)."""
        try:
            pubkey_bytes = base64.b64decode(pubkey_b64, validate=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "candidate-pool: dropped listing for provider_id=%r — "
                "malformed pubkey_b64: %s",
                provider_id, exc,
            )
            return None
        return hashlib.sha256(pubkey_bytes).digest()

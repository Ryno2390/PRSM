"""Phase 3 Task 4: EligibilityFilter.

Pure function: takes a list of ProviderListings + a DispatchPolicy,
returns the policy-compliant subset. No network, no async, no I/O.
Short-circuits on first rejection for each listing.

Filter order (docs/2026-04-20-phase3-marketplace-design.md §3.3):
  1. TTL not expired
  2. max_price_per_shard_ftns ceiling
  3. min_price_per_shard_ftns floor (anti-loss-leader)
  4. require_tee → tee_capable iff True
  5. min_stake_tier ordinal comparison
  6. required_dtype ∈ supported_dtypes
  7. min_capacity_shards_per_sec
  8. min_reputation_score (consults ReputationTracker if wired)
"""
from __future__ import annotations

import time
from typing import List, Optional

from prsm.marketplace.listing import ProviderListing
from prsm.marketplace.policy import DispatchPolicy


class EligibilityFilter:
    """Filters marketplace listings by DispatchPolicy.

    Reputation is consulted via an injected ReputationTracker (Task 6).
    For Tasks 1-5, the tracker can be None — min_reputation_score is
    skipped when no tracker is wired.
    """

    _TIER_ORDER = {
        "open": 0,
        "standard": 1,
        "premium": 2,
        "critical": 3,
    }

    def __init__(self, reputation_tracker=None):
        self._reputation = reputation_tracker

    def filter(
        self,
        listings: List[ProviderListing],
        policy: DispatchPolicy,
        at_unix: Optional[int] = None,
    ) -> List[ProviderListing]:
        """Return listings that pass every policy check.

        Each listing is evaluated independently — the output preserves
        the input ordering so callers (e.g., TopologyRandomizer in
        Task 7) can apply their own randomization downstream."""
        now = at_unix if at_unix is not None else int(time.time())
        out: List[ProviderListing] = []
        min_tier_ord = self._TIER_ORDER.get(policy.min_stake_tier, -1)

        for listing in listings:
            if listing.is_expired(now):
                continue
            if listing.price_per_shard_ftns > policy.max_price_per_shard_ftns:
                continue
            if listing.price_per_shard_ftns < policy.min_price_per_shard_ftns:
                continue
            if policy.require_tee and not listing.tee_capable:
                continue
            listing_tier_ord = self._TIER_ORDER.get(listing.stake_tier, -1)
            if listing_tier_ord < min_tier_ord:
                continue
            if policy.required_dtype not in listing.supported_dtypes:
                continue
            if listing.capacity_shards_per_sec < policy.min_capacity_shards_per_sec:
                continue
            if (
                self._reputation is not None
                and policy.min_reputation_score > 0.0
            ):
                score = self._reputation.score_for(listing.provider_id)
                if score < policy.min_reputation_score:
                    continue
            out.append(listing)

        return out

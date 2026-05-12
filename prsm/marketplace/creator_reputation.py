"""Sprint 287 — CreatorReputationTracker.

Vision §14 "Data quality and Sybil resistance" mitigation item
(1). Creator-side complement to the Phase-3-Task-6 provider-
side ReputationTracker: that one scores compute providers
(success rate + latency + slash history); this one scores
content uploaders on the data-marketplace side (access
frequency + distinct-purchaser breadth + repeat-purchase
rate).

The spam pattern this defends against: a creator uploads
1000 pieces of content, each accessed once by a different
account. Upload earns FTNS but engagement is zero — without
the repeat signal, naive frequency-based reputation would
reward the spam. The repeat-purchase signal is the
discriminator: real value generates return visits.

Score formula (v1, tunable):

  cold-start: total_accesses < MIN_SAMPLES_FOR_SCORE
              → NEUTRAL_SCORE (0.5)

  otherwise:  REACH_WEIGHT * reach_score
            + REPEAT_WEIGHT * repeat_score

  where:
    reach_score  = clip(0..1) of log10(1 + distinct) / 2.0
                   (100 distinct purchasers → 1.0)
    repeat_score = repeat_purchaser_count / distinct
                   (fraction of purchasers who came back for
                    at least one more piece of this creator)

Tier classification, search filtering, and staking gates
ship in follow-on sprints (288-290). This module is the
data substrate they all consume.

Per-node, in-memory. No on-chain anchoring (matches the
provider-side tracker contract — advisory info for the
local node's marketplace UX, not a protocol commitment).
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Tunable score weights ────────────────────────────────

REACH_WEIGHT = 0.6
REPEAT_WEIGHT = 0.4
MIN_SAMPLES_FOR_SCORE = 10
NEUTRAL_SCORE = 0.5

# ── Sprint 288 — tier classification ─────────────────────
# String labels (not Enum) for trivial JSON serialization.
TIER_NEW = "new"          # cold-start; no signal yet
TIER_LOW = "low"          # measured low
TIER_MEDIUM = "medium"
TIER_HIGH = "high"

TIER_THRESHOLD_MEDIUM = 0.55
TIER_THRESHOLD_HIGH = 0.75


def tier_for_score(
    score: float,
    total_accesses: int,
    *,
    threshold_medium: float = TIER_THRESHOLD_MEDIUM,
    threshold_high: float = TIER_THRESHOLD_HIGH,
) -> str:
    """Pure function mapping (score, total_accesses) → tier.

    Cold-start short-circuits to TIER_NEW regardless of score
    — distinct from TIER_LOW which means we have signal AND
    the signal is poor. Downstream sprints (search filtering)
    treat NEW and LOW differently.
    """
    if total_accesses < MIN_SAMPLES_FOR_SCORE:
        return TIER_NEW
    if score >= threshold_high:
        return TIER_HIGH
    if score >= threshold_medium:
        return TIER_MEDIUM
    return TIER_LOW


@dataclass
class CreatorReputationEntry:
    """Aggregated reputation state for one creator.

    purchaser_counts is the load-bearing structure: maps each
    distinct purchaser_id to the number of times they've
    accessed this creator's content. The repeat-purchase
    signal derives from values ≥ 2.
    """
    creator_id: str
    total_accesses: int = 0
    purchaser_counts: Dict[str, int] = field(default_factory=dict)
    first_seen_unix: int = 0
    last_seen_unix: int = 0

    @property
    def distinct_purchasers(self) -> int:
        return len(self.purchaser_counts)

    @property
    def repeat_purchaser_count(self) -> int:
        return sum(
            1 for n in self.purchaser_counts.values() if n >= 2
        )

    def to_dict(self) -> Dict[str, Any]:
        """Public surface drops purchaser_counts (privacy +
        payload size); only aggregates surface."""
        return {
            "creator_id": self.creator_id,
            "total_accesses": self.total_accesses,
            "distinct_purchasers": self.distinct_purchasers,
            "repeat_purchaser_count":
                self.repeat_purchaser_count,
            "first_seen_unix": self.first_seen_unix,
            "last_seen_unix": self.last_seen_unix,
        }


_DEFAULT_MAX_PURCHASERS_PER_CREATOR = 1000


class CreatorReputationTracker:
    """In-memory creator reputation index."""

    def __init__(
        self,
        max_purchasers_per_creator: int = (
            _DEFAULT_MAX_PURCHASERS_PER_CREATOR
        ),
    ) -> None:
        if (
            not isinstance(max_purchasers_per_creator, int)
            or max_purchasers_per_creator <= 0
        ):
            raise ValueError(
                f"max_purchasers_per_creator must be a positive "
                f"integer, got {max_purchasers_per_creator!r}"
            )
        self._max_purchasers_per_creator = (
            max_purchasers_per_creator
        )
        self._creators: Dict[str, CreatorReputationEntry] = {}
        # Per-creator OrderedDict of purchaser-id → access count.
        # OrderedDict preserves insertion order; FIFO eviction
        # drops the oldest first-seen purchaser when the bound
        # is hit. We store this OUT-OF-BAND from the dataclass
        # because dataclass dict[] doesn't preserve order
        # ordering semantics we need here.
        self._purchaser_history: Dict[str, OrderedDict] = {}

    # ── Recording ────────────────────────────────────────

    def record_access(
        self,
        *,
        creator_id: str,
        purchaser_id: str,
        content_id: str,
        timestamp: Optional[float] = None,
    ) -> None:
        if not creator_id:
            raise ValueError("creator_id must be non-empty")
        if not purchaser_id:
            raise ValueError("purchaser_id must be non-empty")
        if not content_id:
            raise ValueError("content_id must be non-empty")

        now_int = int(
            timestamp if timestamp is not None else time.time()
        )
        entry = self._creators.get(creator_id)
        if entry is None:
            entry = CreatorReputationEntry(
                creator_id=creator_id,
                first_seen_unix=now_int,
                last_seen_unix=now_int,
            )
            self._creators[creator_id] = entry
            self._purchaser_history[creator_id] = OrderedDict()
        else:
            entry.last_seen_unix = now_int

        entry.total_accesses += 1

        # Bounded purchaser tracking with FIFO eviction.
        history = self._purchaser_history[creator_id]
        if purchaser_id in history:
            # Existing purchaser — bump count + move to end
            # (MRU semantics; eviction targets least-recently
            # seen).
            history[purchaser_id] = history[purchaser_id] + 1
            history.move_to_end(purchaser_id)
        else:
            # New purchaser. Evict the oldest if at capacity.
            if (
                len(history)
                >= self._max_purchasers_per_creator
            ):
                history.popitem(last=False)
            history[purchaser_id] = 1
        # Keep the dataclass's purchaser_counts mirror in sync
        # (dict copy; cheap because bounded).
        entry.purchaser_counts = dict(history)

    # ── Queries ──────────────────────────────────────────

    def get_entry(
        self, creator_id: str,
    ) -> Optional[CreatorReputationEntry]:
        return self._creators.get(creator_id)

    def access_count(self, creator_id: str) -> int:
        e = self._creators.get(creator_id)
        return e.total_accesses if e else 0

    def distinct_purchasers(self, creator_id: str) -> int:
        e = self._creators.get(creator_id)
        return e.distinct_purchasers if e else 0

    def repeat_purchaser_count(self, creator_id: str) -> int:
        e = self._creators.get(creator_id)
        return e.repeat_purchaser_count if e else 0

    def repeat_purchase_rate(self, creator_id: str) -> float:
        e = self._creators.get(creator_id)
        if e is None or e.distinct_purchasers == 0:
            return 0.0
        return e.repeat_purchaser_count / e.distinct_purchasers

    def known_creators(self) -> List[str]:
        return list(self._creators.keys())

    # ── Score ────────────────────────────────────────────

    def tier_for(self, creator_id: str) -> str:
        """Return the discrete tier label for this creator
        (sprint 288). Cold-start / unknown creators return
        TIER_NEW; otherwise tier follows the score-threshold
        bands."""
        e = self._creators.get(creator_id)
        total = e.total_accesses if e else 0
        score = self.score_for(creator_id)
        return tier_for_score(score=score, total_accesses=total)

    def score_for(self, creator_id: str) -> float:
        """Return a [0.0, 1.0] reputation score for the
        creator. NEUTRAL_SCORE for unknown creators AND for
        known creators with < MIN_SAMPLES_FOR_SCORE total
        accesses (cold-start uncertainty)."""
        e = self._creators.get(creator_id)
        if e is None or e.total_accesses < MIN_SAMPLES_FOR_SCORE:
            return NEUTRAL_SCORE

        distinct = e.distinct_purchasers
        if distinct == 0:
            return NEUTRAL_SCORE

        # Reach: log10(1+distinct) / 2.0 saturates at 100
        # distinct purchasers → 1.0. Bounded above.
        reach_score = min(
            1.0, math.log10(1 + distinct) / 2.0,
        )
        # Repeat: fraction of purchasers who came back.
        repeat_score = e.repeat_purchaser_count / distinct

        score = (
            REACH_WEIGHT * reach_score
            + REPEAT_WEIGHT * repeat_score
        )
        # Defensive clip — handles any future weight changes
        # that don't sum to 1.0.
        return max(0.0, min(1.0, score))

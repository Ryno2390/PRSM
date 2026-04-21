"""Phase 3 Task 6: ReputationTracker.

Lightweight per-provider reputation based on observed dispatch outcomes.
Rolling-window counters (last 1000 events), plus p50/p95 latency from
the last 1000 successful dispatches.

Design contract (docs/2026-04-20-phase3-marketplace-design.md §6):
  - Score = successful / (successful + failed). Preempted events do
    NOT contribute to the denominator — they're honest-work failures
    per Phase 2.1 Line A, not quality signals.
  - New providers (no history) → neutral 0.5. Allows bootstrapping.
  - Providers with < MIN_SAMPLES_FOR_SCORE total → also neutral 0.5.
    Treats "too little data" as uncertainty, not as bad.
  - Tracker is per-node (not federated). Phase 6 adds gossip-based
    reputation sharing.
  - No on-chain anchoring. No signatures. This is advisory info for
    the local requester's EligibilityFilter (Task 4) — it's not a
    protocol commitment anyone else verifies.

Phase 7 Task 6: slashes are a distinct event class.
  - record_slash is fed from the Slashed event on the StakeBond contract
    whenever a successful challenge invalidates a receipt. A slash means
    the provider was caught doing something provably malicious
    (double-spend or invalid signature) and forfeited stake on-chain.
  - Slashes count into score_for via SLASH_WEIGHT so one slash
    effectively pins a previously-perfect provider to ~0.0. Callers that
    want hard exclusion (e.g., EligibilityFilter at critical tier) can
    use has_been_slashed() as a binary filter.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass(frozen=True)
class SlashEvent:
    """Phase 7 Task 6: one on-chain slash against a provider.

    Captured from the StakeBond contract's Slashed event via
    ReputationTracker.record_slash. Immutable — a slash is historical
    record, not mutable state.
    """
    batch_id: str
    slash_amount_wei: int
    reason: str   # "DOUBLE_SPEND" or "INVALID_SIGNATURE"
    recorded_unix: int
    tx_hash: Optional[str] = None  # on-chain tx for audit; None when replayed from logs


@dataclass
class ProviderReputation:
    """Rolling counters + latency samples for one provider.

    All deques are bounded to 1000 elements. When full, the oldest
    element drops as the newest is appended — natural rolling window.
    """
    provider_id: str
    successful_dispatches: Deque[int] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    failed_dispatches: Deque[int] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    preempted_dispatches: Deque[int] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    latency_samples_ms: Deque[float] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    # Phase 7 Task 6: slash history. Kept as a separate stream from
    # failed_dispatches so audit tools can distinguish between ordinary
    # operational failures and on-chain misbehavior.
    slash_events: Deque[SlashEvent] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    first_seen_unix: int = 0
    last_seen_unix: int = 0


class ReputationTracker:
    """In-memory per-provider reputation tracking.

    Consumed by EligibilityFilter (Task 4) via `score_for(provider_id)`
    and by higher-level monitoring via `latency_p50()` / `latency_p95()`.
    """

    MIN_SAMPLES_FOR_SCORE = 10
    NEUTRAL_SCORE = 0.5

    # Phase 7 Task 6: one slash ≈ SLASH_WEIGHT failures in the score
    # denominator. At 100, a single slash on a provider with 20
    # successes drops score to 20/(20+100) = 0.167; a provider with
    # 100 successes drops to 100/(100+100) = 0.5 — still neutral-ish,
    # which is why callers that want hard exclusion should also check
    # has_been_slashed() rather than relying on score alone.
    SLASH_WEIGHT = 100

    def __init__(self):
        self._reputations: Dict[str, ProviderReputation] = {}

    def _touch(self, provider_id: str) -> ProviderReputation:
        """Get or create the reputation entry; update last_seen_unix."""
        rep = self._reputations.get(provider_id)
        now = int(time.time())
        if rep is None:
            rep = ProviderReputation(
                provider_id=provider_id,
                first_seen_unix=now,
                last_seen_unix=now,
            )
            self._reputations[provider_id] = rep
        else:
            rep.last_seen_unix = now
        return rep

    def record_success(self, provider_id: str, latency_ms: float) -> None:
        """Record a successful dispatch. Latency contributes to p50/p95."""
        rep = self._touch(provider_id)
        rep.successful_dispatches.append(int(time.time()))
        rep.latency_samples_ms.append(float(latency_ms))

    def record_failure(self, provider_id: str) -> None:
        """Record a failed dispatch — verification failure, timeout after
        retries, or any other malicious/unrecoverable outcome. Lowers
        the score."""
        rep = self._touch(provider_id)
        rep.failed_dispatches.append(int(time.time()))

    def record_preemption(self, provider_id: str) -> None:
        """Record a preemption event. Does NOT affect the score — it's
        honest-work failure per Phase 2.1 Line A. Tracked for
        observability only (e.g., operator dashboards, T3-tier filtering
        by preemption rate)."""
        rep = self._touch(provider_id)
        rep.preempted_dispatches.append(int(time.time()))

    def record_slash(
        self,
        provider_id: str,
        batch_id: str,
        slash_amount_wei: int,
        reason: str,
        tx_hash: Optional[str] = None,
    ) -> None:
        """Phase 7 Task 6: record an on-chain slash.

        Fed from the StakeBond.Slashed event (or replayed from logs). A
        slash counts as SLASH_WEIGHT failures in score_for and surfaces
        independently via has_been_slashed / slashed_count / get_slash_events.

        batch_id: the BatchSettlementRegistry batch the slash was tied
            to, hex-encoded for human-readable audit. Not decoded to
            bytes here — Reputation is an observability layer, not a
            chain verifier.
        slash_amount_wei: FTNS base units slashed (post-split pre-claim).
        reason: human-readable reason code — "DOUBLE_SPEND" or
            "INVALID_SIGNATURE". Unknown values are stored verbatim so
            a future reason code lands in the audit trail even before
            this module is updated.
        tx_hash: optional on-chain tx hash for cross-referencing with
            block explorers or batch settlement logs.
        """
        if not provider_id:
            raise ValueError("provider_id required")
        if not batch_id:
            raise ValueError("batch_id required")
        if slash_amount_wei < 0:
            raise ValueError(
                f"slash_amount_wei must be non-negative (got {slash_amount_wei})"
            )
        if not reason:
            raise ValueError("reason required")
        rep = self._touch(provider_id)
        rep.slash_events.append(SlashEvent(
            batch_id=batch_id,
            slash_amount_wei=int(slash_amount_wei),
            reason=reason,
            recorded_unix=int(time.time()),
            tx_hash=tx_hash,
        ))

    def has_been_slashed(self, provider_id: str) -> bool:
        """Return True if this provider has any slash on record.

        Use as a hard-exclusion predicate when you don't want any slash
        history near a high-value job, regardless of total sample count.
        Cheaper than score_for — no arithmetic, just a deque emptiness
        check."""
        rep = self._reputations.get(provider_id)
        return bool(rep and rep.slash_events)

    def slashed_count(self, provider_id: str) -> int:
        """Number of slashes currently in the rolling window for this
        provider. 0 for unknown providers."""
        rep = self._reputations.get(provider_id)
        if rep is None:
            return 0
        return len(rep.slash_events)

    def get_slash_events(self, provider_id: str) -> List[SlashEvent]:
        """Return the provider's slash history (oldest first). Empty
        list for unknown providers."""
        rep = self._reputations.get(provider_id)
        if rep is None:
            return []
        return list(rep.slash_events)

    def score_for(self, provider_id: str) -> float:
        """Return the provider's reputation score in [0.0, 1.0].

        Special cases:
          - Unknown provider → NEUTRAL_SCORE (no cold-start dead zone).
          - Total weighted sample count < MIN_SAMPLES_FOR_SCORE →
            NEUTRAL_SCORE (too little data to distinguish).

        Phase 7 Task 6: slashes contribute to the denominator at
        SLASH_WEIGHT each. One slash alone can push a provider into
        the denominator even if they have no failure or success history
        yet, which is intentional — a freshly-caught cheater should not
        sit behind the MIN_SAMPLES_FOR_SCORE cold-start shield.
        """
        rep = self._reputations.get(provider_id)
        if rep is None:
            return self.NEUTRAL_SCORE
        successes = len(rep.successful_dispatches)
        failures = len(rep.failed_dispatches)
        slashes = len(rep.slash_events)
        weighted_failures = failures + self.SLASH_WEIGHT * slashes
        total = successes + weighted_failures
        if total < self.MIN_SAMPLES_FOR_SCORE:
            return self.NEUTRAL_SCORE
        return successes / total

    def latency_p50(self, provider_id: str) -> Optional[float]:
        """Median of the last 1000 success latencies. None if no samples."""
        return self._percentile(provider_id, 0.50)

    def latency_p95(self, provider_id: str) -> Optional[float]:
        """95th percentile of the last 1000 success latencies."""
        return self._percentile(provider_id, 0.95)

    def _percentile(self, provider_id: str, q: float) -> Optional[float]:
        rep = self._reputations.get(provider_id)
        if rep is None or not rep.latency_samples_ms:
            return None
        samples = sorted(rep.latency_samples_ms)
        # Linear interpolation on the sorted array.
        pos = q * (len(samples) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return samples[lo]
        fraction = pos - lo
        return samples[lo] + (samples[hi] - samples[lo]) * fraction

    def get_reputation(
        self, provider_id: str,
    ) -> Optional[ProviderReputation]:
        """Return the raw ProviderReputation record or None if unknown."""
        return self._reputations.get(provider_id)

    def known_providers(self) -> List[str]:
        """List every provider_id the tracker has observed."""
        return list(self._reputations.keys())

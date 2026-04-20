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
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


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
    first_seen_unix: int = 0
    last_seen_unix: int = 0


class ReputationTracker:
    """In-memory per-provider reputation tracking.

    Consumed by EligibilityFilter (Task 4) via `score_for(provider_id)`
    and by higher-level monitoring via `latency_p50()` / `latency_p95()`.
    """

    MIN_SAMPLES_FOR_SCORE = 10
    NEUTRAL_SCORE = 0.5

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

    def score_for(self, provider_id: str) -> float:
        """Return the provider's reputation score in [0.0, 1.0].

        Special cases:
          - Unknown provider → NEUTRAL_SCORE (no cold-start dead zone).
          - Total (success + failure) < MIN_SAMPLES_FOR_SCORE →
            NEUTRAL_SCORE (too little data to distinguish).
        """
        rep = self._reputations.get(provider_id)
        if rep is None:
            return self.NEUTRAL_SCORE
        successes = len(rep.successful_dispatches)
        failures = len(rep.failed_dispatches)
        total = successes + failures
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

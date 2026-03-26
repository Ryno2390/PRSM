"""
KL Divergence Filter — Surprise Threshold Gate
===============================================

The KL filter is the first gate in the BSC pipeline.  It receives a
SurpriseScore from the predictor and decides, based on a configurable
epsilon threshold, whether the chunk is surprising enough to be forwarded
to semantic de-duplication.

Mathematical note
-----------------
The ``SurpriseScore.score`` produced by the predictor is already a
normalised proxy for KL divergence (the cross-entropy of the one-hot
actual distribution relative to the model's predicted distribution,
averaged over chunk tokens, then sigmoid-normalised against the adaptive
baseline).  This module applies the threshold decision and provides
diagnostic context for why a chunk was accepted or rejected.

Adaptive threshold
------------------
A static epsilon can miss bursts of low-surprise information that
collectively represent a significant context shift.  The optional
``AdaptiveKLFilter`` subclass tracks a short sliding window of recent
scores and raises epsilon temporarily when many high-surprise chunks
arrive in quick succession (preventing the whiteboard from flooding
during a lively discussion).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque

from .predictor import SurpriseScore

logger = logging.getLogger(__name__)


class FilterDecision(str, Enum):
    """Outcome of the KL filter evaluation."""
    PROMOTE    = "promote"     # Score exceeded epsilon — forward to semantic dedup
    DISCARD    = "discard"     # Score below epsilon — routine, not worth promoting
    RATE_LIMIT = "rate_limit"  # Adaptive filter suppressed a burst


@dataclass
class KLFilterResult:
    """Result of a single KL filter evaluation."""

    decision: FilterDecision
    score: float
    """The normalised surprise score from the predictor."""

    epsilon: float
    """The epsilon value that was applied (may differ from config if adaptive)."""

    reason: str
    """Human-readable explanation, useful for debugging and Nightly Synthesis."""


class KLFilter:
    """
    Threshold-based gate that promotes high-surprise chunks.

    Parameters
    ----------
    epsilon : float
        Surprise threshold in [0, 1].  Chunks with score > epsilon are
        promoted.  Defaults to ``BSCDeploymentConfig.epsilon`` (0.55).
    min_token_count : int
        Chunks shorter than this many tokens are always discarded regardless
        of their surprise score (avoids promoting stray punctuation).
    """

    def __init__(self, epsilon: float = 0.55, min_token_count: int = 8) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be strictly between 0 and 1")
        self._epsilon = epsilon
        self._min_token_count = min_token_count

    def evaluate(self, ss: SurpriseScore) -> KLFilterResult:
        """
        Apply the threshold decision to a SurpriseScore.

        Parameters
        ----------
        ss : SurpriseScore
            Output from ``BSCPredictor.score_surprise``.

        Returns
        -------
        KLFilterResult
        """
        # Too short to be meaningful
        if ss.token_count < self._min_token_count:
            return KLFilterResult(
                decision=FilterDecision.DISCARD,
                score=ss.score,
                epsilon=self._epsilon,
                reason=(
                    f"chunk too short ({ss.token_count} tokens < "
                    f"min {self._min_token_count})"
                ),
            )

        if ss.score > self._epsilon:
            return KLFilterResult(
                decision=FilterDecision.PROMOTE,
                score=ss.score,
                epsilon=self._epsilon,
                reason=(
                    f"surprise score {ss.score:.3f} exceeds epsilon {self._epsilon:.3f} "
                    f"(perplexity={ss.raw_perplexity:.1f}, "
                    f"baseline={ss.adaptive_baseline:.1f})"
                ),
            )

        return KLFilterResult(
            decision=FilterDecision.DISCARD,
            score=ss.score,
            epsilon=self._epsilon,
            reason=(
                f"surprise score {ss.score:.3f} ≤ epsilon {self._epsilon:.3f} — "
                "expected information, discarding"
            ),
        )

    @property
    def epsilon(self) -> float:
        return self._epsilon


class AdaptiveKLFilter(KLFilter):
    """
    KL filter that temporarily raises epsilon during high-surprise bursts
    to prevent whiteboard flooding.

    When the rolling mean of recent scores exceeds *burst_threshold*, the
    effective epsilon is raised by *burst_penalty* until the window cools
    down.  This preserves the "only news, not noise" invariant even when an
    agent is in the middle of a chain of discoveries.

    Parameters
    ----------
    epsilon : float
        Base epsilon (same as ``KLFilter``).
    window_size : int
        Number of recent evaluations included in the rolling window.
    burst_threshold : float
        If the rolling mean score exceeds this value, burst mode activates.
    burst_penalty : float
        How much to add to epsilon during burst mode.
    min_token_count : int
        Same as ``KLFilter``.
    """

    def __init__(
        self,
        epsilon: float = 0.55,
        window_size: int = 20,
        burst_threshold: float = 0.70,
        burst_penalty: float = 0.15,
        min_token_count: int = 8,
    ) -> None:
        super().__init__(epsilon=epsilon, min_token_count=min_token_count)
        self._window: Deque[float] = deque(maxlen=window_size)
        self._burst_threshold = burst_threshold
        self._burst_penalty = burst_penalty

    def evaluate(self, ss: SurpriseScore) -> KLFilterResult:
        self._window.append(ss.score)

        # Compute effective epsilon
        effective_epsilon = self._epsilon
        in_burst = False
        if len(self._window) >= 5:
            rolling_mean = sum(self._window) / len(self._window)
            if rolling_mean > self._burst_threshold:
                effective_epsilon = min(
                    0.95, self._epsilon + self._burst_penalty
                )
                in_burst = True

        if ss.token_count < self._min_token_count:
            return KLFilterResult(
                decision=FilterDecision.DISCARD,
                score=ss.score,
                epsilon=effective_epsilon,
                reason=f"chunk too short ({ss.token_count} tokens)",
            )

        if in_burst and ss.score <= effective_epsilon:
            return KLFilterResult(
                decision=FilterDecision.RATE_LIMIT,
                score=ss.score,
                epsilon=effective_epsilon,
                reason=(
                    f"burst mode active (rolling mean={sum(self._window)/len(self._window):.3f} "
                    f"> {self._burst_threshold}); effective epsilon raised to {effective_epsilon:.3f}"
                ),
            )

        if ss.score > effective_epsilon:
            return KLFilterResult(
                decision=FilterDecision.PROMOTE,
                score=ss.score,
                epsilon=effective_epsilon,
                reason=(
                    f"surprise {ss.score:.3f} > epsilon {effective_epsilon:.3f}"
                    + (" [burst mode]" if in_burst else "")
                ),
            )

        return KLFilterResult(
            decision=FilterDecision.DISCARD,
            score=ss.score,
            epsilon=effective_epsilon,
            reason=f"surprise {ss.score:.3f} ≤ epsilon {effective_epsilon:.3f}",
        )

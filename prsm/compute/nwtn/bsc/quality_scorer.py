"""
Quality Scorer — Multi-Dimensional Chunk Quality Evaluation
============================================================

The QualityScorer evaluates promoted chunks across four independent quality
dimensions, providing a richer signal than surprise alone.  It is designed to
complement the KL filter: while the KL filter asks "is this *novel*?", the
QualityScorer asks "is this *correct and useful*?".

Motivation
----------
Surprising ≠ correct.  Wrong information that is novel still passes the KL
filter.  The Anthropic harness paper identified this as a critical gap:
"Agents are terrible at self-evaluation" and need dedicated evaluators.  This
module is the P0 (no-LLM) implementation of that evaluator.

Quality Dimensions
------------------
1. **Factual Consistency** (0–1):
   Does the chunk contradict existing whiteboard entries?  Uses cosine
   similarity with sign-inversion heuristic: high similarity at the
   embedding level combined with negation markers suggests contradiction.
   Range: 1.0 = fully consistent, 0.0 = strong contradiction.

2. **Source Reliability** (0–1):
   Score derived from the agent's historical accuracy in the DomainAuthority
   system.  Agents with poor track records get penalised.  Falls back to 0.5
   (neutral) for unknown agents.

3. **Actionability** (0–1):
   Heuristic: does the chunk contain concrete, actor-impacting content?
   Signals: file paths, decision keywords, status changes, specific artifacts.
   Vague observations and meta-commentary score low.

4. **Coherence** (0–1):
   Perplexity in both directions — too-low perplexity suggests boilerplate,
   too-high suggests nonsense.  We want the "Goldilocks zone": semantically
   grounded but not merely a template repetition.

Design Decisions
----------------
- No LLM dependency for P0.  All scoring is heuristic + embedding-based.
- Graceful degradation: any dimension scoring failure returns 0.5 (neutral).
- Each dimension is independently configurable with per-weight tuning knobs.
- Output is a typed ``QualityScore`` dataclass with per-dimension fields for
  the tuning loop to consume.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Actionability heuristics ──────────────────────────────────────────────────

# Keywords that strongly suggest a concrete decision or status change
_DECISION_KEYWORDS: frozenset[str] = frozenset({
    "decided", "decision", "approved", "rejected", "merged", "closed", "opened",
    "deployed", "reverted", "blocked", "unblocked", "resolved", "escalated",
    "assigned", "implemented", "completed", "shipped", "deprecated", "removed",
    "added", "updated", "fixed", "created", "deleted", "migrated",
})

# Regex patterns for concrete artifacts
_FILE_PATH_RE = re.compile(
    r"(?:"
    r"(?:/[\w.\-]+)+|"           # Unix absolute path: /foo/bar.py
    r"(?:[\w.\-]+/)+[\w.\-]+|"  # Relative path: foo/bar.py
    r"[\w.\-]+\.(?:py|ts|js|go|rs|java|yaml|yml|json|toml|md|sh|sql|tf|env)"
    r")",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://\S+")
_ISSUE_REF_RE = re.compile(r"(?:#\d+|[A-Z]+-\d+)")  # #123 or JIRA-456
_CODE_BLOCK_RE = re.compile(r"(?:```|\bcode\b|`[^`]+`)")

# Words that suggest vague meta-commentary rather than concrete content
_VAGUE_MARKERS: frozenset[str] = frozenset({
    "perhaps", "maybe", "seems", "appears", "might", "could", "should consider",
    "it looks like", "wondering", "curious", "interesting", "note that",
    "fyi", "just", "probably", "unclear", "unsure", "tbd", "todo",
})

# Contradiction markers — if chunk has high embedding similarity to a whiteboard
# entry AND contains these negation patterns, flag as potential contradiction
_NEGATION_PATTERNS = re.compile(
    r"\b(?:not|no longer|never|cannot|can't|won't|doesn't|isn't|aren't|wasn't|"
    r"weren't|hasn't|haven't|invalid|incorrect|wrong|false|deprecated|void)\b",
    re.IGNORECASE,
)

# Coherence: normal perplexity range for high-quality agent output
# Below _COHERENCE_FLOOR → boilerplate/template; above _COHERENCE_CEILING → nonsense
_COHERENCE_FLOOR = 10.0
_COHERENCE_CEILING = 500.0
_COHERENCE_SWEET_SPOT = (20.0, 200.0)  # (min, max) ideal range


@dataclass
class QualityScore:
    """
    Multi-dimensional quality evaluation of a single chunk.

    All dimension scores are in [0, 1] where 1.0 is best.
    """

    factual_consistency: float
    """
    1.0 = no contradictions detected with existing whiteboard entries.
    0.0 = strong contradiction signal detected.
    """

    source_reliability: float
    """
    1.0 = agent has excellent historical track record.
    0.5 = unknown agent (neutral default).
    0.0 = agent has poor track record.
    """

    actionability: float
    """
    1.0 = chunk contains concrete artifacts, decisions, or status changes.
    0.0 = vague observation with no actionable content.
    """

    coherence: float
    """
    1.0 = perplexity in the Goldilocks zone (semantically grounded, not boilerplate).
    0.0 = extreme perplexity (either pure template or gibberish).
    """

    overall: float
    """
    Weighted combination of the four dimension scores.
    Computed externally by :class:`QualityScorer` with configurable weights.
    """

    dimension_weights: dict = field(default_factory=dict)
    """The weights used to compute ``overall``.  Retained for the tuning loop."""

    diagnostics: dict = field(default_factory=dict)
    """Per-dimension diagnostic details for debugging and threshold tuning."""


class QualityScorer:
    """
    Evaluates a chunk across four quality dimensions and produces an overall
    quality score in [0, 1].

    Parameters
    ----------
    factual_weight : float
        Weight for factual consistency in the overall score.  Default 0.30.
    reliability_weight : float
        Weight for source reliability.  Default 0.25.
    actionability_weight : float
        Weight for actionability.  Default 0.25.
    coherence_weight : float
        Weight for coherence.  Default 0.20.
    contradiction_similarity_threshold : float
        Cosine similarity above which a whiteboard entry is considered a
        potential contradiction candidate (then checked for negation markers).
        Default 0.75.

    Notes
    -----
    Weights must sum to 1.0.  The class validates this on construction.
    All dimension scoring methods degrade gracefully: any exception returns
    a neutral 0.5 for that dimension.
    """

    def __init__(
        self,
        factual_weight: float = 0.30,
        reliability_weight: float = 0.25,
        actionability_weight: float = 0.25,
        coherence_weight: float = 0.20,
        contradiction_similarity_threshold: float = 0.75,
    ) -> None:
        total = factual_weight + reliability_weight + actionability_weight + coherence_weight
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"QualityScorer: dimension weights must sum to 1.0, got {total:.4f}"
            )
        self._weights = {
            "factual": factual_weight,
            "reliability": reliability_weight,
            "actionability": actionability_weight,
            "coherence": coherence_weight,
        }
        self._contradiction_threshold = contradiction_similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        chunk: str,
        *,
        source_agent: str,
        raw_perplexity: float,
        whiteboard_entries: Sequence[str] = (),
        whiteboard_embeddings: Optional[List[List[float]]] = None,
        chunk_embedding: Optional[List[float]] = None,
        agent_accuracy_rate: Optional[float] = None,
    ) -> QualityScore:
        """
        Compute a :class:`QualityScore` for ``chunk``.

        Parameters
        ----------
        chunk : str
            The text chunk to evaluate.
        source_agent : str
            Agent identifier; used for source reliability scoring.
        raw_perplexity : float
            Raw perplexity from the BSC predictor (already available from the
            KL filter stage — no recomputation needed).
        whiteboard_entries : Sequence[str]
            Existing whiteboard text entries (for contradiction detection).
            Pass an empty sequence if the whiteboard is empty.
        whiteboard_embeddings : list of list of float, optional
            Pre-computed embeddings for ``whiteboard_entries``.  If provided,
            contradiction detection uses cosine similarity.  If None, falls
            back to lightweight keyword heuristics only.
        chunk_embedding : list of float, optional
            Pre-computed embedding for ``chunk``.  If provided, avoids
            redundant embedding computation.
        agent_accuracy_rate : float, optional
            Historical accuracy rate in [0, 1] for ``source_agent``.  If None,
            defaults to 0.5 (neutral — unknown agent).

        Returns
        -------
        QualityScore
        """
        factual, factual_diag = self._score_factual_consistency(
            chunk,
            whiteboard_entries=whiteboard_entries,
            whiteboard_embeddings=whiteboard_embeddings,
            chunk_embedding=chunk_embedding,
        )
        reliability, reliability_diag = self._score_source_reliability(
            source_agent, agent_accuracy_rate=agent_accuracy_rate
        )
        actionability, actionability_diag = self._score_actionability(chunk)
        coherence, coherence_diag = self._score_coherence(raw_perplexity)

        overall = (
            self._weights["factual"] * factual
            + self._weights["reliability"] * reliability
            + self._weights["actionability"] * actionability
            + self._weights["coherence"] * coherence
        )
        overall = max(0.0, min(1.0, overall))

        logger.debug(
            "QualityScorer: agent=%s factual=%.3f reliability=%.3f "
            "actionability=%.3f coherence=%.3f overall=%.3f",
            source_agent, factual, reliability, actionability, coherence, overall,
        )

        return QualityScore(
            factual_consistency=factual,
            source_reliability=reliability,
            actionability=actionability,
            coherence=coherence,
            overall=overall,
            dimension_weights=dict(self._weights),
            diagnostics={
                "factual": factual_diag,
                "reliability": reliability_diag,
                "actionability": actionability_diag,
                "coherence": coherence_diag,
            },
        )

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _score_factual_consistency(
        self,
        chunk: str,
        whiteboard_entries: Sequence[str],
        whiteboard_embeddings: Optional[List[List[float]]],
        chunk_embedding: Optional[List[float]],
    ) -> Tuple[float, dict]:
        """
        Detect contradictions between the chunk and existing whiteboard entries.

        Strategy
        --------
        If embeddings are available:
          1. Compute cosine similarity between the chunk and each entry.
          2. For entries above the contradiction similarity threshold, check
             whether the chunk contains negation markers that are absent from
             the whiteboard entry (or vice-versa).  If so, flag as potential
             contradiction.
          3. Score = 1 − max_contradiction_signal.

        If no embeddings:
          Lightweight heuristic: check for semantic overlap keywords + negation.
          This is less precise but never blocks on errors.

        Returns
        -------
        (score, diagnostics)
        """
        try:
            if not whiteboard_entries:
                return 1.0, {"reason": "whiteboard empty — no contradictions possible"}

            # Embedding-based contradiction detection
            if (
                whiteboard_embeddings is not None
                and chunk_embedding is not None
                and len(whiteboard_embeddings) == len(whiteboard_entries)
            ):
                return self._factual_with_embeddings(
                    chunk, chunk_embedding, whiteboard_entries, whiteboard_embeddings
                )

            # Fallback: keyword-based heuristic
            return self._factual_keyword_heuristic(chunk, whiteboard_entries)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "QualityScorer._score_factual_consistency: failed (%s), defaulting to 0.5",
                exc,
            )
            return 0.5, {"error": str(exc), "fallback": True}

    def _factual_with_embeddings(
        self,
        chunk: str,
        chunk_embedding: List[float],
        entries: Sequence[str],
        entry_embeddings: List[List[float]],
    ) -> Tuple[float, dict]:
        """Embedding-based factual consistency check."""
        import numpy as np

        chunk_vec = np.array(chunk_embedding, dtype=np.float32)
        chunk_norm = np.linalg.norm(chunk_vec)
        if chunk_norm == 0:
            return 0.5, {"reason": "zero-norm chunk embedding"}

        chunk_has_negation = bool(_NEGATION_PATTERNS.search(chunk))

        max_contradiction_signal = 0.0
        flagged_entry_idx: Optional[int] = None

        for idx, (entry, emb) in enumerate(zip(entries, entry_embeddings)):
            entry_vec = np.array(emb, dtype=np.float32)
            entry_norm = np.linalg.norm(entry_vec)
            if entry_norm == 0:
                continue
            similarity = float(np.dot(chunk_vec, entry_vec) / (chunk_norm * entry_norm))

            if similarity >= self._contradiction_threshold:
                # High similarity + contradictory negation pattern = contradiction signal
                entry_has_negation = bool(_NEGATION_PATTERNS.search(entry))

                # Contradiction heuristic:
                #   chunk negates something ← entry doesn't, or vice-versa
                negation_flip = chunk_has_negation != entry_has_negation
                contradiction_signal = similarity * (0.9 if negation_flip else 0.2)

                if contradiction_signal > max_contradiction_signal:
                    max_contradiction_signal = contradiction_signal
                    flagged_entry_idx = idx

        score = 1.0 - max_contradiction_signal
        score = max(0.0, min(1.0, score))

        diag: dict = {
            "method": "embedding",
            "max_contradiction_signal": round(max_contradiction_signal, 4),
            "chunk_has_negation": chunk_has_negation,
        }
        if flagged_entry_idx is not None:
            diag["flagged_entry_preview"] = str(entries[flagged_entry_idx])[:80]

        return score, diag

    def _factual_keyword_heuristic(
        self, chunk: str, entries: Sequence[str]
    ) -> Tuple[float, dict]:
        """Lightweight keyword-based factual consistency check (no embeddings)."""
        chunk_lower = chunk.lower()
        chunk_words = set(re.findall(r"\b\w{4,}\b", chunk_lower))
        chunk_has_negation = bool(_NEGATION_PATTERNS.search(chunk))

        max_overlap_with_negation = 0.0

        for entry in entries:
            entry_lower = entry.lower()
            entry_words = set(re.findall(r"\b\w{4,}\b", entry_lower))

            overlap = chunk_words & entry_words
            if not overlap or not entry_words:
                continue

            overlap_ratio = len(overlap) / len(entry_words)

            if overlap_ratio >= 0.4:
                entry_has_negation = bool(_NEGATION_PATTERNS.search(entry))
                if chunk_has_negation != entry_has_negation:
                    signal = overlap_ratio * 0.6  # weaker signal than embeddings
                    max_overlap_with_negation = max(max_overlap_with_negation, signal)

        score = 1.0 - max_overlap_with_negation
        return max(0.0, min(1.0, score)), {
            "method": "keyword_heuristic",
            "max_contradiction_signal": round(max_overlap_with_negation, 4),
            "chunk_has_negation": chunk_has_negation,
        }

    def _score_source_reliability(
        self, source_agent: str, agent_accuracy_rate: Optional[float]
    ) -> Tuple[float, dict]:
        """
        Score based on the agent's historical accuracy from the DomainAuthority
        system.

        If ``agent_accuracy_rate`` is not provided (unknown agent), returns
        the neutral default of 0.5.

        The mapping is non-linear: we reward agents with very high accuracy more
        strongly than we penalise poor accuracy (moderate asymmetry encourages
        contribution without overly punishing mistakes).
        """
        try:
            if agent_accuracy_rate is None:
                return 0.5, {"reason": "unknown agent — neutral default", "agent": source_agent}

            rate = max(0.0, min(1.0, agent_accuracy_rate))

            # Non-linear mapping: [0, 0.5] → [0, 0.3], [0.5, 1.0] → [0.3, 1.0]
            # This gives a steeper reward curve for accuracy above 0.5.
            if rate <= 0.5:
                score = 0.6 * rate  # 0 → 0, 0.5 → 0.3
            else:
                # 0.5 → 0.3, 1.0 → 1.0: linear from 0.3 to 1.0
                score = 0.3 + 1.4 * (rate - 0.5)

            score = max(0.0, min(1.0, score))
            return score, {
                "agent": source_agent,
                "accuracy_rate": round(rate, 4),
                "mapped_score": round(score, 4),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "QualityScorer._score_source_reliability: failed (%s), defaulting to 0.5", exc
            )
            return 0.5, {"error": str(exc), "fallback": True}

    def _score_actionability(self, chunk: str) -> Tuple[float, dict]:
        """
        Heuristic actionability scorer.

        Signals that boost score
        ~~~~~~~~~~~~~~~~~~~~~~~~
        - Contains file paths or URLs (concrete artifact references)
        - Contains decision verbs (decided, merged, deployed, etc.)
        - Contains issue / PR references (#123, JIRA-456)
        - Contains code block markers

        Signals that suppress score
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - Excessive vague markers (perhaps, maybe, fyi, etc.)
        - Very short chunks (< 8 words) with no concrete content

        Returns
        -------
        (score, diagnostics)
        """
        try:
            words = chunk.lower().split()
            word_count = len(words)

            if word_count == 0:
                return 0.0, {"reason": "empty chunk"}

            # Positive signals
            has_file_path = bool(_FILE_PATH_RE.search(chunk))
            has_url = bool(_URL_RE.search(chunk))
            has_issue_ref = bool(_ISSUE_REF_RE.search(chunk))
            has_code = bool(_CODE_BLOCK_RE.search(chunk))

            chunk_lower = chunk.lower()
            decision_hits = sum(
                1 for kw in _DECISION_KEYWORDS if kw in chunk_lower
            )
            has_decision = decision_hits > 0

            # Negative signals
            vague_hits = sum(1 for vm in _VAGUE_MARKERS if vm in chunk_lower)
            vague_density = vague_hits / max(word_count / 10.0, 1.0)

            # Scoring
            positive_score = sum([
                0.30 * int(has_file_path),
                0.25 * int(has_decision),
                0.20 * int(has_url),
                0.15 * int(has_issue_ref),
                0.10 * int(has_code),
            ])
            # Multiple decision keywords stack (up to 0.40 bonus)
            positive_score += min(0.15, 0.05 * (decision_hits - 1)) if decision_hits > 1 else 0.0

            # Vague marker penalty
            vague_penalty = min(0.4, 0.10 * vague_density)
            score = max(0.0, min(1.0, positive_score - vague_penalty))

            # Minimum floor: at least 0.10 for any non-empty chunk (don't
            # completely zero out chunks that merely lack concrete markers)
            score = max(0.10, score)

            return score, {
                "has_file_path": has_file_path,
                "has_url": has_url,
                "has_issue_ref": has_issue_ref,
                "has_code": has_code,
                "decision_hits": decision_hits,
                "vague_hits": vague_hits,
                "vague_penalty": round(vague_penalty, 4),
                "positive_score": round(positive_score, 4),
            }

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "QualityScorer._score_actionability: failed (%s), defaulting to 0.5", exc
            )
            return 0.5, {"error": str(exc), "fallback": True}

    def _score_coherence(self, raw_perplexity: float) -> Tuple[float, dict]:
        """
        Map the predictor's raw perplexity to a coherence score.

        Mapping rationale
        -----------------
        - Perplexity < 10 → likely boilerplate / exact template repetition → score 0
        - Perplexity in [20, 200] → healthy zone → score 1.0
        - Perplexity > 500 → nonsense / corrupted → score 0
        - Linear ramps between the zones.

        Returns
        -------
        (score, diagnostics)
        """
        try:
            p = max(0.0, float(raw_perplexity))

            if p <= _COHERENCE_FLOOR:
                # Too low: boilerplate
                score = p / _COHERENCE_FLOOR  # 0 → 0, 10 → 1.0
            elif p <= _COHERENCE_SWEET_SPOT[0]:
                # Ramp up from floor to sweet spot
                score = (p - _COHERENCE_FLOOR) / (_COHERENCE_SWEET_SPOT[0] - _COHERENCE_FLOOR)
            elif p <= _COHERENCE_SWEET_SPOT[1]:
                # Sweet spot
                score = 1.0
            elif p <= _COHERENCE_CEILING:
                # Ramp down from sweet spot to ceiling
                score = 1.0 - (p - _COHERENCE_SWEET_SPOT[1]) / (
                    _COHERENCE_CEILING - _COHERENCE_SWEET_SPOT[1]
                )
            else:
                # Above ceiling: nonsense
                score = 0.0

            score = max(0.0, min(1.0, score))
            return score, {
                "raw_perplexity": round(p, 2),
                "floor": _COHERENCE_FLOOR,
                "ceiling": _COHERENCE_CEILING,
                "sweet_spot": _COHERENCE_SWEET_SPOT,
            }

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "QualityScorer._score_coherence: failed (%s), defaulting to 0.5", exc
            )
            return 0.5, {"error": str(exc), "fallback": True}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> dict:
        """Read-only view of the current dimension weights."""
        return dict(self._weights)

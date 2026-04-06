"""
Collision Detector
==================

Compares outputs from diversified pipelines to detect tampering.
Accounts for DP noise variance when comparing.
"""

import json
import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CollisionDetector:
    """Detects output divergence between diversified pipelines."""

    def __init__(
        self,
        dp_epsilon: float = 8.0,
        tolerance_multiplier: float = 3.0,
    ):
        self.dp_epsilon = dp_epsilon
        self.tolerance_multiplier = tolerance_multiplier

    def _expected_noise_std(self) -> float:
        """Expected standard deviation of DP noise."""
        if self.dp_epsilon == float("inf") or self.dp_epsilon <= 0:
            return 0.0
        # sigma = clip_norm * sqrt(2 * ln(1.25/delta)) / epsilon
        # Using clip_norm=1.0, delta=1e-5
        return 1.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / self.dp_epsilon

    def compare_pipelines(
        self,
        output_a: bytes,
        output_b: bytes,
    ) -> Tuple[bool, float]:
        """Compare two pipeline outputs.

        Returns (match: bool, divergence_score: float).
        Match is True if divergence is within expected DP noise tolerance.
        """
        try:
            arr_a = np.array(json.loads(output_a), dtype=np.float64).flatten()
            arr_b = np.array(json.loads(output_b), dtype=np.float64).flatten()
        except (json.JSONDecodeError, ValueError, TypeError):
            # Non-numeric: exact byte comparison
            return output_a == output_b, 0.0 if output_a == output_b else 1.0

        if arr_a.size != arr_b.size:
            return False, 1.0

        if arr_a.size == 0:
            return True, 0.0

        # Compute divergence as normalized L2 distance
        diff = np.linalg.norm(arr_a - arr_b)
        norm = max(np.linalg.norm(arr_a), np.linalg.norm(arr_b), 1e-10)
        divergence = diff / norm

        # Threshold: expected DP noise std * tolerance multiplier
        expected_std = self._expected_noise_std()
        threshold = expected_std * self.tolerance_multiplier

        # If no DP (epsilon=inf), threshold is near zero — require exact match
        if threshold < 1e-10:
            threshold = 1e-6

        match = bool(divergence <= threshold)
        return match, float(divergence)

    def detect_collision(
        self,
        outputs: List[bytes],
    ) -> Dict[str, Any]:
        """Compare all pipeline outputs pairwise.

        Returns detection report with match status and flagged indices.
        """
        if len(outputs) < 2:
            return {"match": True, "comparisons": 0, "flagged_indices": []}

        comparisons = 0
        divergences = []
        flagged = set()

        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                match, div = self.compare_pipelines(outputs[i], outputs[j])
                comparisons += 1
                divergences.append({"i": i, "j": j, "match": match, "divergence": div})
                if not match:
                    flagged.add(i)
                    flagged.add(j)

        all_match = len(flagged) == 0

        return {
            "match": all_match,
            "comparisons": comparisons,
            "divergences": divergences,
            "flagged_indices": sorted(flagged),
        }

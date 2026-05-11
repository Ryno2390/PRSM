"""
Privacy Budget Tracker
======================

Enforces cumulative differential privacy epsilon limits
across inference sessions.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class PrivacySpend:
    """A single privacy budget expenditure."""
    epsilon: float
    operation: str
    # Sprint 263 — added job_id field. Pre-fix the 3 record_spend
    # callsites in api.py passed the job_id positionally where
    # model_id was expected, so the audit log was structurally
    # confused (model_id contained job_id; actual model_id lost).
    job_id: str = ""
    model_id: str = ""
    timestamp: float = field(default_factory=time.time)


class PrivacyBudgetTracker:
    """Tracks cumulative epsilon spend and enforces limits."""

    def __init__(self, max_epsilon: float = 100.0):
        self.max_epsilon = max_epsilon
        self._spends: List[PrivacySpend] = []

    @property
    def total_spent(self) -> float:
        return sum(s.epsilon for s in self._spends)

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_epsilon - self.total_spent)

    def can_spend(self, epsilon: float) -> bool:
        return self.total_spent + epsilon <= self.max_epsilon

    def record_spend(
        self,
        epsilon: float,
        operation: str,
        job_id: str = "",
        model_id: str = "",
    ) -> bool:
        """Record a privacy spend. Returns False if would exceed budget.

        Sprint 263 — third positional arg is now ``job_id`` (was
        ``model_id``, which collided with how api.py callers actually
        pass it). ``model_id`` keyword preserved for sites that
        legitimately want to record the model.

        Rejects non-finite (NaN / +inf / -inf) and non-positive ε before
        the budget check. The non-positive guard is load-bearing: a
        negative ε would CREDIT the budget back, letting an operator
        write `record_spend(-50.0, ...)` to dodge a future ceiling.
        Caught by the Phase 3.x.4 round-1 review.
        """
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            logger.warning(
                f"Privacy spend rejected: epsilon must be finite and "
                f"positive, got {epsilon!r}"
            )
            return False
        if not self.can_spend(epsilon):
            logger.warning(
                f"Privacy budget exceeded: {self.total_spent + epsilon:.1f} > {self.max_epsilon}"
            )
            return False
        self._spends.append(PrivacySpend(
            epsilon=epsilon,
            operation=operation,
            job_id=job_id,
            model_id=model_id,
        ))
        return True

    def get_audit_report(self) -> Dict[str, Any]:
        return {
            "max_epsilon": self.max_epsilon,
            "total_spent": self.total_spent,
            "remaining": self.remaining,
            "num_operations": len(self._spends),
            "spends": [
                {
                    "epsilon": s.epsilon,
                    "operation": s.operation,
                    "job_id": s.job_id,
                    "model_id": s.model_id,
                    "timestamp": s.timestamp,
                }
                for s in self._spends[-20:]  # Last 20
            ],
        }

    def reset(self) -> None:
        self._spends.clear()

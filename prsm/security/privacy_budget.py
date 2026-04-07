"""
Privacy Budget Tracker
======================

Enforces cumulative differential privacy epsilon limits
across inference sessions.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PrivacySpend:
    """A single privacy budget expenditure."""
    epsilon: float
    operation: str
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

    def record_spend(self, epsilon: float, operation: str, model_id: str = "") -> bool:
        """Record a privacy spend. Returns False if would exceed budget."""
        if not self.can_spend(epsilon):
            logger.warning(
                f"Privacy budget exceeded: {self.total_spent + epsilon:.1f} > {self.max_epsilon}"
            )
            return False
        self._spends.append(PrivacySpend(
            epsilon=epsilon,
            operation=operation,
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
                {"epsilon": s.epsilon, "operation": s.operation, "model_id": s.model_id}
                for s in self._spends[-20:]  # Last 20
            ],
        }

    def reset(self) -> None:
        self._spends.clear()

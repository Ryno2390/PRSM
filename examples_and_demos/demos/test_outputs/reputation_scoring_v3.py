"""
Dynamic reputation scoring system for PRSM model registry.

Replaces static performance_score with a dynamic calculation based on:
- Usage frequency (usage_count)
- User ratings (avg_rating)
- Success rate (success_rate)
- Node uptime (uptime_pct)

Integration:
1. Call patch_model_registry() to monkeypatch ModelRegistry methods
2. Call update_reputation() after each model invocation
3. Scores are automatically used in model selection
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import statistics
from collections import deque

from prsm.compute.federation.model_registry import ModelRegistry, ModelDetails
from prsm.compute.federation.model_registry import ModelCapability, ModelProvider

@dataclass
class ReputationRecord:
    """Stores reputation metrics for a single model."""
    model_id: str
    usage_count: int = 0
    ratings: List[float] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    uptime_samples: deque = field(default_factory=lambda: deque(maxlen=30))
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def avg_rating(self) -> float:
        return statistics.mean(self.ratings) if self.ratings else 3.0  # Default to neutral

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0

    @property
    def uptime_pct(self) -> float:
        return statistics.mean(self.uptime_samples) if self.uptime_samples else 1.0

class ReputationScoringSystem:
    """Maintains and calculates dynamic reputation scores for models."""
    
    def __init__(self):
        self.records: Dict[str, ReputationRecord] = {}

    def update_reputation(
        self,
        model_id: str,
        rating: Optional[float] = None,
        success: bool = True,
        response_time_ms: Optional[float] = None
    ) -> None:
        """Update reputation metrics after a model invocation.
        
        Args:
            model_id: Identifier for the model
            rating: Optional user rating (1.0-5.0)
            success: Whether the invocation succeeded
            response_time_ms: Optional response time (for future QoS metrics)
        """
        record = self.records.setdefault(model_id, ReputationRecord(model_id))
        record.usage_count += 1
        
        if rating is not None:
            record.ratings.append(max(1.0, min(5.0, rating)))  # Clamp to valid range
            
        if success:
            record.success_count += 1
        else:
            record.failure_count += 1
            
        record.last_updated = datetime.utcnow()

    def get_reputation_score(self, model_id: str) -> float:
        """Calculate current reputation score (0.0-1.0) for a model."""
        # OPEN: Actual scoring formula pending scientific review
        record = self.records.get(model_id)
        if not record:
            return 0.8  # Default for unrated models
            
        return min(1.0, max(0.0, record.avg_rating / 5.0))  # Placeholder

    def get_all_scores(self) -> Dict[str, float]:
        """Get current reputation scores for all tracked models."""
        return {model_id: self.get_reputation_score(model_id) 
                for model_id in self.records}

def patch_model_registry(registry: ModelRegistry, scoring_system: ReputationScoringSystem) -> None:
    """Monkeypatch ModelRegistry to use dynamic reputation scores."""
    original_select = registry.select_best_model
    
    def patched_select(capability, domain=None):
        candidates = registry.discover_models(capability, domain)
        return max(
            candidates,
            key=lambda m: m.availability * scoring_system.get_reputation_score(m.model_id),
            default=None
        )
    
    registry.select_best_model = patched_select


# --- (continued from later round) ---

...

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math
from pydantic import BaseModel, Field, validator
from prsm.models.model_registry import ModelRegistry

class Rating(BaseModel):
    """A single rating for a model with timestamp and score."""
    score: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    rater_id: str  # Identifier for the entity providing the rating
    
    @validator('score')
    def round_score(cls, v):
        """Round score to 2 decimal places for consistency."""
        return round(v, 2)

class ModelReputation(BaseModel):
    """Reputation data for a single model."""
    model_id: str
    ratings: List[Rating] = []
    decay_rate: float = Field(default=0.95, ge=0.8, le=0.99)  # 5% decay per time unit
    
    def current_score(self) -> float:
        """Calculate the current reputation score with time decay."""
        if not self.ratings:
            return 0.0  # Cold start for new models
            
        now = datetime.utcnow()
        total_weight = 0.0
        weighted_sum = 0.0
        
        for rating in sorted(self.ratings, key=lambda x: x.timestamp, reverse=True):
            age = (now - rating.timestamp).total_seconds() / 86400  # Age in days
            weight = math.pow(self.decay_rate, age)
            weighted_sum += rating.score * weight
            total_weight += weight
            
        return min(max(weighted_sum / total_weight, 0.0), 1.0)

class ReputationSystem:
    """Core reputation scoring system for model federation."""
    def __init__(self):
        self.models: Dict[str, ModelReputation] = {}
    
    def register_model(self, model_id: str, decay_rate: Optional[float] = None) -> None:
        """Register a new model in the reputation system."""
        if model_id not in self.models:
            kwargs = {'model_id': model_id}
            if decay_rate is not None:
                kwargs['decay_rate'] = decay_rate
            self.models[model_id] = ModelReputation(**kwargs)
    
    def add_rating(self, model_id: str, score: float, rater_id: str) -> None:
        """Add a new rating for a model."""
        if model_id not in self.models:
            self.register_model(model_id)
            
        self.models[model_id].ratings.append(
            Rating(score=score, timestamp=datetime.utcnow(), rater_id=rater_id)
        )
    
    def get_score(self, model_id: str) -> float:
        """Get the current reputation score for a model."""
        if model_id not in self.models:
            return 0.0
        return self.models[model_id].current_score()
    
    def patch_model_registry(self, registry: ModelRegistry) -> ModelRegistry:
        """Patch a ModelRegistry instance to include reputation scoring."""
        original_method = registry.get_model_metadata
        
        def wrapped_get_model_metadata(model_id: str, *args, **kwargs):
            metadata = original_method(model_id, *args, **kwargs)
            metadata['reputation_score'] = self.get_score(model_id)
            return metadata
        
        registry.get_model_metadata = wrapped_get_model_metadata
        return registry

# Singleton instance
_reputation_system = ReputationSystem()

def get_reputation_system() -> ReputationSystem:
    """Get the singleton reputation system instance."""
    return _reputation_system

def patch_model_registry(registry: ModelRegistry) -> ModelRegistry:
    """One-line function to patch a ModelRegistry with reputation support."""
    return get_reputation_system().patch_model_registry(registry)

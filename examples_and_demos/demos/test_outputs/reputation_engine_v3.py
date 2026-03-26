from datetime import datetime, timedelta
from typing import Optional
import math
from pydantic import BaseModel, Field, validator
from prsm.compute.federation.model_registry import ModelRegistry, ModelDetails

class ReputationRecord(BaseModel):
    """Core reputation record storing all components needed for scoring."""
    model_id: str = Field(..., description="Unique identifier for the model")
    last_updated: datetime = Field(default_factory=datetime.utcnow, 
                                 description="Timestamp of last update")
    total_interactions: int = Field(default=0, ge=0,
                                  description="Total number of interactions")
    successful_interactions: int = Field(default=0, ge=0,
                                       description="Count of successful interactions")
    failed_interactions: int = Field(default=0, ge=0,
                                   description="Count of failed interactions")
    sybil_attempts: int = Field(default=0, ge=0,
                              description="Count of detected Sybil attacks")
    last_decay_applied: datetime = Field(default_factory=datetime.utcnow,
                                       description="When decay was last applied")
    
    @validator('successful_interactions', 'failed_interactions', 'sybil_attempts')
    def validate_interactions(cls, v, values):
        if 'total_interactions' in values and v > values['total_interactions']:
            raise ValueError("Component interactions cannot exceed total interactions")
        return v

class ReputationEngine:
    """Implements the reputation scoring system with decay and attack mitigation."""
    
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self._records: dict[str, ReputationRecord] = {}
        
        # Parameters from data scientist's formula
        self.SUCCESS_WEIGHT = 0.6
        self.FAILURE_WEIGHT = -0.8
        self.SYBIL_PENALTY = -1.5
        self.DECAY_RATE = 0.95  # 5% decay per day
        self.COLD_START_SCORE = 0.5
        
    async def update_reputation(self, model_id: str, success: bool, 
                              is_sybil_attempt: bool = False) -> float:
        """Update reputation based on a new interaction event."""
        record = await self._get_or_create_record(model_id)
        
        # Apply time decay before updating
        await self._apply_decay(record)
        
        # Update interaction counts
        record.total_interactions += 1
        if success:
            record.successful_interactions += 1
        else:
            record.failed_interactions += 1
            
        if is_sybil_attempt:
            record.sybil_attempts += 1
            
        record.last_updated = datetime.utcnow()
        self._records[model_id] = record
        
        return await self.get_reputation_score(model_id)
    
    async def get_reputation_score(self, model_id: str) -> float:
        """Calculate current reputation score with cold-start handling."""
        record = await self._get_or_create_record(model_id)
        
        # Apply time decay before calculation
        await self._apply_decay(record)
        
        if record.total_interactions == 0:
            return self.COLD_START_SCORE
            
        # Base score components
        success_ratio = record.successful_interactions / record.total_interactions
        failure_ratio = record.failed_interactions / record.total_interactions
        sybil_ratio = record.sybil_attempts / record.total_interactions
        
        # Calculate weighted score
        score = (
            (success_ratio * self.SUCCESS_WEIGHT) +
            (failure_ratio * self.FAILURE_WEIGHT) +
            (sybil_ratio * self.SYBIL_PENALTY)
        )
        
        # Apply bounds [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
    
    async def _apply_decay(self, record: ReputationRecord) -> None:
        """Apply exponential decay based on time since last update."""
        now = datetime.utcnow()
        days_since_update = (now - record.last_decay_applied).total_seconds() / 86400
        
        if days_since_update >= 1:  # Only decay if at least one day passed
            decay_factor = math.pow(self.DECAY_RATE, days_since_update)
            
            record.successful_interactions = int(record.successful_interactions * decay_factor)
            record.failed_interactions = int(record.failed_interactions * decay_factor)
            record.sybil_attempts = int(record.sybil_attempts * decay_factor)
            record.total_interactions = (record.successful_interactions + 
                                       record.failed_interactions + 
                                       record.sybil_attempts)
            
            record.last_decay_applied = now
    
    async def _get_or_create_record(self, model_id: str) -> ReputationRecord:
        """Get existing record or create new one after verifying model exists."""
        # Verify model exists in registry
        model = await self.registry.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found in registry")
            
        if model_id not in self._records:
            self._records[model_id] = ReputationRecord(model_id=model_id)
            
        return self._records[model_id]

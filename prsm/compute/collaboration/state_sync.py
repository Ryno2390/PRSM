"""
PRSM Stream-Based State Syncing
===============================

Moves beyond file-based Git versioning.
Researchers can fork and branch specific 'states' of models and datasets.
"""

import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

@dataclass
class ValidationReport:
    """Impact-Aware Logging: Captures accuracy, bias, and cost of a state change"""
    accuracy_score: float
    bias_index: float
    compute_cost_ftns: float
    benchmark_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ModelState:
    """A specific version of a model's weights or logic"""
    state_id: str
    parent_state_id: Optional[str]
    model_id: str
    logic_cid: str # CID for the System 2 logic (e.g. MCTS parameters)
    weights_cid: str # CID for the System 1 weights (e.g. SSM checkpoints)
    validation: ValidationReport
    author_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class CollaborativeStateSpace:
    """
    Manages the branching and merging of model/data states.
    """
    def __init__(self):
        self.states: Dict[str, ModelState] = {}
        self.heads: Dict[str, str] = {} # branch_name -> state_id

    def create_state(self, 
                     model_id: str, 
                     logic_cid: str, 
                     weights_cid: str, 
                     author_id: str,
                     validation: ValidationReport,
                     parent_id: str = None,
                     branch: str = "main") -> ModelState:
        """
        Creates a new 'Commit' in the state stream.
        """
        state_id = hashlib.sha256(f"{model_id}-{logic_cid}-{weights_cid}-{datetime.now()}".encode()).hexdigest()[:16]
        
        new_state = ModelState(
            state_id=state_id,
            parent_state_id=parent_id or self.heads.get(branch),
            model_id=model_id,
            logic_cid=logic_cid,
            weights_cid=weights_cid,
            validation=validation,
            author_id=author_id
        )
        
        self.states[state_id] = new_state
        self.heads[branch] = state_id
        
        logger.info(f"🚀 New Model State: {state_id} on branch '{branch}' by {author_id}")
        return new_state

    def fork_component(self, state_id: str, new_logic_cid: Optional[str] = None, new_weights_cid: Optional[str] = None) -> Tuple[str, str]:
        """
        Modular Collaboration: Keep weights, change logic (or vice versa).
        """
        base_state = self.states.get(state_id)
        if not base_state:
            raise ValueError("Base state not found")
            
        final_logic = new_logic_cid or base_state.logic_cid
        final_weights = new_weights_cid or base_state.weights_cid
        
        return final_logic, final_weights
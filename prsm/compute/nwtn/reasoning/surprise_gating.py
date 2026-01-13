"""
PRSM Surprise Gating Engine
===========================

Implements Bayesian Surprise Gating to optimize context window usage.
Surprise = -log P(Data | Context). 
Only information that meaningfully changes the model's belief is preserved.
"""

import math
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class SurpriseGater:
    """
    The Information Entropy Filter.
    Ensures repetitive data doesn't clog the context window.
    """
    def __init__(self, surprise_threshold: float = 0.5):
        self.surprise_threshold = surprise_threshold
        self.last_kl_divergence: float = 0.0

    def calculate_surprise(self, predicted_outcome: Any, actual_outcome: Any) -> float:
        """
        Calculates a simplified 'Surprise' metric.
        In production, this would use KL-Divergence between the model's 
        prior and posterior distributions.
        """
        # Simplified: If outcomes are strings, we look at the semantic delta
        # If they are numerical (e.g. battery voltage), we look at the deviation
        if isinstance(actual_outcome, (int, float)) and isinstance(predicted_outcome, (int, float)):
            if predicted_outcome == 0: return 1.0
            deviation = abs(actual_outcome - predicted_outcome) / predicted_outcome
            return deviation
        
        # String-based heuristic: how many new keywords appeared?
        if isinstance(actual_outcome, str) and isinstance(predicted_outcome, str):
            actual_words = set(actual_outcome.lower().split())
            predicted_words = set(predicted_outcome.lower().split())
            new_info = actual_words - predicted_words
            # Ratio of new words to total words
            return len(new_info) / max(1, len(actual_words))
            
        return 1.0 # Unknown types are always surprising

    def should_gate(self, surprise_score: float) -> bool:
        """Returns True if the information should be DISCARDED (low surprise)"""
        return surprise_score < self.surprise_threshold

class SurprisePayload:
    """
    Compact data structure for network synchronization.
    Only contains the information that triggered the gate.
    """
    def __init__(self, data: Any, surprise_score: float, context_id: str):
        self.data = data
        self.surprise_score = surprise_score
        self.context_id = context_id
        self.timestamp = None # To be set by sync layer

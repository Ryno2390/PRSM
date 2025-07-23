"""
NWTN Core Types
===============

Shared data types and enums used throughout the NWTN system.
This module prevents circular dependencies by providing common types.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from decimal import Decimal


class ThinkingMode(Enum):
    """Reasoning complexity levels for NWTN processing"""
    QUICK = "quick"
    INTERMEDIATE = "intermediate" 
    DEEP = "deep"


class ReasoningType(Enum):
    """Types of reasoning engines available in NWTN"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"
    COUNTERFACTUAL = "counterfactual"


class QueryComplexity(Enum):
    """Query complexity levels for processing estimation"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ClarificationStatus(Enum):
    """Status of query clarification process"""
    NOT_NEEDED = "not_needed"
    REQUESTED = "requested"
    PROVIDED = "provided"
    INSUFFICIENT = "insufficient"


@dataclass
class QueryAnalysis:
    """Analysis of user query for processing optimization"""
    query_id: str
    complexity: QueryComplexity
    estimated_tokens: int
    required_reasoning_types: List[ReasoningType]
    clarification_status: ClarificationStatus
    estimated_processing_time_seconds: float
    confidence_score: float = 0.0
    
    # Cost estimation
    estimated_ftns_cost: Optional[Decimal] = None
    cost_breakdown: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningEngineResult:
    """Result from a single reasoning engine"""
    engine_type: ReasoningType
    confidence_score: float
    reasoning_steps: List[str]
    evidence: List[str]
    conclusion: str
    processing_time_seconds: float
    
    # Quality metrics
    logical_consistency: float = 0.0
    empirical_grounding: float = 0.0
    novelty_score: float = 0.0
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalogicalChain:
    """Multi-hop analogical reasoning chain"""
    chain_id: str
    source_domain: str
    target_domain: str
    hops: List[Dict[str, Any]]
    confidence_score: float
    novelty_score: float
    
    # Chain validation
    validated: bool = False
    validation_scores: Dict[ReasoningType, float] = field(default_factory=dict)


@dataclass
class MetaReasoningResult:
    """Complete result from NWTN meta-reasoning process"""
    query_id: str
    original_query: str
    thinking_mode: ThinkingMode
    
    # Core results
    response: str
    confidence_score: float
    reasoning_engines_used: List[ReasoningType]
    engine_results: List[ReasoningEngineResult]
    
    # Advanced features
    analogical_chains: List[AnalogicalChain] = field(default_factory=list)
    breakthrough_insights: List[str] = field(default_factory=list)
    cross_domain_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing metadata
    processing_time_seconds: float = 0.0
    total_ftns_cost: Optional[Decimal] = None
    cost_breakdown: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    cross_validation_score: float = 0.0
    empirical_grounding_score: float = 0.0
    reasoning_completeness: float = 0.0
    
    # System information
    system_version: str = "1.0.0"
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)
    supporting_evidence_count: int = 0


@dataclass
class NWTNResponse:
    """Complete NWTN response for user consumption"""
    query_id: str
    natural_language_response: str
    
    # Core metadata
    confidence_score: float
    reasoning_modes_used: List[str]
    processing_time_seconds: float
    
    # Cost information
    actual_cost_breakdown: Dict[str, Any] = field(default_factory=dict)
    total_cost_ftns: Optional[Decimal] = None
    
    # Enhanced features
    structured_insights: List[Dict[str, Any]] = field(default_factory=list)
    analogical_chains: List[AnalogicalChain] = field(default_factory=list)
    breakthrough_insights: List[str] = field(default_factory=list)
    cross_domain_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality indicators
    citation_count: int = 0
    evidence_grounding_score: float = 0.0
    
    # Response metadata
    response_timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None


@dataclass
class BreakthroughModeConfig:
    """Configuration for breakthrough discovery modes"""
    mode: str = "balanced"  # conservative, balanced, creative, revolutionary
    analogical_chain_depth: int = 4
    contrarian_weight: float = 0.5
    assumption_flip_probability: float = 0.3
    cross_domain_exploration: bool = True
    frontier_detection_enabled: bool = True


@dataclass
class EnhancedUserConfig:
    """Enhanced user configuration for NWTN processing"""
    thinking_complexity: str = "intermediate"
    verbosity_preferences: str = "standard"
    quality_vs_speed_preference: float = 0.7
    breakthrough_discovery_enabled: bool = False
    contrarian_analysis_enabled: bool = False


# Type aliases for common patterns
ProcessingContext = Dict[str, Any]
ReasoningParameters = Dict[str, Union[str, int, float, bool]]
ValidationScores = Dict[ReasoningType, float]
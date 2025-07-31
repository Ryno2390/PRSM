"""
NWTN Breakthrough Mode Configuration System
===========================================

Implements user-configurable breakthrough intensity modes from conservative
academic research to revolutionary moonshot exploration.

Based on NWTN Novel Idea Generation Roadmap Phase 3.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class BreakthroughMode(Enum):
    """User-selectable breakthrough intensity modes"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced" 
    CREATIVE = "creative"
    REVOLUTIONARY = "revolutionary"
    CUSTOM = "custom"

@dataclass
class CandidateTypeDistribution:
    """Distribution of reasoning candidate types for different modes"""
    synthesis: float = 0.0           # Academic synthesis
    methodological: float = 0.0     # Methodological improvements
    empirical: float = 0.0          # Empirical analysis
    applied: float = 0.0            # Applied research
    theoretical: float = 0.0        # Theoretical exploration
    
    # Breakthrough-oriented candidates
    contrarian: float = 0.0         # Opposes consensus
    cross_domain_transplant: float = 0.0  # Distant field solutions
    assumption_flip: float = 0.0    # Inverts core assumptions
    speculative_moonshot: float = 0.0     # Ignores current limits
    historical_analogy: float = 0.0       # Different era solutions

@dataclass
class AnalogicalReasoningConfig:
    """Configuration for analogical chain reasoning system"""
    max_chain_depth: int = 4              # Maximum hops in A→B→C→D chains (1-6)
    consistency_threshold: float = 0.6     # Semantic consistency requirement (0.3-0.9)
    similarity_threshold: float = 0.7      # Analogical similarity requirement (0.4-0.9)
    breakthrough_threshold: float = 0.6    # Breakthrough potential threshold (0.3-0.9)
    cross_domain_aggressiveness: float = 0.5  # Willingness to bridge distant domains (0.2-0.9)
    pattern_recognition_depth: str = "moderate"  # "shallow", "moderate", "deep"
    
@dataclass 
class ContrarianAnalysisConfig:
    """Configuration for contrarian paper identification and analysis"""
    consensus_sensitivity: float = 0.7     # Sensitivity to consensus detection (0.3-0.9)
    contradiction_tolerance: float = 0.6   # Tolerance for contradictory evidence (0.2-0.8)
    credibility_filter: str = "moderate"   # "strict", "moderate", "permissive", "speculative"
    historical_depth: int = 5              # Years to look back for consensus evolution (1-20)
    breakthrough_potential_weight: float = 0.6  # Weight for breakthrough potential (0.2-0.9)
    contrarian_source_preference: str = "balanced"  # "recent", "balanced", "historical"

@dataclass
class CrossDomainBridgeConfig:
    """Configuration for cross-domain ontology bridging"""
    domain_distance_preference: float = 0.6  # Preference for distant domains (0.2-0.9)
    bridge_quality_threshold: float = 0.5    # Quality threshold for bridges (0.3-0.8)
    concept_mapping_strictness: str = "moderate"  # "strict", "moderate", "flexible"
    ontological_depth: int = 3                # Depth of ontological analysis (1-5)
    cross_pollination_aggressiveness: float = 0.5  # Aggressiveness in cross-domain transfer (0.2-0.9)
    domain_expertise_weighting: bool = True   # Weight by domain expertise

@dataclass
class ReasoningEngineConfig:
    """Configuration for the 7 reasoning engines in System 1 (Creative) and System 2 (Validation) modes"""
    
    # System 1 (Creative Generation) Parameters - Control creativity and exploration
    analogical_creativity: float = 0.5          # Analogical reasoning creativity (0.2-0.9)
    deductive_speculation: float = 0.3          # Deductive reasoning speculation level (0.1-0.8)  
    inductive_boldness: float = 0.4             # Inductive reasoning boldness (0.2-0.9)
    abductive_novelty: float = 0.5              # Abductive reasoning novelty seeking (0.2-0.9)
    counterfactual_extremeness: float = 0.3     # Counterfactual reasoning extremeness (0.1-0.9)
    probabilistic_tail_exploration: float = 0.4 # Probabilistic reasoning tail exploration (0.1-0.9)
    causal_mechanism_novelty: float = 0.5       # Causal reasoning mechanism novelty (0.2-0.9)
    
    # System 2 (Validation) Parameters - Control rigor and validation
    validation_strictness: float = 0.7          # Overall validation strictness (0.3-0.9)
    evidence_requirement: float = 0.6           # Evidence quality requirement (0.3-0.9)
    logical_rigor: float = 0.8                  # Logical consistency requirement (0.5-0.9)
    consistency_enforcement: float = 0.7        # Cross-engine consistency enforcement (0.3-0.9)
    contradiction_tolerance: float = 0.3        # Tolerance for contradictions (0.1-0.8)
    
    # Cross-System Parameters - Control interaction between System 1 and System 2
    creative_validation_balance: float = 0.5    # Balance between creativity and validation (0.1-0.9)
    system_transition_threshold: float = 0.6    # Threshold for System 1 → System 2 transition (0.3-0.9)
    reasoning_depth_multiplier: float = 1.0     # Multiplier for reasoning depth (0.5-2.0)

@dataclass
class IntegrationValidationConfig:
    """Configuration for integration and validation system"""
    validation_strictness: str = "moderate"      # "strict", "moderate", "permissive" 
    quality_gate_threshold: float = 0.6          # Overall quality threshold (0.3-0.9)
    cross_system_consistency_weight: float = 0.7 # Weight for cross-system consistency (0.3-0.9)
    validation_timeout_seconds: int = 300        # Maximum time for validation (60-600)
    benchmark_comparison_enabled: bool = True    # Enable historical benchmark comparison
    impossible_problem_testing: bool = False     # Test against impossible problems (advanced mode)

@dataclass
class BreakthroughModeConfig:
    """Complete configuration for a breakthrough mode"""
    mode: BreakthroughMode
    name: str
    description: str
    use_cases: List[str]
    
    # Candidate distribution
    candidate_distribution: CandidateTypeDistribution
    
    # Quality and risk parameters
    confidence_threshold: float
    novelty_weight: float
    consensus_weight: float
    
    # Special features
    assumption_challenging_enabled: bool = False
    wild_hypothesis_enabled: bool = False
    impossibility_exploration_enabled: bool = False
    
    # Pricing modifiers
    complexity_multiplier: float = 1.0
    quality_tier: str = "standard"
    
    # Output preferences
    citation_preferences: str = "balanced"
    explanation_depth: str = "standard"
    
    # Enhanced system configurations
    analogical_config: AnalogicalReasoningConfig = field(default_factory=AnalogicalReasoningConfig)
    contrarian_config: ContrarianAnalysisConfig = field(default_factory=ContrarianAnalysisConfig)
    bridge_config: CrossDomainBridgeConfig = field(default_factory=CrossDomainBridgeConfig)
    validation_config: IntegrationValidationConfig = field(default_factory=IntegrationValidationConfig)
    reasoning_engine_config: ReasoningEngineConfig = field(default_factory=ReasoningEngineConfig)

class BreakthroughModeManager:
    """Manages breakthrough mode configurations and applies them to NWTN reasoning"""
    
    def __init__(self):
        self.modes = self._initialize_default_modes()
        self.context_suggestions = self._initialize_context_suggestions()
        
    def _initialize_default_modes(self) -> Dict[BreakthroughMode, BreakthroughModeConfig]:
        """Initialize the four default breakthrough modes"""
        
        modes = {}
        
        # CONSERVATIVE MODE (Academic/Clinical/Regulatory)
        modes[BreakthroughMode.CONSERVATIVE] = BreakthroughModeConfig(
            mode=BreakthroughMode.CONSERVATIVE,
            name="Conservative",
            description="Established approaches with high confidence and safety",
            use_cases=[
                "Medical research and clinical applications",
                "Regulatory compliance and safety-critical systems", 
                "Academic research requiring peer review",
                "Financial analysis and risk assessment",
                "Legal research and precedent analysis"
            ],
            candidate_distribution=CandidateTypeDistribution(
                synthesis=0.30,
                methodological=0.25,
                empirical=0.20,
                applied=0.15,
                theoretical=0.10,
                # No breakthrough candidates - focus on established approaches
            ),
            confidence_threshold=0.8,
            novelty_weight=0.2,
            consensus_weight=0.8,
            assumption_challenging_enabled=False,
            complexity_multiplier=0.8,  # Simpler, more established reasoning
            quality_tier="high_quality",
            citation_preferences="high_impact_established",
            explanation_depth="thorough_with_citations",
            # Enhanced system configurations for Conservative mode
            analogical_config=AnalogicalReasoningConfig(
                max_chain_depth=2,              # Conservative: shorter chains
                consistency_threshold=0.8,      # High consistency requirement
                similarity_threshold=0.8,       # High similarity requirement
                breakthrough_threshold=0.4,     # Lower breakthrough seeking
                cross_domain_aggressiveness=0.2,  # Conservative domain bridging
                pattern_recognition_depth="shallow"
            ),
            contrarian_config=ContrarianAnalysisConfig(
                consensus_sensitivity=0.9,      # Very sensitive to consensus
                contradiction_tolerance=0.3,    # Low tolerance for contradictions
                credibility_filter="strict",    # Strict credibility requirements
                historical_depth=10,            # Look back further for stability
                breakthrough_potential_weight=0.3,  # Lower breakthrough seeking
                contrarian_source_preference="historical"  # Prefer established contrarian sources
            ),
            bridge_config=CrossDomainBridgeConfig(
                domain_distance_preference=0.3,  # Prefer nearby domains
                bridge_quality_threshold=0.7,    # High quality threshold
                concept_mapping_strictness="strict",  # Strict mapping requirements
                ontological_depth=2,              # Shallow ontological analysis
                cross_pollination_aggressiveness=0.2,  # Conservative transfer
                domain_expertise_weighting=True   # Weight by expertise
            ),
            validation_config=IntegrationValidationConfig(
                validation_strictness="strict",       # Strict validation
                quality_gate_threshold=0.8,          # High quality requirement
                cross_system_consistency_weight=0.9,  # High consistency requirement
                validation_timeout_seconds=600,      # Allow thorough validation
                benchmark_comparison_enabled=True,   # Enable benchmarking
                impossible_problem_testing=False     # No impossible problems
            ),
            reasoning_engine_config=ReasoningEngineConfig(
                # System 1 (Creative) - Conservative parameters
                analogical_creativity=0.3,           # Low creativity for established approaches
                deductive_speculation=0.2,           # Minimal speculation
                inductive_boldness=0.2,              # Conservative induction
                abductive_novelty=0.3,               # Low novelty seeking
                counterfactual_extremeness=0.2,      # Minimal counterfactual exploration
                probabilistic_tail_exploration=0.2,  # Focus on high-probability outcomes
                causal_mechanism_novelty=0.3,        # Conservative causal reasoning
                # System 2 (Validation) - Strict parameters  
                validation_strictness=0.9,           # Maximum validation rigor
                evidence_requirement=0.9,            # High evidence standards
                logical_rigor=0.9,                   # Maximum logical consistency
                consistency_enforcement=0.9,         # Strict consistency enforcement
                contradiction_tolerance=0.1,         # Minimal contradiction tolerance
                # Cross-System - Conservative balance
                creative_validation_balance=0.2,     # Heavy emphasis on validation
                system_transition_threshold=0.8,     # High threshold for creative ideas
                reasoning_depth_multiplier=1.2       # Thorough but not excessive reasoning
            )
        )
        
        # BALANCED MODE (Business Innovation)
        modes[BreakthroughMode.BALANCED] = BreakthroughModeConfig(
            mode=BreakthroughMode.BALANCED,
            name="Balanced", 
            description="Mix of proven approaches with moderate innovation",
            use_cases=[
                "Business strategy and competitive analysis",
                "Product development and market research",
                "Technology adoption decisions",
                "Investment analysis and due diligence",
                "Academic research with practical applications"
            ],
            candidate_distribution=CandidateTypeDistribution(
                # 60% conventional, 40% breakthrough
                synthesis=0.20,
                methodological=0.15,
                empirical=0.15,
                applied=0.10,
                contrarian=0.15,
                cross_domain_transplant=0.15,
                assumption_flip=0.05,
                speculative_moonshot=0.03,
                historical_analogy=0.02
            ),
            confidence_threshold=0.6,
            novelty_weight=0.4,
            consensus_weight=0.6,
            assumption_challenging_enabled=True,
            complexity_multiplier=1.0,  # Standard complexity
            quality_tier="standard",
            citation_preferences="balanced_impact",
            explanation_depth="standard",
            # Enhanced system configurations for Balanced mode
            analogical_config=AnalogicalReasoningConfig(
                max_chain_depth=4,              # Standard chain length
                consistency_threshold=0.6,      # Moderate consistency requirement
                similarity_threshold=0.7,       # Standard similarity requirement
                breakthrough_threshold=0.6,     # Balanced breakthrough seeking
                cross_domain_aggressiveness=0.5,  # Moderate domain bridging
                pattern_recognition_depth="moderate"
            ),
            contrarian_config=ContrarianAnalysisConfig(
                consensus_sensitivity=0.7,      # Standard consensus detection
                contradiction_tolerance=0.6,    # Moderate contradiction tolerance
                credibility_filter="moderate",  # Moderate credibility requirements
                historical_depth=5,             # Standard historical depth
                breakthrough_potential_weight=0.6,  # Balanced breakthrough seeking
                contrarian_source_preference="balanced"  # Balanced source preference
            ),
            bridge_config=CrossDomainBridgeConfig(
                domain_distance_preference=0.6,  # Moderate domain distance
                bridge_quality_threshold=0.5,    # Moderate quality threshold
                concept_mapping_strictness="moderate",  # Moderate mapping requirements
                ontological_depth=3,              # Standard ontological analysis
                cross_pollination_aggressiveness=0.5,  # Moderate transfer
                domain_expertise_weighting=True   # Weight by expertise
            ),
            validation_config=IntegrationValidationConfig(
                validation_strictness="moderate",     # Moderate validation
                quality_gate_threshold=0.6,          # Standard quality requirement
                cross_system_consistency_weight=0.7,  # Standard consistency requirement
                validation_timeout_seconds=300,      # Standard validation time
                benchmark_comparison_enabled=True,   # Enable benchmarking
                impossible_problem_testing=False     # No impossible problems
            ),
            reasoning_engine_config=ReasoningEngineConfig(
                # System 1 (Creative) - Balanced parameters
                analogical_creativity=0.5,           # Moderate creativity
                deductive_speculation=0.3,           # Conservative speculation
                inductive_boldness=0.4,              # Moderate inductive reasoning
                abductive_novelty=0.5,               # Balanced novelty seeking
                counterfactual_extremeness=0.3,      # Moderate counterfactual exploration
                probabilistic_tail_exploration=0.4,  # Balanced probability exploration
                causal_mechanism_novelty=0.5,        # Moderate causal innovation
                # System 2 (Validation) - Balanced parameters
                validation_strictness=0.7,           # Standard validation rigor
                evidence_requirement=0.6,            # Moderate evidence standards
                logical_rigor=0.8,                   # Good logical consistency
                consistency_enforcement=0.7,         # Standard consistency enforcement
                contradiction_tolerance=0.3,         # Moderate contradiction tolerance
                # Cross-System - Balanced approach
                creative_validation_balance=0.5,     # Equal emphasis on creativity and validation
                system_transition_threshold=0.6,     # Standard threshold for creative ideas
                reasoning_depth_multiplier=1.0       # Standard reasoning depth
            )
        )
        
        # CREATIVE MODE (R&D Exploration)
        modes[BreakthroughMode.CREATIVE] = BreakthroughModeConfig(
            mode=BreakthroughMode.CREATIVE,
            name="Creative",
            description="Explore novel possibilities and innovative approaches",
            use_cases=[
                "Research and development initiatives", 
                "Innovation workshops and brainstorming",
                "Startup strategy and disruption analysis",
                "Creative problem solving for complex challenges",
                "Technology foresighting and trend analysis"
            ],
            candidate_distribution=CandidateTypeDistribution(
                # 30% conventional, 70% breakthrough
                synthesis=0.10,
                methodological=0.10,
                empirical=0.05,
                applied=0.05,
                contrarian=0.25,
                cross_domain_transplant=0.20,
                assumption_flip=0.15,
                speculative_moonshot=0.07,
                historical_analogy=0.03
            ),
            confidence_threshold=0.4,
            novelty_weight=0.7,
            consensus_weight=0.3,
            assumption_challenging_enabled=True,
            wild_hypothesis_enabled=True,
            complexity_multiplier=1.3,  # More complex creative reasoning
            quality_tier="premium",
            citation_preferences="novel_and_emerging",
            explanation_depth="creative_with_analogies",
            # Enhanced system configurations for Creative mode
            analogical_config=AnalogicalReasoningConfig(
                max_chain_depth=5,              # Creative: longer chains
                consistency_threshold=0.5,      # Lower consistency for creativity
                similarity_threshold=0.6,       # Lower similarity for creativity
                breakthrough_threshold=0.7,     # Higher breakthrough seeking
                cross_domain_aggressiveness=0.7,  # Aggressive domain bridging
                pattern_recognition_depth="deep"
            ),
            contrarian_config=ContrarianAnalysisConfig(
                consensus_sensitivity=0.5,      # Less sensitive to consensus
                contradiction_tolerance=0.7,    # High tolerance for contradictions
                credibility_filter="permissive",  # Permissive credibility requirements
                historical_depth=3,             # Focus on recent contrarian sources
                breakthrough_potential_weight=0.8,  # High breakthrough seeking
                contrarian_source_preference="recent"  # Prefer recent contrarian sources
            ),
            bridge_config=CrossDomainBridgeConfig(
                domain_distance_preference=0.8,  # Prefer distant domains
                bridge_quality_threshold=0.4,    # Lower quality threshold for creativity
                concept_mapping_strictness="flexible",  # Flexible mapping requirements
                ontological_depth=4,              # Deeper ontological analysis
                cross_pollination_aggressiveness=0.7,  # Aggressive transfer
                domain_expertise_weighting=False  # Don't weight by expertise
            ),
            validation_config=IntegrationValidationConfig(
                validation_strictness="permissive",   # Permissive validation
                quality_gate_threshold=0.4,          # Lower quality requirement
                cross_system_consistency_weight=0.5,  # Lower consistency requirement
                validation_timeout_seconds=180,      # Shorter validation time
                benchmark_comparison_enabled=True,   # Enable benchmarking
                impossible_problem_testing=False     # No impossible problems
            ),
            reasoning_engine_config=ReasoningEngineConfig(
                # System 1 (Creative) - High creativity parameters
                analogical_creativity=0.7,           # High creativity for novel connections
                deductive_speculation=0.5,           # Moderate speculation
                inductive_boldness=0.6,              # Bold inductive leaps
                abductive_novelty=0.7,               # High novelty seeking
                counterfactual_extremeness=0.5,      # Moderate counterfactual exploration
                probabilistic_tail_exploration=0.6,  # Explore low-probability high-impact scenarios
                causal_mechanism_novelty=0.7,        # Novel causal mechanisms
                # System 2 (Validation) - Relaxed parameters for creativity
                validation_strictness=0.6,           # Relaxed validation to allow creativity
                evidence_requirement=0.5,            # Lower evidence standards
                logical_rigor=0.7,                   # Good but not excessive logical consistency
                consistency_enforcement=0.5,         # Relaxed consistency for creative ideas
                contradiction_tolerance=0.7,         # High tolerance for creative contradictions
                # Cross-System - Creative emphasis
                creative_validation_balance=0.7,     # Heavy emphasis on creativity
                system_transition_threshold=0.4,     # Low threshold for creative ideas
                reasoning_depth_multiplier=1.3       # Deeper reasoning for creative exploration
            )
        )
        
        # REVOLUTIONARY MODE (Moonshot Projects)
        modes[BreakthroughMode.REVOLUTIONARY] = BreakthroughModeConfig(
            mode=BreakthroughMode.REVOLUTIONARY,
            name="Revolutionary",
            description="Challenge everything and explore radical possibilities",
            use_cases=[
                "Moonshot and breakthrough innovation projects",
                "Paradigm shift research and analysis",
                "Venture capital and disruptive technology assessment",
                "Blue-sky research and theoretical exploration",
                "Crisis response and unconventional solution generation"
            ],
            candidate_distribution=CandidateTypeDistribution(
                # 10% conventional, 90% breakthrough  
                synthesis=0.05,
                methodological=0.03,
                empirical=0.02,
                contrarian=0.30,
                cross_domain_transplant=0.25,
                assumption_flip=0.20,
                speculative_moonshot=0.10,
                historical_analogy=0.05
            ),
            confidence_threshold=0.2,
            novelty_weight=0.9,
            consensus_weight=0.1,
            assumption_challenging_enabled=True,
            wild_hypothesis_enabled=True,
            impossibility_exploration_enabled=True,
            complexity_multiplier=1.8,  # Maximum complexity and creativity
            quality_tier="excellence",
            citation_preferences="contrarian_and_speculative",
            explanation_depth="visionary_with_scenarios",
            # Enhanced system configurations for Revolutionary mode
            analogical_config=AnalogicalReasoningConfig(
                max_chain_depth=6,              # Revolutionary: maximum chains
                consistency_threshold=0.3,      # Very low consistency for maximum creativity
                similarity_threshold=0.4,       # Very low similarity for breakthrough connections
                breakthrough_threshold=0.9,     # Maximum breakthrough seeking
                cross_domain_aggressiveness=0.9,  # Maximum domain bridging
                pattern_recognition_depth="deep"
            ),
            contrarian_config=ContrarianAnalysisConfig(
                consensus_sensitivity=0.3,      # Very low sensitivity to consensus
                contradiction_tolerance=0.8,    # Maximum tolerance for contradictions
                credibility_filter="speculative",  # Accept speculative sources
                historical_depth=2,             # Focus on most recent contrarian sources
                breakthrough_potential_weight=0.9,  # Maximum breakthrough seeking
                contrarian_source_preference="recent"  # Prefer cutting-edge contrarian sources
            ),
            bridge_config=CrossDomainBridgeConfig(
                domain_distance_preference=0.9,  # Prefer maximally distant domains
                bridge_quality_threshold=0.3,    # Very low quality threshold for maximum creativity
                concept_mapping_strictness="flexible",  # Maximum flexibility
                ontological_depth=5,              # Maximum ontological analysis
                cross_pollination_aggressiveness=0.9,  # Maximum transfer aggressiveness
                domain_expertise_weighting=False  # Ignore domain expertise constraints
            ),
            validation_config=IntegrationValidationConfig(
                validation_strictness="permissive",   # Very permissive validation
                quality_gate_threshold=0.3,          # Very low quality requirement
                cross_system_consistency_weight=0.3,  # Minimal consistency requirement
                validation_timeout_seconds=120,      # Fastest validation time
                benchmark_comparison_enabled=True,   # Enable benchmarking
                impossible_problem_testing=True     # Enable impossible problem testing
            ),
            reasoning_engine_config=ReasoningEngineConfig(
                # System 1 (Creative) - Maximum creativity parameters
                analogical_creativity=0.9,           # Maximum creativity for breakthrough connections
                deductive_speculation=0.8,           # High speculation
                inductive_boldness=0.9,              # Maximum inductive boldness
                abductive_novelty=0.9,               # Maximum novelty seeking
                counterfactual_extremeness=0.9,      # Extreme counterfactual exploration
                probabilistic_tail_exploration=0.9,  # Maximum tail exploration for breakthroughs
                causal_mechanism_novelty=0.9,        # Revolutionary causal mechanisms
                # System 2 (Validation) - Minimal constraints for maximum creativity
                validation_strictness=0.4,           # Minimal validation to preserve breakthroughs
                evidence_requirement=0.3,            # Minimal evidence requirements
                logical_rigor=0.5,                   # Relaxed logical consistency
                consistency_enforcement=0.3,         # Minimal consistency enforcement
                contradiction_tolerance=0.8,         # Maximum tolerance for contradictions
                # Cross-System - Revolutionary emphasis
                creative_validation_balance=0.9,     # Maximum emphasis on creativity
                system_transition_threshold=0.2,     # Very low threshold for creative ideas
                reasoning_depth_multiplier=1.8       # Maximum reasoning depth for breakthroughs
            )
        )
        
        return modes
    
    def _initialize_context_suggestions(self) -> Dict[str, BreakthroughMode]:
        """Initialize context-aware mode suggestions based on query analysis"""
        return {
            # Conservative contexts
            "medical": BreakthroughMode.CONSERVATIVE,
            "safety": BreakthroughMode.CONSERVATIVE,
            "clinical": BreakthroughMode.CONSERVATIVE,
            "regulatory": BreakthroughMode.CONSERVATIVE,
            "compliance": BreakthroughMode.CONSERVATIVE,
            "legal": BreakthroughMode.CONSERVATIVE,
            "financial": BreakthroughMode.CONSERVATIVE,
            
            # Balanced contexts  
            "business": BreakthroughMode.BALANCED,
            "strategy": BreakthroughMode.BALANCED,
            "market": BreakthroughMode.BALANCED,
            "product": BreakthroughMode.BALANCED,
            "investment": BreakthroughMode.BALANCED,
            "academic": BreakthroughMode.BALANCED,
            
            # Creative contexts
            "innovation": BreakthroughMode.CREATIVE,
            "research": BreakthroughMode.CREATIVE,
            "development": BreakthroughMode.CREATIVE,
            "startup": BreakthroughMode.CREATIVE,
            "creative": BreakthroughMode.CREATIVE,
            "brainstorm": BreakthroughMode.CREATIVE,
            
            # Revolutionary contexts
            "moonshot": BreakthroughMode.REVOLUTIONARY,
            "breakthrough": BreakthroughMode.REVOLUTIONARY,
            "disruptive": BreakthroughMode.REVOLUTIONARY,
            "paradigm": BreakthroughMode.REVOLUTIONARY,
            "revolutionary": BreakthroughMode.REVOLUTIONARY,
            "impossible": BreakthroughMode.REVOLUTIONARY,
            "radical": BreakthroughMode.REVOLUTIONARY
        }
    
    def get_mode_config(self, mode: BreakthroughMode) -> BreakthroughModeConfig:
        """Get configuration for a specific breakthrough mode"""
        if mode not in self.modes:
            logger.warning(f"Unknown breakthrough mode: {mode}, defaulting to BALANCED")
            return self.modes[BreakthroughMode.BALANCED]
        return self.modes[mode]
    
    def suggest_mode_from_query(self, query: str) -> BreakthroughMode:
        """Suggest appropriate breakthrough mode based on query analysis"""
        query_lower = query.lower()
        
        # Check for context keywords
        for keyword, suggested_mode in self.context_suggestions.items():
            if keyword in query_lower:
                logger.info(f"Suggested {suggested_mode.value} mode based on keyword '{keyword}' in query")
                return suggested_mode
        
        # Default suggestion logic
        if any(word in query_lower for word in ["safe", "proven", "established", "standard"]):
            return BreakthroughMode.CONSERVATIVE
        elif any(word in query_lower for word in ["novel", "innovative", "creative", "new"]):
            return BreakthroughMode.CREATIVE
        elif any(word in query_lower for word in ["wild", "crazy", "impossible", "radical"]):
            return BreakthroughMode.REVOLUTIONARY
        else:
            return BreakthroughMode.BALANCED
    
    def get_mode_pricing_info(self, mode: BreakthroughMode) -> Dict[str, Any]:
        """Get pricing information for a breakthrough mode"""
        config = self.get_mode_config(mode)
        
        return {
            "complexity_multiplier": config.complexity_multiplier,
            "quality_tier": config.quality_tier,
            "estimated_processing_time": self._estimate_processing_time(config),
            "typical_use_cases": config.use_cases[:3],  # Top 3 use cases
            "breakthrough_intensity": self._calculate_breakthrough_intensity(config)
        }
    
    def _estimate_processing_time(self, config: BreakthroughModeConfig) -> str:
        """Estimate processing time based on mode complexity"""
        base_time = 5  # minutes
        adjusted_time = base_time * config.complexity_multiplier
        
        if adjusted_time < 3:
            return "1-3 minutes"
        elif adjusted_time < 7:
            return "3-7 minutes"  
        elif adjusted_time < 15:
            return "7-15 minutes"
        else:
            return "15-30 minutes"
    
    def _calculate_breakthrough_intensity(self, config: BreakthroughModeConfig) -> str:
        """Calculate breakthrough intensity description"""
        dist = config.candidate_distribution
        breakthrough_percentage = (
            dist.contrarian + 
            dist.cross_domain_transplant + 
            dist.assumption_flip + 
            dist.speculative_moonshot + 
            dist.historical_analogy
        ) * 100
        
        if breakthrough_percentage < 10:
            return "Minimal (focus on established approaches)"
        elif breakthrough_percentage < 30:
            return "Low (mostly proven with some innovation)"
        elif breakthrough_percentage < 60:
            return "Moderate (balanced innovation and proven approaches)"
        elif breakthrough_percentage < 80:
            return "High (significant innovation and creative approaches)"
        else:
            return "Maximum (radical innovation and paradigm-shifting approaches)"
    
    def create_reasoning_context(self, mode: BreakthroughMode, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced reasoning context with breakthrough mode configuration"""
        config = self.get_mode_config(mode)
        
        # Merge mode configuration with base context
        enhanced_context = base_context.copy()
        enhanced_context.update({
            "breakthrough_mode": mode.value,
            "breakthrough_config": {
                "candidate_distribution": config.candidate_distribution.__dict__,
                "confidence_threshold": config.confidence_threshold,
                "novelty_weight": config.novelty_weight,
                "consensus_weight": config.consensus_weight,
                "assumption_challenging_enabled": config.assumption_challenging_enabled,
                "wild_hypothesis_enabled": config.wild_hypothesis_enabled,
                "impossibility_exploration_enabled": config.impossibility_exploration_enabled,
                "citation_preferences": config.citation_preferences,
                "explanation_depth": config.explanation_depth
            },
            "complexity_multiplier": config.complexity_multiplier,
            "quality_tier": config.quality_tier,
            # Enhanced system configurations
            "enhanced_config": self.get_enhanced_config(mode)
        })
        
        return enhanced_context
    
    def get_all_modes_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available breakthrough modes"""
        modes_info = {}
        
        for mode_enum, config in self.modes.items():
            modes_info[mode_enum.value] = {
                "name": config.name,
                "description": config.description,
                "use_cases": config.use_cases,
                "complexity_multiplier": config.complexity_multiplier,
                "quality_tier": config.quality_tier,
                "confidence_threshold": config.confidence_threshold,
                "breakthrough_intensity": self._calculate_breakthrough_intensity(config),
                "estimated_time": self._estimate_processing_time(config),
                "special_features": {
                    "assumption_challenging": config.assumption_challenging_enabled,
                    "wild_hypothesis": config.wild_hypothesis_enabled,
                    "impossibility_exploration": config.impossibility_exploration_enabled
                }
            }
        
        return modes_info
    
    def get_enhanced_config(self, mode: BreakthroughMode) -> Dict[str, Any]:
        """Get enhanced system configurations for a breakthrough mode"""
        config = self.get_mode_config(mode)
        
        return {
            "analogical_reasoning": {
                "max_chain_depth": config.analogical_config.max_chain_depth,
                "consistency_threshold": config.analogical_config.consistency_threshold,
                "similarity_threshold": config.analogical_config.similarity_threshold,
                "breakthrough_threshold": config.analogical_config.breakthrough_threshold,
                "cross_domain_aggressiveness": config.analogical_config.cross_domain_aggressiveness,
                "pattern_recognition_depth": config.analogical_config.pattern_recognition_depth
            },
            "contrarian_analysis": {
                "consensus_sensitivity": config.contrarian_config.consensus_sensitivity,
                "contradiction_tolerance": config.contrarian_config.contradiction_tolerance,
                "credibility_filter": config.contrarian_config.credibility_filter,
                "historical_depth": config.contrarian_config.historical_depth,
                "breakthrough_potential_weight": config.contrarian_config.breakthrough_potential_weight,
                "contrarian_source_preference": config.contrarian_config.contrarian_source_preference
            },
            "cross_domain_bridging": {
                "domain_distance_preference": config.bridge_config.domain_distance_preference,
                "bridge_quality_threshold": config.bridge_config.bridge_quality_threshold,
                "concept_mapping_strictness": config.bridge_config.concept_mapping_strictness,
                "ontological_depth": config.bridge_config.ontological_depth,
                "cross_pollination_aggressiveness": config.bridge_config.cross_pollination_aggressiveness,
                "domain_expertise_weighting": config.bridge_config.domain_expertise_weighting
            },
            "integration_validation": {
                "validation_strictness": config.validation_config.validation_strictness,
                "quality_gate_threshold": config.validation_config.quality_gate_threshold,
                "cross_system_consistency_weight": config.validation_config.cross_system_consistency_weight,
                "validation_timeout_seconds": config.validation_config.validation_timeout_seconds,
                "benchmark_comparison_enabled": config.validation_config.benchmark_comparison_enabled,
                "impossible_problem_testing": config.validation_config.impossible_problem_testing
            },
            "reasoning_engines": {
                # System 1 (Creative Generation) Parameters
                "analogical_creativity": config.reasoning_engine_config.analogical_creativity,
                "deductive_speculation": config.reasoning_engine_config.deductive_speculation,
                "inductive_boldness": config.reasoning_engine_config.inductive_boldness,
                "abductive_novelty": config.reasoning_engine_config.abductive_novelty,
                "counterfactual_extremeness": config.reasoning_engine_config.counterfactual_extremeness,
                "probabilistic_tail_exploration": config.reasoning_engine_config.probabilistic_tail_exploration,
                "causal_mechanism_novelty": config.reasoning_engine_config.causal_mechanism_novelty,
                # System 2 (Validation) Parameters
                "validation_strictness": config.reasoning_engine_config.validation_strictness,
                "evidence_requirement": config.reasoning_engine_config.evidence_requirement,
                "logical_rigor": config.reasoning_engine_config.logical_rigor,
                "consistency_enforcement": config.reasoning_engine_config.consistency_enforcement,
                "contradiction_tolerance": config.reasoning_engine_config.contradiction_tolerance,
                # Cross-System Parameters
                "creative_validation_balance": config.reasoning_engine_config.creative_validation_balance,
                "system_transition_threshold": config.reasoning_engine_config.system_transition_threshold,
                "reasoning_depth_multiplier": config.reasoning_engine_config.reasoning_depth_multiplier
            }
        }
    
    def create_custom_mode(self, 
                          base_mode: BreakthroughMode,
                          analogical_overrides: Optional[Dict[str, Any]] = None,
                          contrarian_overrides: Optional[Dict[str, Any]] = None,
                          bridge_overrides: Optional[Dict[str, Any]] = None,
                          validation_overrides: Optional[Dict[str, Any]] = None,
                          reasoning_engine_overrides: Optional[Dict[str, Any]] = None) -> BreakthroughModeConfig:
        """Create a custom breakthrough mode with user-specified overrides"""
        
        base_config = self.get_mode_config(base_mode)
        
        # Create new enhanced configs with overrides
        analogical_config = AnalogicalReasoningConfig(**base_config.analogical_config.__dict__)
        if analogical_overrides:
            for key, value in analogical_overrides.items():
                if hasattr(analogical_config, key):
                    setattr(analogical_config, key, value)
        
        contrarian_config = ContrarianAnalysisConfig(**base_config.contrarian_config.__dict__)
        if contrarian_overrides:
            for key, value in contrarian_overrides.items():
                if hasattr(contrarian_config, key):
                    setattr(contrarian_config, key, value)
        
        bridge_config = CrossDomainBridgeConfig(**base_config.bridge_config.__dict__)
        if bridge_overrides:
            for key, value in bridge_overrides.items():
                if hasattr(bridge_config, key):
                    setattr(bridge_config, key, value)
        
        validation_config = IntegrationValidationConfig(**base_config.validation_config.__dict__)
        if validation_overrides:
            for key, value in validation_overrides.items():
                if hasattr(validation_config, key):
                    setattr(validation_config, key, value)
        
        reasoning_engine_config = ReasoningEngineConfig(**base_config.reasoning_engine_config.__dict__)
        if reasoning_engine_overrides:
            for key, value in reasoning_engine_overrides.items():
                if hasattr(reasoning_engine_config, key):
                    setattr(reasoning_engine_config, key, value)
        
        # Create custom config with overrides
        custom_config = BreakthroughModeConfig(
            mode=BreakthroughMode.CUSTOM,
            name=f"Custom ({base_config.name})",
            description=f"Customized version of {base_config.name} mode",
            use_cases=base_config.use_cases.copy(),
            candidate_distribution=base_config.candidate_distribution,
            confidence_threshold=base_config.confidence_threshold,
            novelty_weight=base_config.novelty_weight,
            consensus_weight=base_config.consensus_weight,
            assumption_challenging_enabled=base_config.assumption_challenging_enabled,
            wild_hypothesis_enabled=base_config.wild_hypothesis_enabled,
            impossibility_exploration_enabled=base_config.impossibility_exploration_enabled,
            complexity_multiplier=base_config.complexity_multiplier,
            quality_tier=base_config.quality_tier,
            citation_preferences=base_config.citation_preferences,
            explanation_depth=base_config.explanation_depth,
            analogical_config=analogical_config,
            contrarian_config=contrarian_config,
            bridge_config=bridge_config,
            validation_config=validation_config,
            reasoning_engine_config=reasoning_engine_config
        )
        
        return custom_config
    
    def validate_enhanced_config(self, config: BreakthroughModeConfig) -> Dict[str, List[str]]:
        """Validate enhanced configuration parameters and return any warnings/errors"""
        warnings = []
        errors = []
        
        # Validate analogical reasoning config
        if config.analogical_config.max_chain_depth < 1 or config.analogical_config.max_chain_depth > 6:
            errors.append("Analogical chain depth must be between 1 and 6")
        if not (0.3 <= config.analogical_config.consistency_threshold <= 0.9):
            warnings.append("Analogical consistency threshold outside recommended range (0.3-0.9)")
        
        # Validate contrarian analysis config
        if not (0.3 <= config.contrarian_config.consensus_sensitivity <= 0.9):
            warnings.append("Consensus sensitivity outside recommended range (0.3-0.9)")
        if config.contrarian_config.historical_depth < 1 or config.contrarian_config.historical_depth > 20:
            errors.append("Historical depth must be between 1 and 20 years")
        
        # Validate cross-domain bridge config
        if not (1 <= config.bridge_config.ontological_depth <= 5):
            errors.append("Ontological depth must be between 1 and 5")
        
        # Validate integration config
        if not (60 <= config.validation_config.validation_timeout_seconds <= 600):
            warnings.append("Validation timeout outside recommended range (60-600 seconds)")
        
        # Validate reasoning engine config
        if not (0.2 <= config.reasoning_engine_config.analogical_creativity <= 0.9):
            warnings.append("Analogical creativity outside recommended range (0.2-0.9)")
        if not (0.1 <= config.reasoning_engine_config.deductive_speculation <= 0.8):
            warnings.append("Deductive speculation outside recommended range (0.1-0.8)")
        if not (0.3 <= config.reasoning_engine_config.validation_strictness <= 0.9):
            warnings.append("Validation strictness outside recommended range (0.3-0.9)")
        if not (0.5 <= config.reasoning_engine_config.reasoning_depth_multiplier <= 2.0):
            warnings.append("Reasoning depth multiplier outside recommended range (0.5-2.0)")
        
        return {
            "warnings": warnings,
            "errors": errors
        }

# Global instance for easy access
breakthrough_mode_manager = BreakthroughModeManager()

def get_breakthrough_mode_config(mode: BreakthroughMode) -> BreakthroughModeConfig:
    """Convenience function to get breakthrough mode configuration"""
    return breakthrough_mode_manager.get_mode_config(mode)

def suggest_breakthrough_mode(query: str) -> BreakthroughMode:
    """Convenience function to suggest breakthrough mode from query"""
    return breakthrough_mode_manager.suggest_mode_from_query(query)
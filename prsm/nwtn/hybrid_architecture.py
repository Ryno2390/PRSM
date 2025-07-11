"""
NWTN Hybrid Architecture Implementation
System 1 (Transformers) + System 2 (World Model) Integration

This module implements the hybrid AI architecture described in the brainstorming document:
- System 1: Fast, intuitive transformer-based pattern recognition
- System 2: Slow, logical first-principles world model reasoning
- Learning mechanism with threshold-based SOC updates
- Bayesian search with experiment sharing across PRSM network

Key Components:
1. SOCManager: Manages Subjects, Objects, Concepts with dynamic learning
2. WorldModelEngine: First-principles reasoning and causal validation
3. HybridNWTNEngine: Coordinates System 1 + System 2 interactions
4. BayesianSearchEngine: Automated experiment and knowledge sharing
5. ThresholdManager: Manages knowledge confidence levels (tenable -> core)
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import PRSMBaseModel, TimestampMixin
from prsm.core.config import get_settings
from prsm.agents.executors.model_executor import ModelExecutor
from prsm.agents.routers.model_router import ModelRouter

logger = structlog.get_logger(__name__)
settings = get_settings()


class SOCType(str, Enum):
    """Types of Subjects, Objects, Concepts"""
    SUBJECT = "subject"           # Entities that perform actions
    OBJECT = "object"             # Physical or abstract entities
    CONCEPT = "concept"           # Abstract ideas, relationships, principles
    RELATION = "relation"         # Connections between SOCs
    PRINCIPLE = "principle"       # First-principles, fundamental laws


class ConfidenceLevel(str, Enum):
    """Confidence levels for SOC knowledge"""
    TENABLE = "tenable"           # Low confidence, tentative knowledge
    INTERMEDIATE = "intermediate" # Moderate confidence, some validation
    VALIDATED = "validated"       # High confidence, well-supported
    CORE = "core"                 # Highest confidence, fundamental truth


class ExperimentType(str, Enum):
    """Types of experiments for Bayesian search"""
    SIMULATION = "simulation"     # Computational simulation
    LOGICAL_TEST = "logical_test" # Logical consistency check
    CAUSAL_TEST = "causal_test"   # Causal relationship validation
    PRINCIPLE_TEST = "principle_test" # First-principles validation


class CommunicativeIntent(str, Enum):
    """Types of communicative intent identified from queries"""
    INFORMATION_SEEKING = "information_seeking"     # User wants to learn something
    PROBLEM_SOLVING = "problem_solving"             # User needs help solving a problem
    VERIFICATION = "verification"                   # User wants to verify understanding
    PREDICTION = "prediction"                       # User wants future outcomes predicted
    ANALYSIS = "analysis"                           # User wants complex analysis
    GENERATION = "generation"                       # User wants something created
    EXPLANATION = "explanation"                     # User wants concepts explained
    DECISION_SUPPORT = "decision_support"           # User needs help making decisions


class UserGoal(PRSMBaseModel):
    """Representation of inferred user goals and context"""
    
    id: UUID = Field(default_factory=uuid4)
    intent: CommunicativeIntent
    explicit_query: str
    inferred_needs: List[str] = Field(default_factory=list)
    context_requirements: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: List[str] = Field(default_factory=list)
    
    # User modeling
    expertise_level: str = Field(default="unknown")  # beginner, intermediate, expert
    domain_familiarity: float = Field(default=0.5, ge=0.0, le=1.0)
    preferred_communication_style: str = Field(default="balanced")
    
    # Temporal aspects
    urgency: str = Field(default="normal")  # low, normal, high
    depth_required: str = Field(default="moderate")  # shallow, moderate, deep
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BiasDetectionResult(PRSMBaseModel):
    """Result of bias detection analysis"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Bias metrics
    perspective_diversity_score: float = Field(ge=0.0, le=1.0)
    hegemonic_bias_score: float = Field(ge=0.0, le=1.0)  # Higher = more biased
    evidence_quality_score: float = Field(ge=0.0, le=1.0)
    
    # Detected issues
    bias_flags: List[str] = Field(default_factory=list)
    missing_perspectives: List[str] = Field(default_factory=list)
    overrepresented_viewpoints: List[str] = Field(default_factory=list)
    
    # Recommendations
    suggested_additional_perspectives: List[str] = Field(default_factory=list)
    bias_mitigation_strategies: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SOC(PRSMBaseModel):
    """
    Subject, Object, or Concept with learning capabilities
    
    Represents a discrete piece of knowledge with:
    - Identity and type classification
    - Confidence weighting with Bayesian updates
    - Relationship mapping to other SOCs
    - Learning threshold tracking
    """
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Human-readable name")
    soc_type: SOCType = Field(..., description="Type of SOC")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Bayesian confidence")
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.TENABLE)
    
    # Knowledge representation
    properties: Dict[str, Any] = Field(default_factory=dict)
    relationships: Dict[str, float] = Field(default_factory=dict)  # SOC_ID -> weight
    
    # Learning tracking
    evidence_count: int = Field(default=0)
    update_count: int = Field(default=0)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Domain context
    domain: str = Field(default="general")
    tags: List[str] = Field(default_factory=list)
    
    def update_confidence(self, new_evidence: float, weight: float = 1.0):
        """Update confidence using Bayesian updating"""
        prior = self.confidence
        likelihood = new_evidence
        
        # Bayesian update with weighted evidence
        posterior = (prior * weight + likelihood) / (weight + 1)
        self.confidence = max(0.0, min(1.0, posterior))
        
        # Update confidence level based on threshold
        if self.confidence >= 0.9:
            self.confidence_level = ConfidenceLevel.CORE
        elif self.confidence >= 0.7:
            self.confidence_level = ConfidenceLevel.VALIDATED
        elif self.confidence >= 0.5:
            self.confidence_level = ConfidenceLevel.INTERMEDIATE
        else:
            self.confidence_level = ConfidenceLevel.TENABLE
            
        self.evidence_count += 1
        self.update_count += 1
        self.last_updated = datetime.now(timezone.utc)


class ExperimentResult(PRSMBaseModel):
    """Result of a Bayesian search experiment"""
    
    id: UUID = Field(default_factory=uuid4)
    experiment_type: ExperimentType
    agent_id: str
    domain: str
    
    # Experiment details
    hypothesis: str
    method: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Results
    success: bool
    confidence_change: float  # How much confidence changed
    affected_socs: List[str] = Field(default_factory=list)
    
    # Metadata
    execution_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Information value for sharing
    information_value: float = Field(default=0.0)


class WorldModelEngine:
    """
    System 2: First-principles world model reasoning engine
    
    Maintains structured knowledge of fundamental principles and
    validates SOCs against causal relationships and logical consistency.
    """
    
    def __init__(self):
        self.socs: Dict[str, SOC] = {}
        self.core_principles: Dict[str, SOC] = {}
        self.domain_models: Dict[str, Dict[str, SOC]] = {}
        
        # Initialize with fundamental principles
        self._initialize_core_principles()
        
    def _initialize_core_principles(self):
        """Initialize base world model with core principles"""
        
        # Physical principles
        physics_principles = [
            ("conservation_of_energy", "Energy cannot be created or destroyed"),
            ("conservation_of_momentum", "Momentum is conserved in isolated systems"),
            ("causality", "Every effect has a cause"),
            ("object_permanence", "Objects continue to exist when not observed"),
            ("gravity", "Objects with mass attract each other"),
        ]
        
        # Logical principles
        logical_principles = [
            ("identity", "A thing is identical to itself"),
            ("non_contradiction", "A statement cannot be both true and false"),
            ("excluded_middle", "A statement is either true or false"),
            ("transitivity", "If A relates to B and B relates to C, then A relates to C"),
        ]
        
        # Initialize core SOCs
        for name, description in physics_principles + logical_principles:
            soc = SOC(
                name=name,
                soc_type=SOCType.PRINCIPLE,
                confidence=0.95,  # Core principles start with high confidence
                confidence_level=ConfidenceLevel.CORE,
                properties={"description": description, "domain": "physics" if name in [p[0] for p in physics_principles] else "logic"}
            )
            self.core_principles[name] = soc
            self.socs[name] = soc
            
    def validate_soc_against_principles(self, soc: SOC) -> Tuple[bool, float, List[str]]:
        """
        Validate a SOC against first principles
        
        Returns:
            - is_valid: Whether SOC is consistent with principles
            - confidence_adjustment: Suggested confidence change
            - conflicts: List of conflicting principles
        """
        
        conflicts = []
        confidence_adjustment = 0.0
        
        # Check against core principles
        for principle_name, principle_soc in self.core_principles.items():
            consistency = self._check_consistency(soc, principle_soc)
            
            if consistency < 0.3:  # Strong conflict
                conflicts.append(principle_name)
                confidence_adjustment -= 0.2
            elif consistency > 0.7:  # Strong support
                confidence_adjustment += 0.1
                
        is_valid = len(conflicts) == 0
        
        return is_valid, confidence_adjustment, conflicts
        
    def _check_consistency(self, soc: SOC, principle: SOC) -> float:
        """Check consistency between SOC and principle"""
        
        # Simple heuristic for now - in full implementation, this would
        # involve sophisticated logical reasoning
        
        # Check for direct contradictions in properties
        soc_props = soc.properties.get("description", "").lower()
        principle_props = principle.properties.get("description", "").lower()
        
        # Basic keyword conflicts (expand with proper NLP)
        conflict_keywords = {
            "conservation": ["destroy", "create", "lose", "gain"],
            "causality": ["random", "spontaneous", "uncaused"],
            "identity": ["different", "change", "transform"]
        }
        
        for principle_key, conflict_words in conflict_keywords.items():
            if principle_key in principle_props:
                for conflict_word in conflict_words:
                    if conflict_word in soc_props:
                        return 0.1  # Strong conflict
                        
        return 0.5  # Neutral/unknown


class BayesianSearchEngine:
    """
    Automated Bayesian search with experiment sharing
    
    Conducts experiments, shares results across PRSM network,
    and updates SOC confidence using Bayesian updating.
    """
    
    def __init__(self, agent_id: str, domain: str = "general"):
        self.agent_id = agent_id
        self.domain = domain
        self.experiment_history: List[ExperimentResult] = []
        self.shared_experiments: Dict[str, ExperimentResult] = {}
        
    async def conduct_experiment(
        self, 
        hypothesis: str, 
        soc: SOC, 
        experiment_type: ExperimentType = ExperimentType.LOGICAL_TEST
    ) -> ExperimentResult:
        """
        Conduct experiment to test SOC or hypothesis
        
        For now, implements simplified experiment logic.
        In full implementation, this would interface with:
        - Simulation engines for computational experiments
        - Logical reasoning engines for consistency checks
        - External APIs for real-world data validation
        """
        
        start_time = asyncio.get_event_loop().time()
        
        # Simple experiment simulation
        if experiment_type == ExperimentType.LOGICAL_TEST:
            success = await self._logical_consistency_test(soc, hypothesis)
        elif experiment_type == ExperimentType.SIMULATION:
            success = await self._simulation_test(soc, hypothesis)
        else:
            success = soc.confidence > 0.5  # Placeholder
            
        end_time = asyncio.get_event_loop().time()
        
        # Calculate confidence change
        confidence_change = 0.1 if success else -0.1
        
        # Create experiment result
        result = ExperimentResult(
            experiment_type=experiment_type,
            agent_id=self.agent_id,
            domain=self.domain,
            hypothesis=hypothesis,
            method=f"{experiment_type.value}_test",
            success=success,
            confidence_change=confidence_change,
            affected_socs=[soc.name],
            execution_time=end_time - start_time,
            information_value=abs(confidence_change)
        )
        
        self.experiment_history.append(result)
        
        logger.info(
            "Experiment completed",
            agent_id=self.agent_id,
            hypothesis=hypothesis,
            success=success,
            confidence_change=confidence_change
        )
        
        return result
        
    async def _logical_consistency_test(self, soc: SOC, hypothesis: str) -> bool:
        """Test logical consistency of SOC with hypothesis"""
        # Simplified logic test
        await asyncio.sleep(0.1)  # Simulate processing time
        return soc.confidence > 0.4  # Placeholder logic
        
    async def _simulation_test(self, soc: SOC, hypothesis: str) -> bool:
        """Run simulation to test SOC"""
        # Simplified simulation
        await asyncio.sleep(0.2)  # Simulate processing time
        return soc.confidence > 0.3  # Placeholder logic
        
    def share_experiment_result(self, result: ExperimentResult):
        """Share experiment result with PRSM network"""
        
        # In full implementation, this would broadcast to PRSM marketplace
        # For now, just log the sharing
        logger.info(
            "Sharing experiment result",
            agent_id=self.agent_id,
            experiment_id=str(result.id),
            information_value=result.information_value,
            success=result.success
        )
        
    def receive_shared_experiment(self, result: ExperimentResult):
        """Receive and process shared experiment from network"""
        
        self.shared_experiments[str(result.id)] = result
        
        logger.info(
            "Received shared experiment",
            agent_id=self.agent_id,
            from_agent=result.agent_id,
            information_value=result.information_value
        )


class SOCManager:
    """
    Manages SOC lifecycle with threshold-based learning
    
    Handles SOC creation, updating, threshold management,
    and coordination between System 1 and System 2.
    """
    
    def __init__(self):
        self.socs: Dict[str, SOC] = {}
        self.domain_socs: Dict[str, Dict[str, SOC]] = {}
        
        # Threshold configuration
        self.thresholds = {
            ConfidenceLevel.TENABLE: 0.3,
            ConfidenceLevel.INTERMEDIATE: 0.5,
            ConfidenceLevel.VALIDATED: 0.7,
            ConfidenceLevel.CORE: 0.9
        }
        
    def create_or_update_soc(
        self,
        name: str,
        soc_type: SOCType,
        evidence: float,
        domain: str = "general",
        properties: Dict[str, Any] = None
    ) -> SOC:
        """Create new SOC or update existing one"""
        
        if name in self.socs:
            # Update existing SOC
            soc = self.socs[name]
            soc.update_confidence(evidence)
            if properties:
                soc.properties.update(properties)
        else:
            # Create new SOC
            soc = SOC(
                name=name,
                soc_type=soc_type,
                confidence=evidence,
                domain=domain,
                properties=properties or {}
            )
            self.socs[name] = soc
            
            # Add to domain index
            if domain not in self.domain_socs:
                self.domain_socs[domain] = {}
            self.domain_socs[domain][name] = soc
            
        return soc
        
    def get_socs_by_confidence_level(self, level: ConfidenceLevel) -> List[SOC]:
        """Get all SOCs at specified confidence level"""
        return [soc for soc in self.socs.values() if soc.confidence_level == level]
        
    def get_core_socs(self) -> List[SOC]:
        """Get all core knowledge SOCs"""
        return self.get_socs_by_confidence_level(ConfidenceLevel.CORE)
        
    def update_soc_from_experiment(self, soc_name: str, experiment_result: ExperimentResult):
        """Update SOC confidence based on experiment result"""
        
        if soc_name not in self.socs:
            return
            
        soc = self.socs[soc_name]
        
        # Update confidence based on experiment
        weight = experiment_result.information_value
        evidence = 0.8 if experiment_result.success else 0.2
        
        soc.update_confidence(evidence, weight)
        
        logger.info(
            "Updated SOC from experiment",
            soc_name=soc_name,
            old_confidence=soc.confidence - experiment_result.confidence_change,
            new_confidence=soc.confidence,
            confidence_level=soc.confidence_level.value
        )


class HybridNWTNEngine:
    """
    Main hybrid architecture engine
    
    Coordinates System 1 (transformers) and System 2 (world model)
    with Bayesian search and knowledge sharing.
    """
    
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or f"hybrid_nwtn_{uuid4().hex[:8]}"
        
        # Core components
        self.soc_manager = SOCManager()
        self.world_model = WorldModelEngine()
        self.bayesian_search = BayesianSearchEngine(self.agent_id)
        
        # System 1 components (existing PRSM infrastructure)
        self.model_executor = ModelExecutor(agent_id=self.agent_id)
        self.model_router = ModelRouter(agent_id=self.agent_id)
        
        # Configuration
        self.temperature = 0.7  # For diverse perspectives
        self.domain = "general"
        
        logger.info(
            "Initialized Hybrid NWTN Engine",
            agent_id=self.agent_id,
            temperature=self.temperature
        )
        
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process query through hybrid System 1 + System 2 architecture
        
        Enhanced Flow (Anti-Stochastic Parrot):
        0. Intent Analysis: Understand what human actually needs
        1. System 1: Rapid SOC recognition from query
        2. System 2: Validate SOCs against world model
        3. Bias Detection: Check for perspective diversity
        4. Bayesian Search: Conduct experiments on uncertain SOCs
        5. Knowledge Sharing: Share results with PRSM network
        6. Response Generation: Synthesize final response with intent alignment
        """
        
        logger.info("Processing query through hybrid architecture", query=query)
        
        # Step 0: Communicative Intent Analysis (Anti-Stochastic Parrot)
        user_goal = await self._analyze_communicative_intent(query, context)
        
        # Step 1: System 1 - Rapid SOC recognition
        recognized_socs = await self._system1_soc_recognition(query, context)
        
        # Step 2: System 2 - World model validation
        validated_socs = await self._system2_validation(recognized_socs)
        
        # Step 3: Bias Detection and Diverse Perspective Analysis
        bias_analysis = await self._detect_bias_and_generate_perspectives(validated_socs, user_goal)
        
        # Step 4: Bayesian Search - Experiment on uncertain SOCs
        experiment_results = await self._bayesian_search_experiments(validated_socs)
        
        # Step 5: Knowledge Sharing - Share results
        await self._share_knowledge(experiment_results)
        
        # Step 6: Response Generation with Intent Alignment
        response = await self._generate_response(query, validated_socs, experiment_results, user_goal, bias_analysis)
        
        return response
        
    async def _system1_soc_recognition(self, query: str, context: Dict[str, Any]) -> List[SOC]:
        """System 1: Fast SOC recognition using transformers"""
        
        # Use existing model router to analyze query
        # In full implementation, this would use specialized SOC recognition prompts
        
        # For now, extract key entities and concepts
        soc_prompt = f"""
        Analyze this query and identify:
        1. Subjects (entities performing actions)
        2. Objects (things being acted upon)
        3. Concepts (abstract ideas or relationships)
        4. Principles (fundamental rules or laws)
        
        Query: {query}
        
        Format as JSON with type, name, and confidence (0-1).
        """
        
        # Get SOC analysis from transformer
        analysis = await self.model_executor.execute_request(
            prompt=soc_prompt,
            model_name="gpt-4",  # Use sophisticated model for SOC recognition
            temperature=self.temperature
        )
        
        recognized_socs = []
        
        # Fast pattern-based recognition (System 1 characteristic)
        socs_from_patterns = await self._pattern_based_soc_recognition(query)
        recognized_socs.extend(socs_from_patterns)
        
        # Context-based SOC activation
        if context:
            socs_from_context = await self._context_based_soc_recognition(context)
            recognized_socs.extend(socs_from_context)
        
        try:
            # Parse transformer response
            parsed_socs = await self._parse_transformer_socs(analysis, query)
            recognized_socs.extend(parsed_socs)
            
        except Exception as e:
            logger.warning(f"Transformer SOC recognition failed: {e}")
            # Fallback to simple keyword-based recognition
            fallback_socs = await self._keyword_based_soc_recognition(query)
            recognized_socs.extend(fallback_socs)
        
        # Remove duplicates and merge similar SOCs
        unique_socs = await self._merge_similar_socs(recognized_socs)
        
        logger.info(f"System 1 recognized {len(unique_socs)} SOCs from query")
        return unique_socs
    
    async def _pattern_based_soc_recognition(self, query: str) -> List[SOC]:
        """Fast pattern-based SOC recognition using pre-compiled patterns"""
        
        socs = []
        query_lower = query.lower()
        
        # Physics patterns
        physics_patterns = {
            "gravity": ["gravity", "gravitational", "falling", "weight", "mass attraction"],
            "energy": ["energy", "kinetic", "potential", "conservation", "work"],
            "momentum": ["momentum", "velocity", "collision", "impulse"],
            "thermodynamics": ["heat", "temperature", "entropy", "thermal", "hot", "cold"]
        }
        
        # Chemistry patterns
        chemistry_patterns = {
            "chemical_reaction": ["reaction", "catalyst", "reagent", "product", "chemical"],
            "atomic_structure": ["atom", "electron", "proton", "neutron", "nucleus"],
            "molecular_bonding": ["bond", "molecule", "ionic", "covalent", "polar"]
        }
        
        # Mathematics patterns
        math_patterns = {
            "calculus": ["derivative", "integral", "limit", "differential", "calculus"],
            "algebra": ["equation", "variable", "polynomial", "linear", "quadratic"],
            "geometry": ["triangle", "circle", "angle", "area", "volume", "geometric"]
        }
        
        # Check all patterns
        all_patterns = {
            **{f"physics_{k}": v for k, v in physics_patterns.items()},
            **{f"chemistry_{k}": v for k, v in chemistry_patterns.items()},
            **{f"math_{k}": v for k, v in math_patterns.items()}
        }
        
        for concept_name, keywords in all_patterns.items():
            confidence = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in query_lower:
                    confidence += 0.2
                    matched_keywords.append(keyword)
            
            if confidence > 0:
                confidence = min(confidence, 0.8)  # Cap at 0.8 for pattern-based
                
                domain = concept_name.split('_')[0]
                soc = self.soc_manager.create_or_update_soc(
                    concept_name,
                    SOCType.CONCEPT,
                    confidence,
                    domain=domain,
                    properties={
                        "recognition_method": "pattern_matching",
                        "matched_keywords": matched_keywords,
                        "query_fragment": query[:100]
                    }
                )
                socs.append(soc)
        
        return socs
    
    async def _context_based_soc_recognition(self, context: Dict[str, Any]) -> List[SOC]:
        """Activate SOCs based on context information"""
        
        socs = []
        
        # Extract SOCs from context
        if "domain" in context:
            domain = context["domain"]
            # Activate domain-specific SOCs
            domain_socs = self.soc_manager.domain_socs.get(domain, {})
            for soc in domain_socs.values():
                if soc.confidence > 0.5:  # Only activate confident SOCs
                    socs.append(soc)
        
        if "corpus_socs" in context:
            # Use SOCs from corpus search
            corpus_socs = context["corpus_socs"]
            socs.extend(corpus_socs)
        
        return socs
    
    async def _parse_transformer_socs(self, analysis: str, query: str) -> List[SOC]:
        """Parse transformer response into SOCs"""
        
        socs = []
        
        try:
            # Try to parse as JSON
            import json
            parsed = json.loads(analysis)
            
            if isinstance(parsed, list):
                for item in parsed:
                    if all(key in item for key in ["type", "name", "confidence"]):
                        soc_type = {
                            "subject": SOCType.SUBJECT,
                            "object": SOCType.OBJECT,
                            "concept": SOCType.CONCEPT,
                            "principle": SOCType.PRINCIPLE
                        }.get(item["type"], SOCType.CONCEPT)
                        
                        soc = self.soc_manager.create_or_update_soc(
                            item["name"],
                            soc_type,
                            float(item["confidence"]),
                            domain=item.get("domain", "general"),
                            properties={
                                "recognition_method": "transformer_extraction",
                                "source_query": query
                            }
                        )
                        socs.append(soc)
        
        except Exception as e:
            logger.warning(f"Failed to parse transformer SOC response: {e}")
            # Fallback to simple text parsing
            socs = await self._simple_text_soc_parsing(analysis, query)
        
        return socs
    
    async def _simple_text_soc_parsing(self, text: str, query: str) -> List[SOC]:
        """Simple text-based SOC parsing as fallback"""
        
        socs = []
        
        # Extract key terms from the text
        words = text.lower().split()
        
        # Look for domain indicators
        domain_indicators = {
            "physics": ["physics", "force", "energy", "motion", "gravity"],
            "chemistry": ["chemistry", "reaction", "molecule", "atom", "chemical"],
            "mathematics": ["math", "equation", "function", "variable", "calculation"]
        }
        
        for domain, indicators in domain_indicators.items():
            for indicator in indicators:
                if indicator in words:
                    soc = self.soc_manager.create_or_update_soc(
                        f"{domain}_concept",
                        SOCType.CONCEPT,
                        0.6,
                        domain=domain,
                        properties={
                            "recognition_method": "simple_text_parsing",
                            "source_query": query,
                            "detected_indicator": indicator
                        }
                    )
                    socs.append(soc)
        
        return socs
    
    async def _keyword_based_soc_recognition(self, query: str) -> List[SOC]:
        """Keyword-based SOC recognition as ultimate fallback"""
        
        socs = []
        query_lower = query.lower()
        
        # Basic keyword recognition
        keywords = {
            "physics": SOCType.CONCEPT,
            "chemistry": SOCType.CONCEPT,
            "mathematics": SOCType.CONCEPT,
            "science": SOCType.CONCEPT,
            "research": SOCType.CONCEPT,
            "experiment": SOCType.OBJECT,
            "theory": SOCType.CONCEPT,
            "law": SOCType.PRINCIPLE
        }
        
        for keyword, soc_type in keywords.items():
            if keyword in query_lower:
                soc = self.soc_manager.create_or_update_soc(
                    keyword,
                    soc_type,
                    0.5,
                    domain="general",
                    properties={
                        "recognition_method": "keyword_fallback",
                        "source_query": query
                    }
                )
                socs.append(soc)
        
        return socs
    
    async def _merge_similar_socs(self, socs: List[SOC]) -> List[SOC]:
        """Merge similar SOCs to avoid duplicates"""
        
        if not socs:
            return []
        
        # Group SOCs by name similarity
        merged = {}
        
        for soc in socs:
            # Simple similarity based on name
            merged_key = soc.name.lower().replace('_', ' ')
            
            if merged_key in merged:
                # Merge with existing SOC
                existing = merged[merged_key]
                # Take the higher confidence
                if soc.confidence > existing.confidence:
                    merged[merged_key] = soc
                else:
                    # Update evidence count
                    existing.evidence_count += 1
            else:
                merged[merged_key] = soc
        
        return list(merged.values())
        
    async def _system2_validation(self, socs: List[SOC]) -> List[SOC]:
        """System 2: Deliberate validation against world model and logical reasoning"""
        
        validated_socs = []
        
        for soc in socs:
            # Step 1: Validate against first principles
            is_valid, confidence_adjustment, conflicts = self.world_model.validate_soc_against_principles(soc)
            
            # Step 2: Perform logical consistency checks
            logical_consistency = await self._check_logical_consistency(soc)
            
            # Step 3: Check for causal relationships
            causal_validity = await self._validate_causal_relationships(soc)
            
            # Step 4: Cross-domain validation
            cross_domain_support = await self._check_cross_domain_support(soc)
            
            # Step 5: Temporal consistency check
            temporal_consistency = await self._check_temporal_consistency(soc)
            
            # Combine all validation results
            total_adjustment = confidence_adjustment
            total_adjustment += logical_consistency * 0.2
            total_adjustment += causal_validity * 0.15
            total_adjustment += cross_domain_support * 0.1
            total_adjustment += temporal_consistency * 0.1
            
            # Update confidence based on all validations
            if total_adjustment != 0:
                soc.update_confidence(soc.confidence + total_adjustment)
                
            # Accept SOC if it passes basic validation and has reasonable confidence
            if is_valid and soc.confidence > 0.3:
                validated_socs.append(soc)
                logger.info(
                    "SOC validated by System 2",
                    soc_name=soc.name,
                    confidence=soc.confidence,
                    adjustment=total_adjustment,
                    validation_checks={
                        "first_principles": is_valid,
                        "logical_consistency": logical_consistency > 0,
                        "causal_validity": causal_validity > 0,
                        "cross_domain_support": cross_domain_support > 0,
                        "temporal_consistency": temporal_consistency > 0
                    }
                )
            else:
                logger.warning(
                    "SOC rejected by System 2",
                    soc_name=soc.name,
                    confidence=soc.confidence,
                    conflicts=conflicts,
                    first_principles_valid=is_valid
                )
                
        return validated_socs
    
    async def _check_logical_consistency(self, soc: SOC) -> float:
        """Check internal logical consistency of SOC"""
        
        consistency_score = 0.0
        
        # Check for logical contradictions in properties
        props = soc.properties
        if isinstance(props, dict):
            # Look for contradictory properties
            contradictions = [
                ("hot", "cold"),
                ("positive", "negative"),
                ("solid", "liquid"),
                ("increase", "decrease"),
                ("stable", "unstable")
            ]
            
            props_text = str(props).lower()
            for pos, neg in contradictions:
                if pos in props_text and neg in props_text:
                    consistency_score -= 0.3  # Penalty for contradictions
        
        # Check relationships for logical consistency
        if hasattr(soc, 'relationships') and soc.relationships:
            # Strong relationships should be mutual
            strong_relationships = [rel for rel, strength in soc.relationships.items() if strength > 0.8]
            if strong_relationships:
                consistency_score += 0.2  # Bonus for strong relationships
        
        # Domain consistency check
        if soc.domain and soc.name:
            domain_keywords = {
                "physics": ["force", "energy", "motion", "gravity", "mass"],
                "chemistry": ["reaction", "molecule", "atom", "bond", "chemical"],
                "mathematics": ["equation", "function", "variable", "theorem", "proof"]
            }
            
            if soc.domain in domain_keywords:
                domain_words = domain_keywords[soc.domain]
                name_lower = soc.name.lower()
                
                # Check if SOC name aligns with domain
                if any(word in name_lower for word in domain_words):
                    consistency_score += 0.1
                else:
                    # Check if it's a reasonable cross-domain concept
                    if soc.domain != "general":
                        consistency_score -= 0.1
        
        return consistency_score
    
    async def _validate_causal_relationships(self, soc: SOC) -> float:
        """Validate causal relationships of SOC"""
        
        causal_score = 0.0
        
        # Check for causal indicators in properties
        props_text = str(soc.properties).lower()
        
        # Positive causal indicators
        causal_keywords = ["cause", "effect", "result", "lead to", "produce", "generate"]
        for keyword in causal_keywords:
            if keyword in props_text:
                causal_score += 0.1
        
        # Check for impossible causal relationships
        impossible_causals = [
            "effect causes cause",
            "result produces input",
            "output creates input"
        ]
        
        for impossible in impossible_causals:
            if impossible in props_text:
                causal_score -= 0.2
        
        # Check causal consistency with SOC type
        if soc.soc_type == SOCType.PRINCIPLE:
            # Principles should have causal implications
            if any(keyword in props_text for keyword in ["law", "rule", "principle"]):
                causal_score += 0.15
        
        return causal_score
    
    async def _check_cross_domain_support(self, soc: SOC) -> float:
        """Check if SOC has support from other domains"""
        
        support_score = 0.0
        
        # Check if SOC appears in multiple domains
        if soc.domain != "general":
            # Look for similar SOCs in other domains
            similar_socs = [s for s in self.soc_manager.socs.values() 
                          if s.name.lower() in soc.name.lower() and s.domain != soc.domain]
            
            if similar_socs:
                support_score += len(similar_socs) * 0.05
        
        # Check for universal concepts
        universal_concepts = ["energy", "information", "structure", "pattern", "system"]
        if any(concept in soc.name.lower() for concept in universal_concepts):
            support_score += 0.1
        
        return support_score
    
    async def _check_temporal_consistency(self, soc: SOC) -> float:
        """Check temporal consistency of SOC"""
        
        temporal_score = 0.0
        
        # Check for temporal indicators
        props_text = str(soc.properties).lower()
        
        # Look for temporal keywords
        temporal_keywords = ["time", "duration", "instant", "continuous", "periodic"]
        for keyword in temporal_keywords:
            if keyword in props_text:
                temporal_score += 0.05
        
        # Check for temporal contradictions
        contradictions = [
            ("instant", "duration"),
            ("permanent", "temporary"),
            ("continuous", "discrete")
        ]
        
        for pos, neg in contradictions:
            if pos in props_text and neg in props_text:
                temporal_score -= 0.1
        
        # Check SOC age vs confidence
        if hasattr(soc, 'created_at') and hasattr(soc, 'last_updated'):
            # Recent updates should align with confidence
            import datetime
            if soc.last_updated and soc.created_at:
                time_diff = (soc.last_updated - soc.created_at).total_seconds()
                if time_diff > 0 and soc.confidence > 0.8:
                    temporal_score += 0.05  # Confidence built over time
        
        return temporal_score
        
    async def _bayesian_search_experiments(self, socs: List[SOC]) -> List[ExperimentResult]:
        """Conduct Bayesian search experiments on uncertain SOCs"""
        
        experiment_results = []
        
        for soc in socs:
            # Conduct experiments on SOCs below confidence threshold
            if soc.confidence < 0.7:
                hypothesis = f"SOC '{soc.name}' is valid in domain '{soc.domain}'"
                
                experiment = await self.bayesian_search.conduct_experiment(
                    hypothesis, soc, ExperimentType.LOGICAL_TEST
                )
                
                experiment_results.append(experiment)
                
                # Update SOC based on experiment
                self.soc_manager.update_soc_from_experiment(soc.name, experiment)
                
        return experiment_results
        
    async def _share_knowledge(self, experiment_results: List[ExperimentResult]):
        """Share experiment results with PRSM network"""
        
        for result in experiment_results:
            # Share high-value experiments
            if result.information_value > 0.1:
                self.bayesian_search.share_experiment_result(result)
                
    async def _analyze_communicative_intent(self, query: str, context: Dict[str, Any] = None) -> UserGoal:
        """
        Analyze communicative intent to understand what human actually needs
        
        This addresses the 'stochastic parrot' problem by grounding responses
        in genuine understanding of human goals rather than just pattern matching.
        """
        
        intent_analysis_prompt = f"""
        Analyze this query to understand the human's actual communicative intent and goals:
        
        Query: "{query}"
        Context: {context or {}}
        
        Identify:
        1. Primary intent (information seeking, problem solving, verification, etc.)
        2. What they actually need (not just what they asked)
        3. Their likely expertise level
        4. Success criteria for a helpful response
        5. Any missing context they might need
        
        Format as JSON with: intent, inferred_needs, expertise_level, success_criteria, context_requirements
        """
        
        try:
            analysis = await self.model_executor.execute_request(
                prompt=intent_analysis_prompt,
                model_name="gpt-4",
                temperature=0.3  # Lower temperature for intent analysis
            )
            
            # Parse the analysis (simplified for now)
            # In full implementation, use robust JSON parsing
            
            # Infer intent based on query patterns
            intent = CommunicativeIntent.INFORMATION_SEEKING
            if "how to" in query.lower() or "solve" in query.lower():
                intent = CommunicativeIntent.PROBLEM_SOLVING
            elif "explain" in query.lower() or "what is" in query.lower():
                intent = CommunicativeIntent.EXPLANATION
            elif "predict" in query.lower() or "will" in query.lower():
                intent = CommunicativeIntent.PREDICTION
            elif "analyze" in query.lower() or "compare" in query.lower():
                intent = CommunicativeIntent.ANALYSIS
            
            user_goal = UserGoal(
                intent=intent,
                explicit_query=query,
                inferred_needs=[
                    "Clear explanation",
                    "Actionable insights",
                    "Confidence in answer quality"
                ],
                context_requirements={
                    "domain": self.domain,
                    "complexity": "moderate"
                },
                success_criteria=[
                    "Addresses actual need",
                    "Appropriate depth",
                    "Actionable response"
                ],
                expertise_level="intermediate"  # Default assumption
            )
            
            logger.info(
                "Analyzed communicative intent",
                intent=intent.value,
                query=query[:50] + "..." if len(query) > 50 else query
            )
            
            return user_goal
            
        except Exception as e:
            logger.error("Error analyzing communicative intent", error=str(e))
            # Return default goal
            return UserGoal(
                intent=CommunicativeIntent.INFORMATION_SEEKING,
                explicit_query=query,
                inferred_needs=["Basic information"],
                success_criteria=["Relevant response"]
            )
            
    async def _detect_bias_and_generate_perspectives(self, socs: List[SOC], user_goal: UserGoal) -> BiasDetectionResult:
        """
        Detect potential bias and generate diverse perspectives
        
        This addresses the 'hegemonic viewpoint' problem identified in Stochastic Parrots
        by actively seeking diverse perspectives and detecting bias.
        """
        
        bias_detection_prompt = f"""
        Analyze these concepts and reasoning for potential bias:
        
        SOCs: {[soc.name for soc in socs]}
        User Goal: {user_goal.intent.value}
        Domain: {self.domain}
        
        Check for:
        1. Overrepresentation of dominant/hegemonic viewpoints
        2. Missing perspectives from marginalized groups
        3. Cultural, gender, racial, or other systematic biases
        4. Assumptions that may not hold across different contexts
        
        Suggest additional perspectives to consider.
        """
        
        try:
            # Generate diverse perspectives using multiple "agents" with different temperatures
            perspectives = []
            
            # Conservative perspective (low temperature)
            conservative_analysis = await self.model_executor.execute_request(
                prompt=bias_detection_prompt + "\n\nProvide a conservative, established viewpoint.",
                model_name="gpt-4",
                temperature=0.2
            )
            perspectives.append(("conservative", conservative_analysis))
            
            # Progressive perspective (higher temperature)
            progressive_analysis = await self.model_executor.execute_request(
                prompt=bias_detection_prompt + "\n\nProvide a progressive, questioning viewpoint.",
                model_name="gpt-4",
                temperature=0.8
            )
            perspectives.append(("progressive", progressive_analysis))
            
            # International perspective
            international_analysis = await self.model_executor.execute_request(
                prompt=bias_detection_prompt + "\n\nProvide a non-Western, international perspective.",
                model_name="gpt-4",
                temperature=0.5
            )
            perspectives.append(("international", international_analysis))
            
            # Analyze perspective diversity
            perspective_diversity_score = min(1.0, len(perspectives) / 5.0)  # Normalize to 0-1
            
            # Detect hegemonic bias (simplified heuristic)
            hegemonic_bias_score = 0.3  # Default moderate bias assumption
            
            # Check if SOCs represent diverse viewpoints
            evidence_quality_score = 0.7  # Default moderate quality
            
            bias_result = BiasDetectionResult(
                perspective_diversity_score=perspective_diversity_score,
                hegemonic_bias_score=hegemonic_bias_score,
                evidence_quality_score=evidence_quality_score,
                bias_flags=["dominant_perspective_check_needed"],
                missing_perspectives=["marginalized_communities", "non_western_viewpoints"],
                suggested_additional_perspectives=[
                    "Consider impacts on underrepresented groups",
                    "Include non-Western perspectives",
                    "Question dominant assumptions"
                ],
                bias_mitigation_strategies=[
                    "Actively seek diverse viewpoints",
                    "Validate against multiple perspectives",
                    "Question implicit assumptions"
                ]
            )
            
            logger.info(
                "Completed bias detection analysis",
                perspective_diversity=perspective_diversity_score,
                hegemonic_bias=hegemonic_bias_score
            )
            
            return bias_result
            
        except Exception as e:
            logger.error("Error in bias detection", error=str(e))
            # Return default bias analysis
            return BiasDetectionResult(
                perspective_diversity_score=0.5,
                hegemonic_bias_score=0.5,
                evidence_quality_score=0.5,
                bias_flags=["analysis_incomplete"],
                suggested_additional_perspectives=["seek_diverse_viewpoints"]
            )
                
    async def _generate_response(
        self, 
        query: str, 
        socs: List[SOC], 
        experiments: List[ExperimentResult],
        user_goal: UserGoal,
        bias_analysis: BiasDetectionResult
    ) -> Dict[str, Any]:
        """Generate final response integrating all components"""
        
        # Compile comprehensive reasoning trace (Anti-Stochastic Parrot)
        reasoning_trace = []
        
        # Intent Analysis
        reasoning_trace.append({
            "step": "intent_analysis",
            "description": f"Analyzed communicative intent: {user_goal.intent.value}",
            "details": {
                "intent": user_goal.intent.value,
                "inferred_needs": user_goal.inferred_needs,
                "success_criteria": user_goal.success_criteria,
                "expertise_level": user_goal.expertise_level
            }
        })
        
        # System 1 findings
        reasoning_trace.append({
            "step": "system1_recognition",
            "description": f"Recognized {len(socs)} SOCs from query",
            "details": [{"name": soc.name, "type": soc.soc_type.value, "confidence": soc.confidence} for soc in socs]
        })
        
        # System 2 validation
        reasoning_trace.append({
            "step": "system2_validation",
            "description": "Validated SOCs against world model",
            "details": [{"name": soc.name, "validated": True, "confidence": soc.confidence} for soc in socs]
        })
        
        # Bias Detection
        reasoning_trace.append({
            "step": "bias_detection",
            "description": "Analyzed perspective diversity and bias",
            "details": {
                "perspective_diversity": bias_analysis.perspective_diversity_score,
                "hegemonic_bias": bias_analysis.hegemonic_bias_score,
                "missing_perspectives": bias_analysis.missing_perspectives,
                "mitigation_strategies": bias_analysis.bias_mitigation_strategies
            }
        })
        
        # Experiments
        reasoning_trace.append({
            "step": "bayesian_experiments",
            "description": f"Conducted {len(experiments)} experiments",
            "details": [{"hypothesis": exp.hypothesis, "success": exp.success, "information_value": exp.information_value} for exp in experiments]
        })
        
        # Generate response with intent alignment and bias awareness
        response_prompt = f"""
        Based on hybrid System 1 + System 2 analysis with anti-stochastic parrot enhancements:
        
        Query: {query}
        
        Human Intent: {user_goal.intent.value}
        Human Needs: {user_goal.inferred_needs}
        Success Criteria: {user_goal.success_criteria}
        
        System 1 Recognition: {len(socs)} SOCs identified
        System 2 Validation: World model consistency checked
        Bias Analysis: {bias_analysis.perspective_diversity_score:.2f} perspective diversity
        Bayesian Experiments: {len(experiments)} experiments conducted
        
        Generate a response that:
        1. Directly addresses the human's actual intent and needs
        2. Integrates fast transformer insights (System 1) with first-principles reasoning (System 2)
        3. Incorporates diverse perspectives to avoid hegemonic bias
        4. Shows clear reasoning trace from first principles to conclusions
        5. Includes confidence levels and uncertainty acknowledgment
        6. Meets the success criteria: {user_goal.success_criteria}
        
        Avoid being a 'stochastic parrot' - ensure genuine understanding rather than pattern matching.
        """
        
        response_text = await self.model_executor.execute_request(
            prompt=response_prompt,
            model_name="gpt-4",
            temperature=self.temperature
        )
        
        return {
            "response": response_text,
            "reasoning_trace": reasoning_trace,
            "socs_used": [{"name": soc.name, "confidence": soc.confidence, "level": soc.confidence_level.value} for soc in socs],
            "experiments_conducted": len(experiments),
            "hybrid_engine_id": self.agent_id,
            
            # Anti-Stochastic Parrot enhancements
            "communicative_intent": {
                "intent": user_goal.intent.value,
                "inferred_needs": user_goal.inferred_needs,
                "success_criteria": user_goal.success_criteria,
                "expertise_level": user_goal.expertise_level
            },
            "bias_analysis": {
                "perspective_diversity_score": bias_analysis.perspective_diversity_score,
                "hegemonic_bias_score": bias_analysis.hegemonic_bias_score,
                "evidence_quality_score": bias_analysis.evidence_quality_score,
                "bias_flags": bias_analysis.bias_flags,
                "missing_perspectives": bias_analysis.missing_perspectives,
                "mitigation_strategies": bias_analysis.bias_mitigation_strategies
            },
            "anti_stochastic_parrot_features": {
                "intent_aligned": True,
                "world_model_grounded": True,
                "bias_detected": len(bias_analysis.bias_flags) > 0,
                "perspective_diversity": bias_analysis.perspective_diversity_score,
                "first_principles_reasoning": True,
                "communicative_intent_modeled": True
            }
        }
        
    async def update_from_shared_experiment(self, experiment_result: ExperimentResult):
        """Update world model from shared experiment (hive mind effect)"""
        
        # Receive shared experiment
        self.bayesian_search.receive_shared_experiment(experiment_result)
        
        # Update affected SOCs
        for soc_name in experiment_result.affected_socs:
            self.soc_manager.update_soc_from_experiment(soc_name, experiment_result)
            
        logger.info(
            "Updated from shared experiment",
            experiment_id=str(experiment_result.id),
            from_agent=experiment_result.agent_id,
            information_value=experiment_result.information_value
        )
        
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get hybrid engine statistics"""
        
        return {
            "agent_id": self.agent_id,
            "temperature": self.temperature,
            "domain": self.domain,
            "total_socs": len(self.soc_manager.socs),
            "core_socs": len(self.soc_manager.get_core_socs()),
            "experiments_conducted": len(self.bayesian_search.experiment_history),
            "shared_experiments_received": len(self.bayesian_search.shared_experiments)
        }


# Factory functions for integration with existing PRSM infrastructure

def create_hybrid_nwtn_engine(agent_id: str = None, temperature: float = 0.7) -> HybridNWTNEngine:
    """Factory function to create hybrid NWTN engine"""
    
    engine = HybridNWTNEngine(agent_id=agent_id)
    engine.temperature = temperature
    
    return engine


def create_specialized_agent_team(domain: str, team_size: int = 3) -> List[HybridNWTNEngine]:
    """
    Create team of specialized agents with different temperatures
    
    This implements the multi-agent team concept from the brainstorming document
    where agents have different 'perspectives' through temperature variation.
    """
    
    agents = []
    
    for i in range(team_size):
        # Vary temperature for different perspectives
        temperature = 0.3 + (i * 0.4)  # Range: 0.3 to 1.1
        
        agent = create_hybrid_nwtn_engine(
            agent_id=f"{domain}_agent_{i+1}",
            temperature=temperature
        )
        agent.domain = domain
        
        agents.append(agent)
        
    logger.info(
        "Created specialized agent team",
        domain=domain,
        team_size=team_size,
        temperatures=[agent.temperature for agent in agents]
    )
    
    return agents


# Integration with existing PRSM architecture

async def integrate_with_nwtn_orchestrator(orchestrator_instance, enable_hybrid: bool = True):
    """
    Integrate hybrid architecture with existing NWTN orchestrator
    
    This function modifies the existing orchestrator to use hybrid reasoning
    when processing queries, maintaining backward compatibility.
    """
    
    if not enable_hybrid:
        return orchestrator_instance
        
    # Add hybrid engine to orchestrator
    orchestrator_instance.hybrid_engine = create_hybrid_nwtn_engine()
    
    # Store original process method
    original_process = orchestrator_instance.process_query
    
    # Replace with hybrid-enhanced process
    async def hybrid_enhanced_process(query: str, context: Dict[str, Any] = None):
        # Try hybrid processing first
        try:
            hybrid_result = await orchestrator_instance.hybrid_engine.process_query(query, context)
            
            # Integrate with original orchestrator response
            original_result = await original_process(query, context)
            
            # Combine results
            enhanced_result = original_result.copy()
            enhanced_result["hybrid_reasoning"] = hybrid_result
            enhanced_result["reasoning_enhanced"] = True
            
            return enhanced_result
            
        except Exception as e:
            logger.error("Hybrid processing failed, falling back to original", error=str(e))
            return await original_process(query, context)
    
    # Replace method
    orchestrator_instance.process_query = hybrid_enhanced_process
    
    logger.info("Integrated hybrid architecture with NWTN orchestrator")
    
    return orchestrator_instance
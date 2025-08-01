"""
Enhanced Model Router Agent
Advanced routing with specialist matching, marketplace integration, and teacher selection
"""

import asyncio
import time
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from prsm.agents.base import BaseAgent
from prsm.core.config import get_settings
from prsm.core.models import (
    AgentType, ArchitectTask, TeacherModel, ModelType, 
    FTNSTransaction, TimestampMixin
)
from prsm.federation.model_registry import ModelRegistry
from prsm.agents.routers.marketplace_integration import marketplace_integration
from prsm.agents.routers.performance_tracker import performance_tracker, MetricType, ModelPerformanceProfile
from prsm.agents.routers.tool_router import ToolRouter, ToolRequest, ToolRoutingDecision

logger = structlog.get_logger(__name__)
settings = get_settings()


class RoutingStrategy(str, Enum):
    """Routing strategies for different use cases"""
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    COST_OPTIMIZED = "cost_optimized" 
    LATENCY_OPTIMIZED = "latency_optimized"
    ACCURACY_OPTIMIZED = "accuracy_optimized"
    MARKETPLACE_PREFERRED = "marketplace_preferred"
    TEACHER_SELECTION = "teacher_selection"
    HYBRID_CHEMISTRY = "hybrid_chemistry"  # New hybrid chemistry routing


class ModelSource(str, Enum):
    """Source of model availability"""
    LOCAL_REGISTRY = "local_registry"
    MARKETPLACE = "marketplace"
    P2P_NETWORK = "p2p_network"
    TEACHER_POOL = "teacher_pool"
    HYBRID_EXECUTOR = "hybrid_executor"  # New hybrid executor source


class MarketplaceRequest(BaseModel):
    """Request for marketplace model routing"""
    request_id: UUID = Field(default_factory=uuid4)
    task_description: str
    domain_requirements: List[str] = Field(default_factory=list)
    performance_requirements: Dict[str, float] = Field(default_factory=dict)
    budget_limit: Optional[float] = None
    latency_requirements: Optional[float] = None
    quality_threshold: float = 0.7
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RoutingDecision(TimestampMixin):
    """Complete routing decision with metadata"""
    decision_id: UUID = Field(default_factory=uuid4)
    task_id: Optional[UUID] = None
    strategy_used: RoutingStrategy
    primary_candidate: "ModelCandidate"
    backup_candidates: List["ModelCandidate"] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    routing_time: float
    reasoning: str
    cost_estimate: Optional[float] = None
    expected_latency: Optional[float] = None


class TeacherSelection(BaseModel):
    """Teacher model selection for student training"""
    selection_id: UUID = Field(default_factory=uuid4)
    student_model_id: str
    domain: str
    teacher_candidates: List[UUID] = Field(default_factory=list)
    selected_teacher: Optional[UUID] = None
    selection_criteria: Dict[str, float] = Field(default_factory=dict)
    expected_improvement: Optional[float] = None
    training_cost_estimate: Optional[float] = None


class ModelCandidate(BaseModel):
    """Enhanced candidate model for task execution"""
    model_id: str
    name: Optional[str] = None
    specialization: str
    model_type: ModelType = ModelType.GENERAL
    source: ModelSource = ModelSource.LOCAL_REGISTRY
    performance_score: float = Field(ge=0.0, le=1.0)
    compatibility_score: float = Field(ge=0.0, le=1.0)
    availability_score: float = Field(ge=0.0, le=1.0, default=1.0)
    cost_score: float = Field(ge=0.0, le=1.0, default=0.5)  # Lower cost = higher score
    latency_score: float = Field(ge=0.0, le=1.0, default=0.5)  # Lower latency = higher score
    overall_score: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Marketplace-specific attributes
    marketplace_url: Optional[str] = None
    cost_per_token: Optional[float] = None
    estimated_latency: Optional[float] = None
    provider_reputation: Optional[float] = None
    
    # Teacher-specific attributes
    teaching_effectiveness: Optional[float] = None
    curriculum_quality: Optional[float] = None
    student_success_rate: Optional[float] = None
    
    # Additional metadata
    capabilities: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_overall_score(self, strategy: RoutingStrategy) -> float:
        """Calculate overall score based on routing strategy"""
        if strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            score = (self.performance_score * 0.5 + 
                    self.compatibility_score * 0.3 + 
                    self.availability_score * 0.2)
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            score = (self.cost_score * 0.4 + 
                    self.compatibility_score * 0.3 + 
                    self.performance_score * 0.3)
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            score = (self.latency_score * 0.4 + 
                    self.availability_score * 0.3 + 
                    self.compatibility_score * 0.3)
        elif strategy == RoutingStrategy.ACCURACY_OPTIMIZED:
            score = (self.performance_score * 0.6 + 
                    self.compatibility_score * 0.4)
        elif strategy == RoutingStrategy.MARKETPLACE_PREFERRED:
            marketplace_bonus = 0.2 if self.source == ModelSource.MARKETPLACE else 0.0
            score = (self.performance_score * 0.3 + 
                    self.compatibility_score * 0.3 + 
                    self.cost_score * 0.2 + 
                    marketplace_bonus)
        elif strategy == RoutingStrategy.TEACHER_SELECTION:
            if self.model_type == ModelType.TEACHER:
                score = (self.teaching_effectiveness or 0.5) * 0.4 + \
                       (self.curriculum_quality or 0.5) * 0.3 + \
                       (self.student_success_rate or 0.5) * 0.3
            else:
                score = 0.1  # Non-teacher models get low score for teaching
        elif strategy == RoutingStrategy.HYBRID_CHEMISTRY:
            if self.source == ModelSource.HYBRID_EXECUTOR:
                # Hybrid executors get significant bonus for chemistry tasks
                score = (self.performance_score * 0.4 + 
                        self.compatibility_score * 0.4 + 
                        self.availability_score * 0.2)
                if self.specialization == "chemistry":
                    score += 0.2  # Extra bonus for chemistry specialization
            else:
                # Traditional models get lower scores for chemistry
                score = (self.performance_score * 0.3 + 
                        self.compatibility_score * 0.3 + 
                        self.availability_score * 0.2 + 
                        self.cost_score * 0.2)
        else:
            # Default balanced scoring
            score = (self.performance_score * 0.4 + 
                    self.compatibility_score * 0.3 + 
                    self.availability_score * 0.2 + 
                    self.cost_score * 0.1)
        
        self.overall_score = min(score, 1.0)
        return self.overall_score


class ModelRouter(BaseAgent):
    """
    Enhanced Model Router for PRSM with MCP Tool Integration
    
    Advanced routing capabilities:
    - Intelligent specialist matching with domain expertise
    - Marketplace integration for external model discovery
    - Teacher model selection for student training
    - Multi-strategy routing optimization
    - Performance-based adaptive routing
    - MCP Tool routing for tool-augmented models
    
    MCP Tool Integration:
    - Routes tool requests from models to appropriate MCP tools
    - Coordinates model-tool workflows for enhanced capabilities
    - Manages tool execution security and sandboxing
    - Tracks tool usage and performance metrics
    """
    
    def __init__(self, model_registry: Optional[ModelRegistry] = None, agent_id: Optional[str] = None):
        super().__init__(agent_id=agent_id, agent_type=AgentType.ROUTER)
        self.model_registry = model_registry or ModelRegistry()
        self.routing_cache: Dict[str, List[ModelCandidate]] = {}
        self.routing_decisions: List[RoutingDecision] = []
        self.performance_history: Dict[str, List[float]] = {}  # model_id -> performance scores
        self.marketplace_endpoints: List[str] = self._initialize_marketplace_endpoints()
        self.teacher_pool: Dict[str, List[TeacherModel]] = {}  # domain -> teachers
        
        # MCP Tool Integration
        self.tool_router = ToolRouter()
        self.tool_routing_decisions: List[ToolRoutingDecision] = []
        self.model_tool_associations: Dict[str, List[str]] = {}  # model_id -> tool_ids
        
        # Hybrid Architecture Integration
        self.hybrid_executors: Dict[str, Any] = {}  # domain -> hybrid executor
        self.chemistry_patterns = self._initialize_chemistry_patterns()
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
        
        logger.info("Enhanced ModelRouter with MCP Tool and Hybrid Architecture integration initialized",
                   agent_id=self.agent_id,
                   marketplace_endpoints=len(self.marketplace_endpoints),
                   available_tools=len(self.tool_router.tool_registry.tools))
    
    def _initialize_marketplace_endpoints(self) -> List[str]:
        """Initialize marketplace endpoints for model discovery"""
        # In production, these would be loaded from configuration
        return [
            "https://api.huggingface.co/models",
            "https://api.openai.com/models", 
            "https://api.anthropic.com/models",
            "https://api.cohere.ai/models"
        ]
    
    def _initialize_chemistry_patterns(self) -> List[str]:
        """Initialize chemistry detection patterns for automatic routing"""
        return [
            # Chemical reactions and processes
            "reaction", "chemical reaction", "synthesis", "catalysis", "catalyst",
            "reactant", "product", "reagent", "solvent", "precipitation",
            "oxidation", "reduction", "combustion", "polymerization", "hydrolysis",
            
            # Molecular and atomic terms
            "molecule", "molecular", "atom", "atomic", "bond", "ionic", "covalent",
            "electron", "proton", "neutron", "orbital", "valence", "compound",
            "element", "periodic table", "isotope", "ion", "radical",
            
            # Chemical properties and measurements
            "ph", "acidity", "basicity", "molarity", "concentration", "solubility",
            "boiling point", "melting point", "density", "viscosity", "conductivity",
            "electronegativity", "polarity", "dipole", "entropy", "enthalpy",
            
            # Thermodynamics and kinetics
            "gibbs free energy", "activation energy", "equilibrium", "rate constant",
            "temperature", "pressure", "thermodynamic", "kinetic", "mechanism",
            "pathway", "intermediate", "transition state", "endothermic", "exothermic",
            
            # Chemical notation and formulas
            "smiles", "formula", "chemical formula", "molecular formula",
            "structural formula", "lewis structure", "chemical equation",
            
            # Laboratory and analytical
            "titration", "spectroscopy", "chromatography", "mass spectrometry",
            "nmr", "ir", "uv", "lab", "laboratory", "analytical", "qualitative",
            "quantitative", "purity", "yield", "synthesis",
            
            # Biochemistry
            "protein", "enzyme", "amino acid", "nucleotide", "dna", "rna",
            "metabolic", "biochemical", "substrate", "inhibitor", "cofactor",
            
            # Materials chemistry
            "polymer", "crystal", "crystalline", "amorphous", "alloy", "ceramic",
            "composite", "nanomaterial", "surface", "interface", "coating"
        ]
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking integration"""
        try:
            # Initialize performance tracker asynchronously
            asyncio.create_task(performance_tracker.initialize())
            logger.info("Performance tracking integration initialized")
        except Exception as e:
            logger.warning("Failed to initialize performance tracking", error=str(e))
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Enhanced processing with comprehensive routing decision
        
        Args:
            input_data: Task information or routing request
            context: Optional context with routing preferences
            
        Returns:
            RoutingDecision: Complete routing decision with metadata
        """
        start_time = time.time()
        
        # Extract routing parameters
        strategy = RoutingStrategy.PERFORMANCE_OPTIMIZED
        if context:
            strategy = RoutingStrategy(context.get("strategy", "performance_optimized"))
        
        # Extract task information
        if isinstance(input_data, ArchitectTask):
            task = input_data
            task_description = task.instruction
            complexity = task.complexity_score
            task_id = task.task_id
        elif isinstance(input_data, dict):
            task_description = input_data.get("task", str(input_data))
            complexity = input_data.get("complexity", 0.5)
            task_id = input_data.get("task_id")
        else:
            task_description = str(input_data)
            complexity = 0.5
            task_id = None
        
        # Auto-detect chemistry tasks and route to hybrid executor
        if self._is_chemistry_task(task_description) and strategy != RoutingStrategy.HYBRID_CHEMISTRY:
            logger.info("Auto-detected chemistry task, routing to hybrid executor",
                       agent_id=self.agent_id,
                       task_description=task_description[:100])
            strategy = RoutingStrategy.HYBRID_CHEMISTRY
        
        logger.info("Processing enhanced routing request",
                   agent_id=self.agent_id,
                   task_length=len(task_description),
                   complexity=complexity,
                   strategy=strategy.value)
        
        # Get model candidates from all sources
        candidates = await self._discover_all_candidates(task_description, complexity, strategy)
        
        if not candidates:
            logger.warning("No candidates found for task", task=task_description[:100])
            # Create a basic decision with no candidates
            return RoutingDecision(
                task_id=task_id,
                strategy_used=strategy,
                primary_candidate=ModelCandidate(
                    model_id="fallback",
                    specialization="general",
                    performance_score=0.1,
                    compatibility_score=0.1
                ),
                backup_candidates=[],
                confidence_score=0.0,
                routing_time=time.time() - start_time,
                reasoning="No suitable candidates found"
            )
        
        # Calculate scores for all candidates
        for candidate in candidates:
            candidate.calculate_overall_score(strategy)
        
        # Sort by overall score
        candidates.sort(key=lambda c: c.overall_score, reverse=True)
        
        # Create routing decision
        decision = RoutingDecision(
            task_id=task_id,
            strategy_used=strategy,
            primary_candidate=candidates[0],
            backup_candidates=candidates[1:min(4, len(candidates))],  # Up to 3 backups
            confidence_score=self._calculate_confidence(candidates),
            routing_time=time.time() - start_time,
            reasoning=self._generate_reasoning(task_description, candidates[0], strategy),
            cost_estimate=self._estimate_cost(candidates[0]),
            expected_latency=self._estimate_latency(candidates[0])
        )
        
        # Store decision for learning
        self.routing_decisions.append(decision)
        
        logger.info("Enhanced routing completed",
                   agent_id=self.agent_id,
                   candidates_found=len(candidates),
                   primary_model=candidates[0].model_id,
                   confidence=decision.confidence_score,
                   routing_time=f"{decision.routing_time:.3f}s")
        
        return decision
    
    async def _discover_all_candidates(self, task_description: str, complexity: float, 
                                    strategy: RoutingStrategy) -> List[ModelCandidate]:
        """Discover candidates from all available sources"""
        candidates = []
        
        # 1. Local registry candidates
        local_candidates = await self._discover_local_candidates(task_description, complexity)
        candidates.extend(local_candidates)
        
        # 2. Marketplace candidates (if strategy permits)
        if strategy in [RoutingStrategy.MARKETPLACE_PREFERRED, RoutingStrategy.COST_OPTIMIZED]:
            marketplace_candidates = await self._discover_marketplace_candidates(task_description)
            candidates.extend(marketplace_candidates)
        
        # 3. Teacher candidates (for teaching tasks)
        if strategy == RoutingStrategy.TEACHER_SELECTION:
            teacher_candidates = await self._discover_teacher_candidates(task_description)
            candidates.extend(teacher_candidates)
        
        # 4. Hybrid executor candidates (for chemistry tasks)
        if strategy == RoutingStrategy.HYBRID_CHEMISTRY:
            hybrid_candidates = await self._discover_hybrid_candidates(task_description)
            candidates.extend(hybrid_candidates)
        
        # 5. P2P network candidates
        p2p_candidates = await self._discover_p2p_candidates(task_description)
        candidates.extend(p2p_candidates)
        
        return candidates
    
    async def _discover_local_candidates(self, task_description: str, complexity: float) -> List[ModelCandidate]:
        """Discover candidates from local model registry with performance enhancement"""
        candidates = []
        task_category = await self._categorize_task(task_description)
        
        # Get specialists for the task category
        specialist_ids = await self.model_registry.discover_specialists(task_category)
        
        for model_id in specialist_ids:
            model_details = await self.model_registry.get_model_details(model_id)
            if model_details:
                # Get enhanced performance data
                performance_profile = await performance_tracker.get_model_performance(model_id)
                
                compatibility_score = await self._calculate_compatibility(
                    task_description, model_details, complexity
                )
                
                # Use real performance data if available
                if performance_profile:
                    default_score = self._get_model_performance_score(model_details)
                    performance_score = performance_profile.accuracy_score or default_score
                    actual_latency = performance_profile.response_time_avg
                    availability_score = performance_profile.availability_score
                    
                    # Check for performance issues
                    issues = await performance_tracker.detect_performance_issues(model_id)
                    if issues:
                        # Reduce scores for models with issues
                        performance_score *= 0.8
                        availability_score *= 0.9
                else:
                    performance_score = self._get_model_performance_score(model_details)
                    actual_latency = 0.5  # Default for local models
                    availability_score = 1.0
                
                candidate = ModelCandidate(
                    model_id=model_id,
                    name=getattr(model_details, 'name', model_id),
                    specialization=self._get_model_specialization(model_details),
                    source=ModelSource.LOCAL_REGISTRY,
                    performance_score=performance_score,
                    compatibility_score=compatibility_score,
                    availability_score=availability_score,
                    cost_score=0.9,  # Local models are cost-effective
                    latency_score=self._calculate_latency_score(actual_latency),
                    capabilities=getattr(model_details, 'capabilities', []),
                    limitations=getattr(model_details, 'limitations', [])
                )
                candidates.append(candidate)
        
        logger.debug("Local candidates discovery completed",
                    task_category=task_category,
                    candidates_found=len(candidates))
        
        return candidates
    
    async def _discover_marketplace_candidates(self, task_description: str) -> List[ModelCandidate]:
        """Discover candidates from marketplace APIs using real integrations"""
        candidates = []
        
        try:
            # Use real marketplace integration
            marketplace_models = await marketplace_integration.discover_marketplace_models(
                task_description, limit=10
            )
            
            for marketplace_model in marketplace_models:
                # Calculate compatibility score for task
                compatibility_score = await self._calculate_marketplace_compatibility_enhanced(
                    task_description, marketplace_model
                )
                
                # Convert marketplace model to router candidate
                candidate = ModelCandidate(
                    model_id=marketplace_model.model_id,
                    name=marketplace_model.name,
                    specialization=marketplace_model.specialization,
                    source=ModelSource.MARKETPLACE,
                    performance_score=marketplace_model.performance_score,
                    compatibility_score=compatibility_score,
                    availability_score=marketplace_model.availability_score,
                    cost_score=self._calculate_cost_score(marketplace_model.cost_per_token),
                    latency_score=self._calculate_latency_score(marketplace_model.estimated_latency),
                    marketplace_url=marketplace_model.marketplace_url,
                    cost_per_token=marketplace_model.cost_per_token,
                    estimated_latency=marketplace_model.estimated_latency,
                    provider_reputation=marketplace_model.provider_reputation,
                    capabilities=marketplace_model.capabilities,
                    limitations=marketplace_model.limitations,
                    last_updated=marketplace_model.last_updated
                )
                candidates.append(candidate)
                
            logger.info("Real marketplace discovery completed",
                       task=task_description[:50],
                       candidates_found=len(candidates))
                       
        except Exception as e:
            logger.error("Error in marketplace discovery, falling back to simulation", 
                        error=str(e))
            
            # Fallback to simulation if real integration fails
            candidates = await self._discover_marketplace_candidates_fallback(task_description)
        
        return candidates
    
    async def _discover_marketplace_candidates_fallback(self, task_description: str) -> List[ModelCandidate]:
        """Fallback marketplace discovery with simulated data"""
        candidates = []
        
        # Minimal fallback candidates
        fallback_models = [
            {
                "model_id": "fallback_gpt4",
                "name": "GPT-4 (Fallback)",
                "specialization": "general",
                "performance_score": 0.90,
                "cost_per_token": 0.03,
                "estimated_latency": 2.5,
                "provider_reputation": 0.9,
                "marketplace_url": "https://api.openai.com/v1/chat/completions"
            }
        ]
        
        for model_data in fallback_models:
            compatibility_score = await self._calculate_marketplace_compatibility(
                task_description, model_data
            )
            
            candidate = ModelCandidate(
                model_id=model_data["model_id"],
                name=model_data["name"],
                specialization=model_data["specialization"],
                source=ModelSource.MARKETPLACE,
                performance_score=model_data["performance_score"],
                compatibility_score=compatibility_score,
                availability_score=0.7,  # Lower availability for fallback
                cost_score=self._calculate_cost_score(model_data["cost_per_token"]),
                latency_score=self._calculate_latency_score(model_data["estimated_latency"]),
                marketplace_url=model_data["marketplace_url"],
                cost_per_token=model_data["cost_per_token"],
                estimated_latency=model_data["estimated_latency"],
                provider_reputation=model_data["provider_reputation"]
            )
            candidates.append(candidate)
        
        return candidates
    
    async def _discover_teacher_candidates(self, task_description: str) -> List[ModelCandidate]:
        """Discover teacher model candidates"""
        candidates = []
        domain = await self._extract_domain_from_task(task_description)
        
        # Simulate teacher model discovery
        teacher_models = [
            {
                "teacher_id": "teacher_physics_01",
                "name": "Physics Specialist Teacher",
                "specialization": "physics",
                "teaching_effectiveness": 0.85,
                "curriculum_quality": 0.9,
                "student_success_rate": 0.78
            },
            {
                "teacher_id": "teacher_math_01", 
                "name": "Mathematics Expert Teacher",
                "specialization": "mathematics",
                "teaching_effectiveness": 0.88,
                "curriculum_quality": 0.85,
                "student_success_rate": 0.82
            }
        ]
        
        for teacher_data in teacher_models:
            if domain == "any" or teacher_data["specialization"] == domain:
                compatibility_score = 0.8 if teacher_data["specialization"] == domain else 0.6
                
                candidate = ModelCandidate(
                    model_id=teacher_data["teacher_id"],
                    name=teacher_data["name"],
                    specialization=teacher_data["specialization"],
                    model_type=ModelType.TEACHER,
                    source=ModelSource.TEACHER_POOL,
                    performance_score=teacher_data["teaching_effectiveness"],
                    compatibility_score=compatibility_score,
                    availability_score=0.9,
                    teaching_effectiveness=teacher_data["teaching_effectiveness"],
                    curriculum_quality=teacher_data["curriculum_quality"],
                    student_success_rate=teacher_data["student_success_rate"]
                )
                candidates.append(candidate)
        
        return candidates
    
    async def _discover_p2p_candidates(self, task_description: str) -> List[ModelCandidate]:
        """Discover candidates from P2P network - Using Claude API as primary model"""
        candidates = []
        
        # Use Claude-3.5-Sonnet as the primary model instead of invalid p2p models
        p2p_models = [
            {
                "model_id": "claude-3-5-sonnet-20241022",
                "name": "Claude-3.5-Sonnet via NWTN",
                "specialization": "general",
                "performance_score": 0.95,
                "estimated_latency": 2.0
            }
        ]
        
        for model_data in p2p_models:
            compatibility_score = 0.7  # P2P models have moderate compatibility
            latency_score = max(0.1, 1.0 - (model_data["estimated_latency"] / 10.0))
            
            candidate = ModelCandidate(
                model_id=model_data["model_id"],
                name=model_data["name"],
                specialization=model_data["specialization"],
                source=ModelSource.P2P_NETWORK,
                performance_score=model_data["performance_score"],
                compatibility_score=compatibility_score,
                availability_score=0.6,  # P2P availability is variable
                cost_score=0.95,  # P2P models are usually low cost
                latency_score=latency_score
            )
            candidates.append(candidate)
        
        return candidates
    
    def _is_chemistry_task(self, task_description: str) -> bool:
        """Detect if a task is chemistry-related for automatic hybrid routing"""
        task_lower = task_description.lower()
        
        # Check for chemistry patterns
        chemistry_score = 0
        for pattern in self.chemistry_patterns:
            if pattern in task_lower:
                chemistry_score += 1
                
        # Weighted scoring for compound patterns
        if "chemical" in task_lower and any(word in task_lower for word in ["reaction", "synthesis", "compound", "molecule"]):
            chemistry_score += 2
            
        if any(word in task_lower for word in ["catalyst", "catalysis"]) and any(word in task_lower for word in ["reaction", "mechanism", "pathway"]):
            chemistry_score += 2
            
        if "thermodynamic" in task_lower or "kinetic" in task_lower:
            chemistry_score += 1
            
        if any(notation in task_lower for notation in ["smiles", "chemical formula", "molecular formula"]):
            chemistry_score += 2
            
        # Decision threshold: 2 or more chemistry indicators
        is_chemistry = chemistry_score >= 2
        
        if is_chemistry:
            logger.info("Chemistry task detected",
                       task_description=task_description[:100],
                       chemistry_score=chemistry_score)
        
        return is_chemistry
    
    async def _discover_hybrid_candidates(self, task_description: str) -> List[ModelCandidate]:
        """Discover hybrid architecture candidates for chemistry tasks"""
        candidates = []
        
        try:
            # Check if chemistry hybrid executor is available
            if "chemistry" not in self.hybrid_executors:
                # Try to import and initialize chemistry hybrid executor
                try:
                    from prsm.nwtn.chemistry_hybrid_executor import ChemistryHybridExecutor
                    
                    hybrid_config = {
                        "domain": "chemistry",
                        "temperature": 0.7,
                        "enable_learning": True,
                        "enable_world_model": True,
                        "enable_bayesian_search": True
                    }
                    
                    self.hybrid_executors["chemistry"] = ChemistryHybridExecutor(hybrid_config)
                    logger.info("Chemistry hybrid executor initialized for routing")
                    
                except ImportError:
                    logger.warning("Chemistry hybrid executor not available, using mock")
                    self.hybrid_executors["chemistry"] = "mock_chemistry_hybrid"
                    
            # Create chemistry hybrid candidate - use Claude-3.5-Sonnet for robust performance
            chemistry_candidate = ModelCandidate(
                model_id="claude-3-5-sonnet-20241022",
                name="Claude-3.5-Sonnet Architecture",
                specialization="chemistry",
                model_type=ModelType.GENERAL,
                source=ModelSource.HYBRID_EXECUTOR,
                performance_score=0.95,  # High performance for chemistry tasks
                compatibility_score=0.98,  # Very high compatibility for chemistry
                availability_score=1.0,   # Always available locally
                cost_score=0.9,          # Cost-effective (local execution)
                latency_score=0.8,       # Good latency for complex reasoning
                capabilities=[
                    "chemical_reaction_prediction",
                    "thermodynamic_analysis", 
                    "kinetic_modeling",
                    "molecular_property_prediction",
                    "mechanism_pathway_analysis",
                    "catalyst_optimization",
                    "world_model_reasoning",
                    "bayesian_experimentation",
                    "first_principles_physics"
                ],
                limitations=[
                    "specialized_for_chemistry",
                    "requires_chemistry_knowledge_base"
                ]
            )
            
            candidates.append(chemistry_candidate)
            
            # Also add general hybrid executor if available
            if "general" not in self.hybrid_executors:
                try:
                    from prsm.nwtn.hybrid_architecture import create_hybrid_nwtn_engine
                    
                    # Create general hybrid agent
                    hybrid_agent = create_hybrid_nwtn_engine("general_hybrid", 0.7)
                    self.hybrid_executors["general"] = hybrid_agent
                    logger.info("General hybrid executor initialized for routing")
                    
                except ImportError:
                    logger.warning("General hybrid executor not available")
                    
            if "general" in self.hybrid_executors:
                general_candidate = ModelCandidate(
                    model_id="general_hybrid_executor",
                    name="General Hybrid Architecture",
                    specialization="general",
                    model_type=ModelType.GENERAL,
                    source=ModelSource.HYBRID_EXECUTOR,
                    performance_score=0.85,  # Good general performance
                    compatibility_score=0.8,  # Good compatibility for chemistry
                    availability_score=1.0,   # Always available locally
                    cost_score=0.9,          # Cost-effective
                    latency_score=0.7,       # Reasonable latency
                    capabilities=[
                        "world_model_reasoning",
                        "bayesian_search",
                        "first_principles_analysis",
                        "multi_domain_knowledge",
                        "soc_recognition",
                        "causal_reasoning"
                    ],
                    limitations=[
                        "general_purpose",
                        "may_lack_domain_specialization"
                    ]
                )
                
                candidates.append(general_candidate)
                
        except Exception as e:
            logger.error("Error discovering hybrid candidates", error=str(e))
            
        logger.info("Hybrid candidates discovered",
                   task_description=task_description[:50],
                   candidates_found=len(candidates))
        
        return candidates
    
    async def _categorize_task(self, task_description: str) -> str:
        """Enhanced task categorization with domain awareness"""
        task_lower = task_description.lower()
        
        # Scientific domain categorization
        if any(word in task_lower for word in ["physics", "quantum", "energy", "force", "particle"]):
            return "physics"
        elif any(word in task_lower for word in ["chemistry", "chemical", "molecule", "reaction", "catalyst"]):
            return "chemistry"
        elif any(word in task_lower for word in ["biology", "biological", "cell", "protein", "gene", "organism"]):
            return "biology"
        elif any(word in task_lower for word in ["mathematics", "math", "equation", "theorem", "proof", "calculus"]):
            return "mathematics"
        elif any(word in task_lower for word in ["computer", "algorithm", "programming", "software", "code"]):
            return "computer_science"
        
        # Task type categorization
        elif any(word in task_lower for word in ["research", "study", "investigate", "analyze"]):
            return "research"
        elif any(word in task_lower for word in ["data", "dataset", "statistics", "analytics"]):
            return "data_analysis"
        elif any(word in task_lower for word in ["explain", "define", "what is", "how does"]):
            return "explanation"
        elif any(word in task_lower for word in ["create", "generate", "build", "design"]):
            return "creation"
        elif any(word in task_lower for word in ["optimize", "improve", "enhance"]):
            return "optimization"
        elif any(word in task_lower for word in ["translate", "language", "linguistic"]):
            return "natural_language_processing"
        elif any(word in task_lower for word in ["image", "visual", "picture", "photo"]):
            return "computer_vision"
        else:
            return "general"
    
    async def _calculate_compatibility(self, task_description: str, model_details: Any, 
                                     complexity: float) -> float:
        """Enhanced compatibility calculation"""
        compatibility = 0.4  # Base compatibility
        
        # Complexity matching
        performance = self._get_model_performance_score(model_details)
        if complexity > 0.8 and performance > 0.9:
            compatibility += 0.3  # High complexity, high performance
        elif complexity > 0.6 and performance > 0.7:
            compatibility += 0.2  # Medium-high compatibility
        elif complexity < 0.3:
                compatibility += 0.1  # Simple tasks work with most models
        
        # Specialization matching
        task_category = await self._categorize_task(task_description)
        if hasattr(model_details, 'specialization'):
            if model_details.specialization == task_category:
                compatibility += 0.3  # Exact specialization match
            elif task_category in model_details.specialization or \
                 model_details.specialization in task_category:
                compatibility += 0.15  # Partial specialization match
            elif model_details.specialization == "general":
                compatibility += 0.05  # General models have slight bonus
        
        # Performance history bonus
        model_id = getattr(model_details, 'model_id', 'unknown')
        if model_id in self.performance_history:
            recent_performance = self.performance_history[model_id][-5:]  # Last 5 scores
            if recent_performance:
                avg_performance = sum(recent_performance) / len(recent_performance)
                if avg_performance > 0.8:
                    compatibility += 0.1
                elif avg_performance > 0.6:
                    compatibility += 0.05
        
        return min(compatibility, 1.0)
    
    async def _calculate_marketplace_compatibility(self, task_description: str, 
                                                 model_data: Dict[str, Any]) -> float:
        """Calculate compatibility for marketplace models"""
        compatibility = 0.5  # Base marketplace compatibility
        
        task_category = await self._categorize_task(task_description)
        specialization = model_data.get("specialization", "general")
        
        # Specialization matching
        if specialization == task_category:
            compatibility += 0.3
        elif specialization == "general":
            compatibility += 0.1
        
        # Provider reputation bonus
        reputation = model_data.get("provider_reputation", 0.5)
        compatibility += reputation * 0.2
        
        return min(compatibility, 1.0)
    
    async def _calculate_marketplace_compatibility_enhanced(self, task_description: str, 
                                                          marketplace_model) -> float:
        """Enhanced compatibility calculation for marketplace models"""
        compatibility = 0.5  # Base marketplace compatibility
        
        task_category = await self._categorize_task(task_description)
        specialization = marketplace_model.specialization
        
        # Exact specialization match
        if specialization == task_category:
            compatibility += 0.3
        elif task_category in specialization or specialization in task_category:
            compatibility += 0.15
        elif specialization == "general":
            compatibility += 0.1
        
        # Provider reputation bonus
        compatibility += marketplace_model.provider_reputation * 0.2
        
        # Capability matching
        task_lower = task_description.lower()
        for capability in marketplace_model.capabilities:
            if capability.lower() in task_lower:
                compatibility += 0.05
        
        # Performance bonus for high-performing models
        if marketplace_model.performance_score > 0.9:
            compatibility += 0.1
        elif marketplace_model.performance_score > 0.8:
            compatibility += 0.05
        
        return min(compatibility, 1.0)
    
    def _get_model_performance_score(self, model_details: Any) -> float:
        """Safely get performance score from model details (dict or object)"""
        if hasattr(model_details, 'performance_score'):
            return model_details.performance_score
        elif isinstance(model_details, dict) and 'performance_score' in model_details:
            return model_details['performance_score']
        else:
            return 0.8  # Default performance score
    
    def _get_model_specialization(self, model_details: Any) -> str:
        """Safely get specialization from model details (dict or object)"""
        if hasattr(model_details, 'specialization'):
            return model_details.specialization
        elif isinstance(model_details, dict) and 'specialization' in model_details:
            return model_details['specialization']
        else:
            return "general"  # Default specialization
    
    def _calculate_cost_score(self, cost_per_token: Optional[float]) -> float:
        """Calculate cost score (lower cost = higher score)"""
        if cost_per_token is None:
            return 0.5  # Default score for unknown cost
        
        # Normalize cost to score (0.05 = max cost for score calculation)
        max_cost = 0.05
        cost_score = max(0.1, 1.0 - (cost_per_token / max_cost))
        return min(cost_score, 1.0)
    
    def _calculate_latency_score(self, estimated_latency: Optional[float]) -> float:
        """Calculate latency score (lower latency = higher score)"""
        if estimated_latency is None:
            return 0.5  # Default score for unknown latency
        
        # Normalize latency to score (10.0 = max latency for score calculation)
        max_latency = 10.0
        latency_score = max(0.1, 1.0 - (estimated_latency / max_latency))
        return min(latency_score, 1.0)
    
    async def _extract_domain_from_task(self, task_description: str) -> str:
        """Extract domain for teacher selection"""
        task_category = await self._categorize_task(task_description)
        
        # Map task categories to domains
        domain_mapping = {
            "physics": "physics",
            "chemistry": "chemistry", 
            "biology": "biology",
            "mathematics": "mathematics",
            "computer_science": "computer_science"
        }
        
        return domain_mapping.get(task_category, "any")
    
    def _calculate_confidence(self, candidates: List[ModelCandidate]) -> float:
        """Calculate confidence in routing decision"""
        if not candidates:
            return 0.0
        
        # Base confidence on score distribution
        top_score = candidates[0].overall_score
        
        if len(candidates) == 1:
            return top_score
        
        # Calculate score gap between top candidates
        second_score = candidates[1].overall_score if len(candidates) > 1 else 0.0
        score_gap = top_score - second_score
        
        # Higher confidence when there's a clear winner
        confidence = top_score * (0.7 + 0.3 * min(score_gap * 2, 1.0))
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, task_description: str, candidate: ModelCandidate, 
                          strategy: RoutingStrategy) -> str:
        """Generate human-readable reasoning for the routing decision"""
        reasons = []
        
        # Primary selection reason
        if candidate.source == ModelSource.LOCAL_REGISTRY:
            reasons.append(f"Selected local model '{candidate.name or candidate.model_id}'")
        elif candidate.source == ModelSource.MARKETPLACE:
            reasons.append(f"Selected marketplace model '{candidate.name}'")
        elif candidate.source == ModelSource.TEACHER_POOL:
            reasons.append(f"Selected teacher model '{candidate.name}'")
        else:
            reasons.append(f"Selected P2P model '{candidate.name or candidate.model_id}'")
        
        # Specialization reasoning
        if candidate.specialization != "general":
            reasons.append(f"specializing in {candidate.specialization}")
        
        # Strategy-specific reasoning
        if strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            reasons.append(f"for optimal performance (score: {candidate.performance_score:.2f})")
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            reasons.append(f"for cost efficiency (cost score: {candidate.cost_score:.2f})")
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            reasons.append(f"for low latency (latency score: {candidate.latency_score:.2f})")
        elif strategy == RoutingStrategy.TEACHER_SELECTION:
            effectiveness = candidate.teaching_effectiveness or 0.0
            reasons.append(f"for teaching effectiveness ({effectiveness:.2f})")
        elif strategy == RoutingStrategy.HYBRID_CHEMISTRY:
            if candidate.source == ModelSource.HYBRID_EXECUTOR:
                reasons.append(f"using hybrid architecture for superior chemistry reasoning")
                if candidate.specialization == "chemistry":
                    reasons.append("with domain-specific chemistry knowledge")
                reasons.append("featuring System 1 + System 2 architecture")
            else:
                reasons.append(f"using traditional approach")
        
        # Confidence indicator
        reasons.append(f"with {candidate.overall_score:.0%} overall suitability")
        
        return "; ".join(reasons)
    
    def _estimate_cost(self, candidate: ModelCandidate) -> Optional[float]:
        """Estimate cost for using this model"""
        if candidate.cost_per_token:
            # Estimate based on typical token usage (1000 tokens)
            return candidate.cost_per_token * 1000
        elif candidate.source == ModelSource.LOCAL_REGISTRY:
            return 0.0  # Local models are free
        elif candidate.source == ModelSource.P2P_NETWORK:
            return 0.1  # P2P models have minimal cost
        else:
            return None  # Unknown cost
    
    def _estimate_latency(self, candidate: ModelCandidate) -> Optional[float]:
        """Estimate latency for this model"""
        if candidate.estimated_latency:
            return candidate.estimated_latency
        elif candidate.source == ModelSource.LOCAL_REGISTRY:
            return 0.5  # Local models are fast
        elif candidate.source == ModelSource.MARKETPLACE:
            return 2.0  # Marketplace models have API overhead
        elif candidate.source == ModelSource.P2P_NETWORK:
            return 5.0  # P2P models can be slower
        else:
            return None
    
    # Enhanced public interface methods
    async def match_to_specialist(self, task: ArchitectTask) -> List[ModelCandidate]:
        """Match task to specialist models (execution plan interface)"""
        decision = await self.process(task, {"strategy": "performance_optimized"})
        candidates = [decision.primary_candidate] + decision.backup_candidates
        return candidates
    
    async def select_teacher_for_training(self, student_model: str, domain: str) -> str:
        """Select teacher model for student training (execution plan interface)"""
        task_description = f"Teach {domain} concepts to student model {student_model}"
        decision = await self.process(task_description, {"strategy": "teacher_selection"})
        return decision.primary_candidate.model_id
    
    async def route_to_marketplace(self, task: ArchitectTask) -> MarketplaceRequest:
        """Route task to marketplace models (execution plan interface)"""
        decision = await self.process(task, {"strategy": "marketplace_preferred"})
        
        return MarketplaceRequest(
            task_description=task.instruction,
            domain_requirements=[task.required_expertise] if hasattr(task, 'required_expertise') else [],
            performance_requirements={"min_score": 0.7},
            budget_limit=100.0,
            latency_requirements=5.0,
            quality_threshold=0.8
        )
    
    async def route_with_strategy(self, task_description: str, strategy: str) -> RoutingDecision:
        """Route with specific strategy"""
        return await self.process(task_description, {"strategy": strategy})
    
    async def get_routing_analytics(self) -> Dict[str, Any]:
        """Get routing performance analytics"""
        if not self.routing_decisions:
            return {"total_decisions": 0}
        
        total_decisions = len(self.routing_decisions)
        avg_confidence = sum(d.confidence_score for d in self.routing_decisions) / total_decisions
        avg_routing_time = sum(d.routing_time for d in self.routing_decisions) / total_decisions
        
        # Strategy usage
        strategy_usage = {}
        for decision in self.routing_decisions:
            strategy = decision.strategy_used.value if hasattr(decision.strategy_used, 'value') else str(decision.strategy_used)
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        # Source distribution
        source_usage = {}
        for decision in self.routing_decisions:
            source = decision.primary_candidate.source.value if hasattr(decision.primary_candidate.source, 'value') else str(decision.primary_candidate.source)
            source_usage[source] = source_usage.get(source, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "average_confidence": avg_confidence,
            "average_routing_time": avg_routing_time,
            "strategy_usage": strategy_usage,
            "source_distribution": source_usage,
            "cache_hit_rate": len(self.routing_cache) / max(total_decisions, 1)
        }
    
    async def record_execution_feedback(self, decision_id: UUID, model_id: str, 
                                      metrics: Dict[str, float], 
                                      context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record performance feedback after model execution
        
        Args:
            decision_id: ID of the routing decision
            model_id: ID of the model that was executed
            metrics: Performance metrics (response_time, accuracy, success, etc.)
            context: Additional context (task_id, user_id, etc.)
            
        Returns:
            True if feedback was recorded successfully
        """
        try:
            # Record individual metrics
            for metric_name, value in metrics.items():
                try:
                    metric_type = MetricType(metric_name.lower())
                    await performance_tracker.record_metric(
                        model_id=model_id,
                        metric_type=metric_type,
                        value=value,
                        context=context
                    )
                except ValueError:
                    # Skip unknown metric types
                    logger.debug("Unknown metric type", metric_name=metric_name)
                    continue
            
            # Update local performance history
            if "accuracy" in metrics:
                self.update_model_performance(model_id, metrics["accuracy"])
            
            logger.info("Execution feedback recorded",
                       decision_id=str(decision_id),
                       model_id=model_id,
                       metrics=list(metrics.keys()))
            
            return True
            
        except Exception as e:
            logger.error("Failed to record execution feedback",
                        decision_id=str(decision_id),
                        model_id=model_id,
                        error=str(e))
            return False
    
    async def get_model_performance_insights(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive performance insights for a model"""
        try:
            profile = await performance_tracker.get_model_performance(model_id)
            
            if not profile:
                return {"error": "Model performance data not found"}
            
            # Get performance trends
            trends = await performance_tracker.get_performance_trends(model_id, hours=168)  # 1 week
            
            # Get performance issues
            issues = await performance_tracker.detect_performance_issues(model_id)
            
            return {
                "model_id": model_id,
                "performance_profile": {
                    "overall_rank": profile.overall_rank,
                    "performance_grade": profile.performance_grade.value,
                    "accuracy_score": profile.accuracy_score,
                    "response_time_avg": profile.response_time_avg,
                    "success_rate": profile.success_rate,
                    "availability_score": profile.availability_score,
                    "total_requests": profile.total_requests,
                    "last_used": profile.last_used.isoformat() if profile.last_used else None
                },
                "trends": {
                    metric_type.value: len(values) for metric_type, values in trends.items()
                },
                "issues": issues,
                "recommendations": await self._generate_performance_recommendations(profile, issues)
            }
            
        except Exception as e:
            logger.error("Failed to get performance insights", model_id=model_id, error=str(e))
            return {"error": str(e)}
    
    async def get_top_performing_models(self, category: Optional[str] = None, 
                                      limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing models with routing context"""
        try:
            top_models = await performance_tracker.get_top_models(category=category, limit=limit)
            
            results = []
            for profile in top_models:
                # Get recent routing decisions for this model
                recent_decisions = [
                    d for d in self.routing_decisions[-100:]  # Last 100 decisions
                    if d.primary_candidate.model_id == profile.model_id
                ]
                
                results.append({
                    "model_id": profile.model_id,
                    "model_name": profile.model_name,
                    "provider": profile.provider,
                    "performance_grade": profile.performance_grade.value,
                    "overall_rank": profile.overall_rank,
                    "key_metrics": {
                        "accuracy": profile.accuracy_score,
                        "response_time": profile.response_time_avg,
                        "success_rate": profile.success_rate,
                        "availability": profile.availability_score
                    },
                    "usage_stats": {
                        "total_requests": profile.total_requests,
                        "recent_routing_decisions": len(recent_decisions),
                        "last_used": profile.last_used.isoformat() if profile.last_used else None
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error("Failed to get top performing models", error=str(e))
            return []
    
    async def _generate_performance_recommendations(self, profile: ModelPerformanceProfile, 
                                                  issues: List[str]) -> List[str]:
        """Generate actionable performance recommendations"""
        recommendations = []
        
        # Response time recommendations
        if profile.response_time_avg > 5.0:
            recommendations.append("Consider using this model for non-time-critical tasks")
        elif profile.response_time_avg < 1.0:
            recommendations.append("Excellent for real-time applications")
        
        # Accuracy recommendations
        if profile.accuracy_score > 0.9:
            recommendations.append("Suitable for high-accuracy requirements")
        elif profile.accuracy_score < 0.7:
            recommendations.append("May need additional validation or fallback models")
        
        # Usage recommendations
        if profile.total_requests < 10:
            recommendations.append("Limited usage data - performance metrics may be unreliable")
        
        # Issue-based recommendations
        if "Critical success rate" in str(issues):
            recommendations.append("Avoid using for production workloads until issues are resolved")
        
        if "Response time degrading" in str(issues):
            recommendations.append("Monitor closely - performance may be declining")
        
        return recommendations
    
    def update_model_performance(self, model_id: str, performance_score: float):
        """Update performance history for a model"""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        self.performance_history[model_id].append(performance_score)
        
        # Keep only last 20 scores
        if len(self.performance_history[model_id]) > 20:
            self.performance_history[model_id] = self.performance_history[model_id][-20:]
        
        logger.info("Model performance updated",
                   model_id=model_id,
                   performance_score=performance_score,
                   history_length=len(self.performance_history[model_id]))
    
    # Simplified interface methods for backward compatibility
    async def route_to_best_model(self, task_description: str) -> Optional[str]:
        """Route to single best model for a task"""
        decision = await self.process(task_description)
        return decision.primary_candidate.model_id
    
    async def route_to_multiple_models(self, task_description: str, 
                                     count: int = 3) -> List[str]:
        """Route to multiple models for parallel execution"""
        decision = await self.process(task_description)
        candidates = [decision.primary_candidate] + decision.backup_candidates
        return [c.model_id for c in candidates[:count]]
    
    def clear_cache(self):
        """Clear routing cache"""
        cache_size = len(self.routing_cache)
        self.routing_cache.clear()
        logger.info("Routing cache cleared",
                   agent_id=self.agent_id,
                   cleared_entries=cache_size)
    
    # ===============================
    # MCP Tool Integration Methods
    # ===============================
    
    async def route_tool_request(self, model_id: str, tool_request: ToolRequest) -> ToolRoutingDecision:
        """
        Route a tool request from a model to appropriate MCP tools
        
        This method enables models to request tools during execution, creating
        powerful tool-augmented AI workflows where models can:
        - Access real-time data through APIs
        - Perform calculations and computations
        - Interact with file systems and databases
        - Execute code in secure sandboxes
        - Communicate with external services
        
        Args:
            model_id: ID of the requesting model
            tool_request: Specification of tool requirements
            
        Returns:
            ToolRoutingDecision: Complete routing decision with selected tools
        """
        logger.info("Processing tool request from model",
                   agent_id=self.agent_id,
                   model_id=model_id,
                   request_id=str(tool_request.request_id))
        
        # Route through tool router
        decision = await self.tool_router.process(tool_request)
        
        # Store decision for analytics
        self.tool_routing_decisions.append(decision)
        
        # Update model-tool associations
        if model_id not in self.model_tool_associations:
            self.model_tool_associations[model_id] = []
        
        tool_id = decision.primary_tool.tool_spec.tool_id
        if tool_id not in self.model_tool_associations[model_id]:
            self.model_tool_associations[model_id].append(tool_id)
        
        logger.info("Tool request routed successfully",
                   agent_id=self.agent_id,
                   model_id=model_id,
                   selected_tool=tool_id,
                   confidence=decision.confidence_score)
        
        return decision
    
    async def get_tools_for_model(self, model_id: str, task_description: str = None) -> List[str]:
        """
        Get recommended tools for a specific model and task
        
        This method analyzes a model's capabilities and the current task
        to recommend appropriate MCP tools that could enhance the model's
        performance and capabilities.
        
        Args:
            model_id: ID of the model
            task_description: Optional description of the current task
            
        Returns:
            List of recommended tool IDs
        """
        # Get model details for capability analysis
        model_details = await self.model_registry.get_model_details(model_id)
        
        if not model_details:
            logger.warning("Model not found for tool recommendation", model_id=model_id)
            return []
        
        # Create tool request based on model capabilities and task
        if task_description:
            tool_request = ToolRequest(
                model_id=model_id,
                task_description=task_description
            )
        else:
            # Recommend general tools based on model specialization
            specialization = getattr(model_details, 'specialization', 'general')
            tool_request = ToolRequest(
                model_id=model_id,
                task_description=f"General {specialization} tasks"
            )
        
        # Get tool routing decision
        decision = await self.tool_router.process(tool_request)
        
        # Return all suitable tools (primary + backups)
        recommended_tools = [decision.primary_tool.tool_spec.tool_id]
        recommended_tools.extend([t.tool_spec.tool_id for t in decision.backup_tools])
        
        return recommended_tools[:5]  # Limit to top 5 recommendations
    
    async def associate_model_with_tools(self, model_id: str, tool_ids: List[str]) -> bool:
        """
        Create persistent association between a model and tools
        
        This enables automatic tool availability for models, creating
        specialized tool-augmented model configurations.
        
        Args:
            model_id: ID of the model
            tool_ids: List of tool IDs to associate
            
        Returns:
            True if association was successful
        """
        try:
            # Validate that tools exist
            valid_tools = []
            for tool_id in tool_ids:
                tool = self.tool_router.tool_registry.get_tool(tool_id)
                if tool:
                    valid_tools.append(tool_id)
                else:
                    logger.warning("Tool not found for association", 
                                  tool_id=tool_id, model_id=model_id)
            
            if not valid_tools:
                return False
            
            # Create/update association
            self.model_tool_associations[model_id] = valid_tools
            
            logger.info("Model-tool association created",
                       agent_id=self.agent_id,
                       model_id=model_id,
                       tool_count=len(valid_tools))
            
            return True
            
        except Exception as e:
            logger.error("Failed to associate model with tools",
                        model_id=model_id,
                        error=str(e))
            return False
    
    async def execute_model_with_tools(self, model_id: str, task: str, 
                                     available_tools: List[str] = None) -> Dict[str, Any]:
        """
        Execute a model with access to specified MCP tools
        
        This method coordinates the execution of a model while providing
        it with access to MCP tools for enhanced capabilities. The model
        can request tools during execution through the tool router.
        
        Args:
            model_id: ID of the model to execute
            task: Task description for the model
            available_tools: List of tool IDs available to the model
            
        Returns:
            Execution result with tool usage information
        """
        start_time = time.time()
        
        # Get associated tools if none specified
        if available_tools is None:
            available_tools = self.model_tool_associations.get(model_id, [])
        
        logger.info("Executing model with tool access",
                   agent_id=self.agent_id,
                   model_id=model_id,
                   available_tools=len(available_tools))
        
        try:
            # Create execution context with tool access
            execution_context = {
                "model_id": model_id,
                "available_tools": available_tools,
                "tool_router": self.tool_router,
                "tool_enabled": True
            }
            
            # For now, simulate model execution with tool access
            # In production, this would integrate with the actual model executor
            result = await self._simulate_tool_enhanced_execution(
                model_id, task, available_tools
            )
            
            execution_time = time.time() - start_time
            
            return {
                "model_id": model_id,
                "task": task,
                "result": result,
                "execution_time": execution_time,
                "tools_used": result.get("tools_used", []),
                "tool_execution_count": result.get("tool_execution_count", 0),
                "success": True
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Model execution with tools failed",
                        model_id=model_id,
                        error=str(e))
            
            return {
                "model_id": model_id,
                "task": task,
                "result": None,
                "execution_time": execution_time,
                "error": str(e),
                "success": False
            }
    
    async def _simulate_tool_enhanced_execution(self, model_id: str, task: str, 
                                              available_tools: List[str]) -> Dict[str, Any]:
        """Simulate model execution with tool access"""
        task_lower = task.lower()
        tools_used = []
        tool_execution_count = 0
        
        # Simulate tool usage based on task characteristics
        if "search" in task_lower or "find" in task_lower:
            if "web_search" in available_tools:
                tools_used.append("web_search")
                tool_execution_count += 1
        
        if "file" in task_lower or "read" in task_lower:
            if "file_reader" in available_tools:
                tools_used.append("file_reader")
                tool_execution_count += 1
        
        if "calculate" in task_lower or "compute" in task_lower:
            if "python_executor" in available_tools:
                tools_used.append("python_executor")
                tool_execution_count += 1
        
        if "database" in task_lower or "query" in task_lower:
            if "database_query" in available_tools:
                tools_used.append("database_query")
                tool_execution_count += 1
        
        # Generate enhanced result
        base_result = f"Enhanced analysis of: {task}"
        if tools_used:
            base_result += f"\n\nTools utilized: {', '.join(tools_used)}"
            base_result += f"\nTool-enhanced insights: Additional data and capabilities provided through {len(tools_used)} tools"
        
        return {
            "type": "tool_enhanced_response",
            "content": base_result,
            "tools_used": tools_used,
            "tool_execution_count": tool_execution_count,
            "enhancement_factor": len(tools_used) * 0.2 + 1.0,  # 20% improvement per tool
            "confidence": 0.9 + len(tools_used) * 0.02  # Higher confidence with tools
        }
    
    def get_tool_usage_analytics(self) -> Dict[str, Any]:
        """Get analytics on tool usage by models"""
        return {
            "total_tool_routing_decisions": len(self.tool_routing_decisions),
            "models_with_tool_access": len(self.model_tool_associations),
            "tool_associations": dict(self.model_tool_associations),
            "recent_tool_decisions": [
                {
                    "model_id": "unknown",  # Would need to track this
                    "tool_selected": d.primary_tool.tool_spec.tool_id,
                    "confidence": d.confidence_score,
                    "routing_time": d.routing_time
                }
                for d in self.tool_routing_decisions[-10:]
            ],
            "tool_router_analytics": self.tool_router.get_tool_analytics()
        }
    
    async def optimize_model_tool_pairing(self, model_id: str) -> Dict[str, Any]:
        """
        Optimize tool selection for a specific model based on usage patterns
        
        Analyzes historical performance to recommend the best tool combinations
        for a given model, creating optimized tool-augmented configurations.
        
        Args:
            model_id: ID of the model to optimize
            
        Returns:
            Optimization recommendations and metrics
        """
        # Get historical performance data
        model_decisions = [
            d for d in self.tool_routing_decisions 
            if hasattr(d, 'model_id') and getattr(d, 'model_id', None) == model_id
        ]
        
        if not model_decisions:
            return {
                "model_id": model_id,
                "optimization_available": False,
                "reason": "Insufficient usage data for optimization"
            }
        
        # Analyze tool performance for this model
        tool_performance = {}
        for decision in model_decisions:
            tool_id = decision.primary_tool.tool_spec.tool_id
            if tool_id not in tool_performance:
                tool_performance[tool_id] = {
                    "usage_count": 0,
                    "avg_confidence": 0.0,
                    "avg_routing_time": 0.0
                }
            
            tool_performance[tool_id]["usage_count"] += 1
            tool_performance[tool_id]["avg_confidence"] += decision.confidence_score
            tool_performance[tool_id]["avg_routing_time"] += decision.routing_time
        
        # Calculate averages
        for tool_id, stats in tool_performance.items():
            count = stats["usage_count"]
            stats["avg_confidence"] /= count
            stats["avg_routing_time"] /= count
        
        # Generate recommendations
        recommended_tools = sorted(
            tool_performance.items(),
            key=lambda x: (x[1]["avg_confidence"], -x[1]["avg_routing_time"]),
            reverse=True
        )[:5]  # Top 5 tools
        
        return {
            "model_id": model_id,
            "optimization_available": True,
            "total_decisions_analyzed": len(model_decisions),
            "recommended_tools": [
                {
                    "tool_id": tool_id,
                    "usage_count": stats["usage_count"],
                    "avg_confidence": stats["avg_confidence"],
                    "avg_routing_time": stats["avg_routing_time"]
                }
                for tool_id, stats in recommended_tools
            ],
            "optimization_score": sum(stats["avg_confidence"] for _, stats in recommended_tools) / len(recommended_tools) if recommended_tools else 0.0
        }


# Factory function
def create_enhanced_router(model_registry: Optional[ModelRegistry] = None) -> ModelRouter:
    """Create an enhanced model router agent"""
    return ModelRouter(model_registry=model_registry)
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


class ModelSource(str, Enum):
    """Source of model availability"""
    LOCAL_REGISTRY = "local_registry"
    MARKETPLACE = "marketplace"
    P2P_NETWORK = "p2p_network"
    TEACHER_POOL = "teacher_pool"


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
    Enhanced Model Router for PRSM
    
    Advanced routing capabilities:
    - Intelligent specialist matching with domain expertise
    - Marketplace integration for external model discovery
    - Teacher model selection for student training
    - Multi-strategy routing optimization
    - Performance-based adaptive routing
    """
    
    def __init__(self, model_registry: Optional[ModelRegistry] = None, agent_id: Optional[str] = None):
        super().__init__(agent_id=agent_id, agent_type=AgentType.ROUTER)
        self.model_registry = model_registry or ModelRegistry()
        self.routing_cache: Dict[str, List[ModelCandidate]] = {}
        self.routing_decisions: List[RoutingDecision] = []
        self.performance_history: Dict[str, List[float]] = {}  # model_id -> performance scores
        self.marketplace_endpoints: List[str] = self._initialize_marketplace_endpoints()
        self.teacher_pool: Dict[str, List[TeacherModel]] = {}  # domain -> teachers
        
        logger.info("Enhanced ModelRouter initialized",
                   agent_id=self.agent_id,
                   marketplace_endpoints=len(self.marketplace_endpoints))
    
    def _initialize_marketplace_endpoints(self) -> List[str]:
        """Initialize marketplace endpoints for model discovery"""
        # In production, these would be loaded from configuration
        return [
            "https://api.huggingface.co/models",
            "https://api.openai.com/models", 
            "https://api.anthropic.com/models",
            "https://api.cohere.ai/models"
        ]
    
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
        
        # 4. P2P network candidates
        p2p_candidates = await self._discover_p2p_candidates(task_description)
        candidates.extend(p2p_candidates)
        
        return candidates
    
    async def _discover_local_candidates(self, task_description: str, complexity: float) -> List[ModelCandidate]:
        """Discover candidates from local model registry"""
        candidates = []
        task_category = await self._categorize_task(task_description)
        
        # Get specialists for the task category
        specialist_ids = await self.model_registry.discover_specialists(task_category)
        
        for model_id in specialist_ids:
            model_details = await self.model_registry.get_model_details(model_id)
            if model_details:
                compatibility_score = await self._calculate_compatibility(
                    task_description, model_details, complexity
                )
                
                candidate = ModelCandidate(
                    model_id=model_id,
                    name=getattr(model_details, 'name', model_id),
                    specialization=model_details.specialization,
                    source=ModelSource.LOCAL_REGISTRY,
                    performance_score=model_details.performance_score,
                    compatibility_score=compatibility_score,
                    availability_score=1.0,  # Local models are always available
                    cost_score=0.9,  # Local models are cost-effective
                    latency_score=0.8,  # Local models have good latency
                    capabilities=getattr(model_details, 'capabilities', []),
                    limitations=getattr(model_details, 'limitations', [])
                )
                candidates.append(candidate)
        
        return candidates
    
    async def _discover_marketplace_candidates(self, task_description: str) -> List[ModelCandidate]:
        """Discover candidates from marketplace APIs"""
        candidates = []
        
        # Simulate marketplace discovery (in production, this would make real API calls)
        marketplace_models = [
            {
                "model_id": "marketplace_gpt4",
                "name": "GPT-4 Marketplace",
                "specialization": "general",
                "performance_score": 0.95,
                "cost_per_token": 0.03,
                "estimated_latency": 2.5,
                "provider_reputation": 0.9,
                "marketplace_url": "https://api.openai.com/v1/chat/completions"
            },
            {
                "model_id": "marketplace_claude",
                "name": "Claude-3 Marketplace", 
                "specialization": "reasoning",
                "performance_score": 0.92,
                "cost_per_token": 0.025,
                "estimated_latency": 2.0,
                "provider_reputation": 0.85,
                "marketplace_url": "https://api.anthropic.com/v1/messages"
            }
        ]
        
        for model_data in marketplace_models:
            compatibility_score = await self._calculate_marketplace_compatibility(
                task_description, model_data
            )
            
            # Calculate cost and latency scores (lower values = higher scores)
            cost_score = max(0.1, 1.0 - (model_data["cost_per_token"] / 0.05))
            latency_score = max(0.1, 1.0 - (model_data["estimated_latency"] / 10.0))
            
            candidate = ModelCandidate(
                model_id=model_data["model_id"],
                name=model_data["name"],
                specialization=model_data["specialization"],
                source=ModelSource.MARKETPLACE,
                performance_score=model_data["performance_score"],
                compatibility_score=compatibility_score,
                availability_score=0.8,  # Marketplace availability varies
                cost_score=cost_score,
                latency_score=latency_score,
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
        """Discover candidates from P2P network"""
        candidates = []
        
        # Simulate P2P model discovery
        p2p_models = [
            {
                "model_id": "p2p_specialist_01",
                "name": "Community Specialist",
                "specialization": "research",
                "performance_score": 0.75,
                "estimated_latency": 5.0
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
        if hasattr(model_details, 'performance_score'):
            performance = model_details.performance_score
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


# Factory function
def create_enhanced_router(model_registry: Optional[ModelRegistry] = None) -> ModelRouter:
    """Create an enhanced model router agent"""
    return ModelRouter(model_registry=model_registry)
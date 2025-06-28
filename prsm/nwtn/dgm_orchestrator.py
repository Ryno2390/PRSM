"""
DGM-Enhanced NWTN Orchestrator

Production implementation of Darwin Gödel Machine enhanced TaskOrchestrator
for PRSM's core orchestration system. Enables recursive self-improvement
of orchestration strategies through empirical validation.

Based on the DGM-Enhanced Evolution System roadmap Phase 2.1.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import uuid
from decimal import Decimal

# PRSM Core imports
from prsm.core.models import (
    UserInput, PRSMSession, ClarifiedPrompt, PRSMResponse,
    ReasoningStep, AgentType, TaskStatus, ArchitectTask,
    TaskHierarchy, AgentResponse, SafetyFlag, ContextUsage
)
from prsm.core.config import get_settings
from prsm.nwtn.context_manager import ContextManager
from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.tokenomics.ftns_service import FTNSService

# DGM Evolution imports
from ..evolution.archive import EvolutionArchive, SolutionNode
from ..evolution.self_modification import SelfModifyingComponent
from ..evolution.exploration import OpenEndedExplorationEngine
from ..evolution.models import (
    ComponentType, ModificationProposal, EvaluationResult, 
    ModificationResult, Checkpoint, SafetyStatus, RiskLevel, ImpactLevel,
    SelectionStrategy
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class OrchestrationPattern:
    """Orchestration pattern configuration for evolution."""
    
    # Core orchestration strategy
    routing_strategy: str = "intelligent_adaptive"
    load_balancing_algorithm: str = "least_response_time"
    parallel_execution_factor: float = 0.7
    
    # Context allocation strategy
    context_allocation_method: str = "dynamic_budget"
    context_reserve_ratio: float = 0.2
    max_context_per_task: int = 4096
    
    # Agent coordination
    agent_selection_criteria: List[str] = field(default_factory=lambda: ["performance", "cost", "latency"])
    hierarchy_depth_limit: int = 5
    timeout_strategy: str = "adaptive"
    
    # Performance optimization
    caching_strategy: str = "semantic_aware"
    prefetch_enabled: bool = True
    result_compression: bool = True
    
    # Quality control
    quality_threshold: float = 0.85
    error_retry_strategy: str = "exponential_backoff"
    safety_validation_level: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization."""
        return {
            "routing_strategy": self.routing_strategy,
            "load_balancing_algorithm": self.load_balancing_algorithm,
            "parallel_execution_factor": self.parallel_execution_factor,
            "context_allocation_method": self.context_allocation_method,
            "context_reserve_ratio": self.context_reserve_ratio,
            "max_context_per_task": self.max_context_per_task,
            "agent_selection_criteria": self.agent_selection_criteria,
            "hierarchy_depth_limit": self.hierarchy_depth_limit,
            "timeout_strategy": self.timeout_strategy,
            "caching_strategy": self.caching_strategy,
            "prefetch_enabled": self.prefetch_enabled,
            "result_compression": self.result_compression,
            "quality_threshold": self.quality_threshold,
            "error_retry_strategy": self.error_retry_strategy,
            "safety_validation_level": self.safety_validation_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrchestrationPattern':
        """Create pattern from dictionary."""
        return cls(**data)


@dataclass
class OrchestrationMetrics:
    """Performance metrics for orchestration evaluation."""
    
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Performance metrics
    total_latency_ms: float = 0.0
    processing_latency_ms: float = 0.0
    network_latency_ms: float = 0.0
    
    # Success metrics
    success: bool = False
    quality_score: float = 0.0
    safety_violations: int = 0
    
    # Resource metrics
    context_tokens_used: int = 0
    ftns_cost: Decimal = Decimal('0')
    compute_units_consumed: float = 0.0
    
    # Agent metrics
    agents_invoked: int = 0
    parallel_tasks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Error metrics
    errors_encountered: int = 0
    retries_performed: int = 0
    timeouts_occurred: int = 0
    
    @property
    def duration_seconds(self) -> float:
        """Calculate total duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def throughput_tasks_per_second(self) -> float:
        """Calculate task throughput."""
        if self.duration_seconds > 0:
            return self.agents_invoked / self.duration_seconds
        return 0.0
    
    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            return self.cache_hits / total_cache_requests
        return 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.agents_invoked > 0:
            return self.errors_encountered / self.agents_invoked
        return 0.0


class DGMEnhancedNWTNOrchestrator(NWTNOrchestrator, SelfModifyingComponent):
    """
    DGM-Enhanced NWTN Orchestrator with recursive self-improvement capabilities.
    
    Extends the core NWTNOrchestrator with Darwin Gödel Machine evolution
    capabilities, enabling the orchestrator to automatically discover and
    implement better orchestration strategies through empirical validation.
    
    Key Features:
    - Archive-based evolution of orchestration patterns
    - Empirical performance validation
    - Open-ended exploration of orchestration strategies
    - Safety-constrained self-modification
    - Performance-driven recursive improvement
    
    Integration:
    - Maintains full compatibility with existing NWTN architecture
    - Preserves all safety constraints and governance requirements
    - Extends functionality without breaking existing interfaces
    """
    
    def __init__(
        self,
        orchestrator_id: Optional[str] = None,
        context_manager: Optional[ContextManager] = None,
        ftns_service: Optional[FTNSService] = None,
        enable_evolution: bool = True
    ):
        # Initialize base orchestrator
        NWTNOrchestrator.__init__(self, context_manager, ftns_service)
        
        # Initialize DGM components
        orchestrator_id = orchestrator_id or f"nwtn_orchestrator_{uuid.uuid4().hex[:8]}"
        SelfModifyingComponent.__init__(self, orchestrator_id, ComponentType.TASK_ORCHESTRATOR)
        
        # Evolution system components
        self.enable_evolution = enable_evolution
        if enable_evolution:
            self.orchestration_archive = EvolutionArchive(
                archive_id=f"{orchestrator_id}_patterns",
                component_type=ComponentType.TASK_ORCHESTRATOR
            )
            self.exploration_engine = OpenEndedExplorationEngine(self.orchestration_archive)
        
        # Current orchestration pattern
        self.current_pattern = OrchestrationPattern()
        self.current_solution_id: Optional[str] = None
        
        # Performance tracking
        self.session_metrics: Dict[str, OrchestrationMetrics] = {}
        self.performance_history: List[EvaluationResult] = []
        self.evaluation_window = 100  # Evaluate over last 100 sessions
        
        # Pattern evolution configuration
        self.evolution_threshold = 0.05  # 5% improvement required for pattern adoption
        self.evolution_frequency = 50  # Evolve every 50 sessions
        self.session_count = 0
        
        # Initialize archive with current pattern
        if enable_evolution:
            asyncio.create_task(self._initialize_orchestration_archive())
    
    async def _initialize_orchestration_archive(self):
        """Initialize archive with current orchestration pattern."""
        try:
            initial_solution = SolutionNode(
                component_type=ComponentType.TASK_ORCHESTRATOR,
                configuration=self.current_pattern.to_dict(),
                generation=0
            )
            
            # Evaluate initial performance with baseline metrics
            initial_performance = await self._create_baseline_evaluation()
            initial_solution.add_evaluation(initial_performance)
            
            self.current_solution_id = await self.orchestration_archive.add_solution(initial_solution)
            
            logger.info(f"DGM orchestrator initialized with solution {self.current_solution_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestration archive: {e}")
    
    async def orchestrate_session(
        self, 
        user_input: UserInput, 
        session: PRSMSession
    ) -> PRSMResponse:
        """
        Enhanced orchestration with performance tracking and evolution.
        
        Args:
            user_input: User's input to process
            session: Current PRSM session
            
        Returns:
            PRSMResponse with complete orchestration results
        """
        session_start = datetime.utcnow()
        metrics = OrchestrationMetrics(
            session_id=session.session_id,
            start_time=session_start
        )
        
        try:
            # Store metrics for tracking
            self.session_metrics[session.session_id] = metrics
            
            # Execute core orchestration with pattern-based configuration
            response = await self._execute_pattern_based_orchestration(
                user_input, session, metrics
            )
            
            # Finalize metrics
            metrics.end_time = datetime.utcnow()
            metrics.success = response.status == TaskStatus.COMPLETED
            metrics.quality_score = self._calculate_response_quality(response)
            
            # Increment session count and check for evolution trigger
            self.session_count += 1
            if (self.enable_evolution and 
                self.session_count % self.evolution_frequency == 0):
                asyncio.create_task(self._trigger_orchestration_evolution())
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestration failed for session {session.session_id}: {e}")
            metrics.end_time = datetime.utcnow()
            metrics.success = False
            metrics.errors_encountered += 1
            
            # Return error response
            return PRSMResponse(
                session_id=session.session_id,
                response_id=str(uuid.uuid4()),
                status=TaskStatus.FAILED,
                error_message=str(e),
                reasoning_trace=[
                    ReasoningStep(
                        step_id=str(uuid.uuid4()),
                        description=f"Orchestration failed: {e}",
                        timestamp=datetime.utcnow()
                    )
                ],
                context_usage=ContextUsage(
                    tokens_allocated=0,
                    tokens_used=0,
                    cost_ftns=Decimal('0')
                )
            )
    
    async def _execute_pattern_based_orchestration(
        self,
        user_input: UserInput,
        session: PRSMSession,
        metrics: OrchestrationMetrics
    ) -> PRSMResponse:
        """Execute orchestration using current pattern configuration."""
        
        # Apply current orchestration pattern
        await self._apply_orchestration_pattern(self.current_pattern, metrics)
        
        # Execute base orchestration logic
        # This would call the parent NWTNOrchestrator methods
        response = await super().orchestrate_query(user_input, session)
        
        # Track pattern-specific metrics
        await self._track_pattern_metrics(metrics, response)
        
        return response
    
    async def _apply_orchestration_pattern(
        self, 
        pattern: OrchestrationPattern, 
        metrics: OrchestrationMetrics
    ):
        """Apply orchestration pattern configuration."""
        
        # Configure routing strategy
        if hasattr(self, 'router'):
            self.router.strategy = pattern.routing_strategy
            self.router.load_balancing = pattern.load_balancing_algorithm
        
        # Configure context allocation
        if hasattr(self, 'context_manager'):
            self.context_manager.allocation_method = pattern.context_allocation_method
            self.context_manager.reserve_ratio = pattern.context_reserve_ratio
            self.context_manager.max_context_per_task = pattern.max_context_per_task
        
        # Configure performance settings
        self.parallel_execution_factor = pattern.parallel_execution_factor
        self.quality_threshold = pattern.quality_threshold
        
        # Log pattern application
        logger.debug(f"Applied orchestration pattern: {pattern.routing_strategy}")
    
    async def _track_pattern_metrics(
        self, 
        metrics: OrchestrationMetrics, 
        response: PRSMResponse
    ):
        """Track metrics specific to current orchestration pattern."""
        
        # Extract metrics from response
        if response.context_usage:
            metrics.context_tokens_used = response.context_usage.tokens_used
            metrics.ftns_cost = response.context_usage.cost_ftns
        
        # Count safety violations
        metrics.safety_violations = len([
            step for step in response.reasoning_trace 
            if any(flag.severity == "HIGH" for flag in getattr(step, 'safety_flags', []))
        ])
        
        # Track agent invocations
        metrics.agents_invoked = len([
            step for step in response.reasoning_trace
            if step.step_type in ['ARCHITECT', 'PROMPTER', 'ROUTER', 'EXECUTOR', 'COMPILER']
        ])
    
    def _calculate_response_quality(self, response: PRSMResponse) -> float:
        """Calculate quality score for response."""
        quality_score = 0.0
        
        # Base quality from status
        if response.status == TaskStatus.COMPLETED:
            quality_score = 0.8
        elif response.status == TaskStatus.PARTIAL:
            quality_score = 0.5
        else:
            quality_score = 0.1
        
        # Bonus for comprehensive reasoning trace
        if len(response.reasoning_trace) >= 5:
            quality_score += 0.1
        
        # Penalty for safety violations
        safety_violations = sum(
            1 for step in response.reasoning_trace
            for flag in getattr(step, 'safety_flags', [])
            if flag.severity == "HIGH"
        )
        quality_score -= safety_violations * 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, quality_score))
    
    async def _trigger_orchestration_evolution(self):
        """Trigger evolution of orchestration patterns."""
        try:
            logger.info("Triggering orchestration pattern evolution")
            
            # Evaluate current performance
            current_evaluation = await self.evaluate_performance()
            self.performance_history.append(current_evaluation)
            
            # Attempt self-improvement
            improvement_result = await self.self_improve([current_evaluation])
            
            if improvement_result and improvement_result.success:
                logger.info(f"Orchestration evolution successful: {improvement_result.performance_delta:.3f} improvement")
            else:
                logger.info("No orchestration improvement found or failed")
                
            # Evolve alternative patterns for exploration
            await self._evolve_orchestration_patterns()
            
        except Exception as e:
            logger.error(f"Orchestration evolution failed: {e}")
    
    async def _evolve_orchestration_patterns(self):
        """Evolve new orchestration patterns through DGM exploration."""
        try:
            # Select parent patterns for evolution
            parent_solutions = await self.exploration_engine.select_parents_for_evolution(
                k_parents=4,
                strategy=SelectionStrategy.QUALITY_DIVERSITY
            )
            
            if not parent_solutions:
                logger.warning("No parent solutions available for pattern evolution")
                return
            
            # Generate new patterns
            new_patterns = []
            for parent in parent_solutions:
                try:
                    # Mutate parent pattern
                    parent_pattern = OrchestrationPattern.from_dict(parent.configuration)
                    mutated_pattern = await self._mutate_orchestration_pattern(parent_pattern)
                    
                    # Create new solution
                    new_solution = SolutionNode(
                        parent_ids=[parent.id],
                        component_type=ComponentType.TASK_ORCHESTRATOR,
                        configuration=mutated_pattern.to_dict(),
                        generation=parent.generation + 1
                    )
                    
                    # Validate pattern
                    if await self._validate_orchestration_pattern(mutated_pattern):
                        await self.orchestration_archive.add_solution(new_solution)
                        new_patterns.append(mutated_pattern)
                        
                except Exception as e:
                    logger.error(f"Failed to evolve pattern from {parent.id}: {e}")
            
            logger.info(f"Evolved {len(new_patterns)} new orchestration patterns")
            
        except Exception as e:
            logger.error(f"Pattern evolution failed: {e}")
    
    async def _mutate_orchestration_pattern(
        self, 
        parent_pattern: OrchestrationPattern
    ) -> OrchestrationPattern:
        """Mutate orchestration pattern for evolution."""
        
        # Create mutated copy
        mutated_dict = parent_pattern.to_dict().copy()
        
        # Select mutation type
        import random
        mutation_types = [
            "routing_strategy",
            "load_balancing",
            "context_allocation",
            "performance_tuning",
            "quality_control"
        ]
        
        mutation_type = random.choice(mutation_types)
        
        if mutation_type == "routing_strategy":
            strategies = ["intelligent_adaptive", "cost_optimized", "latency_optimized", "quality_focused"]
            current = mutated_dict["routing_strategy"]
            alternatives = [s for s in strategies if s != current]
            if alternatives:
                mutated_dict["routing_strategy"] = random.choice(alternatives)
        
        elif mutation_type == "load_balancing":
            algorithms = ["least_response_time", "round_robin", "weighted_capacity", "least_connections"]
            current = mutated_dict["load_balancing_algorithm"]
            alternatives = [a for a in algorithms if a != current]
            if alternatives:
                mutated_dict["load_balancing_algorithm"] = random.choice(alternatives)
        
        elif mutation_type == "context_allocation":
            # Adjust context parameters
            mutated_dict["context_reserve_ratio"] = max(0.1, min(0.5, 
                mutated_dict["context_reserve_ratio"] * random.uniform(0.8, 1.2)))
            mutated_dict["max_context_per_task"] = max(1024, min(8192,
                int(mutated_dict["max_context_per_task"] * random.uniform(0.8, 1.2))))
        
        elif mutation_type == "performance_tuning":
            # Adjust performance parameters
            mutated_dict["parallel_execution_factor"] = max(0.1, min(1.0,
                mutated_dict["parallel_execution_factor"] * random.uniform(0.9, 1.1)))
            mutated_dict["prefetch_enabled"] = random.choice([True, False])
            mutated_dict["result_compression"] = random.choice([True, False])
        
        elif mutation_type == "quality_control":
            # Adjust quality parameters
            mutated_dict["quality_threshold"] = max(0.5, min(0.95,
                mutated_dict["quality_threshold"] * random.uniform(0.95, 1.05)))
            retry_strategies = ["exponential_backoff", "linear_backoff", "immediate_retry", "no_retry"]
            mutated_dict["error_retry_strategy"] = random.choice(retry_strategies)
        
        return OrchestrationPattern.from_dict(mutated_dict)
    
    async def _validate_orchestration_pattern(self, pattern: OrchestrationPattern) -> bool:
        """Validate orchestration pattern for safety and correctness."""
        
        # Validate routing strategy
        valid_strategies = ["intelligent_adaptive", "cost_optimized", "latency_optimized", "quality_focused"]
        if pattern.routing_strategy not in valid_strategies:
            return False
        
        # Validate load balancing algorithm
        valid_algorithms = ["least_response_time", "round_robin", "weighted_capacity", "least_connections"]
        if pattern.load_balancing_algorithm not in valid_algorithms:
            return False
        
        # Validate numeric parameters
        if not (0.0 <= pattern.parallel_execution_factor <= 1.0):
            return False
        if not (0.1 <= pattern.context_reserve_ratio <= 0.5):
            return False
        if not (1024 <= pattern.max_context_per_task <= 8192):
            return False
        if not (0.5 <= pattern.quality_threshold <= 0.95):
            return False
        if not (1 <= pattern.hierarchy_depth_limit <= 10):
            return False
        
        return True
    
    # Implement SelfModifyingComponent interface
    
    async def propose_modification(self, evaluation_logs: List[EvaluationResult]) -> Optional[ModificationProposal]:
        """Propose orchestration pattern modification based on performance."""
        
        if not evaluation_logs:
            return None
        
        latest_eval = evaluation_logs[-1]
        
        # Check if improvement is needed
        if latest_eval.performance_score >= 0.9:
            logger.info("Orchestration performance is excellent, no modification needed")
            return None
        
        # Analyze performance bottlenecks
        analysis = await self._analyze_performance_bottlenecks(evaluation_logs)
        
        if not analysis["needs_improvement"]:
            return None
        
        # Generate pattern improvements
        improved_pattern = await self._generate_improved_pattern(analysis)
        
        if not improved_pattern:
            return None
        
        # Create modification proposal
        proposal = ModificationProposal(
            solution_id=self.current_solution_id or "unknown",
            component_type=ComponentType.TASK_ORCHESTRATOR,
            modification_type="pattern_update",
            description=f"Improve orchestration {analysis['primary_bottleneck']}",
            rationale=analysis["improvement_rationale"],
            config_changes=improved_pattern.to_dict(),
            estimated_performance_impact=analysis["estimated_improvement"],
            risk_level=RiskLevel.LOW,
            impact_level=ImpactLevel.MEDIUM,
            safety_considerations=["performance_monitoring", "pattern_validation"],
            rollback_plan="Revert to previous orchestration pattern",
            proposer_id=self.component_id
        )
        
        return proposal
    
    async def apply_modification(self, modification: ModificationProposal) -> ModificationResult:
        """Apply orchestration pattern modification."""
        
        start_time = datetime.utcnow()
        
        try:
            # Store previous pattern for rollback
            previous_pattern = self.current_pattern
            
            # Apply new pattern
            new_pattern = OrchestrationPattern.from_dict(modification.config_changes)
            self.current_pattern = new_pattern
            
            # Create new solution in archive
            new_solution = SolutionNode(
                parent_ids=[self.current_solution_id] if self.current_solution_id else [],
                component_type=ComponentType.TASK_ORCHESTRATOR,
                configuration=new_pattern.to_dict(),
                generation=await self._get_current_generation() + 1
            )
            
            self.current_solution_id = await self.orchestration_archive.add_solution(new_solution)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Applied orchestration pattern modification: {modification.description}")
            
            return ModificationResult(
                modification_id=modification.id,
                success=True,
                execution_time_seconds=execution_time,
                functionality_preserved=True,
                safety_status=SafetyStatus.SAFE,
                executor_id=self.component_id,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to apply orchestration modification: {e}")
            
            return ModificationResult(
                modification_id=modification.id,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                execution_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                executor_id=self.component_id,
                timestamp=datetime.utcnow()
            )
    
    async def validate_modification(self, modification: ModificationProposal) -> bool:
        """Validate orchestration pattern modification."""
        
        try:
            # Validate configuration structure
            if not isinstance(modification.config_changes, dict):
                return False
            
            # Create pattern from changes and validate
            pattern = OrchestrationPattern.from_dict(modification.config_changes)
            return await self._validate_orchestration_pattern(pattern)
            
        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            return False
    
    async def create_checkpoint(self) -> Checkpoint:
        """Create checkpoint of current orchestration state."""
        
        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            component_id=self.component_id,
            component_type=ComponentType.TASK_ORCHESTRATOR,
            state_snapshot={
                "current_solution_id": self.current_solution_id,
                "session_count": self.session_count,
                "metrics_count": len(self.session_metrics)
            },
            configuration_snapshot=self.current_pattern.to_dict(),
            timestamp=datetime.utcnow(),
            storage_location=f"checkpoint_{self.component_id}_{datetime.utcnow().timestamp()}"
        )
        
        logger.info(f"Created orchestration checkpoint: {checkpoint.id}")
        return checkpoint
    
    async def rollback_to_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Rollback orchestration to checkpoint state."""
        
        try:
            # Restore pattern configuration
            self.current_pattern = OrchestrationPattern.from_dict(
                checkpoint.configuration_snapshot
            )
            
            # Restore state
            state = checkpoint.state_snapshot
            self.current_solution_id = state.get("current_solution_id")
            self.session_count = state.get("session_count", 0)
            
            logger.info(f"Rolled back to checkpoint: {checkpoint.id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def evaluate_performance(self) -> EvaluationResult:
        """Evaluate current orchestration performance."""
        
        # Get recent session metrics
        recent_sessions = list(self.session_metrics.values())[-self.evaluation_window:]
        
        if not recent_sessions:
            # Return baseline evaluation if no sessions
            return EvaluationResult(
                solution_id=self.current_solution_id or "unknown",
                component_type=ComponentType.TASK_ORCHESTRATOR,
                performance_score=0.5,
                task_success_rate=0.5,
                tasks_evaluated=0,
                tasks_successful=0,
                evaluation_duration_seconds=0.0,
                evaluation_tier="baseline",
                evaluator_version="1.0",
                benchmark_suite="orchestration_patterns"
            )
        
        # Calculate performance metrics
        successful_sessions = sum(1 for s in recent_sessions if s.success)
        success_rate = successful_sessions / len(recent_sessions)
        
        # Calculate average latency
        latencies = [s.total_latency_ms for s in recent_sessions if s.total_latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # Calculate throughput
        total_duration = sum(s.duration_seconds for s in recent_sessions)
        total_tasks = sum(s.agents_invoked for s in recent_sessions)
        throughput = total_tasks / total_duration if total_duration > 0 else 0
        
        # Calculate composite performance score
        latency_score = max(0, 1 - (avg_latency / 5000))  # Normalize to 5s max
        throughput_score = min(1, throughput / 10)  # Normalize to 10 tasks/s max
        quality_score = sum(s.quality_score for s in recent_sessions) / len(recent_sessions)
        
        performance_score = (
            success_rate * 0.4 +
            quality_score * 0.3 +
            latency_score * 0.2 +
            throughput_score * 0.1
        )
        
        return EvaluationResult(
            solution_id=self.current_solution_id or "unknown",
            component_type=ComponentType.TASK_ORCHESTRATOR,
            performance_score=performance_score,
            task_success_rate=success_rate,
            latency_ms=avg_latency,
            throughput_rps=throughput,
            tasks_evaluated=len(recent_sessions),
            tasks_successful=successful_sessions,
            evaluation_duration_seconds=total_duration,
            evaluation_tier="comprehensive",
            evaluator_version="1.0",
            benchmark_suite="orchestration_patterns"
        )
    
    async def _analyze_performance_bottlenecks(self, evaluations: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze performance bottlenecks in orchestration."""
        
        latest_eval = evaluations[-1]
        
        analysis = {
            "needs_improvement": False,
            "primary_bottleneck": "none",
            "improvement_rationale": "",
            "estimated_improvement": 0.0
        }
        
        # Identify primary bottleneck
        if latest_eval.task_success_rate < 0.8:
            analysis["needs_improvement"] = True
            analysis["primary_bottleneck"] = "reliability"
            analysis["improvement_rationale"] = "Low success rate requires reliability improvements"
            analysis["estimated_improvement"] = 0.15
        elif latest_eval.latency_ms and latest_eval.latency_ms > 2000:
            analysis["needs_improvement"] = True
            analysis["primary_bottleneck"] = "latency"
            analysis["improvement_rationale"] = "High latency requires performance optimization"
            analysis["estimated_improvement"] = 0.1
        elif latest_eval.performance_score < 0.7:
            analysis["needs_improvement"] = True
            analysis["primary_bottleneck"] = "overall_performance"
            analysis["improvement_rationale"] = "Overall performance below target"
            analysis["estimated_improvement"] = 0.05
        
        return analysis
    
    async def _generate_improved_pattern(self, analysis: Dict[str, Any]) -> Optional[OrchestrationPattern]:
        """Generate improved orchestration pattern based on analysis."""
        
        bottleneck = analysis["primary_bottleneck"]
        improved_pattern = OrchestrationPattern.from_dict(self.current_pattern.to_dict())
        
        if bottleneck == "reliability":
            improved_pattern.error_retry_strategy = "exponential_backoff"
            improved_pattern.quality_threshold = min(0.95, improved_pattern.quality_threshold + 0.05)
            improved_pattern.safety_validation_level = "enhanced"
        
        elif bottleneck == "latency":
            improved_pattern.routing_strategy = "latency_optimized"
            improved_pattern.load_balancing_algorithm = "least_response_time"
            improved_pattern.prefetch_enabled = True
            improved_pattern.parallel_execution_factor = min(1.0, improved_pattern.parallel_execution_factor + 0.1)
        
        elif bottleneck == "overall_performance":
            improved_pattern.routing_strategy = "intelligent_adaptive"
            improved_pattern.caching_strategy = "semantic_aware"
            improved_pattern.result_compression = True
        
        else:
            return None
        
        # Validate improved pattern
        if await self._validate_orchestration_pattern(improved_pattern):
            return improved_pattern
        
        return None
    
    async def _get_current_generation(self) -> int:
        """Get current generation number."""
        if (self.current_solution_id and 
            self.current_solution_id in self.orchestration_archive.solutions):
            return self.orchestration_archive.solutions[self.current_solution_id].generation
        return 0
    
    async def _create_baseline_evaluation(self) -> EvaluationResult:
        """Create baseline evaluation for initial pattern."""
        return EvaluationResult(
            solution_id="baseline",
            component_type=ComponentType.TASK_ORCHESTRATOR,
            performance_score=0.6,  # Reasonable baseline
            task_success_rate=0.8,
            latency_ms=1000.0,
            throughput_rps=5.0,
            tasks_evaluated=10,
            tasks_successful=8,
            evaluation_duration_seconds=10.0,
            evaluation_tier="baseline",
            evaluator_version="1.0",
            benchmark_suite="orchestration_patterns"
        )
    
    # Additional methods for integration with existing NWTN architecture
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics."""
        
        if not self.enable_evolution:
            return {"evolution_disabled": True}
        
        recent_sessions = list(self.session_metrics.values())[-50:]  # Last 50 sessions
        
        stats = {
            "total_sessions": self.session_count,
            "archive_solutions": len(self.orchestration_archive.solutions) if self.orchestration_archive else 0,
            "current_pattern": self.current_pattern.to_dict(),
            "performance_trend": "improving" if len(self.performance_history) >= 2 and 
                               self.performance_history[-1].performance_score > self.performance_history[-2].performance_score 
                               else "stable",
        }
        
        if recent_sessions:
            stats.update({
                "recent_success_rate": sum(1 for s in recent_sessions if s.success) / len(recent_sessions),
                "average_latency_ms": sum(s.total_latency_ms for s in recent_sessions) / len(recent_sessions),
                "average_quality_score": sum(s.quality_score for s in recent_sessions) / len(recent_sessions)
            })
        
        return stats
    
    async def force_pattern_evolution(self) -> Dict[str, Any]:
        """Force evolution of orchestration patterns (for testing/admin use)."""
        
        if not self.enable_evolution:
            return {"error": "Evolution is disabled"}
        
        try:
            await self._trigger_orchestration_evolution()
            
            stats = await self.orchestration_archive.archive_statistics()
            
            return {
                "evolution_triggered": True,
                "archive_solutions": stats.total_solutions,
                "current_performance": stats.best_performance,
                "diversity_score": stats.diversity_score
            }
            
        except Exception as e:
            logger.error(f"Forced evolution failed: {e}")
            return {"error": str(e)}
    
    async def get_evolution_insights(self) -> Dict[str, Any]:
        """Get insights into the evolution process."""
        
        if not self.enable_evolution:
            return {"evolution_disabled": True}
        
        try:
            archive_stats = await self.orchestration_archive.archive_statistics()
            exploration_metrics = await self.exploration_engine.calculate_exploration_metrics()
            stepping_stones = await self.orchestration_archive.identify_stepping_stones()
            
            return {
                "archive_statistics": {
                    "total_solutions": archive_stats.total_solutions,
                    "active_solutions": archive_stats.active_solutions,
                    "generations": archive_stats.generations,
                    "best_performance": archive_stats.best_performance,
                    "diversity_score": archive_stats.diversity_score,
                    "stepping_stones": archive_stats.stepping_stones_discovered
                },
                "exploration_metrics": {
                    "exploration_diversity": exploration_metrics.exploration_diversity,
                    "breakthrough_rate": exploration_metrics.breakthrough_rate,
                    "novelty_score": exploration_metrics.novelty_score,
                    "quality_diversity_ratio": exploration_metrics.quality_diversity_ratio
                },
                "stepping_stones_found": len(stepping_stones),
                "performance_history_length": len(self.performance_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get evolution insights: {e}")
            return {"error": str(e)}
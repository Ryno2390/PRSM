"""
DGM Evolution System Examples

Example implementations showing how PRSM components can be enhanced
with Darwin Gödel Machine evolution capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import uuid

from .archive import EvolutionArchive, SolutionNode
from .self_modification import SelfModifyingComponent
from .exploration import OpenEndedExplorationEngine
from .models import (
    ComponentType, ModificationProposal, EvaluationResult, 
    ModificationResult, Checkpoint, SafetyStatus, RiskLevel, ImpactLevel
)

# Import PRSM components (these would be actual imports in real implementation)
try:
    from ..nwtn.orchestrator import TaskOrchestrator
    from ..agents.executors.unified_router import UnifiedModelRouter
    from prsm.economy.governance.voting import GovernanceSystem
except ImportError:
    # Mock classes for standalone testing
    class TaskOrchestrator:
        def __init__(self):
            self.routing_strategy = "round_robin"
            self.load_balancing = {"enabled": True, "algorithm": "least_connections"}
            self.performance_metrics = {"latency": 100, "throughput": 50}
    
    class IntelligentRouter:
        def __init__(self):
            self.routing_rules = {"default": "fastest"}
            self.cost_optimization = {"enabled": True, "budget_limit": 1000}
    
    class GovernanceSystem:
        def __init__(self):
            pass


logger = logging.getLogger(__name__)


class DGMEnhancedTaskOrchestrator(TaskOrchestrator, SelfModifyingComponent):
    """
    DGM-enhanced TaskOrchestrator that can evolve its orchestration strategies.
    
    Demonstrates integration of evolution capabilities with existing PRSM components.
    """
    
    def __init__(self, orchestrator_id: str = None):
        # Initialize base components
        TaskOrchestrator.__init__(self)
        orchestrator_id = orchestrator_id or f"orchestrator_{uuid.uuid4().hex[:8]}"
        SelfModifyingComponent.__init__(self, orchestrator_id, ComponentType.TASK_ORCHESTRATOR)
        
        # Evolution system components
        self.orchestration_archive = EvolutionArchive(
            archive_id=f"{orchestrator_id}_archive",
            component_type=ComponentType.TASK_ORCHESTRATOR
        )
        self.exploration_engine = OpenEndedExplorationEngine(self.orchestration_archive)
        
        # Current orchestration configuration
        self.current_config = {
            "routing_strategy": self.routing_strategy,
            "load_balancing": self.load_balancing,
            "performance_targets": {
                "max_latency_ms": 500,
                "min_throughput_rps": 100,
                "success_rate": 0.95
            },
            "optimization_objectives": ["latency", "cost", "quality"]
        }
        
        # Performance tracking
        self.evaluation_window = 100  # Evaluate over last 100 tasks
        self.task_history: List[Dict[str, Any]] = []
        
        # Initialize archive with current configuration
        asyncio.create_task(self._initialize_archive())
    
    async def _initialize_archive(self):
        """Initialize archive with current orchestration configuration."""
        initial_solution = SolutionNode(
            component_type=ComponentType.TASK_ORCHESTRATOR,
            configuration=self.current_config.copy(),
            generation=0
        )
        
        # Evaluate initial performance
        initial_performance = await self.evaluate_performance()
        initial_solution.add_evaluation(initial_performance)
        
        await self.orchestration_archive.add_solution(initial_solution)
        self.current_solution_id = initial_solution.id
        
        logger.info(f"Initialized DGM-enhanced TaskOrchestrator with solution {initial_solution.id}")
    
    async def propose_modification(self, evaluation_logs: List[EvaluationResult]) -> Optional[ModificationProposal]:
        """
        Analyze orchestration performance and propose improvements.
        
        Args:
            evaluation_logs: Recent performance evaluations
            
        Returns:
            ModificationProposal for orchestration improvement
        """
        if not evaluation_logs:
            return None
        
        # Analyze performance gaps
        performance_analysis = await self._analyze_orchestration_performance(evaluation_logs)
        
        if not performance_analysis["needs_improvement"]:
            logger.info("Orchestration performance is satisfactory, no modification needed")
            return None
        
        # Identify improvement strategy
        improvement_strategy = await self._identify_improvement_strategy(performance_analysis)
        
        # Generate configuration changes
        config_changes = await self._generate_config_changes(improvement_strategy)
        
        if not config_changes:
            return None
        
        # Create modification proposal
        proposal = ModificationProposal(
            solution_id=self.current_solution_id,
            component_type=ComponentType.TASK_ORCHESTRATOR,
            modification_type="config_update",
            description=f"Improve orchestration {improvement_strategy['focus_area']}",
            rationale=improvement_strategy["rationale"],
            config_changes=config_changes,
            estimated_performance_impact=improvement_strategy["estimated_impact"],
            risk_level=self._assess_risk_level(config_changes),
            impact_level=self._assess_impact_level(config_changes),
            safety_considerations=["performance_monitoring", "rollback_capability"],
            rollback_plan="Revert to previous configuration if performance degrades",
            proposer_id=self.component_id
        )
        
        logger.info(f"Proposed orchestration modification: {proposal.description}")
        return proposal
    
    async def apply_modification(self, modification: ModificationProposal) -> ModificationResult:
        """
        Apply orchestration configuration changes.
        
        Args:
            modification: The modification to apply
            
        Returns:
            ModificationResult with application status
        """
        start_time = datetime.utcnow()
        
        try:
            # Store current configuration for rollback
            previous_config = self.current_config.copy()
            
            # Apply configuration changes
            for key, value in modification.config_changes.items():
                if key in self.current_config:
                    self.current_config[key] = value
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            # Update orchestrator with new configuration
            await self._apply_orchestration_config(self.current_config)
            
            # Create new solution node
            new_solution = SolutionNode(
                parent_ids=[self.current_solution_id],
                component_type=ComponentType.TASK_ORCHESTRATOR,
                configuration=self.current_config.copy(),
                generation=await self._get_current_generation() + 1
            )
            
            # Add to archive
            new_solution_id = await self.orchestration_archive.add_solution(new_solution)
            self.current_solution_id = new_solution_id
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ModificationResult(
                modification_id=modification.id,
                success=True,
                execution_time_seconds=execution_time,
                resources_used={"config_keys_updated": len(modification.config_changes)},
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
        """
        Validate orchestration modification preserves core functionality.
        
        Args:
            modification: The modification to validate
            
        Returns:
            True if modification is valid
        """
        # Check configuration key validity
        valid_keys = set(self.current_config.keys())
        for key in modification.config_changes.keys():
            if key not in valid_keys:
                logger.error(f"Invalid configuration key: {key}")
                return False
        
        # Check value types and ranges
        for key, value in modification.config_changes.items():
            if not await self._validate_config_value(key, value):
                logger.error(f"Invalid configuration value for {key}: {value}")
                return False
        
        # Check for dangerous modifications
        if modification.risk_level == RiskLevel.CRITICAL:
            logger.error("Critical risk modifications require additional approval")
            return False
        
        return True
    
    async def create_checkpoint(self) -> Checkpoint:
        """
        Create checkpoint of current orchestration state.
        
        Returns:
            Checkpoint containing orchestration state
        """
        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            component_id=self.component_id,
            component_type=ComponentType.TASK_ORCHESTRATOR,
            state_snapshot={
                "current_solution_id": self.current_solution_id,
                "task_history_size": len(self.task_history),
                "performance_metrics": self.performance_metrics.copy()
            },
            configuration_snapshot=self.current_config.copy(),
            timestamp=datetime.utcnow(),
            storage_location=f"checkpoint_{self.component_id}_{datetime.utcnow().timestamp()}"
        )
        
        logger.info(f"Created orchestration checkpoint: {checkpoint.id}")
        return checkpoint
    
    async def rollback_to_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """
        Rollback orchestration to checkpoint state.
        
        Args:
            checkpoint: The checkpoint to rollback to
            
        Returns:
            True if rollback successful
        """
        try:
            # Restore configuration
            self.current_config = checkpoint.configuration_snapshot.copy()
            
            # Apply restored configuration
            await self._apply_orchestration_config(self.current_config)
            
            # Restore state
            self.current_solution_id = checkpoint.state_snapshot["current_solution_id"]
            
            logger.info(f"Successfully rolled back to checkpoint: {checkpoint.id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def evaluate_performance(self) -> EvaluationResult:
        """
        Evaluate current orchestration performance.
        
        Returns:
            EvaluationResult with performance metrics
        """
        # Calculate performance metrics from recent task history
        recent_tasks = self.task_history[-self.evaluation_window:] if self.task_history else []
        
        if not recent_tasks:
            # Return baseline performance if no task history
            return EvaluationResult(
                solution_id=getattr(self, 'current_solution_id', 'unknown'),
                component_type=ComponentType.TASK_ORCHESTRATOR,
                performance_score=0.5,
                task_success_rate=0.5,
                tasks_evaluated=0,
                tasks_successful=0,
                evaluation_duration_seconds=0.0,
                evaluation_tier="quick",
                evaluator_version="1.0",
                benchmark_suite="orchestration_benchmark"
            )
        
        # Calculate metrics
        successful_tasks = sum(1 for task in recent_tasks if task.get("success", False))
        success_rate = successful_tasks / len(recent_tasks)
        
        # Calculate latency metrics
        latencies = [task.get("latency_ms", 0) for task in recent_tasks if "latency_ms" in task]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # Calculate throughput
        time_span = (recent_tasks[-1]["timestamp"] - recent_tasks[0]["timestamp"]).total_seconds()
        throughput = len(recent_tasks) / time_span if time_span > 0 else 0
        
        # Composite performance score
        latency_score = max(0, 1 - (avg_latency / 1000))  # Normalize to 0-1
        throughput_score = min(1, throughput / 100)  # Normalize to 0-1
        performance_score = (success_rate * 0.5) + (latency_score * 0.3) + (throughput_score * 0.2)
        
        return EvaluationResult(
            solution_id=self.current_solution_id,
            component_type=ComponentType.TASK_ORCHESTRATOR,
            performance_score=performance_score,
            task_success_rate=success_rate,
            latency_ms=avg_latency,
            throughput_rps=throughput,
            tasks_evaluated=len(recent_tasks),
            tasks_successful=successful_tasks,
            evaluation_duration_seconds=time_span,
            evaluation_tier="comprehensive",
            evaluator_version="1.0",
            benchmark_suite="orchestration_benchmark"
        )
    
    async def evolve_orchestration_strategies(self) -> List[SolutionNode]:
        """
        Evolve orchestration strategies using DGM exploration.
        
        Returns:
            List of new orchestration solution candidates
        """
        logger.info("Starting orchestration strategy evolution")
        
        # Select parent solutions for evolution
        parent_solutions = await self.exploration_engine.select_parents_for_evolution(
            k_parents=4,
            focus_areas=["routing_strategy", "load_balancing", "optimization_objectives"]
        )
        
        if not parent_solutions:
            logger.warning("No parent solutions available for evolution")
            return []
        
        # Generate new orchestration strategies
        new_solutions = []
        for parent in parent_solutions:
            try:
                # Generate modification based on parent
                modified_config = await self._mutate_orchestration_config(parent.configuration)
                
                # Create new solution
                new_solution = SolutionNode(
                    parent_ids=[parent.id],
                    component_type=ComponentType.TASK_ORCHESTRATOR,
                    configuration=modified_config,
                    generation=parent.generation + 1
                )
                
                # Validate new configuration
                if await self._validate_orchestration_config(modified_config):
                    new_solutions.append(new_solution)
                    await self.orchestration_archive.add_solution(new_solution)
                    
            except Exception as e:
                logger.error(f"Failed to evolve from parent {parent.id}: {e}")
        
        logger.info(f"Generated {len(new_solutions)} new orchestration strategies")
        return new_solutions
    
    async def _analyze_orchestration_performance(self, evaluation_logs: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze orchestration performance to identify improvement areas."""
        if not evaluation_logs:
            return {"needs_improvement": False}
        
        latest_eval = evaluation_logs[-1]
        
        # Performance thresholds
        min_performance = 0.7
        min_success_rate = 0.9
        max_latency = 500  # ms
        
        analysis = {
            "needs_improvement": False,
            "issues": [],
            "current_performance": latest_eval.performance_score,
            "current_success_rate": latest_eval.task_success_rate,
            "current_latency": latest_eval.latency_ms or 0
        }
        
        # Check performance issues
        if latest_eval.performance_score < min_performance:
            analysis["needs_improvement"] = True
            analysis["issues"].append("low_overall_performance")
        
        if latest_eval.task_success_rate < min_success_rate:
            analysis["needs_improvement"] = True
            analysis["issues"].append("low_success_rate")
        
        if latest_eval.latency_ms and latest_eval.latency_ms > max_latency:
            analysis["needs_improvement"] = True
            analysis["issues"].append("high_latency")
        
        return analysis
    
    async def _identify_improvement_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify specific improvement strategy based on performance analysis."""
        issues = analysis.get("issues", [])
        
        if "high_latency" in issues:
            return {
                "focus_area": "latency_optimization",
                "rationale": "High latency detected, optimizing for speed",
                "estimated_impact": 0.2,
                "target_configs": ["routing_strategy", "load_balancing"]
            }
        
        if "low_success_rate" in issues:
            return {
                "focus_area": "reliability_improvement",
                "rationale": "Low success rate detected, improving reliability",
                "estimated_impact": 0.15,
                "target_configs": ["routing_strategy", "performance_targets"]
            }
        
        if "low_overall_performance" in issues:
            return {
                "focus_area": "general_optimization",
                "rationale": "Overall performance below target, general optimization",
                "estimated_impact": 0.1,
                "target_configs": ["optimization_objectives", "performance_targets"]
            }
        
        return {
            "focus_area": "exploration",
            "rationale": "No specific issues detected, exploring new strategies",
            "estimated_impact": 0.05,
            "target_configs": ["routing_strategy"]
        }
    
    async def _generate_config_changes(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration changes based on improvement strategy."""
        changes = {}
        focus_area = strategy["focus_area"]
        
        if focus_area == "latency_optimization":
            changes["routing_strategy"] = "fastest_response"
            changes["load_balancing"] = {
                "enabled": True,
                "algorithm": "least_response_time"
            }
        
        elif focus_area == "reliability_improvement":
            changes["routing_strategy"] = "most_reliable"
            changes["performance_targets"] = {
                **self.current_config["performance_targets"],
                "success_rate": 0.98
            }
        
        elif focus_area == "general_optimization":
            changes["optimization_objectives"] = ["quality", "latency", "cost"]
            changes["performance_targets"] = {
                **self.current_config["performance_targets"],
                "max_latency_ms": 400,
                "min_throughput_rps": 120
            }
        
        elif focus_area == "exploration":
            # Try a new routing strategy
            strategies = ["round_robin", "weighted_round_robin", "least_connections", "fastest_response"]
            current_strategy = self.current_config.get("routing_strategy", "round_robin")
            new_strategies = [s for s in strategies if s != current_strategy]
            if new_strategies:
                changes["routing_strategy"] = new_strategies[0]  # Pick first alternative
        
        return changes
    
    async def _mutate_orchestration_config(self, parent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate orchestration configuration for evolution."""
        mutated_config = parent_config.copy()
        
        # Mutation strategies
        mutations = [
            self._mutate_routing_strategy,
            self._mutate_load_balancing,
            self._mutate_performance_targets,
            self._mutate_optimization_objectives
        ]
        
        # Apply random mutation
        import random
        mutation_function = random.choice(mutations)
        mutated_config = await mutation_function(mutated_config)
        
        return mutated_config
    
    async def _mutate_routing_strategy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate routing strategy."""
        strategies = ["round_robin", "weighted_round_robin", "least_connections", "fastest_response", "most_reliable"]
        current = config.get("routing_strategy", "round_robin")
        alternatives = [s for s in strategies if s != current]
        
        if alternatives:
            config["routing_strategy"] = alternatives[0]
        
        return config
    
    async def _mutate_load_balancing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate load balancing configuration."""
        algorithms = ["round_robin", "least_connections", "least_response_time", "weighted_round_robin"]
        current_lb = config.get("load_balancing", {})
        
        new_algorithm = algorithms[0]  # Simple mutation for demo
        config["load_balancing"] = {
            "enabled": True,
            "algorithm": new_algorithm
        }
        
        return config
    
    async def _mutate_performance_targets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate performance targets."""
        current_targets = config.get("performance_targets", {})
        
        # Slightly adjust targets
        if "max_latency_ms" in current_targets:
            current_latency = current_targets["max_latency_ms"]
            # Randomly adjust by ±10%
            import random
            adjustment = random.uniform(0.9, 1.1)
            new_latency = int(current_latency * adjustment)
            current_targets["max_latency_ms"] = max(100, min(1000, new_latency))
        
        config["performance_targets"] = current_targets
        return config
    
    async def _mutate_optimization_objectives(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate optimization objectives."""
        objectives = ["latency", "cost", "quality", "throughput", "reliability"]
        current_objectives = config.get("optimization_objectives", ["latency", "cost"])
        
        # Add or remove an objective
        import random
        if len(current_objectives) < 3 and random.random() < 0.5:
            # Add objective
            available = [obj for obj in objectives if obj not in current_objectives]
            if available:
                current_objectives.append(available[0])
        elif len(current_objectives) > 1 and random.random() < 0.5:
            # Remove objective
            current_objectives.pop()
        
        config["optimization_objectives"] = current_objectives
        return config
    
    async def _apply_orchestration_config(self, config: Dict[str, Any]):
        """Apply orchestration configuration to the orchestrator."""
        # Update orchestrator properties based on configuration
        if "routing_strategy" in config:
            self.routing_strategy = config["routing_strategy"]
        
        if "load_balancing" in config:
            self.load_balancing = config["load_balancing"]
        
        # Additional configuration application would go here
        logger.info(f"Applied orchestration configuration: {config}")
    
    async def _validate_config_value(self, key: str, value: Any) -> bool:
        """Validate configuration value."""
        if key == "routing_strategy":
            valid_strategies = ["round_robin", "weighted_round_robin", "least_connections", 
                              "fastest_response", "most_reliable"]
            return value in valid_strategies
        
        if key == "load_balancing":
            if not isinstance(value, dict):
                return False
            return "enabled" in value and "algorithm" in value
        
        if key == "performance_targets":
            if not isinstance(value, dict):
                return False
            # Validate target values are reasonable
            if "max_latency_ms" in value and not (50 <= value["max_latency_ms"] <= 5000):
                return False
            return True
        
        return True
    
    async def _validate_orchestration_config(self, config: Dict[str, Any]) -> bool:
        """Validate complete orchestration configuration."""
        required_keys = ["routing_strategy", "load_balancing", "performance_targets"]
        
        for key in required_keys:
            if key not in config:
                return False
            if not await self._validate_config_value(key, config[key]):
                return False
        
        return True
    
    def _assess_risk_level(self, config_changes: Dict[str, Any]) -> RiskLevel:
        """Assess risk level of configuration changes."""
        if "routing_strategy" in config_changes:
            return RiskLevel.MEDIUM
        
        if len(config_changes) > 2:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _assess_impact_level(self, config_changes: Dict[str, Any]) -> ImpactLevel:
        """Assess impact level of configuration changes."""
        if "routing_strategy" in config_changes:
            return ImpactLevel.MEDIUM
        
        if "performance_targets" in config_changes:
            return ImpactLevel.MEDIUM
        
        return ImpactLevel.LOW
    
    async def _get_current_generation(self) -> int:
        """Get current generation number."""
        if hasattr(self, 'current_solution_id') and self.current_solution_id in self.orchestration_archive.solutions:
            return self.orchestration_archive.solutions[self.current_solution_id].generation
        return 0
    
    def record_task_execution(self, task_id: str, success: bool, latency_ms: float, **kwargs):
        """Record task execution for performance evaluation."""
        task_record = {
            "task_id": task_id,
            "success": success,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow(),
            **kwargs
        }
        
        self.task_history.append(task_record)
        
        # Keep history size manageable
        if len(self.task_history) > self.evaluation_window * 2:
            self.task_history = self.task_history[-self.evaluation_window:]


# Example usage and testing
async def demonstrate_dgm_orchestrator():
    """Demonstrate DGM-enhanced TaskOrchestrator capabilities."""
    print("🚀 Demonstrating DGM-Enhanced TaskOrchestrator")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = DGMEnhancedTaskOrchestrator("demo_orchestrator")
    
    # Wait for initialization
    await asyncio.sleep(1)
    
    # Simulate task executions
    print("\n📊 Simulating task executions...")
    for i in range(50):
        # Simulate varying performance
        import random
        success = random.random() > 0.1  # 90% success rate
        latency = random.uniform(100, 800)  # 100-800ms latency
        
        orchestrator.record_task_execution(
            task_id=f"task_{i}",
            success=success,
            latency_ms=latency
        )
    
    # Evaluate current performance
    print("\n🔍 Evaluating orchestration performance...")
    performance = await orchestrator.evaluate_performance()
    print(f"Performance Score: {performance.performance_score:.3f}")
    print(f"Success Rate: {performance.task_success_rate:.3f}")
    print(f"Average Latency: {performance.latency_ms:.1f}ms")
    
    # Trigger self-improvement
    print("\n🧬 Triggering self-improvement...")
    improvement_result = await orchestrator.self_improve([performance])
    
    if improvement_result and improvement_result.success:
        print(f"✅ Self-improvement successful!")
        print(f"Performance delta: {improvement_result.performance_delta:.3f}")
    else:
        print("ℹ️ No improvement needed or failed")
    
    # Demonstrate evolution
    print("\n🔄 Evolving orchestration strategies...")
    new_strategies = await orchestrator.evolve_orchestration_strategies()
    print(f"Generated {len(new_strategies)} new strategies")
    
    # Show archive statistics
    print("\n📈 Archive Statistics:")
    stats = await orchestrator.orchestration_archive.archive_statistics()
    print(f"Total Solutions: {stats.total_solutions}")
    print(f"Active Solutions: {stats.active_solutions}")
    print(f"Generations: {stats.generations}")
    print(f"Best Performance: {stats.best_performance:.3f}")
    print(f"Diversity Score: {stats.diversity_score:.3f}")
    
    print("\n✨ DGM demonstration completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_dgm_orchestrator())
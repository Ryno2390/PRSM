"""
Bayesian Search Engine for NWTN Hybrid Architecture
Automated experiment generation, execution, and knowledge sharing

This module implements the Bayesian search mechanism described in the brainstorming document:
- Automated experiment generation for testing SOCs and hypotheses
- Bayesian updating of confidence based on experimental results
- Knowledge sharing across PRSM network (including failures)
- Hive mind updates where validated knowledge propagates to all agents

Key Features:
1. Experiment Design: Automated generation of testable hypotheses
2. Experiment Execution: Simulation, logical, and causal testing
3. Bayesian Updates: Confidence adjustment based on results
4. Knowledge Sharing: Broadcast results to PRSM network
5. Failure Mining: Extract value from negative results
6. Hive Mind: Propagate validated knowledge to all agents

Integration Points:
- PRSM Marketplace: Share experiment results and validated knowledge
- IPFS: Persistent storage of experimental data
- Federation: Consensus on core knowledge updates
- Tokenomics: FTNS rewards for valuable experiments (including failures)
"""

import asyncio
import json
import math
import random
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import PRSMBaseModel, TimestampMixin
from prsm.core.config import get_settings
from prsm.nwtn.hybrid_architecture import SOC, SOCType, ConfidenceLevel, ExperimentResult, ExperimentType
from prsm.nwtn.world_model_engine import WorldModelEngine, ValidationResult, CausalRelationType

logger = structlog.get_logger(__name__)
settings = get_settings()


class HypothesisType(str, Enum):
    """Types of hypotheses for experiments"""
    CAUSAL = "causal"                    # X causes Y
    CORRELATION = "correlation"          # X correlates with Y
    PROPERTY = "property"                # X has property Y
    CONSISTENCY = "consistency"          # X is consistent with Y
    THRESHOLD = "threshold"              # X exceeds threshold Y
    RELATIONSHIP = "relationship"        # X relates to Y in manner Z


class ExperimentMethodType(str, Enum):
    """Methods for conducting experiments"""
    SIMULATION = "simulation"            # Computational simulation
    LOGICAL_DEDUCTION = "logical_deduction"  # Logical reasoning
    CONSISTENCY_CHECK = "consistency_check"  # Check against principles
    CAUSAL_ANALYSIS = "causal_analysis"  # Analyze causal chains
    STATISTICAL_TEST = "statistical_test"  # Statistical validation
    CROSS_VALIDATION = "cross_validation"  # Cross-check with other sources


class ExperimentStatus(str, Enum):
    """Status of experiments"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Hypothesis(PRSMBaseModel):
    """Testable hypothesis about SOCs or world model"""
    
    id: UUID = Field(default_factory=uuid4)
    hypothesis_type: HypothesisType
    description: str
    
    # Target SOCs
    primary_soc: str
    secondary_soc: Optional[str] = None
    
    # Prediction
    predicted_outcome: bool
    confidence_prediction: float = Field(ge=0.0, le=1.0)
    
    # Context
    domain: str = Field(default="general")
    conditions: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    
    
class ExperimentDesign(PRSMBaseModel):
    """Design specification for an experiment"""
    
    id: UUID = Field(default_factory=uuid4)
    hypothesis: Hypothesis
    method: ExperimentMethodType
    
    # Experiment parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_duration: float = Field(default=1.0)  # seconds
    resource_requirements: Dict[str, float] = Field(default_factory=dict)
    
    # Success criteria
    success_criteria: List[str] = Field(default_factory=list)
    validation_method: str = Field(default="outcome_match")
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class ExperimentExecution(PRSMBaseModel):
    """Record of experiment execution"""
    
    id: UUID = Field(default_factory=uuid4)
    experiment_design: ExperimentDesign
    
    # Execution details
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: ExperimentStatus = Field(default=ExperimentStatus.PENDING)
    
    # Results
    outcome: Optional[bool] = None
    confidence_outcome: Optional[float] = None
    measurements: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis
    hypothesis_confirmed: Optional[bool] = None
    confidence_change: float = Field(default=0.0)
    information_value: float = Field(default=0.0)
    
    # Sharing
    shared_with_network: bool = Field(default=False)
    sharing_timestamp: Optional[datetime] = None
    
    
class KnowledgeUpdate(PRSMBaseModel):
    """Update to knowledge base from experiment"""
    
    id: UUID = Field(default_factory=uuid4)
    experiment_id: UUID
    agent_id: str
    
    # Knowledge changes
    affected_socs: List[str] = Field(default_factory=list)
    confidence_updates: Dict[str, float] = Field(default_factory=dict)
    new_relations: List[str] = Field(default_factory=list)
    
    # Propagation
    propagated_to_network: bool = Field(default=False)
    propagation_timestamp: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    information_value: float = Field(default=0.0)


class BayesianSearchEngine:
    """
    Bayesian search engine for automated experimentation
    
    Implements the core experimental learning loop:
    1. Generate hypotheses about uncertain SOCs
    2. Design experiments to test hypotheses
    3. Execute experiments and collect results
    4. Update SOC confidence using Bayesian methods
    5. Share results with PRSM network
    6. Receive and integrate shared knowledge
    """
    
    def __init__(self, agent_id: str, world_model: WorldModelEngine, domain: str = "general"):
        self.agent_id = agent_id
        self.world_model = world_model
        self.domain = domain
        
        # Experiment management
        self.pending_experiments: Dict[str, ExperimentDesign] = {}
        self.running_experiments: Dict[str, ExperimentExecution] = {}
        self.completed_experiments: Dict[str, ExperimentExecution] = {}
        
        # Knowledge management
        self.knowledge_updates: List[KnowledgeUpdate] = []
        self.shared_knowledge: Dict[str, KnowledgeUpdate] = {}
        
        # Statistics
        self.experiment_stats = {
            "total_experiments": 0,
            "successful_experiments": 0,
            "failed_experiments": 0,
            "shared_experiments": 0,
            "received_knowledge": 0,
            "information_value_generated": 0.0
        }
        
        logger.info(
            "Bayesian Search Engine initialized",
            agent_id=agent_id,
            domain=domain
        )
        
    async def generate_hypotheses_for_soc(self, soc: SOC, max_hypotheses: int = 5) -> List[Hypothesis]:
        """
        Generate testable hypotheses for a SOC
        
        This is where the system becomes creative - generating novel hypotheses
        to test about uncertain SOCs.
        """
        
        hypotheses = []
        
        # Generate different types of hypotheses based on SOC confidence
        if soc.confidence < 0.7:  # Uncertain SOCs get more extensive testing
            
            # Causal hypotheses
            if soc.soc_type in [SOCType.CONCEPT, SOCType.PRINCIPLE]:
                for relation_type in [CausalRelationType.CAUSES, CausalRelationType.ENABLES]:
                    hypothesis = Hypothesis(
                        hypothesis_type=HypothesisType.CAUSAL,
                        description=f"SOC '{soc.name}' {relation_type.value} measurable effects in {soc.domain}",
                        primary_soc=soc.name,
                        predicted_outcome=True,
                        confidence_prediction=soc.confidence,
                        domain=soc.domain,
                        priority=1.0 - soc.confidence  # Lower confidence = higher priority
                    )
                    hypotheses.append(hypothesis)
                    
            # Consistency hypotheses
            hypothesis = Hypothesis(
                hypothesis_type=HypothesisType.CONSISTENCY,
                description=f"SOC '{soc.name}' is consistent with domain principles in {soc.domain}",
                primary_soc=soc.name,
                predicted_outcome=True,
                confidence_prediction=soc.confidence,
                domain=soc.domain,
                priority=1.0 - soc.confidence
            )
            hypotheses.append(hypothesis)
            
            # Property hypotheses
            if soc.properties:
                for prop_name, prop_value in soc.properties.items():
                    hypothesis = Hypothesis(
                        hypothesis_type=HypothesisType.PROPERTY,
                        description=f"SOC '{soc.name}' property '{prop_name}' is accurate",
                        primary_soc=soc.name,
                        predicted_outcome=True,
                        confidence_prediction=soc.confidence,
                        domain=soc.domain,
                        conditions=[f"property_{prop_name}"]
                    )
                    hypotheses.append(hypothesis)
                    
        # Limit to max_hypotheses
        hypotheses = sorted(hypotheses, key=lambda h: h.priority, reverse=True)[:max_hypotheses]
        
        logger.info(
            "Generated hypotheses for SOC",
            soc_name=soc.name,
            hypothesis_count=len(hypotheses),
            soc_confidence=soc.confidence
        )
        
        return hypotheses
        
    async def design_experiment(self, hypothesis: Hypothesis) -> ExperimentDesign:
        """
        Design experiment to test hypothesis
        
        Chooses appropriate experimental method based on hypothesis type
        and available resources.
        """
        
        # Choose method based on hypothesis type
        if hypothesis.hypothesis_type == HypothesisType.CAUSAL:
            method = ExperimentMethodType.CAUSAL_ANALYSIS
        elif hypothesis.hypothesis_type == HypothesisType.CONSISTENCY:
            method = ExperimentMethodType.CONSISTENCY_CHECK
        elif hypothesis.hypothesis_type == HypothesisType.PROPERTY:
            method = ExperimentMethodType.LOGICAL_DEDUCTION
        else:
            method = ExperimentMethodType.SIMULATION
            
        # Set experiment parameters
        parameters = {
            "target_soc": hypothesis.primary_soc,
            "domain": hypothesis.domain,
            "conditions": hypothesis.conditions,
            "confidence_threshold": 0.7
        }
        
        # Estimate duration based on complexity
        duration = 1.0  # Base duration
        if hypothesis.hypothesis_type == HypothesisType.CAUSAL:
            duration = 2.0  # Causal analysis takes longer
        elif method == ExperimentMethodType.SIMULATION:
            duration = 3.0  # Simulations take longest
            
        # Define success criteria
        success_criteria = [
            f"hypothesis_outcome_matches_prediction",
            f"confidence_change_exceeds_threshold",
            f"information_value_above_minimum"
        ]
        
        design = ExperimentDesign(
            hypothesis=hypothesis,
            method=method,
            parameters=parameters,
            expected_duration=duration,
            success_criteria=success_criteria,
            priority=hypothesis.priority
        )
        
        logger.info(
            "Designed experiment",
            hypothesis_id=str(hypothesis.id),
            method=method.value,
            expected_duration=duration
        )
        
        return design
        
    async def execute_experiment(self, design: ExperimentDesign) -> ExperimentExecution:
        """
        Execute experiment according to design
        
        This is where the actual "experimentation" happens - testing
        hypotheses through various methods.
        """
        
        execution = ExperimentExecution(
            experiment_design=design,
            status=ExperimentStatus.RUNNING
        )
        
        self.running_experiments[str(execution.id)] = execution
        
        try:
            # Execute based on method
            if design.method == ExperimentMethodType.CONSISTENCY_CHECK:
                result = await self._execute_consistency_check(design)
            elif design.method == ExperimentMethodType.CAUSAL_ANALYSIS:
                result = await self._execute_causal_analysis(design)
            elif design.method == ExperimentMethodType.LOGICAL_DEDUCTION:
                result = await self._execute_logical_deduction(design)
            elif design.method == ExperimentMethodType.SIMULATION:
                result = await self._execute_simulation(design)
            else:
                result = await self._execute_default_test(design)
                
            # Update execution with results
            execution.outcome = result["outcome"]
            execution.confidence_outcome = result["confidence"]
            execution.measurements = result["measurements"]
            execution.hypothesis_confirmed = result["outcome"] == design.hypothesis.predicted_outcome
            execution.status = ExperimentStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            
            # Calculate information value
            execution.information_value = self._calculate_information_value(execution)
            
            # Calculate confidence change
            execution.confidence_change = self._calculate_confidence_change(execution)
            
            # Update statistics
            self.experiment_stats["total_experiments"] += 1
            if execution.hypothesis_confirmed:
                self.experiment_stats["successful_experiments"] += 1
            else:
                self.experiment_stats["failed_experiments"] += 1
                
            self.experiment_stats["information_value_generated"] += execution.information_value
            
            # Move to completed
            self.completed_experiments[str(execution.id)] = execution
            del self.running_experiments[str(execution.id)]
            
            logger.info(
                "Experiment completed",
                experiment_id=str(execution.id),
                outcome=execution.outcome,
                hypothesis_confirmed=execution.hypothesis_confirmed,
                information_value=execution.information_value,
                confidence_change=execution.confidence_change
            )
            
        except Exception as e:
            execution.status = ExperimentStatus.FAILED
            execution.completed_at = datetime.now(timezone.utc)
            self.experiment_stats["failed_experiments"] += 1
            
            logger.error(
                "Experiment failed",
                experiment_id=str(execution.id),
                error=str(e)
            )
            
        return execution
        
    async def _execute_consistency_check(self, design: ExperimentDesign) -> Dict[str, Any]:
        """Execute consistency check experiment"""
        
        # Simulate processing time
        await asyncio.sleep(design.expected_duration * 0.1)
        
        # Get SOC from world model
        soc_name = design.parameters["target_soc"]
        
        # Find SOC in world model (simplified)
        domain_model = self.world_model.get_domain_model(design.parameters["domain"])
        
        if domain_model and soc_name in domain_model.core_principles:
            soc = domain_model.core_principles[soc_name]
            
            # Validate against world model
            validation_result = await self.world_model.validate_soc_against_world_model(soc)
            
            return {
                "outcome": validation_result.is_valid,
                "confidence": validation_result.confidence_score,
                "measurements": {
                    "supporting_principles": len(validation_result.supporting_principles),
                    "conflicting_principles": len(validation_result.conflicting_principles),
                    "validation_score": validation_result.confidence_score
                }
            }
        else:
            # Default behavior for unknown SOCs
            return {
                "outcome": random.choice([True, False]),
                "confidence": random.uniform(0.3, 0.8),
                "measurements": {"method": "default_consistency_check"}
            }
            
    async def _execute_causal_analysis(self, design: ExperimentDesign) -> Dict[str, Any]:
        """Execute causal analysis experiment"""
        
        await asyncio.sleep(design.expected_duration * 0.1)
        
        # Analyze causal relationships
        soc_name = design.parameters["target_soc"]
        related_socs = self.world_model.get_related_socs(soc_name)
        
        # Simple causal analysis
        causal_strength = len(related_socs) * 0.1
        outcome = causal_strength > 0.3
        
        return {
            "outcome": outcome,
            "confidence": min(0.9, 0.5 + causal_strength),
            "measurements": {
                "related_socs": len(related_socs),
                "causal_strength": causal_strength,
                "method": "causal_analysis"
            }
        }
        
    async def _execute_logical_deduction(self, design: ExperimentDesign) -> Dict[str, Any]:
        """Execute logical deduction experiment"""
        
        await asyncio.sleep(design.expected_duration * 0.1)
        
        # Simple logical deduction
        # In full implementation, this would use formal logic
        
        outcome = random.choice([True, False])
        confidence = random.uniform(0.4, 0.9)
        
        return {
            "outcome": outcome,
            "confidence": confidence,
            "measurements": {"method": "logical_deduction"}
        }
        
    async def _execute_simulation(self, design: ExperimentDesign) -> Dict[str, Any]:
        """Execute simulation experiment"""
        
        await asyncio.sleep(design.expected_duration * 0.1)
        
        # Simulate complex experiment
        # In full implementation, this would run actual simulations
        
        # Bias toward success for higher confidence predictions
        prediction_confidence = design.hypothesis.confidence_prediction
        success_probability = 0.5 + (prediction_confidence - 0.5) * 0.6
        
        outcome = random.random() < success_probability
        confidence = random.uniform(0.4, 0.9)
        
        return {
            "outcome": outcome,
            "confidence": confidence,
            "measurements": {
                "simulation_steps": random.randint(100, 1000),
                "success_probability": success_probability,
                "method": "simulation"
            }
        }
        
    async def _execute_default_test(self, design: ExperimentDesign) -> Dict[str, Any]:
        """Execute default test for unknown methods"""
        
        await asyncio.sleep(design.expected_duration * 0.1)
        
        outcome = random.choice([True, False])
        confidence = random.uniform(0.3, 0.7)
        
        return {
            "outcome": outcome,
            "confidence": confidence,
            "measurements": {"method": "default_test"}
        }
        
    def _calculate_information_value(self, execution: ExperimentExecution) -> float:
        """
        Calculate information value of experiment
        
        This is crucial - even failed experiments provide valuable information
        by ruling out incorrect hypotheses.
        """
        
        # Base information value
        base_value = 0.1
        
        # Confidence change contributes to value
        confidence_contribution = abs(execution.confidence_change) * 0.5
        
        # Surprising results (prediction mismatch) are more valuable
        surprise_bonus = 0.0
        if execution.hypothesis_confirmed is False:
            surprise_bonus = 0.3  # Failed predictions are valuable!
            
        # Domain-specific bonuses
        domain_bonus = 0.0
        if execution.experiment_design.hypothesis.domain in ["physics", "chemistry", "biology"]:
            domain_bonus = 0.1
            
        total_value = base_value + confidence_contribution + surprise_bonus + domain_bonus
        
        return min(1.0, total_value)
        
    def _calculate_confidence_change(self, execution: ExperimentExecution) -> float:
        """Calculate how much confidence should change based on experiment"""
        
        # Bayesian update calculation
        if execution.hypothesis_confirmed:
            # Hypothesis confirmed - increase confidence
            change = execution.information_value * 0.1
        else:
            # Hypothesis rejected - decrease confidence
            change = -execution.information_value * 0.1
            
        return change
        
    async def update_soc_from_experiment(self, soc: SOC, execution: ExperimentExecution):
        """Update SOC confidence based on experiment results"""
        
        # Apply Bayesian update
        old_confidence = soc.confidence
        
        # Update confidence
        soc.update_confidence(
            soc.confidence + execution.confidence_change,
            weight=execution.information_value
        )
        
        # Create knowledge update record
        knowledge_update = KnowledgeUpdate(
            experiment_id=execution.id,
            agent_id=self.agent_id,
            affected_socs=[soc.name],
            confidence_updates={soc.name: execution.confidence_change},
            information_value=execution.information_value
        )
        
        self.knowledge_updates.append(knowledge_update)
        
        logger.info(
            "Updated SOC from experiment",
            soc_name=soc.name,
            old_confidence=old_confidence,
            new_confidence=soc.confidence,
            confidence_change=execution.confidence_change,
            information_value=execution.information_value
        )
        
    async def share_experiment_result(self, execution: ExperimentExecution):
        """
        Share experiment result with PRSM network
        
        This is where the "hive mind" effect happens - sharing both
        successes and failures across the network.
        """
        
        # Only share high-value experiments
        if execution.information_value > 0.1:
            
            # Create shareable experiment result
            shared_result = ExperimentResult(
                experiment_type=ExperimentType.LOGICAL_TEST,  # Map to ExperimentType
                agent_id=self.agent_id,
                domain=execution.experiment_design.hypothesis.domain,
                hypothesis=execution.experiment_design.hypothesis.description,
                method=execution.experiment_design.method.value,
                success=execution.hypothesis_confirmed or False,
                confidence_change=execution.confidence_change,
                affected_socs=[execution.experiment_design.hypothesis.primary_soc],
                execution_time=execution.experiment_design.expected_duration,
                information_value=execution.information_value
            )
            
            # Mark as shared
            execution.shared_with_network = True
            execution.sharing_timestamp = datetime.now(timezone.utc)
            
            # Update statistics
            self.experiment_stats["shared_experiments"] += 1
            
            # In full implementation, this would broadcast to PRSM marketplace
            logger.info(
                "Shared experiment result with network",
                experiment_id=str(execution.id),
                information_value=execution.information_value,
                hypothesis_confirmed=execution.hypothesis_confirmed,
                shared_result_id=str(shared_result.id)
            )
            
            return shared_result
            
        return None
        
    async def receive_shared_knowledge(self, knowledge_update: KnowledgeUpdate):
        """
        Receive and integrate shared knowledge from other agents
        
        This implements the "hive mind" effect where validated knowledge
        from other agents updates local SOCs.
        """
        
        # Store shared knowledge
        self.shared_knowledge[str(knowledge_update.id)] = knowledge_update
        
        # Update statistics
        self.experiment_stats["received_knowledge"] += 1
        
        logger.info(
            "Received shared knowledge",
            from_agent=knowledge_update.agent_id,
            affected_socs=knowledge_update.affected_socs,
            information_value=knowledge_update.information_value,
            confidence_updates=knowledge_update.confidence_updates
        )
        
    async def run_experiment_cycle(self, soc: SOC, max_experiments: int = 3) -> List[ExperimentExecution]:
        """
        Run complete experiment cycle for a SOC
        
        This is the main entry point for automated experimentation:
        1. Generate hypotheses
        2. Design experiments
        3. Execute experiments
        4. Update SOC confidence
        5. Share results
        """
        
        logger.info(
            "Starting experiment cycle",
            soc_name=soc.name,
            soc_confidence=soc.confidence,
            max_experiments=max_experiments
        )
        
        # Generate hypotheses
        hypotheses = await self.generate_hypotheses_for_soc(soc, max_experiments)
        
        executions = []
        
        for hypothesis in hypotheses:
            try:
                # Design experiment
                design = await self.design_experiment(hypothesis)
                
                # Execute experiment
                execution = await self.execute_experiment(design)
                
                # Update SOC based on results
                await self.update_soc_from_experiment(soc, execution)
                
                # Share valuable results
                await self.share_experiment_result(execution)
                
                executions.append(execution)
                
            except Exception as e:
                logger.error(
                    "Error in experiment cycle",
                    hypothesis_id=str(hypothesis.id),
                    error=str(e)
                )
                
        logger.info(
            "Experiment cycle completed",
            soc_name=soc.name,
            experiments_completed=len(executions),
            new_confidence=soc.confidence
        )
        
        return executions
        
    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get experiment statistics"""
        
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "experiment_stats": self.experiment_stats.copy(),
            "pending_experiments": len(self.pending_experiments),
            "running_experiments": len(self.running_experiments),
            "completed_experiments": len(self.completed_experiments),
            "knowledge_updates": len(self.knowledge_updates),
            "shared_knowledge_received": len(self.shared_knowledge)
        }
        
    async def get_most_informative_failures(self, limit: int = 10) -> List[ExperimentExecution]:
        """
        Get most informative failed experiments
        
        This highlights the value of failures - they provide crucial
        information about what doesn't work.
        """
        
        failed_experiments = [
            execution for execution in self.completed_experiments.values()
            if execution.hypothesis_confirmed is False
        ]
        
        # Sort by information value
        failed_experiments.sort(key=lambda x: x.information_value, reverse=True)
        
        return failed_experiments[:limit]
        
    async def export_experiment_history(self) -> Dict[str, Any]:
        """Export experiment history for sharing or analysis"""
        
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "experiment_stats": self.experiment_stats,
            "completed_experiments": {
                k: v.dict() for k, v in self.completed_experiments.items()
            },
            "knowledge_updates": [ku.dict() for ku in self.knowledge_updates],
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }


# Factory functions for integration

def create_bayesian_search_engine(
    agent_id: str, 
    world_model: WorldModelEngine, 
    domain: str = "general"
) -> BayesianSearchEngine:
    """Create Bayesian search engine instance"""
    
    return BayesianSearchEngine(agent_id, world_model, domain)


def create_domain_specialized_search_engine(
    agent_id: str,
    world_model: WorldModelEngine,
    domain: str,
    temperature: float = 0.7
) -> BayesianSearchEngine:
    """
    Create domain-specialized search engine
    
    This can be used to create agents with different exploration strategies
    by varying the temperature parameter.
    """
    
    engine = BayesianSearchEngine(agent_id, world_model, domain)
    
    # Adjust exploration based on temperature
    if temperature > 0.8:
        # High temperature - more exploratory
        engine.experiment_stats["exploration_bias"] = 0.8
    elif temperature < 0.4:
        # Low temperature - more conservative
        engine.experiment_stats["exploration_bias"] = 0.2
    else:
        # Medium temperature - balanced
        engine.experiment_stats["exploration_bias"] = 0.5
        
    return engine


async def create_experiment_sharing_network(agents: List[BayesianSearchEngine]):
    """
    Create network for sharing experiments between agents
    
    This implements the hive mind effect where agents share knowledge
    """
    
    logger.info(
        "Creating experiment sharing network",
        agent_count=len(agents)
    )
    
    # In full implementation, this would set up PRSM marketplace connections
    # For now, just log the network creation
    
    for agent in agents:
        logger.info(
            "Agent added to sharing network",
            agent_id=agent.agent_id,
            domain=agent.domain
        )
        
    return agents
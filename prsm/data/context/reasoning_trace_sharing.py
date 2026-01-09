"""
Reasoning Trace Sharing Between Agents

🧠 COGNITION.AI INSIGHTS INTEGRATION:
- Comprehensive agent reasoning trail propagation with full context sharing
- Inter-agent reasoning context synchronization for coordinated decision-making
- Reasoning step dependency tracking and validation across agent interactions
- Implicit decision capture and explicit reasoning communication between agents
- Conflict detection and resolution when agents have different reasoning paths

This module implements sophisticated reasoning trace sharing that addresses
Cognition.AI's core insight: "actions carry implicit decisions" and agents
need to share full reasoning context, not just final outputs.
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import (
    PRSMBaseModel, TimestampMixin, ReasoningStep, AgentType, TaskStatus, SafetyLevel
)
from prsm.data.context.enhanced_context_compression import (
    ContextSegment, ContextType, ContextImportance, EnhancedContextCompressionEngine
)
from prsm.compute.agents.executors.model_executor import ModelExecutor

logger = structlog.get_logger(__name__)


class ReasoningTraceLevel(str, Enum):
    """Levels of reasoning trace detail"""
    MINIMAL = "minimal"           # Only final decisions and critical steps
    STANDARD = "standard"         # Key reasoning steps and intermediate results
    DETAILED = "detailed"         # Complete reasoning process with alternatives
    COMPREHENSIVE = "comprehensive" # Full trace including failed attempts and reconsiderations


class ReasoningConflictType(str, Enum):
    """Types of reasoning conflicts between agents"""
    CONTRADICTORY_CONCLUSIONS = "contradictory_conclusions"
    INCOMPATIBLE_ASSUMPTIONS = "incompatible_assumptions"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    DIFFERENT_METHODOLOGIES = "different_methodologies"
    PRIORITY_DISAGREEMENT = "priority_disagreement"
    RESOURCE_CONTENTION = "resource_contention"
    TIMING_CONFLICTS = "timing_conflicts"
    SAFETY_CONCERNS = "safety_concerns"


class ReasoningStepType(str, Enum):
    """Types of reasoning steps for detailed tracking"""
    PROBLEM_ANALYSIS = "problem_analysis"
    HYPOTHESIS_FORMATION = "hypothesis_formation"
    EVIDENCE_GATHERING = "evidence_gathering"
    OPTION_EVALUATION = "option_evaluation"
    DECISION_MAKING = "decision_making"
    ACTION_PLANNING = "action_planning"
    RESULT_VALIDATION = "result_validation"
    ERROR_CORRECTION = "error_correction"
    ASSUMPTION_VERIFICATION = "assumption_verification"
    ALTERNATIVE_CONSIDERATION = "alternative_consideration"


class ImplicitDecisionType(str, Enum):
    """Types of implicit decisions that need to be captured"""
    PARAMETER_SELECTION = "parameter_selection"
    METHOD_CHOICE = "method_choice"
    PRIORITY_ORDERING = "priority_ordering"
    RISK_TOLERANCE = "risk_tolerance"
    QUALITY_THRESHOLD = "quality_threshold"
    RESOURCE_ALLOCATION = "resource_allocation"
    TIMING_DECISION = "timing_decision"
    SCOPE_LIMITATION = "scope_limitation"
    ASSUMPTION_ACCEPTANCE = "assumption_acceptance"
    TRADE_OFF_RESOLUTION = "trade_off_resolution"


class EnhancedReasoningStep(TimestampMixin):
    """Enhanced reasoning step with comprehensive context"""
    step_id: UUID = Field(default_factory=uuid4)
    parent_step_id: Optional[UUID] = None
    agent_id: str
    agent_type: AgentType
    
    # Step Classification
    step_type: ReasoningStepType
    step_description: str
    step_rationale: str
    
    # Input/Output Context
    input_context: Dict[str, Any] = Field(default_factory=dict)
    output_result: Dict[str, Any] = Field(default_factory=dict)
    context_dependencies: List[UUID] = Field(default_factory=list)
    
    # Decision Information
    explicit_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    implicit_decisions: List[Dict[ImplicitDecisionType, Any]] = Field(default_factory=list)
    decision_criteria: List[str] = Field(default_factory=list)
    
    # Reasoning Process
    assumptions_made: List[str] = Field(default_factory=list)
    evidence_considered: List[Dict[str, Any]] = Field(default_factory=list)
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_options: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Quality and Confidence
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    uncertainty_factors: List[str] = Field(default_factory=list)
    validation_performed: bool = False
    validation_results: Optional[Dict[str, Any]] = None
    
    # Inter-Agent Context
    shared_with_agents: List[str] = Field(default_factory=list)
    dependent_on_agents: List[str] = Field(default_factory=list)
    conflicts_detected: List[ReasoningConflictType] = Field(default_factory=list)
    
    # Execution Metadata
    execution_time_seconds: float = Field(default=0.0)
    computational_cost: float = Field(default=0.0)
    memory_usage: Optional[int] = None


class ReasoningTraceGraph(TimestampMixin):
    """Graph representation of interconnected reasoning traces"""
    trace_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    root_task_id: str
    
    # Graph Structure
    reasoning_steps: Dict[UUID, EnhancedReasoningStep] = Field(default_factory=dict)
    step_dependencies: Dict[UUID, List[UUID]] = Field(default_factory=dict)
    agent_contributions: Dict[str, List[UUID]] = Field(default_factory=dict)
    
    # Trace Metadata
    trace_level: ReasoningTraceLevel = ReasoningTraceLevel.STANDARD
    total_steps: int = Field(default=0)
    participating_agents: List[str] = Field(default_factory=list)
    
    # Conflict Analysis
    detected_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    resolved_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Quality Metrics
    trace_coherence_score: float = Field(ge=0.0, le=1.0, default=1.0)
    decision_consistency_score: float = Field(ge=0.0, le=1.0, default=1.0)
    agent_coordination_score: float = Field(ge=0.0, le=1.0, default=1.0)
    
    # Synchronization
    last_sync_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sync_conflicts: List[str] = Field(default_factory=list)


class ReasoningConflictResolution(TimestampMixin):
    """Conflict resolution result between agent reasoning"""
    resolution_id: UUID = Field(default_factory=uuid4)
    conflict_type: ReasoningConflictType
    involved_agents: List[str] = Field(default_factory=list)
    conflicting_steps: List[UUID] = Field(default_factory=list)
    
    # Conflict Details
    conflict_description: str
    conflict_severity: SafetyLevel = SafetyLevel.MEDIUM
    impact_assessment: Dict[str, Any] = Field(default_factory=dict)
    
    # Resolution Process
    resolution_strategy: str
    resolution_rationale: str
    consensus_reached: bool = False
    
    # Resolution Result
    agreed_reasoning: Optional[Dict[str, Any]] = None
    compromise_solution: Optional[Dict[str, Any]] = None
    escalation_required: bool = False
    
    # Quality Assessment
    resolution_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    stakeholder_satisfaction: Dict[str, float] = Field(default_factory=dict)
    
    # Follow-up
    monitoring_required: bool = False
    follow_up_actions: List[str] = Field(default_factory=list)


class AgentReasoningContext(TimestampMixin):
    """Agent-specific reasoning context for synchronization"""
    context_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    agent_type: AgentType
    session_id: UUID
    
    # Current Reasoning State
    active_reasoning_steps: List[UUID] = Field(default_factory=list)
    completed_reasoning_steps: List[UUID] = Field(default_factory=list)
    pending_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Context Dependencies
    required_context_from_agents: Dict[str, List[str]] = Field(default_factory=dict)
    provided_context_to_agents: Dict[str, List[str]] = Field(default_factory=dict)
    blocking_dependencies: List[UUID] = Field(default_factory=list)
    
    # Reasoning Preferences
    reasoning_style: str = "systematic"  # systematic, exploratory, cautious, aggressive
    decision_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    collaboration_preference: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Quality Tracking
    reasoning_quality_score: float = Field(ge=0.0, le=1.0, default=0.8)
    consistency_with_peers: float = Field(ge=0.0, le=1.0, default=0.8)
    contribution_value: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Synchronization Status
    last_sync_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sync_status: str = "synchronized"  # synchronized, out_of_sync, conflict, updating
    pending_sync_requests: List[str] = Field(default_factory=list)


class ReasoningTraceSharingEngine:
    """
    Reasoning Trace Sharing Engine
    
    Implements comprehensive reasoning trace sharing between agents:
    - Full reasoning context propagation beyond simple message passing
    - Implicit decision capture and explicit communication of reasoning paths
    - Inter-agent reasoning synchronization with conflict detection and resolution
    - Dependency tracking and validation across agent interactions
    - Quality assessment and consistency validation of shared reasoning
    """
    
    def __init__(self, session_id: UUID):
        self.session_id = session_id
        self.model_executor = ModelExecutor()
        self.context_engine = EnhancedContextCompressionEngine()
        
        # Reasoning Trace Storage
        self.reasoning_traces: Dict[UUID, ReasoningTraceGraph] = {}
        self.agent_contexts: Dict[str, AgentReasoningContext] = {}
        self.conflict_resolutions: List[ReasoningConflictResolution] = []
        
        # Synchronization Management
        self.sync_queue: List[Dict[str, Any]] = []
        self.sync_locks: Dict[str, bool] = {}
        self.last_global_sync: datetime = datetime.now(timezone.utc)
        
        # Quality Tracking
        self.sharing_metrics = {
            "total_steps_shared": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "average_sync_time": 0.0,
            "coordination_effectiveness": 0.8
        }
        
        logger.info("ReasoningTraceSharingEngine initialized", session_id=session_id)
    
    async def create_reasoning_trace(
        self,
        root_task_id: str,
        trace_level: ReasoningTraceLevel = ReasoningTraceLevel.STANDARD
    ) -> ReasoningTraceGraph:
        """
        Create new reasoning trace graph for multi-agent coordination
        """
        trace = ReasoningTraceGraph(
            session_id=self.session_id,
            root_task_id=root_task_id,
            trace_level=trace_level
        )
        
        self.reasoning_traces[trace.trace_id] = trace
        
        logger.info("Reasoning trace created",
                   trace_id=trace.trace_id,
                   task_id=root_task_id,
                   level=trace_level)
        
        return trace
    
    async def add_reasoning_step(
        self,
        trace_id: UUID,
        agent_id: str,
        agent_type: AgentType,
        step_type: ReasoningStepType,
        step_description: str,
        input_context: Dict[str, Any],
        output_result: Dict[str, Any],
        parent_step_id: Optional[UUID] = None
    ) -> EnhancedReasoningStep:
        """
        Add comprehensive reasoning step with full context capture
        """
        if trace_id not in self.reasoning_traces:
            logger.error("Reasoning trace not found", trace_id=trace_id)
            raise ValueError(f"Reasoning trace {trace_id} not found")
        
        trace = self.reasoning_traces[trace_id]
        
        # Create enhanced reasoning step
        reasoning_step = EnhancedReasoningStep(
            parent_step_id=parent_step_id,
            agent_id=agent_id,
            agent_type=agent_type,
            step_type=step_type,
            step_description=step_description,
            input_context=input_context,
            output_result=output_result
        )
        
        # Capture implicit decisions
        reasoning_step.implicit_decisions = await self._capture_implicit_decisions(
            input_context, output_result, step_type
        )
        
        # Generate step rationale
        reasoning_step.step_rationale = await self._generate_step_rationale(
            reasoning_step, trace
        )
        
        # Identify assumptions
        reasoning_step.assumptions_made = await self._identify_assumptions(
            input_context, output_result, reasoning_step.step_rationale
        )
        
        # Assess confidence and uncertainty
        reasoning_step.confidence_score = await self._assess_step_confidence(reasoning_step)
        reasoning_step.uncertainty_factors = await self._identify_uncertainty_factors(reasoning_step)
        
        # Add to trace
        trace.reasoning_steps[reasoning_step.step_id] = reasoning_step
        trace.total_steps += 1
        
        # Update agent contributions
        if agent_id not in trace.agent_contributions:
            trace.agent_contributions[agent_id] = []
        trace.agent_contributions[agent_id].append(reasoning_step.step_id)
        
        if agent_id not in trace.participating_agents:
            trace.participating_agents.append(agent_id)
        
        # Update step dependencies
        if parent_step_id:
            if parent_step_id not in trace.step_dependencies:
                trace.step_dependencies[parent_step_id] = []
            trace.step_dependencies[parent_step_id].append(reasoning_step.step_id)
        
        # Update agent context
        await self._update_agent_context(agent_id, agent_type, reasoning_step)
        
        # Check for conflicts
        conflicts = await self._detect_reasoning_conflicts(reasoning_step, trace)
        if conflicts:
            trace.detected_conflicts.extend(conflicts)
            reasoning_step.conflicts_detected = [c["type"] for c in conflicts]
        
        # Share with relevant agents
        await self._share_reasoning_step(reasoning_step, trace)
        
        # Update sharing metrics
        self.sharing_metrics["total_steps_shared"] += 1
        
        logger.info("Reasoning step added and shared",
                   step_id=reasoning_step.step_id,
                   agent_id=agent_id,
                   step_type=step_type,
                   conflicts_detected=len(reasoning_step.conflicts_detected))
        
        return reasoning_step
    
    async def _capture_implicit_decisions(
        self,
        input_context: Dict[str, Any],
        output_result: Dict[str, Any],
        step_type: ReasoningStepType
    ) -> List[Dict[ImplicitDecisionType, Any]]:
        """Capture implicit decisions made during reasoning step"""
        implicit_decisions = []
        
        # Analyze input/output differences to identify implicit decisions
        try:
            # Parameter selection decisions
            if "parameters" in input_context and "chosen_parameters" in output_result:
                implicit_decisions.append({
                    ImplicitDecisionType.PARAMETER_SELECTION: {
                        "available_options": input_context.get("parameters"),
                        "chosen_option": output_result.get("chosen_parameters"),
                        "selection_criteria": "optimal_performance"
                    }
                })
            
            # Method choice decisions
            if "available_methods" in input_context and "method_used" in output_result:
                implicit_decisions.append({
                    ImplicitDecisionType.METHOD_CHOICE: {
                        "available_methods": input_context.get("available_methods"),
                        "chosen_method": output_result.get("method_used"),
                        "choice_rationale": "best_suited_for_task"
                    }
                })
            
            # Quality threshold decisions
            if "quality_score" in output_result:
                quality_score = output_result.get("quality_score", 0.5)
                if quality_score >= 0.7:  # Implicit acceptance threshold
                    implicit_decisions.append({
                        ImplicitDecisionType.QUALITY_THRESHOLD: {
                            "threshold_used": 0.7,
                            "actual_quality": quality_score,
                            "decision": "accepted"
                        }
                    })
            
            # Risk tolerance decisions
            if step_type in [ReasoningStepType.DECISION_MAKING, ReasoningStepType.ACTION_PLANNING]:
                risk_level = self._assess_implicit_risk_tolerance(input_context, output_result)
                implicit_decisions.append({
                    ImplicitDecisionType.RISK_TOLERANCE: {
                        "assessed_risk_level": risk_level,
                        "tolerance_demonstrated": "moderate",
                        "risk_mitigation": output_result.get("safety_measures", [])
                    }
                })
            
        except Exception as e:
            logger.error("Failed to capture implicit decisions", error=str(e))
        
        return implicit_decisions
    
    def _assess_implicit_risk_tolerance(
        self,
        input_context: Dict[str, Any],
        output_result: Dict[str, Any]
    ) -> str:
        """Assess implicit risk tolerance from decision patterns"""
        # Look for safety measures, validation steps, backup plans
        safety_indicators = [
            output_result.get("safety_measures", []),
            output_result.get("validation_steps", []),
            output_result.get("backup_plans", []),
            output_result.get("error_handling", [])
        ]
        
        total_safety_measures = sum(len(measures) for measures in safety_indicators if measures)
        
        if total_safety_measures >= 5:
            return "conservative"
        elif total_safety_measures >= 2:
            return "moderate"
        else:
            return "aggressive"
    
    async def _generate_step_rationale(
        self,
        reasoning_step: EnhancedReasoningStep,
        trace: ReasoningTraceGraph
    ) -> str:
        """Generate comprehensive rationale for reasoning step"""
        try:
            rationale_prompt = f"""
            Generate a clear rationale for this reasoning step:
            
            Step Type: {reasoning_step.step_type.value}
            Description: {reasoning_step.step_description}
            Agent: {reasoning_step.agent_type.value}
            
            Input Context: {json.dumps(reasoning_step.input_context, indent=2)}
            Output Result: {json.dumps(reasoning_step.output_result, indent=2)}
            
            Explain:
            1. Why this step was necessary
            2. How the input led to the output
            3. What assumptions or decisions were made
            4. How this fits into the overall reasoning process
            
            Provide a concise but comprehensive rationale:
            """
            
            response = await self.model_executor.process({
                "task": rationale_prompt,
                "models": ["gpt-3.5-turbo"],
                "parallel": False
            })
            
            if response and response[0].success:
                return response[0].result.get("content", "Step executed to advance reasoning process")
            else:
                return f"{reasoning_step.step_type.value} performed to process {reasoning_step.step_description}"
                
        except Exception as e:
            logger.error("Failed to generate step rationale", error=str(e))
            return f"Reasoning step: {reasoning_step.step_description}"
    
    async def _identify_assumptions(
        self,
        input_context: Dict[str, Any],
        output_result: Dict[str, Any],
        rationale: str
    ) -> List[str]:
        """Identify assumptions made during reasoning step"""
        assumptions = []
        
        # Look for assumption indicators in rationale
        assumption_keywords = [
            "assuming", "given that", "presume", "suppose", "if we assume",
            "taking for granted", "based on the premise", "under the assumption"
        ]
        
        rationale_lower = rationale.lower()
        for keyword in assumption_keywords:
            if keyword in rationale_lower:
                # Extract assumption context
                start_idx = rationale_lower.find(keyword)
                assumption_text = rationale[start_idx:start_idx + 100]
                assumptions.append(assumption_text.strip())
        
        # Identify implicit assumptions from input/output patterns
        if "data_quality" not in input_context and "confidence" in output_result:
            assumptions.append("Input data is of sufficient quality for analysis")
        
        if "time_constraints" not in input_context and "execution_time" in output_result:
            assumptions.append("No strict time constraints on processing")
        
        if len(assumptions) == 0:
            assumptions.append("Standard operational assumptions apply")
        
        return list(set(assumptions))  # Remove duplicates
    
    async def _assess_step_confidence(self, reasoning_step: EnhancedReasoningStep) -> float:
        """Assess confidence level for reasoning step"""
        base_confidence = 0.7  # Default confidence
        
        # Adjust based on step type
        confidence_adjustments = {
            ReasoningStepType.PROBLEM_ANALYSIS: 0.8,
            ReasoningStepType.HYPOTHESIS_FORMATION: 0.6,
            ReasoningStepType.EVIDENCE_GATHERING: 0.9,
            ReasoningStepType.DECISION_MAKING: 0.7,
            ReasoningStepType.RESULT_VALIDATION: 0.8,
            ReasoningStepType.ERROR_CORRECTION: 0.6
        }
        
        step_confidence = confidence_adjustments.get(reasoning_step.step_type, base_confidence)
        
        # Adjust based on available information
        if len(reasoning_step.input_context) > 3:
            step_confidence += 0.1  # More context = higher confidence
        
        if len(reasoning_step.assumptions_made) > 2:
            step_confidence -= 0.1  # More assumptions = lower confidence
        
        if reasoning_step.output_result.get("validation_performed", False):
            step_confidence += 0.15  # Validation increases confidence
        
        return max(0.0, min(1.0, step_confidence))
    
    async def _identify_uncertainty_factors(self, reasoning_step: EnhancedReasoningStep) -> List[str]:
        """Identify factors contributing to uncertainty in reasoning step"""
        uncertainty_factors = []
        
        # Check for uncertainty indicators
        if len(reasoning_step.assumptions_made) > 2:
            uncertainty_factors.append("Multiple assumptions required")
        
        if reasoning_step.confidence_score < 0.7:
            uncertainty_factors.append("Limited confidence in step outcome")
        
        if "incomplete_data" in str(reasoning_step.input_context):
            uncertainty_factors.append("Incomplete input data")
        
        if "ambiguous" in reasoning_step.step_description.lower():
            uncertainty_factors.append("Ambiguous problem specification")
        
        if reasoning_step.step_type == ReasoningStepType.HYPOTHESIS_FORMATION:
            uncertainty_factors.append("Speculative reasoning involved")
        
        # Check output for uncertainty indicators
        output_str = str(reasoning_step.output_result).lower()
        if any(word in output_str for word in ["uncertain", "unclear", "possibly", "might", "could be"]):
            uncertainty_factors.append("Uncertain outcome expressed")
        
        return uncertainty_factors
    
    async def _update_agent_context(
        self,
        agent_id: str,
        agent_type: AgentType,
        reasoning_step: EnhancedReasoningStep
    ):
        """Update agent's reasoning context with new step"""
        if agent_id not in self.agent_contexts:
            self.agent_contexts[agent_id] = AgentReasoningContext(
                agent_id=agent_id,
                agent_type=agent_type,
                session_id=self.session_id
            )
        
        context = self.agent_contexts[agent_id]
        
        # Update step tracking
        if reasoning_step.step_id in context.active_reasoning_steps:
            context.active_reasoning_steps.remove(reasoning_step.step_id)
        context.completed_reasoning_steps.append(reasoning_step.step_id)
        
        # Update quality tracking
        step_quality = reasoning_step.confidence_score
        current_quality = context.reasoning_quality_score
        context.reasoning_quality_score = (current_quality * 0.8) + (step_quality * 0.2)
        
        # Update last sync time
        context.last_sync_time = datetime.now(timezone.utc)
    
    async def _detect_reasoning_conflicts(
        self,
        new_step: EnhancedReasoningStep,
        trace: ReasoningTraceGraph
    ) -> List[Dict[str, Any]]:
        """Detect conflicts between new reasoning step and existing steps"""
        conflicts = []
        
        # Check against other agents' reasoning steps
        for step_id, existing_step in trace.reasoning_steps.items():
            if existing_step.agent_id != new_step.agent_id:
                # Look for potential conflicts
                conflict = await self._analyze_step_conflict(new_step, existing_step)
                if conflict:
                    conflicts.append(conflict)
                    
                    # Update conflict tracking
                    self.sharing_metrics["conflicts_detected"] += 1
        
        return conflicts
    
    async def _analyze_step_conflict(
        self,
        step1: EnhancedReasoningStep,
        step2: EnhancedReasoningStep
    ) -> Optional[Dict[str, Any]]:
        """Analyze potential conflict between two reasoning steps"""
        # Check for contradictory conclusions
        if step1.step_type == step2.step_type == ReasoningStepType.DECISION_MAKING:
            decision1 = step1.output_result.get("decision")
            decision2 = step2.output_result.get("decision")
            
            if decision1 and decision2 and decision1 != decision2:
                return {
                    "type": ReasoningConflictType.CONTRADICTORY_CONCLUSIONS,
                    "severity": SafetyLevel.HIGH,
                    "description": f"Agents {step1.agent_id} and {step2.agent_id} reached different decisions",
                    "step1_id": step1.step_id,
                    "step2_id": step2.step_id,
                    "conflicting_decisions": {"agent1": decision1, "agent2": decision2}
                }
        
        # Check for incompatible assumptions
        common_assumptions = set(step1.assumptions_made) & set(step2.assumptions_made)
        if len(common_assumptions) == 0 and len(step1.assumptions_made) > 0 and len(step2.assumptions_made) > 0:
            # Different assumptions might indicate conflict
            return {
                "type": ReasoningConflictType.INCOMPATIBLE_ASSUMPTIONS,
                "severity": SafetyLevel.MEDIUM,
                "description": f"Agents {step1.agent_id} and {step2.agent_id} are using different assumptions",
                "step1_id": step1.step_id,
                "step2_id": step2.step_id,
                "conflicting_assumptions": {
                    "agent1": step1.assumptions_made,
                    "agent2": step2.assumptions_made
                }
            }
        
        # Check for timing conflicts
        time_diff = abs((step1.created_at - step2.created_at).total_seconds())
        if time_diff < 5 and step1.step_type == step2.step_type:  # Same type within 5 seconds
            return {
                "type": ReasoningConflictType.TIMING_CONFLICTS,
                "severity": SafetyLevel.LOW,
                "description": f"Concurrent reasoning steps of same type by different agents",
                "step1_id": step1.step_id,
                "step2_id": step2.step_id,
                "time_difference": time_diff
            }
        
        return None
    
    async def _share_reasoning_step(
        self,
        reasoning_step: EnhancedReasoningStep,
        trace: ReasoningTraceGraph
    ):
        """Share reasoning step with relevant agents"""
        # Determine which agents need this reasoning step
        relevant_agents = await self._identify_relevant_agents(reasoning_step, trace)
        
        for agent_id in relevant_agents:
            if agent_id != reasoning_step.agent_id:  # Don't share with originating agent
                await self._send_reasoning_context_to_agent(
                    agent_id, reasoning_step, trace
                )
                reasoning_step.shared_with_agents.append(agent_id)
        
        # Update agent dependencies
        await self._update_agent_dependencies(reasoning_step, trace)
    
    async def _identify_relevant_agents(
        self,
        reasoning_step: EnhancedReasoningStep,
        trace: ReasoningTraceGraph
    ) -> List[str]:
        """Identify which agents need to receive this reasoning step"""
        relevant_agents = []
        
        # All participating agents should receive critical steps
        if reasoning_step.step_type in [
            ReasoningStepType.DECISION_MAKING,
            ReasoningStepType.ERROR_CORRECTION,
            ReasoningStepType.RESULT_VALIDATION
        ]:
            relevant_agents.extend(trace.participating_agents)
        
        # Agents with dependencies should receive related steps
        for agent_id, context in self.agent_contexts.items():
            if reasoning_step.step_id in context.blocking_dependencies:
                relevant_agents.append(agent_id)
        
        # Remove duplicates and originating agent
        relevant_agents = list(set(relevant_agents))
        if reasoning_step.agent_id in relevant_agents:
            relevant_agents.remove(reasoning_step.agent_id)
        
        return relevant_agents
    
    async def _send_reasoning_context_to_agent(
        self,
        target_agent_id: str,
        reasoning_step: EnhancedReasoningStep,
        trace: ReasoningTraceGraph
    ):
        """Send reasoning context to specific agent"""
        if target_agent_id not in self.agent_contexts:
            return
        
        target_context = self.agent_contexts[target_agent_id]
        
        # Add to required context
        if reasoning_step.agent_id not in target_context.required_context_from_agents:
            target_context.required_context_from_agents[reasoning_step.agent_id] = []
        
        target_context.required_context_from_agents[reasoning_step.agent_id].append(
            str(reasoning_step.step_id)
        )
        
        # Update provided context tracking
        source_agent_id = reasoning_step.agent_id
        if source_agent_id in self.agent_contexts:
            source_context = self.agent_contexts[source_agent_id]
            if target_agent_id not in source_context.provided_context_to_agents:
                source_context.provided_context_to_agents[target_agent_id] = []
            
            source_context.provided_context_to_agents[target_agent_id].append(
                str(reasoning_step.step_id)
            )
    
    async def _update_agent_dependencies(
        self,
        reasoning_step: EnhancedReasoningStep,
        trace: ReasoningTraceGraph
    ):
        """Update agent dependency tracking"""
        # Check if this step resolves any blocking dependencies
        for agent_id, context in self.agent_contexts.items():
            if reasoning_step.step_id in context.blocking_dependencies:
                context.blocking_dependencies.remove(reasoning_step.step_id)
        
        # Check if this step creates new dependencies
        if reasoning_step.context_dependencies:
            reasoning_step.dependent_on_agents = [
                trace.reasoning_steps[dep_id].agent_id
                for dep_id in reasoning_step.context_dependencies
                if dep_id in trace.reasoning_steps
            ]
    
    async def synchronize_agent_reasoning(
        self,
        requesting_agent_id: str,
        trace_id: UUID
    ) -> Dict[str, Any]:
        """
        Synchronize reasoning context for requesting agent
        """
        logger.info("Starting agent reasoning synchronization",
                   agent_id=requesting_agent_id,
                   trace_id=trace_id)
        
        if trace_id not in self.reasoning_traces:
            logger.error("Reasoning trace not found for synchronization", trace_id=trace_id)
            return {"error": "Trace not found"}
        
        trace = self.reasoning_traces[trace_id]
        
        if requesting_agent_id not in self.agent_contexts:
            logger.error("Agent context not found", agent_id=requesting_agent_id)
            return {"error": "Agent context not found"}
        
        agent_context = self.agent_contexts[requesting_agent_id]
        sync_start_time = time.time()
        
        try:
            # Collect relevant reasoning steps
            relevant_steps = await self._collect_relevant_steps_for_agent(
                requesting_agent_id, trace
            )
            
            # Compress context for agent consumption
            compressed_context = await self._compress_reasoning_context_for_agent(
                relevant_steps, requesting_agent_id
            )
            
            # Detect and resolve conflicts
            conflict_resolutions = await self._resolve_agent_conflicts(
                requesting_agent_id, trace
            )
            
            # Generate coordination recommendations
            coordination_notes = await self._generate_coordination_recommendations(
                requesting_agent_id, trace, relevant_steps
            )
            
            # Update synchronization status
            agent_context.last_sync_time = datetime.now(timezone.utc)
            agent_context.sync_status = "synchronized"
            
            # Update metrics
            sync_time = time.time() - sync_start_time
            self._update_sync_metrics(sync_time)
            
            sync_result = {
                "agent_id": requesting_agent_id,
                "trace_id": str(trace_id),
                "sync_timestamp": datetime.now(timezone.utc).isoformat(),
                "relevant_steps_count": len(relevant_steps),
                "compressed_context": compressed_context,
                "conflict_resolutions": conflict_resolutions,
                "coordination_notes": coordination_notes,
                "sync_quality_score": await self._assess_sync_quality(
                    requesting_agent_id, relevant_steps
                ),
                "sync_time_seconds": sync_time
            }
            
            logger.info("Agent reasoning synchronization completed",
                       agent_id=requesting_agent_id,
                       steps_synchronized=len(relevant_steps),
                       conflicts_resolved=len(conflict_resolutions))
            
            return sync_result
            
        except Exception as e:
            logger.error("Agent reasoning synchronization failed",
                        agent_id=requesting_agent_id,
                        error=str(e))
            
            agent_context.sync_status = "out_of_sync"
            return {
                "error": f"Synchronization failed: {str(e)}",
                "agent_id": requesting_agent_id,
                "sync_timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _collect_relevant_steps_for_agent(
        self,
        agent_id: str,
        trace: ReasoningTraceGraph
    ) -> List[EnhancedReasoningStep]:
        """Collect reasoning steps relevant to specific agent"""
        relevant_steps = []
        agent_context = self.agent_contexts[agent_id]
        
        # Get steps since last sync
        for step_id, step in trace.reasoning_steps.items():
            if step.created_at > agent_context.last_sync_time:
                # Include if step is relevant to this agent
                if (step.agent_id != agent_id and  # Not from this agent
                    (step.step_type in [ReasoningStepType.DECISION_MAKING, ReasoningStepType.ERROR_CORRECTION] or
                     step.agent_id in agent_context.required_context_from_agents or
                     len(step.conflicts_detected) > 0)):
                    relevant_steps.append(step)
        
        # Sort by timestamp
        relevant_steps.sort(key=lambda x: x.created_at)
        
        return relevant_steps
    
    async def _compress_reasoning_context_for_agent(
        self,
        relevant_steps: List[EnhancedReasoningStep],
        agent_id: str
    ) -> Dict[str, Any]:
        """Compress reasoning context for agent consumption"""
        if not relevant_steps:
            return {"message": "No new reasoning context to synchronize"}
        
        # Convert steps to context segments
        context_segments = []
        for step in relevant_steps:
            segment = ContextSegment(
                content=json.dumps({
                    "step_type": step.step_type.value,
                    "description": step.step_description,
                    "rationale": step.step_rationale,
                    "decisions": step.explicit_decisions,
                    "assumptions": step.assumptions_made,
                    "confidence": step.confidence_score,
                    "conflicts": [c.value for c in step.conflicts_detected]
                }),
                context_type=ContextType.REASONING_TRACE,
                importance=ContextImportance.HIGH,
                agent_id=step.agent_id,
                timestamp=step.created_at,
                token_count=len(step.step_description.split()) + len(step.step_rationale.split())
            )
            context_segments.append(segment)
        
        # Compress using context engine
        try:
            compressed = await self.context_engine.compress_context(
                context_segments,
                target_token_limit=1500  # Reasonable limit for agent consumption
            )
            
            return {
                "compressed_steps": len(context_segments),
                "compression_ratio": compressed.compression_ratio,
                "key_decisions": [step.explicit_decisions for step in relevant_steps if step.explicit_decisions],
                "critical_conflicts": [step.conflicts_detected for step in relevant_steps if step.conflicts_detected],
                "summary": compressed.summary_context
            }
            
        except Exception as e:
            logger.error("Failed to compress reasoning context", error=str(e))
            return {
                "steps_count": len(relevant_steps),
                "summary": f"Synchronizing {len(relevant_steps)} reasoning steps from other agents"
            }
    
    async def _resolve_agent_conflicts(
        self,
        agent_id: str,
        trace: ReasoningTraceGraph
    ) -> List[Dict[str, Any]]:
        """Resolve conflicts affecting specific agent"""
        resolutions = []
        
        # Find unresolved conflicts involving this agent
        for conflict in trace.detected_conflicts:
            if not conflict.get("resolved", False):
                involved_steps = [
                    conflict.get("step1_id"),
                    conflict.get("step2_id")
                ]
                
                # Check if agent is involved
                agent_involved = False
                for step_id in involved_steps:
                    if step_id and step_id in trace.reasoning_steps:
                        step = trace.reasoning_steps[step_id]
                        if step.agent_id == agent_id:
                            agent_involved = True
                            break
                
                if agent_involved:
                    resolution = await self._attempt_conflict_resolution(
                        conflict, trace
                    )
                    resolutions.append(resolution)
                    
                    # Update metrics
                    if resolution.get("resolved", False):
                        self.sharing_metrics["conflicts_resolved"] += 1
        
        return resolutions
    
    async def _attempt_conflict_resolution(
        self,
        conflict: Dict[str, Any],
        trace: ReasoningTraceGraph
    ) -> Dict[str, Any]:
        """Attempt to resolve specific reasoning conflict"""
        conflict_type = conflict.get("type")
        
        resolution = {
            "conflict_id": str(uuid4()),
            "conflict_type": conflict_type,
            "resolution_strategy": "automated_analysis",
            "resolved": False,
            "resolution_confidence": 0.0
        }
        
        try:
            if conflict_type == ReasoningConflictType.CONTRADICTORY_CONCLUSIONS:
                resolution = await self._resolve_contradictory_conclusions(conflict, trace)
            elif conflict_type == ReasoningConflictType.INCOMPATIBLE_ASSUMPTIONS:
                resolution = await self._resolve_incompatible_assumptions(conflict, trace)
            elif conflict_type == ReasoningConflictType.TIMING_CONFLICTS:
                resolution = await self._resolve_timing_conflicts(conflict, trace)
            else:
                resolution["resolution_strategy"] = "escalation_required"
                resolution["resolution_notes"] = "Conflict type requires human intervention"
            
        except Exception as e:
            logger.error("Conflict resolution failed", error=str(e))
            resolution["error"] = str(e)
        
        return resolution
    
    async def _resolve_contradictory_conclusions(
        self,
        conflict: Dict[str, Any],
        trace: ReasoningTraceGraph
    ) -> Dict[str, Any]:
        """Resolve contradictory conclusions between agents"""
        step1_id = conflict.get("step1_id")
        step2_id = conflict.get("step2_id")
        
        if not step1_id or not step2_id:
            return {"resolved": False, "error": "Missing step IDs"}
        
        step1 = trace.reasoning_steps.get(step1_id)
        step2 = trace.reasoning_steps.get(step2_id)
        
        if not step1 or not step2:
            return {"resolved": False, "error": "Steps not found"}
        
        # Analyze confidence scores
        if step1.confidence_score > step2.confidence_score + 0.2:
            # Step1 has significantly higher confidence
            resolution = {
                "resolved": True,
                "resolution_strategy": "confidence_based",
                "chosen_conclusion": step1.output_result.get("decision"),
                "rationale": f"Agent {step1.agent_id} decision chosen due to higher confidence ({step1.confidence_score:.2f} vs {step2.confidence_score:.2f})",
                "resolution_confidence": 0.8
            }
        elif step2.confidence_score > step1.confidence_score + 0.2:
            # Step2 has significantly higher confidence
            resolution = {
                "resolved": True,
                "resolution_strategy": "confidence_based",
                "chosen_conclusion": step2.output_result.get("decision"),
                "rationale": f"Agent {step2.agent_id} decision chosen due to higher confidence ({step2.confidence_score:.2f} vs {step1.confidence_score:.2f})",
                "resolution_confidence": 0.8
            }
        else:
            # Similar confidence - require escalation or compromise
            resolution = {
                "resolved": False,
                "resolution_strategy": "escalation_required",
                "rationale": "Similar confidence levels require human review or additional evidence",
                "suggested_actions": [
                    "Gather additional evidence",
                    "Seek third-party validation",
                    "Implement compromise solution"
                ]
            }
        
        return resolution
    
    async def _resolve_incompatible_assumptions(
        self,
        conflict: Dict[str, Any],
        trace: ReasoningTraceGraph
    ) -> Dict[str, Any]:
        """Resolve incompatible assumptions between agents"""
        # For assumption conflicts, identify which assumptions are more valid
        conflicting_assumptions = conflict.get("conflicting_assumptions", {})
        
        resolution = {
            "resolved": True,
            "resolution_strategy": "assumption_validation",
            "rationale": "Identified need for explicit assumption validation",
            "recommended_actions": [
                "Validate assumptions against known facts",
                "Seek consensus on shared assumptions",
                "Document assumption differences"
            ],
            "resolution_confidence": 0.6
        }
        
        return resolution
    
    async def _resolve_timing_conflicts(
        self,
        conflict: Dict[str, Any],
        trace: ReasoningTraceGraph
    ) -> Dict[str, Any]:
        """Resolve timing conflicts between concurrent reasoning"""
        resolution = {
            "resolved": True,
            "resolution_strategy": "coordination_improvement",
            "rationale": "Concurrent reasoning steps indicate need for better coordination",
            "recommended_actions": [
                "Implement agent coordination protocols",
                "Add synchronization checkpoints",
                "Establish reasoning step priorities"
            ],
            "resolution_confidence": 0.7
        }
        
        return resolution
    
    async def _generate_coordination_recommendations(
        self,
        agent_id: str,
        trace: ReasoningTraceGraph,
        relevant_steps: List[EnhancedReasoningStep]
    ) -> List[str]:
        """Generate recommendations for improved agent coordination"""
        recommendations = []
        
        # Analyze coordination patterns
        agent_context = self.agent_contexts[agent_id]
        
        # Check for dependency issues
        if len(agent_context.blocking_dependencies) > 0:
            recommendations.append("Consider parallel processing for independent tasks")
        
        # Check for conflict patterns
        conflicted_steps = [step for step in relevant_steps if step.conflicts_detected]
        if len(conflicted_steps) > 2:
            recommendations.append("Implement proactive consensus building")
        
        # Check for communication efficiency
        if len(relevant_steps) > 10:
            recommendations.append("Use selective reasoning step sharing to reduce information overload")
        
        # Check for quality patterns
        if agent_context.reasoning_quality_score < 0.7:
            recommendations.append("Increase validation and confidence checking")
        
        # General coordination recommendations
        recommendations.extend([
            "Share decision criteria explicitly with other agents",
            "Validate assumptions before proceeding with critical decisions",
            "Use incremental synchronization for long-running tasks"
        ])
        
        return recommendations
    
    async def _assess_sync_quality(
        self,
        agent_id: str,
        relevant_steps: List[EnhancedReasoningStep]
    ) -> float:
        """Assess quality of synchronization for agent"""
        if not relevant_steps:
            return 1.0  # Perfect sync if no steps to sync
        
        quality_factors = []
        
        # Information completeness
        complete_steps = [step for step in relevant_steps if step.step_rationale and step.assumptions_made]
        completeness = len(complete_steps) / len(relevant_steps)
        quality_factors.append(completeness)
        
        # Conflict resolution effectiveness
        conflicted_steps = [step for step in relevant_steps if step.conflicts_detected]
        if conflicted_steps:
            resolved_conflicts = [step for step in conflicted_steps if "resolved" in str(step.conflicts_detected)]
            conflict_resolution_rate = len(resolved_conflicts) / len(conflicted_steps)
            quality_factors.append(conflict_resolution_rate)
        else:
            quality_factors.append(1.0)  # No conflicts = perfect resolution
        
        # Confidence levels
        avg_confidence = sum(step.confidence_score for step in relevant_steps) / len(relevant_steps)
        quality_factors.append(avg_confidence)
        
        # Overall quality score
        return sum(quality_factors) / len(quality_factors)
    
    def _update_sync_metrics(self, sync_time: float):
        """Update synchronization performance metrics"""
        current_avg = self.sharing_metrics["average_sync_time"]
        total_syncs = self.sharing_metrics.get("total_syncs", 0) + 1
        
        self.sharing_metrics["average_sync_time"] = (
            (current_avg * (total_syncs - 1) + sync_time) / total_syncs
        )
        self.sharing_metrics["total_syncs"] = total_syncs
        
        # Update coordination effectiveness
        if self.sharing_metrics["conflicts_detected"] > 0:
            resolution_rate = (
                self.sharing_metrics["conflicts_resolved"] / 
                self.sharing_metrics["conflicts_detected"]
            )
            self.sharing_metrics["coordination_effectiveness"] = (
                self.sharing_metrics["coordination_effectiveness"] * 0.8 + 
                resolution_rate * 0.2
            )
    
    async def get_reasoning_trace_summary(self, trace_id: UUID) -> Dict[str, Any]:
        """Get comprehensive summary of reasoning trace"""
        if trace_id not in self.reasoning_traces:
            return {"error": "Trace not found"}
        
        trace = self.reasoning_traces[trace_id]
        
        # Calculate summary statistics
        summary = {
            "trace_id": str(trace_id),
            "total_steps": trace.total_steps,
            "participating_agents": trace.participating_agents,
            "agent_contributions": {
                agent_id: len(steps) for agent_id, steps in trace.agent_contributions.items()
            },
            "conflicts_detected": len(trace.detected_conflicts),
            "conflicts_resolved": len(trace.resolved_conflicts),
            "trace_coherence_score": trace.trace_coherence_score,
            "decision_consistency_score": trace.decision_consistency_score,
            "agent_coordination_score": trace.agent_coordination_score,
            "last_sync": trace.last_sync_timestamp.isoformat(),
            "step_type_distribution": await self._calculate_step_type_distribution(trace),
            "quality_metrics": await self._calculate_trace_quality_metrics(trace)
        }
        
        return summary
    
    async def _calculate_step_type_distribution(self, trace: ReasoningTraceGraph) -> Dict[str, int]:
        """Calculate distribution of reasoning step types"""
        distribution = {}
        
        for step in trace.reasoning_steps.values():
            step_type = step.step_type.value
            distribution[step_type] = distribution.get(step_type, 0) + 1
        
        return distribution
    
    async def _calculate_trace_quality_metrics(self, trace: ReasoningTraceGraph) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for trace"""
        if not trace.reasoning_steps:
            return {}
        
        steps = list(trace.reasoning_steps.values())
        
        metrics = {
            "average_confidence": sum(step.confidence_score for step in steps) / len(steps),
            "steps_with_validation": len([s for s in steps if s.validation_performed]) / len(steps),
            "steps_with_conflicts": len([s for s in steps if s.conflicts_detected]) / len(steps),
            "assumption_density": sum(len(s.assumptions_made) for s in steps) / len(steps),
            "implicit_decision_capture": len([s for s in steps if s.implicit_decisions]) / len(steps)
        }
        
        return metrics


# Factory Functions
def create_reasoning_trace_sharing_engine(session_id: UUID) -> ReasoningTraceSharingEngine:
    """Create reasoning trace sharing engine for session"""
    return ReasoningTraceSharingEngine(session_id)


def create_enhanced_reasoning_step(
    agent_id: str,
    agent_type: AgentType,
    step_type: ReasoningStepType,
    description: str,
    input_context: Dict[str, Any],
    output_result: Dict[str, Any]
) -> EnhancedReasoningStep:
    """Create enhanced reasoning step with comprehensive context"""
    return EnhancedReasoningStep(
        agent_id=agent_id,
        agent_type=agent_type,
        step_type=step_type,
        step_description=description,
        input_context=input_context,
        output_result=output_result
    )
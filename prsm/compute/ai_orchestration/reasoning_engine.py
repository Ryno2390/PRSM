#!/usr/bin/env python3
"""
Advanced AI Reasoning Engine
============================

Sophisticated reasoning system supporting multiple reasoning paradigms,
chain-of-thought processing, meta-reasoning, and adaptive reasoning strategies.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
import uuid
from pathlib import Path
import math

from .model_manager import ModelManager, ModelInstance, ModelCapability
from .task_distributor import Task, TaskResult, TaskDistributor

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning approaches"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    PROBABILISTIC = "probabilistic"
    LOGICAL = "logical"
    CREATIVE = "creative"
    METACOGNITIVE = "metacognitive"


class ReasoningStrategy(Enum):
    """Reasoning execution strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    TREE_SEARCH = "tree_search"
    BEAM_SEARCH = "beam_search"
    MONTE_CARLO = "monte_carlo"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


class StepStatus(Enum):
    """Reasoning step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ReasoningStep:
    """Individual reasoning step"""
    step_id: str
    name: str
    reasoning_type: ReasoningType
    description: str = ""
    
    # Step configuration
    input_data: Dict[str, Any] = field(default_factory=dict)
    prompt_template: str = ""
    model_requirements: Set[ModelCapability] = field(default_factory=set)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    
    # Execution configuration
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout_seconds: int = 60
    
    # Results
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    execution_time_ms: float = 0.0
    model_used: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    
    # Quality metrics
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    novelty_score: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "reasoning_type": self.reasoning_type.value,
            "description": self.description,
            "input_data": self.input_data,
            "prompt_template": self.prompt_template,
            "model_requirements": [cap.value for cap in self.model_requirements],
            "depends_on": self.depends_on,
            "provides": self.provides,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value,
            "result": self.result,
            "confidence_score": self.confidence_score,
            "execution_time_ms": self.execution_time_ms,
            "model_used": self.model_used,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "coherence_score": self.coherence_score,
            "relevance_score": self.relevance_score,
            "novelty_score": self.novelty_score,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class ReasoningChain:
    """Chain of reasoning steps"""
    chain_id: str
    name: str
    description: str = ""
    
    # Chain configuration
    steps: List[ReasoningStep] = field(default_factory=list)
    strategy: ReasoningStrategy = ReasoningStrategy.SEQUENTIAL
    
    # Global parameters
    global_context: Dict[str, Any] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    current_step_index: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    
    # Quality assessment
    overall_coherence: float = 0.0
    chain_confidence: float = 0.0
    logical_consistency: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def add_step(self, step: ReasoningStep):
        """Add reasoning step to chain"""
        self.steps.append(step)
    
    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        """Get step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_dependencies_graph(self) -> Dict[str, List[str]]:
        """Get dependency graph for steps"""
        graph = {}
        for step in self.steps:
            graph[step.step_id] = step.depends_on.copy()
        return graph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chain_id": self.chain_id,
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "strategy": self.strategy.value,
            "global_context": self.global_context,
            "shared_memory": self.shared_memory,
            "current_step_index": self.current_step_index,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "overall_coherence": self.overall_coherence,
            "chain_confidence": self.chain_confidence,
            "logical_consistency": self.logical_consistency,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class ReasoningResult:
    """Result of reasoning chain execution"""
    chain_id: str
    success: bool
    final_result: Dict[str, Any]
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    overall_confidence: float = 0.0
    coherence_score: float = 0.0
    consistency_score: float = 0.0
    completeness_score: float = 0.0
    
    # Performance metrics
    total_execution_time_ms: float = 0.0
    steps_executed: int = 0
    models_used: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chain_id": self.chain_id,
            "success": self.success,
            "final_result": self.final_result,
            "step_results": self.step_results,
            "overall_confidence": self.overall_confidence,
            "coherence_score": self.coherence_score,
            "consistency_score": self.consistency_score,
            "completeness_score": self.completeness_score,
            "total_execution_time_ms": self.total_execution_time_ms,
            "steps_executed": self.steps_executed,
            "models_used": self.models_used,
            "total_cost": self.total_cost,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat()
        }


class ReasoningTemplateLibrary:
    """Library of reasoning templates and patterns"""
    
    def __init__(self):
        self.templates = {
            ReasoningType.DEDUCTIVE: self._deductive_templates(),
            ReasoningType.INDUCTIVE: self._inductive_templates(),
            ReasoningType.ABDUCTIVE: self._abductive_templates(),
            ReasoningType.ANALOGICAL: self._analogical_templates(),
            ReasoningType.CAUSAL: self._causal_templates(),
            ReasoningType.COUNTERFACTUAL: self._counterfactual_templates(),
            ReasoningType.PROBABILISTIC: self._probabilistic_templates(),
            ReasoningType.CREATIVE: self._creative_templates(),
            ReasoningType.METACOGNITIVE: self._metacognitive_templates()
        }
    
    def get_template(self, reasoning_type: ReasoningType, template_name: str) -> Optional[str]:
        """Get specific reasoning template"""
        return self.templates.get(reasoning_type, {}).get(template_name)
    
    def list_templates(self, reasoning_type: ReasoningType) -> List[str]:
        """List available templates for reasoning type"""
        return list(self.templates.get(reasoning_type, {}).keys())
    
    def _deductive_templates(self) -> Dict[str, str]:
        """Deductive reasoning templates"""
        return {
            "modus_ponens": """
Given the following premise and rule:
Premise: {premise}
Rule: {rule}

Apply deductive reasoning using modus ponens to derive the conclusion.
Clearly state your logical steps and the final conclusion.
            """,
            
            "syllogism": """
Given the major premise and minor premise:
Major Premise: {major_premise}
Minor Premise: {minor_premise}

Use syllogistic reasoning to derive the logical conclusion.
Show your work step by step.
            """,
            
            "proof_by_contradiction": """
To prove: {statement_to_prove}

Use proof by contradiction:
1. Assume the negation of the statement
2. Derive a logical contradiction
3. Conclude that the original statement must be true

Work through this systematically.
            """
        }
    
    def _inductive_templates(self) -> Dict[str, str]:
        """Inductive reasoning templates"""
        return {
            "pattern_recognition": """
Examine the following examples:
{examples}

Use inductive reasoning to:
1. Identify patterns in the data
2. Formulate a general rule or principle
3. Assess the strength of your induction
4. Make predictions based on the pattern
            """,
            
            "statistical_inference": """
Given the sample data:
{sample_data}

Use statistical induction to:
1. Analyze the sample characteristics
2. Make inferences about the population
3. Estimate confidence levels
4. Draw general conclusions
            """,
            
            "generalization": """
Based on these specific observations:
{observations}

Use inductive generalization to:
1. Identify common features
2. Form a general hypothesis
3. Consider potential exceptions
4. Evaluate the reliability of your generalization
            """
        }
    
    def _abductive_templates(self) -> Dict[str, str]:
        """Abductive reasoning templates"""
        return {
            "best_explanation": """
Observation: {observation}
Possible explanations: {possible_explanations}

Use abductive reasoning to:
1. Evaluate each explanation's plausibility
2. Consider explanatory power and simplicity
3. Assess available evidence
4. Select the best explanation and justify your choice
            """,
            
            "hypothesis_formation": """
Given the puzzling facts:
{facts}

Use abductive reasoning to:
1. Generate multiple hypotheses
2. Evaluate each hypothesis for explanatory adequacy
3. Consider parsimony and plausibility
4. Propose the most likely explanation
            """,
            
            "diagnostic_reasoning": """
Symptoms/Evidence: {symptoms}
Context: {context}

Use diagnostic abduction to:
1. Generate possible diagnoses/causes
2. Evaluate likelihood of each
3. Consider differential diagnosis
4. Recommend further investigation if needed
            """
        }
    
    def _analogical_templates(self) -> Dict[str, str]:
        """Analogical reasoning templates"""
        return {
            "structural_analogy": """
Source domain: {source_domain}
Target domain: {target_domain}

Use structural analogical reasoning to:
1. Map corresponding elements between domains
2. Transfer relationships and properties
3. Generate predictions or solutions for the target
4. Evaluate the strength and limitations of the analogy
            """,
            
            "case_based_reasoning": """
Current situation: {current_situation}
Similar past cases: {past_cases}

Use case-based analogical reasoning to:
1. Identify the most relevant similar case
2. Map similarities and differences
3. Adapt the solution from the similar case
4. Justify your adaptation strategy
            """,
            
            "metaphorical_reasoning": """
Concept to understand: {target_concept}
Metaphorical source: {source_metaphor}

Use metaphorical reasoning to:
1. Explore the metaphorical mapping
2. Identify illuminating correspondences
3. Generate insights about the target concept
4. Consider the limits of the metaphor
            """
        }
    
    def _causal_templates(self) -> Dict[str, str]:
        """Causal reasoning templates"""
        return {
            "causal_chain": """
Initial condition: {initial_condition}
Final outcome: {final_outcome}

Use causal reasoning to:
1. Identify intermediate causal links
2. Trace the causal chain from cause to effect
3. Consider alternative causal pathways
4. Assess the strength of causal relationships
            """,
            
            "root_cause_analysis": """
Problem/Effect: {problem}
Observable symptoms: {symptoms}

Use root cause analysis to:
1. Work backward from the problem
2. Identify potential contributing factors
3. Determine the fundamental root cause(s)
4. Distinguish between symptoms and causes
            """,
            
            "mechanism_explanation": """
Phenomenon: {phenomenon}
Context: {context}

Explain the causal mechanism by:
1. Identifying the key components involved
2. Describing how they interact causally
3. Tracing the step-by-step process
4. Explaining why this mechanism produces the phenomenon
            """
        }
    
    def _counterfactual_templates(self) -> Dict[str, str]:
        """Counterfactual reasoning templates"""
        return {
            "what_if_analysis": """
Actual situation: {actual_situation}
Counterfactual condition: {counterfactual_condition}

Use counterfactual reasoning to:
1. Imagine the alternative scenario
2. Trace through likely consequences
3. Compare with the actual outcome
4. Draw insights about causal relationships
            """,
            
            "alternative_histories": """
Historical event: {historical_event}
Key decision point: {decision_point}

Explore counterfactual history by:
1. Identifying the critical juncture
2. Considering alternative decisions/events
3. Tracing plausible alternative outcomes
4. Assessing how different things might have been
            """,
            
            "scenario_planning": """
Current situation: {current_situation}
Key uncertainties: {uncertainties}

Use counterfactual scenario planning to:
1. Develop multiple future scenarios
2. Consider how different factors might play out
3. Identify robust strategies across scenarios
4. Prepare for various contingencies
            """
        }
    
    def _probabilistic_templates(self) -> Dict[str, str]:
        """Probabilistic reasoning templates"""
        return {
            "bayesian_inference": """
Prior beliefs: {prior_beliefs}
New evidence: {new_evidence}

Use Bayesian reasoning to:
1. Start with your prior probability
2. Consider the likelihood of the evidence
3. Calculate the posterior probability
4. Update your beliefs based on the evidence
            """,
            
            "risk_assessment": """
Scenario: {scenario}
Potential risks: {potential_risks}

Use probabilistic risk assessment to:
1. Estimate the probability of each risk
2. Assess the potential impact/severity
3. Calculate risk scores (probability × impact)
4. Prioritize risks and recommend mitigation
            """,
            
            "decision_under_uncertainty": """
Decision options: {decision_options}
Uncertain outcomes: {uncertain_outcomes}

Use probabilistic decision analysis to:
1. Assign probabilities to uncertain outcomes
2. Calculate expected values for each option
3. Consider risk tolerance and utility
4. Recommend the optimal decision strategy
            """
        }
    
    def _creative_templates(self) -> Dict[str, str]:
        """Creative reasoning templates"""
        return {
            "brainstorming": """
Challenge: {challenge}
Constraints: {constraints}

Use creative brainstorming to:
1. Generate many diverse ideas without judgment
2. Build on and combine ideas
3. Think outside conventional boundaries
4. Select and develop the most promising concepts
            """,
            
            "lateral_thinking": """
Problem: {problem}
Current approach: {current_approach}

Use lateral thinking to:
1. Challenge assumptions about the problem
2. Reframe the problem from different angles
3. Use random stimuli to generate new perspectives
4. Develop unconventional solutions
            """,
            
            "design_thinking": """
User needs: {user_needs}
Design constraints: {design_constraints}

Use design thinking to:
1. Empathize with user perspectives
2. Define the core problem to solve
3. Ideate multiple solution concepts
4. Prototype and iterate on promising ideas
            """
        }
    
    def _metacognitive_templates(self) -> Dict[str, str]:
        """Metacognitive reasoning templates"""
        return {
            "strategy_selection": """
Problem type: {problem_type}
Available strategies: {available_strategies}

Use metacognitive reasoning to:
1. Analyze the problem characteristics
2. Evaluate strategy appropriateness
3. Select the most suitable approach
4. Monitor and adjust strategy as needed
            """,
            
            "confidence_assessment": """
Reasoning process: {reasoning_process}
Conclusion reached: {conclusion}

Use metacognitive assessment to:
1. Evaluate the strength of your reasoning
2. Identify potential weaknesses or gaps
3. Assess your confidence level
4. Consider what additional information would help
            """,
            
            "error_detection": """
Reasoning chain: {reasoning_chain}
Potential issues: {potential_issues}

Use metacognitive error detection to:
1. Review each step of your reasoning
2. Look for logical fallacies or biases
3. Check for consistency and coherence
4. Identify and correct any errors found
            """
        }


class ReasoningQualityAssessor:
    """Quality assessment for reasoning chains and results"""
    
    def __init__(self):
        self.assessment_criteria = {
            "coherence": self._assess_coherence,
            "consistency": self._assess_consistency,
            "completeness": self._assess_completeness,
            "relevance": self._assess_relevance,
            "novelty": self._assess_novelty,
            "confidence": self._assess_confidence
        }
    
    async def assess_reasoning_chain(self, chain: ReasoningChain) -> Dict[str, float]:
        """Assess quality of reasoning chain"""
        
        quality_scores = {}
        
        for criterion, assessor in self.assessment_criteria.items():
            try:
                score = await assessor(chain)
                quality_scores[criterion] = score
            except Exception as e:
                logger.error(f"Quality assessment error for {criterion}: {e}")
                quality_scores[criterion] = 0.0
        
        return quality_scores
    
    async def assess_step_result(self, step: ReasoningStep) -> Dict[str, float]:
        """Assess quality of individual reasoning step"""
        
        quality_scores = {}
        
        # Step-specific quality assessment
        if step.result:
            quality_scores["coherence"] = await self._assess_step_coherence(step)
            quality_scores["relevance"] = await self._assess_step_relevance(step)
            quality_scores["confidence"] = step.confidence_score
        else:
            quality_scores = {criterion: 0.0 for criterion in self.assessment_criteria.keys()}
        
        return quality_scores
    
    async def _assess_coherence(self, chain: ReasoningChain) -> float:
        """Assess logical coherence of reasoning chain"""
        
        if not chain.steps:
            return 0.0
        
        coherence_score = 0.0
        step_count = 0
        
        # Assess coherence between consecutive steps
        for i in range(len(chain.steps) - 1):
            current_step = chain.steps[i]
            next_step = chain.steps[i + 1]
            
            if current_step.status == StepStatus.COMPLETED and next_step.status == StepStatus.COMPLETED:
                # Simple coherence check (would be enhanced with actual NLP analysis)
                step_coherence = self._calculate_step_coherence(current_step, next_step)
                coherence_score += step_coherence
                step_count += 1
        
        return coherence_score / max(1, step_count)
    
    async def _assess_consistency(self, chain: ReasoningChain) -> float:
        """Assess logical consistency of reasoning chain"""
        
        completed_steps = [s for s in chain.steps if s.status == StepStatus.COMPLETED]
        
        if len(completed_steps) < 2:
            return 100.0  # Single step is consistent by default
        
        # Check for contradictions between steps
        consistency_score = 100.0
        contradictions = 0
        
        for i in range(len(completed_steps)):
            for j in range(i + 1, len(completed_steps)):
                if self._detect_contradiction(completed_steps[i], completed_steps[j]):
                    contradictions += 1
        
        # Reduce score based on contradictions
        max_comparisons = (len(completed_steps) * (len(completed_steps) - 1)) // 2
        if max_comparisons > 0:
            consistency_score -= (contradictions / max_comparisons) * 100
        
        return max(0.0, consistency_score)
    
    async def _assess_completeness(self, chain: ReasoningChain) -> float:
        """Assess completeness of reasoning chain"""
        
        if not chain.steps:
            return 0.0
        
        completed_steps = len([s for s in chain.steps if s.status == StepStatus.COMPLETED])
        total_steps = len(chain.steps)
        
        return (completed_steps / total_steps) * 100
    
    async def _assess_relevance(self, chain: ReasoningChain) -> float:
        """Assess relevance of reasoning steps to overall goal"""
        
        if not chain.steps:
            return 0.0
        
        # Average relevance of individual steps
        relevance_scores = [s.relevance_score for s in chain.steps if s.status == StepStatus.COMPLETED]
        
        if not relevance_scores:
            return 0.0
        
        return sum(relevance_scores) / len(relevance_scores)
    
    async def _assess_novelty(self, chain: ReasoningChain) -> float:
        """Assess novelty/creativity of reasoning approach"""
        
        if not chain.steps:
            return 0.0
        
        # Average novelty of individual steps
        novelty_scores = [s.novelty_score for s in chain.steps if s.status == StepStatus.COMPLETED]
        
        if not novelty_scores:
            return 0.0
        
        return sum(novelty_scores) / len(novelty_scores)
    
    async def _assess_confidence(self, chain: ReasoningChain) -> float:
        """Assess overall confidence in reasoning chain"""
        
        if not chain.steps:
            return 0.0
        
        # Weighted average of step confidence scores
        confidence_scores = []
        weights = []
        
        for step in chain.steps:
            if step.status == StepStatus.COMPLETED:
                confidence_scores.append(step.confidence_score)
                # Weight by step importance (could be enhanced)
                weights.append(1.0)
        
        if not confidence_scores:
            return 0.0
        
        weighted_sum = sum(c * w for c, w in zip(confidence_scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight
    
    async def _assess_step_coherence(self, step: ReasoningStep) -> float:
        """Assess coherence of individual step"""
        # Placeholder for actual coherence assessment
        return step.coherence_score if step.coherence_score > 0 else 75.0
    
    async def _assess_step_relevance(self, step: ReasoningStep) -> float:
        """Assess relevance of individual step"""
        # Placeholder for actual relevance assessment
        return step.relevance_score if step.relevance_score > 0 else 80.0
    
    def _calculate_step_coherence(self, step1: ReasoningStep, step2: ReasoningStep) -> float:
        """Calculate coherence between two steps"""
        # Simplified coherence calculation (would use actual NLP analysis)
        
        # Check if step2 uses outputs from step1
        if any(output in step2.input_data.keys() for output in step1.provides):
            return 90.0
        
        # Check reasoning type compatibility
        compatible_pairs = {
            (ReasoningType.DEDUCTIVE, ReasoningType.LOGICAL),
            (ReasoningType.INDUCTIVE, ReasoningType.PROBABILISTIC),
            (ReasoningType.ABDUCTIVE, ReasoningType.CAUSAL),
            (ReasoningType.ANALOGICAL, ReasoningType.CREATIVE)
        }
        
        if (step1.reasoning_type, step2.reasoning_type) in compatible_pairs:
            return 80.0
        
        return 60.0  # Default coherence
    
    def _detect_contradiction(self, step1: ReasoningStep, step2: ReasoningStep) -> bool:
        """Detect logical contradictions between steps"""
        # Simplified contradiction detection (would use actual logical analysis)
        
        if not step1.result or not step2.result:
            return False
        
        # Check for explicit contradictions in results
        result1 = step1.result.get("conclusion", "")
        result2 = step2.result.get("conclusion", "")
        
        # Simple keyword-based contradiction detection
        contradiction_keywords = [
            ("true", "false"),
            ("yes", "no"),
            ("possible", "impossible"),
            ("likely", "unlikely")
        ]
        
        for pos, neg in contradiction_keywords:
            if pos in result1.lower() and neg in result2.lower():
                return True
            if neg in result1.lower() and pos in result2.lower():
                return True
        
        return False


class ReasoningEngine:
    """Main reasoning engine orchestrator"""
    
    def __init__(self, model_manager: ModelManager, task_distributor: TaskDistributor):
        self.model_manager = model_manager
        self.task_distributor = task_distributor
        
        # Core components
        self.template_library = ReasoningTemplateLibrary()
        self.quality_assessor = ReasoningQualityAssessor()
        
        # Chain registry
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        
        # Execution tracking
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            "total_chains_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time_ms": 0.0,
            "avg_confidence_score": 0.0,
            "reasoning_types_used": {}
        }
        
        logger.info("Reasoning Engine initialized")
    
    def create_reasoning_chain(self, name: str, description: str = "", 
                             strategy: ReasoningStrategy = ReasoningStrategy.SEQUENTIAL) -> ReasoningChain:
        """Create a new reasoning chain"""
        
        chain_id = f"chain_{uuid.uuid4().hex[:8]}"
        
        chain = ReasoningChain(
            chain_id=chain_id,
            name=name,
            description=description,
            strategy=strategy
        )
        
        self.reasoning_chains[chain_id] = chain
        
        logger.info(f"Created reasoning chain: {name}")
        
        return chain
    
    def add_reasoning_step(self, chain_id: str, step_name: str, reasoning_type: ReasoningType,
                          prompt_template: Optional[str] = None, 
                          model_requirements: Optional[Set[ModelCapability]] = None,
                          depends_on: Optional[List[str]] = None) -> Optional[ReasoningStep]:
        """Add a reasoning step to a chain"""
        
        if chain_id not in self.reasoning_chains:
            logger.error(f"Reasoning chain not found: {chain_id}")
            return None
        
        chain = self.reasoning_chains[chain_id]
        
        step_id = f"step_{uuid.uuid4().hex[:8]}"
        
        # Get template if not provided
        if not prompt_template:
            available_templates = self.template_library.list_templates(reasoning_type)
            if available_templates:
                template_name = available_templates[0]  # Use first available template
                prompt_template = self.template_library.get_template(reasoning_type, template_name)
        
        step = ReasoningStep(
            step_id=step_id,
            name=step_name,
            reasoning_type=reasoning_type,
            prompt_template=prompt_template or "",
            model_requirements=model_requirements or set(),
            depends_on=depends_on or []
        )
        
        chain.add_step(step)
        
        logger.info(f"Added reasoning step: {step_name} to chain {chain.name}")
        
        return step
    
    async def execute_reasoning_chain(self, chain_id: str, 
                                    input_context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """Execute a reasoning chain"""
        
        if chain_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain not found: {chain_id}")
        
        chain = self.reasoning_chains[chain_id]
        
        # Set input context
        if input_context:
            chain.global_context.update(input_context)
        
        # Start execution
        chain.started_at = datetime.now(timezone.utc)
        start_time = chain.started_at
        
        try:
            # Execute based on strategy
            if chain.strategy == ReasoningStrategy.SEQUENTIAL:
                result = await self._execute_sequential(chain)
            elif chain.strategy == ReasoningStrategy.PARALLEL:
                result = await self._execute_parallel(chain)
            elif chain.strategy == ReasoningStrategy.TREE_SEARCH:
                result = await self._execute_tree_search(chain)
            elif chain.strategy == ReasoningStrategy.ENSEMBLE:
                result = await self._execute_ensemble(chain)
            else:
                result = await self._execute_sequential(chain)  # Default fallback
            
            # Calculate execution time
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result.total_execution_time_ms = execution_time
            
            # Assess overall quality
            quality_scores = await self.quality_assessor.assess_reasoning_chain(chain)
            result.coherence_score = quality_scores.get("coherence", 0.0)
            result.consistency_score = quality_scores.get("consistency", 0.0)
            result.completeness_score = quality_scores.get("completeness", 0.0)
            result.overall_confidence = quality_scores.get("confidence", 0.0)
            
            # Update chain quality metrics
            chain.overall_coherence = result.coherence_score
            chain.chain_confidence = result.overall_confidence
            chain.logical_consistency = result.consistency_score
            
            # Update statistics
            self.stats["total_chains_executed"] += 1
            if result.success:
                self.stats["successful_executions"] += 1
            else:
                self.stats["failed_executions"] += 1
            
            self._update_avg_metrics(execution_time, result.overall_confidence)
            
            # Track reasoning types used
            for step in chain.steps:
                if step.status == StepStatus.COMPLETED:
                    reasoning_type = step.reasoning_type.value
                    self.stats["reasoning_types_used"][reasoning_type] = \
                        self.stats["reasoning_types_used"].get(reasoning_type, 0) + 1
            
            chain.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Reasoning chain executed: {chain.name} (Success: {result.success})")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = ReasoningResult(
                chain_id=chain_id,
                success=False,
                final_result={},
                total_execution_time_ms=execution_time,
                errors=[str(e)]
            )
            
            self.stats["total_chains_executed"] += 1
            self.stats["failed_executions"] += 1
            
            logger.error(f"Reasoning chain execution failed: {chain.name} - {e}")
            
            return result
    
    async def _execute_sequential(self, chain: ReasoningChain) -> ReasoningResult:
        """Execute reasoning chain sequentially"""
        
        result = ReasoningResult(
            chain_id=chain.chain_id,
            success=True,
            final_result={}
        )
        
        # Build dependency-ordered execution list
        execution_order = self._resolve_dependencies(chain)
        
        if not execution_order:
            result.success = False
            result.errors.append("Unable to resolve step dependencies")
            return result
        
        # Execute steps in order
        for step_id in execution_order:
            step = chain.get_step(step_id)
            if not step:
                continue
            
            try:
                step_result = await self._execute_reasoning_step(step, chain)
                result.step_results.append(step_result.to_dict() if hasattr(step_result, 'to_dict') else step_result)
                result.steps_executed += 1
                
                # Update shared memory with step results
                if step.result:
                    for output_key in step.provides:
                        if output_key in step.result:
                            chain.shared_memory[output_key] = step.result[output_key]
                
                if step.status == StepStatus.COMPLETED:
                    chain.completed_steps += 1
                else:
                    chain.failed_steps += 1
                    result.warnings.append(f"Step {step.name} did not complete successfully")
                
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error_message = str(e)
                chain.failed_steps += 1
                result.errors.append(f"Step {step.name} failed: {e}")
        
        # Compile final result
        if chain.completed_steps > 0:
            result.final_result = chain.shared_memory.copy()
        else:
            result.success = False
            result.errors.append("No steps completed successfully")
        
        return result
    
    async def _execute_parallel(self, chain: ReasoningChain) -> ReasoningResult:
        """Execute reasoning chain with parallel processing where possible"""
        
        result = ReasoningResult(
            chain_id=chain.chain_id,
            success=True,
            final_result={}
        )
        
        # Group steps by dependency level
        dependency_levels = self._group_by_dependency_level(chain)
        
        # Execute each level in parallel
        for level_steps in dependency_levels:
            if not level_steps:
                continue
            
            # Execute all steps in this level concurrently
            step_tasks = []
            for step_id in level_steps:
                step = chain.get_step(step_id)
                if step:
                    task = asyncio.create_task(self._execute_reasoning_step(step, chain))
                    step_tasks.append((step, task))
            
            # Wait for all steps in this level to complete
            for step, task in step_tasks:
                try:
                    step_result = await task
                    result.step_results.append(step_result.to_dict() if hasattr(step_result, 'to_dict') else step_result)
                    result.steps_executed += 1
                    
                    # Update shared memory
                    if step.result:
                        for output_key in step.provides:
                            if output_key in step.result:
                                chain.shared_memory[output_key] = step.result[output_key]
                    
                    if step.status == StepStatus.COMPLETED:
                        chain.completed_steps += 1
                    else:
                        chain.failed_steps += 1
                
                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.error_message = str(e)
                    chain.failed_steps += 1
                    result.errors.append(f"Step {step.name} failed: {e}")
        
        # Compile final result
        if chain.completed_steps > 0:
            result.final_result = chain.shared_memory.copy()
        else:
            result.success = False
            result.errors.append("No steps completed successfully")
        
        return result
    
    async def _execute_tree_search(self, chain: ReasoningChain) -> ReasoningResult:
        """Execute reasoning chain using tree search strategy"""
        # Placeholder for tree search implementation
        logger.warning("Tree search strategy not yet implemented, falling back to sequential")
        return await self._execute_sequential(chain)
    
    async def _execute_ensemble(self, chain: ReasoningChain) -> ReasoningResult:
        """Execute reasoning chain using ensemble of different approaches"""
        # Placeholder for ensemble implementation
        logger.warning("Ensemble strategy not yet implemented, falling back to sequential")
        return await self._execute_sequential(chain)
    
    async def _execute_reasoning_step(self, step: ReasoningStep, chain: ReasoningChain) -> Dict[str, Any]:
        """Execute an individual reasoning step"""
        
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        start_time = step.started_at
        
        try:
            # Prepare input data from dependencies and context
            step_input = self._prepare_step_input(step, chain)
            
            # Create task for model execution
            task_data = {
                "step_id": step.step_id,
                "reasoning_type": step.reasoning_type.value,
                "prompt": self._format_prompt(step.prompt_template, step_input),
                "max_tokens": step.max_tokens,
                "temperature": step.temperature
            }
            
            # Select appropriate model
            required_capabilities = step.model_requirements or {ModelCapability.REASONING}
            best_model = self.model_manager.select_best_model(required_capabilities)
            
            if not best_model:
                raise Exception(f"No suitable model found for step {step.name}")
            
            # Execute on model
            execution_result = await self.model_manager.execute_request(
                best_model.model_id,
                task_data
            )
            
            # Process result
            step.result = self._process_step_result(execution_result, step)
            step.model_used = best_model.model_id
            step.status = StepStatus.COMPLETED
            
            # Calculate execution time
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            step.execution_time_ms = execution_time
            
            # Assess step quality
            quality_scores = await self.quality_assessor.assess_step_result(step)
            step.coherence_score = quality_scores.get("coherence", 0.0)
            step.relevance_score = quality_scores.get("relevance", 0.0)
            step.confidence_score = quality_scores.get("confidence", 0.0)
            
            step.completed_at = datetime.now(timezone.utc)
            
            return {
                "step_id": step.step_id,
                "success": True,
                "result": step.result,
                "execution_time_ms": execution_time,
                "model_used": best_model.model_id,
                "quality_scores": quality_scores
            }
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.execution_time_ms = execution_time
            step.completed_at = datetime.now(timezone.utc)
            
            return {
                "step_id": step.step_id,
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time
            }
    
    def _resolve_dependencies(self, chain: ReasoningChain) -> List[str]:
        """Resolve step dependencies and return execution order"""
        
        # Topological sort of dependencies
        in_degree = {}
        graph = chain.get_dependencies_graph()
        
        # Initialize in-degree count
        for step_id in graph:
            in_degree[step_id] = 0
        
        for step_id, dependencies in graph.items():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[step_id] += 1
        
        # Find steps with no dependencies
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Reduce in-degree for dependent steps
            for step_id, dependencies in graph.items():
                if current in dependencies:
                    in_degree[step_id] -= 1
                    if in_degree[step_id] == 0:
                        queue.append(step_id)
        
        # Check for circular dependencies
        if len(execution_order) != len(graph):
            logger.error("Circular dependencies detected in reasoning chain")
            return []
        
        return execution_order
    
    def _group_by_dependency_level(self, chain: ReasoningChain) -> List[List[str]]:
        """Group steps by dependency level for parallel execution"""
        
        levels = []
        remaining_steps = set(step.step_id for step in chain.steps)
        completed_steps = set()
        
        while remaining_steps:
            current_level = []
            
            # Find steps that can be executed (all dependencies satisfied)
            for step_id in list(remaining_steps):
                step = chain.get_step(step_id)
                if step and all(dep in completed_steps for dep in step.depends_on):
                    current_level.append(step_id)
            
            if not current_level:
                # No progress possible - circular dependency
                logger.error("Circular dependencies prevent parallel execution")
                break
            
            levels.append(current_level)
            
            # Remove current level steps from remaining
            for step_id in current_level:
                remaining_steps.remove(step_id)
                completed_steps.add(step_id)
        
        return levels
    
    def _prepare_step_input(self, step: ReasoningStep, chain: ReasoningChain) -> Dict[str, Any]:
        """Prepare input data for reasoning step"""
        
        step_input = {}
        
        # Add global context
        step_input.update(chain.global_context)
        
        # Add step-specific input
        step_input.update(step.input_data)
        
        # Add dependency outputs
        for dep_id in step.depends_on:
            dep_step = chain.get_step(dep_id)
            if dep_step and dep_step.result:
                # Add all outputs from dependency
                for output_key in dep_step.provides:
                    if output_key in dep_step.result:
                        step_input[output_key] = dep_step.result[output_key]
        
        # Add shared memory
        step_input.update(chain.shared_memory)
        
        return step_input
    
    def _format_prompt(self, template: str, input_data: Dict[str, Any]) -> str:
        """Format prompt template with input data"""
        
        try:
            return template.format(**input_data)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template  # Return unformatted template if formatting fails
    
    def _process_step_result(self, execution_result: Dict[str, Any], step: ReasoningStep) -> Dict[str, Any]:
        """Process and structure step execution result"""
        
        # Extract relevant information from model response
        processed_result = {
            "reasoning_type": step.reasoning_type.value,
            "raw_response": execution_result.get("response", ""),
            "tokens_used": execution_result.get("tokens_used", 0),
            "model_response": execution_result
        }
        
        # Try to extract structured information (would be enhanced with actual parsing)
        response_text = execution_result.get("response", "")
        
        # Simple extraction (would use more sophisticated NLP)
        if "conclusion:" in response_text.lower():
            parts = response_text.lower().split("conclusion:")
            if len(parts) > 1:
                processed_result["conclusion"] = parts[1].strip()
        
        if "confidence:" in response_text.lower():
            parts = response_text.lower().split("confidence:")
            if len(parts) > 1:
                confidence_text = parts[1].strip().split()[0]
                try:
                    confidence = float(confidence_text.rstrip('%'))
                    processed_result["confidence"] = confidence
                    step.confidence_score = confidence
                except ValueError:
                    pass
        
        return processed_result
    
    def _update_avg_metrics(self, execution_time_ms: float, confidence_score: float):
        """Update average metrics"""
        
        total_executions = self.stats["total_chains_executed"]
        
        # Update average execution time
        current_avg_time = self.stats["avg_execution_time_ms"]
        self.stats["avg_execution_time_ms"] = \
            (current_avg_time * (total_executions - 1) + execution_time_ms) / total_executions
        
        # Update average confidence
        current_avg_confidence = self.stats["avg_confidence_score"]
        self.stats["avg_confidence_score"] = \
            (current_avg_confidence * (total_executions - 1) + confidence_score) / total_executions
    
    def get_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Get reasoning chain by ID"""
        return self.reasoning_chains.get(chain_id)
    
    def list_reasoning_chains(self) -> List[ReasoningChain]:
        """List all reasoning chains"""
        return list(self.reasoning_chains.values())
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive reasoning engine statistics"""
        return {
            "engine_statistics": self.stats,
            "total_chains": len(self.reasoning_chains),
            "active_executions": len(self.active_executions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def shutdown(self):
        """Graceful shutdown of reasoning engine"""
        logger.info("Shutting down Reasoning Engine")
        
        # Cancel active executions
        for execution_id, task in self.active_executions.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.active_executions.clear()
        
        logger.info("Reasoning Engine shutdown complete")


# Export main classes
__all__ = [
    'ReasoningType',
    'ReasoningStrategy',
    'StepStatus',
    'ReasoningStep',
    'ReasoningChain',
    'ReasoningResult',
    'ReasoningTemplateLibrary',
    'ReasoningQualityAssessor',
    'ReasoningEngine'
]
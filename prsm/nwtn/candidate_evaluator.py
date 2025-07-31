#!/usr/bin/env python3
"""
Candidate Evaluator for NWTN System 1 → System 2 → Attribution Pipeline
========================================================================

This module implements System 2 "methodical evaluation" that takes diverse candidate
answers from System 1 and evaluates them using the existing MetaReasoningEngine's
sophisticated reasoning capabilities.

Part of Phase 2 of the NWTN System 1 → System 2 → Attribution roadmap.
"""

import asyncio
import structlog
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum
import json

from prsm.nwtn.candidate_answer_generator import (
    CandidateGenerationResult,
    CandidateAnswer,
    CandidateType,
    SourceContribution
)
from prsm.nwtn.meta_reasoning_engine import (
    MetaReasoningEngine,
    ThinkingMode,
    MetaReasoningResult,
    ReasoningEngine,
    ReasoningMode
)
from prsm.nwtn.breakthrough_modes import BreakthroughModeConfig

logger = structlog.get_logger(__name__)


class EvaluationCriteria(Enum):
    """Criteria for evaluating candidate answers"""
    RELEVANCE = "relevance"  # How well does it answer the query?
    ACCURACY = "accuracy"    # How accurate is the information?
    EVIDENCE = "evidence"    # How strong is the supporting evidence?
    COHERENCE = "coherence"  # How logically consistent is the answer?
    NOVELTY = "novelty"      # How novel or insightful is the answer?
    COMPLETENESS = "completeness"  # How complete is the answer?
    RELIABILITY = "reliability"    # How reliable are the sources?


@dataclass
class EvaluationScore:
    """Score for a single evaluation criterion"""
    criterion: EvaluationCriteria
    score: float  # 0.0 to 1.0
    reasoning: str
    confidence: float  # 0.0 to 1.0
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class CandidateEvaluation:
    """Complete evaluation of a candidate answer"""
    candidate_id: str
    candidate_answer: CandidateAnswer
    evaluation_scores: List[EvaluationScore]
    overall_score: float
    ranking_position: int
    reasoning_results: List[MetaReasoningResult]  # Results from each reasoning engine
    source_confidence: Dict[str, float]  # Confidence in each source
    evaluation_summary: str
    strengths: List[str]
    weaknesses: List[str]
    evaluation_time: float
    evaluation_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class EvaluationResult:
    """Complete result of candidate evaluation process"""
    query: str
    candidate_evaluations: List[CandidateEvaluation]
    best_candidate: Optional[CandidateEvaluation]
    evaluation_time_seconds: float
    thinking_mode_used: ThinkingMode
    evaluation_criteria_used: List[EvaluationCriteria]
    overall_confidence: float
    evaluation_summary: str
    source_lineage: Dict[str, List[str]]  # paper_id -> list of reasoning steps
    evaluation_id: str = field(default_factory=lambda: str(uuid4()))
    
    @property
    def confidence(self) -> float:
        """Compatibility property - returns overall_confidence"""
        return self.overall_confidence


class RelevanceScorer:
    """Evaluates how well candidate answers address the query"""
    
    def __init__(self, meta_reasoning_engine: MetaReasoningEngine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.initialized = False
    
    async def initialize(self):
        """Initialize the relevance scorer"""
        self.initialized = True
        logger.info("RelevanceScorer initialized")
    
    async def score_relevance(self, query: str, candidate: CandidateAnswer) -> EvaluationScore:
        """Score how well a candidate answers the query"""
        try:
            # Create evaluation context
            context = {
                'evaluation_type': 'relevance',
                'original_query': query,
                'candidate_answer': candidate.answer_text,
                'candidate_type': candidate.answer_type.value,
                'source_count': len(candidate.source_contributions)
            }
            
            # Use meta-reasoning to evaluate relevance
            relevance_query = f"How well does this answer address the question '{query}'? Answer: {candidate.answer_text}"
            
            reasoning_result = await self.meta_reasoning_engine.meta_reason(
                query=relevance_query,
                context=context,
                thinking_mode=ThinkingMode.QUICK,
                include_world_model=True
            )
            
            # Extract relevance score from reasoning result
            relevance_score = self._extract_relevance_score(reasoning_result)
            
            return EvaluationScore(
                criterion=EvaluationCriteria.RELEVANCE,
                score=relevance_score,
                reasoning=f"Meta-reasoning assessment: {reasoning_result.synthesis_summary[:200]}...",
                confidence=reasoning_result.confidence_score,
                supporting_evidence=[f"Reasoning engines used: {len(reasoning_result.reasoning_results)}"]
            )
            
        except Exception as e:
            logger.error(f"Failed to score relevance: {e}")
            return EvaluationScore(
                criterion=EvaluationCriteria.RELEVANCE,
                score=0.5,  # Default moderate score
                reasoning=f"Error during relevance evaluation: {str(e)}",
                confidence=0.3
            )
    
    def _extract_relevance_score(self, reasoning_result: MetaReasoningResult) -> float:
        """Extract relevance score from meta-reasoning result"""
        # Use confidence score as proxy for relevance
        base_score = reasoning_result.confidence_score
        
        # Adjust based on synthesis quality
        if reasoning_result.synthesis_summary:
            # Higher scores for more comprehensive synthesis
            synthesis_length = len(reasoning_result.synthesis_summary)
            if synthesis_length > 200:
                base_score += 0.1
            elif synthesis_length > 100:
                base_score += 0.05
        
        # Consider number of reasoning engines that provided results
        if reasoning_result.reasoning_results:
            engine_coverage = len(reasoning_result.reasoning_results) / 7.0  # 7 total engines
            base_score = (base_score + engine_coverage) / 2.0
        
        return max(0.0, min(1.0, base_score))


class ConfidenceScorer:
    """Evaluates evidence quality and reasoning consistency"""
    
    def __init__(self, meta_reasoning_engine: MetaReasoningEngine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.initialized = False
    
    async def initialize(self):
        """Initialize the confidence scorer"""
        self.initialized = True
        logger.info("ConfidenceScorer initialized")
    
    async def score_confidence(self, query: str, candidate: CandidateAnswer) -> EvaluationScore:
        """Score the confidence/evidence quality of a candidate"""
        try:
            # Create evaluation context
            context = {
                'evaluation_type': 'confidence',
                'original_query': query,
                'candidate_answer': candidate.answer_text,
                'source_contributions': [
                    {
                        'paper_id': contrib.paper_id,
                        'title': contrib.title,
                        'contribution_weight': contrib.contribution_weight,
                        'quality_score': contrib.quality_score
                    } for contrib in candidate.source_contributions
                ],
                'reasoning_chain': candidate.reasoning_chain
            }
            
            # Use meta-reasoning to evaluate evidence quality
            confidence_query = f"How strong is the evidence supporting this answer to '{query}'? Answer: {candidate.answer_text}. Sources: {[c.title for c in candidate.source_contributions]}"
            
            reasoning_result = await self.meta_reasoning_engine.meta_reason(
                query=confidence_query,
                context=context,
                thinking_mode=ThinkingMode.INTERMEDIATE,  # Use deeper reasoning for confidence
                include_world_model=True
            )
            
            # Extract confidence score
            confidence_score = self._extract_confidence_score(reasoning_result, candidate)
            
            return EvaluationScore(
                criterion=EvaluationCriteria.EVIDENCE,
                score=confidence_score,
                reasoning=f"Evidence assessment: {reasoning_result.synthesis_summary[:200]}...",
                confidence=reasoning_result.confidence_score,
                supporting_evidence=[
                    f"Source quality: {sum(c.quality_score for c in candidate.source_contributions) / len(candidate.source_contributions):.2f}",
                    f"Reasoning chain length: {len(candidate.reasoning_chain)}",
                    f"Meta-reasoning engines: {len(reasoning_result.reasoning_results)}"
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to score confidence: {e}")
            return EvaluationScore(
                criterion=EvaluationCriteria.EVIDENCE,
                score=0.5,
                reasoning=f"Error during confidence evaluation: {str(e)}",
                confidence=0.3
            )
    
    def _extract_confidence_score(self, reasoning_result: MetaReasoningResult, candidate: CandidateAnswer) -> float:
        """Extract confidence score from meta-reasoning result and candidate data"""
        # Start with meta-reasoning confidence
        base_score = reasoning_result.confidence_score
        
        # Factor in source quality
        if candidate.source_contributions:
            avg_source_quality = sum(c.quality_score for c in candidate.source_contributions) / len(candidate.source_contributions)
            base_score = (base_score + avg_source_quality) / 2.0
        
        # Factor in reasoning chain depth
        if candidate.reasoning_chain:
            reasoning_bonus = min(len(candidate.reasoning_chain) / 5.0, 0.2)  # Max 0.2 bonus
            base_score += reasoning_bonus
        
        # Factor in candidate's original confidence
        base_score = (base_score + candidate.confidence_score) / 2.0
        
        return max(0.0, min(1.0, base_score))


class SourceTracker:
    """Tracks source lineage through the evaluation process"""
    
    def __init__(self):
        self.source_lineage = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize the source tracker"""
        self.initialized = True
        logger.info("SourceTracker initialized")
    
    def track_source_usage(self, paper_id: str, reasoning_step: str):
        """Track how a source was used in reasoning"""
        if paper_id not in self.source_lineage:
            self.source_lineage[paper_id] = []
        self.source_lineage[paper_id].append(reasoning_step)
    
    def get_source_lineage(self) -> Dict[str, List[str]]:
        """Get complete source lineage"""
        return self.source_lineage.copy()
    
    def reset(self):
        """Reset source tracking for new evaluation"""
        self.source_lineage.clear()


class CandidateEvaluator:
    """
    System 2 methodical evaluation component that evaluates candidate answers
    using the existing MetaReasoningEngine's sophisticated reasoning capabilities
    """
    
    def __init__(self, meta_reasoning_engine: Optional[MetaReasoningEngine] = None):
        self.meta_reasoning_engine = meta_reasoning_engine or MetaReasoningEngine()
        self.relevance_scorer = RelevanceScorer(self.meta_reasoning_engine)
        self.confidence_scorer = ConfidenceScorer(self.meta_reasoning_engine)
        self.source_tracker = SourceTracker()
        self.initialized = False
        
        # Evaluation parameters
        self.default_criteria = [
            EvaluationCriteria.RELEVANCE,
            EvaluationCriteria.EVIDENCE,
            EvaluationCriteria.COHERENCE,
            EvaluationCriteria.COMPLETENESS
        ]
        self.default_thinking_mode = ThinkingMode.INTERMEDIATE
        
        # Evaluation statistics
        self.evaluation_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'average_evaluation_time': 0.0,
            'criteria_usage': {c: 0 for c in EvaluationCriteria}
        }
    
    async def initialize(self):
        """Initialize the candidate evaluator"""
        try:
            # Initialize meta-reasoning engine
            await self.meta_reasoning_engine.initialize_external_knowledge_base()
            
            # Initialize scoring components
            await self.relevance_scorer.initialize()
            await self.confidence_scorer.initialize()
            await self.source_tracker.initialize()
            
            self.initialized = True
            logger.info("CandidateEvaluator initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CandidateEvaluator: {e}")
            return False
    
    async def evaluate_candidates(self, 
                                candidate_result: CandidateGenerationResult,
                                evaluation_criteria: Optional[List[EvaluationCriteria]] = None,
                                thinking_mode: Optional[ThinkingMode] = None,
                                context: Optional[Dict[str, Any]] = None,
                                breakthrough_config: Optional[BreakthroughModeConfig] = None) -> EvaluationResult:
        """
        Evaluate candidate answers using System 2 meta-reasoning with breakthrough mode awareness
        
        Args:
            candidate_result: Result from CandidateAnswerGenerator
            evaluation_criteria: Criteria to evaluate (default: relevance, evidence, coherence, completeness)
            thinking_mode: Thinking mode for meta-reasoning (default: intermediate)
            context: Additional context for evaluation
            breakthrough_config: Breakthrough mode configuration for System 2 validation parameters
            
        Returns:
            EvaluationResult with ranked candidates and detailed evaluations
        """
        start_time = datetime.now(timezone.utc)
        
        if not self.initialized:
            await self.initialize()
        
        criteria = evaluation_criteria or self.default_criteria
        thinking_mode = thinking_mode or self.default_thinking_mode
        context = context or {}
        
        # Apply breakthrough mode configuration for System 2 validation
        if breakthrough_config:
            # Adjust evaluation strictness based on breakthrough mode
            validation_strictness = breakthrough_config.reasoning_engine_config.validation_strictness
            evidence_requirement = breakthrough_config.reasoning_engine_config.evidence_requirement
            logical_rigor = breakthrough_config.reasoning_engine_config.logical_rigor
            
            # Enhanced criteria based on breakthrough mode
            if validation_strictness > 0.7:  # High validation mode
                if EvaluationCriteria.ACCURACY not in criteria:
                    criteria.append(EvaluationCriteria.ACCURACY)
                if EvaluationCriteria.RELIABILITY not in criteria:
                    criteria.append(EvaluationCriteria.RELIABILITY)
            
            context.update({
                "breakthrough_mode": True,
                "validation_strictness": validation_strictness,
                "evidence_requirement": evidence_requirement,
                "logical_rigor": logical_rigor,
                "reasoning_mode": ReasoningMode.SYSTEM2_VALIDATION
            })
            
            logger.info("System 2 validation configured with breakthrough parameters",
                       validation_strictness=validation_strictness,
                       evidence_requirement=evidence_requirement,
                       logical_rigor=logical_rigor)
        
        # Reset source tracking
        self.source_tracker.reset()
        
        try:
            logger.info("Starting candidate evaluation",
                       query=candidate_result.query[:50],
                       candidates=len(candidate_result.candidate_answers),
                       criteria=len(criteria),
                       thinking_mode=thinking_mode.value)
            
            # Evaluate each candidate
            candidate_evaluations = []
            
            for candidate in candidate_result.candidate_answers:
                try:
                    evaluation = await self._evaluate_single_candidate(
                        candidate_result.query,
                        candidate,
                        criteria,
                        thinking_mode,
                        context,
                        breakthrough_config
                    )
                    candidate_evaluations.append(evaluation)
                    
                    # Track source usage
                    for contribution in candidate.source_contributions:
                        self.source_tracker.track_source_usage(
                            contribution.paper_id,
                            f"Candidate {candidate.candidate_id} evaluation"
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to evaluate candidate {candidate.candidate_id}: {e}")
                    continue
            
            # Rank candidates by overall score
            candidate_evaluations.sort(key=lambda x: x.overall_score, reverse=True)
            for i, evaluation in enumerate(candidate_evaluations):
                evaluation.ranking_position = i + 1
            
            # Identify best candidate
            best_candidate = candidate_evaluations[0] if candidate_evaluations else None
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(candidate_evaluations)
            
            # Generate evaluation summary
            evaluation_summary = self._generate_evaluation_summary(candidate_evaluations, criteria)
            
            # Calculate evaluation time
            evaluation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update statistics
            self._update_evaluation_stats(evaluation_time, len(candidate_evaluations), criteria)
            
            result = EvaluationResult(
                query=candidate_result.query,
                candidate_evaluations=candidate_evaluations,
                best_candidate=best_candidate,
                evaluation_time_seconds=evaluation_time,
                thinking_mode_used=thinking_mode,
                evaluation_criteria_used=criteria,
                overall_confidence=overall_confidence,
                evaluation_summary=evaluation_summary,
                source_lineage=self.source_tracker.get_source_lineage()
            )
            
            logger.info("Candidate evaluation completed",
                       query=candidate_result.query[:50],
                       candidates_evaluated=len(candidate_evaluations),
                       best_score=best_candidate.overall_score if best_candidate else 0.0,
                       evaluation_time=evaluation_time,
                       overall_confidence=overall_confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Candidate evaluation failed: {e}")
            return EvaluationResult(
                query=candidate_result.query,
                candidate_evaluations=[],
                best_candidate=None,
                evaluation_time_seconds=0.0,
                thinking_mode_used=thinking_mode,
                evaluation_criteria_used=criteria,
                overall_confidence=0.0,
                evaluation_summary="Evaluation failed due to error",
                source_lineage={}
            )
    
    async def _evaluate_single_candidate(self, 
                                       query: str,
                                       candidate: CandidateAnswer,
                                       criteria: List[EvaluationCriteria],
                                       thinking_mode: ThinkingMode,
                                       context: Optional[Dict[str, Any]] = None,
                                       breakthrough_config: Optional[BreakthroughModeConfig] = None) -> CandidateEvaluation:
        """Evaluate a single candidate answer"""
        evaluation_start = datetime.now(timezone.utc)
        
        try:
            # Collect evaluation scores for each criterion
            evaluation_scores = []
            reasoning_results = []
            
            for criterion in criteria:
                try:
                    if criterion == EvaluationCriteria.RELEVANCE:
                        score = await self.relevance_scorer.score_relevance(query, candidate)
                    elif criterion == EvaluationCriteria.EVIDENCE:
                        score = await self.confidence_scorer.score_confidence(query, candidate)
                    else:
                        # For other criteria, use breakthrough-enhanced meta-reasoning evaluation
                        score = await self._evaluate_general_criterion(query, candidate, criterion, thinking_mode, context)
                    
                    evaluation_scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate criterion {criterion.value}: {e}")
                    # Add fallback score
                    evaluation_scores.append(EvaluationScore(
                        criterion=criterion,
                        score=0.5,
                        reasoning=f"Error evaluating {criterion.value}: {str(e)}",
                        confidence=0.3
                    ))
            
            # Calculate overall score
            overall_score = sum(score.score for score in evaluation_scores) / len(evaluation_scores)
            
            # Generate source confidence scores
            source_confidence = self._calculate_source_confidence(candidate, evaluation_scores)
            
            # Generate evaluation summary
            evaluation_summary = self._generate_candidate_summary(candidate, evaluation_scores)
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(evaluation_scores)
            
            # Calculate evaluation time
            evaluation_time = (datetime.now(timezone.utc) - evaluation_start).total_seconds()
            
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                candidate_answer=candidate,
                evaluation_scores=evaluation_scores,
                overall_score=overall_score,
                ranking_position=0,  # Will be set during ranking
                reasoning_results=reasoning_results,
                source_confidence=source_confidence,
                evaluation_summary=evaluation_summary,
                strengths=strengths,
                weaknesses=weaknesses,
                evaluation_time=evaluation_time
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate candidate {candidate.candidate_id}: {e}")
            raise
    
    async def _evaluate_general_criterion(self, query: str, candidate: CandidateAnswer, 
                                        criterion: EvaluationCriteria, thinking_mode: ThinkingMode,
                                        context: Optional[Dict[str, Any]] = None) -> EvaluationScore:
        """Evaluate a general criterion using meta-reasoning"""
        try:
            # Create enhanced evaluation context with breakthrough parameters
            eval_context = {
                'evaluation_type': criterion.value,
                'original_query': query,
                'candidate_answer': candidate.answer_text,
                'candidate_type': candidate.answer_type.value
            }
            
            # Add breakthrough mode context if available
            if context:
                eval_context.update(context)
                
                # Apply stricter evaluation for breakthrough modes
                if context.get("breakthrough_mode") and context.get("validation_strictness", 0) > 0.7:
                    eval_context['evaluation_strictness'] = 'high'
                    eval_context['evidence_threshold'] = context.get("evidence_requirement", 0.6)
                    eval_context['logical_rigor'] = context.get("logical_rigor", 0.8)
            
            # Create evaluation query
            criterion_queries = {
                EvaluationCriteria.ACCURACY: f"How accurate is this answer to '{query}'? Answer: {candidate.answer_text}",
                EvaluationCriteria.COHERENCE: f"How logically coherent is this answer to '{query}'? Answer: {candidate.answer_text}",
                EvaluationCriteria.NOVELTY: f"How novel or insightful is this answer to '{query}'? Answer: {candidate.answer_text}",
                EvaluationCriteria.COMPLETENESS: f"How complete is this answer to '{query}'? Answer: {candidate.answer_text}",
                EvaluationCriteria.RELIABILITY: f"How reliable are the sources for this answer to '{query}'? Answer: {candidate.answer_text}"
            }
            
            evaluation_query = criterion_queries.get(criterion, f"Evaluate this answer for {criterion.value}: {candidate.answer_text}")
            
            # Use breakthrough-enhanced meta-reasoning for evaluation
            reasoning_result = await self.meta_reasoning_engine.meta_reason(
                query=evaluation_query,
                context=eval_context,
                thinking_mode=thinking_mode,
                reasoning_mode=eval_context.get("reasoning_mode", ReasoningMode.SYSTEM2_VALIDATION),
                breakthrough_config=context.get("breakthrough_config") if context else None,
                include_world_model=True
            )
            
            # Extract and adjust score based on breakthrough parameters
            score = reasoning_result.confidence_score
            
            # Apply breakthrough mode adjustments to scoring
            if context and context.get("breakthrough_mode"):
                validation_strictness = context.get("validation_strictness", 0.7)
                
                # Apply stricter scoring thresholds for high validation modes
                if validation_strictness > 0.8:
                    # In conservative/high-validation modes, penalize lower scores more
                    if score < 0.6:
                        score *= 0.8  # Reduce score for borderline cases
                elif validation_strictness < 0.5:
                    # In creative/revolutionary modes, be more lenient with scores
                    if score > 0.4:
                        score = min(1.0, score * 1.1)  # Boost reasonable scores
            
            # Create breakthrough-enhanced reasoning description
            reasoning_desc = f"{criterion.value.title()} assessment: {reasoning_result.synthesis_summary[:150]}..."
            if context and context.get("breakthrough_mode"):
                mode_info = f" [System 2 validation: strictness={context.get('validation_strictness', 0.7):.2f}]"
                reasoning_desc += mode_info
            
            return EvaluationScore(
                criterion=criterion,
                score=score,
                reasoning=reasoning_desc,
                confidence=reasoning_result.confidence_score,
                supporting_evidence=[
                    f"Meta-reasoning engines: {len(reasoning_result.reasoning_results)}",
                    f"Breakthrough mode: {context.get('breakthrough_mode', False) if context else False}"
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate criterion {criterion.value}: {e}")
            return EvaluationScore(
                criterion=criterion,
                score=0.5,
                reasoning=f"Error evaluating {criterion.value}: {str(e)}",
                confidence=0.3
            )
    
    def _calculate_source_confidence(self, candidate: CandidateAnswer, 
                                   evaluation_scores: List[EvaluationScore]) -> Dict[str, float]:
        """Calculate confidence scores for each source"""
        source_confidence = {}
        
        for contribution in candidate.source_contributions:
            # Base confidence from source quality
            confidence = contribution.quality_score
            
            # Adjust based on evaluation scores
            avg_evaluation_score = sum(score.score for score in evaluation_scores) / len(evaluation_scores)
            confidence = (confidence + avg_evaluation_score) / 2.0
            
            # Factor in contribution weight
            confidence *= contribution.contribution_weight
            
            source_confidence[contribution.paper_id] = confidence
        
        return source_confidence
    
    def _generate_candidate_summary(self, candidate: CandidateAnswer, 
                                  evaluation_scores: List[EvaluationScore]) -> str:
        """Generate summary of candidate evaluation"""
        scores_text = ", ".join([f"{score.criterion.value}: {score.score:.2f}" for score in evaluation_scores])
        return f"Candidate {candidate.answer_type.value} evaluated with scores: {scores_text}"
    
    def _identify_strengths_weaknesses(self, evaluation_scores: List[EvaluationScore]) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses based on evaluation scores"""
        strengths = []
        weaknesses = []
        
        for score in evaluation_scores:
            if score.score > 0.7:
                strengths.append(f"Strong {score.criterion.value} (score: {score.score:.2f})")
            elif score.score < 0.4:
                weaknesses.append(f"Weak {score.criterion.value} (score: {score.score:.2f})")
        
        return strengths, weaknesses
    
    def _calculate_overall_confidence(self, evaluations: List[CandidateEvaluation]) -> float:
        """Calculate overall confidence in the evaluation results"""
        if not evaluations:
            return 0.0
        
        # Average of all evaluation scores
        all_scores = []
        for evaluation in evaluations:
            all_scores.extend([score.score for score in evaluation.evaluation_scores])
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    def _generate_evaluation_summary(self, evaluations: List[CandidateEvaluation], 
                                   criteria: List[EvaluationCriteria]) -> str:
        """Generate summary of overall evaluation results"""
        if not evaluations:
            return "No candidates were successfully evaluated"
        
        best_score = evaluations[0].overall_score
        criteria_text = ", ".join([c.value for c in criteria])
        
        return f"Evaluated {len(evaluations)} candidates using {criteria_text}. Best candidate scored {best_score:.2f}"
    
    def _update_evaluation_stats(self, evaluation_time: float, candidates_evaluated: int, 
                               criteria: List[EvaluationCriteria]):
        """Update evaluation statistics"""
        self.evaluation_stats['total_evaluations'] += 1
        self.evaluation_stats['successful_evaluations'] += 1 if candidates_evaluated > 0 else 0
        
        # Update average evaluation time
        total_time = (self.evaluation_stats['average_evaluation_time'] * 
                     (self.evaluation_stats['total_evaluations'] - 1) + evaluation_time)
        self.evaluation_stats['average_evaluation_time'] = total_time / self.evaluation_stats['total_evaluations']
        
        # Update criteria usage
        for criterion in criteria:
            self.evaluation_stats['criteria_usage'][criterion] += 1
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            **self.evaluation_stats,
            'success_rate': (self.evaluation_stats['successful_evaluations'] / 
                           max(1, self.evaluation_stats['total_evaluations']))
        }
    
    async def configure_evaluation_params(self, 
                                        default_criteria: Optional[List[EvaluationCriteria]] = None,
                                        default_thinking_mode: Optional[ThinkingMode] = None):
        """Configure evaluation parameters"""
        if default_criteria is not None:
            self.default_criteria = default_criteria
        if default_thinking_mode is not None:
            self.default_thinking_mode = default_thinking_mode
        
        logger.info("Evaluation parameters configured",
                   criteria=len(self.default_criteria),
                   thinking_mode=self.default_thinking_mode.value)


# Factory function for easy instantiation
async def create_candidate_evaluator(meta_reasoning_engine: Optional[MetaReasoningEngine] = None) -> CandidateEvaluator:
    """Create and initialize a candidate evaluator"""
    evaluator = CandidateEvaluator(meta_reasoning_engine)
    await evaluator.initialize()
    return evaluator
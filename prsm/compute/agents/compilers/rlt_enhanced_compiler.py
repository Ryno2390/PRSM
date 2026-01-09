"""
RLT-Enhanced Hierarchical Compiler

Extends the HierarchicalCompiler with RLT (Reinforcement Learning Teachers)
quality metrics integration for explanation quality assessment, logical coherence
evaluation, and quality-weighted result compilation.

Key RLT Enhancements:
- Explanation quality assessment at all compilation levels
- Logical coherence evaluation across teacher outputs
- Quality-weighted synthesis and conflict resolution
- Multi-level quality scoring with RLT reward integration
- Teaching effectiveness impact on compilation confidence
"""

import asyncio
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import structlog

from .hierarchical_compiler import HierarchicalCompiler, IntermediateResult, MidResult, FinalResponse
from ..routers.rlt_enhanced_router import RLTTeacherSelection, RLTTeacherCandidate
from ...teachers.rlt.quality_monitor import QualityMetrics, QualityMonitor, MonitoringConfig
from ...teachers.rlt.student_comprehension_evaluator import ComprehensionMetrics, EvaluationConfig
from ...teachers.rlt.dense_reward_trainer import RLTTrainingConfig
from prsm.core.models import (
    UserInput, PRSMSession, AgentResponse, ReasoningStep, 
    TaskStatus, ContextUsage
)

logger = structlog.get_logger(__name__)


class RLTQualityAssessment:
    """RLT-specific quality assessment metrics for teacher explanations"""
    
    def __init__(
        self,
        explanation_id: str,
        teacher_id: str,
        explanation_quality: float = 0.0,
        logical_coherence: float = 0.0,
        concept_coverage: float = 0.0,
        student_comprehension_prediction: float = 0.0,
        dense_reward_score: float = 0.0,
        teaching_effectiveness: float = 0.0
    ):
        self.explanation_id = explanation_id
        self.teacher_id = teacher_id
        self.explanation_quality = max(0.0, min(1.0, explanation_quality))
        self.logical_coherence = max(0.0, min(1.0, logical_coherence))
        self.concept_coverage = max(0.0, min(1.0, concept_coverage))
        self.student_comprehension_prediction = max(0.0, min(1.0, student_comprehension_prediction))
        self.dense_reward_score = max(0.0, min(1.0, dense_reward_score))
        self.teaching_effectiveness = max(0.0, min(1.0, teaching_effectiveness))
        self.timestamp = datetime.now(timezone.utc)
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall RLT quality score with weighted components"""
        return (
            self.explanation_quality * 0.25 +
            self.logical_coherence * 0.20 +
            self.concept_coverage * 0.15 +
            self.student_comprehension_prediction * 0.20 +
            self.dense_reward_score * 0.15 +
            self.teaching_effectiveness * 0.05
        )
    
    def get_quality_breakdown(self) -> Dict[str, float]:
        """Get detailed quality metric breakdown"""
        return {
            "explanation_quality": self.explanation_quality,
            "logical_coherence": self.logical_coherence,
            "concept_coverage": self.concept_coverage,
            "student_comprehension_prediction": self.student_comprehension_prediction,
            "dense_reward_score": self.dense_reward_score,
            "teaching_effectiveness": self.teaching_effectiveness,
            "overall_quality": self.calculate_overall_quality()
        }


class RLTCompilationMetrics:
    """Comprehensive compilation metrics enhanced with RLT quality assessments"""
    
    def __init__(self):
        self.teacher_quality_assessments: Dict[str, RLTQualityAssessment] = {}
        self.synthesis_quality_impact: float = 0.0
        self.conflict_resolution_quality: float = 0.0
        self.overall_explanation_coherence: float = 0.0
        self.quality_weighted_confidence: float = 0.0
        self.teaching_effectiveness_factor: float = 0.0
        self.compilation_timestamp = datetime.now(timezone.utc)
    
    def add_teacher_assessment(self, assessment: RLTQualityAssessment):
        """Add teacher quality assessment to compilation metrics"""
        self.teacher_quality_assessments[assessment.teacher_id] = assessment
    
    def calculate_aggregate_quality(self) -> Dict[str, float]:
        """Calculate aggregate quality metrics across all teachers"""
        if not self.teacher_quality_assessments:
            return {"aggregate_quality": 0.0, "quality_variance": 0.0, "teacher_count": 0}
        
        qualities = [assessment.calculate_overall_quality() 
                    for assessment in self.teacher_quality_assessments.values()]
        
        return {
            "aggregate_quality": np.mean(qualities),
            "quality_variance": np.var(qualities),
            "teacher_count": len(qualities),
            "min_quality": np.min(qualities),
            "max_quality": np.max(qualities)
        }


class RLTEnhancedCompiler(HierarchicalCompiler):
    """
    RLT-Enhanced Hierarchical Compiler
    
    Extends the base HierarchicalCompiler with RLT teacher quality assessment,
    logical coherence evaluation, and quality-weighted compilation strategies.
    
    RLT Enhancements:
    - Quality assessment at elemental, mid-level, and final compilation stages
    - Teaching effectiveness integration into confidence calculations
    - Logical coherence evaluation across multiple teacher outputs
    - Quality-weighted synthesis and conflict resolution
    - Multi-level quality scoring with RLT reward integration
    """
    
    def __init__(
        self,
        agent_id: str,
        quality_monitor: Optional[QualityMonitor] = None,
        monitoring_config: Optional[MonitoringConfig] = None,
        rlt_config: Optional[RLTTrainingConfig] = None
    ):
        super().__init__(agent_id)
        
        # RLT-specific components
        self.quality_monitor = quality_monitor or QualityMonitor(
            monitoring_config or MonitoringConfig()
        )
        self.rlt_config = rlt_config or RLTTrainingConfig()
        
        # Quality assessment caching
        self.quality_assessments_cache: Dict[str, RLTQualityAssessment] = {}
        self.compilation_metrics_history: List[RLTCompilationMetrics] = []
        
        # Performance tracking
        self.quality_assessment_times: List[float] = []
        self.compilation_enhancement_times: List[float] = []
        
        logger.info(
            "RLT Enhanced Compiler initialized",
            agent_id=agent_id,
            quality_monitor_enabled=self.quality_monitor is not None
        )
    
    async def _assess_rlt_explanation_quality(
        self,
        teacher_output: Any,
        teacher_id: str,
        context: Dict[str, Any]
    ) -> RLTQualityAssessment:
        """Assess RLT-specific explanation quality metrics for teacher output"""
        start_time = time.time()
        
        try:
            # Extract explanation content from teacher output
            explanation_content = self._extract_explanation_content(teacher_output)
            explanation_id = f"{teacher_id}_{uuid4().hex[:8]}"
            
            # Initialize quality metrics
            quality_metrics = {
                "explanation_quality": 0.0,
                "logical_coherence": 0.0,
                "concept_coverage": 0.0,
                "student_comprehension_prediction": 0.0,
                "dense_reward_score": 0.0,
                "teaching_effectiveness": 0.0
            }
            
            # 1. Explanation Quality Assessment
            quality_metrics["explanation_quality"] = await self._evaluate_explanation_quality(
                explanation_content, context
            )
            
            # 2. Logical Coherence Evaluation
            quality_metrics["logical_coherence"] = await self._evaluate_logical_coherence(
                explanation_content, context
            )
            
            # 3. Concept Coverage Assessment
            quality_metrics["concept_coverage"] = await self._evaluate_concept_coverage(
                explanation_content, context
            )
            
            # 4. Student Comprehension Prediction
            quality_metrics["student_comprehension_prediction"] = await self._predict_student_comprehension(
                explanation_content, context
            )
            
            # 5. Dense Reward Score (if available)
            quality_metrics["dense_reward_score"] = self._extract_dense_reward_score(
                teacher_output, context
            )
            
            # 6. Teaching Effectiveness (from historical data)
            quality_metrics["teaching_effectiveness"] = await self._assess_teaching_effectiveness(
                teacher_id, context
            )
            
            # Create comprehensive assessment
            assessment = RLTQualityAssessment(
                explanation_id=explanation_id,
                teacher_id=teacher_id,
                **quality_metrics
            )
            
            # Cache assessment for reuse
            self.quality_assessments_cache[explanation_id] = assessment
            
            assessment_time = time.time() - start_time
            self.quality_assessment_times.append(assessment_time)
            
            logger.debug(
                "RLT explanation quality assessed",
                teacher_id=teacher_id,
                explanation_id=explanation_id,
                overall_quality=assessment.calculate_overall_quality(),
                assessment_time_ms=assessment_time * 1000
            )
            
            return assessment
            
        except Exception as e:
            logger.error(
                "Failed to assess RLT explanation quality",
                teacher_id=teacher_id,
                error=str(e)
            )
            # Return default assessment on error
            return RLTQualityAssessment(
                explanation_id=f"{teacher_id}_error",
                teacher_id=teacher_id
            )
    
    def _extract_explanation_content(self, teacher_output: Any) -> str:
        """Extract explanation content from teacher output"""
        if isinstance(teacher_output, dict):
            # Try common explanation keys
            for key in ["explanation", "response", "content", "output", "text"]:
                if key in teacher_output and isinstance(teacher_output[key], str):
                    return teacher_output[key]
            return str(teacher_output)
        elif isinstance(teacher_output, str):
            return teacher_output
        else:
            return str(teacher_output)
    
    async def _evaluate_explanation_quality(self, explanation: str, context: Dict[str, Any]) -> float:
        """Evaluate overall explanation quality"""
        if not explanation or len(explanation.strip()) < 10:
            return 0.1
        
        quality_score = 0.5  # Base score
        
        # Length and structure assessment
        word_count = len(explanation.split())
        if 50 <= word_count <= 500:
            quality_score += 0.2
        elif word_count > 20:
            quality_score += 0.1
        
        # Content structure indicators
        structure_indicators = [
            "because", "therefore", "first", "second", "next", "finally",
            "for example", "specifically", "in other words", "to illustrate"
        ]
        structure_score = min(0.2, len([ind for ind in structure_indicators 
                                      if ind in explanation.lower()]) * 0.04)
        quality_score += structure_score
        
        # Mathematical or technical content (if applicable)
        if any(term in context.get("domain", "").lower() for term in ["math", "science", "technical"]):
            technical_indicators = ["equation", "formula", "theorem", "proof", "calculate", "solve"]
            technical_score = min(0.1, len([ind for ind in technical_indicators 
                                          if ind in explanation.lower()]) * 0.02)
            quality_score += technical_score
        
        return min(1.0, quality_score)
    
    async def _evaluate_logical_coherence(self, explanation: str, context: Dict[str, Any]) -> float:
        """Evaluate logical flow and coherence of explanation"""
        if not explanation:
            return 0.0
        
        coherence_score = 0.5  # Base score
        
        # Sentence transition analysis
        sentences = explanation.split('.')
        if len(sentences) > 1:
            # Check for logical connectors
            connectors = ["however", "therefore", "thus", "consequently", "furthermore", 
                         "moreover", "in addition", "as a result", "because of this"]
            connector_count = sum(1 for sentence in sentences 
                                for connector in connectors 
                                if connector in sentence.lower())
            coherence_score += min(0.3, connector_count * 0.1)
        
        # Repetition and consistency check
        words = explanation.lower().split()
        unique_words = set(words)
        if len(unique_words) > len(words) * 0.6:  # Good vocabulary diversity
            coherence_score += 0.1
        
        # Question-answer alignment (if question is in context)
        if "question" in context:
            question_words = set(context["question"].lower().split())
            explanation_words = set(explanation.lower().split())
            overlap = len(question_words & explanation_words) / max(len(question_words), 1)
            coherence_score += min(0.1, overlap)
        
        return min(1.0, coherence_score)
    
    async def _evaluate_concept_coverage(self, explanation: str, context: Dict[str, Any]) -> float:
        """Evaluate how well the explanation covers relevant concepts"""
        if not explanation:
            return 0.0
        
        coverage_score = 0.4  # Base score
        
        # Domain-specific concept coverage
        domain = context.get("domain", "general").lower()
        
        concept_sets = {
            "mathematics": ["number", "equation", "function", "variable", "solution", "proof"],
            "science": ["theory", "experiment", "hypothesis", "evidence", "analysis", "result"],
            "programming": ["code", "function", "variable", "algorithm", "logic", "output"],
            "general": ["concept", "idea", "principle", "method", "approach", "understanding"]
        }
        
        relevant_concepts = concept_sets.get(domain, concept_sets["general"])
        concept_matches = sum(1 for concept in relevant_concepts 
                            if concept in explanation.lower())
        
        coverage_score += min(0.4, concept_matches * 0.08)
        
        # Depth indicators
        depth_indicators = ["detail", "specifically", "in depth", "thoroughly", "comprehensive"]
        depth_score = min(0.2, len([ind for ind in depth_indicators 
                                  if ind in explanation.lower()]) * 0.05)
        coverage_score += depth_score
        
        return min(1.0, coverage_score)
    
    async def _predict_student_comprehension(self, explanation: str, context: Dict[str, Any]) -> float:
        """Predict likely student comprehension based on explanation characteristics"""
        if not explanation:
            return 0.0
        
        comprehension_score = 0.3  # Base score
        
        # Readability assessment
        sentences = explanation.split('.')
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()])
        
        # Optimal sentence length for comprehension (10-20 words)
        if 10 <= avg_sentence_length <= 20:
            comprehension_score += 0.2
        elif 5 <= avg_sentence_length <= 30:
            comprehension_score += 0.1
        
        # Complexity assessment
        complex_words = [word for word in explanation.split() if len(word) > 8]
        complexity_ratio = len(complex_words) / max(len(explanation.split()), 1)
        
        if complexity_ratio < 0.3:  # Not too complex
            comprehension_score += 0.2
        elif complexity_ratio < 0.5:
            comprehension_score += 0.1
        
        # Examples and analogies
        example_indicators = ["for example", "such as", "like", "similar to", "imagine", "think of"]
        example_score = min(0.3, len([ind for ind in example_indicators 
                                    if ind in explanation.lower()]) * 0.1)
        comprehension_score += example_score
        
        return min(1.0, comprehension_score)
    
    def _extract_dense_reward_score(self, teacher_output: Any, context: Dict[str, Any]) -> float:
        """Extract dense reward score from teacher output if available"""
        if isinstance(teacher_output, dict):
            # Check for RLT reward components
            if "dense_rewards" in teacher_output:
                rewards = teacher_output["dense_rewards"]
                if isinstance(rewards, dict):
                    r_ss = rewards.get("r_ss", 0.0)
                    r_kl = rewards.get("r_kl", 0.0)
                    return min(1.0, (r_ss + r_kl) / 2.0)
            
            # Check for quality scores
            if "quality_score" in teacher_output:
                return min(1.0, teacher_output["quality_score"])
        
        return 0.5  # Default neutral score
    
    async def _assess_teaching_effectiveness(self, teacher_id: str, context: Dict[str, Any]) -> float:
        """Assess teaching effectiveness based on historical data"""
        try:
            # Use quality monitor to get historical effectiveness
            if hasattr(self.quality_monitor, 'get_teacher_effectiveness'):
                effectiveness = await self.quality_monitor.get_teacher_effectiveness(teacher_id)
                return min(1.0, max(0.0, effectiveness))
        except Exception as e:
            logger.debug(f"Could not retrieve teaching effectiveness for {teacher_id}: {e}")
        
        # Default effectiveness based on teacher ID patterns
        if "advanced" in teacher_id.lower() or "expert" in teacher_id.lower():
            return 0.8
        elif "basic" in teacher_id.lower():
            return 0.6
        else:
            return 0.7  # Default moderate effectiveness
    
    async def _calculate_rlt_weighted_confidence(
        self,
        responses: List[Any],
        quality_assessments: Dict[str, RLTQualityAssessment],
        base_confidence: float
    ) -> float:
        """Calculate confidence weighted by RLT explanation quality"""
        if not quality_assessments:
            return base_confidence
        
        # Calculate quality-weighted confidence
        total_quality = 0.0
        total_weight = 0.0
        
        for response in responses:
            # Find corresponding quality assessment
            teacher_id = self._extract_teacher_id(response)
            if teacher_id in quality_assessments:
                assessment = quality_assessments[teacher_id]
                quality = assessment.calculate_overall_quality()
                weight = 1.0 + quality  # Higher quality gets more weight
                
                total_quality += quality * weight
                total_weight += weight
        
        if total_weight > 0:
            avg_quality = total_quality / total_weight
            # Adjust base confidence by quality factor
            quality_bonus = (avg_quality - 0.5) * 0.3  # Up to 30% bonus/penalty
            return min(1.0, max(0.0, base_confidence + quality_bonus))
        
        return base_confidence
    
    def _extract_teacher_id(self, response: Any) -> str:
        """Extract teacher ID from response"""
        if isinstance(response, dict):
            for key in ["teacher_id", "agent_id", "model_id", "id"]:
                if key in response:
                    return str(response[key])
        return "unknown_teacher"
    
    async def _resolve_conflicts_with_rlt_quality(
        self,
        conflicts: List[str],
        responses: List[Any],
        quality_assessments: Dict[str, RLTQualityAssessment]
    ) -> List[str]:
        """Resolve conflicts using RLT quality scores as decisive factors"""
        if not conflicts or not quality_assessments:
            return conflicts
        
        resolved_conflicts = []
        
        for conflict in conflicts:
            # Find responses involved in this conflict
            conflict_responses = [r for r in responses if conflict in str(r)]
            
            if len(conflict_responses) <= 1:
                resolved_conflicts.append(conflict)
                continue
            
            # Find highest quality response for this conflict
            best_response = None
            best_quality = -1.0
            
            for response in conflict_responses:
                teacher_id = self._extract_teacher_id(response)
                if teacher_id in quality_assessments:
                    quality = quality_assessments[teacher_id].calculate_overall_quality()
                    if quality > best_quality:
                        best_quality = quality
                        best_response = response
            
            if best_response is not None:
                # Use the highest quality response to resolve conflict
                resolution = f"Resolved by highest quality teacher (quality: {best_quality:.2f}): {best_response}"
                resolved_conflicts.append(resolution)
            else:
                resolved_conflicts.append(conflict)
        
        return resolved_conflicts
    
    async def compile_elemental(
        self,
        responses: List[Any],
        synthesis_strategy: str = "consensus",
        context: Optional[Dict[str, Any]] = None
    ) -> IntermediateResult:
        """Enhanced elemental compilation with RLT quality assessment"""
        start_time = time.time()
        context = context or {}
        
        # Perform RLT quality assessments for all teacher responses
        quality_assessments = {}
        for response in responses:
            teacher_id = self._extract_teacher_id(response)
            assessment = await self._assess_rlt_explanation_quality(
                response, teacher_id, context
            )
            quality_assessments[teacher_id] = assessment
        
        # Call parent's compile_elemental to get base result
        base_result = await super().compile_elemental(responses, synthesis_strategy, context)
        
        # Enhance with RLT quality metrics
        enhanced_confidence = await self._calculate_rlt_weighted_confidence(
            responses, quality_assessments, base_result.confidence
        )
        
        # Create RLT compilation metrics
        compilation_metrics = RLTCompilationMetrics()
        for assessment in quality_assessments.values():
            compilation_metrics.add_teacher_assessment(assessment)
        
        # Enhanced conflict resolution with quality scoring
        if hasattr(base_result, 'conflicts') and base_result.conflicts:
            resolved_conflicts = await self._resolve_conflicts_with_rlt_quality(
                base_result.conflicts, responses, quality_assessments
            )
            base_result.conflicts = resolved_conflicts
        
        # Update result with RLT enhancements
        base_result.confidence = enhanced_confidence
        
        # Add RLT metrics to metadata
        if not hasattr(base_result, 'metadata'):
            base_result.metadata = {}
        
        base_result.metadata['rlt_quality_metrics'] = compilation_metrics.calculate_aggregate_quality()
        base_result.metadata['rlt_teacher_assessments'] = {
            tid: assessment.get_quality_breakdown() 
            for tid, assessment in quality_assessments.items()
        }
        
        compilation_time = time.time() - start_time
        self.compilation_enhancement_times.append(compilation_time)
        
        logger.info(
            "RLT-enhanced elemental compilation completed",
            response_count=len(responses),
            quality_assessments=len(quality_assessments),
            enhanced_confidence=enhanced_confidence,
            compilation_time_ms=compilation_time * 1000
        )
        
        return base_result
    
    async def compile_mid_level(
        self,
        intermediate_results: List[IntermediateResult],
        context: Optional[Dict[str, Any]] = None
    ) -> MidResult:
        """Enhanced mid-level compilation with RLT quality integration"""
        start_time = time.time()
        
        # Extract RLT quality metrics from intermediate results
        all_quality_metrics = []
        for result in intermediate_results:
            if hasattr(result, 'metadata') and 'rlt_quality_metrics' in result.metadata:
                all_quality_metrics.append(result.metadata['rlt_quality_metrics'])
        
        # Call parent's compile_mid_level
        base_result = await super().compile_mid_level(intermediate_results, context)
        
        # Enhance with aggregated RLT quality insights
        if all_quality_metrics:
            aggregate_quality = np.mean([metrics['aggregate_quality'] for metrics in all_quality_metrics])
            quality_variance = np.mean([metrics['quality_variance'] for metrics in all_quality_metrics])
            
            # Adjust confidence based on quality consistency
            quality_consistency_bonus = max(0.0, (0.1 - quality_variance)) * 2  # Reward consistency
            enhanced_confidence = min(1.0, base_result.confidence + quality_consistency_bonus)
            base_result.confidence = enhanced_confidence
            
            # Add aggregated RLT insights to metadata
            if not hasattr(base_result, 'metadata'):
                base_result.metadata = {}
            
            base_result.metadata['rlt_aggregate_quality'] = aggregate_quality
            base_result.metadata['rlt_quality_consistency'] = 1.0 - quality_variance
            base_result.metadata['rlt_quality_enhancement'] = quality_consistency_bonus
        
        compilation_time = time.time() - start_time
        
        logger.info(
            "RLT-enhanced mid-level compilation completed",
            intermediate_count=len(intermediate_results),
            aggregate_quality=all_quality_metrics[0]['aggregate_quality'] if all_quality_metrics else 0.0,
            compilation_time_ms=compilation_time * 1000
        )
        
        return base_result
    
    async def compile_final(
        self,
        mid_results: List[MidResult],
        context: Optional[Dict[str, Any]] = None
    ) -> FinalResponse:
        """Enhanced final compilation with comprehensive RLT quality assessment"""
        start_time = time.time()
        
        # Extract all RLT quality data from mid-level results
        quality_data = {
            "aggregate_qualities": [],
            "quality_consistencies": [],
            "teacher_assessments": {}
        }
        
        for result in mid_results:
            if hasattr(result, 'metadata'):
                if 'rlt_aggregate_quality' in result.metadata:
                    quality_data["aggregate_qualities"].append(result.metadata['rlt_aggregate_quality'])
                if 'rlt_quality_consistency' in result.metadata:
                    quality_data["quality_consistencies"].append(result.metadata['rlt_quality_consistency'])
        
        # Call parent's compile_final
        base_result = await super().compile_final(mid_results, context)
        
        # Apply final RLT quality enhancements
        if quality_data["aggregate_qualities"]:
            overall_quality = np.mean(quality_data["aggregate_qualities"])
            overall_consistency = np.mean(quality_data["quality_consistencies"]) if quality_data["quality_consistencies"] else 0.5
            
            # Final quality-based confidence adjustment
            quality_factor = (overall_quality * 0.7) + (overall_consistency * 0.3)
            final_enhancement = (quality_factor - 0.5) * 0.2  # Up to 20% enhancement
            
            enhanced_final_confidence = min(1.0, max(0.0, base_result.confidence + final_enhancement))
            base_result.confidence = enhanced_final_confidence
            
            # Add comprehensive RLT quality summary
            if not hasattr(base_result, 'metadata'):
                base_result.metadata = {}
            
            base_result.metadata['rlt_final_quality_summary'] = {
                "overall_explanation_quality": overall_quality,
                "quality_consistency": overall_consistency,
                "quality_enhancement_factor": final_enhancement,
                "final_enhanced_confidence": enhanced_final_confidence,
                "quality_assessment_count": len(self.quality_assessments_cache),
                "compilation_method": "rlt_enhanced"
            }
        
        # Store compilation metrics for analysis
        compilation_metrics = RLTCompilationMetrics()
        compilation_metrics.overall_explanation_coherence = quality_data.get("aggregate_qualities", [0.5])[0] if quality_data["aggregate_qualities"] else 0.5
        compilation_metrics.quality_weighted_confidence = base_result.confidence
        self.compilation_metrics_history.append(compilation_metrics)
        
        compilation_time = time.time() - start_time
        
        logger.info(
            "RLT-enhanced final compilation completed",
            mid_results_count=len(mid_results),
            final_confidence=base_result.confidence,
            overall_quality=quality_data.get("aggregate_qualities", [0.0])[0] if quality_data["aggregate_qualities"] else 0.0,
            compilation_time_ms=compilation_time * 1000
        )
        
        return base_result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for RLT-enhanced compilation"""
        base_metrics = super().get_performance_metrics()
        
        rlt_metrics = {
            "quality_assessments_performed": len(self.quality_assessments_cache),
            "avg_quality_assessment_time_ms": np.mean(self.quality_assessment_times) * 1000 if self.quality_assessment_times else 0.0,
            "avg_compilation_enhancement_time_ms": np.mean(self.compilation_enhancement_times) * 1000 if self.compilation_enhancement_times else 0.0,
            "compilation_metrics_history_count": len(self.compilation_metrics_history),
            "cache_hit_ratio": len(self.quality_assessments_cache) / max(len(self.quality_assessment_times), 1),
        }
        
        # Add RLT-specific performance data
        if self.compilation_metrics_history:
            latest_metrics = self.compilation_metrics_history[-1]
            rlt_metrics.update({
                "latest_overall_coherence": latest_metrics.overall_explanation_coherence,
                "latest_quality_weighted_confidence": latest_metrics.quality_weighted_confidence,
                "avg_teaching_effectiveness": np.mean([
                    assessment.teaching_effectiveness 
                    for assessment in self.quality_assessments_cache.values()
                ]) if self.quality_assessments_cache else 0.0
            })
        
        # Merge with base metrics
        combined_metrics = {**base_metrics, **rlt_metrics}
        combined_metrics["compilation_type"] = "rlt_enhanced"
        
        return combined_metrics
    
    async def clear_quality_cache(self):
        """Clear quality assessment cache to free memory"""
        cache_size = len(self.quality_assessments_cache)
        self.quality_assessments_cache.clear()
        
        logger.debug(f"Cleared RLT quality assessment cache ({cache_size} entries)")
    
    def get_quality_insights(self) -> Dict[str, Any]:
        """Get insights from quality assessments for optimization"""
        if not self.quality_assessments_cache:
            return {"insights": "No quality assessments available"}
        
        assessments = list(self.quality_assessments_cache.values())
        
        insights = {
            "total_assessments": len(assessments),
            "avg_explanation_quality": np.mean([a.explanation_quality for a in assessments]),
            "avg_logical_coherence": np.mean([a.logical_coherence for a in assessments]),
            "avg_concept_coverage": np.mean([a.concept_coverage for a in assessments]),
            "avg_student_comprehension_prediction": np.mean([a.student_comprehension_prediction for a in assessments]),
            "avg_teaching_effectiveness": np.mean([a.teaching_effectiveness for a in assessments]),
            "quality_distribution": {
                "high_quality_count": len([a for a in assessments if a.calculate_overall_quality() > 0.8]),
                "medium_quality_count": len([a for a in assessments if 0.5 < a.calculate_overall_quality() <= 0.8]),
                "low_quality_count": len([a for a in assessments if a.calculate_overall_quality() <= 0.5])
            },
            "top_performing_teachers": sorted(
                [(a.teacher_id, a.calculate_overall_quality()) for a in assessments],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
        
        return insights
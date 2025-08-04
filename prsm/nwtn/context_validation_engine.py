"""
Context Validation Engine for NWTN Reasoning Context Aggregator.

This module validates that the rich reasoning context captures key insights
and provides recommendations for improvement when context aggregation
is incomplete or low quality.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .reasoning_context_types import (
    RichReasoningContext, ContextValidationResult, EngineType,
    EngineInsight, SynthesisPattern, AnalogicalConnection,
    ConfidenceAnalysis, BreakthroughAnalysis
)


logger = logging.getLogger(__name__)


class ContextValidationEngine:
    """
    Validates that rich context captures key reasoning insights and
    provides recommendations for improvement when context aggregation
    is incomplete or low quality.
    """
    
    def __init__(self):
        """Initialize the context validation engine"""
        self.validation_criteria = {
            'engine_coverage': {
                'weight': 0.25,
                'min_score': 0.7,
                'description': 'Coverage of reasoning engines'
            },
            'insight_preservation': {
                'weight': 0.30,
                'min_score': 0.8,
                'description': 'Preservation of key insights'
            },
            'confidence_mapping': {
                'weight': 0.15,
                'min_score': 0.6,
                'description': 'Confidence level mapping accuracy'
            },
            'analogical_richness': {
                'weight': 0.15,
                'min_score': 0.5,
                'description': 'Richness of analogical connections'
            },
            'breakthrough_detection': {
                'weight': 0.15,
                'min_score': 0.5,
                'description': 'Detection of breakthrough potential'
            }
        }
        
        logger.info("ContextValidationEngine initialized with validation criteria")
    
    async def validate_context_completeness(self, 
                                          rich_context: RichReasoningContext,
                                          original_reasoning_result: Optional[Any] = None) -> ContextValidationResult:
        """
        Validate that context aggregation captured key insights.
        
        Args:
            rich_context: The aggregated rich reasoning context
            original_reasoning_result: Optional original reasoning result for comparison
            
        Returns:
            ContextValidationResult with scores and recommendations
        """
        
        logger.info("Starting context validation")
        start_time = datetime.now()
        
        try:
            # Run individual validation checks
            validation_checks = {}
            
            # 1. Engine Coverage Validation
            logger.debug("Validating engine coverage")
            validation_checks['engine_coverage'] = await self._check_engine_coverage(rich_context)
            
            # 2. Insight Preservation Validation
            logger.debug("Validating insight preservation")
            validation_checks['insight_preservation'] = await self._check_insight_preservation(
                rich_context, original_reasoning_result
            )
            
            # 3. Confidence Mapping Validation
            logger.debug("Validating confidence mapping")
            validation_checks['confidence_mapping'] = await self._check_confidence_mapping(rich_context)
            
            # 4. Analogical Richness Validation
            logger.debug("Validating analogical richness")
            validation_checks['analogical_richness'] = await self._check_analogical_richness(rich_context)
            
            # 5. Breakthrough Detection Validation
            logger.debug("Validating breakthrough detection")
            validation_checks['breakthrough_detection'] = await self._check_breakthrough_detection(rich_context)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(validation_checks)
            
            # Generate recommendations
            recommendations = self._generate_improvement_recommendations(validation_checks)
            
            # Identify missing components
            missing_components = self._identify_missing_components(rich_context)
            
            # Identify quality issues
            quality_issues = self._identify_quality_issues(validation_checks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ContextValidationResult(
                overall_score=overall_score,
                validation_checks=validation_checks,
                recommendations=recommendations,
                missing_components=missing_components,
                quality_issues=quality_issues,
                timestamp=datetime.now()
            )
            
            logger.info(f"Context validation completed in {processing_time:.2f}s with score {overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during context validation: {e}")
            return ContextValidationResult(
                overall_score=0.0,
                validation_checks={},
                recommendations=[f"Validation failed: {str(e)}"],
                missing_components=["Unable to determine due to validation error"],
                quality_issues=[f"Validation error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    # ========================================================================
    # Individual Validation Check Methods
    # ========================================================================
    
    async def _check_engine_coverage(self, rich_context: RichReasoningContext) -> float:
        """
        Check how well the context covers different reasoning engines.
        
        Returns:
            Score from 0.0 to 1.0 indicating engine coverage quality
        """
        
        # Count available engine insights
        available_engines = len(rich_context.engine_insights)
        total_possible_engines = len(EngineType)
        
        # Base coverage score
        coverage_ratio = available_engines / total_possible_engines
        
        # Quality assessment of available engine insights
        quality_scores = []
        for engine_type, insight in rich_context.engine_insights.items():
            quality_score = self._assess_engine_insight_quality(insight)
            quality_scores.append(quality_score)
        
        average_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Combined score: coverage ratio weighted with quality
        engine_coverage_score = (coverage_ratio * 0.6) + (average_quality * 0.4)
        
        logger.debug(f"Engine coverage: {available_engines}/{total_possible_engines} engines, "
                    f"quality: {average_quality:.2f}, score: {engine_coverage_score:.2f}")
        
        return min(1.0, engine_coverage_score)
    
    async def _check_insight_preservation(self, 
                                        rich_context: RichReasoningContext,
                                        original_reasoning_result: Optional[Any] = None) -> float:
        """
        Check how well key insights from original reasoning are preserved.
        
        Returns:
            Score from 0.0 to 1.0 indicating insight preservation quality
        """
        
        # If we don't have original result, assess based on richness
        if original_reasoning_result is None:
            return self._assess_insight_richness(rich_context)
        
        # Compare with original reasoning result
        preservation_score = 0.7  # Placeholder - would compare actual insights
        
        # Assess richness of preserved insights
        richness_score = self._assess_insight_richness(rich_context)
        
        # Combined score
        insight_preservation_score = (preservation_score * 0.6) + (richness_score * 0.4)
        
        logger.debug(f"Insight preservation score: {insight_preservation_score:.2f}")
        
        return insight_preservation_score
    
    async def _check_confidence_mapping(self, rich_context: RichReasoningContext) -> float:
        """
        Check how well confidence levels are mapped and analyzed.
        
        Returns:
            Score from 0.0 to 1.0 indicating confidence mapping quality
        """
        
        score_components = []
        
        # 1. Check if confidence analysis exists
        if rich_context.confidence_analysis:
            score_components.append(0.4)
        
        # 2. Check engine confidence levels
        if rich_context.engine_confidence_levels:
            confidence_coverage = len(rich_context.engine_confidence_levels) / len(rich_context.engine_insights)
            score_components.append(confidence_coverage * 0.3)
        
        # 3. Check uncertainty mapping
        if rich_context.uncertainty_mapping:
            score_components.append(0.3)
        
        confidence_mapping_score = sum(score_components)
        
        logger.debug(f"Confidence mapping score: {confidence_mapping_score:.2f}")
        
        return confidence_mapping_score
    
    async def _check_analogical_richness(self, rich_context: RichReasoningContext) -> float:
        """
        Check the richness of analogical connections and cross-domain insights.
        
        Returns:
            Score from 0.0 to 1.0 indicating analogical richness
        """
        
        score_components = []
        
        # 1. Number of analogical connections
        num_connections = len(rich_context.analogical_connections)
        if num_connections > 0:
            # Score based on number (diminishing returns)
            connection_score = min(1.0, num_connections / 5.0) * 0.4
            score_components.append(connection_score)
        
        # 2. Quality of analogical connections
        if rich_context.analogical_connections:
            quality_scores = [
                conn.connection_strength * conn.confidence 
                for conn in rich_context.analogical_connections
            ]
            average_quality = np.mean(quality_scores)
            score_components.append(average_quality * 0.3)
        
        # 3. Cross-domain bridges
        if rich_context.cross_domain_bridges:
            bridge_score = min(1.0, len(rich_context.cross_domain_bridges) / 3.0) * 0.3
            score_components.append(bridge_score)
        
        analogical_richness_score = sum(score_components)
        
        logger.debug(f"Analogical richness score: {analogical_richness_score:.2f}")
        
        return analogical_richness_score
    
    async def _check_breakthrough_detection(self, rich_context: RichReasoningContext) -> float:
        """
        Check how well breakthrough potential is detected and analyzed.
        
        Returns:
            Score from 0.0 to 1.0 indicating breakthrough detection quality
        """
        
        score_components = []
        
        # 1. Breakthrough analysis exists
        if rich_context.breakthrough_analysis:
            # Quality based on breakthrough score and category
            breakthrough_score = rich_context.breakthrough_analysis.overall_breakthrough_score
            score_components.append(breakthrough_score * 0.4)
        
        # 2. Emergent insights identified
        num_emergent = len(rich_context.emergent_insights)
        if num_emergent > 0:
            emergent_score = min(1.0, num_emergent / 3.0) * 0.3
            score_components.append(emergent_score)
        
        # 3. Novelty assessment
        if rich_context.novelty_assessment:
            novelty_score = rich_context.novelty_assessment.novelty_score * 0.3
            score_components.append(novelty_score)
        
        breakthrough_detection_score = sum(score_components)
        
        logger.debug(f"Breakthrough detection score: {breakthrough_detection_score:.2f}")
        
        return breakthrough_detection_score
    
    # ========================================================================
    # Utility Assessment Methods
    # ========================================================================
    
    def _assess_engine_insight_quality(self, insight: EngineInsight) -> float:
        """Assess the quality of an individual engine insight"""
        
        quality_factors = []
        
        # 1. Number of primary findings
        findings_score = min(1.0, len(insight.primary_findings) / 3.0)
        quality_factors.append(findings_score * 0.3)
        
        # 2. Confidence level
        quality_factors.append(insight.confidence_level * 0.3)
        
        # 3. Supporting evidence
        evidence_score = min(1.0, len(insight.supporting_evidence) / 2.0)
        quality_factors.append(evidence_score * 0.2)
        
        # 4. Reasoning trace depth
        trace_score = min(1.0, len(insight.reasoning_trace) / 5.0)
        quality_factors.append(trace_score * 0.2)
        
        return sum(quality_factors)
    
    def _assess_insight_richness(self, rich_context: RichReasoningContext) -> float:
        """Assess the overall richness of insights in the context"""
        
        richness_factors = []
        
        # 1. Total insights across engines
        total_findings = sum(len(insight.primary_findings) for insight in rich_context.engine_insights.values())
        findings_richness = min(1.0, total_findings / 10.0)
        richness_factors.append(findings_richness * 0.4)
        
        # 2. Synthesis patterns
        pattern_richness = min(1.0, len(rich_context.synthesis_patterns) / 3.0)
        richness_factors.append(pattern_richness * 0.3)
        
        # 3. Emergent insights
        emergent_richness = min(1.0, len(rich_context.emergent_insights) / 2.0)
        richness_factors.append(emergent_richness * 0.3)
        
        return sum(richness_factors)
    
    # ========================================================================
    # Overall Score and Recommendations
    # ========================================================================
    
    def _calculate_overall_score(self, validation_checks: Dict[str, float]) -> float:
        """Calculate weighted overall validation score"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for check_name, score in validation_checks.items():
            if check_name in self.validation_criteria:
                weight = self.validation_criteria[check_name]['weight']
                total_score += score * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 0.0
        
        return overall_score
    
    def _generate_improvement_recommendations(self, validation_checks: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving context aggregation"""
        
        recommendations = []
        
        for check_name, score in validation_checks.items():
            if check_name in self.validation_criteria:
                criteria = self.validation_criteria[check_name]
                min_score = criteria['min_score']
                
                if score < min_score:
                    recommendation = self._get_improvement_recommendation(check_name, score, min_score)
                    recommendations.append(recommendation)
        
        # Add general recommendations based on overall patterns
        low_scores = [name for name, score in validation_checks.items() if score < 0.5]
        if len(low_scores) >= 3:
            recommendations.append(
                "Multiple validation checks scored low. Consider reviewing the "
                "entire context aggregation pipeline for systematic improvements."
            )
        
        return recommendations
    
    def _get_improvement_recommendation(self, check_name: str, current_score: float, min_score: float) -> str:
        """Get specific improvement recommendation for a validation check"""
        
        gap = min_score - current_score
        
        recommendations = {
            'engine_coverage': {
                'small': "Consider implementing analyzers for missing reasoning engines.",
                'large': "Significant engine coverage gap detected. Prioritize implementing "
                        "all reasoning engine analyzers and ensure they extract meaningful insights."
            },
            'insight_preservation': {
                'small': "Enhance insight extraction to capture more nuanced findings.",
                'large': "Major insight preservation issue. Review extraction logic to ensure "
                        "key reasoning insights are not lost during aggregation."
            },
            'confidence_mapping': {
                'small': "Improve confidence analysis depth and uncertainty mapping.",
                'large': "Confidence mapping is inadequate. Implement comprehensive confidence "
                        "analysis and uncertainty quantification systems."
            },
            'analogical_richness': {
                'small': "Enhance analogical connection detection and cross-domain analysis.",
                'large': "Analogical reasoning insights are poorly captured. Strengthen "
                        "analogical analysis and cross-domain bridging detection."
            },
            'breakthrough_detection': {
                'small': "Improve breakthrough potential assessment and novelty detection.",
                'large': "Breakthrough detection is insufficient. Implement robust systems "
                        "for identifying emergent insights and paradigm-shifting potential."
            }
        }
        
        severity = 'large' if gap > 0.3 else 'small'
        
        if check_name in recommendations:
            return recommendations[check_name][severity]
        else:
            return f"Improve {check_name} (current: {current_score:.2f}, target: {min_score:.2f})"
    
    def _identify_missing_components(self, rich_context: RichReasoningContext) -> List[str]:
        """Identify missing components in the rich context"""
        
        missing = []
        
        # Check for missing major components
        if not rich_context.engine_insights:
            missing.append("Engine insights")
        
        if not rich_context.confidence_analysis:
            missing.append("Confidence analysis")
        
        if not rich_context.breakthrough_analysis:
            missing.append("Breakthrough analysis")
        
        if not rich_context.analogical_connections:
            missing.append("Analogical connections")
        
        if not rich_context.synthesis_patterns:
            missing.append("Synthesis patterns")
        
        if not rich_context.reasoning_quality:
            missing.append("Reasoning quality assessment")
        
        # Check for missing engine types
        available_engines = set(rich_context.engine_insights.keys())
        all_engines = set(EngineType)
        missing_engines = all_engines - available_engines
        
        if missing_engines:
            missing.append(f"Engine insights for: {', '.join([e.value for e in missing_engines])}")
        
        return missing
    
    def _identify_quality_issues(self, validation_checks: Dict[str, float]) -> List[str]:
        """Identify quality issues based on validation check results"""
        
        issues = []
        
        # Identify issues based on low scores
        for check_name, score in validation_checks.items():
            if score < 0.3:
                issues.append(f"Severe quality issue in {check_name} (score: {score:.2f})")
            elif score < 0.5:
                issues.append(f"Quality concern in {check_name} (score: {score:.2f})")
        
        # Check for specific patterns
        if validation_checks.get('engine_coverage', 0) < 0.4:
            issues.append("Insufficient reasoning engine coverage may lead to incomplete analysis")
        
        if validation_checks.get('insight_preservation', 0) < 0.6:
            issues.append("Poor insight preservation may result in loss of key reasoning insights")
        
        return issues


# ============================================================================
# Context Validation Utilities
# ============================================================================

class ContextQualityMetrics:
    """Utility class for calculating context quality metrics"""
    
    @staticmethod
    def calculate_insight_density(rich_context: RichReasoningContext) -> float:
        """Calculate insight density (insights per engine)"""
        
        if not rich_context.engine_insights:
            return 0.0
        
        total_insights = sum(len(insight.primary_findings) for insight in rich_context.engine_insights.values())
        return total_insights / len(rich_context.engine_insights)
    
    @staticmethod
    def calculate_confidence_variance(rich_context: RichReasoningContext) -> float:
        """Calculate variance in confidence levels across engines"""
        
        if len(rich_context.engine_confidence_levels) < 2:
            return 0.0
        
        confidence_values = list(rich_context.engine_confidence_levels.values())
        return np.var(confidence_values)
    
    @staticmethod
    def calculate_synthesis_complexity(rich_context: RichReasoningContext) -> float:
        """Calculate complexity of synthesis patterns"""
        
        if not rich_context.synthesis_patterns:
            return 0.0
        
        complexity_score = 0.0
        for pattern in rich_context.synthesis_patterns:
            # More participating engines = higher complexity
            engine_complexity = len(pattern.participating_engines) / len(EngineType)
            # Pattern strength contributes to complexity
            strength_factor = pattern.strength
            complexity_score += engine_complexity * strength_factor
        
        return complexity_score / len(rich_context.synthesis_patterns)


class ContextCompletionSuggestions:
    """Utility class for suggesting context completion improvements"""
    
    @staticmethod
    def suggest_missing_engine_implementations(rich_context: RichReasoningContext) -> List[str]:
        """Suggest which engine analyzers need implementation"""
        
        available_engines = set(rich_context.engine_insights.keys())
        all_engines = set(EngineType)
        missing_engines = all_engines - available_engines
        
        suggestions = []
        for engine in missing_engines:
            suggestions.append(f"Implement {engine.value} context analyzer for richer insights")
        
        return suggestions
    
    @staticmethod
    def suggest_quality_improvements(validation_result: ContextValidationResult) -> List[str]:
        """Suggest specific quality improvements based on validation results"""
        
        suggestions = []
        
        # Analyze validation scores for specific suggestions
        checks = validation_result.validation_checks
        
        if checks.get('analogical_richness', 0) < 0.5:
            suggestions.append("Enhance analogical reasoning analysis to capture cross-domain insights")
        
        if checks.get('breakthrough_detection', 0) < 0.5:
            suggestions.append("Strengthen breakthrough potential assessment algorithms")
        
        if checks.get('confidence_mapping', 0) < 0.6:
            suggestions.append("Implement more sophisticated confidence and uncertainty analysis")
        
        return suggestions
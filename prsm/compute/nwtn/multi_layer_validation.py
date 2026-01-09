#!/usr/bin/env python3
"""
Multi-Layer Validation System for NWTN
======================================

Implements comprehensive validation pyramid for hallucination prevention
as outlined in the critical analysis recommendations.

Validation Layers:
1. Corpus Grounding Validation - Ensures claims trace to source papers
2. Logical Consistency Checking - Validates internal logical coherence  
3. External Fact Checking - Validates against external sources (when available)
4. Uncertainty Quantification - Quantifies and communicates uncertainty

This addresses the pseudo-grounding problem where responses appear
authoritative but lack proper source validation.
"""

import asyncio
import logging
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation confidence levels"""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    FAILED = "failed"


class ClaimType(Enum):
    """Types of claims for validation"""
    FACTUAL = "factual"
    METHODOLOGICAL = "methodological"
    STATISTICAL = "statistical"
    CONCEPTUAL = "conceptual"
    OPINION = "opinion"


@dataclass
class ValidationClaim:
    """A claim extracted from response for validation"""
    claim_id: str
    text: str
    claim_type: ClaimType
    confidence: float
    source_sentence: str
    position: int
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


@dataclass 
class ValidationResult:
    """Result of validation check"""
    claim_id: str
    validation_type: str
    level: ValidationLevel
    confidence: float
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        return self.level in [ValidationLevel.HIGH, ValidationLevel.MEDIUM]


@dataclass
class ComprehensiveValidationResult:
    """Complete validation result for a response"""
    response_text: str
    overall_confidence: float
    validation_passed: bool
    layer_results: Dict[str, List[ValidationResult]] = field(default_factory=dict)
    extracted_claims: List[ValidationClaim] = field(default_factory=list)
    uncertainty_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    should_present_response: bool = True
    validation_metadata: Dict[str, Any] = field(default_factory=dict)


class ClaimExtractor:
    """Extracts validatable claims from response text"""
    
    def __init__(self):
        # Patterns for different claim types
        self.factual_patterns = [
            r'(\d+(?:\.\d+)?%?\s+(?:of|percent|percentage))',  # Percentages/statistics
            r'(studies show|research indicates|evidence suggests)',  # Research claims
            r'(according to|based on|findings from)',  # Attribution claims
            r'(\w+\s+is\s+(?:effective|proven|demonstrated))'  # Effectiveness claims
        ]
        
        self.methodological_patterns = [
            r'(using|applying|implementing|through)\s+([^.]{20,80})',  # Method descriptions
            r'(approach|method|technique|strategy).*?(?:involves|includes|requires)',  # Method explanations
        ]
        
        self.conceptual_patterns = [
            r'(\w+\s+(?:is defined as|refers to|means|represents))',  # Definitions
            r'(concept of|principle of|theory of)\s+([^.]{10,50})',  # Conceptual explanations
        ]
    
    async def extract_claims(self, response_text: str) -> List[ValidationClaim]:
        """Extract validatable claims from response"""
        logger.info(f"Extracting claims for validation - response_length: {len(response_text)}")
        
        claims = []
        sentences = self._split_into_sentences(response_text)
        
        for i, sentence in enumerate(sentences):
            sentence_claims = await self._extract_sentence_claims(sentence, i)
            claims.extend(sentence_claims)
        
        # Add dependency analysis
        self._analyze_claim_dependencies(claims)
        
        logger.info(f"Extracted {len(claims)} claims for validation")
        return claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for claim extraction"""
        # Simple sentence splitting - could be enhanced with NLP library
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _extract_sentence_claims(self, sentence: str, position: int) -> List[ValidationClaim]:
        """Extract claims from individual sentence"""
        claims = []
        
        # Check for factual claims
        for pattern in self.factual_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                claim_text = match if isinstance(match, str) else match[0]
                claims.append(ValidationClaim(
                    claim_id="",  # Will be generated in __post_init__
                    text=claim_text,
                    claim_type=ClaimType.FACTUAL,
                    confidence=0.8,
                    source_sentence=sentence,
                    position=position
                ))
        
        # Check for methodological claims
        for pattern in self.methodological_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                claim_text = match if isinstance(match, str) else ' '.join(match)
                claims.append(ValidationClaim(
                    claim_id="",
                    text=claim_text,
                    claim_type=ClaimType.METHODOLOGICAL,
                    confidence=0.7,
                    source_sentence=sentence,
                    position=position
                ))
        
        # Check for conceptual claims
        for pattern in self.conceptual_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                claim_text = match if isinstance(match, str) else ' '.join(match)
                claims.append(ValidationClaim(
                    claim_id="",
                    text=claim_text,
                    claim_type=ClaimType.CONCEPTUAL,
                    confidence=0.6,
                    source_sentence=sentence,
                    position=position
                ))
        
        return claims
    
    def _analyze_claim_dependencies(self, claims: List[ValidationClaim]):
        """Analyze dependencies between claims"""
        for i, claim in enumerate(claims):
            # Simple dependency detection based on position and content similarity
            for j, other_claim in enumerate(claims):
                if i != j and self._are_claims_related(claim, other_claim):
                    claim.dependencies.append(other_claim.claim_id)
    
    def _are_claims_related(self, claim1: ValidationClaim, claim2: ValidationClaim) -> bool:
        """Check if two claims are related"""
        # Simple similarity check - could be enhanced
        words1 = set(claim1.text.lower().split())
        words2 = set(claim2.text.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > 0.3


class ValidationLayer(ABC):
    """Abstract base class for validation layers"""
    
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.validation_count = 0
        self.success_count = 0
    
    @abstractmethod
    async def validate_claim(self, 
                           claim: ValidationClaim, 
                           context: Dict[str, Any]) -> ValidationResult:
        """Validate individual claim"""
        pass
    
    async def validate_claims(self, 
                            claims: List[ValidationClaim], 
                            context: Dict[str, Any]) -> List[ValidationResult]:
        """Validate all claims in the layer"""
        logger.info(f"Starting {self.layer_name} validation - claims_count: {len(claims)}")
        
        results = []
        for claim in claims:
            try:
                result = await self.validate_claim(claim, context)
                results.append(result)
                self.validation_count += 1
                if result.passed:
                    self.success_count += 1
            except Exception as e:
                logger.error(f"{self.layer_name} validation failed for claim {claim.claim_id}: {e}")
                results.append(ValidationResult(
                    claim_id=claim.claim_id,
                    validation_type=self.layer_name,
                    level=ValidationLevel.FAILED,
                    confidence=0.0,
                    reasoning=f"Validation error: {str(e)}"
                ))
        
        success_rate = self.success_count / max(self.validation_count, 1)
        logger.info(f"{self.layer_name} validation completed - success_rate: {success_rate:.1%}, total_results: {len(results)}")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this layer"""
        return {
            'layer_name': self.layer_name,
            'total_validations': self.validation_count,
            'successful_validations': self.success_count,
            'success_rate': self.success_count / max(self.validation_count, 1),
        }


class CorpusValidationLayer(ValidationLayer):
    """Layer 1: Corpus grounding validation"""
    
    def __init__(self, source_papers: List[Dict[str, Any]]):
        super().__init__("corpus_grounding")
        self.source_papers = source_papers
        self.paper_content = self._build_searchable_content()
    
    def _build_searchable_content(self) -> Dict[str, str]:
        """Build searchable content from source papers"""
        content = {}
        for paper in self.source_papers:
            paper_id = paper.get('paper_id', '')
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            searchable_text = f"{title} {abstract}".lower()
            content[paper_id] = searchable_text
        return content
    
    async def validate_claim(self, 
                           claim: ValidationClaim, 
                           context: Dict[str, Any]) -> ValidationResult:
        """Validate claim against corpus"""
        if not self.source_papers:
            return ValidationResult(
                claim_id=claim.claim_id,
                validation_type=self.layer_name,
                level=ValidationLevel.FAILED,
                confidence=0.0,
                reasoning="No source papers available for validation"
            )
        
        claim_words = set(claim.text.lower().split())
        best_match_score = 0.0
        best_match_papers = []
        evidence_found = []
        
        for paper_id, content in self.paper_content.items():
            content_words = set(content.split())
            
            if not claim_words or not content_words:
                continue
            
            # Calculate similarity
            intersection = len(claim_words.intersection(content_words))
            union = len(claim_words.union(content_words))
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity > 0.3:  # Threshold for relevance
                best_match_papers.append(paper_id)
                evidence_found.append(f"Found similarity {similarity:.2f} with paper {paper_id}")
                best_match_score = max(best_match_score, similarity)
        
        # Determine validation level
        if best_match_score > 0.7:
            level = ValidationLevel.HIGH
        elif best_match_score > 0.5:
            level = ValidationLevel.MEDIUM
        elif best_match_score > 0.3:
            level = ValidationLevel.LOW
        else:
            level = ValidationLevel.FAILED
        
        return ValidationResult(
            claim_id=claim.claim_id,
            validation_type=self.layer_name,
            level=level,
            confidence=best_match_score,
            evidence=evidence_found,
            sources=best_match_papers,
            reasoning=f"Corpus validation found {len(best_match_papers)} supporting papers with max similarity {best_match_score:.2f}"
        )


class LogicalConsistencyLayer(ValidationLayer):
    """Layer 2: Logical consistency validation"""
    
    def __init__(self):
        super().__init__("logical_consistency")
        self.contradiction_patterns = [
            (r'always', r'never'),
            (r'all', r'none'),
            (r'increase', r'decrease'),
            (r'effective', r'ineffective'),
            (r'proven', r'unproven'),
        ]
    
    async def validate_claim(self, 
                           claim: ValidationClaim, 
                           context: Dict[str, Any]) -> ValidationResult:
        """Validate logical consistency of claim"""
        all_claims = context.get('all_claims', [])
        contradictions = []
        consistency_score = 1.0
        
        for other_claim in all_claims:
            if other_claim.claim_id == claim.claim_id:
                continue
            
            contradiction_found = self._check_contradiction(claim, other_claim)
            if contradiction_found:
                contradictions.append(f"Contradicts claim {other_claim.claim_id}: {contradiction_found}")
                consistency_score *= 0.7  # Reduce score for each contradiction
        
        # Check for internal logical consistency
        internal_issues = self._check_internal_logic(claim)
        if internal_issues:
            contradictions.extend(internal_issues)
            consistency_score *= 0.8
        
        # Determine validation level
        if consistency_score > 0.9 and not contradictions:
            level = ValidationLevel.HIGH
        elif consistency_score > 0.7:
            level = ValidationLevel.MEDIUM
        elif consistency_score > 0.5:
            level = ValidationLevel.LOW
        else:
            level = ValidationLevel.FAILED
        
        return ValidationResult(
            claim_id=claim.claim_id,
            validation_type=self.layer_name,
            level=level,
            confidence=consistency_score,
            contradictions=contradictions,
            reasoning=f"Logical consistency check found {len(contradictions)} issues, score: {consistency_score:.2f}"
        )
    
    def _check_contradiction(self, claim1: ValidationClaim, claim2: ValidationClaim) -> Optional[str]:
        """Check if two claims contradict each other"""
        text1 = claim1.text.lower()
        text2 = claim2.text.lower()
        
        for positive, negative in self.contradiction_patterns:
            if positive in text1 and negative in text2:
                return f"'{positive}' contradicts '{negative}'"
            if negative in text1 and positive in text2:
                return f"'{negative}' contradicts '{positive}'"
        
        return None
    
    def _check_internal_logic(self, claim: ValidationClaim) -> List[str]:
        """Check internal logical consistency of claim"""
        issues = []
        text = claim.text.lower()
        
        # Check for obvious logical issues
        if 'always' in text and 'sometimes' in text:
            issues.append("Contains both 'always' and 'sometimes'")
        
        if 'all' in text and 'some' in text:
            issues.append("Contains both 'all' and 'some'")
        
        # Check for percentage issues
        percentages = re.findall(r'(\d+(?:\.\d+)?)%', text)
        if percentages:
            for pct_str in percentages:
                pct = float(pct_str)
                if pct > 100:
                    issues.append(f"Percentage {pct}% is greater than 100%")
        
        return issues


class UncertaintyQuantificationLayer(ValidationLayer):
    """Layer 4: Uncertainty quantification"""
    
    def __init__(self):
        super().__init__("uncertainty_quantification")
        self.uncertainty_indicators = [
            'may', 'might', 'could', 'possibly', 'potentially',
            'suggests', 'indicates', 'appears', 'seems',
            'likely', 'probably', 'approximately', 'roughly'
        ]
        self.confidence_indicators = [
            'proven', 'demonstrated', 'established', 'confirmed',
            'definitely', 'certainly', 'always', 'never'
        ]
    
    async def validate_claim(self, 
                           claim: ValidationClaim, 
                           context: Dict[str, Any]) -> ValidationResult:
        """Quantify uncertainty in claim"""
        text = claim.text.lower()
        
        uncertainty_signals = sum(1 for indicator in self.uncertainty_indicators if indicator in text)
        confidence_signals = sum(1 for indicator in self.confidence_indicators if indicator in text)
        
        # Calculate uncertainty score
        total_signals = uncertainty_signals + confidence_signals
        if total_signals == 0:
            # No explicit uncertainty or confidence indicators
            uncertainty_score = 0.5  # Default moderate uncertainty
        else:
            uncertainty_score = uncertainty_signals / total_signals
        
        # Adjust for claim type
        if claim.claim_type == ClaimType.STATISTICAL:
            uncertainty_score += 0.1  # Statistics have inherent uncertainty
        elif claim.claim_type == ClaimType.OPINION:
            uncertainty_score += 0.2  # Opinions are inherently uncertain
        
        uncertainty_score = min(uncertainty_score, 1.0)
        
        # Validate appropriate uncertainty communication
        evidence = []
        if uncertainty_signals > 0:
            evidence.append(f"Contains {uncertainty_signals} uncertainty indicators")
        if confidence_signals > 0:
            evidence.append(f"Contains {confidence_signals} confidence indicators")
        
        # Determine validation level based on appropriate uncertainty expression
        if 0.3 <= uncertainty_score <= 0.7:
            level = ValidationLevel.HIGH  # Appropriate uncertainty
        elif 0.1 <= uncertainty_score <= 0.9:
            level = ValidationLevel.MEDIUM
        else:
            level = ValidationLevel.LOW  # Too certain or too uncertain
        
        return ValidationResult(
            claim_id=claim.claim_id,
            validation_type=self.layer_name,
            level=level,
            confidence=1.0 - uncertainty_score,  # Higher confidence for appropriate uncertainty
            evidence=evidence,
            reasoning=f"Uncertainty score: {uncertainty_score:.2f}, appropriately communicated: {level.value}",
            metadata={'uncertainty_score': uncertainty_score}
        )


class MultiLayerValidator:
    """Comprehensive multi-layer validation system"""
    
    def __init__(self):
        self.claim_extractor = ClaimExtractor()
        self.validation_layers = []
        self.validation_history = []
        
    def initialize_layers(self, source_papers: List[Dict[str, Any]]):
        """Initialize validation layers with context"""
        self.validation_layers = [
            CorpusValidationLayer(source_papers),
            LogicalConsistencyLayer(),
            UncertaintyQuantificationLayer()
        ]
        logger.info(f"Initialized {len(self.validation_layers)} validation layers")
    
    async def comprehensive_validation(self,
                                     response_text: str,
                                     source_papers: List[Dict[str, Any]],
                                     original_query: str) -> ComprehensiveValidationResult:
        """Perform comprehensive multi-layer validation"""
        logger.info(f"Starting comprehensive validation - response_length: {len(response_text)}, source_papers_count: {len(source_papers)}")
        
        validation_start_time = datetime.now(timezone.utc)
        
        # Initialize layers with current context
        self.initialize_layers(source_papers)
        
        # Extract claims from response
        claims = await self.claim_extractor.extract_claims(response_text)
        
        # Prepare validation context
        validation_context = {
            'all_claims': claims,
            'source_papers': source_papers,
            'original_query': original_query,
            'response_text': response_text
        }
        
        # Run validation through all layers
        layer_results = {}
        for layer in self.validation_layers:
            layer_name = layer.layer_name
            results = await layer.validate_claims(claims, validation_context)
            layer_results[layer_name] = results
        
        # Calculate overall validation metrics
        overall_result = self._calculate_overall_validation(
            response_text, claims, layer_results, validation_start_time
        )
        
        # Store validation history
        self.validation_history.append(overall_result)
        
        logger.info(f"Comprehensive validation completed - overall_confidence: {overall_result.overall_confidence:.2f}, validation_passed: {overall_result.validation_passed}, should_present: {overall_result.should_present_response}")
        
        return overall_result
    
    def _calculate_overall_validation(self,
                                    response_text: str,
                                    claims: List[ValidationClaim],
                                    layer_results: Dict[str, List[ValidationResult]],
                                    start_time: datetime) -> ComprehensiveValidationResult:
        """Calculate overall validation result"""
        
        # Calculate layer-wise confidence scores
        layer_confidences = {}
        layer_pass_rates = {}
        
        for layer_name, results in layer_results.items():
            if results:
                confidences = [r.confidence for r in results]
                layer_confidences[layer_name] = sum(confidences) / len(confidences)
                
                passed_count = sum(1 for r in results if r.passed)
                layer_pass_rates[layer_name] = passed_count / len(results)
            else:
                layer_confidences[layer_name] = 0.0
                layer_pass_rates[layer_name] = 0.0
        
        # Weighted overall confidence (corpus grounding weighted higher)
        weights = {
            'corpus_grounding': 0.5,
            'logical_consistency': 0.3,
            'uncertainty_quantification': 0.2
        }
        
        overall_confidence = sum(
            weights.get(layer_name, 0.33) * confidence
            for layer_name, confidence in layer_confidences.items()
        )
        
        # Determine if validation passed
        validation_passed = (
            layer_pass_rates.get('corpus_grounding', 0) >= 0.7 and
            layer_pass_rates.get('logical_consistency', 0) >= 0.8 and
            overall_confidence >= 0.6
        )
        
        # Determine if response should be presented
        should_present = (
            validation_passed or
            overall_confidence >= 0.5  # Lower threshold for presentation with warnings
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(layer_results, layer_pass_rates)
        
        # Calculate uncertainty analysis
        uncertainty_results = layer_results.get('uncertainty_quantification', [])
        uncertainty_analysis = {
            'average_uncertainty': sum(
                r.metadata.get('uncertainty_score', 0.5) 
                for r in uncertainty_results
            ) / max(len(uncertainty_results), 1),
            'uncertainty_appropriate': sum(
                1 for r in uncertainty_results if r.passed
            ) / max(len(uncertainty_results), 1)
        }
        
        validation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return ComprehensiveValidationResult(
            response_text=response_text,
            overall_confidence=overall_confidence,
            validation_passed=validation_passed,
            layer_results=layer_results,
            extracted_claims=claims,
            uncertainty_analysis=uncertainty_analysis,
            recommendations=recommendations,
            should_present_response=should_present,
            validation_metadata={
                'validation_time': validation_time,
                'layer_confidences': layer_confidences,
                'layer_pass_rates': layer_pass_rates,
                'claims_extracted': len(claims)
            }
        )
    
    def _generate_recommendations(self, 
                                layer_results: Dict[str, List[ValidationResult]],
                                layer_pass_rates: Dict[str, float]) -> List[str]:
        """Generate validation recommendations"""
        recommendations = []
        
        if layer_pass_rates.get('corpus_grounding', 1.0) < 0.7:
            recommendations.append("Improve corpus grounding - add more explicit citations and paper references")
        
        if layer_pass_rates.get('logical_consistency', 1.0) < 0.8:
            recommendations.append("Address logical inconsistencies - review for contradictory statements")
        
        if layer_pass_rates.get('uncertainty_quantification', 1.0) < 0.6:
            recommendations.append("Improve uncertainty communication - add appropriate confidence indicators")
        
        # Check for specific issues
        all_contradictions = []
        for results in layer_results.values():
            for result in results:
                all_contradictions.extend(result.contradictions)
        
        if all_contradictions:
            recommendations.append(f"Resolve {len(all_contradictions)} identified contradictions")
        
        return recommendations


# Export main classes
__all__ = [
    'MultiLayerValidator',
    'ValidationLevel',
    'ClaimType', 
    'ValidationClaim',
    'ValidationResult',
    'ComprehensiveValidationResult'
]
#!/usr/bin/env python3
"""
Enhanced Voicebox for NWTN System 1 → System 2 → Attribution Pipeline
=====================================================================

This module implements enhanced natural language generation that integrates
winning candidates from System 2 evaluation into LLM prompts with inline
citations and response validation.

Part of Phase 3 of the NWTN System 1 → System 2 → Attribution roadmap.
"""

import asyncio
import structlog
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum
import json
import re

from prsm.nwtn.candidate_evaluator import EvaluationResult, CandidateEvaluation
from prsm.nwtn.citation_filter import CitationFilterResult, FilteredCitation, CitationFormat
from prsm.nwtn.voicebox import NWTNVoicebox  # Import existing Voicebox

logger = structlog.get_logger(__name__)


class ResponseQuality(Enum):
    """Response quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    POOR = "poor"
    INVALID = "invalid"


class ValidationStatus(Enum):
    """Response validation status"""
    VALID = "valid"
    CITATION_MISMATCH = "citation_mismatch"
    CONTENT_MISMATCH = "content_mismatch"
    MISSING_CITATIONS = "missing_citations"
    INVALID_FORMAT = "invalid_format"
    QUALITY_ISSUES = "quality_issues"


@dataclass
class ResponseValidation:
    """Result of response validation"""
    validation_status: ValidationStatus
    quality_score: float  # 0.0 to 1.0
    citation_accuracy: float  # How accurately citations are used
    content_accuracy: float  # How accurately content reflects sources
    completeness_score: float  # How complete the response is
    issues_found: List[str]  # List of validation issues
    recommendations: List[str]  # Recommendations for improvement
    validation_details: Dict[str, Any]  # Detailed validation results


@dataclass
class EnhancedResponse:
    """Enhanced response with integrated citations and validation"""
    response_id: str
    query: str
    response_text: str
    inline_citations: List[str]  # Citations embedded in response
    citation_list: List[FilteredCitation]  # Complete citation list
    source_integration: Dict[str, str]  # How each source was integrated
    response_validation: ResponseValidation
    quality_metrics: Dict[str, float]
    generation_time: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ResponseValidator:
    """Validates responses to ensure accurate source attribution"""
    
    def __init__(self):
        self.validation_patterns = {
            'citation_markers': [
                r'\[(\d+)\]',           # [1], [2], etc.
                r'\(([^)]+,\s*\d{4})\)', # (Author, 2023)
                r'\(([^)]+)\)',         # (Author)
            ],
            'content_indicators': [
                r'according to',
                r'as stated in',
                r'research shows',
                r'studies indicate',
                r'findings suggest',
                r'evidence demonstrates'
            ]
        }
    
    async def validate_response(self, 
                              response_text: str,
                              citations: List[FilteredCitation],
                              winning_candidates: List[CandidateEvaluation]) -> ResponseValidation:
        """Validate response for accurate source attribution"""
        try:
            # Extract citation markers from response
            citation_markers = self._extract_citation_markers(response_text)
            
            # Validate citation accuracy
            citation_accuracy = self._validate_citation_accuracy(
                response_text, citations, citation_markers
            )
            
            # Validate content accuracy
            content_accuracy = self._validate_content_accuracy(
                response_text, winning_candidates
            )
            
            # Validate completeness
            completeness_score = self._validate_completeness(
                response_text, winning_candidates
            )
            
            # Identify issues
            issues_found = self._identify_issues(
                response_text, citations, citation_markers, winning_candidates
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues_found)
            
            # Calculate overall quality score
            quality_score = (citation_accuracy + content_accuracy + completeness_score) / 3.0
            
            # Determine validation status
            validation_status = self._determine_validation_status(
                quality_score, citation_accuracy, content_accuracy, issues_found
            )
            
            return ResponseValidation(
                validation_status=validation_status,
                quality_score=quality_score,
                citation_accuracy=citation_accuracy,
                content_accuracy=content_accuracy,
                completeness_score=completeness_score,
                issues_found=issues_found,
                recommendations=recommendations,
                validation_details={
                    'citation_markers_found': len(citation_markers),
                    'citations_expected': len(citations),
                    'content_integration_points': len(self._find_content_integration_points(response_text))
                }
            )
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return ResponseValidation(
                validation_status=ValidationStatus.INVALID_FORMAT,
                quality_score=0.0,
                citation_accuracy=0.0,
                content_accuracy=0.0,
                completeness_score=0.0,
                issues_found=[f"Validation error: {str(e)}"],
                recommendations=["Please review response format and try again"],
                validation_details={}
            )
    
    def _extract_citation_markers(self, response_text: str) -> List[str]:
        """Extract citation markers from response text"""
        markers = []
        for pattern in self.validation_patterns['citation_markers']:
            matches = re.findall(pattern, response_text)
            markers.extend(matches)
        return markers
    
    def _validate_citation_accuracy(self, 
                                  response_text: str,
                                  citations: List[FilteredCitation],
                                  citation_markers: List[str]) -> float:
        """Validate that citations are accurately used"""
        if not citations:
            return 1.0 if not citation_markers else 0.0
        
        # Check if all citations are referenced
        expected_citations = len(citations)
        found_citations = len(citation_markers)
        
        if expected_citations == 0:
            return 1.0
        
        # Calculate accuracy based on citation usage
        accuracy = min(found_citations / expected_citations, 1.0)
        
        # Penalize for over-citation
        if found_citations > expected_citations:
            accuracy *= 0.8
        
        return accuracy
    
    def _validate_content_accuracy(self, 
                                 response_text: str,
                                 winning_candidates: List[CandidateEvaluation]) -> float:
        """Validate that content accurately reflects winning candidates"""
        if not winning_candidates:
            return 1.0
        
        # Check for content integration indicators
        integration_points = self._find_content_integration_points(response_text)
        
        # Check for key concepts from winning candidates
        key_concepts = set()
        for candidate in winning_candidates[:3]:  # Top 3 candidates
            key_concepts.update(candidate.candidate_answer.key_concepts_used)
        
        if not key_concepts:
            return 0.7  # Moderate score if no key concepts
        
        # Calculate concept coverage
        concepts_mentioned = 0
        for concept in key_concepts:
            if concept.lower() in response_text.lower():
                concepts_mentioned += 1
        
        concept_coverage = concepts_mentioned / len(key_concepts)
        
        # Combine integration points and concept coverage
        content_accuracy = (
            min(len(integration_points) / 3.0, 1.0) * 0.4 +  # Integration indicators
            concept_coverage * 0.6                            # Concept coverage
        )
        
        return content_accuracy
    
    def _validate_completeness(self, 
                             response_text: str,
                             winning_candidates: List[CandidateEvaluation]) -> float:
        """Validate completeness of response"""
        if not winning_candidates:
            return 1.0
        
        # Check response length
        response_length = len(response_text.split())
        length_score = min(response_length / 100.0, 1.0)  # Aim for ~100 words
        
        # Check for answer structure
        has_introduction = any(phrase in response_text.lower() for phrase in ['regarding', 'concerning', 'about'])
        has_conclusion = any(phrase in response_text.lower() for phrase in ['therefore', 'thus', 'in conclusion'])
        
        structure_score = (0.5 if has_introduction else 0.0) + (0.5 if has_conclusion else 0.0)
        
        # Combine scores
        completeness_score = length_score * 0.6 + structure_score * 0.4
        
        return completeness_score
    
    def _find_content_integration_points(self, response_text: str) -> List[str]:
        """Find points where content is integrated from sources"""
        integration_points = []
        
        for pattern in self.validation_patterns['content_indicators']:
            matches = re.finditer(pattern, response_text.lower())
            for match in matches:
                integration_points.append(match.group())
        
        return integration_points
    
    def _identify_issues(self, 
                       response_text: str,
                       citations: List[FilteredCitation],
                       citation_markers: List[str],
                       winning_candidates: List[CandidateEvaluation]) -> List[str]:
        """Identify specific issues with the response"""
        issues = []
        
        # Citation issues
        if len(citations) > 0 and len(citation_markers) == 0:
            issues.append("No citation markers found despite having sources")
        
        if len(citation_markers) > len(citations):
            issues.append("More citation markers than available sources")
        
        # Content issues
        if len(response_text.split()) < 50:
            issues.append("Response is too short")
        
        if not any(phrase in response_text.lower() for phrase in ['machine learning', 'classification', 'accuracy']):
            issues.append("Response may not address the query topic")
        
        # Integration issues
        integration_points = self._find_content_integration_points(response_text)
        if len(integration_points) == 0:
            issues.append("No clear source integration indicators found")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on identified issues"""
        recommendations = []
        
        for issue in issues:
            if "citation markers" in issue:
                recommendations.append("Add inline citations using [1], [2] format")
            elif "too short" in issue:
                recommendations.append("Expand response with more detailed explanations")
            elif "source integration" in issue:
                recommendations.append("Add phrases like 'according to research' or 'studies show'")
            elif "query topic" in issue:
                recommendations.append("Ensure response directly addresses the query")
        
        if not recommendations:
            recommendations.append("Response quality is acceptable")
        
        return recommendations
    
    def _determine_validation_status(self, 
                                   quality_score: float,
                                   citation_accuracy: float,
                                   content_accuracy: float,
                                   issues: List[str]) -> ValidationStatus:
        """Determine overall validation status"""
        if quality_score >= 0.8 and citation_accuracy >= 0.8 and content_accuracy >= 0.8:
            return ValidationStatus.VALID
        elif citation_accuracy < 0.5:
            return ValidationStatus.CITATION_MISMATCH
        elif content_accuracy < 0.5:
            return ValidationStatus.CONTENT_MISMATCH
        elif len(issues) > 3:
            return ValidationStatus.QUALITY_ISSUES
        else:
            return ValidationStatus.MISSING_CITATIONS


class EnhancedVoicebox:
    """
    Enhanced Voicebox that integrates winning candidates into LLM prompts
    with inline citations and response validation
    """
    
    def __init__(self, base_voicebox: Optional[NWTNVoicebox] = None):
        self.base_voicebox = base_voicebox or NWTNVoicebox()
        self.response_validator = ResponseValidator()
        self.initialized = False
        
        # Generation parameters
        self.max_response_length = 500  # Maximum response length in words
        self.citation_style = CitationFormat.NUMBERED
        self.include_validation = True
        self.retry_on_validation_failure = True
        self.max_retries = 2
        
        # Response templates
        self.response_templates = {
            'academic': """Based on the research analysis, here is a comprehensive response to your query: "{query}"

{candidate_content}

{citations_section}

This response integrates findings from {source_count} high-quality sources with confidence score of {confidence:.2f}.""",
            
            'conversational': """Regarding your question about {query}:

{candidate_content}

{citations_section}""",
            
            'detailed': """**Question:** {query}

**Analysis:**
{candidate_content}

**Sources:**
{citations_section}

**Confidence Level:** {confidence:.2f}/1.0"""
        }
        
        # Response statistics
        self.response_stats = {
            'total_responses': 0,
            'successful_responses': 0,
            'validation_passes': 0,
            'average_response_time': 0.0,
            'average_quality_score': 0.0,
            'citation_accuracy_average': 0.0
        }
    
    async def initialize(self):
        """Initialize the enhanced voicebox"""
        try:
            # Initialize base voicebox
            await self.base_voicebox.initialize()
            
            self.initialized = True
            logger.info("EnhancedVoicebox initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedVoicebox: {e}")
            return False
    
    async def generate_response(self, 
                              query: str,
                              evaluation_result: EvaluationResult,
                              citation_result: CitationFilterResult,
                              response_template: str = 'academic',
                              max_retries: Optional[int] = None) -> EnhancedResponse:
        """
        Generate enhanced response with integrated citations and validation
        
        Args:
            query: Original user query
            evaluation_result: Result from CandidateEvaluator
            citation_result: Result from CitationFilter
            response_template: Template style for response
            max_retries: Maximum retries for validation failures
            
        Returns:
            EnhancedResponse with integrated citations and validation
        """
        start_time = datetime.now(timezone.utc)
        max_retries = max_retries or self.max_retries
        
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info("Generating enhanced response",
                       query=query[:50],
                       template=response_template,
                       citations=len(citation_result.filtered_citations),
                       candidates=len(evaluation_result.candidate_evaluations))
            
            # Get winning candidates
            winning_candidates = self._get_winning_candidates(evaluation_result)
            
            # Generate response with retries
            response_text, validation_result = await self._generate_with_validation(
                query, winning_candidates, citation_result, response_template, max_retries
            )
            
            # Extract inline citations
            inline_citations = self._extract_inline_citations(response_text)
            
            # Calculate source integration
            source_integration = self._calculate_source_integration(
                response_text, citation_result.filtered_citations
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(validation_result, citation_result)
            
            # Calculate generation time
            generation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update statistics
            self._update_response_stats(generation_time, validation_result, quality_metrics)
            
            enhanced_response = EnhancedResponse(
                response_id=str(uuid4()),
                query=query,
                response_text=response_text,
                inline_citations=inline_citations,
                citation_list=citation_result.filtered_citations,
                source_integration=source_integration,
                response_validation=validation_result,
                quality_metrics=quality_metrics,
                generation_time=generation_time
            )
            
            logger.info("Enhanced response generated",
                       query=query[:50],
                       validation_status=validation_result.validation_status.value,
                       quality_score=validation_result.quality_score,
                       generation_time=generation_time,
                       response_length=len(response_text.split()))
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Enhanced response generation failed: {e}")
            return self._create_error_response(query, str(e))
    
    async def _generate_with_validation(self, 
                                      query: str,
                                      winning_candidates: List[CandidateEvaluation],
                                      citation_result: CitationFilterResult,
                                      response_template: str,
                                      max_retries: int) -> Tuple[str, ResponseValidation]:
        """Generate response with validation and retry logic"""
        best_response = None
        best_validation = None
        
        for attempt in range(max_retries + 1):
            try:
                # Generate response
                response_text = await self._generate_response_text(
                    query, winning_candidates, citation_result, response_template
                )
                
                # Validate response
                validation_result = await self.response_validator.validate_response(
                    response_text, citation_result.filtered_citations, winning_candidates
                )
                
                # Check if validation passes
                if validation_result.validation_status == ValidationStatus.VALID:
                    return response_text, validation_result
                
                # Keep track of best attempt
                if best_validation is None or validation_result.quality_score > best_validation.quality_score:
                    best_response = response_text
                    best_validation = validation_result
                
                # Log retry attempt
                logger.info(f"Response validation failed, retrying (attempt {attempt + 1}/{max_retries + 1})",
                           validation_status=validation_result.validation_status.value,
                           quality_score=validation_result.quality_score)
                
            except Exception as e:
                logger.warning(f"Response generation attempt {attempt + 1} failed: {e}")
                continue
        
        # Return best attempt if no valid response found
        if best_response and best_validation:
            return best_response, best_validation
        
        # Create fallback response
        fallback_response = self._create_fallback_response(query, winning_candidates, citation_result)
        fallback_validation = ResponseValidation(
            validation_status=ValidationStatus.QUALITY_ISSUES,
            quality_score=0.5,
            citation_accuracy=0.5,
            content_accuracy=0.5,
            completeness_score=0.5,
            issues_found=["Generated fallback response due to validation failures"],
            recommendations=["Manual review recommended"],
            validation_details={}
        )
        
        return fallback_response, fallback_validation
    
    async def _generate_response_text(self, 
                                    query: str,
                                    winning_candidates: List[CandidateEvaluation],
                                    citation_result: CitationFilterResult,
                                    response_template: str) -> str:
        """Generate response text using winning candidates and citations"""
        # Get template
        template = self.response_templates.get(response_template, self.response_templates['academic'])
        
        # Generate candidate content
        candidate_content = self._generate_candidate_content(winning_candidates, citation_result)
        
        # Generate citations section
        citations_section = self._generate_citations_section(citation_result.filtered_citations)
        
        # Format response
        response_text = template.format(
            query=query,
            candidate_content=candidate_content,
            citations_section=citations_section,
            source_count=len(citation_result.filtered_citations),
            confidence=citation_result.attribution_confidence
        )
        
        return response_text
    
    def _generate_candidate_content(self, 
                                  winning_candidates: List[CandidateEvaluation],
                                  citation_result: CitationFilterResult) -> str:
        """Generate content from winning candidates with inline citations"""
        if not winning_candidates:
            return "No suitable candidates found for response generation."
        
        # Use top candidate as primary content
        best_candidate = winning_candidates[0]
        content_parts = []
        
        # Add primary answer with inline citation
        primary_citation = self._find_citation_for_candidate(best_candidate, citation_result)
        if primary_citation:
            content_parts.append(f"{best_candidate.candidate_answer.answer_text} {primary_citation.inline_reference}")
        else:
            content_parts.append(best_candidate.candidate_answer.answer_text)
        
        # Add supporting information from other candidates
        for candidate in winning_candidates[1:3]:  # Up to 2 additional candidates
            if candidate.overall_score > 0.6:  # Only high-scoring candidates
                supporting_citation = self._find_citation_for_candidate(candidate, citation_result)
                if supporting_citation:
                    content_parts.append(f"Additionally, {candidate.candidate_answer.answer_text} {supporting_citation.inline_reference}")
        
        return " ".join(content_parts)
    
    def _generate_citations_section(self, citations: List[FilteredCitation]) -> str:
        """Generate formatted citations section"""
        if not citations:
            return "No citations available."
        
        citations_text = "**References:**\n"
        for i, citation in enumerate(citations, 1):
            citations_text += f"{i}. {citation.citation_text}\n"
        
        return citations_text
    
    def _find_citation_for_candidate(self, 
                                   candidate: CandidateEvaluation,
                                   citation_result: CitationFilterResult) -> Optional[FilteredCitation]:
        """Find corresponding citation for a candidate"""
        candidate_sources = [contrib.paper_id for contrib in candidate.candidate_answer.source_contributions]
        
        for citation in citation_result.filtered_citations:
            if citation.paper_id in candidate_sources:
                return citation
        
        return None
    
    def _get_winning_candidates(self, evaluation_result: EvaluationResult) -> List[CandidateEvaluation]:
        """Get winning candidates sorted by score"""
        return sorted(
            evaluation_result.candidate_evaluations,
            key=lambda x: x.overall_score,
            reverse=True
        )[:3]  # Top 3 candidates
    
    def _extract_inline_citations(self, response_text: str) -> List[str]:
        """Extract inline citations from response text"""
        citations = []
        
        # Find numbered citations [1], [2], etc.
        numbered_citations = re.findall(r'\[(\d+)\]', response_text)
        citations.extend(numbered_citations)
        
        # Find author-year citations (Author, Year)
        author_year_citations = re.findall(r'\(([^)]+,\s*\d{4})\)', response_text)
        citations.extend(author_year_citations)
        
        return citations
    
    def _calculate_source_integration(self, 
                                    response_text: str,
                                    citations: List[FilteredCitation]) -> Dict[str, str]:
        """Calculate how each source was integrated into the response"""
        source_integration = {}
        
        for citation in citations:
            # Look for key concepts from this source in the response
            concepts_found = []
            for concept in citation.key_concepts:
                if concept.lower() in response_text.lower():
                    concepts_found.append(concept)
            
            if concepts_found:
                source_integration[citation.paper_id] = f"Concepts used: {', '.join(concepts_found)}"
            else:
                source_integration[citation.paper_id] = "Referenced but concepts not directly mentioned"
        
        return source_integration
    
    def _calculate_quality_metrics(self, 
                                 validation_result: ResponseValidation,
                                 citation_result: CitationFilterResult) -> Dict[str, float]:
        """Calculate quality metrics for the response"""
        return {
            'overall_quality': validation_result.quality_score,
            'citation_accuracy': validation_result.citation_accuracy,
            'content_accuracy': validation_result.content_accuracy,
            'completeness': validation_result.completeness_score,
            'attribution_confidence': citation_result.attribution_confidence,
            'source_utilization': len(citation_result.filtered_citations) / max(1, citation_result.original_sources)
        }
    
    def _create_fallback_response(self, 
                                query: str,
                                winning_candidates: List[CandidateEvaluation],
                                citation_result: CitationFilterResult) -> str:
        """Create fallback response when validation fails"""
        if winning_candidates:
            best_candidate = winning_candidates[0]
            return f"Based on the available research, {best_candidate.candidate_answer.answer_text} Please note that this response may not meet all quality standards."
        
        return f"I apologize, but I cannot provide a high-quality response to your query: {query}. Please try rephrasing your question."
    
    def _create_error_response(self, query: str, error: str) -> EnhancedResponse:
        """Create error response"""
        return EnhancedResponse(
            response_id=str(uuid4()),
            query=query,
            response_text=f"I apologize, but I encountered an error generating a response: {error}",
            inline_citations=[],
            citation_list=[],
            source_integration={},
            response_validation=ResponseValidation(
                validation_status=ValidationStatus.INVALID_FORMAT,
                quality_score=0.0,
                citation_accuracy=0.0,
                content_accuracy=0.0,
                completeness_score=0.0,
                issues_found=[f"Generation error: {error}"],
                recommendations=["Please try again with a different query"],
                validation_details={}
            ),
            quality_metrics={},
            generation_time=0.0
        )
    
    def _update_response_stats(self, 
                             generation_time: float,
                             validation_result: ResponseValidation,
                             quality_metrics: Dict[str, float]):
        """Update response statistics"""
        self.response_stats['total_responses'] += 1
        
        if validation_result.validation_status == ValidationStatus.VALID:
            self.response_stats['successful_responses'] += 1
            self.response_stats['validation_passes'] += 1
        
        # Update averages
        total_responses = self.response_stats['total_responses']
        
        # Average response time
        total_time = (self.response_stats['average_response_time'] * (total_responses - 1) + generation_time)
        self.response_stats['average_response_time'] = total_time / total_responses
        
        # Average quality score
        total_quality = (self.response_stats['average_quality_score'] * (total_responses - 1) + validation_result.quality_score)
        self.response_stats['average_quality_score'] = total_quality / total_responses
        
        # Average citation accuracy
        total_citation = (self.response_stats['citation_accuracy_average'] * (total_responses - 1) + validation_result.citation_accuracy)
        self.response_stats['citation_accuracy_average'] = total_citation / total_responses
    
    def get_response_statistics(self) -> Dict[str, Any]:
        """Get response generation statistics"""
        return {
            **self.response_stats,
            'success_rate': (self.response_stats['successful_responses'] / 
                           max(1, self.response_stats['total_responses'])),
            'validation_pass_rate': (self.response_stats['validation_passes'] / 
                                   max(1, self.response_stats['total_responses']))
        }
    
    async def configure_response_params(self, 
                                      max_response_length: Optional[int] = None,
                                      citation_style: Optional[CitationFormat] = None,
                                      include_validation: Optional[bool] = None,
                                      max_retries: Optional[int] = None):
        """Configure response generation parameters"""
        if max_response_length is not None:
            self.max_response_length = max_response_length
        if citation_style is not None:
            self.citation_style = citation_style
        if include_validation is not None:
            self.include_validation = include_validation
        if max_retries is not None:
            self.max_retries = max_retries
        
        logger.info("Response parameters configured",
                   max_response_length=self.max_response_length,
                   citation_style=self.citation_style.value,
                   include_validation=self.include_validation,
                   max_retries=self.max_retries)


# Factory function for easy instantiation
async def create_enhanced_voicebox(base_voicebox: Optional[NWTNVoicebox] = None) -> EnhancedVoicebox:
    """Create and initialize an enhanced voicebox"""
    voicebox = EnhancedVoicebox(base_voicebox)
    await voicebox.initialize()
    return voicebox
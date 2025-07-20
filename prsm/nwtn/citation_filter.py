#!/usr/bin/env python3
"""
Citation Filter for NWTN System 1 → System 2 → Attribution Pipeline
===================================================================

This module implements accurate citation filtering that extracts sources from
highest-scoring candidates and ensures only papers that actually influenced
the final answer are cited.

Part of Phase 3 of the NWTN System 1 → System 2 → Attribution roadmap.
"""

import asyncio
import structlog
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum
import json
import re

from prsm.nwtn.candidate_evaluator import (
    EvaluationResult,
    CandidateEvaluation,
    EvaluationCriteria
)
from prsm.nwtn.candidate_answer_generator import (
    CandidateAnswer,
    SourceContribution
)

logger = structlog.get_logger(__name__)


class CitationFormat(Enum):
    """Citation format types"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    INLINE = "inline"
    NUMBERED = "numbered"
    AUTHOR_YEAR = "author_year"


class CitationRelevance(Enum):
    """Citation relevance levels"""
    CRITICAL = "critical"        # Essential to the answer
    IMPORTANT = "important"      # Significantly supports the answer
    SUPPORTING = "supporting"    # Provides additional context
    BACKGROUND = "background"    # General background information
    MINIMAL = "minimal"         # Minimal contribution


@dataclass
class FilteredCitation:
    """Represents a filtered citation with attribution confidence"""
    paper_id: str
    title: str
    authors: str
    arxiv_id: str
    publish_date: str
    contribution_score: float  # How much this source contributed to final answer
    relevance_level: CitationRelevance
    attribution_confidence: float  # Confidence in attribution
    usage_description: str  # How this source was used
    key_concepts: List[str]  # Key concepts from this source
    candidate_references: List[str]  # Which candidates used this source
    citation_text: str  # Formatted citation text
    inline_reference: str  # Inline reference format (e.g., "[1]", "(Smith, 2023)")


@dataclass
class CitationFilterResult:
    """Result of citation filtering process"""
    query: str
    original_sources: int  # Total sources from System 1
    filtered_citations: List[FilteredCitation]
    sources_removed: int  # Number of sources filtered out
    attribution_confidence: float  # Overall confidence in attribution
    citation_summary: str  # Summary of citation filtering
    filtering_criteria: Dict[str, Any]  # Criteria used for filtering
    filter_id: str = field(default_factory=lambda: str(uuid4()))


class CitationFormatter:
    """Formats citations in different academic styles"""
    
    def __init__(self):
        self.formats = {
            CitationFormat.APA: self._format_apa,
            CitationFormat.MLA: self._format_mla,
            CitationFormat.CHICAGO: self._format_chicago,
            CitationFormat.INLINE: self._format_inline,
            CitationFormat.NUMBERED: self._format_numbered,
            CitationFormat.AUTHOR_YEAR: self._format_author_year
        }
    
    def format_citation(self, citation: FilteredCitation, format_type: CitationFormat) -> str:
        """Format a citation according to specified style"""
        formatter = self.formats.get(format_type, self._format_inline)
        return formatter(citation)
    
    def _format_apa(self, citation: FilteredCitation) -> str:
        """Format citation in APA style"""
        if citation.publish_date:
            year = citation.publish_date[:4] if len(citation.publish_date) >= 4 else citation.publish_date
            return f"{citation.authors} ({year}). {citation.title}. arXiv:{citation.arxiv_id}."
        return f"{citation.authors}. {citation.title}. arXiv:{citation.arxiv_id}."
    
    def _format_mla(self, citation: FilteredCitation) -> str:
        """Format citation in MLA style"""
        return f"{citation.authors}. \"{citation.title}.\" arXiv:{citation.arxiv_id}, {citation.publish_date}."
    
    def _format_chicago(self, citation: FilteredCitation) -> str:
        """Format citation in Chicago style"""
        return f"{citation.authors}. \"{citation.title}.\" arXiv:{citation.arxiv_id} ({citation.publish_date})."
    
    def _format_inline(self, citation: FilteredCitation) -> str:
        """Format citation for inline use"""
        return f"{citation.title} ({citation.authors})"
    
    def _format_numbered(self, citation: FilteredCitation) -> str:
        """Format citation with numbered reference"""
        return f"{citation.authors}. {citation.title}. arXiv:{citation.arxiv_id} ({citation.publish_date})."
    
    def _format_author_year(self, citation: FilteredCitation) -> str:
        """Format citation with author-year inline reference"""
        if citation.publish_date:
            year = citation.publish_date[:4] if len(citation.publish_date) >= 4 else citation.publish_date
            first_author = citation.authors.split(',')[0].strip() if citation.authors else "Unknown"
            return f"({first_author}, {year})"
        return f"({citation.authors})" if citation.authors else "(Unknown)"


class CitationFilter:
    """
    Accurate citation filtering that extracts sources from highest-scoring candidates
    and ensures only papers that actually influenced the final answer are cited
    """
    
    def __init__(self, citation_formatter: Optional[CitationFormatter] = None):
        self.citation_formatter = citation_formatter or CitationFormatter()
        self.initialized = False
        
        # Filtering parameters
        self.default_relevance_threshold = 0.15  # Lowered for more diverse citations
        self.default_contribution_threshold = 0.1   # Lowered for more source inclusion
        self.top_candidates_to_consider = 6  # Increased for better source coverage
        self.max_citations = 15  # Increased maximum citations for complex queries
        
        # Relevance scoring weights
        self.relevance_weights = {
            'candidate_score': 0.4,      # How well the candidate scored
            'contribution_weight': 0.3,   # How much the source contributed to candidate
            'source_quality': 0.2,       # Quality of the source itself
            'concept_overlap': 0.1        # How many key concepts overlap
        }
        
        # Filtering statistics
        self.filter_stats = {
            'total_filterings': 0,
            'successful_filterings': 0,
            'average_filter_time': 0.0,
            'total_sources_processed': 0,
            'total_sources_filtered': 0,
            'relevance_distribution': {level: 0 for level in CitationRelevance}
        }
    
    async def initialize(self):
        """Initialize the citation filter"""
        try:
            self.initialized = True
            logger.info("CitationFilter initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CitationFilter: {e}")
            return False
    
    async def filter_citations(self, 
                             evaluation_result: EvaluationResult,
                             relevance_threshold: Optional[float] = None,
                             contribution_threshold: Optional[float] = None,
                             citation_format: CitationFormat = CitationFormat.INLINE,
                             max_citations: Optional[int] = None) -> CitationFilterResult:
        """
        Filter citations from evaluation result, keeping only sources that actually
        influenced the highest-scoring candidates
        
        Args:
            evaluation_result: Result from CandidateEvaluator
            relevance_threshold: Minimum relevance score to include citation
            contribution_threshold: Minimum contribution score to include citation
            citation_format: Format for citations
            max_citations: Maximum number of citations to include
            
        Returns:
            CitationFilterResult with filtered citations
        """
        start_time = datetime.now(timezone.utc)
        
        if not self.initialized:
            await self.initialize()
        
        relevance_threshold = relevance_threshold or self.default_relevance_threshold
        contribution_threshold = contribution_threshold or self.default_contribution_threshold
        max_citations = max_citations or self.max_citations
        
        try:
            logger.info("Starting citation filtering",
                       query=evaluation_result.query[:50],
                       candidates=len(evaluation_result.candidate_evaluations),
                       relevance_threshold=relevance_threshold,
                       contribution_threshold=contribution_threshold)
            
            # Step 1: Extract sources from top candidates
            candidate_sources = self._extract_candidate_sources(
                evaluation_result.candidate_evaluations,
                self.top_candidates_to_consider
            )
            
            original_sources_count = len(candidate_sources)
            
            # Step 2: Calculate contribution scores for each source
            source_contributions = self._calculate_source_contributions(
                candidate_sources,
                evaluation_result.candidate_evaluations
            )
            
            # Step 3: Filter sources based on relevance and contribution
            filtered_sources = self._apply_filtering_criteria(
                source_contributions,
                relevance_threshold,
                contribution_threshold
            )
            
            # Step 4: Rank and limit citations
            ranked_citations = self._rank_and_limit_citations(
                filtered_sources,
                max_citations
            )
            
            # Step 5: Format citations
            formatted_citations = []
            for i, citation_data in enumerate(ranked_citations):
                citation = self._create_filtered_citation(
                    citation_data,
                    citation_format,
                    i + 1  # Citation number
                )
                formatted_citations.append(citation)
            
            # Step 6: Calculate overall attribution confidence
            attribution_confidence = self._calculate_attribution_confidence(
                formatted_citations,
                evaluation_result
            )
            
            # Step 7: Generate filtering summary
            filtering_summary = self._generate_filtering_summary(
                original_sources_count,
                len(formatted_citations),
                formatted_citations
            )
            
            # Calculate filtering time
            filter_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update statistics
            self._update_filter_stats(
                filter_time,
                original_sources_count,
                len(formatted_citations),
                formatted_citations
            )
            
            result = CitationFilterResult(
                query=evaluation_result.query,
                original_sources=original_sources_count,
                filtered_citations=formatted_citations,
                sources_removed=original_sources_count - len(formatted_citations),
                attribution_confidence=attribution_confidence,
                citation_summary=filtering_summary,
                filtering_criteria={
                    'relevance_threshold': relevance_threshold,
                    'contribution_threshold': contribution_threshold,
                    'max_citations': max_citations,
                    'top_candidates_considered': self.top_candidates_to_consider
                }
            )
            
            logger.info("Citation filtering completed",
                       query=evaluation_result.query[:50],
                       original_sources=original_sources_count,
                       filtered_citations=len(formatted_citations),
                       sources_removed=original_sources_count - len(formatted_citations),
                       attribution_confidence=attribution_confidence,
                       filter_time=filter_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Citation filtering failed: {e}")
            return CitationFilterResult(
                query=evaluation_result.query,
                original_sources=0,
                filtered_citations=[],
                sources_removed=0,
                attribution_confidence=0.0,
                citation_summary="Citation filtering failed due to error",
                filtering_criteria={}
            )
    
    def _extract_candidate_sources(self, 
                                 candidate_evaluations: List[CandidateEvaluation],
                                 top_n: int) -> Dict[str, SourceContribution]:
        """Extract sources from top N candidates"""
        candidate_sources = {}
        
        # Sort candidates by score and take top N
        sorted_candidates = sorted(
            candidate_evaluations,
            key=lambda x: x.overall_score,
            reverse=True
        )[:top_n]
        
        for candidate in sorted_candidates:
            for contribution in candidate.candidate_answer.source_contributions:
                source_id = contribution.paper_id
                if source_id not in candidate_sources:
                    candidate_sources[source_id] = contribution
                # Keep the contribution with highest weight if duplicate
                elif contribution.contribution_weight > candidate_sources[source_id].contribution_weight:
                    candidate_sources[source_id] = contribution
        
        return candidate_sources
    
    def _calculate_source_contributions(self, 
                                      candidate_sources: Dict[str, SourceContribution],
                                      candidate_evaluations: List[CandidateEvaluation]) -> Dict[str, Dict[str, Any]]:
        """Calculate detailed contribution scores for each source"""
        source_contributions = {}
        
        for source_id, contribution in candidate_sources.items():
            # Find all candidates that used this source
            using_candidates = []
            for evaluation in candidate_evaluations:
                for contrib in evaluation.candidate_answer.source_contributions:
                    if contrib.paper_id == source_id:
                        using_candidates.append(evaluation)
                        break
            
            if not using_candidates:
                continue
            
            # Calculate weighted contribution score
            total_score = 0.0
            total_weight = 0.0
            
            for candidate in using_candidates:
                candidate_weight = candidate.overall_score
                source_weight = next(
                    (c.contribution_weight for c in candidate.candidate_answer.source_contributions 
                     if c.paper_id == source_id), 0.0
                )
                
                weighted_score = candidate_weight * source_weight
                total_score += weighted_score
                total_weight += candidate_weight
            
            # Calculate final contribution score
            contribution_score = total_score / max(total_weight, 0.01)
            
            # Extract key concepts used
            key_concepts = set()
            for candidate in using_candidates:
                key_concepts.update(candidate.candidate_answer.key_concepts_used)
            
            # Generate usage description
            usage_description = self._generate_usage_description(contribution, using_candidates)
            
            source_contributions[source_id] = {
                'contribution': contribution,
                'contribution_score': contribution_score,
                'using_candidates': using_candidates,
                'key_concepts': list(key_concepts),
                'usage_description': usage_description,
                'candidate_references': [c.candidate_answer.candidate_id for c in using_candidates]
            }
        
        return source_contributions
    
    def _apply_filtering_criteria(self, 
                                source_contributions: Dict[str, Dict[str, Any]],
                                relevance_threshold: float,
                                contribution_threshold: float) -> Dict[str, Dict[str, Any]]:
        """Apply filtering criteria to remove low-relevance sources"""
        filtered_sources = {}
        
        for source_id, data in source_contributions.items():
            contribution_score = data['contribution_score']
            source_quality = data['contribution'].quality_score
            
            # Calculate relevance score
            relevance_score = (
                contribution_score * self.relevance_weights['candidate_score'] +
                data['contribution'].contribution_weight * self.relevance_weights['contribution_weight'] +
                source_quality * self.relevance_weights['source_quality'] +
                min(len(data['key_concepts']) / 10.0, 1.0) * self.relevance_weights['concept_overlap']
            )
            
            # Apply filtering criteria
            if (relevance_score >= relevance_threshold and 
                contribution_score >= contribution_threshold):
                
                data['relevance_score'] = relevance_score
                data['relevance_level'] = self._determine_relevance_level(relevance_score)
                filtered_sources[source_id] = data
        
        return filtered_sources
    
    def _rank_and_limit_citations(self, 
                                filtered_sources: Dict[str, Dict[str, Any]],
                                max_citations: int) -> List[Dict[str, Any]]:
        """Rank sources by relevance and limit to max citations"""
        # Sort by relevance score (descending)
        sorted_sources = sorted(
            filtered_sources.values(),
            key=lambda x: x['relevance_score'],
            reverse=True
        )
        
        # Limit to max citations
        return sorted_sources[:max_citations]
    
    def _create_filtered_citation(self, 
                                citation_data: Dict[str, Any],
                                citation_format: CitationFormat,
                                citation_number: int) -> FilteredCitation:
        """Create a FilteredCitation object from citation data"""
        contribution = citation_data['contribution']
        
        # Format citation text
        temp_citation = FilteredCitation(
            paper_id=contribution.paper_id,
            title=contribution.title,
            authors=contribution.paper_id,  # Will be replaced
            arxiv_id="",
            publish_date="",
            contribution_score=citation_data['contribution_score'],
            relevance_level=citation_data['relevance_level'],
            attribution_confidence=citation_data['contribution_score'],
            usage_description=citation_data['usage_description'],
            key_concepts=citation_data['key_concepts'],
            candidate_references=citation_data['candidate_references'],
            citation_text="",
            inline_reference=""
        )
        
        # Format citation and inline reference
        citation_text = self.citation_formatter.format_citation(temp_citation, citation_format)
        inline_reference = self._generate_inline_reference(temp_citation, citation_format, citation_number)
        
        return FilteredCitation(
            paper_id=contribution.paper_id,
            title=contribution.title,
            authors=contribution.paper_id,  # Simplified for now
            arxiv_id=contribution.paper_id,
            publish_date="2023",  # Placeholder
            contribution_score=citation_data['contribution_score'],
            relevance_level=citation_data['relevance_level'],
            attribution_confidence=citation_data['contribution_score'],
            usage_description=citation_data['usage_description'],
            key_concepts=citation_data['key_concepts'],
            candidate_references=citation_data['candidate_references'],
            citation_text=citation_text,
            inline_reference=inline_reference
        )
    
    def _generate_usage_description(self, 
                                  contribution: SourceContribution,
                                  using_candidates: List[CandidateEvaluation]) -> str:
        """Generate description of how the source was used"""
        usage_types = set()
        
        for candidate in using_candidates:
            answer_type = candidate.candidate_answer.answer_type.value
            usage_types.add(answer_type)
        
        if len(usage_types) == 1:
            return f"Used in {list(usage_types)[0]} analysis"
        else:
            return f"Used in {', '.join(usage_types)} analyses"
    
    def _determine_relevance_level(self, relevance_score: float) -> CitationRelevance:
        """Determine relevance level based on score"""
        if relevance_score >= 0.8:
            return CitationRelevance.CRITICAL
        elif relevance_score >= 0.6:
            return CitationRelevance.IMPORTANT
        elif relevance_score >= 0.4:
            return CitationRelevance.SUPPORTING
        elif relevance_score >= 0.2:
            return CitationRelevance.BACKGROUND
        else:
            return CitationRelevance.MINIMAL
    
    def _generate_inline_reference(self, 
                                 citation: FilteredCitation,
                                 citation_format: CitationFormat,
                                 citation_number: int) -> str:
        """Generate inline reference format"""
        if citation_format == CitationFormat.NUMBERED:
            return f"[{citation_number}]"
        elif citation_format == CitationFormat.AUTHOR_YEAR:
            return self.citation_formatter.format_citation(citation, CitationFormat.AUTHOR_YEAR)
        else:
            return f"[{citation_number}]"
    
    def _calculate_attribution_confidence(self, 
                                        citations: List[FilteredCitation],
                                        evaluation_result: EvaluationResult) -> float:
        """Calculate overall confidence in attribution"""
        if not citations:
            return 0.0
        
        # Base confidence from evaluation result
        base_confidence = evaluation_result.overall_confidence
        
        # Adjust based on citation quality
        citation_quality = sum(c.attribution_confidence for c in citations) / len(citations)
        
        # Adjust based on source coverage
        total_sources = len(evaluation_result.source_lineage)
        cited_sources = len(citations)
        coverage_factor = min(cited_sources / max(total_sources, 1), 1.0)
        
        # Calculate final confidence
        attribution_confidence = (
            base_confidence * 0.5 +
            citation_quality * 0.3 +
            coverage_factor * 0.2
        )
        
        return max(0.0, min(1.0, attribution_confidence))
    
    def _generate_filtering_summary(self, 
                                  original_sources: int,
                                  filtered_sources: int,
                                  citations: List[FilteredCitation]) -> str:
        """Generate summary of filtering process"""
        if filtered_sources == 0:
            return f"No sources met filtering criteria from {original_sources} original sources"
        
        removed_count = original_sources - filtered_sources
        removal_percentage = (removed_count / max(original_sources, 1)) * 100
        
        relevance_counts = {}
        for citation in citations:
            level = citation.relevance_level
            relevance_counts[level] = relevance_counts.get(level, 0) + 1
        
        relevance_text = ", ".join([f"{count} {level.value}" for level, count in relevance_counts.items()])
        
        return f"Filtered {original_sources} sources to {filtered_sources} citations ({removal_percentage:.1f}% removed). Relevance: {relevance_text}"
    
    def _update_filter_stats(self, 
                           filter_time: float,
                           original_sources: int,
                           filtered_sources: int,
                           citations: List[FilteredCitation]):
        """Update filtering statistics"""
        self.filter_stats['total_filterings'] += 1
        self.filter_stats['successful_filterings'] += 1 if filtered_sources > 0 else 0
        self.filter_stats['total_sources_processed'] += original_sources
        self.filter_stats['total_sources_filtered'] += filtered_sources
        
        # Update average filter time
        total_time = (self.filter_stats['average_filter_time'] * 
                     (self.filter_stats['total_filterings'] - 1) + filter_time)
        self.filter_stats['average_filter_time'] = total_time / self.filter_stats['total_filterings']
        
        # Update relevance distribution
        for citation in citations:
            self.filter_stats['relevance_distribution'][citation.relevance_level] += 1
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        return {
            **self.filter_stats,
            'success_rate': (self.filter_stats['successful_filterings'] / 
                           max(1, self.filter_stats['total_filterings'])),
            'average_sources_per_filter': (self.filter_stats['total_sources_processed'] / 
                                         max(1, self.filter_stats['total_filterings'])),
            'average_filtered_per_filter': (self.filter_stats['total_sources_filtered'] / 
                                          max(1, self.filter_stats['successful_filterings'])),
            'filter_ratio': (self.filter_stats['total_sources_filtered'] / 
                           max(1, self.filter_stats['total_sources_processed']))
        }
    
    async def configure_filter_params(self, 
                                    relevance_threshold: Optional[float] = None,
                                    contribution_threshold: Optional[float] = None,
                                    top_candidates_to_consider: Optional[int] = None,
                                    max_citations: Optional[int] = None):
        """Configure filtering parameters"""
        if relevance_threshold is not None:
            self.default_relevance_threshold = relevance_threshold
        if contribution_threshold is not None:
            self.default_contribution_threshold = contribution_threshold
        if top_candidates_to_consider is not None:
            self.top_candidates_to_consider = top_candidates_to_consider
        if max_citations is not None:
            self.max_citations = max_citations
        
        logger.info("Filter parameters configured",
                   relevance_threshold=self.default_relevance_threshold,
                   contribution_threshold=self.default_contribution_threshold,
                   top_candidates_to_consider=self.top_candidates_to_consider,
                   max_citations=self.max_citations)


# Factory function for easy instantiation
async def create_citation_filter(citation_formatter: Optional[CitationFormatter] = None) -> CitationFilter:
    """Create and initialize a citation filter"""
    filter_instance = CitationFilter(citation_formatter)
    await filter_instance.initialize()
    return filter_instance
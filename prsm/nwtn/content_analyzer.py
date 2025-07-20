#!/usr/bin/env python3
"""
Content Analyzer for NWTN System 1 → System 2 → Attribution Pipeline
==================================================================

This module implements deep content analysis and extraction for the NWTN system,
processing paper abstracts and key sections to extract structured insights.

Part of Phase 1.2 of the NWTN System 1 → System 2 → Attribution roadmap.
"""

import asyncio
import re
import structlog
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum
import json

from prsm.nwtn.semantic_retriever import RetrievedPaper, SemanticSearchResult

logger = structlog.get_logger(__name__)


class ContentQuality(Enum):
    """Content quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class ExtractedConcept:
    """Represents a key concept extracted from paper content"""
    concept: str
    category: str  # e.g., "methodology", "finding", "theory", "application"
    confidence: float  # 0.0 to 1.0
    context: str  # Surrounding text providing context
    paper_id: str
    extraction_method: str = "nlp_analysis"


@dataclass
class ContentSummary:
    """Structured summary of paper content"""
    paper_id: str
    title: str
    main_contributions: List[str]
    key_concepts: List[ExtractedConcept]
    methodologies: List[str]
    findings: List[str]
    applications: List[str]
    limitations: List[str]
    quality_score: float
    quality_level: ContentQuality
    analysis_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    summary_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ContentAnalysisResult:
    """Complete result of content analysis operation"""
    query: str
    analyzed_papers: List[ContentSummary]
    total_concepts_extracted: int
    analysis_time_seconds: float
    quality_distribution: Dict[ContentQuality, int]
    analysis_id: str = field(default_factory=lambda: str(uuid4()))


class ConceptExtractor:
    """Extracts key concepts from paper content using NLP techniques"""
    
    def __init__(self):
        self.concept_patterns = {
            "methodology": [
                r"(method|approach|technique|algorithm|procedure|framework|model)",
                r"(neural network|machine learning|deep learning|reinforcement learning)",
                r"(statistical|probabilistic|mathematical|computational) (model|method|approach)",
                r"(optimization|simulation|analysis|evaluation|assessment)"
            ],
            "finding": [
                r"(result|finding|outcome|conclusion|discovery|observation)",
                r"(showed|demonstrated|revealed|indicated|suggested|found)",
                r"(significant|important|notable|remarkable|unexpected) (result|finding|effect)",
                r"(correlation|relationship|association|connection|link)"
            ],
            "theory": [
                r"(theory|theorem|hypothesis|principle|law|axiom)",
                r"(theoretical|conceptual) (framework|model|foundation)",
                r"(quantum|classical|statistical|information) (theory|mechanics|physics)",
                r"(mathematical|logical|formal) (proof|derivation|formulation)"
            ],
            "application": [
                r"(application|use case|implementation|deployment|practical)",
                r"(real-world|industrial|commercial|clinical) (application|use)",
                r"(system|platform|tool|software|hardware) (for|that|which)",
                r"(applied to|used in|implemented in|deployed for)"
            ]
        }
        
        # Common academic stopwords to filter out
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after", "above",
            "below", "between", "among", "this", "that", "these", "those", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can", "paper", "study", "research",
            "work", "article", "authors", "we", "our", "they", "their", "it", "its", "also",
            "however", "therefore", "thus", "hence", "furthermore", "moreover", "additionally"
        }
    
    def extract_concepts(self, text: str, paper_id: str) -> List[ExtractedConcept]:
        """Extract key concepts from text content"""
        concepts = []
        text_lower = text.lower()
        
        for category, patterns in self.concept_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    # Extract the actual concept phrase
                    concept_text = match.group(0)
                    
                    # Filter out very short or common terms
                    if len(concept_text) < 3 or concept_text.lower() in self.stopwords:
                        continue
                    
                    # Calculate confidence based on context richness
                    confidence = self._calculate_concept_confidence(concept_text, context)
                    
                    if confidence > 0.3:  # Only keep reasonably confident extractions
                        concepts.append(ExtractedConcept(
                            concept=concept_text.strip(),
                            category=category,
                            confidence=confidence,
                            context=context,
                            paper_id=paper_id,
                            extraction_method="regex_pattern"
                        ))
        
        # Remove duplicates and sort by confidence
        unique_concepts = self._deduplicate_concepts(concepts)
        unique_concepts.sort(key=lambda c: c.confidence, reverse=True)
        
        return unique_concepts[:20]  # Return top 20 concepts
    
    def _calculate_concept_confidence(self, concept: str, context: str) -> float:
        """Calculate confidence score for extracted concept"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for longer, more specific terms
        if len(concept) > 10:
            confidence += 0.2
        
        # Boost confidence for technical terms
        technical_indicators = ["algorithm", "model", "method", "analysis", "system", "framework"]
        if any(indicator in concept.lower() for indicator in technical_indicators):
            confidence += 0.15
        
        # Boost confidence for context richness
        if len(context) > 100:
            confidence += 0.1
        
        # Reduce confidence for very common terms
        common_terms = ["the", "and", "for", "with", "this", "that"]
        if any(term in concept.lower() for term in common_terms):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _deduplicate_concepts(self, concepts: List[ExtractedConcept]) -> List[ExtractedConcept]:
        """Remove duplicate concepts, keeping the highest confidence version"""
        seen = {}
        for concept in concepts:
            key = concept.concept.lower().strip()
            if key not in seen or concept.confidence > seen[key].confidence:
                seen[key] = concept
        return list(seen.values())


class ContentAnalyzer:
    """
    Advanced content analysis system for NWTN System 1 candidate generation
    
    Processes paper abstracts and key sections to extract structured insights
    for use in generating candidate answers.
    """
    
    def __init__(self, concept_extractor: Optional[ConceptExtractor] = None):
        self.concept_extractor = concept_extractor or ConceptExtractor()
        self.initialized = False
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            ContentQuality.EXCELLENT: 0.8,
            ContentQuality.GOOD: 0.6,
            ContentQuality.AVERAGE: 0.4,
            ContentQuality.POOR: 0.2,
            ContentQuality.UNUSABLE: 0.0
        }
        
        # Analysis statistics
        self.analysis_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'concepts_extracted': 0,
            'average_analysis_time': 0.0,
            'quality_distribution': {q: 0 for q in ContentQuality}
        }
    
    async def initialize(self):
        """Initialize the content analyzer"""
        try:
            self.initialized = True
            logger.info("Content analyzer initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize content analyzer: {e}")
            return False
    
    async def analyze_retrieved_papers(self, 
                                     search_result: SemanticSearchResult) -> ContentAnalysisResult:
        """
        Analyze all papers from a semantic search result
        
        Args:
            search_result: Result from semantic search containing papers to analyze
            
        Returns:
            ContentAnalysisResult with structured analysis of all papers
        """
        start_time = datetime.now(timezone.utc)
        
        if not self.initialized:
            await self.initialize()
        
        try:
            analyzed_papers = []
            total_concepts = 0
            quality_distribution = {q: 0 for q in ContentQuality}
            
            for paper in search_result.retrieved_papers:
                try:
                    summary = await self.analyze_paper_content(paper)
                    analyzed_papers.append(summary)
                    total_concepts += len(summary.key_concepts)
                    quality_distribution[summary.quality_level] += 1
                    
                    logger.debug("Paper analyzed successfully",
                               paper_id=paper.paper_id,
                               quality=summary.quality_level.value,
                               concepts_extracted=len(summary.key_concepts))
                
                except Exception as e:
                    logger.warning(f"Failed to analyze paper {paper.paper_id}: {e}")
                    continue
            
            # Calculate analysis time
            analysis_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update statistics
            self._update_analysis_stats(analysis_time, len(analyzed_papers), 
                                      total_concepts, quality_distribution)
            
            result = ContentAnalysisResult(
                query=search_result.query,
                analyzed_papers=analyzed_papers,
                total_concepts_extracted=total_concepts,
                analysis_time_seconds=analysis_time,
                quality_distribution=quality_distribution
            )
            
            logger.info("Content analysis completed",
                       query=search_result.query[:50],
                       papers_analyzed=len(analyzed_papers),
                       concepts_extracted=total_concepts,
                       analysis_time=analysis_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return ContentAnalysisResult(
                query=search_result.query,
                analyzed_papers=[],
                total_concepts_extracted=0,
                analysis_time_seconds=0.0,
                quality_distribution={q: 0 for q in ContentQuality}
            )
    
    async def analyze_paper_content(self, paper: RetrievedPaper) -> ContentSummary:
        """
        Analyze individual paper content to extract structured insights
        
        Args:
            paper: Retrieved paper to analyze
            
        Returns:
            ContentSummary with extracted insights
        """
        try:
            # Combine title and abstract for analysis
            full_text = f"{paper.title}. {paper.abstract}"
            
            # Extract key concepts
            key_concepts = self.concept_extractor.extract_concepts(full_text, paper.paper_id)
            
            # Extract structured information
            main_contributions = self._extract_contributions(full_text)
            methodologies = self._extract_methodologies(key_concepts, full_text)
            findings = self._extract_findings(key_concepts, full_text)
            applications = self._extract_applications(key_concepts, full_text)
            limitations = self._extract_limitations(full_text)
            
            # Assess content quality
            quality_score = self._assess_content_quality(
                full_text, key_concepts, main_contributions, methodologies, findings
            )
            quality_level = self._determine_quality_level(quality_score)
            
            summary = ContentSummary(
                paper_id=paper.paper_id,
                title=paper.title,
                main_contributions=main_contributions,
                key_concepts=key_concepts,
                methodologies=methodologies,
                findings=findings,
                applications=applications,
                limitations=limitations,
                quality_score=quality_score,
                quality_level=quality_level
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to analyze paper content: {e}")
            # Return minimal summary on failure
            return ContentSummary(
                paper_id=paper.paper_id,
                title=paper.title,
                main_contributions=[],
                key_concepts=[],
                methodologies=[],
                findings=[],
                applications=[],
                limitations=[],
                quality_score=0.0,
                quality_level=ContentQuality.UNUSABLE
            )
    
    def _extract_contributions(self, text: str) -> List[str]:
        """Extract main contributions from paper text"""
        contributions = []
        
        # Look for contribution indicators
        contribution_patterns = [
            r"contribut[a-z]+[^.]*\.?",
            r"propos[a-z]+[^.]*\.?",
            r"present[a-z]+[^.]*\.?",
            r"introduc[a-z]+[^.]*\.?",
            r"develop[a-z]+[^.]*\.?",
            r"novel[^.]*\.?",
            r"new[^.]*\.?",
            r"improv[a-z]+[^.]*\.?"
        ]
        
        for pattern in contribution_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                contribution = match.group(0).strip()
                if len(contribution) > 20 and contribution not in contributions:
                    contributions.append(contribution)
        
        return contributions[:3]  # Return top 3 contributions
    
    def _extract_methodologies(self, concepts: List[ExtractedConcept], text: str) -> List[str]:
        """Extract methodologies from concepts and text"""
        methodologies = []
        
        # Extract from concepts
        for concept in concepts:
            if concept.category == "methodology":
                methodologies.append(concept.concept)
        
        # Extract from text patterns
        method_patterns = [
            r"using[^.]*\.?",
            r"based on[^.]*\.?",
            r"employing[^.]*\.?",
            r"applying[^.]*\.?",
            r"implementing[^.]*\.?"
        ]
        
        for pattern in method_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                method = match.group(0).strip()
                if len(method) > 15 and method not in methodologies:
                    methodologies.append(method)
        
        return methodologies[:5]  # Return top 5 methodologies
    
    def _extract_findings(self, concepts: List[ExtractedConcept], text: str) -> List[str]:
        """Extract key findings from concepts and text"""
        findings = []
        
        # Extract from concepts
        for concept in concepts:
            if concept.category == "finding":
                findings.append(concept.concept)
        
        # Extract from text patterns
        finding_patterns = [
            r"result[a-z]* show[a-z]*[^.]*\.?",
            r"found that[^.]*\.?",
            r"demonstrated[^.]*\.?",
            r"revealed[^.]*\.?",
            r"indicated[^.]*\.?",
            r"observed[^.]*\.?"
        ]
        
        for pattern in finding_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                finding = match.group(0).strip()
                if len(finding) > 20 and finding not in findings:
                    findings.append(finding)
        
        return findings[:5]  # Return top 5 findings
    
    def _extract_applications(self, concepts: List[ExtractedConcept], text: str) -> List[str]:
        """Extract applications from concepts and text"""
        applications = []
        
        # Extract from concepts
        for concept in concepts:
            if concept.category == "application":
                applications.append(concept.concept)
        
        # Extract from text patterns
        app_patterns = [
            r"can be applied to[^.]*\.?",
            r"useful for[^.]*\.?",
            r"applications include[^.]*\.?",
            r"used in[^.]*\.?",
            r"practical[^.]*\.?"
        ]
        
        for pattern in app_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                application = match.group(0).strip()
                if len(application) > 15 and application not in applications:
                    applications.append(application)
        
        return applications[:3]  # Return top 3 applications
    
    def _extract_limitations(self, text: str) -> List[str]:
        """Extract limitations from text"""
        limitations = []
        
        limitation_patterns = [
            r"limitation[a-z]*[^.]*\.?",
            r"however[^.]*\.?",
            r"but[^.]*\.?",
            r"unfortunately[^.]*\.?",
            r"challenge[a-z]*[^.]*\.?",
            r"problem[a-z]*[^.]*\.?"
        ]
        
        for pattern in limitation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                limitation = match.group(0).strip()
                if len(limitation) > 20 and limitation not in limitations:
                    limitations.append(limitation)
        
        return limitations[:3]  # Return top 3 limitations
    
    def _assess_content_quality(self, text: str, concepts: List[ExtractedConcept], 
                               contributions: List[str], methodologies: List[str], 
                               findings: List[str]) -> float:
        """Assess overall content quality"""
        quality_score = 0.0
        
        # Text length and richness
        if len(text) > 200:
            quality_score += 0.2
        if len(text) > 500:
            quality_score += 0.1
        
        # Concept extraction success
        if len(concepts) > 5:
            quality_score += 0.2
        if len(concepts) > 10:
            quality_score += 0.1
        
        # Structured content extraction
        if contributions:
            quality_score += 0.15
        if methodologies:
            quality_score += 0.15
        if findings:
            quality_score += 0.15
        
        # High-confidence concepts
        high_conf_concepts = [c for c in concepts if c.confidence > 0.7]
        if len(high_conf_concepts) > 3:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _determine_quality_level(self, score: float) -> ContentQuality:
        """Determine quality level from score"""
        for level, threshold in sorted(self.quality_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return level
        return ContentQuality.UNUSABLE
    
    def _update_analysis_stats(self, analysis_time: float, papers_analyzed: int, 
                             concepts_extracted: int, quality_dist: Dict[ContentQuality, int]):
        """Update analysis statistics"""
        self.analysis_stats['total_analyses'] += 1
        self.analysis_stats['successful_analyses'] += 1 if papers_analyzed > 0 else 0
        self.analysis_stats['concepts_extracted'] += concepts_extracted
        
        # Update average analysis time
        total_time = (self.analysis_stats['average_analysis_time'] * 
                     (self.analysis_stats['total_analyses'] - 1) + analysis_time)
        self.analysis_stats['average_analysis_time'] = total_time / self.analysis_stats['total_analyses']
        
        # Update quality distribution
        for quality, count in quality_dist.items():
            self.analysis_stats['quality_distribution'][quality] += count
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get content analysis statistics"""
        return {
            **self.analysis_stats,
            'success_rate': (self.analysis_stats['successful_analyses'] / 
                           max(1, self.analysis_stats['total_analyses'])),
            'average_concepts_per_paper': (self.analysis_stats['concepts_extracted'] / 
                                         max(1, self.analysis_stats['successful_analyses']))
        }


# Factory function for easy instantiation
async def create_content_analyzer() -> ContentAnalyzer:
    """Create and initialize a content analyzer"""
    analyzer = ContentAnalyzer()
    await analyzer.initialize()
    return analyzer
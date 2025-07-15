#!/usr/bin/env python3
"""
Content Quality Filter for PRSM
===============================

This module provides advanced content quality filtering for large-scale
breadth-optimized ingestion, ensuring only high-quality content with
strong analogical reasoning potential is ingested.

Key Features:
1. Multi-dimensional quality assessment
2. Analogical reasoning potential scoring
3. Cross-domain relevance detection
4. Breakthrough potential identification
5. Content deduplication and similarity detection
6. Adaptive quality thresholds
7. Real-time quality monitoring

Designed to maximize the value of ingested content while maintaining
storage efficiency and system performance.
"""

import asyncio
import json
import logging
import hashlib
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import string
from collections import Counter
import math

import structlog

logger = structlog.get_logger(__name__)


class QualityDecision(str, Enum):
    """Quality assessment decisions"""
    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "review"  # Requires human review
    CONDITIONAL = "conditional"  # Accept under certain conditions


class ContentType(str, Enum):
    """Types of content being filtered"""
    RESEARCH_PAPER = "research_paper"
    PREPRINT = "preprint"
    DATASET = "dataset"
    CODE_REPOSITORY = "code_repository"
    REVIEW_ARTICLE = "review_article"
    TECHNICAL_REPORT = "technical_report"
    CONFERENCE_PAPER = "conference_paper"
    BOOK_CHAPTER = "book_chapter"
    THESIS = "thesis"
    PATENT = "patent"


class QualityDimension(str, Enum):
    """Quality assessment dimensions"""
    ACADEMIC_QUALITY = "academic_quality"
    ANALOGICAL_POTENTIAL = "analogical_potential"
    CROSS_DOMAIN_RELEVANCE = "cross_domain_relevance"
    BREAKTHROUGH_POTENTIAL = "breakthrough_potential"
    CONTENT_RICHNESS = "content_richness"
    METHODOLOGICAL_RIGOR = "methodological_rigor"
    NOVELTY_SCORE = "novelty_score"
    CITATION_POTENTIAL = "citation_potential"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for content"""
    
    # Basic quality metrics
    academic_quality: float = 0.0        # 0-1 scale
    content_richness: float = 0.0        # 0-1 scale
    methodological_rigor: float = 0.0    # 0-1 scale
    
    # Analogical reasoning metrics
    analogical_potential: float = 0.0    # 0-1 scale
    cross_domain_relevance: float = 0.0  # 0-1 scale
    conceptual_density: float = 0.0      # 0-1 scale
    
    # Innovation metrics
    breakthrough_potential: float = 0.0  # 0-1 scale
    novelty_score: float = 0.0          # 0-1 scale
    citation_potential: float = 0.0     # 0-1 scale
    
    # Technical metrics
    content_length: int = 0
    keyword_density: float = 0.0
    readability_score: float = 0.0
    structural_quality: float = 0.0
    
    # Similarity metrics
    similarity_to_existing: float = 0.0  # 0-1 scale (lower is better)
    uniqueness_score: float = 0.0       # 0-1 scale (higher is better)
    
    # Overall scores
    overall_quality: float = 0.0         # Weighted combination
    ingestion_priority: float = 0.0      # Priority for ingestion
    
    # Metadata
    assessment_confidence: float = 0.0   # Confidence in assessment
    processing_time: float = 0.0
    quality_factors: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class ContentAnalysis:
    """Comprehensive content analysis"""
    
    # Content identification
    content_id: str
    content_type: ContentType
    source: str
    
    # Content characteristics
    title: str
    abstract: str
    keywords: List[str]
    domain: str
    
    # Quality assessment
    quality_metrics: QualityMetrics
    quality_decision: QualityDecision
    decision_reasoning: str
    
    # Processing metadata
    analysis_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filter_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "content_id": self.content_id,
            "content_type": self.content_type.value,
            "source": self.source,
            "title": self.title,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "domain": self.domain,
            "quality_metrics": {
                "academic_quality": self.quality_metrics.academic_quality,
                "analogical_potential": self.quality_metrics.analogical_potential,
                "cross_domain_relevance": self.quality_metrics.cross_domain_relevance,
                "breakthrough_potential": self.quality_metrics.breakthrough_potential,
                "content_richness": self.quality_metrics.content_richness,
                "overall_quality": self.quality_metrics.overall_quality,
                "ingestion_priority": self.quality_metrics.ingestion_priority
            },
            "quality_decision": self.quality_decision.value,
            "decision_reasoning": self.decision_reasoning,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "filter_version": self.filter_version
        }


@dataclass
class FilterConfig:
    """Configuration for content quality filtering"""
    
    # Quality thresholds - OPTIMIZED FOR SPEED AND BREADTH
    min_academic_quality: float = 0.3
    min_analogical_potential: float = 0.2
    min_cross_domain_relevance: float = 0.2
    min_breakthrough_potential: float = 0.1
    min_content_richness: float = 0.2
    min_overall_quality: float = 0.25
    
    # Content constraints - RELAXED FOR SPEED
    min_content_length: int = 100      # Much lower minimum
    max_content_length: int = 100000   # Maximum word count
    min_abstract_length: int = 20      # Lower minimum abstract length
    max_similarity_threshold: float = 0.95  # Higher similarity tolerance
    
    # Domain-specific settings
    domain_priority_weights: Dict[str, float] = field(default_factory=dict)
    preferred_content_types: List[ContentType] = field(default_factory=list)
    
    # Adaptive settings
    adaptive_thresholds: bool = True
    quality_learning_enabled: bool = True
    batch_size: int = 1000
    
    # Performance settings
    max_processing_time: float = 5.0   # Maximum seconds per content
    parallel_processing: bool = True
    cache_enabled: bool = True


class ContentQualityFilter:
    """
    Advanced Content Quality Filter for PRSM
    
    Provides multi-dimensional quality assessment designed specifically
    for breadth-optimized ingestion with emphasis on analogical reasoning
    potential and cross-domain relevance.
    """
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        
        # Quality assessment models
        self.quality_models = {}
        self.similarity_cache = {}
        self.content_signatures = set()
        
        # Domain knowledge
        self.domain_keywords = {}
        self.cross_domain_mappings = {}
        self.breakthrough_indicators = set()
        
        # Adaptive learning
        self.quality_history = []
        self.threshold_adjustments = {}
        
        # Performance tracking
        self.filter_stats = {
            "total_processed": 0,
            "accepted": 0,
            "rejected": 0,
            "review_required": 0,
            "average_processing_time": 0.0,
            "quality_distribution": {},
            "rejection_reasons": Counter()
        }
        
        # Initialize components
        self._initialize_domain_knowledge()
        self._initialize_breakthrough_indicators()
        
        logger.info("Content Quality Filter initialized")
    
    async def initialize(self):
        """Initialize the content quality filter"""
        
        logger.info("🔍 Initializing Content Quality Filter...")
        
        # Load domain knowledge
        await self._load_domain_knowledge()
        
        # Initialize quality models
        await self._initialize_quality_models()
        
        # Load existing content signatures
        await self._load_content_signatures()
        
        logger.info("✅ Content Quality Filter ready")
    
    async def assess_content_quality(self, content_data: Dict[str, Any]) -> ContentAnalysis:
        """
        Perform comprehensive content quality assessment
        
        Args:
            content_data: Raw content data including title, abstract, keywords, etc.
            
        Returns:
            Complete content analysis with quality assessment
        """
        
        start_time = datetime.now()
        
        try:
            # Extract content characteristics
            content_id = content_data.get("id", "unknown")
            content_type = ContentType(content_data.get("type", "research_paper"))
            source = content_data.get("source", "unknown")
            
            title = content_data.get("title", "")
            abstract = content_data.get("abstract", "")
            keywords = content_data.get("keywords", [])
            domain = content_data.get("domain", "unknown")
            
            logger.debug(f"Assessing content quality: {content_id}")
            
            # Initialize quality metrics
            quality_metrics = QualityMetrics()
            
            # Assess different quality dimensions
            await self._assess_academic_quality(quality_metrics, content_data)
            await self._assess_analogical_potential(quality_metrics, content_data)
            await self._assess_cross_domain_relevance(quality_metrics, content_data)
            await self._assess_breakthrough_potential(quality_metrics, content_data)
            await self._assess_content_richness(quality_metrics, content_data)
            await self._assess_uniqueness(quality_metrics, content_data)
            
            # Calculate overall quality
            quality_metrics.overall_quality = await self._calculate_overall_quality(quality_metrics)
            quality_metrics.ingestion_priority = await self._calculate_ingestion_priority(quality_metrics)
            
            # Make quality decision
            quality_decision, decision_reasoning = await self._make_quality_decision(quality_metrics)
            
            # Record processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            quality_metrics.processing_time = processing_time
            
            # Create content analysis
            analysis = ContentAnalysis(
                content_id=content_id,
                content_type=content_type,
                source=source,
                title=title,
                abstract=abstract,
                keywords=keywords,
                domain=domain,
                quality_metrics=quality_metrics,
                quality_decision=quality_decision,
                decision_reasoning=decision_reasoning
            )
            
            # Update statistics
            await self._update_filter_stats(analysis)
            
            # Learn from assessment if enabled
            if self.config.quality_learning_enabled:
                await self._learn_from_assessment(analysis)
            
            logger.debug(f"Quality assessment completed: {content_id}",
                        overall_quality=quality_metrics.overall_quality,
                        decision=quality_decision.value,
                        processing_time=processing_time)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Quality assessment failed for {content_data.get('id', 'unknown')}: {e}")
            raise
    
    async def batch_assess_quality(self, content_batch: List[Dict[str, Any]]) -> List[ContentAnalysis]:
        """
        Assess quality for a batch of content items
        
        Args:
            content_batch: List of content data dictionaries
            
        Returns:
            List of content analyses
        """
        
        logger.info(f"Batch quality assessment: {len(content_batch)} items")
        
        if self.config.parallel_processing:
            # Process in parallel
            tasks = [self.assess_content_quality(content) for content in content_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            analyses = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch assessment error: {result}")
                else:
                    analyses.append(result)
            
            return analyses
        else:
            # Process sequentially
            analyses = []
            for content in content_batch:
                try:
                    analysis = await self.assess_content_quality(content)
                    analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Content assessment failed: {e}")
                    continue
            
            return analyses
    
    async def get_filter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filter statistics"""
        
        acceptance_rate = self.filter_stats["accepted"] / max(1, self.filter_stats["total_processed"])
        rejection_rate = self.filter_stats["rejected"] / max(1, self.filter_stats["total_processed"])
        
        return {
            "filter_performance": {
                "total_processed": self.filter_stats["total_processed"],
                "accepted": self.filter_stats["accepted"],
                "rejected": self.filter_stats["rejected"],
                "review_required": self.filter_stats["review_required"],
                "acceptance_rate": acceptance_rate,
                "rejection_rate": rejection_rate,
                "average_processing_time": self.filter_stats["average_processing_time"]
            },
            "quality_distribution": self.filter_stats["quality_distribution"],
            "rejection_reasons": dict(self.filter_stats["rejection_reasons"]),
            "adaptive_thresholds": self.threshold_adjustments,
            "content_uniqueness": {
                "unique_signatures": len(self.content_signatures),
                "similarity_cache_size": len(self.similarity_cache)
            }
        }
    
    async def optimize_thresholds(self) -> Dict[str, Any]:
        """Optimize quality thresholds based on performance data"""
        
        if not self.config.adaptive_thresholds or len(self.quality_history) < 100:
            return {"message": "Not enough data for optimization"}
        
        logger.info("🔧 Optimizing quality thresholds...")
        
        # Analyze quality distribution
        quality_scores = [item["overall_quality"] for item in self.quality_history]
        acceptance_outcomes = [item["accepted"] for item in self.quality_history]
        
        # Find optimal thresholds
        optimal_thresholds = {}
        
        # Use statistical analysis to find optimal cutoffs
        for percentile in [50, 75, 90, 95]:
            threshold = np.percentile(quality_scores, percentile)
            
            # Calculate performance at this threshold
            predicted_accepts = sum(1 for score in quality_scores if score >= threshold)
            predicted_rate = predicted_accepts / len(quality_scores)
            
            optimal_thresholds[f"p{percentile}"] = {
                "threshold": threshold,
                "predicted_acceptance_rate": predicted_rate
            }
        
        # Apply conservative optimization
        current_acceptance_rate = sum(acceptance_outcomes) / len(acceptance_outcomes)
        
        if current_acceptance_rate < 0.3:  # Too restrictive
            self.config.min_overall_quality = max(0.4, self.config.min_overall_quality - 0.05)
            self.threshold_adjustments["overall_quality"] = "decreased"
        elif current_acceptance_rate > 0.8:  # Too permissive
            self.config.min_overall_quality = min(0.8, self.config.min_overall_quality + 0.05)
            self.threshold_adjustments["overall_quality"] = "increased"
        
        optimization_result = {
            "current_acceptance_rate": current_acceptance_rate,
            "optimal_thresholds": optimal_thresholds,
            "threshold_adjustments": self.threshold_adjustments,
            "new_min_overall_quality": self.config.min_overall_quality
        }
        
        logger.info("✅ Threshold optimization completed",
                   new_threshold=self.config.min_overall_quality,
                   acceptance_rate=current_acceptance_rate)
        
        return optimization_result
    
    # === Private Methods ===
    
    def _initialize_domain_knowledge(self):
        """Initialize domain-specific knowledge"""
        
        # High-value keywords by domain
        self.domain_keywords = {
            "physics": ["quantum", "particle", "energy", "field", "theory", "experiment"],
            "biology": ["cell", "gene", "protein", "evolution", "organism", "molecular"],
            "computer_science": ["algorithm", "data", "network", "machine", "system", "computation"],
            "mathematics": ["theorem", "proof", "function", "equation", "analysis", "topology"],
            "chemistry": ["molecule", "reaction", "synthesis", "catalyst", "structure", "bond"],
            "neuroscience": ["brain", "neuron", "synaptic", "cognitive", "neural", "memory"],
            "economics": ["market", "economic", "price", "demand", "supply", "welfare"],
            "psychology": ["behavior", "cognitive", "mental", "learning", "memory", "emotion"]
        }
        
        # Cross-domain connection indicators
        self.cross_domain_mappings = {
            "network": ["biology", "computer_science", "sociology", "physics"],
            "optimization": ["mathematics", "computer_science", "economics", "engineering"],
            "information": ["computer_science", "physics", "biology", "communication"],
            "learning": ["psychology", "computer_science", "neuroscience", "education"],
            "system": ["physics", "biology", "computer_science", "engineering"]
        }
    
    def _initialize_breakthrough_indicators(self):
        """Initialize breakthrough detection indicators"""
        
        self.breakthrough_indicators = {
            "novel", "breakthrough", "revolutionary", "paradigm", "first", "unprecedented",
            "innovative", "groundbreaking", "pioneering", "transformative", "disruptive",
            "emergent", "discovery", "invention", "advancement", "progress", "leap",
            "significant", "major", "important", "crucial", "critical", "fundamental"
        }
    
    async def _load_domain_knowledge(self):
        """Load domain-specific knowledge"""
        # In production, this would load from files or databases
        logger.info("Domain knowledge loaded")
    
    async def _initialize_quality_models(self):
        """Initialize quality assessment models"""
        # In production, this would load trained models
        logger.info("Quality models initialized")
    
    async def _load_content_signatures(self):
        """Load existing content signatures for deduplication"""
        # In production, this would load from persistent storage
        logger.info("Content signatures loaded")
    
    async def _assess_academic_quality(self, metrics: QualityMetrics, content: Dict[str, Any]):
        """Assess academic quality of content"""
        
        title = content.get("title", "")
        abstract = content.get("abstract", "")
        
        # Check for academic indicators
        academic_indicators = [
            "study", "research", "analysis", "investigation", "experiment",
            "method", "results", "conclusion", "hypothesis", "theory"
        ]
        
        text = (title + " " + abstract).lower()
        indicator_count = sum(1 for indicator in academic_indicators if indicator in text)
        
        # Basic academic quality score
        metrics.academic_quality = min(1.0, indicator_count / len(academic_indicators))
        
        # Adjust based on content type
        content_type = content.get("type", "research_paper")
        if content_type in ["research_paper", "conference_paper"]:
            metrics.academic_quality *= 1.2
        elif content_type == "preprint":
            metrics.academic_quality *= 0.9
        
        metrics.academic_quality = min(1.0, metrics.academic_quality)
    
    async def _assess_analogical_potential(self, metrics: QualityMetrics, content: Dict[str, Any]):
        """Assess analogical reasoning potential"""
        
        title = content.get("title", "")
        abstract = content.get("abstract", "")
        keywords = content.get("keywords", [])
        
        # Look for analogical indicators
        analogical_indicators = [
            "similar", "like", "analogy", "analogous", "parallel", "equivalent",
            "compare", "comparison", "correspond", "relate", "relationship",
            "pattern", "structure", "framework", "model", "approach"
        ]
        
        text = (title + " " + abstract + " " + " ".join(keywords)).lower()
        
        # Count analogical indicators
        indicator_score = sum(1 for indicator in analogical_indicators if indicator in text)
        indicator_score = min(1.0, indicator_score / 5)  # Normalize
        
        # Check for cross-domain keywords
        cross_domain_score = 0
        for keyword, domains in self.cross_domain_mappings.items():
            if keyword in text and len(domains) > 2:
                cross_domain_score += 0.2
        
        cross_domain_score = min(1.0, cross_domain_score)
        
        # Combined analogical potential
        metrics.analogical_potential = (indicator_score * 0.6 + cross_domain_score * 0.4)
    
    async def _assess_cross_domain_relevance(self, metrics: QualityMetrics, content: Dict[str, Any]):
        """Assess cross-domain relevance"""
        
        title = content.get("title", "")
        abstract = content.get("abstract", "")
        keywords = content.get("keywords", [])
        domain = content.get("domain", "")
        
        text = (title + " " + abstract + " " + " ".join(keywords)).lower()
        
        # Check for keywords from other domains
        other_domain_matches = 0
        total_other_domains = 0
        
        for other_domain, domain_keywords in self.domain_keywords.items():
            if other_domain != domain:
                total_other_domains += 1
                domain_match_count = sum(1 for keyword in domain_keywords if keyword in text)
                if domain_match_count > 0:
                    other_domain_matches += 1
        
        # Cross-domain relevance score
        if total_other_domains > 0:
            metrics.cross_domain_relevance = other_domain_matches / total_other_domains
        else:
            metrics.cross_domain_relevance = 0.0
    
    async def _assess_breakthrough_potential(self, metrics: QualityMetrics, content: Dict[str, Any]):
        """Assess breakthrough potential"""
        
        title = content.get("title", "")
        abstract = content.get("abstract", "")
        
        text = (title + " " + abstract).lower()
        
        # Count breakthrough indicators
        breakthrough_count = sum(1 for indicator in self.breakthrough_indicators if indicator in text)
        
        # Normalize breakthrough score
        metrics.breakthrough_potential = min(1.0, breakthrough_count / 3)
        
        # Boost for novel combinations
        if "novel" in text and any(domain in text for domain in self.domain_keywords.keys()):
            metrics.breakthrough_potential *= 1.2
        
        metrics.breakthrough_potential = min(1.0, metrics.breakthrough_potential)
    
    async def _assess_content_richness(self, metrics: QualityMetrics, content: Dict[str, Any]):
        """Assess content richness and depth"""
        
        title = content.get("title", "")
        abstract = content.get("abstract", "")
        keywords = content.get("keywords", [])
        
        # Content length score
        content_length = len(title) + len(abstract)
        metrics.content_length = content_length
        
        length_score = min(1.0, content_length / 1000)  # Normalize to 1000 chars
        
        # Keyword richness
        keyword_score = min(1.0, len(keywords) / 10)  # Normalize to 10 keywords
        
        # Technical depth indicators
        technical_indicators = [
            "method", "methodology", "approach", "technique", "algorithm",
            "framework", "system", "model", "analysis", "evaluation"
        ]
        
        text = (title + " " + abstract).lower()
        technical_score = sum(1 for indicator in technical_indicators if indicator in text)
        technical_score = min(1.0, technical_score / 5)
        
        # Combined richness score
        metrics.content_richness = (length_score * 0.4 + keyword_score * 0.3 + technical_score * 0.3)
    
    async def _assess_uniqueness(self, metrics: QualityMetrics, content: Dict[str, Any]):
        """Assess content uniqueness to avoid duplicates"""
        
        title = content.get("title", "")
        abstract = content.get("abstract", "")
        
        # Create content signature
        content_signature = hashlib.md5((title + abstract).encode()).hexdigest()
        
        # Check for exact duplicates
        if content_signature in self.content_signatures:
            metrics.similarity_to_existing = 1.0
            metrics.uniqueness_score = 0.0
        else:
            # Add to signatures
            self.content_signatures.add(content_signature)
            
            # Simple similarity check (in production, use more sophisticated methods)
            max_similarity = 0.0
            for existing_sig in list(self.content_signatures)[-1000:]:  # Check last 1000
                # Simple character-level similarity
                similarity = self._calculate_simple_similarity(content_signature, existing_sig)
                max_similarity = max(max_similarity, similarity)
            
            metrics.similarity_to_existing = max_similarity
            metrics.uniqueness_score = 1.0 - max_similarity
    
    def _calculate_simple_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate simple character-level similarity"""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    async def _calculate_overall_quality(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score"""
        
        # Define weights for different quality dimensions
        weights = {
            "academic_quality": 0.20,
            "analogical_potential": 0.25,  # Higher weight for analogical reasoning
            "cross_domain_relevance": 0.20,
            "breakthrough_potential": 0.15,
            "content_richness": 0.15,
            "uniqueness_score": 0.05
        }
        
        # Calculate weighted score
        overall_score = (
            metrics.academic_quality * weights["academic_quality"] +
            metrics.analogical_potential * weights["analogical_potential"] +
            metrics.cross_domain_relevance * weights["cross_domain_relevance"] +
            metrics.breakthrough_potential * weights["breakthrough_potential"] +
            metrics.content_richness * weights["content_richness"] +
            metrics.uniqueness_score * weights["uniqueness_score"]
        )
        
        return min(1.0, overall_score)
    
    async def _calculate_ingestion_priority(self, metrics: QualityMetrics) -> float:
        """Calculate ingestion priority score"""
        
        # Priority based on analogical potential and breakthrough potential
        priority = (
            metrics.analogical_potential * 0.4 +
            metrics.breakthrough_potential * 0.3 +
            metrics.cross_domain_relevance * 0.2 +
            metrics.overall_quality * 0.1
        )
        
        return min(1.0, priority)
    
    async def _make_quality_decision(self, metrics: QualityMetrics) -> Tuple[QualityDecision, str]:
        """Make final quality decision based on metrics"""
        
        # Check hard constraints
        if metrics.content_length < self.config.min_content_length:
            return QualityDecision.REJECT, "Content too short"
        
        if metrics.similarity_to_existing > self.config.max_similarity_threshold:
            return QualityDecision.REJECT, "Content too similar to existing"
        
        # Check quality thresholds
        if metrics.overall_quality < self.config.min_overall_quality:
            return QualityDecision.REJECT, "Overall quality below threshold"
        
        if metrics.analogical_potential < self.config.min_analogical_potential:
            return QualityDecision.REJECT, "Analogical potential too low"
        
        # Accept high-quality content
        if metrics.overall_quality >= 0.8:
            return QualityDecision.ACCEPT, "High quality content"
        
        # Conditional acceptance for moderate quality
        if metrics.breakthrough_potential > 0.7:
            return QualityDecision.ACCEPT, "High breakthrough potential"
        
        if metrics.cross_domain_relevance > 0.8:
            return QualityDecision.ACCEPT, "Strong cross-domain relevance"
        
        # Default acceptance for content meeting thresholds
        return QualityDecision.ACCEPT, "Meets quality thresholds"
    
    async def _update_filter_stats(self, analysis: ContentAnalysis):
        """Update filter statistics"""
        
        self.filter_stats["total_processed"] += 1
        
        if analysis.quality_decision == QualityDecision.ACCEPT:
            self.filter_stats["accepted"] += 1
        elif analysis.quality_decision == QualityDecision.REJECT:
            self.filter_stats["rejected"] += 1
            self.filter_stats["rejection_reasons"][analysis.decision_reasoning] += 1
        elif analysis.quality_decision == QualityDecision.REVIEW:
            self.filter_stats["review_required"] += 1
        
        # Update average processing time
        current_avg = self.filter_stats["average_processing_time"]
        total_processed = self.filter_stats["total_processed"]
        
        self.filter_stats["average_processing_time"] = (
            (current_avg * (total_processed - 1) + analysis.quality_metrics.processing_time) / total_processed
        )
        
        # Update quality distribution
        quality_bucket = f"{int(analysis.quality_metrics.overall_quality * 10)}/10"
        self.filter_stats["quality_distribution"][quality_bucket] = (
            self.filter_stats["quality_distribution"].get(quality_bucket, 0) + 1
        )
    
    async def _learn_from_assessment(self, analysis: ContentAnalysis):
        """Learn from quality assessment for adaptive improvement"""
        
        # Store assessment for learning
        self.quality_history.append({
            "overall_quality": analysis.quality_metrics.overall_quality,
            "analogical_potential": analysis.quality_metrics.analogical_potential,
            "breakthrough_potential": analysis.quality_metrics.breakthrough_potential,
            "accepted": analysis.quality_decision == QualityDecision.ACCEPT,
            "timestamp": analysis.analysis_timestamp
        })
        
        # Keep only recent history
        if len(self.quality_history) > 10000:
            self.quality_history = self.quality_history[-5000:]


# Test function
async def test_quality_filter():
    """Test content quality filter"""
    
    print("🔍 CONTENT QUALITY FILTER TEST")
    print("=" * 50)
    
    # Initialize filter
    filter_system = ContentQualityFilter()
    await filter_system.initialize()
    
    # Test content
    test_content = [
        {
            "id": "test_1",
            "type": "research_paper",
            "title": "Novel Quantum Machine Learning Algorithm for Cross-Domain Pattern Recognition",
            "abstract": "This paper presents a breakthrough approach to machine learning using quantum computing principles. The method demonstrates significant improvements in pattern recognition across multiple domains including biology, physics, and computer science. Our analysis shows unprecedented results in analogical reasoning tasks.",
            "keywords": ["quantum", "machine learning", "pattern recognition", "cross-domain", "analogical"],
            "domain": "computer_science",
            "source": "arxiv"
        },
        {
            "id": "test_2",
            "type": "preprint",
            "title": "Short title",
            "abstract": "Brief abstract.",
            "keywords": ["test"],
            "domain": "unknown",
            "source": "preprint"
        }
    ]
    
    # Test single assessment
    analysis = await filter_system.assess_content_quality(test_content[0])
    print(f"Single Assessment: ✅ Quality: {analysis.quality_metrics.overall_quality:.2f}")
    print(f"Decision: {analysis.quality_decision.value}")
    
    # Test batch assessment
    batch_analyses = await filter_system.batch_assess_quality(test_content)
    print(f"Batch Assessment: ✅ Processed {len(batch_analyses)} items")
    
    # Get statistics
    stats = await filter_system.get_filter_statistics()
    print(f"Filter Stats: ✅ Acceptance rate: {stats['filter_performance']['acceptance_rate']:.1%}")
    
    # Test threshold optimization
    optimization = await filter_system.optimize_thresholds()
    print(f"Optimization: ✅ New threshold: {optimization.get('new_min_overall_quality', 'N/A')}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(test_quality_filter())
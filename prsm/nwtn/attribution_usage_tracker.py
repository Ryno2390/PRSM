#!/usr/bin/env python3
"""
Attribution Usage Tracker for NWTN System 1 → System 2 → Attribution Pipeline
===============================================================================

This module tracks the usage of specific sources in the final response and
integrates with the FTNS payment system to ensure accurate royalty distribution
based on actual source contributions to the generated answer.

Part of Phase 4 of the NWTN System 1 → System 2 → Attribution roadmap.
"""

import asyncio
import structlog
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum
import json

from prsm.nwtn.citation_filter import CitationFilterResult, FilteredCitation
from prsm.nwtn.enhanced_voicebox import EnhancedResponse
from prsm.nwtn.candidate_evaluator import EvaluationResult
from prsm.tokenomics.ftns_service import FTNSService
from prsm.integrations.core.provenance_engine import ProvenanceEngine
from prsm.nwtn.content_royalty_engine import ContentRoyaltyEngine

logger = structlog.get_logger(__name__)


class UsageType(Enum):
    """Types of source usage in the pipeline"""
    RETRIEVED = "retrieved"           # Source was retrieved by System 1
    ANALYZED = "analyzed"             # Source was analyzed by ContentAnalyzer
    CANDIDATE_USED = "candidate_used" # Source contributed to candidate answer
    EVALUATED = "evaluated"           # Source was part of evaluated candidates
    CITED = "cited"                   # Source was cited in final response
    PAID = "paid"                     # Source owner was paid for contribution


class ContributionLevel(Enum):
    """Levels of contribution for payment calculation"""
    CRITICAL = "critical"      # Essential to the final answer (80-100% weight)
    PRIMARY = "primary"        # Major contribution (60-80% weight)
    SUPPORTING = "supporting"  # Supporting evidence (40-60% weight)
    BACKGROUND = "background"  # Background context (20-40% weight)
    MINIMAL = "minimal"        # Minimal contribution (0-20% weight)


@dataclass
class SourceUsage:
    """Detailed usage tracking for a specific source"""
    paper_id: str
    title: str
    authors: str
    usage_types: List[UsageType]
    contribution_level: ContributionLevel
    contribution_weight: float  # 0.0 to 1.0
    attribution_confidence: float  # 0.0 to 1.0
    usage_context: str  # How the source was used
    candidate_references: List[str]  # Which candidates used this source
    citation_position: Optional[int]  # Position in citation list
    quality_score: float  # Quality assessment of the source
    relevance_score: float  # Relevance to the query
    payment_eligible: bool  # Whether source is eligible for payment
    usage_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QueryUsageSession:
    """Complete usage session for a query"""
    session_id: str
    query: str
    user_id: str
    total_cost: float  # Total FTNS cost charged to user
    system_fee: float  # FTNS retained by system
    creator_distribution: float  # FTNS distributed to creators
    sources_used: List[SourceUsage]
    payment_calculations: Dict[str, float]  # paper_id -> payment amount
    audit_trail: List[Dict[str, Any]]  # Complete audit trail
    processing_time: float
    pipeline_stages: Dict[str, Any]  # Performance metrics for each stage
    session_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PaymentDistribution:
    """Payment distribution to content creators"""
    session_id: str
    creator_id: str
    paper_id: str
    payment_amount: float
    contribution_weight: float
    contribution_level: ContributionLevel
    payment_rationale: str
    ftns_transaction_id: Optional[str] = None
    payment_status: str = "pending"
    payment_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AttributionUsageTracker:
    """
    Tracks usage of sources throughout the System 1 → System 2 → Attribution pipeline
    and integrates with FTNS for accurate royalty distribution
    """
    
    def __init__(self, 
                 ftns_service: Optional[FTNSService] = None,
                 provenance_engine: Optional[ProvenanceEngine] = None,
                 royalty_engine: Optional[ContentRoyaltyEngine] = None):
        self.ftns_service = ftns_service
        self.provenance_engine = provenance_engine
        self.royalty_engine = royalty_engine
        self.initialized = False
        
        # Payment distribution parameters
        self.system_fee_percentage = 0.3  # 30% system fee
        self.creator_distribution_percentage = 0.7  # 70% to creators
        
        # Contribution level weights for payment calculation
        self.contribution_weights = {
            ContributionLevel.CRITICAL: 1.0,
            ContributionLevel.PRIMARY: 0.8,
            ContributionLevel.SUPPORTING: 0.6,
            ContributionLevel.BACKGROUND: 0.4,
            ContributionLevel.MINIMAL: 0.2
        }
        
        # Usage session storage
        self.usage_sessions: Dict[str, QueryUsageSession] = {}
        
        # Usage statistics
        self.usage_stats = {
            'total_sessions': 0,
            'total_payments_distributed': 0.0,
            'total_system_fees': 0.0,
            'average_sources_per_query': 0.0,
            'average_payment_per_source': 0.0,
            'payment_distribution_by_level': {level: 0.0 for level in ContributionLevel}
        }
    
    async def initialize(self):
        """Initialize the usage tracker"""
        try:
            # Initialize services if not provided
            if self.ftns_service is None:
                self.ftns_service = FTNSService()
                await self.ftns_service.initialize()
            
            if self.provenance_engine is None:
                self.provenance_engine = ProvenanceEngine()
                await self.provenance_engine.initialize()
            
            if self.royalty_engine is None:
                self.royalty_engine = ContentRoyaltyEngine()
                await self.royalty_engine.initialize()
            
            self.initialized = True
            logger.info("AttributionUsageTracker initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AttributionUsageTracker: {e}")
            return False
    
    async def track_complete_usage(self,
                                 query: str,
                                 user_id: str,
                                 evaluation_result: EvaluationResult,
                                 citation_result: CitationFilterResult,
                                 enhanced_response: EnhancedResponse,
                                 total_cost: float,
                                 processing_time: float,
                                 session_id: Optional[str] = None) -> QueryUsageSession:
        """
        Track complete usage from System 1 → System 2 → Attribution pipeline
        and calculate payment distributions
        
        Args:
            query: Original user query
            user_id: User who submitted the query
            evaluation_result: Result from CandidateEvaluator
            citation_result: Result from CitationFilter
            enhanced_response: Result from EnhancedVoicebox
            total_cost: Total FTNS cost charged to user
            processing_time: Total processing time
            session_id: Session ID from SystemIntegrator (optional, will generate if not provided)
            
        Returns:
            QueryUsageSession with complete usage tracking
        """
        if not self.initialized:
            await self.initialize()
        
        # Use provided session_id or generate a new one
        if session_id is None:
            session_id = str(uuid4())
        
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info("Starting complete usage tracking",
                       session_id=session_id,
                       query=query[:50],
                       user_id=user_id,
                       total_cost=total_cost)
            
            # Step 1: Track source usage throughout pipeline
            sources_used = await self._track_pipeline_sources(
                evaluation_result,
                citation_result,
                enhanced_response
            )
            
            # Step 2: Calculate payment distributions
            payment_calculations = await self._calculate_payment_distributions(
                sources_used,
                total_cost
            )
            
            # Step 3: Generate audit trail
            audit_trail = await self._generate_audit_trail(
                session_id,
                query,
                user_id,
                sources_used,
                payment_calculations,
                evaluation_result,
                citation_result,
                enhanced_response
            )
            
            # Step 4: Create usage session
            usage_session = QueryUsageSession(
                session_id=session_id,
                query=query,
                user_id=user_id,
                total_cost=total_cost,
                system_fee=total_cost * self.system_fee_percentage,
                creator_distribution=total_cost * self.creator_distribution_percentage,
                sources_used=sources_used,
                payment_calculations=payment_calculations,
                audit_trail=audit_trail,
                processing_time=processing_time,
                pipeline_stages=self._extract_pipeline_metrics(
                    evaluation_result, citation_result, enhanced_response
                )
            )
            
            # Step 5: Store session
            self.usage_sessions[session_id] = usage_session
            logger.info("Session stored in usage_sessions",
                       session_id=session_id,
                       total_sessions=len(self.usage_sessions))
            
            # Step 6: Update statistics
            self._update_usage_stats(usage_session)
            
            logger.info("Complete usage tracking completed",
                       session_id=session_id,
                       sources_tracked=len(sources_used),
                       total_payments=sum(payment_calculations.values()),
                       processing_time=processing_time)
            
            return usage_session
            
        except Exception as e:
            logger.error(f"Usage tracking failed: {e}", session_id=session_id)
            raise
    
    async def _track_pipeline_sources(self,
                                    evaluation_result: EvaluationResult,
                                    citation_result: CitationFilterResult,
                                    enhanced_response: EnhancedResponse) -> List[SourceUsage]:
        """Track sources throughout the pipeline"""
        sources_used = []
        
        # Get all sources from citations (these are the ones actually used)
        cited_sources = {citation.paper_id for citation in citation_result.filtered_citations}
        
        # Track each cited source
        for citation in citation_result.filtered_citations:
            usage_types = [UsageType.RETRIEVED, UsageType.ANALYZED, 
                          UsageType.CANDIDATE_USED, UsageType.EVALUATED, UsageType.CITED]
            
            # Determine contribution level based on citation relevance
            contribution_level = self._map_relevance_to_contribution_level(
                citation.relevance_level
            )
            
            # Find which candidates used this source
            candidate_references = []
            for eval_result in evaluation_result.candidate_evaluations:
                for contrib in eval_result.candidate_answer.source_contributions:
                    if contrib.paper_id == citation.paper_id:
                        candidate_references.append(eval_result.candidate_id)
            
            # Check if source is eligible for payment
            payment_eligible = await self._check_payment_eligibility(citation.paper_id)
            
            source_usage = SourceUsage(
                paper_id=citation.paper_id,
                title=citation.title,
                authors=citation.authors,
                usage_types=usage_types,
                contribution_level=contribution_level,
                contribution_weight=citation.contribution_score,
                attribution_confidence=citation.attribution_confidence,
                usage_context=citation.usage_description,
                candidate_references=candidate_references,
                citation_position=citation_result.filtered_citations.index(citation) + 1,
                quality_score=citation.contribution_score,  # Use contribution as quality proxy
                relevance_score=citation.contribution_score,
                payment_eligible=payment_eligible
            )
            
            sources_used.append(source_usage)
        
        return sources_used
    
    async def _calculate_payment_distributions(self,
                                             sources_used: List[SourceUsage],
                                             total_cost: float) -> Dict[str, float]:
        """Calculate payment distributions to content creators"""
        payment_calculations = {}
        
        # Calculate total available for creator distribution
        creator_distribution_pool = total_cost * self.creator_distribution_percentage
        
        # Calculate total weighted contributions
        total_weighted_contribution = 0.0
        eligible_sources = [s for s in sources_used if s.payment_eligible]
        
        for source in eligible_sources:
            weight = self.contribution_weights[source.contribution_level]
            weighted_contribution = source.contribution_weight * weight
            total_weighted_contribution += weighted_contribution
        
        # Distribute payment proportionally
        if total_weighted_contribution > 0:
            for source in eligible_sources:
                weight = self.contribution_weights[source.contribution_level]
                weighted_contribution = source.contribution_weight * weight
                payment_share = weighted_contribution / total_weighted_contribution
                payment_amount = creator_distribution_pool * payment_share
                payment_calculations[source.paper_id] = payment_amount
        
        return payment_calculations
    
    async def _generate_audit_trail(self,
                                  session_id: str,
                                  query: str,
                                  user_id: str,
                                  sources_used: List[SourceUsage],
                                  payment_calculations: Dict[str, float],
                                  evaluation_result: EvaluationResult,
                                  citation_result: CitationFilterResult,
                                  enhanced_response: EnhancedResponse) -> List[Dict[str, Any]]:
        """Generate complete audit trail for the session"""
        audit_trail = []
        
        # Step 1: Query submission
        audit_trail.append({
            'step': 'query_submission',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': {
                'query': query,
                'user_id': user_id,
                'session_id': session_id
            }
        })
        
        # Step 2: System 1 - Candidate generation
        audit_trail.append({
            'step': 'system1_candidate_generation',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': {
                'candidates_generated': len(evaluation_result.candidate_evaluations),
                'sources_analyzed': len(evaluation_result.source_lineage),
                'evaluation_time': evaluation_result.evaluation_time_seconds
            }
        })
        
        # Step 3: System 2 - Evaluation
        audit_trail.append({
            'step': 'system2_evaluation',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': {
                'best_candidate_score': evaluation_result.best_candidate.overall_score,
                'evaluation_confidence': evaluation_result.overall_confidence,
                'thinking_mode': evaluation_result.thinking_mode_used
            }
        })
        
        # Step 4: Attribution - Citation filtering
        audit_trail.append({
            'step': 'attribution_citation_filtering',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': {
                'original_sources': citation_result.original_sources,
                'filtered_citations': len(citation_result.filtered_citations),
                'sources_removed': citation_result.sources_removed,
                'attribution_confidence': citation_result.attribution_confidence
            }
        })
        
        # Step 5: Response generation
        audit_trail.append({
            'step': 'response_generation',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': {
                'response_length': len(enhanced_response.response_text),
                'quality_score': enhanced_response.response_validation.quality_score,
                'citations_used': len(enhanced_response.citation_list),
                'generation_time': enhanced_response.generation_time
            }
        })
        
        # Step 6: Payment calculation
        audit_trail.append({
            'step': 'payment_calculation',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': {
                'total_cost': sum(payment_calculations.values()) + (sum(payment_calculations.values()) * (self.system_fee_percentage / self.creator_distribution_percentage)),
                'creator_distribution': sum(payment_calculations.values()),
                'system_fee': sum(payment_calculations.values()) * (self.system_fee_percentage / self.creator_distribution_percentage),
                'sources_paid': len(payment_calculations),
                'payment_breakdown': payment_calculations
            }
        })
        
        return audit_trail
    
    async def distribute_payments(self, session_id: str) -> List[PaymentDistribution]:
        """Distribute payments to content creators for a session"""
        logger.info("Distribute payments called",
                   session_id=session_id,
                   available_sessions=list(self.usage_sessions.keys()))
        
        if session_id not in self.usage_sessions:
            logger.error("Session not found in usage_sessions",
                        session_id=session_id,
                        available_sessions=list(self.usage_sessions.keys()))
            raise ValueError(f"Session {session_id} not found")
        
        session = self.usage_sessions[session_id]
        distributions = []
        
        try:
            for paper_id, payment_amount in session.payment_calculations.items():
                # Find source usage for this paper
                source_usage = next(
                    (s for s in session.sources_used if s.paper_id == paper_id),
                    None
                )
                
                if source_usage and source_usage.payment_eligible:
                    # Get creator information from provenance
                    creator_info = await self.provenance_engine.get_creator_info(paper_id)
                    
                    if creator_info:
                        # Create FTNS payment
                        ftns_transaction_id = await self.ftns_service.distribute_royalty(
                            creator_info['creator_id'],
                            payment_amount,
                            f"Content usage royalty for paper {paper_id} in session {session_id}"
                        )
                        
                        # Create payment distribution record
                        distribution = PaymentDistribution(
                            session_id=session_id,
                            creator_id=creator_info['creator_id'],
                            paper_id=paper_id,
                            payment_amount=payment_amount,
                            contribution_weight=source_usage.contribution_weight,
                            contribution_level=source_usage.contribution_level,
                            payment_rationale=f"Source contributed {source_usage.contribution_weight:.2f} with {source_usage.contribution_level.value} level",
                            ftns_transaction_id=ftns_transaction_id,
                            payment_status="completed"
                        )
                        
                        distributions.append(distribution)
                        
                        # Update source usage to mark as paid
                        source_usage.usage_types.append(UsageType.PAID)
            
            logger.info("Payment distribution completed",
                       session_id=session_id,
                       distributions=len(distributions),
                       total_distributed=sum(d.payment_amount for d in distributions))
            
            return distributions
            
        except Exception as e:
            logger.error(f"Payment distribution failed: {e}", session_id=session_id)
            raise
    
    def _map_relevance_to_contribution_level(self, relevance_level) -> ContributionLevel:
        """Map citation relevance level to contribution level"""
        mapping = {
            'critical': ContributionLevel.CRITICAL,
            'important': ContributionLevel.PRIMARY,
            'supporting': ContributionLevel.SUPPORTING,
            'background': ContributionLevel.BACKGROUND,
            'minimal': ContributionLevel.MINIMAL
        }
        return mapping.get(relevance_level.value, ContributionLevel.MINIMAL)
    
    async def _check_payment_eligibility(self, paper_id: str) -> bool:
        """Check if source is eligible for payment"""
        try:
            # Check provenance for creator information
            creator_info = await self.provenance_engine.get_creator_info(paper_id)
            return creator_info is not None
        except Exception:
            return False
    
    def _extract_pipeline_metrics(self, evaluation_result, citation_result, enhanced_response) -> Dict[str, Any]:
        """Extract performance metrics from pipeline stages"""
        return {
            'evaluation_time': evaluation_result.evaluation_time_seconds,
            'citation_filtering_time': 0.0,  # Not tracked in current implementation
            'response_generation_time': enhanced_response.generation_time,
            'total_candidates': len(evaluation_result.candidate_evaluations),
            'sources_filtered': citation_result.original_sources - len(citation_result.filtered_citations),
            'response_quality': enhanced_response.response_validation.quality_score
        }
    
    def _update_usage_stats(self, usage_session: QueryUsageSession):
        """Update usage statistics"""
        self.usage_stats['total_sessions'] += 1
        self.usage_stats['total_payments_distributed'] += usage_session.creator_distribution
        self.usage_stats['total_system_fees'] += usage_session.system_fee
        
        # Update averages
        total_sources = sum(len(s.sources_used) for s in self.usage_sessions.values())
        self.usage_stats['average_sources_per_query'] = total_sources / self.usage_stats['total_sessions']
        
        if total_sources > 0:
            self.usage_stats['average_payment_per_source'] = self.usage_stats['total_payments_distributed'] / total_sources
        
        # Update distribution by level
        for source in usage_session.sources_used:
            if source.payment_eligible:
                payment_amount = usage_session.payment_calculations.get(source.paper_id, 0.0)
                self.usage_stats['payment_distribution_by_level'][source.contribution_level] += payment_amount
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage and payment statistics"""
        return {
            **self.usage_stats,
            'active_sessions': len(self.usage_sessions),
            'total_revenue': self.usage_stats['total_payments_distributed'] + self.usage_stats['total_system_fees']
        }
    
    def get_session_details(self, session_id: str) -> Optional[QueryUsageSession]:
        """Get detailed information about a specific session"""
        return self.usage_sessions.get(session_id)
    
    async def generate_usage_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive usage report for a session"""
        session = self.usage_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        return {
            'session_info': {
                'session_id': session.session_id,
                'query': session.query,
                'user_id': session.user_id,
                'timestamp': session.session_timestamp.isoformat(),
                'processing_time': session.processing_time
            },
            'financial_summary': {
                'total_cost': session.total_cost,
                'system_fee': session.system_fee,
                'creator_distribution': session.creator_distribution,
                'payment_breakdown': session.payment_calculations
            },
            'source_usage': [
                {
                    'paper_id': source.paper_id,
                    'title': source.title,
                    'authors': source.authors,
                    'contribution_level': source.contribution_level.value,
                    'contribution_weight': source.contribution_weight,
                    'payment_amount': session.payment_calculations.get(source.paper_id, 0.0),
                    'usage_context': source.usage_context,
                    'citation_position': source.citation_position
                }
                for source in session.sources_used
            ],
            'pipeline_metrics': session.pipeline_stages,
            'audit_trail': session.audit_trail
        }


# Factory function for easy instantiation
async def create_attribution_usage_tracker(
    ftns_service: Optional[FTNSService] = None,
    provenance_engine: Optional[ProvenanceEngine] = None,
    royalty_engine: Optional[ContentRoyaltyEngine] = None
) -> AttributionUsageTracker:
    """Create and initialize an attribution usage tracker"""
    tracker = AttributionUsageTracker(ftns_service, provenance_engine, royalty_engine)
    await tracker.initialize()
    return tracker
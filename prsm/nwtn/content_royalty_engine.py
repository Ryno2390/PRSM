#!/usr/bin/env python3
"""
NWTN Content Royalty Engine
===========================

This module implements sophisticated royalty calculation and distribution for 
content usage in NWTN reasoning. It ensures fair compensation for content creators
based on usage patterns, query complexity, and content quality metrics.

Key Features:
1. Multi-factor royalty calculation (complexity, importance, quality, user tier)
2. Real-time royalty distribution via FTNS tokens
3. Transparent audit trails for all royalty transactions
4. Performance optimization for high-volume usage
5. Integration with PRSM's provenance and FTNS systems

Usage:
    from prsm.nwtn.content_royalty_engine import ContentRoyaltyEngine
    
    royalty_engine = ContentRoyaltyEngine()
    await royalty_engine.initialize()
    
    royalties = await royalty_engine.calculate_usage_royalty(
        content_sources=content_list,
        query_complexity=QueryComplexity.COMPLEX,
        user_tier="premium"
    )
    
    await royalty_engine.distribute_royalties(royalties, session_id)
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from decimal import Decimal, getcontext
import statistics
import json

import structlog

from prsm.core.config import get_settings
from prsm.provenance.enhanced_provenance_system import EnhancedProvenanceSystem
from prsm.tokenomics.ftns_service import FTNSService
from prsm.core.database_service import get_database_service

# Set precision for financial calculations
getcontext().prec = 18

logger = structlog.get_logger(__name__)
settings = get_settings()


class QueryComplexity(str, Enum):
    """Query complexity levels for royalty calculation"""
    SIMPLE = "simple"          # Direct factual questions
    MODERATE = "moderate"      # Analysis requiring 2-3 reasoning modes
    COMPLEX = "complex"        # Multi-step analysis requiring 4+ reasoning modes
    BREAKTHROUGH = "breakthrough"  # Research-level queries requiring full validation


class UserTier(str, Enum):
    """User tiers with different royalty rates"""
    BASIC = "basic"           # Standard users
    PREMIUM = "premium"       # Premium subscribers
    RESEARCHER = "researcher" # Academic/research users
    ENTERPRISE = "enterprise" # Enterprise customers
    CREATOR = "creator"       # Content creators (higher royalty rates)


class ContentImportance(str, Enum):
    """Content importance levels in reasoning"""
    CORE = "core"             # Central to the reasoning conclusion
    SUPPORTING = "supporting" # Provides supporting evidence
    REFERENCE = "reference"   # Background/contextual information
    MINIMAL = "minimal"       # Minor contribution to reasoning


@dataclass
class ContentUsageMetrics:
    """Metrics for content usage in reasoning"""
    content_id: UUID
    importance_level: ContentImportance
    confidence_contribution: float  # How much this content contributed to overall confidence
    reasoning_weight: float         # Weight in the reasoning process (0.0 to 1.0)
    quality_score: float           # Content quality score (0.0 to 1.0)
    citation_count: int            # Number of times this content has been cited
    recency_factor: float          # Recency adjustment (newer content may get bonus)
    domain_relevance: float        # How relevant to the query domain (0.0 to 1.0)


@dataclass  
class RoyaltyCalculation:
    """Result of royalty calculation for a content item"""
    content_id: UUID
    creator_id: str
    creator_address: Optional[str]
    base_royalty: Decimal
    complexity_multiplier: float
    quality_multiplier: float
    importance_multiplier: float
    user_tier_multiplier: float
    final_royalty: Decimal
    calculation_timestamp: datetime
    session_id: UUID
    reasoning_context: Dict[str, Any]


@dataclass
class RoyaltyDistributionResult:
    """Result of royalty distribution process"""
    session_id: UUID
    total_royalties_calculated: Decimal
    total_royalties_distributed: Decimal
    successful_distributions: int
    failed_distributions: int
    distribution_timestamp: datetime
    transaction_ids: List[str]
    creator_distributions: Dict[str, Decimal]
    audit_trail: List[Dict[str, Any]]


class ContentRoyaltyEngine:
    """
    NWTN Content Royalty Engine
    
    Calculates and distributes royalties for content usage in NWTN reasoning
    based on sophisticated multi-factor analysis including query complexity,
    content importance, quality metrics, and user tiers.
    """
    
    def __init__(self):
        self.provenance_system = None
        self.ftns_service = None
        self.database_service = None
        
        # Royalty calculation parameters
        self.base_royalty_rates = {
            QueryComplexity.SIMPLE: Decimal('0.01'),      # 0.01 FTNS per content
            QueryComplexity.MODERATE: Decimal('0.025'),   # 0.025 FTNS per content
            QueryComplexity.COMPLEX: Decimal('0.05'),     # 0.05 FTNS per content
            QueryComplexity.BREAKTHROUGH: Decimal('0.10') # 0.10 FTNS per content
        }
        
        # User tier multipliers (affect royalty rates)
        self.user_tier_multipliers = {
            UserTier.BASIC: 1.0,        # Standard rate
            UserTier.PREMIUM: 1.2,      # 20% bonus for premium users
            UserTier.RESEARCHER: 1.5,   # 50% bonus for researchers
            UserTier.ENTERPRISE: 2.0,   # 100% bonus for enterprise
            UserTier.CREATOR: 0.8       # 20% discount for creators (they earn from their own content)
        }
        
        # Content importance multipliers
        self.importance_multipliers = {
            ContentImportance.CORE: 2.0,        # Core content gets 2x royalty
            ContentImportance.SUPPORTING: 1.5,  # Supporting content gets 1.5x
            ContentImportance.REFERENCE: 1.0,   # Reference content gets standard rate
            ContentImportance.MINIMAL: 0.5      # Minimal contribution gets 0.5x
        }
        
        # Quality score multipliers (applied based on content quality)
        self.quality_thresholds = {
            0.9: 1.5,   # Exceptional quality: 1.5x multiplier
            0.8: 1.3,   # High quality: 1.3x multiplier  
            0.7: 1.1,   # Good quality: 1.1x multiplier
            0.6: 1.0,   # Average quality: standard rate
            0.0: 0.8    # Below average: 0.8x multiplier
        }
        
        # Performance tracking
        self.calculation_times: List[float] = []
        self.distribution_times: List[float] = []
        self.total_calculations: int = 0
        self.total_distributions: int = 0
        self.total_royalties_distributed: Decimal = Decimal('0')
        
        # Caching for performance
        self.creator_cache: Dict[UUID, Dict[str, Any]] = {}
        self.quality_cache: Dict[UUID, float] = {}
        
        logger.info("ContentRoyaltyEngine initialized")
    
    async def initialize(self):
        """Initialize the royalty engine with required services"""
        try:
            # Initialize service dependencies
            self.provenance_system = EnhancedProvenanceSystem()
            self.ftns_service = FTNSService()
            self.database_service = get_database_service()
            
            logger.info("✅ ContentRoyaltyEngine fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ContentRoyaltyEngine: {e}")
            raise
    
    async def calculate_usage_royalty(
        self,
        content_sources: List[UUID],
        query_complexity: QueryComplexity,
        user_tier: str = "basic",
        session_id: Optional[UUID] = None,
        reasoning_context: Optional[Dict[str, Any]] = None
    ) -> List[RoyaltyCalculation]:
        """
        Calculate royalties for content usage in NWTN reasoning
        
        Args:
            content_sources: List of content IDs used in reasoning
            query_complexity: Complexity level of the query
            user_tier: User tier for tier-based pricing
            session_id: Session ID for tracking
            reasoning_context: Additional context about the reasoning
            
        Returns:
            List of royalty calculations for each content item
        """
        start_time = time.perf_counter()
        session_id = session_id or uuid4()
        reasoning_context = reasoning_context or {}
        
        try:
            logger.info("Calculating usage royalties",
                       session_id=str(session_id),
                       content_count=len(content_sources),
                       complexity=query_complexity.value,
                       user_tier=user_tier)
            
            royalty_calculations: List[RoyaltyCalculation] = []
            
            # Get user tier enum
            try:
                user_tier_enum = UserTier(user_tier)
            except ValueError:
                logger.warning(f"Unknown user tier: {user_tier}, defaulting to BASIC")
                user_tier_enum = UserTier.BASIC
            
            # Get base royalty rate for complexity
            base_rate = self.base_royalty_rates[query_complexity]
            user_multiplier = self.user_tier_multipliers[user_tier_enum]
            
            # Calculate usage metrics for all content
            content_metrics = await self._analyze_content_usage_metrics(
                content_sources, reasoning_context
            )
            
            # Calculate royalties for each content item
            for content_id in content_sources:
                try:
                    metrics = content_metrics.get(content_id)
                    if not metrics:
                        logger.warning(f"No metrics found for content {content_id}")
                        continue
                    
                    # Get creator information
                    creator_info = await self._get_creator_info(content_id)
                    if not creator_info:
                        logger.warning(f"No creator info found for content {content_id}")
                        continue
                    
                    # Calculate multipliers
                    importance_multiplier = self.importance_multipliers[metrics.importance_level]
                    quality_multiplier = self._calculate_quality_multiplier(metrics.quality_score)
                    
                    # Calculate final royalty
                    base_royalty = base_rate * Decimal(str(metrics.reasoning_weight))
                    final_royalty = (
                        base_royalty * 
                        Decimal(str(user_multiplier)) *
                        Decimal(str(importance_multiplier)) *
                        Decimal(str(quality_multiplier))
                    )
                    
                    # Create royalty calculation record
                    calculation = RoyaltyCalculation(
                        content_id=content_id,
                        creator_id=creator_info['creator_id'],
                        creator_address=creator_info.get('ftns_address'),
                        base_royalty=base_royalty,
                        complexity_multiplier=float(user_multiplier),
                        quality_multiplier=quality_multiplier,
                        importance_multiplier=importance_multiplier,
                        user_tier_multiplier=user_multiplier,
                        final_royalty=final_royalty,
                        calculation_timestamp=datetime.now(timezone.utc),
                        session_id=session_id,
                        reasoning_context=reasoning_context
                    )
                    
                    royalty_calculations.append(calculation)
                    
                    logger.debug("Royalty calculated",
                               content_id=str(content_id),
                               creator_id=creator_info['creator_id'],
                               final_royalty=float(final_royalty),
                               importance=metrics.importance_level.value)
                
                except Exception as content_error:
                    logger.error(f"Failed to calculate royalty for content {content_id}: {content_error}")
                    continue
            
            # Performance tracking
            calculation_time = time.perf_counter() - start_time
            self.calculation_times.append(calculation_time)
            self.total_calculations += 1
            
            total_royalty = sum(calc.final_royalty for calc in royalty_calculations)
            
            logger.info("Royalty calculation completed",
                       session_id=str(session_id),
                       items_calculated=len(royalty_calculations),
                       total_royalty=float(total_royalty),
                       calculation_time_ms=calculation_time * 1000)
            
            return royalty_calculations
            
        except Exception as e:
            logger.error("Royalty calculation failed",
                        session_id=str(session_id),
                        error=str(e))
            raise
    
    async def distribute_royalties(
        self,
        royalty_calculations: List[RoyaltyCalculation],
        session_id: Optional[UUID] = None
    ) -> RoyaltyDistributionResult:
        """
        Distribute FTNS tokens to content creators based on royalty calculations
        
        Args:
            royalty_calculations: List of calculated royalties to distribute
            session_id: Session ID for tracking
            
        Returns:
            Distribution result with success/failure details
        """
        start_time = time.perf_counter()
        session_id = session_id or uuid4()
        
        try:
            logger.info("Starting royalty distribution",
                       session_id=str(session_id),
                       royalty_count=len(royalty_calculations))
            
            distribution_result = RoyaltyDistributionResult(
                session_id=session_id,
                total_royalties_calculated=Decimal('0'),
                total_royalties_distributed=Decimal('0'),
                successful_distributions=0,
                failed_distributions=0,
                distribution_timestamp=datetime.now(timezone.utc),
                transaction_ids=[],
                creator_distributions={},
                audit_trail=[]
            )
            
            # Group royalties by creator for efficient distribution
            creator_royalties: Dict[str, List[RoyaltyCalculation]] = {}
            for calc in royalty_calculations:
                creator_id = calc.creator_id
                if creator_id not in creator_royalties:
                    creator_royalties[creator_id] = []
                creator_royalties[creator_id].append(calc)
                distribution_result.total_royalties_calculated += calc.final_royalty
            
            # Distribute to each creator
            for creator_id, creator_calcs in creator_royalties.items():
                try:
                    # Calculate total royalty for this creator
                    total_creator_royalty = sum(calc.final_royalty for calc in creator_calcs)
                    
                    # Get creator's FTNS address
                    creator_address = creator_calcs[0].creator_address
                    if not creator_address:
                        logger.warning(f"No FTNS address for creator {creator_id}")
                        distribution_result.failed_distributions += len(creator_calcs)
                        continue
                    
                    # Distribute FTNS tokens
                    success = await self._distribute_to_creator(
                        creator_id=creator_id,
                        creator_address=creator_address,
                        royalty_amount=total_creator_royalty,
                        calculations=creator_calcs,
                        session_id=session_id
                    )
                    
                    if success:
                        distribution_result.successful_distributions += len(creator_calcs)
                        distribution_result.total_royalties_distributed += total_creator_royalty
                        distribution_result.creator_distributions[creator_id] = total_creator_royalty
                        
                        # Add to audit trail
                        distribution_result.audit_trail.append({
                            'creator_id': creator_id,
                            'creator_address': creator_address,
                            'royalty_amount': float(total_creator_royalty),
                            'content_count': len(creator_calcs),
                            'status': 'success',
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                        
                        logger.debug("Creator royalty distributed",
                                   creator_id=creator_id,
                                   royalty_amount=float(total_creator_royalty),
                                   content_count=len(creator_calcs))
                    else:
                        distribution_result.failed_distributions += len(creator_calcs)
                        
                        # Add failure to audit trail
                        distribution_result.audit_trail.append({
                            'creator_id': creator_id,
                            'creator_address': creator_address,
                            'royalty_amount': float(total_creator_royalty),
                            'content_count': len(creator_calcs),
                            'status': 'failed',
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                
                except Exception as creator_error:
                    logger.error(f"Failed to distribute royalty to creator {creator_id}: {creator_error}")
                    distribution_result.failed_distributions += len(creator_calcs)
                    continue
            
            # Performance tracking
            distribution_time = time.perf_counter() - start_time
            self.distribution_times.append(distribution_time)
            self.total_distributions += 1
            self.total_royalties_distributed += distribution_result.total_royalties_distributed
            
            # Store distribution record in database
            await self._store_distribution_record(distribution_result)
            
            logger.info("Royalty distribution completed",
                       session_id=str(session_id),
                       successful_distributions=distribution_result.successful_distributions,
                       failed_distributions=distribution_result.failed_distributions,
                       total_distributed=float(distribution_result.total_royalties_distributed),
                       distribution_time_ms=distribution_time * 1000)
            
            return distribution_result
            
        except Exception as e:
            logger.error("Royalty distribution failed",
                        session_id=str(session_id),
                        error=str(e))
            raise
    
    async def get_creator_earnings_summary(
        self,
        creator_id: str,
        time_period: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get earnings summary for a specific creator
        
        Args:
            creator_id: Creator to get earnings for
            time_period: Optional time period (defaults to last 30 days)
            
        Returns:
            Earnings summary with detailed breakdown
        """
        try:
            # Default to last 30 days
            if time_period is None:
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=30)
                time_period = (start_time, end_time)
            
            # Get distribution records from database
            earnings_data = await self.database_service.get_creator_earnings(
                creator_id=creator_id,
                start_time=time_period[0],
                end_time=time_period[1]
            )
            
            if not earnings_data:
                return {
                    'creator_id': creator_id,
                    'total_earnings': 0.0,
                    'total_content_used': 0,
                    'average_per_usage': 0.0,
                    'earnings_by_complexity': {},
                    'top_earning_content': [],
                    'time_period': {
                        'start': time_period[0].isoformat(),
                        'end': time_period[1].isoformat()
                    }
                }
            
            # Analyze earnings data
            total_earnings = sum(Decimal(str(record['royalty_amount'])) for record in earnings_data)
            total_usage_events = len(earnings_data)
            
            # Group by complexity
            complexity_earnings = {}
            for record in earnings_data:
                complexity = record.get('query_complexity', 'unknown')
                if complexity not in complexity_earnings:
                    complexity_earnings[complexity] = {'count': 0, 'earnings': Decimal('0')}
                complexity_earnings[complexity]['count'] += 1
                complexity_earnings[complexity]['earnings'] += Decimal(str(record['royalty_amount']))
            
            # Get top earning content
            content_earnings = {}
            for record in earnings_data:
                content_id = record['content_id']
                if content_id not in content_earnings:
                    content_earnings[content_id] = {'count': 0, 'earnings': Decimal('0')}
                content_earnings[content_id]['count'] += 1
                content_earnings[content_id]['earnings'] += Decimal(str(record['royalty_amount']))
            
            top_content = sorted(
                content_earnings.items(),
                key=lambda x: x[1]['earnings'],
                reverse=True
            )[:10]
            
            summary = {
                'creator_id': creator_id,
                'total_earnings': float(total_earnings),
                'total_content_used': total_usage_events,
                'average_per_usage': float(total_earnings / total_usage_events) if total_usage_events > 0 else 0.0,
                'earnings_by_complexity': {
                    complexity: {
                        'count': data['count'],
                        'earnings': float(data['earnings']),
                        'average': float(data['earnings'] / data['count']) if data['count'] > 0 else 0.0
                    }
                    for complexity, data in complexity_earnings.items()
                },
                'top_earning_content': [
                    {
                        'content_id': content_id,
                        'usage_count': data['count'],
                        'total_earnings': float(data['earnings']),
                        'average_per_usage': float(data['earnings'] / data['count']) if data['count'] > 0 else 0.0
                    }
                    for content_id, data in top_content
                ],
                'time_period': {
                    'start': time_period[0].isoformat(),
                    'end': time_period[1].isoformat(),
                    'duration_days': (time_period[1] - time_period[0]).days
                }
            }
            
            logger.info("Creator earnings summary generated",
                       creator_id=creator_id,
                       total_earnings=float(total_earnings),
                       usage_events=total_usage_events)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get creator earnings summary: {e}")
            raise
    
    # === Private Helper Methods ===
    
    async def _analyze_content_usage_metrics(
        self,
        content_sources: List[UUID],
        reasoning_context: Dict[str, Any]
    ) -> Dict[UUID, ContentUsageMetrics]:
        """Analyze usage metrics for content items"""
        try:
            metrics: Dict[UUID, ContentUsageMetrics] = {}
            
            # Get reasoning weights and confidence contributions from context
            reasoning_weights = reasoning_context.get('content_weights', {})
            confidence_contributions = reasoning_context.get('confidence_contributions', {})
            
            for content_id in content_sources:
                try:
                    # Get quality score (cached for performance)
                    quality_score = await self._get_content_quality_score(content_id)
                    
                    # Determine importance level based on reasoning weight
                    reasoning_weight = reasoning_weights.get(str(content_id), 1.0 / len(content_sources))
                    importance_level = self._classify_content_importance(reasoning_weight)
                    
                    # Get other metrics
                    citation_count = await self._get_content_citation_count(content_id)
                    recency_factor = await self._calculate_recency_factor(content_id)
                    domain_relevance = reasoning_context.get('domain_relevance', {}).get(str(content_id), 1.0)
                    
                    metrics[content_id] = ContentUsageMetrics(
                        content_id=content_id,
                        importance_level=importance_level,
                        confidence_contribution=confidence_contributions.get(str(content_id), 0.0),
                        reasoning_weight=reasoning_weight,
                        quality_score=quality_score,
                        citation_count=citation_count,
                        recency_factor=recency_factor,
                        domain_relevance=domain_relevance
                    )
                
                except Exception as content_error:
                    logger.warning(f"Failed to analyze metrics for content {content_id}: {content_error}")
                    # Provide default metrics
                    metrics[content_id] = ContentUsageMetrics(
                        content_id=content_id,
                        importance_level=ContentImportance.REFERENCE,
                        confidence_contribution=0.0,
                        reasoning_weight=1.0 / len(content_sources),
                        quality_score=0.7,  # Default quality
                        citation_count=0,
                        recency_factor=1.0,
                        domain_relevance=1.0
                    )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to analyze content usage metrics: {e}")
            return {}
    
    def _classify_content_importance(self, reasoning_weight: float) -> ContentImportance:
        """Classify content importance based on reasoning weight"""
        if reasoning_weight >= 0.4:
            return ContentImportance.CORE
        elif reasoning_weight >= 0.2:
            return ContentImportance.SUPPORTING
        elif reasoning_weight >= 0.05:
            return ContentImportance.REFERENCE
        else:
            return ContentImportance.MINIMAL
    
    def _calculate_quality_multiplier(self, quality_score: float) -> float:
        """Calculate quality multiplier based on quality score"""
        for threshold in sorted(self.quality_thresholds.keys(), reverse=True):
            if quality_score >= threshold:
                return self.quality_thresholds[threshold]
        return self.quality_thresholds[0.0]  # Default for below average
    
    async def _get_creator_info(self, content_id: UUID) -> Optional[Dict[str, Any]]:
        """Get creator information for content"""
        try:
            # Check cache first
            if content_id in self.creator_cache:
                return self.creator_cache[content_id]
            
            # Get from provenance system
            attribution_chain = await self.provenance_system._load_attribution_chain(content_id)
            if not attribution_chain:
                return None
            
            creator_info = {
                'creator_id': attribution_chain.original_creator,
                'ftns_address': attribution_chain.creator_address,
                'creation_timestamp': attribution_chain.creation_timestamp,
                'platform_source': attribution_chain.platform_source
            }
            
            # Cache for performance
            self.creator_cache[content_id] = creator_info
            
            return creator_info
            
        except Exception as e:
            logger.error(f"Failed to get creator info for {content_id}: {e}")
            return None
    
    async def _get_content_quality_score(self, content_id: UUID) -> float:
        """Get quality score for content (cached)"""
        try:
            # Check cache first
            if content_id in self.quality_cache:
                return self.quality_cache[content_id]
            
            # Calculate quality score based on various factors
            # This is a simplified implementation - would be more sophisticated in production
            citation_count = await self._get_content_citation_count(content_id)
            usage_frequency = await self._get_content_usage_frequency(content_id)
            
            # Base quality score formula
            quality_score = min(1.0, 0.5 + (citation_count * 0.01) + (usage_frequency * 0.02))
            
            # Cache the result
            self.quality_cache[content_id] = quality_score
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Failed to get quality score for {content_id}: {e}")
            return 0.7  # Default quality score
    
    async def _get_content_citation_count(self, content_id: UUID) -> int:
        """Get citation count for content"""
        try:
            # This would query the database for citation/reference count
            # Placeholder implementation
            return 5  # Default citation count
        except Exception:
            return 0
    
    async def _get_content_usage_frequency(self, content_id: UUID) -> float:
        """Get usage frequency for content"""
        try:
            # This would calculate how often content is used
            # Placeholder implementation
            return 0.1  # Default usage frequency
        except Exception:
            return 0.0
    
    async def _calculate_recency_factor(self, content_id: UUID) -> float:
        """Calculate recency factor for content"""
        try:
            # Get content creation date
            creator_info = await self._get_creator_info(content_id)
            if not creator_info:
                return 1.0
            
            creation_date = creator_info['creation_timestamp']
            days_old = (datetime.now(timezone.utc) - creation_date).days
            
            # Newer content gets slight bonus (within 30 days)
            if days_old <= 30:
                return 1.1
            elif days_old <= 90:
                return 1.05
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    async def _distribute_to_creator(
        self,
        creator_id: str,
        creator_address: str,
        royalty_amount: Decimal,
        calculations: List[RoyaltyCalculation],
        session_id: UUID
    ) -> bool:
        """Distribute FTNS tokens to a specific creator"""
        try:
            # Create distribution metadata
            distribution_metadata = {
                'session_id': str(session_id),
                'content_count': len(calculations),
                'content_ids': [str(calc.content_id) for calc in calculations],
                'distribution_type': 'content_usage_royalty',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Use FTNS service to distribute tokens
            success = await self.ftns_service.reward_contribution(
                user_id=creator_address,  # Use FTNS address as user ID
                contribution_type="content_usage",
                value=float(royalty_amount),
                metadata=distribution_metadata
            )
            
            if success:
                logger.debug("FTNS tokens distributed to creator",
                           creator_id=creator_id,
                           creator_address=creator_address,
                           amount=float(royalty_amount))
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to distribute FTNS to creator {creator_id}: {e}")
            return False
    
    async def _store_distribution_record(self, distribution_result: RoyaltyDistributionResult):
        """Store distribution record in database for audit trail"""
        try:
            record = {
                'session_id': str(distribution_result.session_id),
                'distribution_timestamp': distribution_result.distribution_timestamp,
                'total_royalties_calculated': float(distribution_result.total_royalties_calculated),
                'total_royalties_distributed': float(distribution_result.total_royalties_distributed),
                'successful_distributions': distribution_result.successful_distributions,
                'failed_distributions': distribution_result.failed_distributions,
                'creator_distributions': {
                    creator: float(amount) 
                    for creator, amount in distribution_result.creator_distributions.items()
                },
                'audit_trail': distribution_result.audit_trail
            }
            
            await self.database_service.store_royalty_distribution_record(record)
            
        except Exception as e:
            logger.error(f"Failed to store distribution record: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the royalty engine"""
        return {
            'total_calculations': self.total_calculations,
            'total_distributions': self.total_distributions,
            'total_royalties_distributed': float(self.total_royalties_distributed),
            'average_calculation_time_ms': (
                sum(self.calculation_times) / len(self.calculation_times) * 1000
                if self.calculation_times else 0
            ),
            'average_distribution_time_ms': (
                sum(self.distribution_times) / len(self.distribution_times) * 1000
                if self.distribution_times else 0
            ),
            'cache_stats': {
                'creator_cache_size': len(self.creator_cache),
                'quality_cache_size': len(self.quality_cache)
            }
        }


# Global content royalty engine instance
_content_royalty_engine = None

async def get_content_royalty_engine() -> ContentRoyaltyEngine:
    """Get the global content royalty engine instance"""
    global _content_royalty_engine
    if _content_royalty_engine is None:
        _content_royalty_engine = ContentRoyaltyEngine()
        await _content_royalty_engine.initialize()
    return _content_royalty_engine
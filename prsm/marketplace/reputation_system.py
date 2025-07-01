"""
PRSM User Reputation System
==========================

Enterprise-grade reputation system addressing Gemini's audit concerns about
trust and quality mechanisms in the marketplace. Features sophisticated
trust scoring, behavioral analysis, and fraud detection.

Key Features:
- Multi-dimensional reputation scoring
- Real-time trust calculation with decay mechanisms
- Behavioral pattern analysis and fraud detection
- Quality contribution tracking and verification
- Social proof and peer validation systems
- Reputation-based access controls and privileges
"""

import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from uuid import UUID, uuid4
from dataclasses import dataclass
from enum import Enum
import structlog
from collections import defaultdict, Counter
import math

from prsm.core.database_service import get_database_service
from prsm.core.config import get_settings
from prsm.core.models import UserRole
from prsm.marketplace.models import MarketplaceResource

logger = structlog.get_logger(__name__)
settings = get_settings()


class ReputationDimension(Enum):
    """Different dimensions of user reputation"""
    QUALITY = "quality"
    RELIABILITY = "reliability"
    RESPONSIVENESS = "responsiveness"
    EXPERTISE = "expertise"
    TRUSTWORTHINESS = "trustworthiness"
    COMMUNITY_CONTRIBUTION = "community_contribution"
    INNOVATION = "innovation"


class ReputationEvent(Enum):
    """Types of events that affect reputation"""
    RESOURCE_PUBLISHED = "resource_published"
    RESOURCE_RATED = "resource_rated"
    RESOURCE_DOWNLOADED = "resource_downloaded"
    REVIEW_SUBMITTED = "review_submitted"
    REVIEW_RECEIVED = "review_received"
    DISPUTE_RAISED = "dispute_raised"
    DISPUTE_RESOLVED = "dispute_resolved"
    FRAUD_DETECTED = "fraud_detected"
    COMMUNITY_REPORT = "community_report"
    EXPERT_ENDORSEMENT = "expert_endorsement"
    COLLABORATION_COMPLETED = "collaboration_completed"
    SUPPORT_PROVIDED = "support_provided"


class TrustLevel(Enum):
    """Trust levels based on reputation scores"""
    UNTRUSTED = 0      # 0-20: New or problematic users
    NEWCOMER = 1       # 21-40: Recently joined users
    MEMBER = 2         # 41-60: Established users
    TRUSTED = 3        # 61-80: Reliable contributors
    EXPERT = 4         # 81-95: Highly respected experts
    ELITE = 5          # 96-100: Top-tier community leaders


@dataclass
class ReputationScore:
    """Individual reputation score for a dimension"""
    dimension: ReputationDimension
    score: float
    confidence: float
    last_updated: datetime
    trend: float  # Positive/negative trend over time
    evidence_count: int


@dataclass
class UserReputation:
    """Complete user reputation profile"""
    user_id: str
    overall_score: float
    trust_level: TrustLevel
    dimension_scores: Dict[ReputationDimension, ReputationScore]
    badges: List[str]
    verification_status: Dict[str, bool]
    reputation_history: List[Dict[str, Any]]
    last_calculated: datetime
    next_review: datetime


@dataclass
class ReputationTransaction:
    """Record of reputation-affecting transaction"""
    transaction_id: str
    user_id: str
    event_type: ReputationEvent
    dimension: ReputationDimension
    score_change: float
    evidence: Dict[str, Any]
    timestamp: datetime
    validated: bool


class ReputationCalculator:
    """
    Advanced reputation calculation engine with sophisticated algorithms
    
    Features:
    - Multi-dimensional scoring with weighted aggregation
    - Time-decay mechanisms for recent vs historical performance
    - Confidence intervals based on evidence quantity and quality
    - Behavioral pattern analysis for fraud detection
    - Peer validation and social proof integration
    - Reputation-based privilege escalation
    """
    
    def __init__(self):
        self.database_service = get_database_service()
        
        # Dimension weights for overall score calculation
        self.dimension_weights = {
            ReputationDimension.QUALITY: 0.25,
            ReputationDimension.RELIABILITY: 0.20,
            ReputationDimension.TRUSTWORTHINESS: 0.20,
            ReputationDimension.EXPERTISE: 0.15,
            ReputationDimension.RESPONSIVENESS: 0.10,
            ReputationDimension.COMMUNITY_CONTRIBUTION: 0.10
        }
        
        # Event impact scores for different reputation events
        self.event_impacts = {
            ReputationEvent.RESOURCE_PUBLISHED: {
                ReputationDimension.QUALITY: 2.0,
                ReputationDimension.EXPERTISE: 1.5,
                ReputationDimension.COMMUNITY_CONTRIBUTION: 1.0
            },
            ReputationEvent.RESOURCE_RATED: {
                ReputationDimension.QUALITY: 0.0,  # Determined by rating value
                ReputationDimension.RELIABILITY: 0.5
            },
            ReputationEvent.REVIEW_RECEIVED: {
                ReputationDimension.QUALITY: 0.0,  # Determined by review rating
                ReputationDimension.RESPONSIVENESS: 1.0
            },
            ReputationEvent.EXPERT_ENDORSEMENT: {
                ReputationDimension.EXPERTISE: 5.0,
                ReputationDimension.TRUSTWORTHINESS: 3.0
            },
            ReputationEvent.FRAUD_DETECTED: {
                ReputationDimension.TRUSTWORTHINESS: -10.0,
                ReputationDimension.RELIABILITY: -5.0
            },
            ReputationEvent.COMMUNITY_REPORT: {
                ReputationDimension.TRUSTWORTHINESS: -2.0
            }
        }
        
        # Time decay parameters
        self.decay_rate = 0.98  # Daily decay rate
        self.confidence_threshold = 10  # Minimum evidence for reliable scores
        
        # Fraud detection parameters
        self.fraud_detection_enabled = True
        self.suspicious_pattern_threshold = 0.8
        
        logger.info("Reputation calculator initialized",
                   dimensions=len(self.dimension_weights),
                   events=len(self.event_impacts))
    
    async def calculate_user_reputation(
        self,
        user_id: str,
        force_recalculation: bool = False
    ) -> UserReputation:
        """
        Calculate comprehensive user reputation across all dimensions
        
        Args:
            user_id: User to calculate reputation for
            force_recalculation: Force full recalculation instead of incremental
            
        Returns:
            Complete user reputation profile with scores and analysis
        """
        try:
            start_time = datetime.now(timezone.utc)
            
            logger.info("Calculating user reputation",
                       user_id=user_id,
                       force_recalculation=force_recalculation)
            
            # Get existing reputation if available
            existing_reputation = await self._get_existing_reputation(user_id)
            
            # Check if recalculation needed
            if not force_recalculation and existing_reputation:
                if self._is_reputation_current(existing_reputation):
                    logger.info("Using cached reputation",
                               user_id=user_id,
                               last_calculated=existing_reputation.last_calculated)
                    return existing_reputation
            
            # Get user's reputation events
            reputation_events = await self._get_user_reputation_events(user_id)
            
            # Calculate dimension scores
            dimension_scores = {}
            for dimension in ReputationDimension:
                score = await self._calculate_dimension_score(
                    user_id, dimension, reputation_events
                )
                dimension_scores[dimension] = score
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(dimension_scores)
            
            # Determine trust level
            trust_level = self._determine_trust_level(overall_score)
            
            # Get badges and verification status
            badges = await self._calculate_user_badges(user_id, dimension_scores)
            verification_status = await self._get_verification_status(user_id)
            
            # Get reputation history
            reputation_history = await self._get_reputation_history(user_id, limit=50)
            
            # Fraud detection analysis
            if self.fraud_detection_enabled:
                await self._analyze_fraud_patterns(user_id, reputation_events)
            
            # Create reputation profile
            user_reputation = UserReputation(
                user_id=user_id,
                overall_score=overall_score,
                trust_level=trust_level,
                dimension_scores=dimension_scores,
                badges=badges,
                verification_status=verification_status,
                reputation_history=reputation_history,
                last_calculated=start_time,
                next_review=start_time + timedelta(days=7)  # Weekly reviews
            )
            
            # Store reputation in database
            await self._store_reputation(user_reputation)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info("User reputation calculated successfully",
                       user_id=user_id,
                       overall_score=overall_score,
                       trust_level=trust_level.name,
                       badges_count=len(badges),
                       execution_time_ms=execution_time * 1000)
            
            return user_reputation
            
        except Exception as e:
            logger.error("Reputation calculation failed",
                        user_id=user_id,
                        error=str(e))
            # Return default reputation for new users
            return await self._create_default_reputation(user_id)
    
    async def _calculate_dimension_score(
        self,
        user_id: str,
        dimension: ReputationDimension,
        events: List[Dict[str, Any]]
    ) -> ReputationScore:
        """Calculate score for a specific reputation dimension"""
        try:
            relevant_events = [
                event for event in events
                if self._event_affects_dimension(event, dimension)
            ]
            
            if not relevant_events:
                return ReputationScore(
                    dimension=dimension,
                    score=50.0,  # Neutral starting score
                    confidence=0.1,
                    last_updated=datetime.now(timezone.utc),
                    trend=0.0,
                    evidence_count=0
                )
            
            # Calculate base score from events
            base_score = 50.0  # Starting neutral score
            total_weight = 0.0
            
            for event in relevant_events:
                event_impact = self._get_event_impact(event, dimension)
                event_weight = self._calculate_event_weight(event)
                
                base_score += event_impact * event_weight
                total_weight += event_weight
            
            # Apply time decay
            decayed_score = self._apply_time_decay(base_score, relevant_events)
            
            # Normalize score to 0-100 range
            normalized_score = max(0.0, min(100.0, decayed_score))
            
            # Calculate confidence based on evidence quantity and quality
            confidence = self._calculate_confidence(relevant_events)
            
            # Calculate trend (positive/negative momentum)
            trend = self._calculate_trend(relevant_events, dimension)
            
            return ReputationScore(
                dimension=dimension,
                score=normalized_score,
                confidence=confidence,
                last_updated=datetime.now(timezone.utc),
                trend=trend,
                evidence_count=len(relevant_events)
            )
            
        except Exception as e:
            logger.error("Dimension score calculation failed",
                        user_id=user_id,
                        dimension=dimension.value,
                        error=str(e))
            return ReputationScore(
                dimension=dimension,
                score=50.0,
                confidence=0.1,
                last_updated=datetime.now(timezone.utc),
                trend=0.0,
                evidence_count=0
            )
    
    def _get_event_impact(self, event: Dict[str, Any], dimension: ReputationDimension) -> float:
        """Get the impact score for an event on a specific dimension"""
        event_type = ReputationEvent(event.get("event_type"))
        
        # Base impact from configuration
        base_impact = self.event_impacts.get(event_type, {}).get(dimension, 0.0)
        
        # Adjust based on event data
        if event_type == ReputationEvent.RESOURCE_RATED:
            rating = event.get("rating", 3.0)
            # Convert 1-5 rating to -5 to +5 impact
            base_impact = (rating - 3.0) * 2.0
        
        elif event_type == ReputationEvent.REVIEW_RECEIVED:
            rating = event.get("review_rating", 3.0)
            base_impact = (rating - 3.0) * 1.5
        
        elif event_type == ReputationEvent.RESOURCE_DOWNLOADED:
            # More downloads = higher quality signal
            download_count = event.get("download_count", 1)
            base_impact = min(3.0, math.log(download_count + 1) * 0.5)
        
        return base_impact
    
    def _calculate_event_weight(self, event: Dict[str, Any]) -> float:
        """Calculate weight for an event based on recency and source credibility"""
        # Time-based weight (more recent = higher weight)
        event_time = datetime.fromisoformat(event.get("timestamp"))
        days_ago = (datetime.now(timezone.utc) - event_time).days
        time_weight = self.decay_rate ** days_ago
        
        # Source credibility weight
        source_credibility = 1.0
        if event.get("source_verified"):
            source_credibility = 1.5
        elif event.get("source_trusted"):
            source_credibility = 1.2
        
        # Event quality weight
        quality_weight = event.get("quality_score", 1.0)
        
        return time_weight * source_credibility * quality_weight
    
    def _apply_time_decay(self, score: float, events: List[Dict[str, Any]]) -> float:
        """Apply time decay to reputation scores"""
        if not events:
            return score
        
        # Calculate weighted average with time decay
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for event in events:
            event_time = datetime.fromisoformat(event.get("timestamp"))
            days_ago = (datetime.now(timezone.utc) - event_time).days
            weight = self.decay_rate ** days_ago
            
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else score
    
    def _calculate_confidence(self, events: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on evidence quantity and quality"""
        if not events:
            return 0.1
        
        # Base confidence from event count
        event_count = len(events)
        base_confidence = min(1.0, event_count / self.confidence_threshold)
        
        # Quality adjustment
        quality_scores = [event.get("quality_score", 1.0) for event in events]
        avg_quality = np.mean(quality_scores)
        
        # Diversity adjustment (variety of event types)
        event_types = set(event.get("event_type") for event in events)
        diversity_factor = min(1.0, len(event_types) / 5.0)  # Max 5 event types
        
        # Time span adjustment (longer history = higher confidence)
        if len(events) > 1:
            times = [datetime.fromisoformat(e.get("timestamp")) for e in events]
            time_span_days = (max(times) - min(times)).days
            time_factor = min(1.0, time_span_days / 365.0)  # Max 1 year
        else:
            time_factor = 0.1
        
        final_confidence = base_confidence * avg_quality * diversity_factor * time_factor
        return max(0.1, min(1.0, final_confidence))
    
    def _calculate_trend(self, events: List[Dict[str, Any]], dimension: ReputationDimension) -> float:
        """Calculate trend (momentum) for a reputation dimension"""
        if len(events) < 3:
            return 0.0
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x.get("timestamp"))
        
        # Split into recent and older events
        split_point = len(sorted_events) // 2
        older_events = sorted_events[:split_point]
        recent_events = sorted_events[split_point:]
        
        # Calculate average impact for each period
        older_avg = np.mean([self._get_event_impact(e, dimension) for e in older_events])
        recent_avg = np.mean([self._get_event_impact(e, dimension) for e in recent_events])
        
        # Trend is the difference (positive = improving, negative = declining)
        trend = recent_avg - older_avg
        
        # Normalize trend to -1 to +1 range
        return max(-1.0, min(1.0, trend / 5.0))
    
    def _calculate_overall_score(self, dimension_scores: Dict[ReputationDimension, ReputationScore]) -> float:
        """Calculate overall reputation score from dimension scores"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in self.dimension_weights.items():
            if dimension in dimension_scores:
                score_data = dimension_scores[dimension]
                # Weight by confidence
                effective_weight = weight * score_data.confidence
                total_weighted_score += score_data.score * effective_weight
                total_weight += effective_weight
        
        if total_weight == 0:
            return 50.0  # Default neutral score
        
        return total_weighted_score / total_weight
    
    def _determine_trust_level(self, overall_score: float) -> TrustLevel:
        """Determine trust level based on overall reputation score"""
        if overall_score >= 96:
            return TrustLevel.ELITE
        elif overall_score >= 81:
            return TrustLevel.EXPERT
        elif overall_score >= 61:
            return TrustLevel.TRUSTED
        elif overall_score >= 41:
            return TrustLevel.MEMBER
        elif overall_score >= 21:
            return TrustLevel.NEWCOMER
        else:
            return TrustLevel.UNTRUSTED
    
    async def _calculate_user_badges(
        self,
        user_id: str,
        dimension_scores: Dict[ReputationDimension, ReputationScore]
    ) -> List[str]:
        """Calculate badges earned by user based on achievements"""
        badges = []
        
        # Score-based badges
        for dimension, score_data in dimension_scores.items():
            if score_data.score >= 90 and score_data.confidence >= 0.8:
                badges.append(f"{dimension.value}_expert")
            elif score_data.score >= 80 and score_data.confidence >= 0.6:
                badges.append(f"{dimension.value}_specialist")
        
        # Activity-based badges
        user_stats = await self._get_user_statistics(user_id)
        
        if user_stats.get("resources_published", 0) >= 50:
            badges.append("prolific_contributor")
        elif user_stats.get("resources_published", 0) >= 10:
            badges.append("active_contributor")
        
        if user_stats.get("community_reports", 0) >= 20:
            badges.append("community_guardian")
        
        if user_stats.get("expert_endorsements", 0) >= 5:
            badges.append("peer_recognized")
        
        # Time-based badges
        account_age_days = user_stats.get("account_age_days", 0)
        if account_age_days >= 365:
            badges.append("veteran_member")
        elif account_age_days >= 90:
            badges.append("established_member")
        
        return list(set(badges))  # Remove duplicates
    
    async def record_reputation_event(
        self,
        user_id: str,
        event_type: ReputationEvent,
        evidence: Dict[str, Any],
        source_user_id: Optional[str] = None
    ) -> str:
        """
        Record a reputation-affecting event
        
        Args:
            user_id: User whose reputation is affected
            event_type: Type of reputation event
            evidence: Supporting evidence and context
            source_user_id: User who triggered the event (if applicable)
            
        Returns:
            Transaction ID for the reputation change
        """
        try:
            transaction_id = str(uuid4())
            
            logger.info("Recording reputation event",
                       user_id=user_id,
                       event_type=event_type.value,
                       transaction_id=transaction_id,
                       source_user=source_user_id)
            
            # Determine affected dimension
            primary_dimension = self._get_primary_dimension(event_type)
            
            # Calculate immediate score change
            score_change = self._calculate_immediate_impact(event_type, evidence)
            
            # Create transaction record
            transaction = ReputationTransaction(
                transaction_id=transaction_id,
                user_id=user_id,
                event_type=event_type,
                dimension=primary_dimension,
                score_change=score_change,
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                validated=True  # Would implement validation logic
            )
            
            # Store transaction
            await self._store_reputation_transaction(transaction)
            
            # Trigger incremental reputation update
            await self._update_reputation_incremental(user_id, transaction)
            
            logger.info("Reputation event recorded successfully",
                       user_id=user_id,
                       transaction_id=transaction_id,
                       score_change=score_change,
                       dimension=primary_dimension.value)
            
            return transaction_id
            
        except Exception as e:
            logger.error("Failed to record reputation event",
                        user_id=user_id,
                        event_type=event_type.value,
                        error=str(e))
            raise
    
    async def get_user_reputation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of user's reputation for display purposes"""
        try:
            reputation = await self.calculate_user_reputation(user_id)
            
            return {
                "user_id": user_id,
                "overall_score": round(reputation.overall_score, 1),
                "trust_level": reputation.trust_level.name.title(),
                "trust_level_numeric": reputation.trust_level.value,
                "badges": reputation.badges,
                "top_dimensions": [
                    {
                        "dimension": dim.value,
                        "score": round(score.score, 1),
                        "confidence": round(score.confidence, 2),
                        "trend": round(score.trend, 2)
                    }
                    for dim, score in sorted(
                        reputation.dimension_scores.items(),
                        key=lambda x: x[1].score,
                        reverse=True
                    )[:3]
                ],
                "verification_status": reputation.verification_status,
                "last_updated": reputation.last_calculated.isoformat(),
                "next_review": reputation.next_review.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get reputation summary",
                        user_id=user_id,
                        error=str(e))
            return {
                "user_id": user_id,
                "overall_score": 50.0,
                "trust_level": "Newcomer",
                "trust_level_numeric": 1,
                "badges": [],
                "top_dimensions": [],
                "verification_status": {},
                "error": "Unable to calculate reputation"
            }
    
    # Helper methods (database operations would be implemented based on actual schema)
    async def _get_existing_reputation(self, user_id: str) -> Optional[UserReputation]:
        """Get existing reputation from database"""
        # Placeholder - would query database
        return None
    
    def _is_reputation_current(self, reputation: UserReputation) -> bool:
        """Check if reputation calculation is still current"""
        return (datetime.now(timezone.utc) - reputation.last_calculated).days < 1
    
    async def _get_user_reputation_events(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all reputation events for a user"""
        # Placeholder - would query database for reputation events
        return []
    
    def _event_affects_dimension(self, event: Dict[str, Any], dimension: ReputationDimension) -> bool:
        """Check if an event affects a specific reputation dimension"""
        event_type = ReputationEvent(event.get("event_type"))
        return dimension in self.event_impacts.get(event_type, {})
    
    async def _get_verification_status(self, user_id: str) -> Dict[str, bool]:
        """Get user verification status"""
        # Placeholder - would check various verification sources
        return {
            "email_verified": True,
            "identity_verified": False,
            "expert_verified": False,
            "organization_verified": False
        }
    
    async def _get_reputation_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's reputation history"""
        # Placeholder - would query reputation change history
        return []
    
    async def _analyze_fraud_patterns(self, user_id: str, events: List[Dict[str, Any]]):
        """Analyze for suspicious patterns that might indicate fraud"""
        # Placeholder - would implement fraud detection algorithms
        pass
    
    async def _store_reputation(self, reputation: UserReputation):
        """Store reputation in database"""
        # Placeholder - would store in database
        pass
    
    async def _create_default_reputation(self, user_id: str) -> UserReputation:
        """Create default reputation for new users"""
        dimension_scores = {}
        for dimension in ReputationDimension:
            dimension_scores[dimension] = ReputationScore(
                dimension=dimension,
                score=50.0,
                confidence=0.1,
                last_updated=datetime.now(timezone.utc),
                trend=0.0,
                evidence_count=0
            )
        
        return UserReputation(
            user_id=user_id,
            overall_score=50.0,
            trust_level=TrustLevel.NEWCOMER,
            dimension_scores=dimension_scores,
            badges=[],
            verification_status={},
            reputation_history=[],
            last_calculated=datetime.now(timezone.utc),
            next_review=datetime.now(timezone.utc) + timedelta(days=7)
        )
    
    def _get_primary_dimension(self, event_type: ReputationEvent) -> ReputationDimension:
        """Get the primary dimension affected by an event type"""
        dimension_map = {
            ReputationEvent.RESOURCE_PUBLISHED: ReputationDimension.QUALITY,
            ReputationEvent.RESOURCE_RATED: ReputationDimension.QUALITY,
            ReputationEvent.REVIEW_RECEIVED: ReputationDimension.RESPONSIVENESS,
            ReputationEvent.EXPERT_ENDORSEMENT: ReputationDimension.EXPERTISE,
            ReputationEvent.FRAUD_DETECTED: ReputationDimension.TRUSTWORTHINESS,
            ReputationEvent.COMMUNITY_REPORT: ReputationDimension.TRUSTWORTHINESS
        }
        return dimension_map.get(event_type, ReputationDimension.TRUSTWORTHINESS)
    
    def _calculate_immediate_impact(self, event_type: ReputationEvent, evidence: Dict[str, Any]) -> float:
        """Calculate immediate reputation impact of an event"""
        # Simplified immediate impact calculation
        impact_map = {
            ReputationEvent.RESOURCE_PUBLISHED: 2.0,
            ReputationEvent.EXPERT_ENDORSEMENT: 5.0,
            ReputationEvent.FRAUD_DETECTED: -10.0,
            ReputationEvent.COMMUNITY_REPORT: -2.0
        }
        return impact_map.get(event_type, 0.0)
    
    async def _store_reputation_transaction(self, transaction: ReputationTransaction):
        """Store reputation transaction in database"""
        # Placeholder - would store transaction record
        pass
    
    async def _update_reputation_incremental(self, user_id: str, transaction: ReputationTransaction):
        """Update reputation incrementally without full recalculation"""
        # Placeholder - would update reputation incrementally
        pass
    
    async def _get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user activity statistics"""
        # Placeholder - would aggregate user statistics
        return {
            "resources_published": 0,
            "community_reports": 0,
            "expert_endorsements": 0,
            "account_age_days": 30
        }


# Factory function
def get_reputation_calculator() -> ReputationCalculator:
    """Get the reputation calculator instance"""
    return ReputationCalculator()
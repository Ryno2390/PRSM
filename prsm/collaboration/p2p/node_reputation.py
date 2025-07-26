"""
Node Reputation System for PRSM P2P Collaboration

This module implements a comprehensive reputation system for P2P nodes,
ensuring reliable and trustworthy shard distribution. It tracks node
behavior, performance, and reliability to make informed decisions about
peer selection and trust levels.

Key Features:
- Multi-factor reputation scoring
- Behavior tracking and analysis
- Performance-based reputation adjustments
- Misbehavior detection and penalties
- Reputation-based peer selection
- Historical reputation analysis
"""

import asyncio
import json
import logging
import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
import math
import hashlib

logger = logging.getLogger(__name__)


class ReputationFactor(Enum):
    """Factors that contribute to reputation"""
    AVAILABILITY = "availability"           # Node uptime and responsiveness
    RELIABILITY = "reliability"             # Successful data transfers
    PERFORMANCE = "performance"             # Speed and efficiency
    HONESTY = "honesty"                    # Data integrity and truthfulness
    CONTRIBUTION = "contribution"           # Network participation
    SECURITY = "security"                  # Security compliance


class BehaviorType(Enum):
    """Types of node behaviors"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MALICIOUS = "malicious"


@dataclass
class ReputationEvent:
    """Single reputation event"""
    event_id: str
    node_id: str
    factor: ReputationFactor
    behavior_type: BehaviorType
    impact_score: float  # -1.0 to 1.0
    timestamp: float
    context: Dict[str, Any]
    verified: bool = False
    
    def __post_init__(self):
        # Ensure impact score is within bounds
        self.impact_score = max(-1.0, min(1.0, self.impact_score))


@dataclass
class ReputationScore:
    """Comprehensive reputation score for a node"""
    node_id: str
    overall_score: float  # 0.0 to 1.0
    factor_scores: Dict[ReputationFactor, float]
    confidence_level: float  # 0.0 to 1.0
    sample_size: int
    last_updated: float
    
    def is_trustworthy(self, threshold: float = 0.7) -> bool:
        """Check if node meets trustworthiness threshold"""
        return self.overall_score >= threshold and self.confidence_level >= 0.5
    
    def get_weighted_score(self, weights: Dict[ReputationFactor, float]) -> float:
        """Calculate weighted reputation score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            if factor in self.factor_scores:
                weighted_sum += self.factor_scores[factor] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


@dataclass
class NodeMetrics:
    """Performance and behavior metrics for a node"""
    node_id: str
    
    # Availability metrics
    uptime_percentage: float = 0.0
    response_time_avg: float = 0.0
    last_seen: float = 0.0
    
    # Reliability metrics
    successful_transfers: int = 0
    failed_transfers: int = 0
    data_integrity_violations: int = 0
    
    # Performance metrics
    avg_transfer_speed: float = 0.0
    avg_latency: float = 0.0
    bandwidth_contribution: float = 0.0
    
    # Security metrics
    security_violations: int = 0
    authentication_failures: int = 0
    
    # Contribution metrics
    data_shared: int = 0  # bytes
    storage_provided: int = 0  # bytes
    network_participation: float = 0.0
    
    @property
    def total_transfers(self) -> int:
        return self.successful_transfers + self.failed_transfers
    
    @property
    def success_rate(self) -> float:
        if self.total_transfers == 0:
            return 1.0  # No data yet, assume good
        return self.successful_transfers / self.total_transfers
    
    @property
    def reliability_score(self) -> float:
        """Calculate reliability score from metrics"""
        if self.total_transfers < 5:
            return 0.5  # Neutral score for new nodes
        
        # Penalize integrity violations heavily
        integrity_penalty = min(0.5, self.data_integrity_violations * 0.1)
        
        return max(0.0, self.success_rate - integrity_penalty)


class ReputationCalculator:
    """Calculates reputation scores based on various factors"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Factor weights (should sum to 1.0)
        self.factor_weights = {
            ReputationFactor.AVAILABILITY: 0.25,
            ReputationFactor.RELIABILITY: 0.25,
            ReputationFactor.PERFORMANCE: 0.15,
            ReputationFactor.HONESTY: 0.20,
            ReputationFactor.CONTRIBUTION: 0.10,
            ReputationFactor.SECURITY: 0.05
        }
        
        # Update weights from config
        if 'factor_weights' in self.config:
            self.factor_weights.update(self.config['factor_weights'])
        
        # Reputation parameters
        self.min_sample_size = self.config.get('min_sample_size', 10)
        self.decay_factor = self.config.get('decay_factor', 0.95)  # Weekly decay
        self.new_node_score = self.config.get('new_node_score', 0.5)
    
    def calculate_availability_score(self, metrics: NodeMetrics, 
                                   events: List[ReputationEvent]) -> float:
        """Calculate availability reputation score"""
        if not events and metrics.uptime_percentage == 0:
            return self.new_node_score
        
        # Base score from uptime percentage
        uptime_score = metrics.uptime_percentage
        
        # Adjust based on response time
        if metrics.response_time_avg > 0:
            # Good response time: < 1s, Poor: > 10s
            response_score = max(0, 1.0 - (metrics.response_time_avg - 1.0) / 9.0)
            uptime_score = (uptime_score + response_score) / 2
        
        # Apply recent availability events
        recent_events = [
            e for e in events 
            if e.factor == ReputationFactor.AVAILABILITY 
            and time.time() - e.timestamp < 86400 * 7  # Last 7 days
        ]
        
        for event in recent_events:
            if event.behavior_type == BehaviorType.NEGATIVE:
                uptime_score *= 0.9  # 10% penalty for negative events
            elif event.behavior_type == BehaviorType.POSITIVE:
                uptime_score = min(1.0, uptime_score * 1.05)  # 5% bonus
        
        return max(0.0, min(1.0, uptime_score))
    
    def calculate_reliability_score(self, metrics: NodeMetrics,
                                  events: List[ReputationEvent]) -> float:
        """Calculate reliability reputation score"""
        base_score = metrics.reliability_score
        
        # Apply reliability events
        reliability_events = [
            e for e in events
            if e.factor == ReputationFactor.RELIABILITY
        ]
        
        # Weight recent events more heavily
        current_time = time.time()
        for event in reliability_events:
            age_weight = math.exp(-(current_time - event.timestamp) / (86400 * 30))  # 30-day decay
            
            if event.behavior_type == BehaviorType.MALICIOUS:
                base_score *= 0.5  # Severe penalty
            elif event.behavior_type == BehaviorType.NEGATIVE:
                base_score *= (1.0 - 0.1 * age_weight)
            elif event.behavior_type == BehaviorType.POSITIVE:
                base_score = min(1.0, base_score * (1.0 + 0.05 * age_weight))
        
        return max(0.0, min(1.0, base_score))
    
    def calculate_performance_score(self, metrics: NodeMetrics) -> float:
        """Calculate performance reputation score"""
        if metrics.avg_transfer_speed == 0:
            return self.new_node_score
        
        # Normalize transfer speed (assume 10 MB/s is excellent)
        speed_score = min(1.0, metrics.avg_transfer_speed / (10 * 1024 * 1024))
        
        # Normalize latency (assume 100ms is good, 1000ms is poor)
        if metrics.avg_latency > 0:
            latency_score = max(0, 1.0 - (metrics.avg_latency - 0.1) / 0.9)
        else:
            latency_score = 0.5
        
        # Combine speed and latency
        performance_score = (speed_score * 0.7 + latency_score * 0.3)
        
        return max(0.0, min(1.0, performance_score))
    
    def calculate_honesty_score(self, metrics: NodeMetrics,
                               events: List[ReputationEvent]) -> float:
        """Calculate honesty reputation score"""
        base_score = 1.0
        
        # Penalize integrity violations heavily
        if metrics.data_integrity_violations > 0:
            base_score *= (0.8 ** metrics.data_integrity_violations)
        
        # Apply honesty events
        honesty_events = [
            e for e in events
            if e.factor == ReputationFactor.HONESTY
        ]
        
        for event in honesty_events:
            if event.behavior_type == BehaviorType.MALICIOUS:
                base_score *= 0.3  # Severe penalty for dishonesty
            elif event.behavior_type == BehaviorType.NEGATIVE:
                base_score *= 0.8
            elif event.behavior_type == BehaviorType.POSITIVE:
                base_score = min(1.0, base_score * 1.1)
        
        return max(0.0, min(1.0, base_score))
    
    def calculate_contribution_score(self, metrics: NodeMetrics) -> float:
        """Calculate contribution reputation score"""
        if metrics.data_shared == 0 and metrics.storage_provided == 0:
            return self.new_node_score
        
        # Score based on data contribution (normalize to 1GB = full score)
        data_score = min(1.0, metrics.data_shared / (1024 * 1024 * 1024))
        
        # Score based on storage contribution (normalize to 10GB = full score)
        storage_score = min(1.0, metrics.storage_provided / (10 * 1024 * 1024 * 1024))
        
        # Network participation score
        participation_score = min(1.0, metrics.network_participation)
        
        # Weighted combination
        contribution_score = (
            data_score * 0.4 +
            storage_score * 0.4 +
            participation_score * 0.2
        )
        
        return max(0.0, min(1.0, contribution_score))
    
    def calculate_security_score(self, metrics: NodeMetrics,
                                events: List[ReputationEvent]) -> float:
        """Calculate security reputation score"""
        base_score = 1.0
        
        # Penalize security violations
        if metrics.security_violations > 0:
            base_score *= (0.7 ** metrics.security_violations)
        
        # Penalize authentication failures
        if metrics.authentication_failures > 0:
            base_score *= (0.9 ** metrics.authentication_failures)
        
        # Apply security events
        security_events = [
            e for e in events
            if e.factor == ReputationFactor.SECURITY
        ]
        
        for event in security_events:
            if event.behavior_type == BehaviorType.MALICIOUS:
                base_score *= 0.2  # Severe penalty for security issues
            elif event.behavior_type == BehaviorType.NEGATIVE:
                base_score *= 0.8
        
        return max(0.0, min(1.0, base_score))
    
    def calculate_overall_score(self, factor_scores: Dict[ReputationFactor, float],
                               sample_size: int) -> Tuple[float, float]:
        """Calculate overall reputation score and confidence"""
        # Weighted average of factor scores
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, score in factor_scores.items():
            weight = self.factor_weights.get(factor, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else self.new_node_score
        
        # Calculate confidence based on sample size
        confidence = min(1.0, sample_size / self.min_sample_size) if sample_size > 0 else 0.0
        
        return overall_score, confidence
    
    def calculate_reputation(self, metrics: NodeMetrics,
                           events: List[ReputationEvent]) -> ReputationScore:
        """Calculate comprehensive reputation score"""
        factor_scores = {
            ReputationFactor.AVAILABILITY: self.calculate_availability_score(metrics, events),
            ReputationFactor.RELIABILITY: self.calculate_reliability_score(metrics, events),
            ReputationFactor.PERFORMANCE: self.calculate_performance_score(metrics),
            ReputationFactor.HONESTY: self.calculate_honesty_score(metrics, events),
            ReputationFactor.CONTRIBUTION: self.calculate_contribution_score(metrics),
            ReputationFactor.SECURITY: self.calculate_security_score(metrics, events)
        }
        
        overall_score, confidence = self.calculate_overall_score(
            factor_scores, len(events)
        )
        
        return ReputationScore(
            node_id=metrics.node_id,
            overall_score=overall_score,
            factor_scores=factor_scores,
            confidence_level=confidence,
            sample_size=len(events),
            last_updated=time.time()
        )


class ReputationTracker:
    """Tracks and manages reputation events and metrics"""
    
    def __init__(self):
        self.events: Dict[str, List[ReputationEvent]] = {}  # node_id -> events
        self.metrics: Dict[str, NodeMetrics] = {}  # node_id -> metrics
        self.max_events_per_node = 1000
        self.event_retention_days = 365
    
    def record_event(self, event: ReputationEvent):
        """Record a reputation event"""
        if event.node_id not in self.events:
            self.events[event.node_id] = []
        
        self.events[event.node_id].append(event)
        
        # Maintain event history limits
        events_list = self.events[event.node_id]
        if len(events_list) > self.max_events_per_node:
            # Remove oldest events
            events_list.sort(key=lambda e: e.timestamp)
            self.events[event.node_id] = events_list[-self.max_events_per_node:]
        
        # Remove expired events
        cutoff_time = time.time() - (self.event_retention_days * 86400)
        self.events[event.node_id] = [
            e for e in self.events[event.node_id]
            if e.timestamp > cutoff_time
        ]
        
        logger.debug(f"Recorded {event.behavior_type.value} event for {event.node_id}: "
                    f"{event.factor.value} (impact: {event.impact_score})")
    
    def update_metrics(self, node_id: str, **metric_updates):
        """Update node metrics"""
        if node_id not in self.metrics:
            self.metrics[node_id] = NodeMetrics(node_id=node_id)
        
        metrics = self.metrics[node_id]
        
        for key, value in metric_updates.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        
        logger.debug(f"Updated metrics for {node_id}: {metric_updates}")
    
    def get_events(self, node_id: str) -> List[ReputationEvent]:
        """Get all events for a node"""
        return self.events.get(node_id, [])
    
    def get_metrics(self, node_id: str) -> NodeMetrics:
        """Get metrics for a node"""
        if node_id not in self.metrics:
            self.metrics[node_id] = NodeMetrics(node_id=node_id)
        return self.metrics[node_id]
    
    def get_recent_events(self, node_id: str, days: int = 7) -> List[ReputationEvent]:
        """Get recent events for a node"""
        cutoff_time = time.time() - (days * 86400)
        return [
            event for event in self.get_events(node_id)
            if event.timestamp > cutoff_time
        ]


class ReputationSystem:
    """
    Main reputation system for PRSM P2P network
    
    Manages node reputation tracking, calculation, and provides
    reputation-based decision making for peer selection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.tracker = ReputationTracker()
        self.calculator = ReputationCalculator(config)
        
        # Cached reputation scores
        self.reputation_cache: Dict[str, ReputationScore] = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        # System parameters
        self.min_reputation_threshold = self.config.get('min_reputation', 0.3)
        self.preferred_reputation_threshold = self.config.get('preferred_reputation', 0.7)
        
        logger.info("Reputation system initialized")
    
    def record_positive_behavior(self, node_id: str, factor: ReputationFactor,
                                impact: float = 0.1, context: Optional[Dict] = None):
        """Record positive behavior for a node"""
        event = ReputationEvent(
            event_id=self._generate_event_id(),
            node_id=node_id,
            factor=factor,
            behavior_type=BehaviorType.POSITIVE,
            impact_score=abs(impact),
            timestamp=time.time(),
            context=context or {}
        )
        
        self.tracker.record_event(event)
        self._invalidate_cache(node_id)
    
    def record_negative_behavior(self, node_id: str, factor: ReputationFactor,
                                impact: float = -0.1, context: Optional[Dict] = None):
        """Record negative behavior for a node"""
        event = ReputationEvent(
            event_id=self._generate_event_id(),
            node_id=node_id,
            factor=factor,
            behavior_type=BehaviorType.NEGATIVE,
            impact_score=-abs(impact),
            timestamp=time.time(),
            context=context or {}
        )
        
        self.tracker.record_event(event)
        self._invalidate_cache(node_id)
    
    def record_malicious_behavior(self, node_id: str, factor: ReputationFactor,
                                 context: Optional[Dict] = None):
        """Record malicious behavior for a node"""
        event = ReputationEvent(
            event_id=self._generate_event_id(),
            node_id=node_id,
            factor=factor,
            behavior_type=BehaviorType.MALICIOUS,
            impact_score=-1.0,  # Maximum negative impact
            timestamp=time.time(),
            context=context or {}
        )
        
        self.tracker.record_event(event)
        self._invalidate_cache(node_id)
        
        logger.warning(f"Recorded malicious behavior for {node_id}: {factor.value}")
    
    def update_node_performance(self, node_id: str, **performance_data):
        """Update node performance metrics"""
        self.tracker.update_metrics(node_id, **performance_data)
        self._invalidate_cache(node_id)
    
    def get_reputation(self, node_id: str, use_cache: bool = True) -> ReputationScore:
        """Get reputation score for a node"""
        # Check cache first
        if use_cache and node_id in self.reputation_cache:
            cached_score = self.reputation_cache[node_id]
            if time.time() - cached_score.last_updated < self.cache_ttl:
                return cached_score
        
        # Calculate fresh reputation score
        metrics = self.tracker.get_metrics(node_id)
        events = self.tracker.get_events(node_id)
        
        reputation = self.calculator.calculate_reputation(metrics, events)
        
        # Cache the result
        self.reputation_cache[node_id] = reputation
        
        return reputation
    
    def is_node_trustworthy(self, node_id: str, 
                           custom_threshold: Optional[float] = None) -> bool:
        """Check if a node is trustworthy"""
        threshold = custom_threshold or self.preferred_reputation_threshold
        reputation = self.get_reputation(node_id)
        
        return reputation.is_trustworthy(threshold)
    
    def select_trustworthy_nodes(self, candidate_nodes: List[str],
                                count: int, min_reputation: Optional[float] = None,
                                preferred_factors: Optional[Dict[ReputationFactor, float]] = None) -> List[str]:
        """Select most trustworthy nodes from candidates"""
        min_rep = min_reputation or self.min_reputation_threshold
        
        # Get reputation scores for all candidates
        node_scores = []
        for node_id in candidate_nodes:
            reputation = self.get_reputation(node_id)
            
            if reputation.overall_score >= min_rep:
                if preferred_factors:
                    score = reputation.get_weighted_score(preferred_factors)
                else:
                    score = reputation.overall_score
                
                node_scores.append((score, node_id, reputation))
        
        # Sort by score and select top nodes
        node_scores.sort(key=lambda x: x[0], reverse=True)
        
        selected_nodes = []
        for score, node_id, reputation in node_scores[:count]:
            selected_nodes.append(node_id)
            logger.debug(f"Selected node {node_id} with reputation {score:.3f}")
        
        return selected_nodes
    
    def get_network_reputation_stats(self) -> Dict[str, Any]:
        """Get network-wide reputation statistics"""
        all_nodes = set(self.tracker.events.keys()) | set(self.tracker.metrics.keys())
        
        if not all_nodes:
            return {'total_nodes': 0}
        
        reputations = [self.get_reputation(node_id) for node_id in all_nodes]
        
        scores = [rep.overall_score for rep in reputations]
        confidences = [rep.confidence_level for rep in reputations]
        
        trustworthy_count = sum(1 for rep in reputations if rep.is_trustworthy())
        
        # Calculate factor averages
        factor_averages = {}
        for factor in ReputationFactor:
            factor_scores = [
                rep.factor_scores.get(factor, 0.0) for rep in reputations
            ]
            factor_averages[factor.value] = statistics.mean(factor_scores) if factor_scores else 0.0
        
        return {
            'total_nodes': len(all_nodes),
            'trustworthy_nodes': trustworthy_count,
            'trustworthy_percentage': (trustworthy_count / len(all_nodes)) * 100,
            'average_reputation': statistics.mean(scores) if scores else 0.0,
            'median_reputation': statistics.median(scores) if scores else 0.0,
            'average_confidence': statistics.mean(confidences) if confidences else 0.0,
            'factor_averages': factor_averages,
            'reputation_distribution': {
                'excellent': sum(1 for s in scores if s >= 0.9),
                'good': sum(1 for s in scores if 0.7 <= s < 0.9),
                'fair': sum(1 for s in scores if 0.5 <= s < 0.7),
                'poor': sum(1 for s in scores if 0.3 <= s < 0.5),
                'very_poor': sum(1 for s in scores if s < 0.3)
            }
        }
    
    def detect_reputation_attacks(self) -> List[Dict[str, Any]]:
        """Detect potential reputation manipulation attacks"""
        attacks = []
        
        # Look for nodes with suspicious reputation patterns
        for node_id in self.tracker.events.keys():
            events = self.tracker.get_events(node_id)
            recent_events = self.tracker.get_recent_events(node_id, 1)  # Last day
            
            if len(recent_events) > 50:  # Unusually high event count
                attacks.append({
                    'type': 'event_flooding',
                    'node_id': node_id,
                    'event_count': len(recent_events),
                    'severity': 'medium'
                })
            
            # Check for rapid reputation changes
            if len(events) >= 10:
                recent_scores = []
                for i in range(min(10, len(events))):
                    event = events[-(i+1)]
                    temp_reputation = self.calculator.calculate_reputation(
                        self.tracker.get_metrics(node_id),
                        events[:-(i)]
                    )
                    recent_scores.append(temp_reputation.overall_score)
                
                if len(recent_scores) >= 3:
                    score_variance = statistics.variance(recent_scores)
                    if score_variance > 0.1:  # High variance suggests manipulation
                        attacks.append({
                            'type': 'reputation_manipulation',
                            'node_id': node_id,
                            'score_variance': score_variance,
                            'severity': 'high' if score_variance > 0.2 else 'medium'
                        })
        
        return attacks
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        return hashlib.sha256(f"{time.time()}:{id(self)}".encode()).hexdigest()[:16]
    
    def _invalidate_cache(self, node_id: str):
        """Invalidate cached reputation for a node"""
        if node_id in self.reputation_cache:
            del self.reputation_cache[node_id]
    
    def export_reputation_data(self, node_id: str) -> Dict[str, Any]:
        """Export all reputation data for a node"""
        reputation = self.get_reputation(node_id)
        events = self.tracker.get_events(node_id)
        metrics = self.tracker.get_metrics(node_id)
        
        return {
            'node_id': node_id,
            'reputation': asdict(reputation),
            'metrics': asdict(metrics),
            'events': [asdict(event) for event in events],
            'exported_at': time.time()
        }
    
    def import_reputation_data(self, data: Dict[str, Any]):
        """Import reputation data for a node"""
        node_id = data['node_id']
        
        # Import metrics
        if 'metrics' in data:
            metrics_data = data['metrics']
            self.tracker.metrics[node_id] = NodeMetrics(**metrics_data)
        
        # Import events
        if 'events' in data:
            events_data = data['events']
            self.tracker.events[node_id] = [
                ReputationEvent(**event_data) for event_data in events_data
            ]
        
        # Invalidate cache to force recalculation
        self._invalidate_cache(node_id)
        
        logger.info(f"Imported reputation data for {node_id}")


# Example usage and testing
async def example_reputation_system():
    """Example of reputation system usage"""
    
    # Initialize reputation system
    reputation_system = ReputationSystem({
        'min_reputation': 0.3,
        'preferred_reputation': 0.7
    })
    
    # Simulate some node behaviors
    nodes = ['node1', 'node2', 'node3', 'node4', 'node5']
    
    # Node1: Excellent behavior
    for _ in range(20):
        reputation_system.record_positive_behavior(
            'node1', ReputationFactor.RELIABILITY, 0.05
        )
        reputation_system.record_positive_behavior(
            'node1', ReputationFactor.AVAILABILITY, 0.03
        )
    
    reputation_system.update_node_performance(
        'node1',
        uptime_percentage=0.98,
        successful_transfers=100,
        failed_transfers=2,
        avg_transfer_speed=50 * 1024 * 1024,
        avg_latency=0.05
    )
    
    # Node2: Good behavior
    for _ in range(15):
        reputation_system.record_positive_behavior(
            'node2', ReputationFactor.RELIABILITY, 0.03
        )
    
    reputation_system.update_node_performance(
        'node2',
        uptime_percentage=0.95,
        successful_transfers=80,
        failed_transfers=5,
        avg_transfer_speed=30 * 1024 * 1024,
        avg_latency=0.08
    )
    
    # Node3: Mixed behavior
    for _ in range(10):
        reputation_system.record_positive_behavior(
            'node3', ReputationFactor.RELIABILITY, 0.02
        )
    
    for _ in range(5):
        reputation_system.record_negative_behavior(
            'node3', ReputationFactor.AVAILABILITY, -0.05
        )
    
    reputation_system.update_node_performance(
        'node3',
        uptime_percentage=0.85,
        successful_transfers=60,
        failed_transfers=15,
        avg_transfer_speed=20 * 1024 * 1024,
        avg_latency=0.15
    )
    
    # Node4: Poor behavior
    for _ in range(8):
        reputation_system.record_negative_behavior(
            'node4', ReputationFactor.RELIABILITY, -0.1
        )
    
    reputation_system.update_node_performance(
        'node4',
        uptime_percentage=0.70,
        successful_transfers=30,
        failed_transfers=20,
        data_integrity_violations=3,
        avg_transfer_speed=10 * 1024 * 1024,
        avg_latency=0.3
    )
    
    # Node5: Malicious behavior
    reputation_system.record_malicious_behavior(
        'node5', ReputationFactor.SECURITY, {'violation_type': 'data_tampering'}
    )
    reputation_system.record_malicious_behavior(
        'node5', ReputationFactor.HONESTY, {'violation_type': 'false_data'}
    )
    
    reputation_system.update_node_performance(
        'node5',
        uptime_percentage=0.60,
        successful_transfers=10,
        failed_transfers=40,
        data_integrity_violations=10,
        security_violations=5,
        avg_transfer_speed=5 * 1024 * 1024,
        avg_latency=0.5
    )
    
    # Display reputation scores
    print("Node Reputation Scores:")
    print("=" * 50)
    
    for node_id in nodes:
        reputation = reputation_system.get_reputation(node_id)
        print(f"\n{node_id}:")
        print(f"  Overall Score: {reputation.overall_score:.3f}")
        print(f"  Confidence: {reputation.confidence_level:.3f}")
        print(f"  Trustworthy: {reputation.is_trustworthy()}")
        print(f"  Factor Scores:")
        for factor, score in reputation.factor_scores.items():
            print(f"    {factor.value}: {score:.3f}")
    
    # Test trustworthy node selection
    print(f"\nTrustworthy Node Selection:")
    print("=" * 50)
    
    trustworthy_nodes = reputation_system.select_trustworthy_nodes(
        nodes, 3, min_reputation=0.5
    )
    print(f"Top 3 trustworthy nodes: {trustworthy_nodes}")
    
    # Network statistics
    print(f"\nNetwork Reputation Statistics:")
    print("=" * 50)
    stats = reputation_system.get_network_reputation_stats()
    print(json.dumps(stats, indent=2))
    
    # Attack detection
    print(f"\nReputation Attack Detection:")
    print("=" * 50)
    attacks = reputation_system.detect_reputation_attacks()
    if attacks:
        for attack in attacks:
            print(f"Detected {attack['type']} from {attack['node_id']} "
                  f"(severity: {attack['severity']})")
    else:
        print("No reputation attacks detected")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_reputation_system())
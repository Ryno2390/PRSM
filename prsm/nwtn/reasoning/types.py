#!/usr/bin/env python3
"""
NWTN Meta-Reasoning Engine Types and Enums
==========================================

Core types, enums, and data structures used across the NWTN meta-reasoning system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID
from datetime import datetime, timezone


class ThinkingMode(Enum):
    """Different thinking modes representing computational budget levels"""
    QUICK = "quick"        # 5-10 engines, minimal permutations, low FTNS cost
    INTERMEDIATE = "intermediate"  # 50-100 engines, partial permutations, medium FTNS cost  
    DEEP = "deep"         # 5040 engines, full permutations, high FTNS cost


class ReasoningEngine(Enum):
    """Available reasoning engines in the system"""
    CLAUDE = "claude"
    GPT4 = "gpt4"
    GEMINI_PRO = "gemini_pro"
    COHERE = "cohere"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


class ReasoningMode(Enum):
    """System 1 vs System 2 reasoning mode differentiation"""
    SYSTEM1_CREATIVE = "system1_creative"      # Divergent, high-risk exploration for candidate generation
    SYSTEM2_VALIDATION = "system2_validation"  # Convergent, rigorous testing for evaluation


class EngineHealthStatus(Enum):
    """Health status of reasoning engines"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"


@dataclass
class EnginePerformanceMetrics:
    """Performance metrics for a reasoning engine"""
    engine_id: str
    engine_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Response metrics
    response_time_ms: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Quality metrics
    reasoning_quality_score: float = 0.0
    coherence_score: float = 0.0
    accuracy_score: float = 0.0
    creativity_score: float = 0.0
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    token_usage_count: int = 0
    cost_per_request: float = 0.0
    
    # Load metrics
    current_requests: int = 0
    queue_depth: int = 0
    throughput_rps: float = 0.0
    
    # Health indicators
    health_status: EngineHealthStatus = EngineHealthStatus.HEALTHY
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    
    def calculate_composite_score(self) -> float:
        """Calculate a composite performance score (0-100)"""
        try:
            # Weight different metrics for overall performance
            weights = {
                'response_time': 0.25,  # Lower is better
                'success_rate': 0.25,   # Higher is better
                'quality': 0.20,        # Higher is better
                'resource_efficiency': 0.15,  # Lower resource usage is better
                'reliability': 0.15     # Higher uptime is better
            }
            
            # Normalize response time (assume 1000ms is baseline)
            response_score = max(0, 100 - (self.response_time_ms / 10))
            
            # Success rate is already 0-100
            success_score = self.success_rate
            
            # Average quality scores
            quality_score = (self.reasoning_quality_score + self.coherence_score + 
                           self.accuracy_score + self.creativity_score) / 4
            
            # Resource efficiency (lower usage = higher score)
            resource_score = max(0, 100 - self.cpu_usage_percent)
            
            # Reliability based on error rate
            reliability_score = max(0, 100 - self.error_rate)
            
            composite = (
                weights['response_time'] * response_score +
                weights['success_rate'] * success_score +
                weights['quality'] * quality_score +
                weights['resource_efficiency'] * resource_score +
                weights['reliability'] * reliability_score
            )
            
            return min(100.0, max(0.0, composite))
            
        except Exception:
            return 50.0  # Default middle score if calculation fails


class PerformanceMetric(Enum):
    """Types of performance metrics tracked"""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    QUALITY_SCORE = "quality_score"
    RESOURCE_USAGE = "resource_usage"
    THROUGHPUT = "throughput"
    COST_EFFICIENCY = "cost_efficiency"


class PerformanceCategory(Enum):
    """Categories for performance analysis"""
    SPEED = "speed"
    QUALITY = "quality"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_engines: int = 0
    active_engines: int = 0
    healthy_engines: int = 0
    avg_response_time: float = 0.0
    total_requests: int = 0
    success_rate: float = 0.0
    overall_quality_score: float = 0.0
    system_load: float = 0.0
    memory_usage_mb: float = 0.0


@dataclass
class PerformanceProfile:
    """Performance profile for an engine or system component"""
    profile_id: str
    name: str
    description: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Historical metrics
    snapshots: List[PerformanceSnapshot] = field(default_factory=list)
    
    # Aggregated metrics
    avg_response_time: float = 0.0
    avg_success_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_resource_usage: float = 0.0
    
    # Performance characteristics
    peak_throughput_rps: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    reliability_score: float = 0.0
    
    # Capacity metrics
    max_concurrent_requests: int = 0
    optimal_load_threshold: float = 0.8
    
    def add_snapshot(self, snapshot: PerformanceSnapshot):
        """Add a performance snapshot"""
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots (last 1000)
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-1000:]
        
        # Update aggregated metrics
        self._update_aggregated_metrics()
    
    def _update_aggregated_metrics(self):
        """Update aggregated performance metrics"""
        if not self.snapshots:
            return
        
        recent_snapshots = self.snapshots[-100:]  # Last 100 snapshots
        
        self.avg_response_time = sum(s.avg_response_time for s in recent_snapshots) / len(recent_snapshots)
        self.avg_success_rate = sum(s.success_rate for s in recent_snapshots) / len(recent_snapshots)
        self.avg_quality_score = sum(s.overall_quality_score for s in recent_snapshots) / len(recent_snapshots)
        self.avg_resource_usage = sum(s.system_load for s in recent_snapshots) / len(recent_snapshots)
        
        # Update min/max response times
        for snapshot in recent_snapshots:
            self.min_response_time = min(self.min_response_time, snapshot.avg_response_time)
            self.max_response_time = max(self.max_response_time, snapshot.avg_response_time)
        
        # Calculate reliability score based on consistency
        response_times = [s.avg_response_time for s in recent_snapshots]
        if len(response_times) > 1:
            import statistics
            std_dev = statistics.stdev(response_times)
            mean_time = statistics.mean(response_times)
            # Lower coefficient of variation = higher reliability
            cv = std_dev / mean_time if mean_time > 0 else 1.0
            self.reliability_score = max(0, 100 - (cv * 100))
        else:
            self.reliability_score = 100.0


class FailureType(Enum):
    """Types of failures that can occur in the reasoning system"""
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    PARSING = "parsing"
    VALIDATION = "validation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    LOGIC_ERROR = "logic_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions that can be taken"""
    RETRY = "retry"
    FALLBACK_ENGINE = "fallback_engine"
    REDUCE_LOAD = "reduce_load"
    CIRCUIT_BREAKER = "circuit_breaker"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    RESTART = "restart"
    MAINTENANCE_MODE = "maintenance_mode"


class FailureDetectionMode(Enum):
    """Modes for failure detection"""
    REACTIVE = "reactive"      # Detect after failure occurs
    PROACTIVE = "proactive"    # Predict failures before they occur
    HYBRID = "hybrid"          # Combination of both


@dataclass
class FailureEvent:
    """Represents a failure event in the system"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    engine_id: str = ""
    failure_type: FailureType = FailureType.UNKNOWN
    severity: int = 1  # 1-10 scale
    
    error_message: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Context information
    request_id: Optional[str] = None
    thinking_mode: Optional[ThinkingMode] = None
    operation: str = ""
    
    # Recovery information
    recovery_attempted: bool = False
    recovery_action: Optional[RecoveryAction] = None
    recovery_successful: bool = False
    recovery_time_ms: float = 0.0
    
    # Impact assessment
    affected_requests: int = 0
    performance_impact: float = 0.0  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert failure event to dictionary"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'engine_id': self.engine_id,
            'failure_type': self.failure_type.value,
            'severity': self.severity,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'request_id': self.request_id,
            'thinking_mode': self.thinking_mode.value if self.thinking_mode else None,
            'operation': self.operation,
            'recovery_attempted': self.recovery_attempted,
            'recovery_action': self.recovery_action.value if self.recovery_action else None,
            'recovery_successful': self.recovery_successful,
            'recovery_time_ms': self.recovery_time_ms,
            'affected_requests': self.affected_requests,
            'performance_impact': self.performance_impact
        }


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from failures"""
    strategy_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    
    # Trigger conditions
    failure_types: List[FailureType] = field(default_factory=list)
    severity_threshold: int = 1
    frequency_threshold: int = 1  # Number of failures before triggering
    time_window_seconds: int = 60
    
    # Recovery actions
    primary_action: RecoveryAction = RecoveryAction.RETRY
    fallback_actions: List[RecoveryAction] = field(default_factory=list)
    
    # Parameters
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    
    # Conditions
    enabled: bool = True
    priority: int = 5  # 1-10, higher = more priority
    
    def should_trigger(self, failure_event: FailureEvent, recent_failures: List[FailureEvent]) -> bool:
        """Determine if this strategy should trigger for a failure"""
        if not self.enabled:
            return False
        
        # Check failure type
        if self.failure_types and failure_event.failure_type not in self.failure_types:
            return False
        
        # Check severity
        if failure_event.severity < self.severity_threshold:
            return False
        
        # Check frequency
        if len(recent_failures) < self.frequency_threshold:
            return False
        
        return True


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for distributing requests"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    PERFORMANCE_BASED = "performance_based"
    RANDOM = "random"
    HASH_BASED = "hash_based"
    ADAPTIVE = "adaptive"


class LoadBalancingMode(Enum):
    """Modes for load balancing operation"""
    ACTIVE_PASSIVE = "active_passive"
    ACTIVE_ACTIVE = "active_active"
    PRIORITY_BASED = "priority_based"


@dataclass
class EngineWorkload:
    """Represents the current workload of an engine"""
    engine_id: str
    current_requests: int = 0
    queue_depth: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 100.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Capacity metrics
    max_capacity: int = 100
    utilization_percent: float = 0.0
    
    # Performance indicators
    performance_score: float = 100.0
    health_status: EngineHealthStatus = EngineHealthStatus.HEALTHY
    
    def calculate_load_score(self) -> float:
        """Calculate a load score for load balancing decisions"""
        # Lower score = better choice for new requests
        try:
            # Base load from current utilization
            load_score = self.utilization_percent
            
            # Adjust for response time (higher response time = higher load score)
            response_time_penalty = min(50, self.avg_response_time / 10)
            load_score += response_time_penalty
            
            # Adjust for success rate (lower success rate = higher load score)
            success_rate_penalty = (100 - self.success_rate) / 2
            load_score += success_rate_penalty
            
            # Adjust for health status
            if self.health_status == EngineHealthStatus.DEGRADED:
                load_score += 25
            elif self.health_status == EngineHealthStatus.UNHEALTHY:
                load_score += 100  # Strongly discourage unhealthy engines
            elif self.health_status == EngineHealthStatus.OFFLINE:
                load_score = float('inf')  # Never route to offline engines
            
            return max(0.0, load_score)
            
        except Exception:
            return 100.0  # Default high load score if calculation fails
    
    def update_utilization(self):
        """Update utilization percentage based on current load"""
        if self.max_capacity > 0:
            self.utilization_percent = (self.current_requests / self.max_capacity) * 100
        else:
            self.utilization_percent = 0.0


# Additional type definitions for meta-reasoning system
@dataclass
class ReasoningRequest:
    """Request for reasoning operation"""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Request content
    query: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Processing parameters
    thinking_mode: ThinkingMode = ThinkingMode.QUICK
    max_engines: int = 10
    timeout_seconds: float = 30.0
    
    # Quality requirements
    min_quality_score: float = 0.7
    preferred_engines: List[ReasoningEngine] = field(default_factory=list)
    
    # FTNS budget
    max_ftns_cost: float = 100.0
    
    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 5  # 1-10 scale


@dataclass
class ReasoningResponse:
    """Response from reasoning operation"""
    request_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Response content
    result: str = ""
    confidence: float = 0.0
    quality_score: float = 0.0
    
    # Processing information
    engines_used: List[str] = field(default_factory=list)
    thinking_mode: ThinkingMode = ThinkingMode.QUICK
    processing_time_ms: float = 0.0
    
    # Cost information
    ftns_cost: float = 0.0
    token_usage: int = 0
    
    # Quality metrics
    coherence_score: float = 0.0
    accuracy_score: float = 0.0
    creativity_score: float = 0.0
    
    # Metadata
    error_message: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def is_successful(self) -> bool:
        """Check if the reasoning was successful"""
        return self.error_message is None and self.confidence > 0.5


# Export all types for use in other modules
__all__ = [
    'ThinkingMode',
    'ReasoningEngine', 
    'EngineHealthStatus',
    'EnginePerformanceMetrics',
    'PerformanceMetric',
    'PerformanceCategory',
    'PerformanceSnapshot',
    'PerformanceProfile',
    'FailureType',
    'RecoveryAction',
    'FailureDetectionMode',
    'FailureEvent',
    'RecoveryStrategy',
    'LoadBalancingStrategy',
    'LoadBalancingMode',
    'EngineWorkload',
    'ReasoningRequest',
    'ReasoningResponse'
]
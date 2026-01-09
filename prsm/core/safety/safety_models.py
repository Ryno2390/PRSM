"""
Safety Models for DGM-Enhanced Self-Modification

Comprehensive data models for safety validation, risk assessment,
and constraint checking in the DGM evolution system.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid

from ..evolution.models import RiskLevel, ComponentType


class SafetyCheckType(str, Enum):
    """Types of safety checks performed."""
    CAPABILITY_BOUNDS = "CAPABILITY_BOUNDS"
    RESOURCE_LIMITS = "RESOURCE_LIMITS"
    BEHAVIORAL_CONSTRAINTS = "BEHAVIORAL_CONSTRAINTS"
    IMPACT_ASSESSMENT = "IMPACT_ASSESSMENT"
    SECURITY_VALIDATION = "SECURITY_VALIDATION"
    GOVERNANCE_APPROVAL = "GOVERNANCE_APPROVAL"


class SafetyStatus(str, Enum):
    """Status of safety validation."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MONITORING = "MONITORING"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class ConstraintViolationType(str, Enum):
    """Types of constraint violations."""
    CAPABILITY_EXCEEDED = "CAPABILITY_EXCEEDED"
    RESOURCE_EXCEEDED = "RESOURCE_EXCEEDED"
    BEHAVIORAL_VIOLATION = "BEHAVIORAL_VIOLATION"
    SECURITY_BREACH = "SECURITY_BREACH"
    GOVERNANCE_VIOLATION = "GOVERNANCE_VIOLATION"


@dataclass
class CapabilityBounds:
    """Defines capability bounds for system components."""
    
    # Processing capabilities
    max_context_length: int = 8192
    max_concurrent_sessions: int = 100
    max_memory_usage_mb: int = 4096
    max_cpu_utilization: float = 0.8
    
    # Modification capabilities
    max_modification_frequency: int = 10  # per hour
    max_modification_scope: str = "component"  # component, system, network
    max_performance_delta: float = 0.5  # maximum improvement per modification
    
    # Network capabilities
    max_network_connections: int = 1000
    max_bandwidth_mbps: float = 100.0
    max_storage_gb: float = 10.0
    
    # Safety constraints
    requires_approval_threshold: float = 0.1  # modifications above this require approval
    emergency_stop_threshold: float = 0.3  # immediate stop threshold
    rollback_timeout_minutes: int = 30


@dataclass
class ResourceLimits:
    """Resource consumption limits for safe operation."""
    
    # Compute resources
    cpu_cores_limit: int = 8
    memory_limit_gb: float = 16.0
    disk_space_limit_gb: float = 100.0
    
    # Network resources
    bandwidth_limit_mbps: float = 100.0
    connections_limit: int = 1000
    request_rate_limit: int = 1000  # per minute
    
    # Time resources
    execution_timeout_seconds: int = 300
    modification_timeout_seconds: int = 600
    evaluation_timeout_seconds: int = 1800
    
    # Economic resources
    ftns_budget_limit: Decimal = Decimal('1000.0')
    modification_cost_limit: Decimal = Decimal('100.0')


@dataclass
class BehavioralConstraints:
    """Behavioral constraints for system components."""
    
    # Prohibited behaviors
    prohibited_actions: List[str] = field(default_factory=lambda: [
        "modify_safety_systems",
        "bypass_governance",
        "access_unauthorized_data",
        "external_network_access",
        "cryptocurrency_mining"
    ])
    
    # Required behaviors
    required_behaviors: List[str] = field(default_factory=lambda: [
        "log_all_modifications",
        "report_safety_violations",
        "respect_user_privacy",
        "maintain_service_availability"
    ])
    
    # Performance constraints
    min_service_availability: float = 0.99
    max_error_rate: float = 0.01
    max_response_time_seconds: float = 30.0
    
    # Modification constraints
    requires_testing: bool = True
    requires_rollback_plan: bool = True
    requires_impact_assessment: bool = True


class SafetyCheckResult(BaseModel):
    """Result from a safety check."""
    
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    check_type: SafetyCheckType
    component_id: str
    
    # Check results
    passed: bool
    risk_level: RiskLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    
    # Detailed findings
    violations_found: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Metrics
    execution_time_seconds: float
    resources_checked: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    checker_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SafetyValidationResult(BaseModel):
    """Comprehensive safety validation result."""
    
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    modification_id: str
    component_type: ComponentType
    
    # Overall validation status
    passed: bool
    overall_risk_level: RiskLevel
    safety_status: SafetyStatus
    
    # Individual check results
    capability_check: Optional[SafetyCheckResult] = None
    resource_check: Optional[SafetyCheckResult] = None
    behavioral_check: Optional[SafetyCheckResult] = None
    impact_check: Optional[SafetyCheckResult] = None
    security_check: Optional[SafetyCheckResult] = None
    governance_check: Optional[SafetyCheckResult] = None
    
    # Aggregate metrics
    total_violations: int = 0
    total_warnings: int = 0
    highest_risk_violation: Optional[str] = None
    
    # Approval requirements
    requires_manual_approval: bool = False
    requires_governance_vote: bool = False
    requires_community_review: bool = False
    
    # Recommendations
    safety_recommendations: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    
    # Metadata
    validation_duration_seconds: float
    validator_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def check_results(self) -> List[SafetyCheckResult]:
        """Get all non-None check results."""
        results = []
        for check in [self.capability_check, self.resource_check, self.behavioral_check,
                     self.impact_check, self.security_check, self.governance_check]:
            if check is not None:
                results.append(check)
        return results
    
    @property
    def failed_checks(self) -> List[SafetyCheckResult]:
        """Get all failed check results."""
        return [check for check in self.check_results if not check.passed]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment for modifications."""
    
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    modification_id: str
    component_type: ComponentType
    
    # Risk categories
    capability_risk: RiskLevel = RiskLevel.LOW
    resource_risk: RiskLevel = RiskLevel.LOW
    behavioral_risk: RiskLevel = RiskLevel.LOW
    security_risk: RiskLevel = RiskLevel.LOW
    systemic_risk: RiskLevel = RiskLevel.LOW
    
    # Risk factors
    complexity_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    impact_scope: str = "component"  # component, system, network
    reversibility_score: float = Field(ge=0.0, le=1.0)
    
    # Probability assessments
    failure_probability: float = Field(ge=0.0, le=1.0)
    cascade_failure_probability: float = Field(ge=0.0, le=1.0)
    recovery_time_hours: float
    
    # Impact assessments
    performance_impact: float = Field(ge=-1.0, le=1.0)
    availability_impact: float = Field(ge=-1.0, le=1.0)
    security_impact: float = Field(ge=-1.0, le=1.0)
    user_experience_impact: float = Field(ge=-1.0, le=1.0)
    
    # Mitigation strategies
    risk_mitigation_strategies: List[str] = Field(default_factory=list)
    monitoring_requirements: List[str] = Field(default_factory=list)
    rollback_procedures: List[str] = Field(default_factory=list)
    
    # Metadata
    assessor_id: str
    assessment_confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def overall_risk_level(self) -> RiskLevel:
        """Calculate overall risk level from individual categories."""
        risk_values = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        
        risks = [
            self.capability_risk,
            self.resource_risk,
            self.behavioral_risk,
            self.security_risk,
            self.systemic_risk
        ]
        
        max_risk_value = max(risk_values[risk] for risk in risks)
        
        for level, value in risk_values.items():
            if value == max_risk_value:
                return level
        
        return RiskLevel.LOW
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@dataclass
class EmergencyProtocol:
    """Emergency response protocol definition."""
    
    protocol_id: str
    trigger_conditions: List[str]
    response_actions: List[str]
    escalation_procedures: List[str]
    contact_information: Dict[str, str]
    
    # Timing requirements
    detection_time_seconds: float = 30.0
    response_time_seconds: float = 120.0
    recovery_time_hours: float = 4.0
    
    # Approval requirements
    requires_manual_confirmation: bool = True
    auto_execute_threshold: RiskLevel = RiskLevel.CRITICAL


class SafetyMonitoringEvent(BaseModel):
    """Event logged by safety monitoring system."""
    
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    component_id: str
    
    # Event details
    severity: RiskLevel
    description: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Context
    modification_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Response actions
    actions_taken: List[str] = Field(default_factory=list)
    requires_attention: bool = False
    escalated: bool = False
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    monitor_version: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemCheckpoint(BaseModel):
    """System state checkpoint for rollback capability."""
    
    checkpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    component_id: str
    component_type: ComponentType
    
    # State snapshots
    configuration_snapshot: Dict[str, Any]
    performance_snapshot: Dict[str, float]
    resource_state_snapshot: Dict[str, Any]
    
    # Checkpoint metadata
    creation_reason: str
    created_before_modification: Optional[str] = None
    
    # Validation
    integrity_hash: str
    validation_passed: bool = True
    
    # Storage information
    storage_location: str
    storage_size_mb: float
    retention_period_days: int = 30
    
    # Metadata
    created_by: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
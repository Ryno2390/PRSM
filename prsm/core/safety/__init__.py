"""
Safety and Security Systems for PRSM

This module provides comprehensive safety mechanisms for DGM-enhanced
self-modification capabilities, ensuring secure and controlled evolution.

Implements Phase 4.1 of the DGM roadmap: Safety-Constrained Self-Modification.
"""

from .safe_modification import (
    SafetyConstrainedModificationSystem,
    CapabilityBoundsChecker,
    ResourceMonitor,
    BehavioralConstraintAnalyzer,
    EmergencyShutdownSystem,
    ResourceMonitoringContext
)

from .safety_models import (
    SafetyCheckType,
    SafetyStatus,
    SafetyCheckResult,
    SafetyValidationResult,
    RiskAssessment,
    CapabilityBounds,
    ResourceLimits,
    BehavioralConstraints,
    EmergencyProtocol,
    SafetyMonitoringEvent,
    SystemCheckpoint,
    ConstraintViolationType
)

__all__ = [
    # Core safety system
    "SafetyConstrainedModificationSystem",
    
    # Safety checkers
    "CapabilityBoundsChecker", 
    "ResourceMonitor",
    "BehavioralConstraintAnalyzer",
    "EmergencyShutdownSystem",
    "ResourceMonitoringContext",
    
    # Safety models and enums
    "SafetyCheckType",
    "SafetyStatus", 
    "SafetyCheckResult",
    "SafetyValidationResult",
    "RiskAssessment",
    "CapabilityBounds",
    "ResourceLimits",
    "BehavioralConstraints",
    "EmergencyProtocol",
    "SafetyMonitoringEvent",
    "SystemCheckpoint",
    "ConstraintViolationType"
]
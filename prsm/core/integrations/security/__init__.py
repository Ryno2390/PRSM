"""
Integration Security Framework
=============================

Advanced security and compliance components for safe integration of external content:
- Enhanced sandbox manager for isolated execution with monitoring
- License compliance scanning with policy enforcement
- Vulnerability detection and assessment
- Threat detection and malware scanning
- Comprehensive audit logging
- Integration with PRSM's safety frameworks
"""

# Core security components
from .sandbox_manager import SandboxManager, SandboxResult
from .license_scanner import LicenseScanner, LicenseResult
from .vulnerability_scanner import VulnerabilityScanner, VulnerabilityResult

# Enhanced security components
from .enhanced_sandbox import EnhancedSandboxManager, EnhancedSandboxResult
from .threat_detector import ThreatDetector, ThreatResult, ThreatLevel, ThreatType
from .audit_logger import AuditLogger, SecurityEvent, EventLevel
from .security_orchestrator import SecurityOrchestrator, SecurityAssessment

# Global instances
from .audit_logger import audit_logger
from .threat_detector import threat_detector
from .enhanced_sandbox import enhanced_sandbox_manager
from .security_orchestrator import security_orchestrator

__all__ = [
    # Core components
    "SandboxManager",
    "SandboxResult",
    "LicenseScanner", 
    "LicenseResult",
    "VulnerabilityScanner",
    "VulnerabilityResult",
    
    # Enhanced components
    "EnhancedSandboxManager",
    "EnhancedSandboxResult",
    "ThreatDetector",
    "ThreatResult",
    "ThreatLevel",
    "ThreatType",
    "AuditLogger",
    "SecurityEvent",
    "EventLevel",
    "SecurityOrchestrator",
    "SecurityAssessment",
    
    # Global instances
    "audit_logger",
    "threat_detector",
    "enhanced_sandbox_manager",
    "security_orchestrator"
]
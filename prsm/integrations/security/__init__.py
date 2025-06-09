"""
Integration Security Framework
=============================

Security and compliance components for safe integration of external content:
- Sandbox manager for isolated model execution
- License compliance scanning
- Vulnerability detection and assessment
- Integration with PRSM's circuit breaker system
"""

from .sandbox_manager import SandboxManager, SandboxResult
from .license_scanner import LicenseScanner, LicenseResult
from .vulnerability_scanner import VulnerabilityScanner, VulnerabilityResult

__all__ = [
    "SandboxManager",
    "SandboxResult",
    "LicenseScanner", 
    "LicenseResult",
    "VulnerabilityScanner",
    "VulnerabilityResult"
]
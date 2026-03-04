"""
PRSM Security Hardening Module

This module provides comprehensive security hardening capabilities including:
- Security audit checklist and framework
- Automated security scanning
- Penetration testing framework
- Secrets management
- Secure environment configuration
"""

from prsm.security.audit_checklist import (
    SecurityCheck,
    SecurityAuditChecklist,
    AuditReport,
    CheckResult,
    CheckSeverity,
)
from prsm.security.scanner import (
    SecurityScanner,
    Vulnerability,
    SecurityIssue,
    SecretLeak,
    ScanResult,
)
from prsm.security.pentest import (
    PenetrationTestSuite,
    SecurityFinding,
    TestType,
    FindingSeverity,
)
from prsm.security.secrets import (
    SecretsManager,
    SecretBackend,
    SecretMetadata,
)
from prsm.security.env_config import (
    SecureEnvironment,
    SecretStrengthResult,
    EnvironmentValidationResult,
)

__all__ = [
    # Audit Checklist
    "SecurityCheck",
    "SecurityAuditChecklist",
    "AuditReport",
    "CheckResult",
    "CheckSeverity",
    # Scanner
    "SecurityScanner",
    "Vulnerability",
    "SecurityIssue",
    "SecretLeak",
    "ScanResult",
    # Penetration Testing
    "PenetrationTestSuite",
    "SecurityFinding",
    "TestType",
    "FindingSeverity",
    # Secrets Management
    "SecretsManager",
    "SecretBackend",
    "SecretMetadata",
    # Environment Configuration
    "SecureEnvironment",
    "SecretStrengthResult",
    "EnvironmentValidationResult",
]

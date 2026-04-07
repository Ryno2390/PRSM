"""
PRSM Security Hardening Module

This module provides comprehensive security hardening capabilities including:
- Security audit checklist and framework
- Automated security scanning
- Penetration testing framework
- Secrets management
- Secure environment configuration
- Integrity verification (Ring 10)
- Privacy budget tracking (Ring 10)
- Pipeline audit logging (Ring 10)
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
from prsm.security.integrity import IntegrityVerifier
from prsm.security.privacy_budget import PrivacyBudgetTracker
from prsm.security.audit_log import PipelineAuditLog

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
    # Integrity Verification (Ring 10)
    "IntegrityVerifier",
    # Privacy Budget (Ring 10)
    "PrivacyBudgetTracker",
    # Pipeline Audit Log (Ring 10)
    "PipelineAuditLog",
]

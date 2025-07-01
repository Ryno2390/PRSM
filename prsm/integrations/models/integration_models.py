"""
Integration Layer Data Models
=============================

Pydantic data models for the PRSM integration layer, extending the existing
PRSM model framework with integration-specific structures.

These models handle:
- Integration source metadata and configuration
- Import requests and results tracking
- Security scan results and compliance data
- Provenance metadata for FTNS creator rewards
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from ...core.models import PRSMBaseModel, TimestampMixin


# === Enums ===

class IntegrationPlatform(str, Enum):
    """Supported integration platforms"""
    GITHUB = "github"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    META_LLAMA = "meta_llama"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    MISTRAL = "mistral"


class ImportStatus(str, Enum):
    """Status of import operations"""
    PENDING = "pending"
    SCANNING = "scanning"
    SECURITY_CHECK = "security_check"
    IMPORTING = "importing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SECURITY_BLOCKED = "security_blocked"
    SECURITY_ERROR = "security_error"


class SecurityRisk(str, Enum):
    """Security risk levels for imported content"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LicenseType(str, Enum):
    """License compliance categories"""
    PERMISSIVE = "permissive"
    COPYLEFT = "copyleft"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


# === Core Integration Models ===

class IntegrationSource(PRSMBaseModel):
    """
    Represents an external platform source for integration
    
    Tracks metadata about external platforms and their configuration
    for seamless integration with PRSM's ecosystem.
    """
    source_id: UUID = Field(default_factory=uuid4)
    platform: IntegrationPlatform
    external_id: str = Field(..., description="Platform-specific identifier (e.g., repo name, model ID)")
    display_name: str
    description: Optional[str] = None
    owner_id: Optional[str] = None  # Platform username/organization
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('external_id')
    @classmethod
    def validate_external_id(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError('external_id cannot be empty')
        return v.strip()


class ConnectorConfig(PRSMBaseModel):
    """
    Configuration for platform-specific connectors
    
    Stores authentication credentials and platform-specific settings
    for each integration connector.
    """
    config_id: UUID = Field(default_factory=uuid4)
    platform: IntegrationPlatform
    user_id: str
    auth_token: Optional[str] = None
    api_key: Optional[str] = None
    oauth_credentials: Optional[Dict[str, str]] = None
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    active: bool = True
    
    @field_validator('auth_token', 'api_key')
    @classmethod
    def validate_sensitive_data(cls, v: Optional[str]) -> Optional[str]:
        # In production, these should be encrypted
        return v


class ImportRequest(TimestampMixin):
    """
    Request to import content from an external platform
    
    Tracks the complete lifecycle of importing external content
    into PRSM's ecosystem with proper provenance and security validation.
    """
    request_id: UUID = Field(default_factory=uuid4)
    user_id: str
    source: IntegrationSource
    import_type: str = Field(..., description="model, dataset, repository, code")
    target_location: Optional[str] = None  # Where to store in PRSM
    import_options: Dict[str, Any] = Field(default_factory=dict)
    security_scan_required: bool = True
    license_check_required: bool = True
    auto_reward_creator: bool = True
    status: ImportStatus = ImportStatus.PENDING
    error_message: Optional[str] = None


class SecurityScanResult(TimestampMixin):
    """
    Results from security scanning of imported content
    
    Provides comprehensive security assessment including vulnerability
    detection, license compliance, and risk assessment.
    """
    scan_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    risk_level: SecurityRisk = SecurityRisk.NONE
    vulnerabilities_found: List[str] = Field(default_factory=list)
    license_compliance: LicenseType = LicenseType.UNKNOWN
    compliance_issues: List[str] = Field(default_factory=list)
    scan_duration: float = Field(default=0.0, description="Scan time in seconds")
    recommendations: List[str] = Field(default_factory=list)
    approved_for_import: bool = False
    scanner_version: str = "1.0.0"
    scan_metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceMetadata(TimestampMixin):
    """
    Provenance tracking for FTNS creator rewards
    
    Maintains complete chain of custody and attribution for imported
    content to enable accurate creator compensation through FTNS.
    """
    provenance_id: UUID = Field(default_factory=uuid4)
    content_id: str
    original_creator: Optional[str] = None
    platform_source: IntegrationPlatform
    external_id: str
    import_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attribution_chain: List[Dict[str, str]] = Field(default_factory=list)
    license_info: Dict[str, Any] = Field(default_factory=dict)
    usage_metrics: Dict[str, int] = Field(default_factory=dict)
    reward_eligible: bool = True
    total_rewards_paid: float = Field(default=0.0)


class ImportResult(TimestampMixin):
    """
    Complete result of an import operation
    
    Aggregates all information about a completed import including
    security results, provenance data, and final status.
    """
    result_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    status: ImportStatus
    imported_content_id: Optional[str] = None
    ipfs_cid: Optional[str] = None
    security_scan: Optional[SecurityScanResult] = None
    provenance: Optional[ProvenanceMetadata] = None
    import_duration: float = Field(default=0.0, description="Total import time in seconds")
    content_size_bytes: Optional[int] = None
    success_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    ftns_rewards_distributed: float = Field(default=0.0)
    integration_metadata: Dict[str, Any] = Field(default_factory=dict)


class IntegrationRecord(TimestampMixin):
    """
    Master record tracking all integration activity
    
    Provides comprehensive tracking of integration operations for
    auditing, analytics, and system optimization.
    """
    record_id: UUID = Field(default_factory=uuid4)
    user_id: str
    platform: IntegrationPlatform
    operation_type: str = Field(..., description="import, sync, scan, reward")
    source_reference: str
    request_id: Optional[UUID] = None
    result_id: Optional[UUID] = None
    status: str
    operation_duration: float = Field(default=0.0)
    resources_consumed: Dict[str, float] = Field(default_factory=dict)
    ftns_cost: float = Field(default=0.0)
    success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


# === Connector Status and Health Models ===

class ConnectorHealth(PRSMBaseModel):
    """Health status for platform connectors"""
    platform: IntegrationPlatform
    status: str = Field(..., description="healthy, degraded, unhealthy, offline")
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: Optional[float] = None
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    rate_limit_remaining: Optional[int] = None
    issues: List[str] = Field(default_factory=list)


class IntegrationStats(PRSMBaseModel):
    """Statistics for integration layer operations"""
    total_imports: int = 0
    successful_imports: int = 0
    failed_imports: int = 0
    total_rewards_distributed: float = 0.0
    platforms_connected: int = 0
    active_connectors: List[IntegrationPlatform] = Field(default_factory=list)
    average_import_time: float = 0.0
    security_scans_performed: int = 0
    high_risk_content_blocked: int = 0
    license_violations_detected: int = 0
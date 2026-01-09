"""
Configuration Management
========================

Secure configuration and credential management for PRSM integration layer.

Components:
- credential_manager: Encrypted credential storage and retrieval
- config_manager: User preferences and platform configuration
- Security features: AES encryption, access controls, audit logging
- API endpoints: REST interfaces for configuration management

Features:
- Encrypted credential storage using Fernet (AES 128)
- Per-user configuration isolation
- Platform-specific settings and preferences
- Import/export capabilities for configuration backup
- Validation and health monitoring
"""

from .credential_manager import credential_manager, CredentialData, CredentialType
from .integration_config import (
    config_manager, IntegrationPreferences, PlatformConfig,
    SecurityConfig, RateLimitConfig, SecurityLevel
)

__all__ = [
    "credential_manager",
    "config_manager", 
    "CredentialData",
    "CredentialType",
    "IntegrationPreferences",
    "PlatformConfig",
    "SecurityConfig", 
    "RateLimitConfig",
    "SecurityLevel"
]
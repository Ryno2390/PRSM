"""
Integration Data Models
=======================

Pydantic data models for integration layer components, extending
PRSM's existing model framework with integration-specific structures.
"""

from .integration_models import (
    IntegrationSource,
    IntegrationRecord,
    ConnectorConfig,
    ImportRequest,
    ImportResult,
    SecurityScanResult,
    ProvenanceMetadata
)

__all__ = [
    "IntegrationSource",
    "IntegrationRecord", 
    "ConnectorConfig",
    "ImportRequest",
    "ImportResult",
    "SecurityScanResult",
    "ProvenanceMetadata"
]
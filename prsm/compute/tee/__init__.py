"""
Trusted Execution Environment (TEE) Runtime
============================================

Confidential compute for PRSM: TEE-backed execution with
differential privacy on intermediate activations.
Ring 7 of the Sovereign-Edge AI architecture.
"""

from prsm.compute.tee.models import (
    TEEType,
    TEECapability,
    DPConfig,
    ConfidentialResult,
    PrivacyLevel,
)

__all__ = [
    "TEEType",
    "TEECapability",
    "DPConfig",
    "ConfidentialResult",
    "PrivacyLevel",
]

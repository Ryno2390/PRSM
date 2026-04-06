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
from prsm.compute.tee.dp_noise import DPNoiseInjector
from prsm.compute.tee.runtime import TEERuntime, SoftwareTEERuntime
from prsm.compute.tee.confidential_executor import ConfidentialExecutor

__all__ = [
    "TEEType",
    "TEECapability",
    "DPConfig",
    "ConfidentialResult",
    "PrivacyLevel",
    "DPNoiseInjector",
    "TEERuntime",
    "SoftwareTEERuntime",
    "ConfidentialExecutor",
]

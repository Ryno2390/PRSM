"""
Mobile Agent Framework
======================

WASM-based mobile agents that travel to data instead of moving data to compute.
Ring 2 of the Sovereign-Edge AI architecture.
"""

from prsm.compute.agents.models import (
    AgentManifest,
    MobileAgent,
    DispatchStatus,
    DispatchRecord,
)

__all__ = [
    "AgentManifest",
    "MobileAgent",
    "DispatchStatus",
    "DispatchRecord",
]

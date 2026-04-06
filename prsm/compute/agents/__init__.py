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
from prsm.compute.agents.dispatcher import AgentDispatcher
from prsm.compute.agents.executor import AgentExecutor

__all__ = [
    "AgentManifest",
    "MobileAgent",
    "DispatchStatus",
    "DispatchRecord",
    "AgentDispatcher",
    "AgentExecutor",
]

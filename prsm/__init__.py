"""
PRSM: Protocol for Recursive Scientific Modeling
A P2P infrastructure protocol for open-source collaboration

PRSM aggregates latent storage, compute, and data from consumer nodes
into a mesh network accessible to third-party LLMs via MCP tools.
Contributors earn FTNS tokens for sharing resources; users leverage
PRSM infrastructure through their preferred LLMs (local or OAuth/API).

Core capabilities:
1. Native content-addressed storage (prsm.storage)
2. P2P federation and consensus (prsm.compute.federation)
3. WASM mobile agent runtime — lightweight, stateless micro-agents
   dispatched by third-party LLMs via MCP (prsm.compute.agents)
4. 10-Ring Sovereign-Edge AI architecture — the execution layer
   third-party LLMs invoke via MCP tools
5. FTNS tokenomics — minted at contribution time, earned by sharing
   latent resources (prsm.economy.tokenomics)
6. MCP server exposing PRSM tools to any LLM (prsm.core.integrations.mcp)
7. On-network governance — node users vote on protocol evolution
   (prsm.governance)
8. FTNS↔USD/USDT conversion for settlement flexibility
   (prsm.compute.chronos)

PRSM is not an AGI framework. Reasoning happens in third-party LLMs;
PRSM provides the infrastructure those LLMs use to access distributed
resources and data.
"""

__version__ = "1.7.0"
__author__ = "PRSM Team"
__email__ = "team@prsm-network.com"
__description__ = "Protocol for Recursive Scientific Modeling - A P2P infrastructure protocol for open-source collaboration"

from prsm.core.config import get_settings, settings
from prsm.core.models import *

__all__ = [
    "settings",
    "get_settings",
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]

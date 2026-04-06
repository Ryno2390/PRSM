"""
PRSM: Protocol for Recursive Scientific Modeling
A decentralized AI framework for scientific discovery

PRSM is a revolutionary framework that replaces monolithic AI models with
a federated ecosystem of specialized agents. The system enables:

- Recursive problem decomposition and collaborative solving
- Decentralized P2P network of specialized AI models
- Democratic governance through token-weighted voting
- Economic sustainability via FTNS token incentives
- Built-in safety through circuit breakers and transparency
- Continuous self-improvement through recursive enhancement

Architectural Components:
1. NWTN Orchestrator - Core AGI coordination system
2. Enhanced Agent Framework - 5-layer agent pipeline
3. Teacher Model Framework - Distilled learning system
4. Safety Infrastructure - Circuit breakers and monitoring
5. P2P Federation - Distributed model network
6. Advanced Tokenomics - FTNS economic model
7. Governance System - Democratic decision-making
8. Recursive Self-Improvement - Continuous evolution
"""

__version__ = "0.28.0"
__author__ = "PRSM Team"
__email__ = "team@prsm.org"
__description__ = "Protocol for Recursive Scientific Modeling - A decentralized AI framework for scientific discovery"

# Import core configuration and data models
# These provide the fundamental settings and data structures for the entire PRSM system
from prsm.core.config import get_settings, settings
from prsm.core.models import *

__all__ = [
    "settings",
    "get_settings",
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]

# ── Legacy compatibility shim ──────────────────────────────────────────
# DEPRECATED: The `prsm.nwtn` import path is a compatibility shim.
# Migration: `from prsm.nwtn import ...` → `from prsm.compute.nwtn import ...`
# Timeline: v0.1.x active (no warning) → v0.2.0 DeprecationWarning → v0.3.0 removed
import sys
import prsm.compute.nwtn
sys.modules['prsm.nwtn'] = sys.modules['prsm.compute.nwtn']
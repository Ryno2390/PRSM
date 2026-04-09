"""
AI Orchestrator
===============

Enterprise AI orchestration for PRSM.
This is an alias to the real AIOrchestrator implementation.
"""

# Re-export the real AIOrchestrator from the compute module
# v1.6.0 scope alignment: prsm.compute.ai_orchestration deleted in PR 3
try:
    from prsm.compute.ai_orchestration.orchestrator import AIOrchestrator
except ImportError:
    AIOrchestrator = None  # type: ignore[assignment,misc]

__all__ = ["AIOrchestrator"]

"""
AI Orchestrator
===============

Enterprise AI orchestration for PRSM.
This is an alias to the real AIOrchestrator implementation.
"""

# Re-export the real AIOrchestrator from the compute module
from prsm.compute.ai_orchestration.orchestrator import AIOrchestrator

__all__ = ["AIOrchestrator"]

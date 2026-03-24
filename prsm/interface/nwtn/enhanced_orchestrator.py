"""
Re-export shim: prsm.interface.nwtn.enhanced_orchestrator
delegates to the NeuroSymbolicOrchestrator in prsm.compute.nwtn.
"""

from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator


class EnhancedNWTNOrchestrator(NeuroSymbolicOrchestrator):
    """Alias for NeuroSymbolicOrchestrator (backward-compat shim)."""
    pass


def get_enhanced_nwtn_orchestrator() -> EnhancedNWTNOrchestrator:
    """Return a shared EnhancedNWTNOrchestrator instance."""
    return EnhancedNWTNOrchestrator(node_id="budget_orchestrator")

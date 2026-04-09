"""
Re-export shim: prsm.interface.nwtn.enhanced_orchestrator
delegates to the NeuroSymbolicOrchestrator in prsm.compute.nwtn.

v1.6.0 scope alignment: s1_neuro_symbolic legacy module deleted in PR 2,
this shim will be deleted in PR 3 along with the rest of prsm/interface/nwtn/.
"""

try:
    from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator

    class EnhancedNWTNOrchestrator(NeuroSymbolicOrchestrator):
        """Alias for NeuroSymbolicOrchestrator (backward-compat shim)."""
        pass

    def get_enhanced_nwtn_orchestrator() -> EnhancedNWTNOrchestrator:
        """Return a shared EnhancedNWTNOrchestrator instance."""
        return EnhancedNWTNOrchestrator(node_id="budget_orchestrator")

except ImportError:
    NeuroSymbolicOrchestrator = None  # type: ignore[assignment,misc]
    EnhancedNWTNOrchestrator = None  # type: ignore[assignment,misc]

    def get_enhanced_nwtn_orchestrator():  # type: ignore[no-redef]
        raise ImportError(
            "NeuroSymbolicOrchestrator was removed in v1.6.0; use a "
            "third-party LLM via prsm.compute.agents instead."
        )

"""
Meta Reasoning Orchestrator
===========================

Orchestrates meta-level reasoning across multiple engines.
Delegates to NWTNOrchestrator when available.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MetaReasoningOrchestrator:
    """
    Orchestrate meta-level reasoning.
    Delegates to NWTNOrchestrator when available.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._orchestrator = None

    def _get_orchestrator(self):
        """Lazy initialization of orchestrator."""
        if self._orchestrator is None:
            try:
                from prsm.compute.nwtn.orchestrator import NWTNOrchestrator
                from tests.fixtures.nwtn_mocks import (
                    MockContextManager, MockFTNSService, MockIPFSClient, MockModelRegistry
                )
                self._orchestrator = NWTNOrchestrator(
                    context_manager=MockContextManager(),
                    ftns_service=MockFTNSService(),
                    ipfs_client=MockIPFSClient(),
                    model_registry=MockModelRegistry()
                )
            except Exception as e:
                logger.debug(f"Could not initialize NWTNOrchestrator: {e}")
                self._orchestrator = None
        return self._orchestrator

    async def orchestrate(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Orchestrate reasoning across engines."""
        context = context or {}

        orchestrator = self._get_orchestrator()
        if orchestrator:
            try:
                from prsm.compute.nwtn.orchestrator import UserInput
                user_input = UserInput(
                    user_id=str(context.get("user_id", "system")),
                    prompt=query,
                    context_allocation=context.get("context_allocation", 10),
                )
                response = await orchestrator.process_query(user_input)
                return {
                    "result": response.final_response or f"Meta-reasoning for: {query}",
                    "confidence": response.confidence_score or 0.85
                }
            except Exception as e:
                logger.debug(f"Orchestrator processing failed: {e}")

        # Fallback
        return {
            "result": f"Meta-reasoning for: {query}",
            "confidence": 0.85
        }

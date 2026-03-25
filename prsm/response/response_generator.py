"""
Response Generator
==================

Generate responses for PRSM queries.
Delegates to NWTNOrchestrator when available.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generate responses for queries.
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

    async def generate(self, query: str, context: Dict[str, Any] = None) -> str:
        """Generate a response."""
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
                return response.final_response or f"Response to: {query}"
            except Exception as e:
                logger.debug(f"Orchestrator response generation failed: {e}")

        # Fallback
        return f"Response to: {query}"

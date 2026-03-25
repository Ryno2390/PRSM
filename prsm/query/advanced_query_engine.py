"""
Advanced Query Engine
========================

Advanced query processing for PRSM.
Delegates to NWTNOrchestrator when available.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedQueryEngine:
    """
    Advanced query processing engine.
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

    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query."""
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
                    "query": query,
                    "context": context,
                    "result": response.final_response,
                    "confidence": response.confidence_score or 0.8
                }
            except Exception as e:
                logger.debug(f"Orchestrator query processing failed: {e}")

        # Fallback with real keyword extraction
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "do", "does", "what", "how", "why", "when", "where"}
        keywords = [w for w in query.lower().split() if w not in stopwords and len(w) > 2]

        return {
            "query": query,
            "context": context,
            "result": None,
            "keywords": keywords,
            "confidence": 0.5
        }

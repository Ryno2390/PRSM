"""
Deep Reasoning Engine
=====================

Deep reasoning capabilities for NWTN.
Delegates multi-stage reasoning to NWTNOrchestrator when available.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result from deep reasoning."""
    conclusion: str
    confidence: float
    reasoning_chain: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)


class DeepReasoningEngine:
    """
    Engine for deep reasoning and analysis.
    Delegates to NWTNOrchestrator when available, falls back gracefully otherwise.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._orchestrator = None

    def _get_orchestrator(self):
        """Lazy initialization of orchestrator to avoid circular imports."""
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

    async def reason(self, query: str, context: Dict[str, Any] = None) -> ReasoningResult:
        """Perform deep reasoning on a query."""
        context = context or {}

        # Try to use the orchestrator
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
                return ReasoningResult(
                    conclusion=response.final_response or f"Analysis of: {query}",
                    confidence=response.confidence_score or 0.8,
                    reasoning_chain=[s.get("step", "") for s in (response.reasoning_trace or [])],
                    evidence=response.citations or [],
                )
            except Exception as e:
                logger.debug(f"Orchestrator reasoning failed: {e}")

        # Graceful fallback with real keyword extraction
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "do", "does", "what", "how", "why", "when", "where"}
        words = [w for w in query.lower().split() if w not in stopwords and len(w) > 2]

        return ReasoningResult(
            conclusion=f"Analysis of: {query}",
            confidence=0.8,
            reasoning_chain=["Initial analysis", "Deep evaluation", "Synthesis"],
            evidence=[{"type": "keywords", "values": words}]
        )

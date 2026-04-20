"""Phase 3 marketplace exception types."""
from __future__ import annotations

from prsm.compute.remote_dispatcher import MissingAttestationError


class NoEligibleProvidersError(RuntimeError):
    """Orchestrator's directory + filter pass returned zero providers.

    Can fire either at orchestration start (no listings match policy at
    all) or mid-dispatch (all remaining eligible providers rejected
    their quotes / preempted / failed)."""


class PriceQuoteRejectedError(RuntimeError):
    """A provider rejected the price-quote request. Not typically raised
    directly — the orchestrator retries on the next eligible provider
    instead. Surfaced only when all providers reject."""


__all__ = [
    "MissingAttestationError",
    "NoEligibleProvidersError",
    "PriceQuoteRejectedError",
]

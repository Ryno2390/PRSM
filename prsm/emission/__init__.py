"""Phase 8 emission-side Python wrappers.

Read-only surface around the on-chain EmissionController + CompensationDistributor
contracts. Used by operator dashboards, Foundation treasury models, and
monitoring alerting (epoch-boundary notifications).

Per docs/2026-04-22-phase8-design-plan.md §4.3 + §6 Task 4.
"""
from prsm.emission.emission_client import (
    EmissionClient,
    EmissionState,
    MintEvent,
)
from prsm.emission.watcher import EmissionWatcher

__all__ = [
    "EmissionClient",
    "EmissionState",
    "EmissionWatcher",
    "MintEvent",
]

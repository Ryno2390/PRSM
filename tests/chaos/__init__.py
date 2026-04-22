"""Phase 6 chaos harness.

In-process deterministic simulation of PRSM's P2P primitives under
adversarial conditions. Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md
§4.3 and §6 Task 7.

The harness does NOT simulate libp2p wire traffic — that's the Task 4
testnet exercise. What it DOES is exercise the liveness + rate-limit
state machines at scale (tens to hundreds of peers) with reproducible
adversarial inputs, so regressions in those primitives surface in CI
without a real network.
"""
from tests.chaos.harness import (
    ChaosReport,
    ChaosScenario,
    SimNetwork,
    SimPeer,
)

__all__ = [
    "ChaosReport",
    "ChaosScenario",
    "SimNetwork",
    "SimPeer",
]

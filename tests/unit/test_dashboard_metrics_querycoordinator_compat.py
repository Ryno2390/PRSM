"""Sprint 453 — F13 fix: dashboard_metrics QueryOrchestrator compat.

Production-blocking bug surfaced during sprint 453 §13 MCP-tool
live-verification sweep: GET /rings/status returned HTTP 500 with
text/plain "Internal Server Error". Trace:

  File "prsm/observability/dashboard_metrics.py", line 51 + 91
    metrics={"traces": len(self._node.agent_forge.traces) ...}
  AttributeError: 'QueryOrchestrator' object has no attribute 'traces'

Root cause: sprint 173 swapped `node.agent_forge` from the legacy
AgentForge module (which had `.traces`) to a QueryOrchestrator
instance (which doesn't). `dashboard_metrics.py` was written for
the legacy shape and crashed on every /rings/status call after
the swap.

Severity: production-blocking for the MCP `prsm_node_status` tool
that depends on /rings/status. Operators using AI assistants to
triage their node got opaque "Cannot reach PRSM node: 500" errors
even though the node was perfectly healthy.

Fix: defensive `getattr(forge, 'traces', []) or []` so the metric
reports 0 when QueryOrchestrator is wired (which doesn't track
per-query traces in the legacy AgentForge sense). Applied at
both occurrences: collect_ring_status + get_summary.

These pins defend against the regression — fixture-drift would
re-introduce the crash by accessing .traces directly.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from prsm.observability.dashboard_metrics import DashboardMetrics


def test_get_summary_with_query_orchestrator_no_traces():
    """The headline regression: agent_forge is a
    QueryOrchestrator (NO .traces attr) → get_summary
    must not raise."""
    node = MagicMock()
    # QueryOrchestrator-shaped: NO .traces attribute.
    # Use spec_set to prevent MagicMock from auto-generating .traces.
    class _QOLike:
        pass
    node.agent_forge = _QOLike()
    node.privacy_budget = None
    # Other ring-status reads use getattr-with-default already;
    # ensure they don't crash either
    node.agent_dispatcher = None
    node.swarm_coordinator = None

    metrics = DashboardMetrics(node=node)
    summary = metrics.get_summary()

    # Headline invariant: no AttributeError
    assert summary["rings_total"] == 10
    # Ring 5 (The Brain) should report 0 traces, not crash
    ring5 = next(r for r in summary["rings"] if r["ring"] == 5)
    assert ring5["name"] == "The Brain"
    assert ring5["initialized"] is True
    # forge_metrics dict still produced with traces_collected=0
    assert summary["forge"]["traces_collected"] == 0


def test_get_summary_with_legacy_forge_having_traces():
    """Backwards-compat: if a legacy AgentForge IS wired
    (has .traces list), the count should be honored."""
    node = MagicMock()
    class _LegacyForge:
        traces = ["trace1", "trace2", "trace3"]
    node.agent_forge = _LegacyForge()
    node.privacy_budget = None

    metrics = DashboardMetrics(node=node)
    summary = metrics.get_summary()
    assert summary["forge"]["traces_collected"] == 3


def test_get_summary_with_no_forge_at_all():
    """If agent_forge is None (legacy default), still don't
    crash — traces_collected key omitted or set to 0."""
    node = MagicMock()
    node.agent_forge = None
    node.privacy_budget = None

    metrics = DashboardMetrics(node=node)
    summary = metrics.get_summary()
    # No forge → forge dict should be empty or have 0
    assert "forge" in summary
    # And rings_total still rendered
    assert summary["rings_total"] == 10


def test_collect_ring_status_with_query_orchestrator():
    """collect_ring_status (called by get_summary) must
    also handle the QueryOrchestrator case at Ring 5
    construction."""
    node = MagicMock()
    class _QOLike:
        pass
    node.agent_forge = _QOLike()

    metrics = DashboardMetrics(node=node)
    rings = metrics.collect_ring_status()

    # Ring 5 is "The Brain" — must not crash building it
    ring5 = next(r for r in rings if r.ring_number == 5)
    assert ring5.name == "The Brain"
    # And the metrics dict must surface traces=0 (not absent)
    assert ring5.metrics.get("traces") == 0

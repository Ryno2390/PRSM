"""Sprint 732 — operator runbook refresh covering sprints 713-731.

Sprint 709 shipped the consolidated operator runbook with the
sprint-697 base + sprints 700-708. The 18 sprints since (711-731)
added 7 new env vars + 1 new CLI command (`prsm node streams`) +
significant transport-layer security upgrades. Without runbook
refresh, operators deploying fresh today wouldn't know:

- The new defenses exist (would deploy without them)
- How to tune them for different deployment sizes
- The observability CLI exists (would have no visibility into
  the in-flight streams sprint 711-731 enables)

These pin tests defend against runbook drift on the load-bearing
operational surface.
"""
from __future__ import annotations

from pathlib import Path


RUNBOOK = (
    Path(__file__).parent.parent.parent
    / "docs" / "operations" / "parallax-inference-deploy.md"
)


def _runbook_text() -> str:
    return RUNBOOK.read_text()


def test_runbook_documents_sprint_713_queue_maxsize_env():
    text = _runbook_text()
    assert "PRSM_CHAIN_STREAM_QUEUE_MAXSIZE" in text, (
        "Sprint 713's bounded receive queue env must be documented"
    )


def test_runbook_documents_sprint_721_725_size_limit_envs():
    text = _runbook_text()
    assert "PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES" in text
    assert "PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES" in text


def test_runbook_documents_sprint_723_726_per_peer_caps():
    text = _runbook_text()
    assert "PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY" in text
    assert "PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY" in text


def test_runbook_documents_sprint_728_729_timeouts():
    text = _runbook_text()
    assert "PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S" in text
    assert "PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S" in text


def test_runbook_documents_prsm_node_streams_cli():
    """Sprint 722's observability CLI must be in the runbook so
    operators know it exists."""
    text = _runbook_text()
    assert "prsm node streams" in text, (
        "Sprint 722 observability CLI must be documented in the "
        "operator runbook"
    )


def test_runbook_documents_transport_sender_binding():
    """Sprints 730-731 are transport-layer security fixes that
    take effect on daemon restart with no operator action — but
    operators should be aware of the security model upgrade."""
    text = _runbook_text()
    assert "sender_id" in text or "sender binding" in text, (
        "Sprint 730-731 transport-layer security upgrade must be "
        "documented so operators know what their daemon protects"
    )

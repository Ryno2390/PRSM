"""Sprint 605 — /admin/chain-exec-ping endpoint.

Source-grep invariants for the new endpoint (full integration
testing impractical without a live multi-host bench — covered by
live attestation during sprint deployment).
"""
from __future__ import annotations

from pathlib import Path


def test_endpoint_defined_in_api():
    src = (
        Path(__file__).parent.parent.parent
        / "prsm" / "node" / "api.py"
    ).read_text(encoding="utf-8")
    assert '"/admin/chain-exec-ping"' in src, (
        "Sprint 605: chain-exec-ping endpoint not defined"
    )


def test_endpoint_uses_build_send_message_adapter():
    src = (
        Path(__file__).parent.parent.parent
        / "prsm" / "node" / "api.py"
    ).read_text(encoding="utf-8")
    assert "build_send_message_adapter" in src, (
        "Sprint 605: endpoint must use sprint-596 adapter"
    )


def test_endpoint_dispatches_to_executor_thread():
    """The adapter is sync + drives the loop; calling it directly
    from the loop thread deadlocks. Endpoint MUST run_in_executor.
    """
    src = (
        Path(__file__).parent.parent.parent
        / "prsm" / "node" / "api.py"
    ).read_text(encoding="utf-8")
    # The chain-exec-ping section must use run_in_executor
    idx = src.find('"/admin/chain-exec-ping"')
    assert idx > 0
    section = src[idx:idx + 4000]
    assert "run_in_executor" in section, (
        "Sprint 605: must run adapter in a worker thread to "
        "avoid deadlocking the loop thread"
    )

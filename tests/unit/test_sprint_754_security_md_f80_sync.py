"""Sprint 754 — SECURITY.md + audit-doc exec summary synced to F80.

Sprint 746 brought both artifacts to F73. Sprints 747-753 closed
F74 through F80 (7 more F-classes). Sprint 754 refreshes both
artifacts to the F80 endpoint.

These pin tests defend each of the F74-F80 fixes against drift
in SECURITY.md — every fix shipped now has at least one
identifiable mention in the doc external auditors read first.
"""
from __future__ import annotations

from pathlib import Path


SECURITY_MD = Path(__file__).parent.parent.parent / "SECURITY.md"


def _text() -> str:
    return SECURITY_MD.read_text()


def test_security_md_documents_f74_info_health_detailed():
    """F74 (sprint 747): /info + /health/detailed recon."""
    text = _text()
    assert "F74" in text


def test_security_md_documents_f75_status_recon():
    """F75 (sprint 748): /status + /rings/status — the operator
    FTNS balance + 13 subsystem stats leak."""
    text = _text()
    assert "F75" in text


def test_security_md_documents_f76_peers_topology():
    """F76 (sprint 749): /peers network topology leak."""
    text = _text()
    assert "F76" in text


def test_security_md_documents_f77_balance_recon():
    """F77 (sprint 750): /balance + /bootstrap/status. The
    operator's complete financial profile leak."""
    text = _text()
    assert "F77" in text


def test_security_md_documents_f78_transactions_staking_settlement():
    """F78 (sprint 751): /transactions + /staking/status +
    /settlement/* — 200-tx history + stake position leaks."""
    text = _text()
    assert "F78" in text


def test_security_md_documents_f79_audit_ledger_sync():
    """F79 (sprint 752): /balance/onchain + /audit/* +
    /ledger/sync/stats. Access-log + sync-state recon."""
    text = _text()
    assert "F79" in text


def test_security_md_documents_f80_agents_spending_privacy():
    """F80 (sprint 753): /agents/spending + /privacy/budget.
    Round-number 50-blocker milestone closure."""
    text = _text()
    assert "F80" in text


def test_security_md_names_full_f30_f80_range():
    """The cumulative arc range should be visible — the
    exec summary in the audit doc + SECURITY.md should both
    quote F30-F80 (or equivalent) so an auditor skimming sees
    the full breadth."""
    text = _text()
    assert (
        "F30 through F80" in text
        or "F30-F80" in text
        or "F30 → F80" in text
    )

"""Sprint 733 — pin tests for the audit-readiness doc executive summary.

The executive summary at the top of `docs/2026-05-22-parallax-
inference-audit-readiness.md` is what an external auditor reads
FIRST. Pre-sprint-733 it covered sprints 680-700 but didn't
mention the F50-F64 hardening arc that happened in this session.

Without the summary update, an auditor would have to scroll to
§7 (known limits) and §8 (changelog) to learn that 35 F-class
production-blockers were closed — easy to miss the depth of
hardening from a 30-second skim.

Sprint 733 updates the exec summary with a wire-protocol-
hardening paragraph naming the 10 audit dimensions + the
foundation-fix pair (F63/F64). These pin tests defend against
drift on the doc's most-visible surface.
"""
from __future__ import annotations

from pathlib import Path


AUDIT_DOC = (
    Path(__file__).parent.parent.parent
    / "docs" / "2026-05-22-parallax-inference-audit-readiness.md"
)


def _exec_summary_text() -> str:
    """Return the executive-summary section (between "## 1." and
    "## 2.")."""
    text = AUDIT_DOC.read_text()
    start = text.find("## 1. Executive summary")
    end = text.find("## 2.")
    assert start > 0 and end > start
    return text[start:end]


def test_exec_summary_names_f_class_arc_range():
    """The summary must name the F-class arc range so an auditor
    skimming sees the depth (35 blockers closed)."""
    summary = _exec_summary_text()
    # Sprint 746 widened from F64 → F73 (admin-auth + recon arc).
    # Accept any current range that includes the cumulative endpoint.
    assert (
        "F30 through F73" in summary
        or "F30 → F73" in summary
        or "F30-F73" in summary
    ), (
        "exec summary must name the F-class arc range so external "
        "auditors see the breadth of hardening (cumulative F30-F73 "
        "as of sprint 745)"
    )


def test_exec_summary_mentions_streaming_unary_parity():
    """The streaming + unary parity is one of the key load-bearing
    claims — 4 cross-path defenses (collision, size, cap, hijack)
    apply to BOTH paths. Auditors looking for "is the unary path
    audited too?" need to see this."""
    # Collapse whitespace so soft-line-wraps don't break the check.
    summary = " ".join(_exec_summary_text().split())
    assert (
        "Streaming + unary" in summary
        or "streaming + unary" in summary
        or "Streaming and unary" in summary
    ), "exec summary must surface streaming + unary parity claim"


def test_exec_summary_documents_transport_sender_binding_upgrade():
    """F63/F64 is foundational — without it the rest of the
    wire-protocol hardening rests on attacker-controlled wire
    data. Auditors need to see this AND that it protects
    ledger_sync FTNS transfers (highest-stakes consequence)."""
    summary = _exec_summary_text()
    assert "ledger_sync" in summary or "FTNS transfers" in summary, (
        "exec summary must surface that sprint 730/731 protect "
        "ledger FTNS transfers (highest-stakes wire-level fix)"
    )
    # Crypto foundation must be named
    assert "sender_id" in summary or "peer_id" in summary


def test_exec_summary_cites_sprint_range():
    """22-sprint arc 711-731 should be visible — gives auditor a
    concrete window to look at in commit history."""
    summary = _exec_summary_text()
    assert "711-731" in summary or "sprints 711" in summary

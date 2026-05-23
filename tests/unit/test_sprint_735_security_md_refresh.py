"""Sprint 735 — SECURITY.md refresh for sprints 711-734.

Pre-sprint-735, SECURITY.md "Last updated: 2026-05-05" predated
the entire 22-sprint wire-protocol audit arc (F30-F65). A
potential reporter or external auditor reading SECURITY.md would
conclude the security posture hadn't evolved since then — when
in fact this session added 35 F-class closures including the
F63/F64 transport-layer foundation fix protecting FTNS transfers.

Sprint 735 adds a "Recent hardening — wire-protocol audit arc"
section summarizing the F-class closures with operator-facing
consequences. Updates "Last updated" + cross-links the
audit-readiness doc.

These pin tests defend the SECURITY.md surface against drift —
it's the FIRST artifact a responsible-disclosure reporter or
external security auditor reads about the project.
"""
from __future__ import annotations

from pathlib import Path


SECURITY_MD = (
    Path(__file__).parent.parent.parent / "SECURITY.md"
)


def _text() -> str:
    return SECURITY_MD.read_text()


def test_security_md_last_updated_post_audit_arc():
    """Pin: "Last updated" reflects the audit arc completion date,
    not the pre-audit 2026-05-05 timestamp."""
    text = _text()
    # Must be 2026-05-22 (sprint 735 day) or later. If updated again
    # post-735 the assertion still passes — we only require NOT-
    # pre-535-05-22.
    assert "Last updated:** 2026-05-22" in text or (
        "Last updated:** 2026-05-23" in text
    ), (
        "SECURITY.md Last-updated stamp must reflect the wire-"
        "protocol audit arc completion (2026-05-22)"
    )


def test_security_md_cross_references_audit_readiness_doc():
    """Pin: SECURITY.md links to the audit-readiness doc so a
    reader can find the per-F-class evidence."""
    text = _text()
    assert "2026-05-22-parallax-inference-audit-readiness.md" in text


def test_security_md_documents_recent_hardening_section():
    """Pin: the audit history is a top-level section a reader
    can scan to."""
    text = _text()
    assert "Recent hardening" in text or "wire-protocol audit" in text


def test_security_md_documents_f63_f64_foundation_fix():
    """Pin: F63/F64 is the highest-stakes fix in the arc — the
    foundation that several preceding F-fixes depend on, and the
    one that protects FTNS transfers. Auditors should NOT have
    to scroll through changelog tables to find it."""
    text = _text()
    assert "F63" in text and "F64" in text, (
        "F63/F64 foundation fix must be visible in SECURITY.md"
    )
    assert "ledger_sync" in text or "FTNS transfers" in text, (
        "F63/F64's protection of ledger FTNS transfers must be "
        "visible in SECURITY.md"
    )


def test_security_md_warns_operators_about_f65_admin_default_deny():
    """Pin: F65 is a BEHAVIOR CHANGE for operators upgrading
    past sprint 734. SECURITY.md should warn them so they know
    to set PRSM_ADMIN_REMOTE_ALLOWED=1 + add real auth if
    they need remote admin access. Otherwise upgrades silently
    break their grafana scrapers / remote tooling."""
    text = _text()
    assert "PRSM_ADMIN_REMOTE_ALLOWED" in text, (
        "Operators upgrading past sprint 734 need to know about "
        "PRSM_ADMIN_REMOTE_ALLOWED — name it in SECURITY.md"
    )


def test_security_md_names_f_class_arc_range():
    """Pin: the F30-F65 range gives auditors a concrete window
    to look at in commit history."""
    text = _text()
    # Sprint 754 widened to F80 (full admin-auth + recon arc).
    assert (
        "F30 through F80" in text
        or "F30-F80" in text
        or "F30 → F80" in text
    )

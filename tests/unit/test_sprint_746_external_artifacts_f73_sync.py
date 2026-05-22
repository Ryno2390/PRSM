"""Sprint 746 — SECURITY.md + audit-doc exec summary synced to F73.

Sprint 735 refreshed SECURITY.md when the arc was at F65. Sprint
733 wrote the audit-doc exec summary at F64. Both have drifted
8-9 F-classes since (F66-F73 closed via sprints 737, 738, 739,
740, 741, 742, 743, 744, 745). External auditors / responsible-
disclosure reporters reading these artifacts would underestimate
the depth of recent hardening.

Sprint 746 brings both artifacts current. These pin tests defend
the new content against drift — every F-class fix shipped now
has at least one identifiable mention in the high-level artifact
an external party reads.
"""
from __future__ import annotations

from pathlib import Path


SECURITY_MD = (
    Path(__file__).parent.parent.parent / "SECURITY.md"
)
AUDIT_DOC = (
    Path(__file__).parent.parent.parent
    / "docs" / "2026-05-22-parallax-inference-audit-readiness.md"
)


def _security_text() -> str:
    return SECURITY_MD.read_text()


def _audit_text() -> str:
    return AUDIT_DOC.read_text()


def test_security_md_documents_dns_rebinding_f71():
    """F71 (sprint 743) is the canonical attack class against
    loopback-as-auth services. Auditors should see it named
    in SECURITY.md, not buried in the audit doc."""
    text = _security_text()
    assert "F71" in text
    assert "DNS-rebinding" in text or "DNS rebinding" in text


def test_security_md_documents_docs_recon_f72():
    """F72 (sprint 744) closes the reconnaissance vector. Worth
    naming in SECURITY.md so auditors know /docs is no longer
    free intel."""
    text = _security_text()
    assert "F72" in text
    assert "openapi" in text.lower() or "/docs" in text


def test_security_md_documents_metrics_leak_f73():
    """F73 (sprint 745) — financial intel + counter state was
    public via Prometheus exposition. Worth naming."""
    text = _security_text()
    assert "F73" in text
    assert "metrics" in text.lower() or "Prometheus" in text


def test_security_md_documents_per_requester_rate_limit_f69():
    """F69 (sprint 741) — false-confidence bug class where env
    name promised protection that wasn't wired. Distinct from
    the auth gates; auditors should see it called out."""
    text = _security_text()
    assert "F69" in text


def test_security_md_documents_http_body_size_f70():
    """F70 (sprint 742) — HTTP-side memory DoS. Sibling of wire-
    protocol F55/F58."""
    text = _security_text()
    assert "F70" in text


def test_audit_doc_exec_summary_mentions_admin_auth_arc():
    """Audit-doc exec summary names the F65-F73 admin-auth arc
    so an external auditor sees both the wire-protocol hardening
    AND the HTTP-surface hardening from a 30-second skim."""
    summary_start = _audit_text().find("## 1. Executive summary")
    summary_end = _audit_text().find("## 2.")
    summary = _audit_text()[summary_start:summary_end]
    assert "F65-F73" in summary or (
        "F65" in summary and "F73" in summary
    )


def test_audit_doc_exec_summary_cites_sprint_745_endpoint():
    """The 35-sprint arc range (711-745) should be visible in
    the exec summary so the breadth is concrete."""
    summary_start = _audit_text().find("## 1. Executive summary")
    summary_end = _audit_text().find("## 2.")
    summary = _audit_text()[summary_start:summary_end]
    assert "711-745" in summary or (
        "711" in summary and "745" in summary
    )

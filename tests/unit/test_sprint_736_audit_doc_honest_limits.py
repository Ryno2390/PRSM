"""Sprint 736 — pin tests for audit-doc § 7 honest limits.

Sprint 729 introduced a real but partial mitigation: the
streaming wall-time bound catches glacial yields but NOT a
single `__next__()` call that hangs indefinitely on the event
loop. This limit was documented in the sprint-729 commit message
but NOT in audit-doc § 7 (the canonical honest-limits section).

Sprint 731 explicitly excluded MSG_GOSSIP from the transport-
level sender-binding. The implication — gossip-origin spoofing
is possible — was acknowledged in commit messages but never
written into § 7 either.

Sprint 736 adds both as § 7.7 and § 7.8. An external auditor
reading § 7 should see ALL the honest scope carveouts the audit
arc didn't close, not just the closed ones.
"""
from __future__ import annotations

from pathlib import Path


AUDIT_DOC = (
    Path(__file__).parent.parent.parent
    / "docs" / "2026-05-22-parallax-inference-audit-readiness.md"
)


def _text() -> str:
    return AUDIT_DOC.read_text()


def test_audit_doc_documents_sprint_729_single_yield_hang():
    """Pin: § 7.7 (or equivalent) acknowledges that sprint 729's
    wall-time bound doesn't catch a single hung __next__() call.
    Auditors need to know this honest carveout."""
    text = _text()
    assert "single-yield hang" in text.lower() or (
        "__next__" in text and "hang" in text.lower()
    ), (
        "audit doc § 7 must acknowledge the sprint-729 single-"
        "yield hang carveout"
    )


def test_audit_doc_documents_gossip_out_of_scope():
    """Pin: § 7.8 (or equivalent) names that MSG_GOSSIP was NOT
    in the F30-F65 audit arc. Mentioning sprint-731 exclusion
    and the gossip-origin spoofing implication is honest scope."""
    text = _text()
    assert "Gossip layer" in text or "gossip-origin" in text, (
        "audit doc § 7 must name that the gossip layer is out of "
        "this audit arc's scope (sprint-731 exclusion)"
    )


def test_audit_doc_sprint_729_carveout_offers_mitigation_path():
    """The honest limit doc should suggest what an operator CAN
    do today (sprint-723 cap + systemd watchdog) rather than
    leaving the reader stranded with "this hangs forever"."""
    text = _text()
    assert "WatchdogSec" in text or "watchdog" in text.lower(), (
        "sprint-729 carveout should suggest operator-side "
        "mitigation (systemd watchdog)"
    )


def test_audit_doc_gossip_carveout_bounds_concrete_impact():
    """The honest limit doc should NOT just say "gossip is
    unsafe"; it should bound the concrete impact (financial path
    is MSG_DIRECT not gossip, so funds aren't at risk; presence
    + reputation are the affected surfaces)."""
    text = _text()
    assert "financial transfer path uses" in text or (
        "MSG_DIRECT" in text and "funds-at-risk" in text.lower()
    ) or (
        "funds aren't at risk" in text.lower()
        or "not funds-at-risk" in text.lower()
    )

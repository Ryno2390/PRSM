"""Sprint 759 — pin tests for the operator runbook scheduling section.

Sprint 755-758 shipped the operator-controlled active-window
scheduling. Sprint 759 documents it in the operator runbook so
operators reading the deploy guide see how to use it.

These pin tests defend the new "Active-window scheduling
(sprints 755-758)" section against drift.
"""
from __future__ import annotations

from pathlib import Path


RUNBOOK = (
    Path(__file__).parent.parent.parent
    / "docs" / "operations" / "parallax-inference-deploy.md"
)


def _text() -> str:
    return RUNBOOK.read_text()


def test_runbook_has_active_window_scheduling_section():
    """Section title grep-findable so operators searching for
    'scheduling' or 'PRSM_ACTIVE_HOURS' land here."""
    text = _text()
    assert "Active-window scheduling" in text
    assert "PRSM_ACTIVE_HOURS" in text


def test_runbook_documents_active_timezone_env():
    """Both env vars must be documented."""
    text = _text()
    assert "PRSM_ACTIVE_TIMEZONE" in text


def test_runbook_explains_503_retry_after_behavior():
    """Operators need to see what happens to inference requests
    outside the window."""
    text = _text()
    assert "503" in text and "Retry-After" in text


def test_runbook_explains_announce_skip_behavior():
    """Operators need to know peers evict during the inactive
    window — explains why their daemon "disappears" from the
    pool but is still running."""
    text = _text()
    assert "announce" in text.lower()
    assert "evict" in text.lower() or "evicts" in text.lower()


def test_runbook_explains_fast_reannounce_on_resume():
    """Sprint 758 — operators see fast re-announce within ~10s
    of window resume. Worth documenting so they know what to
    expect."""
    text = _text()
    assert "immediate re-announce" in text.lower() or (
        "fast" in text.lower() and "announce" in text.lower()
    )


def test_runbook_documents_prsm_node_schedule_cli():
    """Sprint 757 — the inspection CLI must be discoverable
    in the runbook."""
    text = _text()
    assert "prsm node schedule" in text


def test_runbook_shows_cross_midnight_example():
    """The 22:00-08:00 cross-midnight pattern is the canonical
    operator example (overnight only)."""
    text = _text()
    assert "22:00-08:00" in text


def test_runbook_explains_backward_compat_unset_env():
    """Operators who DON'T set the env see no behavior change.
    This is critical for the existing operator fleet — sprint 755
    shipped without breaking anything."""
    text = _text()
    assert "always-active" in text or "backward-compat" in text


def test_runbook_changelog_includes_sprints_755_758():
    """Changelog row for the scheduling arc."""
    text = _text()
    assert "755-758" in text

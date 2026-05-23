"""Sprint 768 — runbook documentation closes the auto-claim arc.

Sprint 765-767 shipped worker + lifecycle + CLI. Sprint 768
documents in the operator runbook so deploying operators see
how to opt in.

Pin tests defend the new "Auto-claim FTNS rewards" runbook
section against drift.
"""
from __future__ import annotations

from pathlib import Path


RUNBOOK = (
    Path(__file__).parent.parent.parent
    / "docs" / "operations" / "parallax-inference-deploy.md"
)


def _text() -> str:
    return RUNBOOK.read_text()


def test_runbook_has_auto_claim_section():
    """Section title grep-findable."""
    text = _text()
    assert "Auto-claim FTNS rewards" in text


def test_runbook_documents_both_env_vars():
    text = _text()
    assert "PRSM_AUTO_CLAIM_THRESHOLD_FTNS" in text
    assert "PRSM_AUTO_CLAIM_INTERVAL_S" in text


def test_runbook_documents_default_disabled():
    """Backward-compat unset → disabled (manual claim only)."""
    text = _text()
    assert "disabled" in text.lower() and "manual" in text.lower()


def test_runbook_documents_prsm_node_auto_claim_cli():
    """CLI command discoverable."""
    text = _text()
    assert "prsm node auto-claim" in text


def test_runbook_documents_failure_safe_loop():
    """Operators need to know transient failures don't crash
    the worker."""
    text = _text()
    assert "DON'T crash" in text or (
        "don't crash" in text.lower()
    ) or "failure counter" in text.lower()


def test_runbook_documents_honest_scope_staking_only():
    """Honest carveout: auto-claim only handles STAKING rewards,
    not content royalties or other payout paths."""
    text = _text()
    assert "STAKING rewards" in text or (
        "staking rewards" in text.lower() and "only" in text.lower()
    )


def test_runbook_changelog_includes_sprints_765_767():
    """Changelog row references the auto-claim arc."""
    text = _text()
    assert "765-767" in text or (
        "765" in text and "767" in text and "auto-claim" in text.lower()
    )


def test_runbook_documents_gas_threshold_guidance():
    """Operators need guidance on minimum-viable threshold so
    they don't burn gas on tiny claims."""
    text = _text()
    assert "gas" in text.lower()
    # Either a concrete recommendation or numeric guidance
    assert (
        "10×" in text
        or "10x" in text.lower()
        or "100 FTNS" in text
    )

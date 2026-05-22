"""Sprint 710 — pin test for README's external-readiness section.

The GitHub repo landing page (README.md) is what every random visitor
sees first. Sprint 710 added a "Verify a PRSM inference receipt in
30 seconds" callout linking the three external-readiness artifacts
(verifier script + sample receipts + audit doc + deploy runbook).

These tests defend the callout against drift: if someone refactors
the README and accidentally drops the demo block, this test fails.
"""
from __future__ import annotations

from pathlib import Path

import pytest


README_PATH = Path(__file__).parent.parent.parent / "README.md"


def _readme_text() -> str:
    return README_PATH.read_text()


def test_readme_links_audit_readiness_doc():
    """Sprint 699's audit-readiness summary is the headline
    external-facing artifact — must be reachable from README."""
    text = _readme_text()
    assert "2026-05-22-parallax-inference-audit-readiness.md" in text


def test_readme_links_verifier_script():
    """Sprint 703's standalone verifier is the most-shareable demo
    surface — README must show the literal command."""
    text = _readme_text()
    assert "verify_prsm_receipt.py" in text
    # The literal pip-install + run command
    assert "pip install web3 cryptography" in text


def test_readme_references_sample_receipts():
    """Sample receipts (sprint 706-708) must be linked so README
    readers can find the 4 reference receipts."""
    text = _readme_text()
    assert "sample-receipts" in text


def test_readme_references_operator_runbook():
    """Sprint 697/709 deploy runbook is the onboarding doc — must
    be reachable for would-be operators."""
    text = _readme_text()
    assert "parallax-inference-deploy.md" in text


def test_readme_shows_tamper_detection_outcome():
    """The demo must surface the success-AND-failure cases so a
    reader sees both `✓ VALID` and `✗ INVALID` are real outcomes
    of the verifier."""
    text = _readme_text()
    assert "VALID" in text
    assert "INVALID" in text or "tamper" in text.lower()


def test_readme_30_second_demo_callout_present():
    """The "verify in 30 seconds" framing is the headline pitch
    for non-engineer readers — pin it."""
    text = _readme_text()
    assert "30 seconds" in text or "30-second" in text

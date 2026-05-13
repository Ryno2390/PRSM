"""Sprint 381 — PRSM-CR-2026-05-13-2 ratification pin.

Council resolution ratifies the 26-sprint cycle (355-380):
Vision §14 item 4 expansion + §7.29 SPOF closure +
operator-runbook + source-identity CI gate + two
regression-discipline rules.

This test pins the resolution document against accidental
deletion and gates the load-bearing claims so revisions
maintain audit-trail integrity.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CR_PATH = (
    REPO_ROOT
    / "docs" / "governance"
    / "PRSM-CR-2026-05-13-2.md"
)


def _read_cr() -> str:
    return CR_PATH.read_text(encoding="utf-8")


def test_cr_file_exists():
    """Pin against accidental deletion. Cross-referenced
    by audit-prep + MEMORY.md."""
    assert CR_PATH.is_file()


def test_cr_documents_authority_section():
    """Authority section names the canonical governance
    charter + voting rules + cross-CR precedent chain."""
    text = _read_cr()
    assert "## 1. Authority" in text
    assert "foundation-governance-charter" in text
    assert "PRSM-CR-2026-05-06" in text  # precedent
    assert "PRSM-CR-2026-05-08" in text  # precedent
    assert "PRSM-POL-2" in text


def test_cr_ratifies_26_sprint_cycle():
    """Sprint range covered must be unambiguous in the
    background section."""
    text = _read_cr()
    assert "355" in text
    assert "380" in text
    assert "26 sprints" in text or "26-sprint" in text


def test_cr_encodes_r_2026_05_13_1_source_identity_rule():
    """R-2026-05-13-1 source-identity CI parity rule is
    the binding council commitment behind the sprint-378
    gate. Must remain visible + load-bearing."""
    text = _read_cr()
    assert "R-2026-05-13-1" in text
    assert "source_identity_pins.json" in text
    assert "source-identity" in text.lower()
    # The enforcement mechanism path is named
    assert (
        "tests/unit/test_source_identity_parity.py"
        in text
    )


def test_cr_encodes_r_2026_05_13_2_runbook_anchor_rule():
    """R-2026-05-13-2 fleet kill-switch runbook anchor
    rule. Must remain visible + load-bearing."""
    text = _read_cr()
    assert "R-2026-05-13-2" in text
    assert "design-only as of 2026-05-13" in text
    assert "fleet-kill-switch-operator-runbook" in text
    # The enforcement mechanism path is named
    assert (
        "tests/unit/test_fleet_kill_switch_runbook.py"
        in text
    )


def test_cr_records_real_bug_discoveries():
    """The two bugs caught during the cycle (sprint 356
    _SEL_FTNS + sprint 363 self-revoke counterexample)
    must remain on the formal record."""
    text = _read_cr()
    assert "_SEL_FTNS" in text
    assert "0x9b03f021" in text
    assert "0xefa21b41" in text
    assert "self-revoke" in text or "self-revoked" in text


def test_cr_documents_explicit_non_scope():
    """§5 non-scope must enumerate what this CR does NOT
    authorize so future scope-creep cannot point at it as
    precedent."""
    text = _read_cr()
    assert "## 5. Explicit non-scope" in text
    # Key non-scope items
    assert "fleet-coordination layer" in text
    assert "Coinbase CDP" in text
    assert "Aerodrome pool-seed" in text


def test_cr_acknowledges_trigger_gated_paths():
    """Trigger-gated paths must be acknowledged as
    pre-committed but NOT pre-authorized."""
    text = _read_cr()
    assert "trigger-gated" in text
    assert "T1" in text and "T6" in text  # §7 trigger set


def test_cr_lists_all_cycle_tags():
    """The cycle's load-bearing tags must be cross-
    referenced in §6 so auditors can git-checkout the
    exact engineering substance."""
    text = _read_cr()
    cycle_tags = [
        "audit-prep-7-29-through-7-33-refresh",
        "audit-prep-7-34-formal-verification-arc",
        "source-identity-parity-gate",
        "fleet-kill-switch-operator-runbook",
        "cli-node-bootstrap-subcommand",
    ]
    for tag in cycle_tags:
        assert tag in text, (
            f"CR missing cross-reference to tag "
            f"{tag!r}. Auditor cannot git-checkout the "
            f"engineering substance without it."
        )


def test_cr_status_is_ratified():
    """Resolution must be marked RATIFIED — not draft."""
    text = _read_cr()
    assert "RATIFIED 2026-05-13" in text


def test_cr_quorum_documented():
    """Sole-founder 1-of-1 quorum must be explicitly
    named — supersession protocol relies on this anchor."""
    text = _read_cr()
    assert "sole-founder" in text.lower()
    assert "1-of-1" in text
    # 2-of-3 expansion commitment from PRSM-POL-1
    assert "2-of-3" in text
    assert "2027" in text

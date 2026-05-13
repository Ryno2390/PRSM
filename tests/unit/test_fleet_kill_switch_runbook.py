"""Sprint 379 — pin the fleet-kill-switch operator runbook.

The runbook lives at
``docs/operations/fleet-kill-switch-operator-runbook.md``
and closes the §7.21 audit-prep honest-scope item
("operator runbook not yet drafted"). This test gates
against accidental deletion + ensures the load-bearing
sections + cross-references stay present.

Anchor stability is critical because:
  - §7.21 audit-prep entry references the runbook
  - PRSM-CR-2026-05-08 Resolution 7 binds the authority
    schedule the runbook documents
  - The §7-promotion-trigger mechanism in PRSM-POL-2 reads
    the runbook for the operator-opt-in contract
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK_PATH = (
    REPO_ROOT
    / "docs" / "operations"
    / "fleet-kill-switch-operator-runbook.md"
)


def _read_runbook() -> str:
    return RUNBOOK_PATH.read_text(encoding="utf-8")


def test_runbook_file_exists():
    """Pin against accidental deletion. The §7.21 audit-
    prep entry cross-references this path."""
    assert RUNBOOK_PATH.is_file(), (
        f"Operator runbook missing: {RUNBOOK_PATH}. This "
        f"closes a §7.21 audit-prep honest-scope item; do "
        f"not delete without an explicit CR superseding it."
    )


def test_runbook_documents_all_seven_kill_switches():
    """The seven canonical per-node kill switches must
    be enumerated. Drift here breaks the audit-visible
    claim that the runbook is exhaustive for the per-node
    layer."""
    text = _read_runbook()
    canonical_env_vars = [
        "PRSM_QUERY_ORCHESTRATOR_ENABLED",
        "PRSM_AGGREGATOR_SHARE_BPS",
        "PRSM_ARBITRATION_PROPOSER_ID",
        "PRSM_DHT_ENABLED",
        "PRSM_KEY_DISTRIBUTION_ADDRESS",
        "PRSM_ONCHAIN_PROVENANCE",
        "PRSM_JOB_HISTORY_DIR",
    ]
    for env_var in canonical_env_vars:
        assert env_var in text, (
            f"Runbook missing canonical kill switch "
            f"{env_var}. Either it was renamed (update the "
            f"runbook + this pin) or accidentally dropped."
        )


def test_runbook_documents_fleet_opt_in_env_var():
    """The fleet-layer opt-in env var name is the
    operator contract; it must remain visible."""
    text = _read_runbook()
    assert "PRSM_FLEET_KILL_SWITCH_ENABLED=1" in text


def test_runbook_documents_authority_thresholds():
    """The four-tier authority schedule from
    FLEET-KILL-SWITCH-SCOPING-1 §6.1 must remain visible.
    These numbers are load-bearing — they're the same
    council ratification in PRSM-CR-2026-05-08
    Resolution 7."""
    text = _read_runbook()
    # P0 disable + deactivation: 3-of-5
    assert "3-of-5" in text
    # List-expansion: 4-of-5 + auditor
    assert "4-of-5" in text


def test_runbook_documents_appeals_process():
    """Operator dispute path must be discoverable."""
    text = _read_runbook()
    assert "Disputing a directive" in text
    assert "prsm_disclosure" in text


def test_runbook_cross_references_load_bearing_docs():
    """Cross-reference list must include the canonical
    upstream docs so auditors can follow the chain."""
    text = _read_runbook()
    expected_refs = [
        "docs/2026-05-08-fleet-kill-switch-scoping.md",
        "EXPLOIT_RESPONSE_PLAYBOOK_ANNEX",
        "PRSM-CR-2026-05-08",
        "PRSM-POL-2",
        "audit-prep",
        "§7.21",
    ]
    for ref in expected_refs:
        assert ref in text, (
            f"Runbook missing cross-reference to {ref!r}. "
            f"Audit-trail chain breaks without it."
        )


def test_runbook_documents_design_only_status():
    """The fleet-coordination LAYER (not the per-node
    layer) is design-only as of sprint 379. Runbook must
    name that explicitly so operators don't expect
    fleet-side behavior to fire."""
    text = _read_runbook()
    assert "design-only" in text.lower()


def test_runbook_quick_reference_card_present():
    """§6 quick-reference card is the print-and-pin
    artifact for incident response. Verify it's intact."""
    text = _read_runbook()
    assert "Quick Reference" in text
    # All 7 env vars appear in the card too (test §1.1 is
    # mirrored in §6 — duplication is intentional)
    assert text.count("PRSM_QUERY_ORCHESTRATOR_ENABLED") >= 2


def test_runbook_changelog_lists_sprint_379():
    """Sprint provenance must be visible. If the runbook
    is significantly revised, a new changelog row + a
    sprint number is the audit-trail expectation."""
    text = _read_runbook()
    assert "sprint" in text.lower()
    assert "379" in text


def test_runbook_does_not_claim_fleet_layer_is_live():
    """As of sprint 379, the fleet-coordination LAYER
    remains design-only. If any future revision flips to
    'shipped' without updating this pin AND a corresponding
    audit-prep §7.X entry documenting the ship, the test
    must catch it.

    This is a load-bearing claim — fleet kill-switch
    implementation is gated on §7 promotion triggers (T1-T6)
    per FLEET-KILL-SWITCH-SCOPING-1. Silent claims of
    'shipped' would mislead auditors about the actual
    fleet's coordination posture.
    """
    text = _read_runbook()
    # The runbook MUST contain the explicit "design-only as
    # of YYYY-MM-DD" anchor. Any future revision flipping
    # the layer to shipped should update this anchor + add
    # a sprint changelog row documenting the §7-trigger
    # that fired.
    assert "design-only as of 2026-05-13" in text

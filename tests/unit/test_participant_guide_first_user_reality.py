"""Sprint 424 — PARTICIPANT_GUIDE first-time-user reality pins.

After the 2026-05-14 user-perspective dogfood (see
`docs/operations/2026-05-14-user-dogfood-findings.md`),
the PARTICIPANT_GUIDE was updated to address three real
friction points a brand-new user hits:

  F1 — `prsm daemon` was deprecated → use `prsm node`
  F2 — `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` is required but
       was undocumented; first query fails with cryptic
       "Agent forge not initialized" error
  F3 — Fresh node has no content shards + zero peer
       discovery; first query fails with "no content
       shards above similarity threshold"

These pins fire if PARTICIPANT_GUIDE is rewritten in a
way that reintroduces the original friction.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs" / "PARTICIPANT_GUIDE.md"
FINDINGS = (
    REPO_ROOT / "docs" / "operations"
    / "2026-05-14-user-dogfood-findings.md"
)


def _read_guide():
    return GUIDE.read_text()


# ── F1: deprecated command guidance ──────────────────────


def test_step_3_uses_prsm_node_not_prsm_daemon():
    """The first user-facing command after install must
    use the current (non-deprecated) CLI. Using
    `prsm daemon` triggers a deprecation warning on the
    user's very first action — bad first impression."""
    text = _read_guide()
    # Step 3 block must contain the canonical start command
    assert "prsm node start" in text, (
        "PARTICIPANT_GUIDE Step 3 must use `prsm node start`, "
        "not the deprecated `prsm daemon start`"
    )


def test_deprecation_note_present():
    """The doc should acknowledge the old command works
    but is deprecated, so users with prior memory aren't
    confused when they search for it."""
    text = _read_guide()
    assert (
        "deprecated" in text.lower()
        and "prsm daemon" in text
    ), (
        "PARTICIPANT_GUIDE must note that `prsm daemon` is "
        "deprecated"
    )


# ── F2: env var requirement documented ───────────────────


def test_query_orchestrator_env_var_documented():
    """The user-facing query workflow requires
    `PRSM_QUERY_ORCHESTRATOR_ENABLED=1`. Pre-fix, this
    was nowhere in the doc and users hit a cryptic
    'Agent forge not initialized' error on their first
    query."""
    text = _read_guide()
    assert "PRSM_QUERY_ORCHESTRATOR_ENABLED" in text, (
        "PARTICIPANT_GUIDE must document the env-var "
        "requirement that gates the canonical user-query "
        "workflow"
    )


# ── F3: fresh-node honest-scope ──────────────────────────


def test_fresh_node_content_expectation_documented():
    """A brand-new node has no content shards. The doc
    must set this expectation explicitly so users aren't
    surprised when their first query fails."""
    text = _read_guide()
    assert (
        "no content shards" in text.lower()
        or "brand-new node has" in text.lower()
        or "first query" in text.lower()
    )


# ── Cross-reference to the dogfood findings doc ──────────


def test_findings_doc_exists_and_linked():
    """The full dogfood findings doc must exist and be
    cross-linked from PARTICIPANT_GUIDE for users who hit
    a friction point and want the full journey."""
    assert FINDINGS.is_file(), (
        f"dogfood findings doc missing at {FINDINGS}"
    )
    text = _read_guide()
    # Linked by filename so future doc moves surface here
    assert "2026-05-14-user-dogfood-findings.md" in text


# ── Findings doc structure pins ──────────────────────────


def test_findings_doc_documents_all_fourteen_frictions():
    """The findings doc enumerates F1-F14. F14 added
    2026-05-15 during sprint 456's multi-node test bench:
    two daemons on same host discover each other via
    bootstrap server but NAT-loopback prevents direct P2P.
    Discovery layer (sprints 319-329) verified working;
    cross-host test bench is the eventual right answer.
    If a finding is silently removed (without an explicit
    closure note), surface that."""
    text = FINDINGS.read_text()
    for marker in (
        "F1 — `prsm daemon`",
        "F2 — `PRSM_QUERY_ORCHESTRATOR_ENABLED",
        "F3 — First query fails",
        "F4 — Content upload fails",
        "F5 — Quote endpoint",
        "F6 — `/onboarding/`",
        "F7 — Locally-uploaded content not retrievable",
        "F8 — BT publisher/requester session isolation",
        "F9 — Upload + query embedding-dim mismatch",
        "F10 — Single-node forge query blocked",
        "F11 — StakingManager TypeError on claim",
        "F12 — Mock executor's ε=∞ for NONE tier",
        "F13 — `/rings/status` 500 breaks `prsm_node_status`",
        "F14 — Multi-node single-host test bench",
    ):
        assert marker in text, (
            f"dogfood finding marker missing: {marker!r}"
        )


def test_f4_closure_documented():
    """Sprint 425 closed F4 end-to-end via 4 layered fixes
    (bencodepy required dep, libtorrent system-install
    docs, content_publisher_wired field on /info,
    result.cid → result.content_id at api.py:5861). The
    closure note must stay attached to F4 so future
    readers can trace the fix arc."""
    text = FINDINGS.read_text()
    assert "Update 2026-05-14 (sprint 425)" in text
    assert "result.cid" in text  # the actual production bug
    assert "content_publisher_wired" in text


def test_findings_doc_has_positive_findings():
    """The report isn't a takedown — it must call out what
    actually worked end-to-end too. Audit-trail signal
    for honest scoping."""
    text = FINDINGS.read_text()
    assert "What's working" in text or "positive findings" in text.lower()
    for surface in (
        "Bootstrap connection",
        "/balance",
        "/info",
        "/compute/forge/quote",
    ):
        assert surface in text, (
            f"positive-findings surface missing: {surface}"
        )

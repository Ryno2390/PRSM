"""Sprint 429 — invariants for ``docs/PRSM_Testing.md``.

This is the authoritative functional-verification roadmap created
after the 2026-05-14 user-perspective dogfood arc. It enumerates
every shipped piece of functionality with a verification status and
drives the systematic test campaign that promotes 🟢 (test-pinned)
features to ✅ (live-verified).

These pins fire when the doc gets edited in a way that:
- Removes the status legend (loses meaning of the symbols)
- Drops one of the canonical Vision §-section headings
- Promotes a row to ✅ without a sprint number in the Sprint column
  (the dogfood arc proved that "tests pass" ≠ "feature works", so
  every ✅ MUST cite the sprint that did the live verification)

They are NOT correctness tests on the features themselves — they
defend the integrity of the audit trail this doc provides.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = REPO_ROOT / "docs" / "PRSM_Testing.md"


def _read():
    return ROADMAP.read_text()


def test_roadmap_exists():
    """The roadmap itself must exist. Removing it loses the
    one-stop verification-status surface."""
    assert ROADMAP.is_file(), (
        f"PRSM_Testing.md missing at {ROADMAP}"
    )


def test_status_legend_present():
    """The status legend defines what every ✅ / 🟢 / ⚠️ / 🔬
    / ⏸️ / ❌ / 🔗 means. Without it, the verification status
    column is opaque to a new reader."""
    text = _read()
    for symbol in ("✅", "🟢", "⚠️", "🔬", "⏸️", "❌", "🔗"):
        assert symbol in text, (
            f"status legend missing symbol: {symbol!r}"
        )
    assert "Verified end-to-end" in text
    assert "Test-pinned" in text


def test_vision_section_headings_present():
    """The doc is organized by Vision §-section per the
    structural decision in sprint 429. If a top-level section
    disappears, surface that — it means the doc has drifted
    from the Vision structure."""
    text = _read()
    for heading in (
        "## §4 — End-to-end user workflow",
        "## §5.1 — Data Layer",
        "## §5.2 — Compute Layer",
        "## §5.3 — Economic Layer",
        "## §5.4 — Provenance Layer",
        "## §7 — Private Inference",
        "## §13 — Operator surfaces",
        "## §14 — Risk mitigations",
    ):
        assert heading in text, (
            f"vision §-section heading missing: {heading!r}"
        )


def test_dogfood_arc_context_documented():
    """The doc's "Why this doc exists" block must call out the
    F4/F7/F8 dogfood lessons. Without that motivation, future
    editors might assume ✅ === green CI (the exact mistake
    this doc was designed to prevent)."""
    text = _read()
    assert "F4" in text and "F7" in text and "F8" in text, (
        "doc must cite the F4/F7/F8 dogfood lessons in its "
        "rationale block"
    )
    assert "result.cid" in text or "fixture" in text.lower()


def test_verification_campaign_priorities_present():
    """The "Verification campaign priorities" section is the
    actionable handle on the doc — it's what tells the
    autonomous loop which sprint to pick next."""
    text = _read()
    assert "Verification campaign priorities" in text
    # Top-priority items must mention the actual surfaces.
    assert "Tier B/C" in text
    assert "PRSM_QUERY_ORCHESTRATOR_ENABLED" in text


def test_every_verified_row_cites_attribution():
    """The dogfood arc lesson: a ✅ is only a ✅ if it's been
    live-tested. Each ✅ row must cite WHO / WHEN — either
    a sprint number in the Sprint column (for code surfaces)
    OR a contract reference / phase tag in the Contract
    column (for mainnet-deploy rows in §5.4 which have a
    different table shape).

    Concretely: at least ONE of the cells flanking the ✅
    must be non-empty.

    Catches drift like: someone flips a row from 🟢 to ✅
    without attributing the verification to anything."""
    text = _read()
    lines = [l for l in text.splitlines() if l.startswith("| ")]
    # Row split: cells around ✅ — capture the cell BEFORE
    # the ✅-only cell and the cell AFTER it.
    pat = re.compile(
        r"\|\s*([^|]*?)\s*\|\s*✅\s*\|\s*([^|]*?)\s*\|"
    )
    seen_verified = 0
    for line in lines:
        # Header rows have "Status" — skip.
        if "Status" in line and "|" in line and "---" not in line:
            continue
        if "✅" not in line:
            continue
        m = pat.search(line)
        if m is None:
            # ✅ may appear in non-tabular contexts (legend,
            # prose) — those don't need attribution.
            continue
        before, after = m.group(1).strip(), m.group(2).strip()
        assert before or after, (
            f"✅ row missing attribution on EITHER side: "
            f"{line[:120]}"
        )
        seen_verified += 1
    assert seen_verified > 0, (
        "no ✅ rows found in roadmap — did the table format "
        "change?"
    )


def test_changelog_present():
    """The changelog records when each sprint touches the doc.
    Removing it loses the audit trail."""
    text = _read()
    assert "## Changelog" in text
    assert "sprint 429" in text
    assert "Initial draft" in text


def test_dogfood_arc_canonical_user_workflow_first_section():
    """Vision §4 (canonical user workflow) MUST be the first
    feature-mapping section. Demoting it would lose the
    "user-perspective" lens the dogfood arc proved valuable."""
    text = _read()
    # §4 must appear BEFORE §5.1 in the doc.
    idx_4 = text.find("## §4 — End-to-end user workflow")
    idx_5_1 = text.find("## §5.1 — Data Layer")
    assert idx_4 > 0 and idx_5_1 > 0
    assert idx_4 < idx_5_1, (
        "§4 (canonical user workflow) must come before §5.1 "
        "in the roadmap — user lens first, then subsystem map"
    )

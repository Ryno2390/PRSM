"""Sprint 486 — Test coverage matrix integrity pins.

User challenged sprint 485's "thoroughly tested" framing
("Have we really thoroughly tested ALL of PRSM's
functionality? I doubt it"). Sprint 486 added a depth-
dimension matrix to PRSM_Testing.md that maps surface area
× testing dimension (HP/EP/CC/SC/AD/FM/LR/DR/OC/XF).

These pins defend the matrix's integrity. They are NOT
correctness tests on the features — they ensure the
matrix doesn't decay silently:

  1. The depth-dimensions legend stays defined (callers
     reading a ✅/⚠️/❌ cell need to know what dimension
     they're rating).
  2. The matrix header row stays in sync with the legend
     (a column reordering shouldn't silently drop a
     dimension).
  3. The "highest-priority untested cells" ranking section
     stays present (the matrix is a risk map; without the
     priority section it's just a wall of red).
  4. The canonical 10 dimensions stay enumerated.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = REPO_ROOT / "docs" / "PRSM_Testing.md"


def _read():
    return ROADMAP.read_text()


def test_coverage_matrix_section_exists():
    """The top-level section heading must remain — removing
    it loses the depth-of-testing surface the user
    explicitly asked for."""
    text = _read()
    assert "## Test coverage matrix — depth dimensions" in text


def test_all_ten_dimensions_defined():
    """Each of the 10 canonical depth dimensions must
    appear in the matrix legend with its abbreviation +
    name + definition row."""
    text = _read()
    # Find the depth-dimensions table.
    idx = text.find("### Depth dimensions")
    assert idx >= 0, "depth-dimensions legend missing"
    end = text.find("### Coverage matrix", idx)
    legend = text[idx:end]
    for abbrev, name in (
        ("**HP**", "Happy path"),
        ("**EP**", "Error path"),
        ("**CC**", "Concurrency"),
        ("**SC**", "Scale"),
        ("**AD**", "Adversarial"),
        ("**FM**", "Failure modes"),
        ("**LR**", "Long-running"),
        ("**DR**", "Disaster recovery"),
        ("**OC**", "Real on-chain"),
        ("**XF**", "Cross-feature"),
    ):
        assert abbrev in legend, (
            f"depth dimension abbreviation missing: {abbrev}"
        )
        assert name in legend, (
            f"depth dimension name missing: {name}"
        )


def test_coverage_matrix_header_row_present():
    """The matrix header must list every dimension column
    — drift means rows can't be parsed by ops tooling."""
    text = _read()
    # The header row.
    header = "| HP | EP | CC | SC | AD | FM | LR | DR | OC | XF |"
    assert header in text, (
        "coverage matrix header row drifted from canonical "
        "10-column form"
    )


def test_highest_priority_untested_cells_section_present():
    """The matrix without the priority-ranking section is
    just a wall of red. The ranking is the actionable
    handle — keep it."""
    text = _read()
    assert "### Highest-priority untested cells" in text
    # The 8 priority items must remain enumerated.
    for marker in (
        "CC — Concurrency",
        "SC — Scale",
        "AD — Adversarial",
        "FM — Failure modes",
        "LR — Long-running",
        "OC — Real on-chain",
        "XF — Cross-feature",
        "§11/§12 Governance",
    ):
        assert marker in text, (
            f"highest-priority cell missing: {marker!r}"
        )


def test_matrix_is_risk_map_not_punch_list():
    """The matrix doc explicitly states it's a risk map,
    not a punch list to clear top-to-bottom. This framing
    matters: future sprints should pick cells by
    production-risk per unit-effort, not by completeness.
    Without this caveat, the matrix becomes a checkbox
    farm."""
    text = _read()
    assert "risk map" in text.lower()
    assert "not a punch list" in text.lower() or (
        "NOT a punch list" in text or "not a punch" in text.lower()
    )


def test_matrix_acknowledges_zero_concurrency_coverage():
    """The CC column should be empty (❌) across nearly all
    subjects — that's the load-bearing finding. If a future
    edit silently fills the CC column without a sprint
    citation, surface it."""
    text = _read()
    # Find the coverage matrix region.
    idx = text.find("### Coverage matrix")
    assert idx >= 0
    end = text.find("### Highest-priority", idx)
    matrix = text[idx:end]
    # Count CC cells with ✅ — should be very few (currently
    # 0 active surfaces; only N/A allowed for
    # surfaces where concurrency is structurally
    # inapplicable). If someone flips many to ✅ without
    # actually testing concurrency, this surfaces it.
    rows = [r for r in matrix.splitlines() if r.startswith("| **")]
    # CC is the 3rd data column (after the subject column).
    # Row split: cells[0] is the empty pre-pipe string,
    # cells[1] is the subject column, cells[2] is HP,
    # cells[3] is EP, cells[4] is CC.
    cc_verified_count = 0
    for row in rows:
        cells = [c.strip() for c in row.split("|")]
        if len(cells) >= 6:
            cc_cell = cells[4]  # CC column
            if "✅" in cc_cell:
                cc_verified_count += 1
    # As of sprint 486, ZERO subjects have CC ✅. The pin
    # allows up to 3 (room for the verification campaign to
    # close concurrency for a few critical paths) but
    # blocks a silent flip of the whole column.
    assert cc_verified_count <= 3, (
        f"concurrency column ✅ count = {cc_verified_count}; "
        f"this load-bearing column shouldn't fill up "
        f"without dedicated sprints. If you ran real "
        f"concurrent-caller tests, bump this threshold; "
        f"otherwise audit the recent edits"
    )

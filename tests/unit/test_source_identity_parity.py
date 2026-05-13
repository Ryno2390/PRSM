"""Sprint 378 — source-identity CI parity check.

Halmos symbolic specs cite canonical source line ranges in
their STRUCTURAL EQUIVALENCE blocks (audit-visible source-
identity claim). When canonical source mutates without a
coordinated spec update, the symbolic proof silently drifts
out of sync with what it claims to mirror.

This test gates that drift: every citation in every spec
must hash to the value in source_identity_pins.json. When
canonical source legitimately changes, the engineer also
updates the spec to match, then re-runs
scripts/update_source_identity_pins.py to refresh the pin
file. CI re-greens.

Closes the §7.34 honest-scope item: "Source-identity CI
parity check between spec contracts + canonical source —
would catch silent drift if someone modifies one without
the other."
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prsm.economy.web3.source_identity import (
    Citation,
    ParityResult,
    hash_canonical_range,
    load_pins,
    parse_citations,
    save_pins,
    scan_specs_dir,
    verify_parity,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS_DIR = REPO_ROOT / "contracts" / "symbolic-proofs" / "test"
PINS_PATH = (
    REPO_ROOT
    / "contracts" / "symbolic-proofs"
    / "source_identity_pins.json"
)


# ── Parser ───────────────────────────────────────────


def test_parse_citations_extracts_python_path():
    text = """
/// Source-identity-mirrors prsm/compute/chain_rpc/client.py:1431
"""
    cits = parse_citations(text)
    assert len(cits) == 1
    assert (
        cits[0].canonical_path
        == "prsm/compute/chain_rpc/client.py"
    )
    assert cits[0].line_start == 1431
    assert cits[0].line_end == 1431


def test_parse_citations_extracts_range():
    text = "see prsm/foo/bar.py:100-200 for details"
    cits = parse_citations(text)
    assert len(cits) == 1
    assert cits[0].line_start == 100
    assert cits[0].line_end == 200


def test_parse_citations_extracts_solidity_path():
    text = "/// contracts/contracts/Foo.sol:10-20"
    cits = parse_citations(text)
    assert len(cits) == 1
    assert (
        cits[0].canonical_path
        == "contracts/contracts/Foo.sol"
    )


def test_parse_citations_ignores_unrooted_paths():
    """Citations MUST start with prsm/ or contracts/.
    Bare names like RoyaltyDistributor.sol:111-155 (without
    the contracts/contracts/ prefix) are ambiguous — the
    parser skips them to avoid silently miss-matching."""
    text = "/// see RoyaltyDistributor.sol:111-155"
    cits = parse_citations(text)
    assert cits == []


def test_parse_citations_dedups_within_file():
    text = (
        "prsm/x.py:10 here, prsm/x.py:10 again, "
        "prsm/x.py:20 different"
    )
    cits = parse_citations(text)
    assert len(cits) == 2
    keys = {c.key for c in cits}
    assert keys == {"prsm/x.py:10", "prsm/x.py:20"}


def test_parse_citations_handles_no_matches():
    assert parse_citations("nothing here") == []
    assert parse_citations("") == []


# ── Citation key ─────────────────────────────────────


def test_citation_key_single_line():
    c = Citation(
        canonical_path="prsm/x.py",
        line_start=10,
        line_end=10,
    )
    assert c.key == "prsm/x.py:10"


def test_citation_key_range():
    c = Citation(
        canonical_path="prsm/x.py",
        line_start=10,
        line_end=20,
    )
    assert c.key == "prsm/x.py:10-20"


# ── Hashing ──────────────────────────────────────────


def test_hash_canonical_range_returns_hex_digest(tmp_path):
    src = tmp_path / "prsm" / "x.py"
    src.parent.mkdir(parents=True)
    src.write_text("line1\nline2\nline3\nline4\n")
    cit = Citation(
        canonical_path="prsm/x.py",
        line_start=2,
        line_end=3,
    )
    h = hash_canonical_range(cit, repo_root=tmp_path)
    # SHA-256 hex = 64 chars
    assert h is not None
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_canonical_range_stable_under_trailing_whitespace(
    tmp_path,
):
    """Trailing whitespace per line is stripped — a benign
    end-of-line edit must NOT trip the parity gate."""
    src = tmp_path / "prsm" / "x.py"
    src.parent.mkdir(parents=True)
    src.write_text("line1\nline2\n")
    cit = Citation(
        canonical_path="prsm/x.py",
        line_start=1,
        line_end=2,
    )
    h1 = hash_canonical_range(cit, repo_root=tmp_path)
    # Add trailing space — hash should be unchanged
    src.write_text("line1   \nline2\t\t\n")
    h2 = hash_canonical_range(cit, repo_root=tmp_path)
    assert h1 == h2


def test_hash_canonical_range_changes_on_content_edit(
    tmp_path,
):
    """Real content change → hash changes. The whole
    point."""
    src = tmp_path / "prsm" / "x.py"
    src.parent.mkdir(parents=True)
    src.write_text("def foo():\n    pass\n")
    cit = Citation(
        canonical_path="prsm/x.py",
        line_start=1,
        line_end=2,
    )
    h1 = hash_canonical_range(cit, repo_root=tmp_path)
    src.write_text("def foo():\n    return 42\n")
    h2 = hash_canonical_range(cit, repo_root=tmp_path)
    assert h1 != h2


def test_hash_canonical_range_returns_none_when_file_missing(
    tmp_path,
):
    cit = Citation(
        canonical_path="prsm/nonexistent.py",
        line_start=1,
        line_end=1,
    )
    assert hash_canonical_range(
        cit, repo_root=tmp_path,
    ) is None


def test_hash_canonical_range_returns_none_when_out_of_range(
    tmp_path,
):
    src = tmp_path / "prsm" / "x.py"
    src.parent.mkdir(parents=True)
    src.write_text("only-one-line\n")
    cit = Citation(
        canonical_path="prsm/x.py",
        line_start=1,
        line_end=100,
    )
    assert hash_canonical_range(
        cit, repo_root=tmp_path,
    ) is None


# ── Pin registry I/O ─────────────────────────────────


def test_load_pins_returns_dict(tmp_path):
    p = tmp_path / "pins.json"
    p.write_text('{"version": 1, "pins": {"a:1": "hexx"}}')
    pins = load_pins(p)
    assert pins == {"a:1": "hexx"}


def test_load_pins_empty_when_file_missing(tmp_path):
    p = tmp_path / "missing.json"
    assert load_pins(p) == {}


def test_load_pins_empty_when_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    assert load_pins(p) == {}


def test_save_pins_sorted_for_diff_friendliness(tmp_path):
    p = tmp_path / "pins.json"
    save_pins({"z:1": "hh1", "a:1": "hh2"}, p)
    text = p.read_text()
    # `a:1` should appear before `z:1`
    assert text.index('"a:1"') < text.index('"z:1"')


def test_save_pins_round_trips(tmp_path):
    p = tmp_path / "pins.json"
    expected = {"prsm/x.py:1": "abc", "prsm/y.sol:5-10": "def"}
    save_pins(expected, p)
    assert load_pins(p) == expected


# ── Directory scan ───────────────────────────────────


def test_scan_specs_dir_returns_sorted(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    (d / "a.t.sol").write_text("/// prsm/z.py:10-20")
    (d / "b.t.sol").write_text("/// prsm/a.py:5")
    cits = scan_specs_dir(d)
    keys = [c.key for c in cits]
    # Sorted alphabetically
    assert keys == ["prsm/a.py:5", "prsm/z.py:10-20"]


def test_scan_specs_dir_dedups_across_files(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    (d / "a.t.sol").write_text("/// prsm/x.py:10")
    (d / "b.t.sol").write_text("/// prsm/x.py:10 again")
    cits = scan_specs_dir(d)
    assert len(cits) == 1


# ── End-to-end verify ────────────────────────────────


def test_verify_parity_passes_on_matching_pins(tmp_path):
    """Set up a tiny tree: spec cites a source file, pin
    matches the current hash → ok=True."""
    src = tmp_path / "prsm" / "x.py"
    src.parent.mkdir(parents=True)
    src.write_text("line1\nline2\nline3\n")
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "x.t.sol").write_text("/// prsm/x.py:1-3")
    cit = Citation("prsm/x.py", 1, 3)
    actual = hash_canonical_range(cit, repo_root=tmp_path)
    pins = tmp_path / "pins.json"
    save_pins({cit.key: actual}, pins)

    result = verify_parity(
        specs_dir=specs,
        pins_path=pins,
        repo_root=tmp_path,
    )
    assert result.ok is True
    assert cit.key in result.passed


def test_verify_parity_detects_drift(tmp_path):
    """Modify the canonical source after pin was set →
    drifted entry, ok=False."""
    src = tmp_path / "prsm" / "x.py"
    src.parent.mkdir(parents=True)
    src.write_text("original\n")
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "x.t.sol").write_text("/// prsm/x.py:1")
    cit = Citation("prsm/x.py", 1, 1)
    h_orig = hash_canonical_range(cit, repo_root=tmp_path)
    pins = tmp_path / "pins.json"
    save_pins({cit.key: h_orig}, pins)
    # Drift: change the source
    src.write_text("MUTATED\n")

    result = verify_parity(
        specs_dir=specs,
        pins_path=pins,
        repo_root=tmp_path,
    )
    assert result.ok is False
    assert len(result.drifted) == 1
    assert result.drifted[0][0] == cit.key


def test_verify_parity_detects_missing_source(tmp_path):
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "x.t.sol").write_text(
        "/// prsm/nonexistent.py:1-10"
    )
    pins = tmp_path / "pins.json"
    save_pins({}, pins)

    result = verify_parity(
        specs_dir=specs,
        pins_path=pins,
        repo_root=tmp_path,
    )
    assert result.ok is False
    assert "prsm/nonexistent.py:1-10" in result.missing_source


def test_verify_parity_detects_missing_pin(tmp_path):
    src = tmp_path / "prsm" / "x.py"
    src.parent.mkdir(parents=True)
    src.write_text("line1\n")
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "x.t.sol").write_text("/// prsm/x.py:1")
    pins = tmp_path / "pins.json"
    save_pins({}, pins)  # No pin for this citation

    result = verify_parity(
        specs_dir=specs,
        pins_path=pins,
        repo_root=tmp_path,
    )
    assert result.ok is False
    assert "prsm/x.py:1" in result.missing_pin


def test_verify_parity_detects_out_of_range(tmp_path):
    src = tmp_path / "prsm" / "x.py"
    src.parent.mkdir(parents=True)
    src.write_text("only line\n")
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "x.t.sol").write_text("/// prsm/x.py:1-100")
    pins = tmp_path / "pins.json"
    save_pins({}, pins)

    result = verify_parity(
        specs_dir=specs,
        pins_path=pins,
        repo_root=tmp_path,
    )
    assert result.ok is False
    assert len(result.out_of_range) == 1
    assert result.out_of_range[0][0] == "prsm/x.py:1-100"


# ── Live CI gate ─────────────────────────────────────


def test_live_source_identity_parity():
    """THE CI gate. Every citation in every shipped spec
    file must hash to the pinned value. If this fails:

      1. If the spec change was intentional + the canonical
         source change is intentional → regenerate pins via
         `scripts/update_source_identity_pins.py`.
      2. If you weren't expecting drift → READ THE DIFFERENCE
         carefully. Either the canonical source mutated
         without a spec update (the bug class this gate
         catches), or the spec was edited without checking
         the canonical mirror.

    This is the §7.34 honest-scope closer — silent drift
    between halmos specs + canonical source is now
    detectable in CI.
    """
    result = verify_parity(
        specs_dir=SPECS_DIR,
        pins_path=PINS_PATH,
        repo_root=REPO_ROOT,
    )
    assert result.ok, (
        f"Source-identity parity check failed.\n"
        f"  {result.summary()}\n"
        f"  drifted: {result.drifted}\n"
        f"  missing_source: {result.missing_source}\n"
        f"  missing_pin: {result.missing_pin}\n"
        f"  out_of_range: {result.out_of_range}\n"
        f"\n"
        f"To regenerate pins (after VERIFYING the spec "
        f"still mirrors the canonical source semantically), "
        f"run: python scripts/update_source_identity_pins.py"
    )
    # At least 1 citation pinned — guards against the
    # registry going empty silently.
    assert len(result.passed) >= 1

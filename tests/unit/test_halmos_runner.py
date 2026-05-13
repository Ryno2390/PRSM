"""Sprint 360 — halmos runner tests.

The runner is the bridge between Python operator tooling
and halmos symbolic execution. Tests cover:
  - fail-soft when halmos / forge isn't installed
  - parsing of canonical halmos output formats
    (PASS / FAIL / mixed)
  - suite-status aggregation
  - timeout + invocation-error handling
  - the symbolic-proof catalog
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.economy.web3.halmos_runner import (
    HalmosRunner,
    SYMBOLIC_PROOF_CATALOG,
    SymbolicProofResult,
    SymbolicProofStatus,
    SymbolicProofSuite,
    _parse_halmos_output,
)


# ── status + catalog ─────────────────────────────────


def test_status_enum_values():
    assert SymbolicProofStatus.PASSED.value == "passed"
    assert SymbolicProofStatus.FAILED.value == "failed"
    assert SymbolicProofStatus.SKIPPED.value == "skipped"
    assert SymbolicProofStatus.ERROR.value == "error"


def test_catalog_has_ftns_supply_cap_proof():
    """Sprint 360 ships FTNSSupplyCapSpec; pin the entry
    so accidental removal trips CI."""
    assert "FTNSSupplyCapSpec" in SYMBOLIC_PROOF_CATALOG
    entry = SYMBOLIC_PROOF_CATALOG["FTNSSupplyCapSpec"]
    assert entry["mirrors_runtime_contract"] == "ftns_token"
    assert "INV-FT-1" in entry["runtime_invariants"]
    assert "INV-FT-2" in entry["runtime_invariants"]


def test_catalog_has_royalty_distributor_solvency_proof():
    """Sprint 361 ships RoyaltyDistributorSolvencySpec —
    the canonical solvency proof across distributeRoyalty +
    claim + recoverStranded. Mirrors INV-RD-1 + INV-RD-4."""
    assert (
        "RoyaltyDistributorSolvencySpec"
        in SYMBOLIC_PROOF_CATALOG
    )
    entry = SYMBOLIC_PROOF_CATALOG[
        "RoyaltyDistributorSolvencySpec"
    ]
    assert (
        entry["mirrors_runtime_contract"]
        == "royalty_distributor"
    )
    # INV-RD-4 is THE solvency invariant per the existing
    # registry; INV-RD-1 covers the NETWORK_FEE_BPS pin
    # that the symbolic spec also verifies.
    assert "INV-RD-4" in entry["runtime_invariants"]
    assert "INV-RD-1" in entry["runtime_invariants"]


# ── parser ───────────────────────────────────────────


def test_parser_handles_all_pass():
    output = """
Running 3 tests for test/FTNSTokenSupplyCap.t.sol:FTNSSupplyCapSpec
[PASS] check_max_supply_constant_value() (paths: 1, time: 0.00s, bounds: [])
[PASS] check_mint_preserves_cap(address,uint256) (paths: 5, time: 0.02s, bounds: [])
[PASS] check_post_construction_in_cap() (paths: 1, time: 0.00s, bounds: [])
Symbolic test result: 3 passed; 0 failed; time: 0.09s
""".strip()
    suite = _parse_halmos_output(
        "FTNSSupplyCapSpec", output,
    )
    assert suite.status == SymbolicProofStatus.PASSED
    assert len(suite.proofs) == 3
    assert all(
        p.status == SymbolicProofStatus.PASSED
        for p in suite.proofs
    )
    # paths_explored captured for the symbolic test
    mint_proof = next(
        p for p in suite.proofs
        if p.name.startswith("check_mint_preserves_cap")
    )
    assert mint_proof.paths_explored == 5


def test_parser_handles_mixed_pass_fail():
    output = """
Running 2 tests
[PASS] check_pass() (paths: 1, time: 0.00s, bounds: [])
[FAIL] check_fail() (paths: 3, time: 0.01s, bounds: [])
""".strip()
    suite = _parse_halmos_output("X", output)
    assert suite.status == SymbolicProofStatus.FAILED
    assert len(suite.proofs) == 2
    statuses = {p.status for p in suite.proofs}
    assert SymbolicProofStatus.PASSED in statuses
    assert SymbolicProofStatus.FAILED in statuses


def test_parser_strips_ansi_color_codes():
    """Halmos emits ANSI color codes around PASS/FAIL.
    Parser must strip them before regex-matching."""
    output = (
        "\x1b[32m[PASS]\x1b[0m check_x() (paths: 1, time: 0.0s, bounds: [])"
    )
    suite = _parse_halmos_output("X", output)
    assert suite.status == SymbolicProofStatus.PASSED
    assert len(suite.proofs) == 1


def test_parser_extracts_halmos_version():
    output = """
halmos 0.3.3
Running 1 tests
[PASS] check_x() (paths: 1, time: 0.00s, bounds: [])
""".strip()
    suite = _parse_halmos_output("X", output)
    assert suite.halmos_version == "0.3.3"


def test_parser_errors_on_empty_output():
    """If halmos produced no parseable lines (e.g., build
    failure), suite reports ERROR — not silently PASSED."""
    suite = _parse_halmos_output("X", "")
    assert suite.status == SymbolicProofStatus.ERROR
    assert "no parseable" in (suite.error or "")


def test_parser_handles_zero_paths():
    """Trivial constant-checks have paths=0 or 1; ensure
    parser tolerates."""
    output = (
        "[PASS] check_x() (paths: 0, time: 0.00s, bounds: [])"
    )
    suite = _parse_halmos_output("X", output)
    assert suite.proofs[0].paths_explored == 0


# ── runner fail-soft behavior ────────────────────────


def test_runner_skipped_when_halmos_missing(tmp_path):
    """No halmos in PATH → SKIPPED with named tool, no
    raise."""
    runner = HalmosRunner(
        halmos_bin=None, forge_bin="/usr/bin/forge",
    )
    with patch(
        "shutil.which",
        side_effect=lambda x: (
            "/usr/bin/forge" if x == "forge" else None
        ),
    ):
        suite = runner.run("FTNSSupplyCapSpec")
    assert suite.status == SymbolicProofStatus.SKIPPED
    assert "halmos" in (suite.error or "")


def test_runner_skipped_when_forge_missing():
    runner = HalmosRunner(
        halmos_bin="/usr/bin/halmos", forge_bin=None,
    )
    with patch(
        "shutil.which",
        side_effect=lambda x: (
            "/usr/bin/halmos" if x == "halmos" else None
        ),
    ):
        suite = runner.run("X")
    assert suite.status == SymbolicProofStatus.SKIPPED
    assert "forge" in (suite.error or "")


def test_runner_skipped_when_both_missing():
    runner = HalmosRunner()
    with patch("shutil.which", return_value=None):
        suite = runner.run("X")
    assert suite.status == SymbolicProofStatus.SKIPPED
    assert "halmos" in (suite.error or "")
    assert "forge" in (suite.error or "")


def test_runner_skipped_when_proofs_dir_missing():
    """Even with both tools installed, missing proofs dir
    SKIPS rather than crashes (e.g., partial repo clone)."""
    runner = HalmosRunner(
        proofs_dir="/tmp/__nonexistent_halmos_dir_xyz__",
        halmos_bin="/usr/bin/halmos",
        forge_bin="/usr/bin/forge",
    )
    with patch(
        "shutil.which",
        side_effect=lambda x: (
            "/usr/bin/halmos" if x == "halmos"
            else "/usr/bin/forge"
        ),
    ):
        suite = runner.run("X")
    assert suite.status == SymbolicProofStatus.SKIPPED
    assert "symbolic-proofs dir" in (suite.error or "")


def test_runner_is_available_checks_both_tools():
    runner = HalmosRunner()
    with patch("shutil.which", return_value=None):
        assert runner.is_available() is False
    with patch("shutil.which", return_value="/usr/bin/halmos"):
        # Both lookups return same value when which is mocked
        # with a string return; that satisfies both halmos +
        # forge.
        assert runner.is_available() is True


def test_runner_missing_tools_reports_both():
    runner = HalmosRunner()
    with patch("shutil.which", return_value=None):
        missing = runner.missing_tools()
    assert "halmos" in missing
    assert "forge" in missing


# ── suite serialization ──────────────────────────────


def test_suite_to_dict_includes_summary():
    suite = SymbolicProofSuite(
        contract="X",
        status=SymbolicProofStatus.PASSED,
        proofs=[
            SymbolicProofResult(
                name="check_a()",
                status=SymbolicProofStatus.PASSED,
                paths_explored=1,
            ),
            SymbolicProofResult(
                name="check_b()",
                status=SymbolicProofStatus.FAILED,
                paths_explored=3,
            ),
            SymbolicProofResult(
                name="check_c()",
                status=SymbolicProofStatus.ERROR,
            ),
        ],
    )
    d = suite.to_dict()
    assert d["summary"]["passed"] == 1
    assert d["summary"]["failed"] == 1
    assert d["summary"]["errored"] == 1
    assert len(d["proofs"]) == 3


def test_proof_result_to_dict_serializable():
    r = SymbolicProofResult(
        name="check_x()",
        status=SymbolicProofStatus.PASSED,
        paths_explored=5,
        time_seconds=0.02,
    )
    d = r.to_dict()
    assert d["name"] == "check_x()"
    assert d["status"] == "passed"
    assert d["paths_explored"] == 5
    assert d["time_seconds"] == 0.02


# ── subprocess timeout handling ──────────────────────


# NOTE: a live end-to-end test that actually invokes
# halmos against the FTNSSupplyCapSpec proof was prototyped
# but conflicts with tests/conftest.py:602 session-scoped
# autouse fixture that mocks subprocess.run globally for
# external-connection safety in CI. The 17 mocked-output
# tests above cover the parser + fail-soft surface; the
# Solidity spec itself is the formal proof and is verified
# manually via:
#   PATH="$PWD/.venv/bin:$HOME/.foundry/bin:$PATH" \
#   cd contracts/symbolic-proofs && \
#   halmos --contract FTNSSupplyCapSpec
# Expected output: "Symbolic test result: 3 passed; 0 failed"
# Future sprint may carve a tools-required marker that
# overrides the autouse subprocess mock for a subset of
# integration tests; out of scope for sprint 360.


def test_runner_handles_timeout(tmp_path):
    """Long-running halmos times out gracefully."""
    proofs_dir = tmp_path / "symbolic-proofs"
    proofs_dir.mkdir()
    runner = HalmosRunner(
        proofs_dir=str(proofs_dir),
        timeout_seconds=1,
        halmos_bin="/usr/bin/halmos",
        forge_bin="/usr/bin/forge",
    )
    import subprocess
    with patch(
        "shutil.which",
        side_effect=lambda x: (
            "/usr/bin/halmos" if x == "halmos"
            else "/usr/bin/forge"
        ),
    ), patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(
            cmd="halmos", timeout=1,
        ),
    ):
        suite = runner.run("X")
    assert suite.status == SymbolicProofStatus.ERROR
    assert "timed out" in (suite.error or "")

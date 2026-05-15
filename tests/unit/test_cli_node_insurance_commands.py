"""Sprint 435 — `prsm node insurance ...` CLI trifecta gap closure.

The /admin/insurance-fund/* REST surface + `prsm_insurance_fund`
MCP tool existed; the CLI lane was the gap per PRSM_Testing.md
§13 "Operator-trifecta gaps". This sprint adds:

- `prsm node insurance status` — read-only fund status
- `prsm node insurance compose-recovery` — produces a multi-sig-
  uploadable recovery tx payload (does NOT execute — Foundation
  Safe holds the transfer privilege per Vision §14)

These pins fire if the commands are silently removed, if
compose-recovery's required-args are dropped (would let
operators ship malformed tx payloads), or if the default JSON
output for compose-recovery shifts to text (would break ops
pipelines piping into safe-cli).
"""
from __future__ import annotations

from click.testing import CliRunner

from prsm.cli import main


def test_node_insurance_group_registered():
    """`prsm node insurance` must exist as a click subcommand
    group, with status + compose-recovery as subcommands."""
    runner = CliRunner()
    result = runner.invoke(main, ["node", "insurance", "--help"])
    assert result.exit_code == 0
    assert "insurance" in result.output.lower()
    assert "status" in result.output
    assert "compose-recovery" in result.output


def test_node_insurance_status_help():
    """`prsm node insurance status` must support --format
    for ops automation."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "insurance", "status", "--help"],
    )
    assert result.exit_code == 0
    assert "--format" in result.output


def test_node_insurance_compose_recovery_help():
    """All three required args (recipient / amount-wei /
    reason) must be advertised as required in --help. Missing
    any of them would produce a malformed tx payload that
    could confuse the multi-sig signer."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "insurance", "compose-recovery", "--help"],
    )
    assert result.exit_code == 0
    assert "--recipient" in result.output
    assert "--amount-wei" in result.output
    assert "--reason" in result.output
    # All three should be marked [required]
    assert result.output.count("[required]") >= 3


def test_compose_recovery_default_format_is_json():
    """compose-recovery defaults to JSON output so operators
    can pipe directly into safe-cli or similar multi-sig
    tools. Text default would break ops pipelines silently."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "insurance", "compose-recovery", "--help"],
    )
    assert result.exit_code == 0
    # Click shows the default in the help line
    assert "default: json" in result.output.lower()


def test_compose_recovery_requires_all_three_args():
    """Click should reject invocation with missing required
    args. Defense against operators omitting --reason and
    shipping an unaudited recovery."""
    runner = CliRunner()
    # Missing --reason
    result = runner.invoke(
        main, [
            "node", "insurance", "compose-recovery",
            "--recipient", "0x" + "00" * 20,
            "--amount-wei", "1",
        ],
    )
    assert result.exit_code != 0
    assert "reason" in result.output.lower()


def test_invariant_compose_does_not_execute():
    """The CLI MUST NOT execute the recovery transfer
    directly — it only composes the multi-sig payload. The
    help text must reflect this so operators reading --help
    know they need to upload to a multi-sig signer.

    This is Vision §14: PRSM never executes; Foundation Safe
    holds the privilege. The CLI is just a composer."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "insurance", "compose-recovery", "--help"],
    )
    text_lower = result.output.lower()
    # Either "does not execute" or "multi-sig must sign"
    assert "not execute" in text_lower or "multi-sig" in text_lower

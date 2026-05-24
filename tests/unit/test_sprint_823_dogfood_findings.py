"""Sprint 823 — dogfood findings + fixes.

Live operator-path walkthrough surfaced one UX bug:

  FINDING #2 (sprint 797's --write flag): when --write or
  --write-path successfully wrote the delegation file, the
  text-mode tail message still told the operator to "Save the
  JSON above as e.g. ~/.prsm/operator_delegation.json" + go
  through the manual systemd export dance. Confusing —
  operator already had the file written + just needs to
  restart the daemon (sprint 797's _merge_operator_delegation
  auto-loads from the default path).

Sprint 823 fix: branch on `written_path is not None` and emit
a different message:
  "Delegation written to <path>
   Next: restart the daemon — sprint-797 auto-loads
   ~/.prsm/operator_delegation.json from that path, or set
   PRSM_OPERATOR_DELEGATION_FILE to override.
   Tip: add --register to also auto-record this binding."

The no-write fallback (operator copy-pastes manually) is
preserved + now also suggests --write as the tip (mirroring
the --register tip from sprint 796).

Other findings during dogfood:
- FINDING #1 (false alarm): parallax-readiness exit code was
  misread due to bash compound-command `$?` capturing the
  tail subshell's exit, not python's. Re-tested with explicit
  `>file 2>&1; echo $?` and got exit 1 correctly when
  required env vars missing.
- FINDING #3 (false alarm): same artifact for `node doctor`
  against unreachable daemon — re-tested cleanly + got
  exit 2 as expected.

Pin tests:
- --write-path successful write → tail message includes
  "Delegation written to" (the new sprint-823 branch).
- --write-path successful write → tail message does NOT
  include "Save the JSON above" (the old confusing line).
- No --write → tail message DOES include "Save the JSON
  above" (preserved fallback).
- No --write → tail message includes "--write" tip (sprint
  823 added the --write hint to the fallback path so first-
  time operators discover it).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args, env=None):
    from prsm.cli import wallet as _wallet_group
    runner = CliRunner()
    return runner.invoke(
        _wallet_group, ["devices"] + list(args),
        env=env or {},
    )


def _pk():
    from eth_account import Account
    acct = Account.create()
    return acct.key.to_0x_hex()


def test_write_path_success_emits_new_message(tmp_path: Path):
    target = tmp_path / "delegation.json"
    result = _invoke(
        [
            "add", "--node-id", "a" * 32,
            "--write-path", str(target),
            "--format", "text",
        ],
        env={"PRIVATE_KEY": _pk()},
    )
    assert result.exit_code == 0, result.output
    assert "Delegation written to" in result.output


def test_write_path_success_omits_old_save_above(tmp_path: Path):
    """The old confusing 'Save the JSON above' line should NOT
    appear when --write-path already wrote the file."""
    target = tmp_path / "delegation.json"
    result = _invoke(
        [
            "add", "--node-id", "b" * 32,
            "--write-path", str(target),
            "--format", "text",
        ],
        env={"PRIVATE_KEY": _pk()},
    )
    assert result.exit_code == 0
    assert "Save the JSON above" not in result.output


def test_no_write_keeps_save_above_fallback():
    """The manual-save fallback path stays intact when no
    --write flag — operators using the bare `add` command
    still see the step-by-step copy/paste guidance."""
    result = _invoke(
        [
            "add", "--node-id", "c" * 32,
            "--format", "text",
        ],
        env={"PRIVATE_KEY": _pk()},
    )
    assert result.exit_code == 0
    assert "Save the JSON above" in result.output


def test_no_write_fallback_suggests_write_flag():
    """First-time operators on the manual path should see
    --write as a tip so they discover the simpler flow."""
    result = _invoke(
        [
            "add", "--node-id", "d" * 32,
            "--format", "text",
        ],
        env={"PRIVATE_KEY": _pk()},
    )
    assert result.exit_code == 0
    assert "--write" in result.output

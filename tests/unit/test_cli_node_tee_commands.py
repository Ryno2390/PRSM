"""Sprint 436 — `prsm node tee ...` CLI trifecta gap closure.

The /admin/tee-policy/* REST surface + `prsm_tee_policy` MCP
existed; CLI was the gap per PRSM_Testing.md §13.

Two subcommands:
- `prsm node tee status` — wraps GET /admin/tee-policy/node-status
  (this node's own attestation tier — operators pre-screen before
  workload dispatch)
- `prsm node tee evaluate --policy-file <path> [--attestation-b64
  <b64>]` — wraps POST /admin/tee-policy/evaluate. Policy comes
  from a JSON file (TEEPolicy schema). Attestation is optional;
  omit it for pre-flight policy validation.

These pins fire if the commands are removed, if the required
--policy-file arg is silently made optional, or if the help text
loses the "pre-screen" / TEEPolicy-schema references.
"""
from __future__ import annotations

import json
import os
import tempfile

from click.testing import CliRunner

from prsm.cli import main


def test_node_tee_group_registered():
    """`prsm node tee` group + status + evaluate subcommands
    must exist."""
    runner = CliRunner()
    result = runner.invoke(main, ["node", "tee", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output
    assert "evaluate" in result.output


def test_node_tee_status_help_documents_format():
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "tee", "status", "--help"],
    )
    assert result.exit_code == 0
    assert "--format" in result.output


def test_node_tee_evaluate_requires_policy_file():
    """The policy-file arg is required — operators must
    NOT be able to invoke evaluate without specifying the
    policy. A missing-policy call would default-permissive
    and is a security footgun."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "tee", "evaluate"],
    )
    assert result.exit_code != 0
    assert "policy-file" in result.output.lower()


def test_node_tee_evaluate_help_references_tee_policy_schema():
    """The --policy-file help text must point operators at
    the TEEPolicy schema so they know what fields are valid.
    Without that pointer, ops would copy-paste broken policies."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "tee", "evaluate", "--help"],
    )
    assert result.exit_code == 0
    assert "TEEPolicy" in result.output


def test_node_tee_evaluate_attestation_optional():
    """attestation-b64 is OPTIONAL — operators run pre-flight
    policy validation (no attestation) BEFORE accepting a
    workload. The help text should reflect this. Making it
    required would block the pre-flight path."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "tee", "evaluate", "--help"],
    )
    # If marked [required], operators couldn't pre-flight.
    # Look for the help-text indicator
    text = result.output.lower()
    assert "optional" in text or "if omitted" in text or "pre-flight" in text


def test_node_tee_evaluate_rejects_broken_json_policy_file():
    """A policy file that isn't valid JSON must surface a
    clear error + exit 1, not silently fall through with a
    default-permissive policy."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as tf:
        tf.write("this is not json")
        broken_path = tf.name
    try:
        runner = CliRunner()
        result = runner.invoke(
            main, [
                "node", "tee", "evaluate",
                "--policy-file", broken_path,
            ],
        )
        # CliRunner sees sys.exit(1) as exit_code==1
        assert result.exit_code == 1
        assert "json" in result.output.lower() or "valid" in result.output.lower()
    finally:
        os.unlink(broken_path)


def test_node_tee_status_help_mentions_pre_screen_purpose():
    """The status command's help text must explain WHY
    operators run it — pre-screening workload eligibility.
    Without that context, operators might think it's just
    a generic health check."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "tee", "status", "--help"],
    )
    assert result.exit_code == 0
    text = result.output.lower()
    assert "pre-screen" in text or "eligible" in text

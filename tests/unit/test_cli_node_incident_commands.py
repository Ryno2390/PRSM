"""Sprint 434 — `prsm node incident ...` CLI trifecta gap closure.

The /admin/incident/* REST surface + `prsm_incident` MCP tool
existed; the CLI lane was the gap per PRSM_Testing.md §13
"Operator-trifecta gaps". This sprint adds the read-only triage
commands (list / details / playbook) operators need at incident
time.

Mutating commands (open / advance / log-event) need more careful
input-parameter UX; deferred. Operators can still hit those via
REST or the `prsm_incident` MCP.

These pins fire if the commands are silently removed or if the
severity/phase enum vocabulary drifts away from the IncidentResponse
schema.
"""
from __future__ import annotations

from click.testing import CliRunner

from prsm.cli import main


def test_node_incident_group_registered():
    """`prsm node incident` must exist as a click subcommand
    group. Help text identifies it as the triage surface."""
    runner = CliRunner()
    result = runner.invoke(main, ["node", "incident", "--help"])
    assert result.exit_code == 0
    assert "incident" in result.output.lower()
    # The three subcommands must be advertised
    assert "list" in result.output
    assert "details" in result.output
    assert "playbook" in result.output


def test_node_incident_list_help():
    """`prsm node incident list` must surface the actual
    severity vocabulary (s0/s1/s2/s3) in --help. Drift to
    minor/major/critical would silently break operators
    typing what the help suggests."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "incident", "list", "--help"],
    )
    assert result.exit_code == 0
    assert "s0" in result.output
    assert "s1" in result.output
    assert "--format" in result.output


def test_node_incident_details_help():
    """`prsm node incident details <id>` must take an
    incident_id argument + support --format."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "incident", "details", "--help"],
    )
    assert result.exit_code == 0
    assert "INCIDENT_ID" in result.output or "incident_id" in result.output.lower()
    assert "--format" in result.output


def test_node_incident_playbook_help():
    """`prsm node incident playbook` must support filtering
    by severity (operators triaging an S1 don't want the
    full S0 wall of text)."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "incident", "playbook", "--help"],
    )
    assert result.exit_code == 0
    assert "--severity" in result.output


def test_node_incident_severity_vocab_matches_schema():
    """The CLI's documented severity values must be the
    same vocabulary the IncidentSeverity enum uses. Drift
    here would let an operator type --severity=critical
    and silently get a 422 from the server."""
    from prsm.economy.web3.incident_response import IncidentSeverity

    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "incident", "list", "--help"],
    )
    # The enum has S0/S1/S2/S3 → lowercase s0/s1/s2/s3 on the wire
    for sev_obj in IncidentSeverity:
        assert sev_obj.value in result.output, (
            f"severity {sev_obj.value!r} from schema not "
            f"documented in CLI --help; operators would type "
            f"the wrong vocabulary"
        )

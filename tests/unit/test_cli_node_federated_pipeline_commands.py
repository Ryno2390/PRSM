"""Sprint 437 — federated + pipeline CLI trifecta gap closure.

Both surfaces are deep (~7 endpoints each: list / details /
execute / round / aggregate / issue-round / update). This sprint
closes the read-only triage path — the operator subset needed at
incident time. Mutating commands deferred per the sprint-434
incident-CLI pattern.
"""
from __future__ import annotations

from click.testing import CliRunner

from prsm.cli import main


# ── Federated group ──────────────────────────────────────


def test_node_federated_group_registered():
    runner = CliRunner()
    result = runner.invoke(main, ["node", "federated", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "details" in result.output


def test_node_federated_list_status_filter_documented():
    """--status filter MUST advertise the JobStatus
    vocabulary so operators don't guess wrong values."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "federated", "list", "--help"],
    )
    assert result.exit_code == 0
    assert "--status" in result.output
    text = result.output.lower()
    assert "jobstatus" in text or "pending" in text or "active" in text


def test_node_federated_details_requires_job_id():
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "federated", "details"],
    )
    assert result.exit_code != 0


def test_node_federated_help_mentions_readonly_scope():
    """Help text must surface that mutating commands are
    deferred — operators reading --help must know they need
    REST or MCP for issue-round/aggregate/update."""
    runner = CliRunner()
    result = runner.invoke(main, ["node", "federated", "--help"])
    text = result.output.lower()
    assert "read-only" in text or "triage" in text


# ── Pipeline group ───────────────────────────────────────


def test_node_pipeline_group_registered():
    runner = CliRunner()
    result = runner.invoke(main, ["node", "pipeline", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "details" in result.output


def test_node_pipeline_list_help_format_option():
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "pipeline", "list", "--help"],
    )
    assert result.exit_code == 0
    assert "--format" in result.output


def test_node_pipeline_details_requires_job_id():
    runner = CliRunner()
    result = runner.invoke(
        main, ["node", "pipeline", "details"],
    )
    assert result.exit_code != 0


# ── Cross-group consistency ──────────────────────────────


def test_both_groups_default_to_text_format():
    """Operator UX consistency: both groups default to text
    output for human readability. JSON is opt-in via
    --format json for ops pipelines."""
    runner = CliRunner()
    for group in ("federated", "pipeline"):
        result = runner.invoke(
            main, ["node", group, "list", "--help"],
        )
        assert "default: text" in result.output.lower(), (
            f"{group} list should default to text format "
            f"for human readability"
        )

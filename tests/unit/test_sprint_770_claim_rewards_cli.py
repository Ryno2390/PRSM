"""Sprint 770 — `prsm node claim-rewards` CLI for manual claim.

Operators want to trigger an immediate claim without waiting
for the auto-claim interval. Sprint 770 ships a CLI that POSTs
to the existing /staking/claim-rewards endpoint.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args=None):
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["claim-rewards"] + (args or []),
    )


def test_claim_rewards_command_registered():
    """Command exists."""
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "claim-rewards" in cmd_names


def test_success_renders_total_text():
    """Happy path: daemon returns 200 → text output shows
    claimed amount."""
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "total_rewards_claimed": "150",
        "stakes_processed": [{"stake_id": "s1"}],
    }
    with patch("httpx.post", return_value=fake):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "150" in result.output
    assert "FTNS" in result.output


def test_success_returns_full_json():
    """JSON output: full payload."""
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "total_rewards_claimed": "75",
        "stakes_processed": [],
    }
    with patch("httpx.post", return_value=fake):
        result = _invoke(["--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total_rewards_claimed"] == "75"


def test_zero_claimed_shows_actionable_message():
    """0 FTNS claimed → operator sees "no rewards above
    threshold" instead of a confusing silent success."""
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "total_rewards_claimed": "0",
        "stakes_processed": [],
    }
    with patch("httpx.post", return_value=fake):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0
    assert "No stakes" in result.output or (
        "no" in result.output.lower() and "threshold" in result.output.lower()
    )


def test_daemon_unreachable_exits_2():
    """Network error → exit code 2 + clear message."""
    with patch(
        "httpx.post",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()


def test_daemon_error_response_exits_1():
    """Daemon returns 503 (staking_manager not initialized) →
    exit 1 with the error surfaced."""
    fake = MagicMock()
    fake.status_code = 503
    fake.text = '{"detail":"Staking manager not initialized"}'
    with patch("httpx.post", return_value=fake):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    assert "503" in result.output


def test_stake_id_flag_passed_as_query_param():
    """--stake-id S1 should pass `?stake_id=S1` to the endpoint."""
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "total_rewards_claimed": "10",
        "stakes_processed": [{"stake_id": "S1"}],
    }
    with patch("httpx.post", return_value=fake) as mock_post:
        result = _invoke(["--stake-id", "S1"])
    assert result.exit_code == 0, result.output
    # The call should have included params={"stake_id": "S1"}
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs.get("params") == {"stake_id": "S1"}


def test_no_stake_id_means_no_query_params():
    """Without --stake-id, the request has params=None
    (omitted from URL)."""
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "total_rewards_claimed": "10",
        "stakes_processed": [],
    }
    with patch("httpx.post", return_value=fake) as mock_post:
        result = _invoke([])
    assert result.exit_code == 0
    assert mock_post.call_args.kwargs.get("params") is None

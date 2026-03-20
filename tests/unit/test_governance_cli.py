"""
Unit tests for governance CLI commands.

Tests cover:
- CLI sends Authorization header
- CLI uses correct query param names
- CLI calls correct endpoint
- CLI sends correct request body format
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from prsm.cli import main


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestGovernanceCli:
    """Tests for prsm governance CLI commands."""

    def test_proposals_sends_auth_header(self, tmp_path, monkeypatch, cli_runner):
        """prsm governance proposals sends Authorization: Bearer header."""
        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock httpx.get to capture headers
        captured_headers = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Proposals retrieved",
            "data": {
                "proposals": [],
                "total_count": 0,
                "filter": "all"
            }
        }

        def capture_get(*args, **kwargs):
            captured_headers.append(kwargs.get("headers", {}))
            return mock_response

        with patch("httpx.get", side_effect=capture_get):
            result = cli_runner.invoke(main, ["governance", "proposals"])

        # Assert Bearer token in Authorization header
        assert len(captured_headers) == 1
        headers = captured_headers[0]
        assert "Authorization" in headers
        assert "Bearer" in headers["Authorization"]
        assert result.exit_code == 0

    def test_proposals_uses_status_filter_param(self, tmp_path, monkeypatch, cli_runner):
        """prsm governance proposals --status active sends status_filter=active (not status=active)."""
        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock httpx.get to capture params
        captured_params = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Proposals retrieved",
            "data": {
                "proposals": [],
                "total_count": 0,
                "filter": "active"
            }
        }

        def capture_get(*args, **kwargs):
            captured_params.append(kwargs.get("params", {}))
            return mock_response

        with patch("httpx.get", side_effect=capture_get):
            result = cli_runner.invoke(main, ["governance", "proposals", "--status", "active"])

        # Assert status_filter param (not status)
        assert len(captured_params) == 1
        params = captured_params[0]
        assert "status_filter" in params
        assert params["status_filter"] == "active"
        assert "status" not in params or params.get("status") is None

    def test_vote_calls_correct_endpoint(self, tmp_path, monkeypatch, cli_runner):
        """prsm governance vote calls POST /api/v1/governance/vote (not /proposals/{id}/vote)."""
        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock httpx.post to capture URL
        captured_url = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Vote cast",
            "data": {
                "vote": {
                    "proposal_id": "proposal-123",
                    "vote_choice": True,
                    "rationale": None,
                    "cast_at": "2024-01-01T00:00:00Z"
                }
            }
        }

        def capture_post(*args, **kwargs):
            captured_url.append(args[0] if args else kwargs.get("url"))
            return mock_response

        with patch("httpx.post", side_effect=capture_post):
            result = cli_runner.invoke(main, ["governance", "vote", "proposal-123", "--choice", "for"])

        # Assert correct endpoint
        assert len(captured_url) == 1
        url = captured_url[0]
        assert url.endswith("/api/v1/governance/vote")
        # Ensure it's NOT the old endpoint pattern
        assert "/proposals/proposal-123/vote" not in url

    def test_vote_sends_bool_vote_choice(self, tmp_path, monkeypatch, cli_runner):
        """prsm governance vote --choice for sends vote_choice=True (not choice='yes')."""
        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock httpx.post to capture JSON body
        captured_body = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Vote cast",
            "data": {
                "vote": {
                    "proposal_id": "proposal-123",
                    "vote_choice": True,
                    "rationale": None,
                    "cast_at": "2024-01-01T00:00:00Z"
                }
            }
        }

        def capture_post(*args, **kwargs):
            captured_body.append(kwargs.get("json", {}))
            return mock_response

        with patch("httpx.post", side_effect=capture_post):
            result = cli_runner.invoke(main, ["governance", "vote", "proposal-123", "--choice", "for"])

        # Assert correct body format
        assert len(captured_body) == 1
        body = captured_body[0]
        
        # vote_choice should be a boolean True (not string "yes")
        assert "vote_choice" in body
        assert body["vote_choice"] is True
        assert isinstance(body["vote_choice"], bool)
        
        # Old field name "choice" must not be present
        assert "choice" not in body
        
        # proposal_id must be present
        assert body["proposal_id"] == "proposal-123"

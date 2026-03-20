"""Unit tests for wallet CLI commands (login, logout, ftns balance/transfer/history/stake)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Import the main CLI group
from prsm.cli import main


class TestWalletCli:

    def test_login_saves_credentials(self, tmp_path, monkeypatch, cli_runner):
        """prsm login saves access_token and api_url to credentials file."""
        # Monkeypatch _CREDENTIALS_FILE to a tmp path
        cred_file = tmp_path / ".prsm" / "credentials.json"
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)
        
        # Mock httpx.post to return 200 with access_token, refresh_token, expires_in
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600
        }
        
        # Patch httpx.post at the module level (since it's imported inside the function)
        with patch("httpx.post", return_value=mock_response):
            result = cli_runner.invoke(main, ["login", "--username", "testuser", "--password", "testpass"])
        
        assert result.exit_code == 0
        assert cred_file.exists()
        
        creds = json.loads(cred_file.read_text())
        assert creds["access_token"] == "test-access-token"
        assert creds["api_url"] == "http://127.0.0.1:8000"
        assert creds["username"] == "testuser"

    def test_login_wrong_password_exits_1(self, cli_runner):
        """prsm login with bad credentials exits 1 with a clear message."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        
        with patch("httpx.post", return_value=mock_response):
            result = cli_runner.invoke(main, ["login", "--username", "u", "--password", "wrong"])
        
        assert result.exit_code == 1
        assert "Invalid username or password" in result.output

    def test_balance_sends_auth_header(self, tmp_path, monkeypatch, cli_runner):
        """prsm ftns balance includes Authorization: Bearer header."""
        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({"access_token": "test-token", "api_url": "http://test"}))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)
        
        # Mock httpx.get to capture the call and return balance response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "balance": 100.0,
            "available_balance": 80.0,
            "locked_balance": 20.0,
            "user_id": "user-123"
        }
        
        captured_kwargs = {}
        def capture_get(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_response
        
        with patch("httpx.get", side_effect=capture_get):
            result = cli_runner.invoke(main, ["ftns", "balance"])
        
        assert result.exit_code == 0
        assert "Authorization" in captured_kwargs.get("headers", {})
        assert "Bearer test-token" in captured_kwargs["headers"]["Authorization"]

    def test_transfer_uses_correct_field_names(self, tmp_path, monkeypatch, cli_runner):
        """prsm ftns transfer sends 'recipient' not 'to_address' in request body."""
        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({"access_token": "test-token", "api_url": "http://test"}))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)
        
        # Mock httpx.post, capture the JSON body
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "recipient": "user2",
            "amount": 10.0,
            "status": "completed"
        }
        
        captured_kwargs = {}
        def capture_post(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_response
        
        with patch("httpx.post", side_effect=capture_post):
            result = cli_runner.invoke(main, ["ftns", "transfer", "--to", "user2", "--amount", "10"])
        
        assert result.exit_code == 0
        body = captured_kwargs.get("json", {})
        assert "recipient" in body
        assert "to_address" not in body
        assert "description" in body
        assert "memo" not in body

    def test_history_calls_correct_endpoint(self, tmp_path, monkeypatch, cli_runner):
        """prsm ftns history calls /transactions not /history."""
        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({"access_token": "test-token", "api_url": "http://test"}))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)
        
        # Mock httpx.get, capture the URL
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "transactions": [],
            "user_id": "user-123"
        }
        
        captured_args = []
        def capture_get(*args, **kwargs):
            captured_args.append(args)
            return mock_response
        
        with patch("httpx.get", side_effect=capture_get):
            result = cli_runner.invoke(main, ["ftns", "history"])
        
        assert result.exit_code == 0
        assert len(captured_args) > 0
        url = captured_args[0][0]
        assert "/api/v1/ftns/transactions" in url
        assert "/api/v1/ftns/history" not in url


@pytest.fixture
def cli_runner():
    return CliRunner()

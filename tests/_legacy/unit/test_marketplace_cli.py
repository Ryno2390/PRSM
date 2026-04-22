"""
Unit tests for marketplace CLI commands.

Tests cover:
- CLI calls correct endpoint for list
- CLI passes filters as query params
- CLI sends correct order body for buy
- CLI requires authentication
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


class TestMarketplaceCli:
    """Tests for prsm marketplace CLI commands."""

    def test_list_calls_correct_endpoint(self, tmp_path, monkeypatch, cli_runner):
        """prsm marketplace list calls GET /api/v1/marketplace/resources."""
        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock httpx.get to capture the URL
        captured_url = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "resources": [],
            "total": 0,
            "page": 1,
            "page_size": 20
        }

        def capture_get(*args, **kwargs):
            captured_url.append(args[0] if args else kwargs.get("url"))
            return mock_response

        with patch("httpx.get", side_effect=capture_get):
            result = cli_runner.invoke(main, ["marketplace", "list"])

        # Assert captured URL ends with /api/v1/marketplace/resources
        assert len(captured_url) == 1
        assert captured_url[0].endswith("/api/v1/marketplace/resources")

    def test_list_passes_filters_as_query_params(self, tmp_path, monkeypatch, cli_runner):
        """prsm marketplace list passes type and query filters as query params."""
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
            "resources": [],
            "total": 0,
            "page": 1,
            "page_size": 20
        }

        def capture_get(*args, **kwargs):
            captured_params.append(kwargs.get("params", {}))
            return mock_response

        with patch("httpx.get", side_effect=capture_get):
            result = cli_runner.invoke(main, [
                "marketplace", "list",
                "--type", "ai_model",
                "--query", "gpt"
            ])

        # Assert filters were passed as params
        assert len(captured_params) == 1
        params = captured_params[0]
        assert params.get("resource_type") == "ai_model"
        assert params.get("query") == "gpt"

    def test_buy_sends_correct_order_body(self, tmp_path, monkeypatch, cli_runner):
        """prsm marketplace buy sends correct POST body to /api/v1/marketplace/orders."""
        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock GET response for resource details
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "resource_id": "res-123",
            "name": "Test Model",
            "resource_type": "ai_model",
            "price": 10.0,
            "creator_id": "creator-1"
        }

        # Mock POST response for order
        mock_post_response = MagicMock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = {
            "order_id": "order-123",
            "resource_id": "res-123",
            "buyer_id": "test-user",
            "order_type": "purchase",
            "quantity": 1,
            "total_price": 10.0,
            "status": "completed"
        }

        captured_post_body = []
        captured_post_url = []

        def capture_get(*args, **kwargs):
            return mock_get_response

        def capture_post(*args, **kwargs):
            captured_post_url.append(args[0] if args else kwargs.get("url"))
            captured_post_body.append(kwargs.get("json", {}))
            return mock_post_response

        with patch("httpx.get", side_effect=capture_get), \
             patch("httpx.post", side_effect=capture_post):
            # Use --yes to skip confirmation prompt
            result = cli_runner.invoke(main, [
                "marketplace", "buy", "res-123",
                "--order-type", "purchase",
                "--quantity", "1",
                "--yes"
            ])

        # Assert POST was called with correct endpoint and body
        assert len(captured_post_url) == 1
        assert captured_post_url[0].endswith("/api/v1/marketplace/orders")
        
        assert len(captured_post_body) == 1
        post_body = captured_post_body[0]
        assert post_body.get("resource_id") == "res-123"
        assert post_body.get("order_type") == "purchase"
        assert post_body.get("quantity") == 1

    def test_list_requires_login(self, tmp_path, monkeypatch, cli_runner):
        """prsm marketplace list exits 1 with login hint when not authenticated."""
        # No credentials file present - point to non-existent file
        non_existent = tmp_path / ".prsm" / "credentials.json"
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", non_existent)

        # Invoke list
        result = cli_runner.invoke(main, ["marketplace", "list"])

        # Assert exit_code == 1
        assert result.exit_code == 1
        # Assert "prsm login" in output
        assert "prsm login" in result.output

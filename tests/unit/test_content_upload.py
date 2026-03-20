"""
Unit tests for content upload CLI and API.

Tests cover:
- CLI calls correct endpoint
- CLI requires authentication
- CLI sends auth header
- API endpoint registers provenance
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from click.testing import CliRunner

from prsm.cli import main


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestContentUploadCli:
    """Tests for prsm storage upload CLI command."""

    def test_upload_calls_correct_endpoint(self, tmp_path, monkeypatch, cli_runner):
        """prsm storage upload calls /api/v1/content/upload (not /api/v1/storage/upload)."""
        # Write a temp file to upload
        tmp_file = tmp_path / "test_upload.txt"
        tmp_file.write_text("test content")

        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock httpx.post to capture the URL
        captured_url = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cid": "QmTestCID123",
            "filename": "test_upload.txt",
            "size_bytes": 12,
            "content_hash": "abc123",
            "creator_id": "test-user",
            "royalty_rate": 0.01,
            "parent_cids": [],
            "provenance_registered": True,
            "access_url": "https://ipfs.io/ipfs/QmTestCID123"
        }

        def capture_post(*args, **kwargs):
            captured_url.append(args[0] if args else kwargs.get("url"))
            return mock_response

        with patch("httpx.post", side_effect=capture_post):
            result = cli_runner.invoke(main, ["storage", "upload", str(tmp_file)])

        # Assert captured URL ends with "/api/v1/content/upload"
        assert len(captured_url) == 1
        assert captured_url[0].endswith("/api/v1/content/upload")

    def test_upload_requires_login(self, tmp_path, monkeypatch, cli_runner):
        """prsm storage upload exits 1 with login hint when not authenticated."""
        # Create a real temp file (click.Path(exists=True) requires it to exist)
        tmp_file = tmp_path / "test_upload.txt"
        tmp_file.write_text("test content")

        # No credentials file present - point to non-existent file
        non_existent = tmp_path / ".prsm" / "credentials.json"
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", non_existent)

        # Invoke upload
        result = cli_runner.invoke(main, ["storage", "upload", str(tmp_file)])

        # Assert exit_code == 1
        assert result.exit_code == 1
        # Assert "prsm login" in output
        assert "prsm login" in result.output

    def test_upload_sends_auth_header(self, tmp_path, monkeypatch, cli_runner):
        """prsm storage upload includes Authorization: Bearer in the request."""
        # Write a temp file to upload
        tmp_file = tmp_path / "test_upload.txt"
        tmp_file.write_text("test content")

        # Write mock credentials file
        cred_file = tmp_path / ".prsm" / "credentials.json"
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text(json.dumps({
            "access_token": "test-token",
            "user_id": "test-user",
            "api_url": "http://localhost:8000"
        }))
        monkeypatch.setattr("prsm.cli._CREDENTIALS_FILE", cred_file)

        # Mock httpx.post to capture headers
        captured_headers = []
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cid": "QmTestCID123",
            "filename": "test_upload.txt",
            "size_bytes": 12,
            "content_hash": "abc123",
            "creator_id": "test-user",
            "royalty_rate": 0.01,
            "parent_cids": [],
            "provenance_registered": True,
            "access_url": "https://ipfs.io/ipfs/QmTestCID123"
        }

        def capture_post(*args, **kwargs):
            captured_headers.append(kwargs.get("headers", {}))
            return mock_response

        with patch("httpx.post", side_effect=capture_post):
            result = cli_runner.invoke(main, ["storage", "upload", str(tmp_file)])

        # Assert "Bearer test-token" in captured headers["Authorization"]
        assert len(captured_headers) == 1
        auth_header = captured_headers[0].get("Authorization", "")
        assert "Bearer test-token" in auth_header

    def test_upload_endpoint_registers_provenance(self, tmp_path):
        """POST /api/v1/content/upload calls ProvenanceQueries.upsert_provenance."""
        import asyncio
        from unittest.mock import AsyncMock

        # Mock get_ipfs_client
        mock_ipfs_result = MagicMock()
        mock_ipfs_result.success = True
        mock_ipfs_result.cid = "QmTestCID123"
        mock_ipfs_result.error = None

        mock_ipfs_client = MagicMock()
        mock_ipfs_client.upload_content = AsyncMock(return_value=mock_ipfs_result)

        # Mock ProvenanceQueries.upsert_provenance
        # Note: get_ipfs_client is imported inside the function, so we patch at the source module
        with patch("prsm.core.ipfs_client.get_ipfs_client", return_value=mock_ipfs_client), \
             patch("prsm.core.database.ProvenanceQueries.upsert_provenance", new_callable=AsyncMock) as mock_upsert:

            mock_upsert.return_value = True

            # Import the upload function directly to test without FastAPI multipart dependency
            from prsm.interface.api.content_api import upload_content_with_provenance
            from starlette.datastructures import UploadFile
            import io

            # Create a mock upload file
            mock_upload = UploadFile(
                filename="test.txt",
                file=io.BytesIO(b"test content")
            )

            # Call the endpoint directly
            async def run_test():
                result = await upload_content_with_provenance(
                    file=mock_upload,
                    description="Test file",
                    royalty_rate=0.01,
                    parent_cids="",
                    replicas=3,
                    current_user="test-user"
                )
                return result

            result = asyncio.run(run_test())

            # Assert result contains expected fields
            assert result.get("provenance_registered") == True
            assert result.get("cid") == "QmTestCID123"

            # Assert ProvenanceQueries.upsert_provenance called with creator_id == authenticated user
            mock_upsert.assert_called_once()
            call_args = mock_upsert.call_args[0][0]
            assert call_args.get("creator_id") == "test-user"

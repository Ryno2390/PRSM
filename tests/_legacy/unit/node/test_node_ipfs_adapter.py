"""Unit tests for _NodeIPFSAdapter in prsm/node/node.py"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib


class TestNodeIPFSAdapter:
    """Tests for the _NodeIPFSAdapter class."""

    @pytest.mark.asyncio
    async def test_store_model_uploads_to_real_ipfs(self):
        """store_model calls IPFSClient.upload_content and returns real CID."""
        # Import the adapter
        from prsm.node.node import _NodeIPFSAdapter

        # Create mock IPFSClient
        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.initialize = AsyncMock()
        mock_upload_result = MagicMock()
        mock_upload_result.success = True
        mock_upload_result.cid = "QmRealCID123456789"
        mock_client.upload_content = AsyncMock(return_value=mock_upload_result)

        adapter = _NodeIPFSAdapter("http://localhost:5001")

        # Patch IPFSClient creation at the source module where it's imported from
        with patch("prsm.core.ipfs_client.IPFSClient", return_value=mock_client), \
             patch("prsm.core.ipfs_client.IPFSConfig") as mock_config:
            mock_config.return_value = MagicMock()
            
            cid = await adapter.store_model(b"model weights", {"version": "1.0"})

        # Assert return value is the real CID from upload
        assert cid == "QmRealCID123456789"
        # Assert upload_content was called twice: once for model, once for metadata
        assert mock_client.upload_content.call_count == 2
        # First call should be for the model data
        first_call = mock_client.upload_content.call_args_list[0]
        assert first_call[0][0] == b"model weights"
        assert first_call[1]["filename"] == "model.bin"
        assert first_call[1]["pin"] is True

    @pytest.mark.asyncio
    async def test_store_model_returns_placeholder_when_ipfs_unavailable(self):
        """store_model logs a warning and returns a placeholder CID when IPFS is down."""
        from prsm.node.node import _NodeIPFSAdapter

        adapter = _NodeIPFSAdapter("http://localhost:5001")
        model_data = b"model weights"

        # Patch IPFSClient.initialize to raise ConnectionRefusedError
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock(side_effect=ConnectionRefusedError("Connection refused"))

        with patch("prsm.core.ipfs_client.IPFSClient", return_value=mock_client), \
             patch("prsm.core.ipfs_client.IPFSConfig") as mock_config:
            mock_config.return_value = MagicMock()
            
            cid = await adapter.store_model(model_data, {})

        # Assert return value starts with "Qm" (placeholder format)
        assert cid.startswith("Qm")
        # Assert return value is the expected placeholder (sha256-based)
        expected_placeholder = f"Qm{hashlib.sha256(model_data).hexdigest()[:44]}"
        assert cid == expected_placeholder

    @pytest.mark.asyncio
    async def test_retrieve_model_fetches_bytes(self):
        """retrieve_model calls IPFSClient.download_content and returns bytes."""
        from prsm.node.node import _NodeIPFSAdapter

        # Create mock IPFSClient
        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.initialize = AsyncMock()
        mock_download_result = MagicMock()
        mock_download_result.success = True
        mock_download_result.metadata = {"content": b"model weights"}
        mock_client.download_content = AsyncMock(return_value=mock_download_result)

        adapter = _NodeIPFSAdapter("http://localhost:5001")

        with patch("prsm.core.ipfs_client.IPFSClient", return_value=mock_client), \
             patch("prsm.core.ipfs_client.IPFSConfig") as mock_config:
            mock_config.return_value = MagicMock()
            
            result = await adapter.retrieve_model("QmABC")

        # Assert return value is the bytes from download
        assert result == b"model weights"
        # Assert download_content was called with the CID
        mock_client.download_content.assert_called_once_with("QmABC")

    @pytest.mark.asyncio
    async def test_retrieve_model_returns_none_when_ipfs_unavailable(self):
        """retrieve_model returns None (not raises) when IPFS is down."""
        from prsm.node.node import _NodeIPFSAdapter

        adapter = _NodeIPFSAdapter("http://localhost:5001")

        # Patch IPFSClient.initialize to raise ConnectionRefusedError
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock(side_effect=ConnectionRefusedError("Connection refused"))

        with patch("prsm.core.ipfs_client.IPFSClient", return_value=mock_client), \
             patch("prsm.core.ipfs_client.IPFSConfig") as mock_config:
            mock_config.return_value = MagicMock()
            
            result = await adapter.retrieve_model("QmABC")

        # Assert return value is None
        assert result is None

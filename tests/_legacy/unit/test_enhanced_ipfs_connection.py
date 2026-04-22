"""Unit tests for enhanced_ipfs connection failure surfacing (Phase 2 Item 2d)."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture
def disconnected_client():
    """PRSMIPFSClient with connection forced to False."""
    from prsm.data.data_layer.enhanced_ipfs import PRSMIPFSClient
    client = PRSMIPFSClient()
    client.connected = False
    client._initialization_started = True  # skip re-init
    return client


@pytest.fixture
def connected_client():
    """PRSMIPFSClient with a mocked connected IPFS backend."""
    from prsm.data.data_layer.enhanced_ipfs import PRSMIPFSClient
    client = PRSMIPFSClient()
    client.connected = True
    client._initialization_started = True
    mock_ipfs = MagicMock()
    mock_ipfs.add_bytes = MagicMock(return_value="QmRealCID123abc")
    mock_ipfs.cat = MagicMock(return_value=b'{"model_data": "68656c6c6f", "metadata": {}}')
    client.client = mock_ipfs
    return client


class TestDisconnectedStorageRaisesExplicitly:
    @pytest.mark.asyncio
    async def test_store_content_raises_connection_error(self, disconnected_client):
        with pytest.raises(ConnectionError, match="IPFS client is not connected"):
            await disconnected_client._store_content(b"some content")

    @pytest.mark.asyncio
    async def test_retrieve_content_raises_connection_error(self, disconnected_client):
        with pytest.raises(ConnectionError, match="IPFS client is not connected"):
            await disconnected_client._retrieve_content("bafybeigtest123")

    @pytest.mark.asyncio
    async def test_store_model_raises_connection_error(self, disconnected_client):
        with pytest.raises(ConnectionError):
            await disconnected_client.store_model(b"model bytes", {"uploader_id": "u1"})

    @pytest.mark.asyncio
    async def test_store_dataset_raises_connection_error(self, disconnected_client):
        with pytest.raises(ConnectionError):
            await disconnected_client.store_dataset(b"dataset", {"uploader_id": "u1"})

    @pytest.mark.asyncio
    async def test_retrieve_with_provenance_raises_connection_error(self, disconnected_client):
        # retrieve_with_provenance() calls _retrieve_content(); ConnectionError propagates
        with pytest.raises(ConnectionError):
            await disconnected_client.retrieve_with_provenance("bafybeigtest456")

    def test_no_simulation_storage_attribute(self, disconnected_client):
        """simulation_storage dict must no longer exist — it was a simulation artifact."""
        assert not hasattr(disconnected_client, 'simulation_storage')


class TestConnectedStorageWorks:
    @pytest.mark.asyncio
    async def test_store_content_returns_cid_when_connected(self, connected_client):
        cid = await connected_client._store_content(b"hello")
        assert cid == "QmRealCID123abc"

    @pytest.mark.asyncio
    async def test_retrieve_content_returns_bytes_when_connected(self, connected_client):
        data = await connected_client._retrieve_content("QmRealCID123abc")
        assert isinstance(data, bytes)

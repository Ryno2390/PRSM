"""Unit tests for DataSpineProxy real IPFS wiring (Phase 3 Item 3e)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def ipfs_adapter():
    from prsm.compute.spine.data_spine_proxy import IPFSClient
    return IPFSClient()


@pytest.fixture
def proxy():
    from prsm.compute.spine.data_spine_proxy import PRSMDataSpineProxy
    return PRSMDataSpineProxy()


class TestIPFSClientAdapter:
    @pytest.mark.asyncio
    async def test_add_content_returns_real_cid(self, ipfs_adapter):
        """add_content() delegates to canonical client and returns CID string."""
        from prsm.core.ipfs_client import IPFSResult
        mock_canonical = AsyncMock()
        mock_canonical.add_content = AsyncMock(
            return_value=IPFSResult(success=True, cid="QmRealCID123")
        )
        mock_canonical.connected = True
        ipfs_adapter._client = mock_canonical

        cid = await ipfs_adapter.add_content(b"test content")
        assert cid == "QmRealCID123"
        mock_canonical.add_content.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_content_raises_on_failure(self, ipfs_adapter):
        """add_content() raises IPFSConnectionError when canonical client fails."""
        from prsm.core.ipfs_client import IPFSResult, IPFSConnectionError
        mock_canonical = AsyncMock()
        mock_canonical.add_content = AsyncMock(
            return_value=IPFSResult(success=False, error="daemon offline")
        )
        mock_canonical.connected = True
        ipfs_adapter._client = mock_canonical

        with pytest.raises(IPFSConnectionError):
            await ipfs_adapter.add_content(b"test content")

    @pytest.mark.asyncio
    async def test_get_content_returns_real_bytes(self, ipfs_adapter):
        """get_content() extracts bytes from result.metadata['content']."""
        from prsm.core.ipfs_client import IPFSResult
        expected_bytes = b"real IPFS data"
        mock_canonical = AsyncMock()
        mock_canonical.get_content = AsyncMock(
            return_value=IPFSResult(
                success=True, cid="QmTest",
                metadata={"content": expected_bytes}
            )
        )
        mock_canonical.connected = True
        ipfs_adapter._client = mock_canonical

        result = await ipfs_adapter.get_content("QmTest")
        assert result == expected_bytes

    @pytest.mark.asyncio
    async def test_get_content_raises_on_failure(self, ipfs_adapter):
        """get_content() raises IPFSConnectionError on failed retrieval."""
        from prsm.core.ipfs_client import IPFSResult, IPFSConnectionError
        mock_canonical = AsyncMock()
        mock_canonical.get_content = AsyncMock(
            return_value=IPFSResult(success=False, error="not found")
        )
        mock_canonical.connected = True
        ipfs_adapter._client = mock_canonical

        with pytest.raises(IPFSConnectionError):
            await ipfs_adapter.get_content("QmMissing")

    @pytest.mark.asyncio
    async def test_pin_content_returns_false_on_error(self, ipfs_adapter):
        """pin_content() returns False on error without raising."""
        mock_canonical = AsyncMock()
        mock_canonical.pin_content = AsyncMock(side_effect=RuntimeError("pin failed"))
        mock_canonical.connected = True
        ipfs_adapter._client = mock_canonical

        result = await ipfs_adapter.pin_content("QmTest")
        assert result is False

    def test_no_random_in_ipfs_adapter(self, ipfs_adapter):
        """The IPFSClient adapter must not reference random."""
        import inspect
        from prsm.compute.spine.data_spine_proxy import IPFSClient
        src = inspect.getsource(IPFSClient)
        assert "random" not in src

    def test_no_random_in_retrieve_from_https(self, proxy):
        """_retrieve_from_https must not reference random."""
        import inspect
        src = inspect.getsource(proxy._retrieve_from_https)
        assert "random" not in src

    def test_no_random_in_test_performance_targets(self, proxy):
        """_test_performance_targets must not reference random."""
        import inspect
        src = inspect.getsource(proxy._test_performance_targets)
        assert "random" not in src

    @pytest.mark.asyncio
    async def test_retrieve_from_https_uses_http_session(self, proxy):
        """_retrieve_from_https uses self.http_session, not fake content."""
        import aiohttp
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.read = AsyncMock(return_value=b"real HTTP response")
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_response)
        proxy.http_session = mock_session

        content, metadata = await proxy._retrieve_from_https("https://example.com/data")
        assert content == b"real HTTP response"
        assert "example.com" in metadata.original_url
        assert not content.startswith(b"HTTPS content from")  # not fake

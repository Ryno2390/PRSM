"""
Sprint 4 Phase 4: Content Retrieval API Tests
===============================================

Tests for the request_content() cross-node download functionality
in ContentUploader, including provider discovery, inline/gateway
transfer modes, content hash verification, and error handling.
"""

import asyncio
import base64
import hashlib
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Set

from prsm.node.content_uploader import ContentUploader, UploadedContent
from prsm.node.content_index import ContentRecord


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_identity():
    """Create a mock node identity."""
    identity = MagicMock()
    identity.node_id = "node-A"
    identity.public_key_b64 = "mock-pubkey-b64"
    identity.sign = MagicMock(return_value="mock-signature")
    return identity


@pytest.fixture
def mock_gossip():
    """Create a mock gossip protocol."""
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    return gossip


@pytest.fixture
def mock_ledger():
    """Create a mock ledger."""
    ledger = MagicMock()
    ledger.credit = AsyncMock()
    return ledger


@pytest.fixture
def mock_transport():
    """Create a mock transport layer."""
    transport = MagicMock()
    transport.on_message = MagicMock()
    transport.send_to_peer = AsyncMock(return_value=True)
    transport.identity = MagicMock()
    transport.identity.node_id = "node-A"
    return transport


@pytest.fixture
def mock_content_index():
    """Create a mock content index."""
    index = MagicMock()
    return index


@pytest.fixture
def uploader(mock_identity, mock_gossip, mock_ledger, mock_transport, mock_content_index):
    """Create a ContentUploader with all mock dependencies."""
    cu = ContentUploader(
        identity=mock_identity,
        gossip=mock_gossip,
        ledger=mock_ledger,
        transport=mock_transport,
        content_index=mock_content_index,
    )
    return cu


# =============================================================================
# HELPER
# =============================================================================

def make_content_record(cid: str, providers: set, content_hash: str = "") -> ContentRecord:
    """Create a ContentRecord for testing."""
    return ContentRecord(
        cid=cid,
        filename="test.txt",
        size_bytes=100,
        content_hash=content_hash,
        creator_id="creator-node",
        providers=providers,
    )


# =============================================================================
# TEST: request_content() - provider discovery
# =============================================================================

class TestProviderDiscovery:
    """Test that request_content() correctly discovers and contacts providers."""

    @pytest.mark.asyncio
    async def test_no_transport_returns_none(self, uploader):
        """Without transport, request_content should return None."""
        uploader.transport = None
        result = await uploader.request_content("QmTestCid")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_providers_returns_none(self, uploader, mock_content_index):
        """When no providers are known, return None."""
        mock_content_index.lookup.return_value = make_content_record(
            "QmTestCid", providers=set()
        )
        result = await uploader.request_content("QmTestCid")
        assert result is None

    @pytest.mark.asyncio
    async def test_self_provider_skipped(self, uploader, mock_content_index):
        """If only provider is self, skip and return None."""
        mock_content_index.lookup.return_value = make_content_record(
            "QmTestCid", providers={"node-A"}  # self
        )
        result = await uploader.request_content("QmTestCid")
        assert result is None

    @pytest.mark.asyncio
    async def test_local_content_returned_without_network(self, uploader, mock_content_index):
        """If we have the content locally (uploaded_content), try IPFS cat first."""
        uploader.uploaded_content["QmLocal"] = UploadedContent(
            cid="QmLocal",
            filename="local.txt",
            size_bytes=5,
            content_hash="abc",
            creator_id="node-A",
        )
        uploader._ipfs_cat = AsyncMock(return_value=b"hello")

        result = await uploader.request_content("QmLocal")

        assert result == b"hello"
        uploader._ipfs_cat.assert_called_once_with("QmLocal")
        # Should not have contacted the content index
        mock_content_index.lookup.assert_not_called()


# =============================================================================
# TEST: Inline transfer mode
# =============================================================================

class TestInlineTransfer:
    """Test content retrieval via inline base64 transfer."""

    @pytest.mark.asyncio
    async def test_inline_content_decoded(self, uploader, mock_content_index, mock_transport):
        """Inline base64 content should be decoded and returned."""
        test_content = b"Hello, PRSM network!"
        content_hash = hashlib.sha256(test_content).hexdigest()

        mock_content_index.lookup.return_value = make_content_record(
            "QmTest", providers={"node-B"}, content_hash=content_hash
        )

        # Simulate: when send_to_peer is called, resolve the pending future
        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            # Simulate the provider responding
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "subtype": "content_response",
                    "request_id": request_id,
                    "cid": "QmTest",
                    "found": True,
                    "transfer_mode": "inline",
                    "data_b64": base64.b64encode(test_content).decode(),
                    "content_hash": content_hash,
                })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmTest", timeout=5.0)

        assert result == test_content

    @pytest.mark.asyncio
    async def test_hash_verification_rejects_mismatch(self, uploader, mock_content_index, mock_transport):
        """Content with wrong hash should be rejected."""
        test_content = b"Tampered content"
        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

        mock_content_index.lookup.return_value = make_content_record(
            "QmBad", providers={"node-B"}, content_hash=wrong_hash
        )

        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "found": True,
                    "transfer_mode": "inline",
                    "data_b64": base64.b64encode(test_content).decode(),
                })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmBad", timeout=5.0)

        # Should return None because hash doesn't match
        assert result is None

    @pytest.mark.asyncio
    async def test_hash_verification_skipped_when_disabled(self, uploader, mock_content_index, mock_transport):
        """When verify_hash=False, content is returned even with mismatched hash."""
        test_content = b"Any content"
        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

        mock_content_index.lookup.return_value = make_content_record(
            "QmAny", providers={"node-B"}, content_hash=wrong_hash
        )

        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "found": True,
                    "transfer_mode": "inline",
                    "data_b64": base64.b64encode(test_content).decode(),
                })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmAny", timeout=5.0, verify_hash=False)

        assert result == test_content


# =============================================================================
# TEST: Gateway transfer mode
# =============================================================================

class TestGatewayTransfer:
    """Test content retrieval via IPFS gateway URL."""

    @pytest.mark.asyncio
    async def test_gateway_content_fetched(self, uploader, mock_content_index, mock_transport):
        """Gateway URL content should be fetched and returned."""
        test_content = b"Large file content from gateway"
        content_hash = hashlib.sha256(test_content).hexdigest()

        mock_content_index.lookup.return_value = make_content_record(
            "QmGateway", providers={"node-B"}, content_hash=content_hash
        )

        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "found": True,
                    "transfer_mode": "gateway",
                    "gateway_url": "http://127.0.0.1:8080/ipfs/QmGateway",
                })
            return True

        mock_transport.send_to_peer = fake_send
        uploader._fetch_from_gateway = AsyncMock(return_value=test_content)

        result = await uploader.request_content("QmGateway", timeout=5.0)

        assert result == test_content
        uploader._fetch_from_gateway.assert_called_once_with(
            "http://127.0.0.1:8080/ipfs/QmGateway"
        )


# =============================================================================
# TEST: Timeout and error handling
# =============================================================================

class TestErrorHandling:
    """Test timeout and error scenarios in content retrieval."""

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, uploader, mock_content_index, mock_transport):
        """If provider doesn't respond within timeout, return None."""
        mock_content_index.lookup.return_value = make_content_record(
            "QmSlow", providers={"node-B"}
        )

        # Don't resolve the future — simulates no response
        mock_transport.send_to_peer = AsyncMock(return_value=True)

        result = await uploader.request_content("QmSlow", timeout=0.5)

        assert result is None

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self, uploader, mock_content_index, mock_transport):
        """If provider responds with found=False, return None."""
        mock_content_index.lookup.return_value = make_content_record(
            "QmMissing", providers={"node-B"}
        )

        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "found": False,
                    "cid": "QmMissing",
                })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmMissing", timeout=5.0)

        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_to_second_provider(self, uploader, mock_content_index, mock_transport):
        """If first provider fails, try the next one."""
        test_content = b"From second provider"
        content_hash = hashlib.sha256(test_content).hexdigest()

        mock_content_index.lookup.return_value = make_content_record(
            "QmFallback", providers={"node-B", "node-C"}, content_hash=content_hash
        )

        call_count = 0

        async def fake_send(peer_id, msg):
            nonlocal call_count
            call_count += 1
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                if call_count == 1:
                    # First provider says not found
                    future.set_result({"found": False})
                else:
                    # Second provider has it
                    future.set_result({
                        "found": True,
                        "transfer_mode": "inline",
                        "data_b64": base64.b64encode(test_content).decode(),
                    })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmFallback", timeout=5.0)

        assert result == test_content
        assert call_count == 2


# =============================================================================
# TEST: Response handler
# =============================================================================

class TestResponseHandler:
    """Test the _handle_content_response future resolution."""

    def test_response_resolves_pending_future(self, uploader):
        """A content_response should resolve the matching pending future."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        uploader._pending_requests["req-123"] = future

        msg = MagicMock()
        msg.payload = {
            "subtype": "content_response",
            "request_id": "req-123",
            "found": True,
            "transfer_mode": "inline",
            "data_b64": base64.b64encode(b"test").decode(),
        }

        uploader._handle_content_response(msg)

        assert future.done()
        assert future.result()["found"] is True
        loop.close()

    def test_unknown_request_id_ignored(self, uploader):
        """A response for an unknown request_id should be silently ignored."""
        msg = MagicMock()
        msg.payload = {
            "subtype": "content_response",
            "request_id": "unknown-id",
            "found": True,
        }

        # Should not raise
        uploader._handle_content_response(msg)

    def test_already_done_future_not_overwritten(self, uploader):
        """If the future is already resolved, don't overwrite it."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result({"found": False})
        uploader._pending_requests["req-done"] = future

        msg = MagicMock()
        msg.payload = {
            "request_id": "req-done",
            "found": True,
        }

        uploader._handle_content_response(msg)

        # Should still have the original result
        assert future.result()["found"] is False
        loop.close()

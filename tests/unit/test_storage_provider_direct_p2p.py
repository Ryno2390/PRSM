"""Tests for StorageProvider direct P2P challenge/proof delivery."""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prsm.node.storage_provider import StorageProvider
from prsm.node.transport import MSG_DIRECT, P2PMessage


def _make_identity():
    mock = MagicMock()
    mock.node_id = "local_node_abc"
    mock.private_key_bytes = b"\x00" * 32
    mock.public_key_bytes = b"\x00" * 32
    return mock


def _make_storage_provider(transport_send_succeeds=True):
    identity = _make_identity()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=1000.0)
    ledger.transfer = AsyncMock()
    transport = MagicMock()
    transport.on_message = MagicMock()
    transport.send_to_peer = AsyncMock(return_value=transport_send_succeeds)
    discovery = MagicMock()
    discovery.provide_content = AsyncMock()

    sp = StorageProvider(
        identity=identity,
        gossip=gossip,
        ledger=ledger,
        transport=transport,
        discovery=discovery,
    )
    return sp


class TestDirectP2PChallenge:
    """Tests for storage challenge delivery via direct P2P."""

    @pytest.mark.asyncio
    async def test_constructor_accepts_transport_and_discovery(self):
        sp = _make_storage_provider()
        assert sp.transport is not None
        assert sp.discovery is not None

    @pytest.mark.asyncio
    async def test_challenge_deduplication(self):
        """Duplicate challenge IDs should be dropped."""
        sp = _make_storage_provider()
        sp._running = True
        sp._seen_challenge_ids = {}

        challenge_data = {
            "challenge": {"challenge_id": "chal_001", "cid": "QmTest"},
            "challenger_id": "remote_node",
            "target_provider_id": "local_node_abc",
        }

        # Mark as seen
        sp._seen_challenge_ids["chal_001"] = time.time()

        # Second call with same ID should be dropped silently
        await sp._on_storage_challenge("storage_challenge", challenge_data, "remote_node")
        # If dedup works, _storage_prover.answer_challenge should NOT be called
        # (since _storage_prover is None in this test, it would raise if reached)

    @pytest.mark.asyncio
    async def test_seen_challenge_cleanup(self):
        """Old entries in _seen_challenge_ids should be evicted."""
        sp = _make_storage_provider()
        sp._seen_challenge_ids = {
            "old_challenge": time.time() - 700,
            "recent_challenge": time.time() - 60,
        }
        sp._cleanup_seen_challenges()
        assert "old_challenge" not in sp._seen_challenge_ids
        assert "recent_challenge" in sp._seen_challenge_ids

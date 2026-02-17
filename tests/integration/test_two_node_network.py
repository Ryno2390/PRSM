"""
Integration test: Two PRSM nodes discover each other,
exchange a compute job, and settle payment.
"""

import asyncio
from unittest.mock import patch

import pytest

_REAL_SLEEP = asyncio.sleep


@pytest.fixture(autouse=True)
def real_asyncio_sleep():
    """Restore real asyncio.sleep for network tests."""
    with patch("asyncio.sleep", _REAL_SLEEP):
        yield

from prsm.node.compute_provider import ComputeProvider, JobType
from prsm.node.compute_requester import ComputeRequester
from prsm.node.config import NodeConfig, NodeRole
from prsm.node.discovery import PeerDiscovery
from prsm.node.gossip import GossipProtocol
from prsm.node.identity import generate_node_identity
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.transport import WebSocketTransport


async def _setup_node(name, p2p_port, bootstrap=None):
    """Create a minimal node stack for testing."""
    identity = generate_node_identity(name)
    transport = WebSocketTransport(identity, host="127.0.0.1", port=p2p_port)
    gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    await ledger.create_wallet(identity.node_id, name)
    await ledger.create_wallet("system")
    await ledger.issue_welcome_grant(identity.node_id, 100.0)

    discovery = PeerDiscovery(
        transport,
        bootstrap_nodes=[bootstrap] if bootstrap else [],
    )
    provider = ComputeProvider(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
    )
    requester = ComputeRequester(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
    )

    return {
        "identity": identity,
        "transport": transport,
        "gossip": gossip,
        "ledger": ledger,
        "discovery": discovery,
        "provider": provider,
        "requester": requester,
    }


async def _start_node(node):
    await node["transport"].start()
    await node["gossip"].start()
    await node["provider"].start()
    await node["requester"].start()
    await node["discovery"].start()


async def _stop_node(node):
    await node["discovery"].stop()
    await node["provider"].stop()
    await node["requester"].stop()
    await node["gossip"].stop()
    await node["transport"].stop()
    await node["ledger"].close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_two_nodes_discover_and_connect():
    """Two nodes find each other via bootstrap."""
    node_a = await _setup_node("node-A", 19400)
    node_b = await _setup_node("node-B", 19401, bootstrap="127.0.0.1:19400")

    try:
        await _start_node(node_a)
        await _start_node(node_b)

        await asyncio.sleep(0.5)

        assert node_a["transport"].peer_count >= 1
        assert node_b["transport"].peer_count >= 1
    finally:
        await _stop_node(node_b)
        await _stop_node(node_a)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_two_nodes_compute_job_and_payment():
    """Node A submits a job, Node B executes it, payment is settled."""
    node_a = await _setup_node("requester-A", 19410)
    node_b = await _setup_node("provider-B", 19411, bootstrap="127.0.0.1:19410")

    try:
        await _start_node(node_a)
        await _start_node(node_b)
        await asyncio.sleep(0.5)

        # Verify connection
        assert node_a["transport"].peer_count >= 1

        # Check initial balances
        balance_a_before = await node_a["ledger"].get_balance(node_a["identity"].node_id)
        balance_b_before = await node_b["ledger"].get_balance(node_b["identity"].node_id)
        assert balance_a_before == 100.0
        assert balance_b_before == 100.0

        # Node A submits a benchmark job
        submitted = await node_a["requester"].submit_job(
            job_type=JobType.BENCHMARK,
            payload={"iterations": 100},
            ftns_budget=5.0,
        )
        assert submitted.job_id is not None

        # Wait for job to be accepted and completed
        result = await node_a["requester"].get_result(submitted.job_id, timeout=10.0)

        assert result is not None, f"Job timed out. Status: {submitted.status.value}, error: {submitted.error}"
        assert "primes_found" in result
        assert submitted.result_verified  # signature verified

        # Verify payment was recorded
        balance_a_after = await node_a["ledger"].get_balance(node_a["identity"].node_id)
        balance_b_after = await node_b["ledger"].get_balance(node_b["identity"].node_id)

        # A paid 5 FTNS, B earned 5 FTNS
        assert balance_a_after == pytest.approx(95.0, abs=0.01)
        assert balance_b_after == pytest.approx(105.0, abs=0.01)

    finally:
        await _stop_node(node_b)
        await _stop_node(node_a)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_two_nodes_embedding_job():
    """End-to-end embedding job between two nodes."""
    node_a = await _setup_node("client", 19420)
    node_b = await _setup_node("worker", 19421, bootstrap="127.0.0.1:19420")

    try:
        await _start_node(node_a)
        await _start_node(node_b)
        await asyncio.sleep(0.5)

        submitted = await node_a["requester"].submit_job(
            job_type=JobType.EMBEDDING,
            payload={"text": "decentralized AI for science", "dimensions": 64},
            ftns_budget=2.0,
        )

        result = await node_a["requester"].get_result(submitted.job_id, timeout=10.0)

        assert result is not None
        assert len(result["embedding"]) == 64
        assert result["provider_node"] == node_b["identity"].node_id

    finally:
        await _stop_node(node_b)
        await _stop_node(node_a)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_insufficient_balance_rejected():
    """Submitting a job with insufficient balance should fail immediately."""
    node_a = await _setup_node("broke-node", 19430)

    try:
        await _start_node(node_a)

        # Try to submit a job worth more than our balance
        with pytest.raises(ValueError, match="Insufficient FTNS balance"):
            await node_a["requester"].submit_job(
                job_type=JobType.BENCHMARK,
                payload={},
                ftns_budget=999.0,
            )
    finally:
        await _stop_node(node_a)

"""
Tests for AgentDispatcher — requester-side dispatch-bid-transfer-settle lifecycle.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.compute.agents.models import (
    AgentManifest,
    DispatchRecord,
    DispatchStatus,
    MobileAgent,
)
from prsm.compute.agents.dispatcher import AgentDispatcher


# ── Helpers ──────────────────────────────────────────────────────────────

VALID_WASM = b"\x00asm\x01\x00\x00\x00"


def _make_identity(node_id="requester-1"):
    identity = MagicMock()
    identity.node_id = node_id
    identity.sign = MagicMock(return_value="sig-requester")
    identity.public_key_b64 = "cmVxdWVzdGVy"
    return identity


def _make_gossip():
    gossip = AsyncMock()
    gossip.publish = AsyncMock(return_value=3)
    gossip.subscribe = MagicMock()
    return gossip


def _make_transport():
    transport = AsyncMock()
    transport.send_to_peer = AsyncMock()
    return transport


def _make_escrow():
    escrow = AsyncMock()
    escrow.create = AsyncMock(return_value="escrow-123")
    escrow.release = AsyncMock()
    escrow.refund = AsyncMock()
    return escrow


# ── Tests ────────────────────────────────────────────────────────────────

def test_create_agent_returns_signed_agent():
    """create_agent returns a signed MobileAgent with correct origin."""
    identity = _make_identity()
    gossip = _make_gossip()
    transport = _make_transport()
    escrow = _make_escrow()

    dispatcher = AgentDispatcher(identity, gossip, transport, escrow)

    manifest = AgentManifest(min_hardware_tier="t2")
    agent = dispatcher.create_agent(VALID_WASM, manifest, ftns_budget=5.0, ttl=60)

    assert isinstance(agent, MobileAgent)
    assert agent.origin_node == "requester-1"
    assert agent.signature == "sig-requester"
    assert agent.ftns_budget == 5.0
    assert agent.ttl == 60
    assert agent.wasm_binary == VALID_WASM


@pytest.mark.asyncio
async def test_dispatch_creates_escrow_and_publishes():
    """dispatch creates escrow and publishes gossip."""
    identity = _make_identity()
    gossip = _make_gossip()
    transport = _make_transport()
    escrow = _make_escrow()

    dispatcher = AgentDispatcher(identity, gossip, transport, escrow)
    manifest = AgentManifest()
    agent = dispatcher.create_agent(VALID_WASM, manifest, ftns_budget=10.0)

    record = await dispatcher.dispatch(agent)

    assert isinstance(record, DispatchRecord)
    assert record.status == DispatchStatus.BIDDING
    escrow.create.assert_awaited_once()
    gossip.publish.assert_awaited_once()
    assert gossip.publish.call_args[0][0] == "agent_dispatch"


@pytest.mark.asyncio
async def test_dispatch_publishes_manifest_without_binary():
    """dispatch publishes manifest only — no WASM binary in gossip payload."""
    identity = _make_identity()
    gossip = _make_gossip()
    transport = _make_transport()
    escrow = _make_escrow()

    dispatcher = AgentDispatcher(identity, gossip, transport, escrow)
    manifest = AgentManifest()
    agent = dispatcher.create_agent(VALID_WASM, manifest, ftns_budget=10.0)

    await dispatcher.dispatch(agent)

    payload = gossip.publish.call_args[0][1]
    # Binary must NOT be in the gossip payload
    assert "wasm_binary" not in payload
    # Manifest info SHOULD be present
    assert "manifest" in payload
    assert "agent_id" in payload


@pytest.mark.asyncio
async def test_on_agent_accept_appends_bid():
    """_on_agent_accept appends bid to record."""
    identity = _make_identity()
    gossip = _make_gossip()
    transport = _make_transport()
    escrow = _make_escrow()

    dispatcher = AgentDispatcher(identity, gossip, transport, escrow)
    manifest = AgentManifest()
    agent = dispatcher.create_agent(VALID_WASM, manifest, ftns_budget=10.0)
    record = await dispatcher.dispatch(agent)

    bid_data = {
        "agent_id": agent.agent_id,
        "provider_id": "provider-A",
        "bid_price": 3.0,
        "hardware_tier": "t2",
        "reputation": 0.9,
    }

    await dispatcher._on_agent_accept("agent_accept", bid_data, "provider-A")

    assert len(record.bids) == 1
    assert record.bids[0]["provider_id"] == "provider-A"


def test_select_best_bid_returns_highest_score():
    """_select_best_bid returns the bid with the best composite score."""
    identity = _make_identity()
    gossip = _make_gossip()
    transport = _make_transport()
    escrow = _make_escrow()

    dispatcher = AgentDispatcher(identity, gossip, transport, escrow)

    bids = [
        {"provider_id": "A", "bid_price": 8.0, "hardware_tier": "t1", "reputation": 0.5},
        {"provider_id": "B", "bid_price": 3.0, "hardware_tier": "t3", "reputation": 0.95},
        {"provider_id": "C", "bid_price": 5.0, "hardware_tier": "t2", "reputation": 0.7},
    ]

    best = dispatcher._select_best_bid(bids, max_budget=10.0)
    # B should win: good headroom, high tier, excellent reputation
    assert best is not None
    assert best["provider_id"] == "B"


@pytest.mark.asyncio
async def test_on_agent_result_success_releases_escrow():
    """_on_agent_result with success releases escrow, sets COMPLETED."""
    identity = _make_identity()
    gossip = _make_gossip()
    transport = _make_transport()
    escrow = _make_escrow()

    dispatcher = AgentDispatcher(identity, gossip, transport, escrow)
    manifest = AgentManifest()
    agent = dispatcher.create_agent(VALID_WASM, manifest, ftns_budget=10.0)
    record = await dispatcher.dispatch(agent)

    result_data = {
        "agent_id": agent.agent_id,
        "status": "success",
        "output_b64": "aGVsbG8=",
        "provider_id": "provider-X",
    }

    await dispatcher._on_agent_result("agent_result", result_data, "provider-X")

    assert record.status == DispatchStatus.COMPLETED
    assert record.result == result_data
    escrow.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_bid_timeout_no_bids_refunds():
    """_check_bid_timeout with no bids refunds escrow and fails record."""
    identity = _make_identity()
    gossip = _make_gossip()
    transport = _make_transport()
    escrow = _make_escrow()

    dispatcher = AgentDispatcher(identity, gossip, transport, escrow)
    manifest = AgentManifest()
    agent = dispatcher.create_agent(VALID_WASM, manifest, ftns_budget=10.0)
    record = await dispatcher.dispatch(agent)

    # No bids arrived
    await dispatcher._check_bid_timeout(agent.agent_id)

    assert record.status == DispatchStatus.FAILED
    escrow.refund.assert_awaited_once()


def test_get_record_returns_none_before_dispatch():
    """get_record returns None for unknown agent_id."""
    identity = _make_identity()
    gossip = _make_gossip()
    transport = _make_transport()
    escrow = _make_escrow()

    dispatcher = AgentDispatcher(identity, gossip, transport, escrow)

    assert dispatcher.get_record("nonexistent-agent") is None

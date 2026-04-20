"""Unit tests for ComputeProvider's shard_execute_request handler.

Phase 2 Task 4. Tests the server-side flow: receive request →
acceptance check → execute → sign → respond.

Tensors use float64 to match execute_shard_locally's current dtype
interpretation — tensor_dtype is carried in the payload for future
generality but the executor interprets bytes as float64.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from prsm.compute.shard_receipt import ReceiptOnlyVerification
from prsm.node.compute_provider import ComputeProvider
from prsm.node.identity import generate_node_identity
from prsm.node.transport import MSG_DIRECT, P2PMessage


def _make_provider(max_concurrent_jobs: int = 10):
    identity = generate_node_identity(display_name="provider")
    transport = MagicMock()
    transport.send_to_peer = AsyncMock()
    transport.on_message = MagicMock()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock()
    ledger = MagicMock()

    provider = ComputeProvider(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
        max_concurrent_jobs=max_concurrent_jobs,
    )
    return provider, identity, transport


def _build_shard_request(
    shard_tensor: np.ndarray,
    input_tensor: np.ndarray,
    job_id: str = "job-1",
    shard_index: int = 0,
    deadline_offset_sec: int = 30,
) -> dict:
    tensor_bytes = shard_tensor.astype(np.float64).tobytes()
    checksum = hashlib.sha256(tensor_bytes).hexdigest()
    return {
        "subtype": "shard_execute_request",
        "job_id": job_id,
        "shard_index": shard_index,
        "tensor_data_b64": base64.b64encode(tensor_bytes).decode(),
        "tensor_shape": list(shard_tensor.shape),
        "tensor_dtype": "float64",
        "input_b64": base64.b64encode(input_tensor.astype(np.float64).tobytes()).decode(),
        "input_shape": list(input_tensor.shape),
        "input_dtype": "float64",
        "checksum": checksum,
        "stake_tier": "STANDARD",
        "escrow_tx_id": "esc-test-1",
        "deadline_unix": int(time.time()) + deadline_offset_sec,
        "request_id": "req-test-1",
        "requester_pubkey_b64": "fake-requester-pubkey",
    }


def test_shard_execute_request_happy_path():
    """A valid shard_execute_request produces a completed response with
    a correct output and a signature that verifies."""
    provider, _identity, transport = _make_provider()

    # execute_shard_locally computes tensor @ input_array.
    # tensor (3, 2) @ input (2,) → output (3,).
    shard_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    input_tensor = np.array([1.0, 1.0], dtype=np.float64)
    expected_output = shard_tensor @ input_tensor  # [3.0, 7.0, 11.0]

    payload = _build_shard_request(shard_tensor, input_tensor)
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock()
    peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_execute_request(msg, peer))

    assert transport.send_to_peer.await_count == 1
    call_args = transport.send_to_peer.await_args.args
    sent_peer_id, sent_msg = call_args[0], call_args[1]
    assert sent_peer_id == "peer-req"
    resp = sent_msg.payload

    assert resp["subtype"] == "shard_execute_response"
    assert resp["request_id"] == "req-test-1"
    assert resp["status"] == "completed"
    assert resp["shard_index"] == 0

    output_bytes = base64.b64decode(resp["output_b64"])
    output = np.frombuffer(output_bytes, dtype=np.float64).reshape(resp["output_shape"])
    np.testing.assert_allclose(output, expected_output, rtol=1e-12)

    verifier = ReceiptOnlyVerification()
    assert asyncio.run(verifier.verify(resp["receipt"], output_bytes)) is True


def test_shard_execute_request_declines_when_over_capacity():
    """When _current_jobs >= max_concurrent_jobs → status=declined, no exec."""
    provider, _identity, transport = _make_provider(max_concurrent_jobs=1)
    provider._current_jobs = 1

    shard_tensor = np.array([[1.0]], dtype=np.float64)
    input_tensor = np.array([1.0], dtype=np.float64)
    payload = _build_shard_request(shard_tensor, input_tensor)
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_execute_request(msg, peer))

    assert transport.send_to_peer.await_count == 1
    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["status"] == "declined"
    assert resp.get("error") == "resource_unavailable"
    assert "output_b64" not in resp or resp.get("output_b64") is None


def test_shard_execute_request_declines_past_deadline():
    """deadline_unix < now → declined."""
    provider, _identity, transport = _make_provider()

    shard_tensor = np.array([[1.0]], dtype=np.float64)
    input_tensor = np.array([1.0], dtype=np.float64)
    payload = _build_shard_request(shard_tensor, input_tensor, deadline_offset_sec=-100)
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_execute_request(msg, peer))

    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["status"] == "declined"
    assert resp.get("error") == "deadline_past"


def test_shard_execute_request_declines_bad_checksum():
    """Declared checksum != actual sha256(tensor_bytes) → declined."""
    provider, _identity, transport = _make_provider()

    shard_tensor = np.array([[1.0, 2.0]], dtype=np.float64)
    input_tensor = np.array([1.0], dtype=np.float64)
    payload = _build_shard_request(shard_tensor, input_tensor)
    payload["checksum"] = "0" * 64

    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_execute_request(msg, peer))

    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["status"] == "declined"
    assert resp.get("error") == "checksum_mismatch"


# ── Phase 3 Task 5: preempt_inflight_shards + price-quote handler ──


def _make_marketplace_advertiser(price_ftns: float = 0.05):
    """Minimal stub with just the current_price_ftns() surface that
    _on_shard_price_quote_request consumes."""
    advertiser = MagicMock()
    advertiser.current_price_ftns = MagicMock(return_value=price_ftns)
    return advertiser


def test_preempt_inflight_shards_sends_preempted_to_all():
    """preempt_inflight_shards iterates every in-flight request and
    emits a status=preempted response to the original peer."""
    provider, _identity, transport = _make_provider()

    # Fake three in-flight requests.
    provider._inflight_requests = {
        "req-1": ("peer-A", 0),
        "req-2": ("peer-A", 1),
        "req-3": ("peer-B", 2),
    }

    asyncio.run(provider.preempt_inflight_shards(reason="spot_eviction"))

    assert transport.send_to_peer.await_count == 3
    statuses = [
        call.args[1].payload["status"]
        for call in transport.send_to_peer.await_args_list
    ]
    assert all(s == "preempted" for s in statuses)
    # Reason propagates as the error field.
    errors = [
        call.args[1].payload["error"]
        for call in transport.send_to_peer.await_args_list
    ]
    assert all(e == "spot_eviction" for e in errors)
    # State cleared so a second call is a no-op.
    assert provider._inflight_requests == {}


def test_preempt_inflight_shards_idempotent_when_empty():
    """No in-flight requests → no side effects, no errors."""
    provider, _identity, transport = _make_provider()
    provider._inflight_requests = {}

    asyncio.run(provider.preempt_inflight_shards())

    transport.send_to_peer.assert_not_called()


def test_preempt_inflight_shards_routes_to_correct_peer():
    """Each response goes back to the peer that originally requested
    that shard — critical when different peers requested different shards."""
    provider, _identity, transport = _make_provider()
    provider._inflight_requests = {
        "req-A": ("peer-alpha", 0),
        "req-B": ("peer-beta", 1),
    }

    asyncio.run(provider.preempt_inflight_shards())

    calls_by_peer = {
        call.args[0]: call.args[1].payload
        for call in transport.send_to_peer.await_args_list
    }
    assert set(calls_by_peer.keys()) == {"peer-alpha", "peer-beta"}
    assert calls_by_peer["peer-alpha"]["shard_index"] == 0
    assert calls_by_peer["peer-beta"]["shard_index"] == 1


def test_price_quote_ack_returns_advertised_price():
    """Happy path: advertiser wired, capacity free, shard fits, price
    under ceiling → ack with the advertised price."""
    provider, _identity, transport = _make_provider()
    provider._marketplace_advertiser = _make_marketplace_advertiser(price_ftns=0.05)

    payload = {
        "subtype": "shard_price_quote_request",
        "request_id": "quote-1",
        "listing_id": "listing-X",
        "shard_index": 0,
        "shard_size_bytes": 1024,
        "max_acceptable_price_ftns": 0.10,
    }
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_price_quote_request(msg, peer))

    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["subtype"] == "shard_price_quote_ack"
    assert resp["quoted_price_ftns"] == 0.05
    assert resp["request_id"] == "quote-1"
    assert resp["listing_id"] == "listing-X"
    assert resp["shard_index"] == 0
    assert resp["quote_expires_unix"] > int(time.time())
    assert resp["provider_id"]
    assert resp["signature"]


def test_price_quote_reject_when_no_advertiser():
    """ComputeProvider without a MarketplaceAdvertiser wired responds
    with reject, not ack — it has no price to quote."""
    provider, _identity, transport = _make_provider()
    provider._marketplace_advertiser = None

    payload = {
        "subtype": "shard_price_quote_request",
        "request_id": "quote-1", "listing_id": "listing-X",
        "shard_index": 0, "shard_size_bytes": 1024,
        "max_acceptable_price_ftns": 0.10,
    }
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_price_quote_request(msg, peer))

    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["subtype"] == "shard_price_quote_reject"
    assert resp["reason"] == "no_active_listing"


def test_price_quote_reject_when_at_capacity():
    provider, _identity, transport = _make_provider(max_concurrent_jobs=1)
    provider._marketplace_advertiser = _make_marketplace_advertiser()
    provider._current_jobs = 1  # at capacity

    payload = {
        "subtype": "shard_price_quote_request",
        "request_id": "quote-1", "listing_id": "listing-X",
        "shard_index": 0, "shard_size_bytes": 1024,
        "max_acceptable_price_ftns": 0.10,
    }
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_price_quote_request(msg, peer))

    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["subtype"] == "shard_price_quote_reject"
    assert resp["reason"] == "overloaded"


def test_price_quote_reject_when_shard_too_large():
    provider, _identity, transport = _make_provider()
    provider._marketplace_advertiser = _make_marketplace_advertiser()

    payload = {
        "subtype": "shard_price_quote_request",
        "request_id": "quote-1", "listing_id": "listing-X",
        "shard_index": 0,
        "shard_size_bytes": provider.MAX_SHARD_BYTES + 1,  # over the cap
        "max_acceptable_price_ftns": 0.10,
    }
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_price_quote_request(msg, peer))

    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["subtype"] == "shard_price_quote_reject"
    assert resp["reason"] == "shard_too_large"


def test_price_quote_reject_when_above_ceiling():
    provider, _identity, transport = _make_provider()
    provider._marketplace_advertiser = _make_marketplace_advertiser(price_ftns=0.15)

    payload = {
        "subtype": "shard_price_quote_request",
        "request_id": "quote-1", "listing_id": "listing-X",
        "shard_index": 0, "shard_size_bytes": 1024,
        "max_acceptable_price_ftns": 0.10,  # provider wants 0.15, requester caps at 0.10
    }
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_price_quote_request(msg, peer))

    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["subtype"] == "shard_price_quote_reject"
    assert resp["reason"] == "above_ceiling"


def test_price_quote_router_dispatches_subtype():
    """_on_direct_message must route shard_price_quote_request to the
    new handler, not drop it silently (as it would for an unknown
    subtype)."""
    provider, _identity, transport = _make_provider()
    provider._marketplace_advertiser = _make_marketplace_advertiser()

    payload = {
        "subtype": "shard_price_quote_request",
        "request_id": "quote-1", "listing_id": "listing-X",
        "shard_index": 0, "shard_size_bytes": 1024,
        "max_acceptable_price_ftns": 0.10,
    }
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_direct_message(msg, peer))

    # Response sent — router picked up the subtype.
    assert transport.send_to_peer.await_count == 1
    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["subtype"] == "shard_price_quote_ack"


def test_inflight_requests_tracked_during_shard_execute():
    """While a shard is executing, the request_id appears in
    _inflight_requests; after completion, it's removed."""
    provider, _identity, transport = _make_provider()

    shard_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_tensor = np.array([1.0, 0.0], dtype=np.float64)
    tensor_bytes = shard_tensor.astype(np.float64).tobytes()
    payload = {
        "subtype": "shard_execute_request",
        "job_id": "job-1", "shard_index": 0,
        "tensor_data_b64": base64.b64encode(tensor_bytes).decode(),
        "tensor_shape": list(shard_tensor.shape),
        "tensor_dtype": "float64",
        "input_b64": base64.b64encode(input_tensor.astype(np.float64).tobytes()).decode(),
        "input_shape": list(input_tensor.shape),
        "input_dtype": "float64",
        "checksum": hashlib.sha256(tensor_bytes).hexdigest(),
        "stake_tier": "STANDARD",
        "escrow_tx_id": "esc-1",
        "deadline_unix": int(time.time()) + 30,
        "request_id": "req-flight-1",
        "requester_pubkey_b64": "fake",
    }
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_execute_request(msg, peer))

    # After completion, nothing in flight.
    assert "req-flight-1" not in provider._inflight_requests

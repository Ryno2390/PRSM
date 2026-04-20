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

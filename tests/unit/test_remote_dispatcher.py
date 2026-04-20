"""Unit tests for RemoteShardDispatcher.

Phase 2 Task 5. Exercises happy path, timeout fallback, bad-signature
refund, size limit, retry-on-timeout, peer-not-connected.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from prsm.compute.model_sharding.models import ModelShard, PipelineStakeTier
from prsm.compute.remote_dispatcher import (
    PeerNotConnectedError,
    RemoteShardDispatcher,
    ShardDispatchError,
    ShardTooLargeError,
)
from prsm.compute.shard_receipt import (
    ReceiptOnlyVerification,
    build_receipt_signing_payload,
)
from prsm.node.identity import generate_node_identity
from prsm.node.transport import MSG_DIRECT, P2PMessage


def _make_shard():
    shard_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tensor_bytes = shard_tensor.tobytes()
    shard = ModelShard(
        shard_id="s-0",
        model_id="m-test",
        shard_index=0,
        total_shards=1,
        tensor_data=tensor_bytes,
        tensor_shape=shard_tensor.shape,
        size_bytes=len(tensor_bytes),
        checksum=hashlib.sha256(tensor_bytes).hexdigest(),
    )
    input_tensor = np.array([1.0, 0.0], dtype=np.float64)
    expected_output = shard_tensor @ input_tensor
    return shard, input_tensor, expected_output


def _make_dispatcher(timeout=0.5, max_retries=0, max_shard_bytes=1024, local_fallback=None):
    identity = generate_node_identity(display_name="requester")
    transport = MagicMock()
    peer_stub = MagicMock(); peer_stub.peer_id = "provider-1"
    transport.get_peer = MagicMock(return_value=peer_stub)
    transport.send_to_peer = AsyncMock()
    transport.on_message = MagicMock()

    escrow = MagicMock()
    escrow.create_escrow = AsyncMock(return_value=MagicMock(escrow_id="esc-1"))
    escrow.release_escrow = AsyncMock()
    escrow.refund_escrow = AsyncMock()

    verifier = ReceiptOnlyVerification()

    dispatcher = RemoteShardDispatcher(
        identity=identity,
        transport=transport,
        payment_escrow=escrow,
        verification_strategy=verifier,
        default_timeout=timeout,
        max_retries=max_retries,
        max_shard_bytes=max_shard_bytes,
        local_fallback=local_fallback,
    )
    return dispatcher, identity, transport, escrow


def _provider_response(shard, input_tensor, request_id, job_id="job-1", tamper_sig=False):
    """Build a valid shard_execute_response payload signed by a fresh provider."""
    provider = generate_node_identity(display_name="provider")

    tensor = np.frombuffer(shard.tensor_data, dtype=np.float64).reshape(shard.tensor_shape)
    output = tensor @ input_tensor
    output_bytes = output.tobytes()
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = int(time.time())

    sig_payload = build_receipt_signing_payload(
        job_id=job_id,
        shard_index=shard.shard_index,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    signature = provider.sign(sig_payload)
    if tamper_sig:
        signature = signature[:-4] + ("AAAA" if signature[-4:] != "AAAA" else "BBBB")

    return {
        "subtype": "shard_execute_response",
        "request_id": request_id,
        "status": "completed",
        "shard_index": shard.shard_index,
        "output_b64": base64.b64encode(output_bytes).decode(),
        "output_shape": list(output.shape),
        "output_dtype": "float64",
        "receipt": {
            "job_id": job_id,
            "shard_index": shard.shard_index,
            "provider_id": provider.node_id,
            "provider_pubkey_b64": provider.public_key_b64,
            "output_hash": output_hash,
            "executed_at_unix": executed_at,
            "signature": signature,
        },
    }


def _wire_send_to_respond(transport, dispatcher, shard, input_tensor, tamper_sig=False):
    """Configure transport.send_to_peer so it synchronously creates a task
    to deliver a matching response via dispatcher._on_direct_message."""
    async def send(peer_id, msg):
        request_id = msg.payload["request_id"]
        resp_payload = _provider_response(shard, input_tensor, request_id, tamper_sig=tamper_sig)
        resp_msg = P2PMessage(
            msg_type=MSG_DIRECT, sender_id="provider-1", payload=resp_payload,
        )
        peer = MagicMock(); peer.peer_id = "provider-1"
        asyncio.create_task(dispatcher._on_direct_message(resp_msg, peer))

    transport.send_to_peer.side_effect = send


def test_dispatch_happy_path():
    """Dispatch sends request, receives valid response, releases escrow, returns output."""
    dispatcher, _, transport, escrow = _make_dispatcher()
    shard, input_tensor, expected = _make_shard()
    _wire_send_to_respond(transport, dispatcher, shard, input_tensor)

    result = asyncio.run(dispatcher.dispatch(
        shard=shard, input_tensor=input_tensor,
        node_id="provider-1", job_id="job-1",
        stake_tier=PipelineStakeTier.STANDARD, escrow_amount_ftns=1.0,
    ))

    np.testing.assert_allclose(result, expected, rtol=1e-12)
    escrow.create_escrow.assert_awaited_once()
    escrow.release_escrow.assert_awaited_once()
    escrow.refund_escrow.assert_not_called()


def test_dispatch_timeout_falls_back():
    """Provider doesn't respond; escrow refunded, local_fallback invoked."""
    fallback_output = np.array([999.0, 999.0], dtype=np.float64)
    fallback = MagicMock(return_value=fallback_output)

    dispatcher, _, transport, escrow = _make_dispatcher(
        timeout=0.1, max_retries=0, local_fallback=fallback,
    )
    shard, input_tensor, _ = _make_shard()

    result = asyncio.run(dispatcher.dispatch(
        shard=shard, input_tensor=input_tensor,
        node_id="provider-1", job_id="job-1",
        stake_tier=PipelineStakeTier.STANDARD, escrow_amount_ftns=1.0,
    ))

    np.testing.assert_allclose(result, fallback_output)
    escrow.refund_escrow.assert_awaited_once()
    escrow.release_escrow.assert_not_called()
    fallback.assert_called_once()


def test_dispatch_bad_signature_refunds():
    """A response with an invalid signature causes escrow refund + raise."""
    dispatcher, _, transport, escrow = _make_dispatcher()
    shard, input_tensor, _ = _make_shard()
    _wire_send_to_respond(transport, dispatcher, shard, input_tensor, tamper_sig=True)

    with pytest.raises(ShardDispatchError):
        asyncio.run(dispatcher.dispatch(
            shard=shard, input_tensor=input_tensor,
            node_id="provider-1", job_id="job-1",
            stake_tier=PipelineStakeTier.STANDARD, escrow_amount_ftns=1.0,
        ))

    escrow.refund_escrow.assert_awaited_once()
    escrow.release_escrow.assert_not_called()


def test_dispatch_size_limit_rejects():
    """Shard > max_shard_bytes raises ShardTooLargeError without network."""
    dispatcher, _, transport, escrow = _make_dispatcher(max_shard_bytes=1024)

    big = np.ones((32, 32), dtype=np.float64)   # 32 * 32 * 8 = 8192 bytes
    tensor_bytes = big.tobytes()
    shard = ModelShard(
        shard_id="s-big",
        model_id="m-test",
        shard_index=0,
        total_shards=1,
        tensor_data=tensor_bytes,
        tensor_shape=big.shape,
        size_bytes=len(tensor_bytes),
        checksum=hashlib.sha256(tensor_bytes).hexdigest(),
    )
    input_tensor = np.ones(32, dtype=np.float64)

    with pytest.raises(ShardTooLargeError):
        asyncio.run(dispatcher.dispatch(
            shard=shard, input_tensor=input_tensor,
            node_id="provider-1", job_id="job-1",
            stake_tier=PipelineStakeTier.STANDARD, escrow_amount_ftns=1.0,
        ))

    transport.send_to_peer.assert_not_called()
    escrow.create_escrow.assert_not_called()


def test_dispatch_retry_on_timeout():
    """First response swallowed; retry succeeds; single release fires."""
    dispatcher, _, transport, escrow = _make_dispatcher(
        timeout=0.1, max_retries=1,
    )
    shard, input_tensor, expected = _make_shard()

    call_count = {"n": 0}

    async def send_sometimes(peer_id, msg):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return
        request_id = msg.payload["request_id"]
        resp_payload = _provider_response(shard, input_tensor, request_id)
        resp_msg = P2PMessage(
            msg_type=MSG_DIRECT, sender_id="provider-1", payload=resp_payload,
        )
        peer = MagicMock(); peer.peer_id = "provider-1"
        asyncio.create_task(dispatcher._on_direct_message(resp_msg, peer))

    transport.send_to_peer.side_effect = send_sometimes

    result = asyncio.run(dispatcher.dispatch(
        shard=shard, input_tensor=input_tensor,
        node_id="provider-1", job_id="job-1",
        stake_tier=PipelineStakeTier.STANDARD, escrow_amount_ftns=1.0,
    ))

    np.testing.assert_allclose(result, expected, rtol=1e-12)
    assert escrow.release_escrow.await_count == 1
    assert escrow.refund_escrow.await_count == 0


def test_dispatch_peer_not_connected():
    """If transport.get_peer returns None and no fallback, raise PeerNotConnectedError."""
    dispatcher, _, transport, escrow = _make_dispatcher()
    transport.get_peer = MagicMock(return_value=None)

    shard, input_tensor, _ = _make_shard()

    with pytest.raises(PeerNotConnectedError):
        asyncio.run(dispatcher.dispatch(
            shard=shard, input_tensor=input_tensor,
            node_id="provider-1", job_id="job-1",
            stake_tier=PipelineStakeTier.STANDARD, escrow_amount_ftns=1.0,
        ))

    escrow.create_escrow.assert_not_called()

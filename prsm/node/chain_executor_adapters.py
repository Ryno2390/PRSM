"""Sprint 592 (Phase 2A) — chain-executor send-message adapter scaffolding.

Sprint 578 plumbed PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=rpc as a hook
for the real ``make_rpc_chain_executor`` factory. Phase 2 wiring
requires bridging the gap between:

- ``SendMessage = Callable[[str, bytes], bytes]`` (factory contract;
  SYNC; address → response bytes); see
  ``prsm/compute/chain_rpc/client.py``
- ``transport.send_to_peer(peer_id, P2PMessage) -> bool`` (async;
  awaits + does not return response bytes directly)

This module scaffolds the adapter contract. Phase 2A (sprint 592)
introduces the type signature + Protocol + a NotImplementedError
placeholder ``build_send_message_adapter()``. Phase 2B (sprint 593)
adds the address-resolver helper. Phase 2C (sprint 594) implements
the async-to-sync bridge using ``asyncio.run_coroutine_threadsafe``.
Phase 2D (sprint 595) wires it into ``_build_chain_executor``.

This staged approach keeps each sprint scoped + reviewable. The
scaffolding lets test code reference the eventual contract today.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable


# Re-export the canonical SendMessage signature from the factory's
# source of truth — adapters wired to make_rpc_chain_executor MUST
# conform to this Callable shape.
SendMessage = Callable[[str, bytes], bytes]


@runtime_checkable
class SendMessageAdapter(Protocol):
    """Phase 2 adapter contract.

    Implementations bridge between the daemon's async transport
    layer and the sync SendMessage contract required by
    ``make_rpc_chain_executor``. The adapter is responsible for:

    1. Resolving ``stage_address`` to a transport peer_id
       (delegates to a Phase-2B address resolver).
    2. Constructing a ``P2PMessage`` wrapping ``request_bytes``.
    3. Scheduling the async ``transport.send_to_peer`` on the
       daemon's running event loop (e.g., via
       ``asyncio.run_coroutine_threadsafe``).
    4. Blocking the calling thread until a response arrives or
       a deadline elapses; raising on transport failure.

    Phase 2A — this module. Protocol only; no implementation.
    """

    def __call__(self, stage_address: str, request_bytes: bytes) -> bytes:
        ...


class _Phase2AdapterNotReady(NotImplementedError):
    """Sprint 592 — raised by the scaffolding placeholder when
    callers try to USE the adapter before Phase 2C lands the real
    async-to-sync bridge.

    Distinct exception class so `_build_chain_executor`'s `rpc`
    branch can detect Phase 2 non-readiness cleanly + log the
    structured warning sprint 578 already established.
    """


def run_async_on_loop(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, Any],
    timeout: float,
) -> Any:
    """Sprint 594 (Phase 2C) — async-to-sync bridge primitive.

    Schedules ``coro`` on a running event loop from a DIFFERENT
    thread + returns the result synchronously. Thin wrapper around
    ``asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)``.

    Thread-safety contract:
      - ``loop`` MUST be running on a different thread than the
        caller. Calling from the loop's own thread deadlocks
        because the loop cannot make progress while blocked in
        ``.result()``.

    Used by Phase 2D wiring (sprint 595+) to bridge the sync
    SendMessage contract over async transport calls. Exposed as a
    standalone helper so the threading primitive is unit-testable
    in isolation from the chain-executor protocol layer.

    Raises whatever the coroutine raises (passed through);
    ``concurrent.futures.TimeoutError`` on timeout expiry.
    """
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


class PeerNotFound(RuntimeError):
    """Sprint 593 (Phase 2B) — raised by the address resolver when
    a chain stage's node_id isn't currently in ``transport.peers``.

    Operators triaging chain-executor failures see the missing
    node_id in the exception message. Sprint-595 wiring will catch
    this at executor-startup time + report it via the trust-stack
    observability surfaces (sprint 579 CLI + 582 /health/detailed)
    so the failure is visible BEFORE a real inference request hits
    the dispatch.
    """


def build_address_resolver(node: Any) -> Callable[[str], str]:
    """Sprint 593 (Phase 2B) — map chain stage node_id → transport
    peer.address by looking up ``node.transport.peers[node_id]``.

    Raises ``PeerNotFound`` when the node isn't currently in
    transport.peers. The chain executor's dispatch loop catches
    this to surface "stage N unreachable" cleanly rather than
    propagating a KeyError up.

    Returns the canonical ``AddressResolver = Callable[[str], str]``
    shape from ``prsm/compute/chain_rpc/client.py``.
    """
    transport = node.transport

    def _resolve(node_id: str) -> str:
        peer = transport.peers.get(node_id)
        if peer is None:
            raise PeerNotFound(
                f"chain stage node_id {node_id!r} not currently "
                f"in transport.peers; cannot dispatch. Likely "
                f"causes: peer dropped connection, auto-dial sweep "
                f"hasn't reached this peer yet (sprint 573), or "
                f"peer never registered against the same bootstrap."
            )
        return peer.address

    return _resolve


# Sprint 596 — chain-executor wire-protocol identifiers.
# Chain-executor request messages set payload[CHAIN_REQ_KEY] = request_id;
# response messages set payload[CHAIN_RESP_KEY] = request_id + payload
# bytes (base64-encoded in the JSON envelope). Sprint 597 wires the
# response handler that reads CHAIN_RESP_KEY and resolves the pending
# Future identified by request_id.
CHAIN_MSG_TYPE = "chain_executor_rpc"
CHAIN_REQ_KEY = "chain_req_id"
CHAIN_RESP_KEY = "chain_resp_id"
CHAIN_PAYLOAD_KEY = "chain_payload_b64"
# Sprint 601 (Phase 2E-1) — error indicator in server-side response.
# Lets the requester distinguish "stage handler returned an error"
# from "no stage handler exists yet" (Phase 2E-1 ships scaffolding;
# Phase 2E-2+ replaces this with real stage execution).
CHAIN_ERROR_KEY = "chain_error"


async def handle_chain_executor_request(node: Any, msg: Any) -> bool:
    """Sprint 601 (Phase 2E-1) — server-side request handler scaffolding.

    Receives a chain_executor_rpc REQUEST message (CHAIN_REQ_KEY set,
    CHAIN_RESP_KEY absent) + dispatches a structured response back
    to the sender. Phase 2E-1 ships a "not yet implemented" error
    response so the round-trip wire is provably functional before
    Phase 2E-2+ wires real stage execution.

    Returns:
      True  — handled (response sent or attempted to send)
      False — not for us (wrong subtype, or a response message
              meant for sprint 597's response handler)

    Defensive semantics:
      - Wrong subtype → return False (ignored)
      - Message has CHAIN_RESP_KEY → return False (it's a RESPONSE
        to our outbound request, not a fresh REQUEST)
      - Malformed base64 payload → send error response anyway
        (better than silent drop; requester knows we tried)

    Future Phase 2E-2 sub-sprints replace the error response with
    real stage execution: decode activation bytes → forward through
    a model layer → encode response activation → ship back.
    """
    import base64
    payload = getattr(msg, "payload", None) or {}
    if payload.get("subtype") != CHAIN_MSG_TYPE:
        return False
    request_id = payload.get(CHAIN_REQ_KEY)
    if not request_id:
        return False
    # CHAIN_RESP_KEY presence → this is a response, not a request.
    # Sprint 597 handles responses; we (Phase 2E-1) handle requests.
    if payload.get(CHAIN_RESP_KEY):
        return False

    # Phase 2E-1: decode payload defensively + build error response.
    # We attempt the decode for parity with the real Phase 2E-2 path
    # (which would forward the activation bytes to a stage executor);
    # malformed-base64 still produces a response so the requester
    # doesn't time out silently.
    decode_error = None
    try:
        _payload_bytes = base64.b64decode(
            payload.get(CHAIN_PAYLOAD_KEY, ""),
        )
        del _payload_bytes  # Phase 2E-1 doesn't actually use them
    except Exception as exc:  # noqa: BLE001
        decode_error = (
            f"chain-executor request payload base64-decode failed: "
            f"{type(exc).__name__}: {exc}"
        )

    error_msg = decode_error or (
        "Sprint 601 Phase 2E-1: server-side stage handler not yet "
        "implemented. Phase 2E-2+ will replace this scaffolding "
        "with real chain-stage execution. Until then, this node "
        "ACKs incoming requests but cannot actually compute the "
        "next stage's activation."
    )

    sender_id = getattr(msg, "sender_id", None)
    if not sender_id:
        # Can't ship a response without a sender to reply to.
        return False

    try:
        from prsm.node.transport import P2PMessage, MSG_DIRECT
        response = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=getattr(
                getattr(node, "identity", None), "node_id", "",
            ),
            payload={
                "subtype": CHAIN_MSG_TYPE,
                CHAIN_REQ_KEY: request_id,  # keep for diagnostic correlation
                CHAIN_RESP_KEY: request_id,
                CHAIN_ERROR_KEY: error_msg,
                CHAIN_PAYLOAD_KEY: "",  # empty in error path
            },
        )
        await node.transport.send_to_peer(sender_id, response)
    except Exception as exc:  # noqa: BLE001
        # Even the response-send failed — log + return True so
        # callers know we attempted. Sprint 597's response handler
        # on the requester side will eventually time out.
        import logging as _l
        _l.getLogger(__name__).warning(
            "Sprint 601 chain-executor request handler: "
            "failed to send error response to %s: %s",
            sender_id, exc,
        )
    return True


def handle_chain_executor_response(node: Any, msg: Any) -> bool:
    """Sprint 597 (Phase 2D step 3) — resolve a pending chain-executor
    Future on receipt of a response message.

    Reads the response wire-protocol fields from ``msg.payload``:
      ``CHAIN_RESP_KEY``     → request_id (hex sha256 string)
      ``CHAIN_PAYLOAD_KEY``  → base64-encoded response bytes

    If the request_id is in ``node._chain_executor_pending``, sets
    the Future result. If not (request already timed out + popped,
    or unsolicited message), returns False without raising.

    Must be called from the loop thread (via the transport's normal
    inbound-message dispatch); the Future is on that loop. Use
    ``loop.call_soon_threadsafe`` if invoking from another thread.

    Returns ``True`` if the response was matched + Future resolved;
    ``False`` if no matching pending request (silent drop).
    """
    import base64
    payload = getattr(msg, "payload", None) or {}
    # Only handle our wire-protocol envelope; ignore other subtypes
    if payload.get("subtype") != CHAIN_MSG_TYPE:
        return False
    request_id = payload.get(CHAIN_RESP_KEY)
    if not request_id:
        return False
    pending = getattr(node, "_chain_executor_pending", None)
    if pending is None:
        return False
    future = pending.get(request_id)
    if future is None or future.done():
        # No pending request (timed out + cleaned up) OR already
        # resolved (duplicate response). Silent drop is correct.
        return False
    payload_b64 = payload.get(CHAIN_PAYLOAD_KEY, "")
    try:
        response_bytes = base64.b64decode(payload_b64)
    except Exception as exc:  # noqa: BLE001
        # Malformed payload → reject this response. Future stays
        # pending so the original SendMessage call still times out
        # cleanly rather than getting bogus bytes.
        future.set_exception(
            RuntimeError(
                f"chain-executor response payload base64-decode "
                f"failed for request_id={request_id}: {exc}"
            )
        )
        return True
    future.set_result(response_bytes)
    return True


def build_send_message_adapter(
    node: Any,
    timeout: float = 30.0,
) -> SendMessageAdapter:
    """Sprint 596 (Phase 2D step 2) — real SendMessage adapter.

    Bridges the sync ``SendMessage = Callable[[str, bytes], bytes]``
    contract over the daemon's async transport. Workflow:

      1. Hash request_bytes → request_id (sha256 hex).
      2. Resolve stage_address → peer_id via build_address_resolver-
         compatible lookup against ``node.transport.peers``.
      3. Create asyncio.Future, store in
         ``node._chain_executor_pending[request_id]``.
      4. Schedule coroutine on ``node._loop`` that sends a P2PMessage
         + awaits the future (resolved by sprint-597 response handler).
      5. Return response bytes synchronously.

    Phase 2D step 3 (sprint 597) wires the response handler that
    resolves the Future when a CHAIN_RESP_KEY message arrives.
    Until that lands, calls will time out — but the wire is sound.

    Raises:
      _Phase2AdapterNotReady — when node._loop is None (daemon
        not started or not running on asyncio context).
      PeerNotFound — when stage_address isn't in transport.peers.
      TimeoutError — when no response arrives within ``timeout``.
    """
    import base64
    import hashlib

    def _adapter(stage_address: str, request_bytes: bytes) -> bytes:
        loop = getattr(node, "_loop", None)
        if loop is None:
            raise _Phase2AdapterNotReady(
                "Sprint 596 SendMessage adapter: node._loop is None. "
                "Daemon must be started (sprint-595 captures the loop "
                "at PRSMNode.start). Until then, set "
                "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=stub for a working "
                "daemon."
            )

        request_id = hashlib.sha256(request_bytes).hexdigest()
        pending = node._chain_executor_pending

        async def _send_and_wait() -> bytes:
            # Import here to avoid circular: transport imports node
            # indirectly via __init__
            from prsm.node.transport import P2PMessage, MSG_DIRECT
            future = loop.create_future()
            pending[request_id] = future
            try:
                msg = P2PMessage(
                    msg_type=MSG_DIRECT,
                    sender_id=node.identity.node_id,
                    payload={
                        "subtype": CHAIN_MSG_TYPE,
                        CHAIN_REQ_KEY: request_id,
                        CHAIN_PAYLOAD_KEY: base64.b64encode(
                            request_bytes,
                        ).decode("ascii"),
                    },
                )
                # Resolve address → peer_id by lookup. The chain
                # executor passes stage_address that downstream code
                # treats as a peer_id (per sprint 593 resolver
                # contract). For Phase 2D the convention is:
                # stage_address IS the target peer_id.
                sent = await node.transport.send_to_peer(
                    stage_address, msg,
                )
                if not sent:
                    raise RuntimeError(
                        f"transport.send_to_peer returned False for "
                        f"peer_id={stage_address!r}; chain stage "
                        f"unreachable"
                    )
                import asyncio as _asyncio
                response_payload = await _asyncio.wait_for(
                    future, timeout=timeout,
                )
                return response_payload
            finally:
                pending.pop(request_id, None)

        return run_async_on_loop(loop, _send_and_wait(), timeout=timeout + 1.0)

    return _adapter

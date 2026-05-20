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

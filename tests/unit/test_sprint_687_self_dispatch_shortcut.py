"""Sprint 687 — F34 fix: self-dispatch shortcut in send_message adapter.

Live-attest of sprint 686 surfaced F34: when Phase-1 allocation
routes a chain stage to the SAME node that's serving the request,
``build_send_message_adapter`` tries to look up the stage's
node_id in ``transport.peers``. But ``transport.peers`` only
holds REMOTE peer connections — never self. The dispatch fails
with "chain stage node_id X not currently in transport.peers".

Sprint 687 detects ``stage_address == node.identity.node_id`` and
executes the stage LOCALLY via the same StageExecutor that the
server-side ``handle_chain_executor_request`` uses. No network
hop, no transport.peers lookup, no auto-dial-sweep dependency.

This is a real distributed-systems requirement, not a workaround:
in a 2-node deployment serving its own requests, one stage of the
chain WILL be self-hosted. The network protocol must accommodate
that without forcing operators to deploy a 3rd node just to
dispatch.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_node_with_loop(self_node_id: str = "a" * 32):
    """Build a fake node container with the minimum surface
    build_send_message_adapter touches: _loop, identity, transport,
    _chain_executor_pending."""
    loop = asyncio.new_event_loop()
    node = MagicMock()
    node._loop = loop
    node.identity.node_id = self_node_id
    node.transport.peers = {}  # empty — self isn't in here
    node._chain_executor_pending = {}
    return node, loop


def test_self_dispatch_short_circuits_to_local_executor():
    """stage_address == node.identity.node_id → call local
    StageExecutor.execute() directly; NEVER touch transport."""
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )
    self_id = "a" * 32
    node, loop = _make_node_with_loop(self_node_id=self_id)
    # Inject a local stage executor that returns deterministic bytes.

    class _Echo:
        calls = []
        async def execute(self, request_bytes: bytes) -> bytes:
            self.calls.append(request_bytes)
            return b"local-result:" + request_bytes

    echo = _Echo()
    node._chain_stage_executor = echo
    node.transport.send_to_peer = AsyncMock(
        side_effect=AssertionError(
            "self-dispatch must NOT touch transport.send_to_peer"
        ),
    )

    send = build_send_message_adapter(node)
    try:
        # Run on a thread since adapter blocks on the future.
        import threading
        result_holder = []
        err_holder = []
        def _run():
            try:
                # Drive the loop on this thread; adapter uses
                # run_coroutine_threadsafe so the loop must be
                # running.
                pass
            except Exception as e:  # noqa: BLE001
                err_holder.append(e)

        # Spin the loop in a worker thread so the adapter's
        # run_coroutine_threadsafe has something to schedule onto.
        def _spin():
            asyncio.set_event_loop(loop)
            loop.run_forever()
        spinner = threading.Thread(target=_spin, daemon=True)
        spinner.start()

        # Call adapter from the main thread (sync call).
        result = send(self_id, b"req-bytes")
        assert result == b"local-result:req-bytes"
        assert echo.calls == [b"req-bytes"]
        # send_to_peer must not have been called for self-dispatch.
        node.transport.send_to_peer.assert_not_called()
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_remote_dispatch_still_uses_transport(monkeypatch):
    """stage_address != self → falls through to normal transport
    path. Pin against accidental self-dispatch over-broadening."""
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )
    self_id = "a" * 32
    remote_id = "b" * 32
    node, loop = _make_node_with_loop(self_node_id=self_id)
    # Pre-populate transport.peers for the remote
    peer = MagicMock()
    peer.address = "1.2.3.4:9001"
    node.transport.peers[remote_id] = peer
    # send_to_peer fails immediately so we don't have to round-trip
    # a real response; we just want to verify it WAS called for the
    # non-self stage_address.
    node.transport.send_to_peer = AsyncMock(return_value=False)

    send = build_send_message_adapter(node)
    import threading
    def _spin():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    spinner = threading.Thread(target=_spin, daemon=True)
    spinner.start()
    try:
        # Will raise because the send returns False + can't redial,
        # but the assertion we care about is that send_to_peer was
        # attempted.
        try:
            send(remote_id, b"req-bytes")
        except Exception:
            pass
        assert node.transport.send_to_peer.called
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_self_dispatch_raises_when_no_local_executor():
    """No node._chain_stage_executor AND no env opt-in →
    _build_stage_executor_from_env returns the stub that raises
    StageExecutionError. Caller sees the executor error, not a
    transport.peers lookup error — which is the right diagnostic."""
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter, StageExecutionError,
    )
    self_id = "a" * 32
    node, loop = _make_node_with_loop(self_node_id=self_id)
    # Don't set _chain_stage_executor — falls to env. Default env
    # is empty → stub executor.
    node._chain_stage_executor = None
    node.transport.send_to_peer = AsyncMock()

    send = build_send_message_adapter(node)
    import threading
    def _spin():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    spinner = threading.Thread(target=_spin, daemon=True)
    spinner.start()
    try:
        with pytest.raises((StageExecutionError, RuntimeError)) as exc_info:
            send(self_id, b"req-bytes")
        # The error mentions stage execution, NOT transport.peers
        assert "transport.peers" not in str(exc_info.value)
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_address_resolver_short_circuits_for_self():
    """build_address_resolver must NOT raise PeerNotFound when
    asked to resolve self's own node_id. Live-attest of sprint 687
    surfaced that the resolver raises BEFORE the send_message
    adapter ever runs — so the send_message-side fix alone is
    insufficient. The resolver returns the node_id unchanged for
    self, signaling the send adapter to short-circuit."""
    from prsm.node.chain_executor_adapters import (
        build_address_resolver, PeerNotFound,
    )
    self_id = "a" * 32
    node = MagicMock()
    node.identity.node_id = self_id
    node.transport.peers = {}  # empty — self NEVER appears here
    resolver = build_address_resolver(node)
    # Must not raise:
    assert resolver(self_id) == self_id


def test_address_resolver_still_raises_for_unknown_remote():
    """Remote node_id absent from transport.peers → still raises
    PeerNotFound (pin against over-broadening the self-shortcut)."""
    from prsm.node.chain_executor_adapters import (
        build_address_resolver, PeerNotFound,
    )
    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.transport.peers = {}
    resolver = build_address_resolver(node)
    with pytest.raises(PeerNotFound):
        resolver("b" * 32)  # unknown remote


def test_parallax_executor_runs_chain_off_event_loop():
    """Sprint 687 F35 pin: parallax_executor.execute MUST call
    chain_executor.execute_chain via run_in_executor, not
    synchronously. The sync call blocks the event loop, which
    deadlocks the sprint 687 self-dispatch path (its scheduled
    coroutine can't run because the loop is blocked waiting for
    execute_chain to return).

    Live-attest of sprint 687 surfaced this — py-spy showed the
    MainThread stuck in future.result(), with execute_chain on
    the stack between FastAPI's async handler and the adapter."""
    import inspect
    from prsm.compute.inference import parallax_executor
    src = inspect.getsource(parallax_executor.ParallaxScheduledExecutor.execute)
    assert "run_in_executor" in src, (
        "parallax_executor.execute must wrap execute_chain in "
        "run_in_executor to avoid deadlock with self-dispatch "
        "(sprint 687 F35)"
    )
    # And the run_in_executor call must be awaited (not fire-and-
    # forget). Look for "await loop.run_in_executor" or equivalent
    # tight binding.
    assert "await " in src and "run_in_executor" in src
    # Pin the exact form so a future refactor that splits them
    # doesn't accidentally drop the await.
    assert (
        "await loop.run_in_executor" in src
        or "await asyncio.get_event_loop().run_in_executor" in src
    )


def test_self_dispatch_path_in_source_guard():
    """Pin against a refactor that removes the sprint-687 self-
    dispatch shortcut. The adapter must contain the self-id check."""
    import inspect
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
    )
    src = inspect.getsource(build_send_message_adapter)
    assert "self_node_id" in src or "identity.node_id" in src, (
        "send_message adapter must compare stage_address against "
        "node.identity.node_id (sprint 687 F34)"
    )
    assert "Sprint 687" in src or "sprint 687" in src

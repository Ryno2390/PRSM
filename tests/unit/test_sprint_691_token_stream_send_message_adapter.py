"""Sprint 691 F40 fix — token_stream_send_message adapter.

Sprint 689 closed F39 (decorator passthrough) but surfaced F40:
make_rpc_chain_executor needs a `token_stream_send_message`
callable to dispatch streaming requests, and the PRSM-side wiring
never provided one. Sprint 691 ships `build_token_stream_send_
message_adapter(node)`:

  - Self-dispatch (stage_address == node.identity.node_id):
    call the local LayerStageServer's handle_token_stream and
    yield each frame. Works whenever PRSM_PARALLAX_STAGE_EXECUTOR_
    KIND=layer_stage + the runner exposes
    run_layer_slice_streaming.

  - Remote (stage_address != self): raise structured
    "remote streaming not yet wired" RuntimeError. Sprint 692+
    will ship the P2P streaming transport. The clear error means
    operators see the gap immediately instead of opaque hangs.

  - Local executor lacks ._server.handle_token_stream: raise
    StageExecutionError naming the gap (operator config issue,
    not a wiring bug).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_node(self_node_id: str = "a" * 32):
    """Build a minimal node container for adapter wiring."""
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = self_node_id
    return node


def test_remote_dispatch_raises_not_yet_wired():
    """stage_address != self → clear RuntimeError naming the gap."""
    from prsm.node.chain_executor_adapters import (
        build_token_stream_send_message_adapter,
    )
    node = _make_node()
    adapter = build_token_stream_send_message_adapter(node)
    with pytest.raises(RuntimeError, match="remote streaming"):
        # Generator only raises on first next() — consume it.
        list(adapter("b" * 32, b"req"))


def test_self_dispatch_with_streaming_server_yields_frames():
    """Self-dispatch + executor._server has handle_token_stream →
    adapter yields each frame from the server's streaming output."""
    from prsm.node.chain_executor_adapters import (
        build_token_stream_send_message_adapter,
    )
    node = _make_node()
    server = MagicMock()
    server.handle_token_stream = MagicMock(
        return_value=iter([b"frame1", b"frame2", b"final"]),
    )
    stage_executor = MagicMock()
    stage_executor._server = server
    node._chain_stage_executor = stage_executor

    adapter = build_token_stream_send_message_adapter(node)
    frames = list(adapter("a" * 32, b"req-bytes"))
    assert frames == [b"frame1", b"frame2", b"final"]
    server.handle_token_stream.assert_called_once_with(b"req-bytes")


def test_self_dispatch_raises_when_server_lacks_streaming():
    """Self-dispatch + executor._server lacks handle_token_stream →
    StageExecutionError naming the gap (operator config issue)."""
    from prsm.node.chain_executor_adapters import (
        build_token_stream_send_message_adapter,
        StageExecutionError,
    )
    node = _make_node()
    # Server with NO handle_token_stream attribute
    server = MagicMock(spec=["handle"])  # unary only
    stage_executor = MagicMock()
    stage_executor._server = server
    node._chain_stage_executor = stage_executor

    adapter = build_token_stream_send_message_adapter(node)
    with pytest.raises(StageExecutionError, match="handle_token_stream"):
        list(adapter("a" * 32, b"req-bytes"))


def test_adapter_is_callable_with_correct_signature():
    """Returned adapter must match TokenStreamSendMessage Protocol:
    Callable[[str, bytes], Iterable[bytes]]."""
    from prsm.node.chain_executor_adapters import (
        build_token_stream_send_message_adapter,
    )
    node = _make_node()
    adapter = build_token_stream_send_message_adapter(node)
    assert callable(adapter)


def test_make_rpc_chain_executor_construction_wires_token_stream():
    """Pin: _build_chain_executor's rpc kind must pass
    token_stream_send_message= to make_rpc_chain_executor. Without
    this, /compute/inference/stream returns the F40
    "STREAMING_NOT_WIRED" error even after sprint 691 ships the
    adapter."""
    import inspect
    from prsm.node import inference_wiring
    src = inspect.getsource(inference_wiring._build_chain_executor)
    assert "token_stream_send_message" in src, (
        "_build_chain_executor must pass token_stream_send_message= "
        "to make_rpc_chain_executor (sprint 691)"
    )
    assert "build_token_stream_send_message_adapter" in src

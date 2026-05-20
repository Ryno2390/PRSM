"""Sprint 606 (Phase 2F-1) — LayerStageServer adapter as StageExecutor.

The existing ``LayerStageServer.handle(request_bytes) -> bytes``
(prsm/compute/chain_rpc/server.py:385) ALREADY has the exact
signature our StageExecutor Protocol expects (sprint 602). It just
needs an async wrapper to fit the Protocol.

Sprint 606 ships ``LayerStageServerStageExecutor`` — a thin async
adapter that delegates ``execute(bytes)`` to ``server.handle(bytes)``.
Phase 2F-2 (sprint 607+) will add the factory that constructs the
underlying LayerStageServer with the operator's identity + registry
+ runner + tee_runtime + anchor.

Tests:
  - Adapter implements StageExecutor Protocol
  - Async execute() returns whatever sync handle() returned
  - Exceptions from handle() propagate via StageExecutionError
    (since LayerStageServer.handle should NEVER raise, this is
     defense-in-depth — wraps unexpected runtime errors)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


def test_module_exposes_layer_stage_executor():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "LayerStageServerStageExecutor")


def test_adapter_implements_stage_executor_protocol():
    from prsm.node.chain_executor_adapters import (
        LayerStageServerStageExecutor, StageExecutor,
    )
    fake_server = MagicMock()
    fake_server.handle = MagicMock(return_value=b"x")
    adapter = LayerStageServerStageExecutor(server=fake_server)
    assert isinstance(adapter, StageExecutor)


@pytest.mark.asyncio
async def test_adapter_returns_handle_result():
    """execute(req_bytes) calls server.handle(req_bytes) + returns its bytes."""
    from prsm.node.chain_executor_adapters import (
        LayerStageServerStageExecutor,
    )
    fake_server = MagicMock()
    fake_server.handle = MagicMock(return_value=b"response-from-handle")
    adapter = LayerStageServerStageExecutor(server=fake_server)
    result = await adapter.execute(b"request-payload")
    assert result == b"response-from-handle"
    fake_server.handle.assert_called_once_with(b"request-payload")


@pytest.mark.asyncio
async def test_adapter_wraps_handle_raise_as_stage_execution_error():
    """LayerStageServer.handle SHOULD never raise (it returns structured
    error bytes), but defense-in-depth: if it does, the adapter wraps
    in StageExecutionError so the request-handler surfaces it cleanly.
    """
    from prsm.node.chain_executor_adapters import (
        LayerStageServerStageExecutor, StageExecutionError,
    )
    fake_server = MagicMock()
    fake_server.handle = MagicMock(
        side_effect=RuntimeError("unexpected handle crash"),
    )
    adapter = LayerStageServerStageExecutor(server=fake_server)
    with pytest.raises(StageExecutionError) as exc_info:
        await adapter.execute(b"req")
    assert "unexpected handle crash" in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_runs_handle_in_executor_thread():
    """server.handle is synchronous + may do CPU-heavy forward pass.
    The adapter MUST dispatch via loop.run_in_executor so it doesn't
    block the event loop thread.
    """
    import inspect
    from prsm.node.chain_executor_adapters import (
        LayerStageServerStageExecutor,
    )
    src = inspect.getsource(LayerStageServerStageExecutor)
    assert "run_in_executor" in src, (
        "Sprint 606: handle() is sync + potentially CPU-heavy; "
        "adapter must run_in_executor so it doesn't block the "
        "event loop thread."
    )

"""Sprint 602 (Phase 2E-2) — StageExecutor protocol + stub.

The server-side request handler (sprint 601) currently returns a
hardcoded "not yet implemented" error. Phase 2E-2 introduces the
abstraction layer: a StageExecutor Protocol that the handler will
delegate to in Phase 2E-4.

Contract:
  async def execute(request_bytes: bytes) -> bytes:
      '''Run a chain stage forward + return response bytes.

      Raises StageExecutionError on failure.
      '''

Sprint 602 ships:
  - StageExecutor Protocol (runtime-checkable)
  - StageExecutionError typed exception
  - build_stub_stage_executor() — placeholder that raises
    StageExecutionError with "not wired" message
"""
from __future__ import annotations

import pytest


def test_module_exposes_stage_executor_symbols():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "StageExecutor")
    assert hasattr(m, "StageExecutionError")
    assert hasattr(m, "build_stub_stage_executor")


def test_stage_executor_is_runtime_checkable_protocol():
    from prsm.node.chain_executor_adapters import StageExecutor

    class _FakeStageExec:
        async def execute(self, request_bytes: bytes) -> bytes:
            return b"stub"

    assert isinstance(_FakeStageExec(), StageExecutor)


def test_stage_executor_protocol_rejects_objects_without_execute():
    from prsm.node.chain_executor_adapters import StageExecutor

    class _NoExecute:
        pass

    assert not isinstance(_NoExecute(), StageExecutor)


def test_stage_execution_error_is_runtime_error():
    """Operators catching generic transport/RPC errors should
    naturally catch StageExecutionError via RuntimeError isinstance.
    """
    from prsm.node.chain_executor_adapters import StageExecutionError
    assert issubclass(StageExecutionError, RuntimeError)


def test_stub_stage_executor_implements_protocol():
    from prsm.node.chain_executor_adapters import (
        build_stub_stage_executor, StageExecutor,
    )
    exe = build_stub_stage_executor()
    assert isinstance(exe, StageExecutor)


@pytest.mark.asyncio
async def test_stub_stage_executor_raises_with_actionable_message():
    """Phase 2E-2 stub: execute() raises StageExecutionError with
    "not yet wired" message + hint for operators.
    """
    from prsm.node.chain_executor_adapters import (
        build_stub_stage_executor, StageExecutionError,
    )
    exe = build_stub_stage_executor()
    with pytest.raises(StageExecutionError) as exc_info:
        await exe.execute(b"sample request")
    msg = str(exc_info.value).lower()
    # Mentions Phase 2E-3 / not yet implemented / not wired
    assert (
        "phase 2e" in msg
        or "not yet implemented" in msg
        or "stub" in msg
    )

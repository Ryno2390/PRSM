"""Sprint 603 (Phase 2E-3) — EchoStageExecutor.

A StageExecutor implementation that returns the request bytes
UNCHANGED. Not real chain execution (no model forward pass), but
makes the full round-trip wire (Mac client → droplet server → echo
→ Mac receives) testable end-to-end without model integration.

Phase 2E-4 (sprint 604) wires this into handle_chain_executor_request
behind an env var so operators can opt into echo-mode for fleet
diagnostics; default stays at sprint-602 stub (which raises) so
production-like behavior isn't silently masked.

Phase 2F+ (future) introduces real-model StageExecutor variants.
"""
from __future__ import annotations

import pytest


def test_module_exposes_echo_stage_executor():
    from prsm.node import chain_executor_adapters as m
    assert hasattr(m, "build_echo_stage_executor")


def test_echo_executor_implements_protocol():
    from prsm.node.chain_executor_adapters import (
        build_echo_stage_executor, StageExecutor,
    )
    exe = build_echo_stage_executor()
    assert isinstance(exe, StageExecutor)


@pytest.mark.asyncio
async def test_echo_executor_returns_input_bytes_unchanged():
    from prsm.node.chain_executor_adapters import build_echo_stage_executor
    exe = build_echo_stage_executor()
    result = await exe.execute(b"hello world")
    assert result == b"hello world"


@pytest.mark.asyncio
async def test_echo_executor_handles_empty_bytes():
    from prsm.node.chain_executor_adapters import build_echo_stage_executor
    exe = build_echo_stage_executor()
    assert await exe.execute(b"") == b""


@pytest.mark.asyncio
async def test_echo_executor_handles_large_payload():
    """Verify the executor doesn't truncate or buffer-limit."""
    from prsm.node.chain_executor_adapters import build_echo_stage_executor
    exe = build_echo_stage_executor()
    payload = b"x" * (256 * 1024)  # 256 KB
    assert await exe.execute(payload) == payload

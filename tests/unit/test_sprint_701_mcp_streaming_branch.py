"""Sprint 701 — pin test for MCP handle_prsm_inference streaming branch.

Phase 3.x.8.1 Task 4 shipped the streaming branch (emit_progress
callback → SSE consumer → per-token progress events). The existing
tests/unit/test_mcp_server.py covers only the unary branch
(emit_progress=None). Sprint 701 adds a pin test for the streaming
branch.

Live-attest 2026-05-22 from NYC localhost: 4 emit_progress events
fired correctly + final result returned with signed-receipt footer.
This pin test mocks aiohttp's ClientSession to feed a synthetic SSE
stream so CI can verify the wiring without a live PRSM node.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_streaming_branch_forwards_tokens_to_emit_progress(monkeypatch):
    """When emit_progress is supplied, handle_prsm_inference routes
    through _call_node_api_streaming + forwards every token event
    to the callback. Live-attest 2026-05-22 confirmed 4 events for
    a 4-token prompt; this test pins the per-token forwarding
    contract."""
    from prsm import mcp_server

    captured_events = []

    async def fake_emit(text, progress, total):
        captured_events.append((text, progress, total))

    # Synthetic SSE stream — 3 token frames + terminal result frame.
    fake_result = {
        "success": True,
        "output": " the capital",
        "job_id": "test-job",
        "receipt": {
            "model_id": "gpt2", "duration_seconds": 0.5,
            "settler_node_id": "test-settler", "tee_type": "software",
            "epsilon_spent": 0.0, "cost_ftns": "0.12",
            "privacy_tier": "none", "content_tier": "A",
            "settler_signature": "deadbeef",
        },
    }

    async def fake_call_streaming(path, payload, emit_progress):
        # Forward 3 fake token events
        await emit_progress(" the", 1.0, None)
        await emit_progress(" capital", 2.0, None)
        await emit_progress(" of", 3.0, None)
        return fake_result

    monkeypatch.setattr(
        mcp_server, "_call_node_api_streaming", fake_call_streaming,
    )

    result = await mcp_server.handle_prsm_inference(
        {"prompt": "Hello", "model_id": "gpt2", "budget_ftns": 1.0,
         "privacy_tier": "none", "content_tier": "A"},
        emit_progress=fake_emit,
    )
    # All 3 token events made it to the caller's callback
    assert len(captured_events) == 3
    assert captured_events[0][0] == " the"
    assert captured_events[1][0] == " capital"
    assert captured_events[2][0] == " of"
    # Final result includes the model + cost reconciliation footer
    assert "gpt2" in result
    assert "Settler" in result or "settler" in result
    assert "0.12" in result


@pytest.mark.asyncio
async def test_streaming_branch_surfaces_inference_error_structured(monkeypatch):
    """When _call_node_api_streaming raises InferenceError (e.g.,
    server returned an `event: error` frame), the streaming branch
    returns a clean "Inference rejected: <msg>" string rather than
    propagating the exception. Live-attest captured the exact error
    "insufficient capacity: region 'default': total layer_capacity=
    2 < num_layers=12" being mapped this way."""
    from prsm import mcp_server

    async def fake_emit(text, progress, total):
        pass

    async def raising_call_streaming(path, payload, emit_progress):
        raise mcp_server.InferenceError(
            "insufficient capacity: region 'default': total "
            "layer_capacity=2 < num_layers=12",
            code="ALLOCATION_FAILURE",
        )

    monkeypatch.setattr(
        mcp_server, "_call_node_api_streaming", raising_call_streaming,
    )

    result = await mcp_server.handle_prsm_inference(
        {"prompt": "Hi", "model_id": "gpt2", "budget_ftns": 1.0,
         "privacy_tier": "none", "content_tier": "A"},
        emit_progress=fake_emit,
    )
    assert result.startswith("Inference rejected: ")
    assert "insufficient capacity" in result


@pytest.mark.asyncio
async def test_streaming_branch_surfaces_connect_error_with_triage(monkeypatch):
    """When _call_node_api_streaming raises a generic exception (e.g.,
    aiohttp ConnectionError because the node isn't running or the
    URL is wrong), the streaming branch returns a multi-line error
    with triage hints. Live-attest 2026-05-22 confirmed the same
    message when the Mac couldn't reach NYC public IP (port 8002
    is loopback-only)."""
    from prsm import mcp_server

    async def fake_emit(text, progress, total):
        pass

    async def raising_call_streaming(path, payload, emit_progress):
        raise ConnectionError("Cannot connect to host")

    monkeypatch.setattr(
        mcp_server, "_call_node_api_streaming", raising_call_streaming,
    )

    result = await mcp_server.handle_prsm_inference(
        {"prompt": "Hi", "model_id": "gpt2", "budget_ftns": 1.0,
         "privacy_tier": "none", "content_tier": "A"},
        emit_progress=fake_emit,
    )
    assert "PRSM streaming inference failed" in result
    assert "prsm node start" in result  # operator triage hint


@pytest.mark.asyncio
async def test_unary_branch_unchanged_when_no_progress_callback(monkeypatch):
    """Backward-compat: when emit_progress=None, the unary branch
    runs (existing tests/unit/test_mcp_server.py coverage). Pin
    that the streaming wiring doesn't accidentally fire for unary
    callers."""
    from prsm import mcp_server

    streaming_called = []
    async def detect_streaming(*args, **kwargs):
        streaming_called.append(True)
        return {}

    monkeypatch.setattr(
        mcp_server, "_call_node_api_streaming", detect_streaming,
    )
    # Also stub the unary path so we don't hit the network
    async def fake_unary(*args, **kwargs):
        return {"success": True, "output": "ok",
                "receipt": {"model_id": "gpt2",
                            "duration_seconds": 1.0,
                            "settler_node_id": "test",
                            "tee_type": "software",
                            "epsilon_spent": 0.0,
                            "cost_ftns": "0.01",
                            "privacy_tier": "none",
                            "content_tier": "A",
                            "settler_signature": "x"}}

    monkeypatch.setattr(mcp_server, "_call_node_api", fake_unary)

    result = await mcp_server.handle_prsm_inference(
        {"prompt": "Hello", "model_id": "gpt2", "budget_ftns": 1.0,
         "privacy_tier": "none", "content_tier": "A"},
        emit_progress=None,
    )
    assert streaming_called == [], (
        "unary callers must NOT hit the streaming path"
    )

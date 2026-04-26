"""Tests for MCP streaming context (Phase 3.x.1 Task 8).

Verifies that prsm_inference and prsm_analyze emit progress notifications
when callers supply a progressToken, and remain backwards-compatible for
non-streaming clients.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.mcp_server import (
    _build_progress_emitter,
    _handler_accepts_emit_progress,
    handle_prsm_analyze,
    handle_prsm_inference,
    handle_prsm_quote,
    handle_prsm_node_status,
)


# ── Handler signature introspection ─────────────────────────────────────────


class TestHandlerAcceptsEmitProgress:
    def test_inference_accepts_emit_progress(self):
        assert _handler_accepts_emit_progress(handle_prsm_inference)

    def test_analyze_accepts_emit_progress(self):
        assert _handler_accepts_emit_progress(handle_prsm_analyze)

    def test_quote_does_not_accept_emit_progress(self):
        # Streaming opt-in — non-streaming tools keep their original signature
        assert not _handler_accepts_emit_progress(handle_prsm_quote)

    def test_node_status_does_not_accept_emit_progress(self):
        assert not _handler_accepts_emit_progress(handle_prsm_node_status)

    def test_non_callable_returns_false(self):
        assert not _handler_accepts_emit_progress("not-a-function")

    def test_lambda_without_param_returns_false(self):
        assert not _handler_accepts_emit_progress(lambda x: x)

    def test_function_with_param_returns_true(self):
        async def _h(args, *, emit_progress=None):
            return ""
        assert _handler_accepts_emit_progress(_h)


# ── Emitter construction from request context ──────────────────────────────


class TestBuildProgressEmitter:
    def test_no_request_context_returns_none(self):
        server = MagicMock()
        # Configure request_context property to raise (simulates no active req)
        type(server).request_context = property(
            lambda self: (_ for _ in ()).throw(LookupError("no active request"))
        )
        emitter = _build_progress_emitter(server)
        assert emitter is None

    def test_no_meta_returns_none(self):
        server = MagicMock()
        ctx = MagicMock()
        ctx.meta = None
        server.request_context = ctx
        emitter = _build_progress_emitter(server)
        assert emitter is None

    def test_no_progress_token_returns_none(self):
        server = MagicMock()
        ctx = MagicMock()
        ctx.meta = MagicMock()
        ctx.meta.progressToken = None
        server.request_context = ctx
        emitter = _build_progress_emitter(server)
        assert emitter is None

    def test_with_progress_token_returns_callable(self):
        server = MagicMock()
        ctx = MagicMock()
        ctx.meta = MagicMock()
        ctx.meta.progressToken = "test-token-42"
        ctx.session = MagicMock()
        ctx.session.send_progress_notification = AsyncMock()
        server.request_context = ctx
        emitter = _build_progress_emitter(server)
        assert emitter is not None
        assert callable(emitter)

    @pytest.mark.asyncio
    async def test_emitter_calls_session_send(self):
        server = MagicMock()
        ctx = MagicMock()
        ctx.meta = MagicMock()
        ctx.meta.progressToken = "tok-99"
        ctx.session = MagicMock()
        ctx.session.send_progress_notification = AsyncMock()
        server.request_context = ctx

        emitter = _build_progress_emitter(server)
        await emitter("Working...", 1.0, 4.0)

        ctx.session.send_progress_notification.assert_awaited_once_with(
            progress_token="tok-99",
            progress=1.0,
            total=4.0,
            message="Working...",
        )

    @pytest.mark.asyncio
    async def test_emitter_swallows_send_failures(self):
        """A dropped notification should not crash the tool response."""
        server = MagicMock()
        ctx = MagicMock()
        ctx.meta = MagicMock()
        ctx.meta.progressToken = "tok"
        ctx.session = MagicMock()
        ctx.session.send_progress_notification = AsyncMock(
            side_effect=RuntimeError("session closed")
        )
        server.request_context = ctx

        emitter = _build_progress_emitter(server)
        # Must not raise
        await emitter("msg", 1.0, 4.0)


# ── prsm_inference streaming behavior ───────────────────────────────────────


class TestPrsmInferenceStreaming:
    @pytest.mark.asyncio
    async def test_emits_4_progress_steps_on_success(self):
        """Streaming-aware inference emits 4 progress notifications."""
        emitter = AsyncMock()
        mock_response = {
            "success": True,
            "output": "Result text.",
            "receipt": {
                "job_id": "job-1",
                "model_id": "mock-llama-3-8b",
                "privacy_tier": "standard",
                "content_tier": "A",
                "tee_type": "software",
                "epsilon_spent": 8.0,
                "cost_ftns": "0.5",
                "duration_seconds": 1.0,
                "settler_node_id": "node-x",
                "settler_signature": "deadbeef",
            },
        }
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = mock_response
            await handle_prsm_inference(
                {"prompt": "hi", "model_id": "mock-llama-3-8b", "budget_ftns": 1.0},
                emit_progress=emitter,
            )
        # Building → Locking → Inference complete → Settling = 4 calls
        assert emitter.await_count == 4
        # Each call is awaited with (msg, progress, total)
        call_args = [c.args for c in emitter.await_args_list]
        # Progress is monotonically increasing toward total
        progress_values = [args[1] for args in call_args]
        assert progress_values == [1.0, 2.0, 3.0, 4.0]
        # Total is consistent
        total_values = [args[2] for args in call_args]
        assert all(t == 4.0 for t in total_values)

    @pytest.mark.asyncio
    async def test_no_emit_when_emitter_is_none(self):
        """Without an emitter, no streaming side effects — pure backwards-compat."""
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {
                "success": True,
                "output": "ok",
                "receipt": {},
            }
            # Call without emit_progress kwarg (i.e. None default)
            result = await handle_prsm_inference(
                {"prompt": "hi", "budget_ftns": 1.0},
                emit_progress=None,
            )
        assert "PRSM Inference Result" in result

    @pytest.mark.asyncio
    async def test_validation_failure_skips_emitter(self):
        """Validation errors return early — no progress notifications."""
        emitter = AsyncMock()
        result = await handle_prsm_inference(
            {"budget_ftns": 1.0},  # missing prompt
            emit_progress=emitter,
        )
        assert "Missing required 'prompt'" in result
        emitter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_failure_skips_emitter(self):
        emitter = AsyncMock()
        result = await handle_prsm_inference(
            {"prompt": "x", "budget_ftns": 0},
            emit_progress=emitter,
        )
        assert "FTNS budget" in result
        emitter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_emit_after_node_unreachable_stops_at_step_2(self):
        """Network failure during dispatch yields only the first 2 progress steps."""
        emitter = AsyncMock()
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.side_effect = Exception("Connection refused")
            await handle_prsm_inference(
                {"prompt": "hi", "budget_ftns": 1.0},
                emit_progress=emitter,
            )
        # Building (step 1) + Locking/dispatching (step 2), then exception → no 3 or 4
        assert emitter.await_count == 2

    @pytest.mark.asyncio
    async def test_signature_includes_emit_progress_kwarg_only(self):
        """emit_progress must be keyword-only to avoid positional-arg surprises."""
        sig = inspect.signature(handle_prsm_inference)
        param = sig.parameters.get("emit_progress")
        assert param is not None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


# ── prsm_analyze streaming behavior ─────────────────────────────────────────


class TestPrsmAnalyzeStreaming:
    @pytest.mark.asyncio
    async def test_emits_4_progress_steps_on_success(self):
        emitter = AsyncMock()
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {
                "response": "Analysis answer.",
                "route": "swarm",
                "job_id": "forge-x",
            }
            await handle_prsm_analyze(
                {"query": "EV trends", "budget_ftns": 5.0},
                emit_progress=emitter,
            )
        assert emitter.await_count == 4

    @pytest.mark.asyncio
    async def test_progress_messages_are_human_readable(self):
        emitter = AsyncMock()
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {"response": "x", "route": "r", "job_id": "j"}
            await handle_prsm_analyze(
                {"query": "q", "budget_ftns": 5.0},
                emit_progress=emitter,
            )
        messages = [c.args[0] for c in emitter.await_args_list]
        # Each message is a complete human-readable status
        for msg in messages:
            assert isinstance(msg, str)
            assert len(msg) > 5
        # Final message indicates completion
        assert "complete" in messages[-1].lower()

    @pytest.mark.asyncio
    async def test_no_emit_when_emitter_is_none(self):
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {"response": "x", "route": "r", "job_id": "j"}
            result = await handle_prsm_analyze(
                {"query": "q", "budget_ftns": 5.0},
                emit_progress=None,
            )
        assert "PRSM Analysis Result" in result

    @pytest.mark.asyncio
    async def test_signature_includes_emit_progress_kwarg_only(self):
        sig = inspect.signature(handle_prsm_analyze)
        param = sig.parameters.get("emit_progress")
        assert param is not None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


# ── Backwards compatibility for the dispatcher ──────────────────────────────


class TestBackwardsCompatibility:
    """Existing tests in test_mcp_server.py call handlers without emit_progress;
    these tests verify that pattern still works."""

    @pytest.mark.asyncio
    async def test_inference_callable_without_kwarg(self):
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {
                "success": True, "output": "x", "receipt": {},
            }
            # Call exactly as existing tests do — no streaming arg
            result = await handle_prsm_inference({
                "prompt": "hi", "budget_ftns": 1.0,
            })
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_analyze_callable_without_kwarg(self):
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {"response": "x", "route": "r", "job_id": "j"}
            result = await handle_prsm_analyze({
                "query": "q", "budget_ftns": 5.0,
            })
        assert isinstance(result, str)

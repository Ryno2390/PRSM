"""Phase 3.x.8.1 Task 3 — unit tests for MCP streaming client
helpers (``_parse_sse`` + ``_call_node_api_streaming``)."""

from __future__ import annotations

import json
from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    InferenceError,
    _call_node_api_streaming,
    _parse_sse,
)


# ──────────────────────────────────────────────────────────────────────────
# Fake aiohttp response (just the surface _parse_sse / streaming client need)
# ──────────────────────────────────────────────────────────────────────────


class _FakeContent:
    """Mimics aiohttp.ClientResponse.content — exposes .iter_any()
    yielding bytes chunks. Tests construct one with a list of chunks
    so they can simulate chunk-boundary edge cases (mid-frame splits,
    multi-frame chunks, etc.)."""

    def __init__(self, chunks: List[bytes]):
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(
        self, *, chunks: List[bytes], status: int = 200,
    ):
        self.content = _FakeContent(chunks)
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    """Mimics aiohttp.ClientSession just enough for the streaming
    client. ``post`` returns a _FakeResponse pre-loaded with chunks."""

    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_url: Optional[str] = None
        self.last_json: Optional[dict] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, json=None, headers=None):  # noqa: A002
        self.last_url = url
        self.last_json = json
        return self._response


def _frame(event_type: str, data: dict | str) -> bytes:
    """Encode an SSE frame for use in test chunks."""
    if not isinstance(data, str):
        data = json.dumps(data)
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────
# _parse_sse direct tests
# ──────────────────────────────────────────────────────────────────────────


async def _drain_sse(response) -> List[Tuple[str, str]]:
    """Helper: collect all (event_type, data) tuples from _parse_sse."""
    return [pair async for pair in _parse_sse(response)]


class TestParseSSE:
    @pytest.mark.asyncio
    async def test_single_frame_in_one_chunk(self):
        response = _FakeResponse(chunks=[
            _frame("token", {"text_delta": "hi"}),
        ])
        events = await _drain_sse(response)
        assert events == [("token", '{"text_delta": "hi"}')]

    @pytest.mark.asyncio
    async def test_multiple_frames_in_one_chunk(self):
        response = _FakeResponse(chunks=[
            _frame("token", {"text_delta": "a"})
            + _frame("token", {"text_delta": "b"})
            + _frame("result", {"output": "ab"}),
        ])
        events = await _drain_sse(response)
        assert [e[0] for e in events] == ["token", "token", "result"]

    @pytest.mark.asyncio
    async def test_frame_split_across_chunk_boundary(self):
        # Chunk 1 contains the start of an event line; chunk 2 contains
        # the rest. Parser must buffer + reassemble correctly.
        response = _FakeResponse(chunks=[
            b"event: token\nda",
            b'ta: {"text_delta": "split"}\n\n',
        ])
        events = await _drain_sse(response)
        assert events == [("token", '{"text_delta": "split"}')]

    @pytest.mark.asyncio
    async def test_event_default_message_when_missing(self):
        # SSE spec: missing event: line implies event_type="message".
        response = _FakeResponse(chunks=[
            b"data: hello\n\n",
        ])
        events = await _drain_sse(response)
        assert events == [("message", "hello")]

    @pytest.mark.asyncio
    async def test_comment_lines_ignored(self):
        # Lines starting with ":" are SSE comments — must be silently
        # ignored, not emitted as events.
        response = _FakeResponse(chunks=[
            b": this is a comment\nevent: token\ndata: x\n\n",
        ])
        events = await _drain_sse(response)
        assert events == [("token", "x")]

    @pytest.mark.asyncio
    async def test_unknown_fields_ignored(self):
        # id: + retry: aren't load-bearing for our use; ignored.
        response = _FakeResponse(chunks=[
            b"id: 42\nretry: 1000\nevent: token\ndata: x\n\n",
        ])
        events = await _drain_sse(response)
        assert events == [("token", "x")]

    @pytest.mark.asyncio
    async def test_crlf_line_endings_tolerated(self):
        # Some HTTP intermediaries normalize line endings to CRLF.
        response = _FakeResponse(chunks=[
            b"event: token\r\ndata: x\r\n\r\n",
        ])
        events = await _drain_sse(response)
        assert events == [("token", "x")]

    @pytest.mark.asyncio
    async def test_data_with_leading_space_stripped(self):
        # Per SSE spec: a single leading space after ``data:`` /
        # ``event:`` is consumed.
        response = _FakeResponse(chunks=[
            b"event: token\ndata: hi\n\n",
        ])
        events = await _drain_sse(response)
        assert events[0][1] == "hi"  # NOT " hi"

    @pytest.mark.asyncio
    async def test_multi_line_data_concatenated_with_newlines(self):
        response = _FakeResponse(chunks=[
            b"event: token\ndata: line1\ndata: line2\n\n",
        ])
        events = await _drain_sse(response)
        assert events == [("token", "line1\nline2")]

    @pytest.mark.asyncio
    async def test_no_trailing_blank_line_still_flushes(self):
        # Defensive: server forgot the trailing blank line. Final
        # buffered data is yielded as a final frame.
        response = _FakeResponse(chunks=[
            b"event: token\ndata: x",
        ])
        events = await _drain_sse(response)
        assert events == [("token", "x")]


# ──────────────────────────────────────────────────────────────────────────
# _call_node_api_streaming integration tests (via _FakeSession patch)
# ──────────────────────────────────────────────────────────────────────────


class _ProgressRecorder:
    def __init__(self):
        self.calls: List[Tuple[str, float, Optional[float]]] = []

    async def __call__(self, message, progress, total):
        self.calls.append((message, progress, total))


def _patch_session(response: _FakeResponse):
    """Replace aiohttp.ClientSession with a fake whose post() returns
    the pre-loaded _FakeResponse. Returns a context-manager-shaped
    patch that callers use as ``with _patch_session(...): ...``."""
    fake_session = _FakeSession(response)

    class _SessionFactory:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return fake_session

        async def __aexit__(self, *args):
            return False

    return patch("aiohttp.ClientSession", _SessionFactory)


class TestCallNodeApiStreaming:
    @pytest.mark.asyncio
    async def test_token_events_forwarded_in_order(self):
        chunks = [
            _frame("token", {
                "sequence_index": 0, "text_delta": "hello",
                "token_id": None, "finish_reason": None,
            }),
            _frame("token", {
                "sequence_index": 1, "text_delta": " ",
                "token_id": None, "finish_reason": None,
            }),
            _frame("token", {
                "sequence_index": 2, "text_delta": "world",
                "token_id": None, "finish_reason": "stop",
            }),
            _frame("result", {
                "success": True, "output": "hello world",
                "request_id": "req-1", "job_id": "j",
            }),
        ]
        response = _FakeResponse(chunks=chunks)
        emit = _ProgressRecorder()
        with _patch_session(response):
            result = await _call_node_api_streaming(
                "/compute/inference/stream",
                {"prompt": "x"},
                emit,
            )
        # Three tokens forwarded; sequence_count is 1-indexed.
        assert len(emit.calls) == 3
        assert emit.calls[0] == ("hello", 1.0, None)
        assert emit.calls[1] == (" ", 2.0, None)
        assert emit.calls[2] == ("world", 3.0, None)
        # Result returned as parsed dict.
        assert result["output"] == "hello world"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_result_event_returned_as_dict(self):
        chunks = [_frame("result", {"success": True, "x": 42})]
        response = _FakeResponse(chunks=chunks)
        emit = _ProgressRecorder()
        with _patch_session(response):
            result = await _call_node_api_streaming(
                "/compute/inference/stream", {}, emit,
            )
        assert result == {"success": True, "x": 42}
        # No tokens means no progress events.
        assert emit.calls == []

    @pytest.mark.asyncio
    async def test_error_event_raises_inference_error(self):
        chunks = [_frame("error", {
            "error": "Insufficient budget: 0.0001 < 0.04",
            "code": "EXECUTION_FAILURE",
            "job_id": "j",
        })]
        response = _FakeResponse(chunks=chunks)
        emit = _ProgressRecorder()
        with _patch_session(response):
            with pytest.raises(InferenceError) as exc_info:
                await _call_node_api_streaming(
                    "/compute/inference/stream", {}, emit,
                )
        assert "Insufficient budget" in str(exc_info.value)
        assert exc_info.value.code == "EXECUTION_FAILURE"

    @pytest.mark.asyncio
    async def test_stream_ends_without_terminal_raises_runtime_error(self):
        # Tokens but no result/error — server died mid-stream.
        chunks = [
            _frame("token", {"text_delta": "a"}),
            _frame("token", {"text_delta": "b"}),
        ]
        response = _FakeResponse(chunks=chunks)
        emit = _ProgressRecorder()
        with _patch_session(response):
            with pytest.raises(RuntimeError, match="without a 'result'"):
                await _call_node_api_streaming(
                    "/compute/inference/stream", {}, emit,
                )

    @pytest.mark.asyncio
    async def test_malformed_token_event_raises_inference_error(self):
        # data: not valid JSON
        chunks = [
            b"event: token\ndata: not-json{\n\n",
        ]
        response = _FakeResponse(chunks=chunks)
        emit = _ProgressRecorder()
        with _patch_session(response):
            with pytest.raises(InferenceError) as exc_info:
                await _call_node_api_streaming(
                    "/compute/inference/stream", {}, emit,
                )
        assert exc_info.value.code == "MALFORMED_RESPONSE"
        assert "malformed token event" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_malformed_result_event_raises_inference_error(self):
        chunks = [
            b"event: result\ndata: not-json}{\n\n",
        ]
        response = _FakeResponse(chunks=chunks)
        emit = _ProgressRecorder()
        with _patch_session(response):
            with pytest.raises(InferenceError) as exc_info:
                await _call_node_api_streaming(
                    "/compute/inference/stream", {}, emit,
                )
        assert exc_info.value.code == "MALFORMED_RESPONSE"

    @pytest.mark.asyncio
    async def test_unknown_event_types_ignored(self):
        # Forward-compat: server adds a new event type the client
        # doesn't know about. Don't crash; ignore + keep going.
        chunks = [
            _frame("future_event", {"some": "data"}),
            _frame("token", {"text_delta": "x"}),
            _frame("result", {"success": True, "output": "x"}),
        ]
        response = _FakeResponse(chunks=chunks)
        emit = _ProgressRecorder()
        with _patch_session(response):
            result = await _call_node_api_streaming(
                "/compute/inference/stream", {}, emit,
            )
        # Token forwarded; result returned. Unknown event silently
        # skipped.
        assert len(emit.calls) == 1
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_inference_error_carries_code_attribute(self):
        # Verifies that the code is reachable as both .code attribute
        # and via the constructor's keyword argument.
        chunks = [_frame("error", {
            "error": "rejected",
            "code": "TIER_GATE",
        })]
        response = _FakeResponse(chunks=chunks)
        emit = _ProgressRecorder()
        with _patch_session(response):
            try:
                await _call_node_api_streaming(
                    "/compute/inference/stream", {}, emit,
                )
                assert False, "should have raised"
            except InferenceError as e:
                assert e.code == "TIER_GATE"
                assert e.message == "rejected"

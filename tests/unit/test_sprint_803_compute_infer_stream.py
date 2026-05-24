"""Sprint 803 — `prsm compute infer --stream` SSE consumer.

Sprint 802 shipped the unary `compute infer` CLI. Sprint 803
adds the streaming variant: same body schema posted to
/compute/inference/stream, but the CLI consumes the SSE event
stream and prints tokens inline as they arrive.

Wire format (per api.py):
  event: token
  data: {"sequence_index": N, "text_delta": "...",
         "token_id": int|null, "finish_reason": str|null}

  event: result
  data: {"success": true, "output": "...", "ftns_charged": "...",
         "receipt": {...}}

  event: error
  data: {"detail": "..."}

CLI behavior:
- --stream flag triggers the streaming path (default = unary).
- Text mode: prints text_delta inline as each token frame
  arrives + a final "[done]" line with receipt summary on
  result event.
- JSON mode: accumulates tokens + emits a single combined
  JSON object at end of stream so callers chain.
- error event → exit 1 with detail surfaced.
- 503 on initial POST → exit 1.
- Unreachable → exit 2.

Pin tests:
- --stream flag in --help.
- Stream POSTs to /compute/inference/stream (NOT /inference).
- Stream body schema matches the unary endpoint (sprint 802).
- Token events print inline (text mode).
- Result event surfaces receipt summary.
- Error event → exit 1.
- 503 initial response → exit 1.
- Unreachable → exit 2.
- JSON mode accumulates + emits combined payload.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args):
    from prsm.cli import compute as _compute_group
    return CliRunner().invoke(_compute_group, ["infer"] + list(args))


# ---- Helpers to build SSE byte streams -------------------------


def _sse_frame(event_type: str, data: dict) -> bytes:
    body = json.dumps(data)
    return f"event: {event_type}\ndata: {body}\n\n".encode("utf-8")


def _stream_response(frames: list[bytes], status_code: int = 200):
    """Build a fake httpx response usable in a `with ctx as r:`
    pattern. Mimics httpx.Response.iter_bytes()."""

    class _FakeResponse:
        def __init__(self):
            self.status_code = status_code

        def iter_bytes(self, chunk_size=None):
            for f in frames:
                yield f

        def iter_lines(self):
            buf = b"".join(frames).decode("utf-8")
            for line in buf.split("\n"):
                yield line

        @property
        def text(self):
            return b"".join(frames).decode("utf-8")

    class _Ctx:
        def __init__(self):
            self._resp = _FakeResponse()

        def __enter__(self):
            return self._resp

        def __exit__(self, *args):
            return False

    return _Ctx()


# ---- Flag exists ----------------------------------------------


def test_stream_flag_exists():
    from prsm.cli import compute as _compute_group
    runner = CliRunner()
    infer_cmd = _compute_group.commands["infer"]
    result = runner.invoke(infer_cmd, ["--help"])
    assert result.exit_code == 0
    assert "--stream" in result.output


# ---- POSTs to /stream endpoint --------------------------------


def test_stream_posts_to_stream_endpoint():
    frames = [
        _sse_frame("result", {
            "success": True,
            "output": " the",
            "ftns_charged": "0.1",
            "receipt": {"settler_signature": "0xabc"},
        }),
    ]
    with patch("httpx.stream", return_value=_stream_response(frames)) as ms:
        result = _invoke([
            "--prompt", "Hi", "--stream", "--format", "json",
        ])
    assert result.exit_code == 0, result.output
    # httpx.stream called as a positional or kw method/URL pair
    # First positional is method ("POST"); second is URL.
    args = ms.call_args.args
    assert args[0] == "POST"
    assert "/compute/inference/stream" in args[1]


def test_stream_body_schema_matches_unary():
    """Body shape identical to unary (sprint 802) — same set of
    canonical field names."""
    frames = [
        _sse_frame("result", {
            "success": True, "output": "", "ftns_charged": "0",
            "receipt": {},
        }),
    ]
    with patch(
        "httpx.stream", return_value=_stream_response(frames),
    ) as ms:
        _invoke([
            "--prompt", "Hi", "--stream", "--format", "json",
        ])
    body = ms.call_args.kwargs.get("json") or {}
    assert "prompt" in body
    assert "model_id" in body
    assert "budget_ftns" in body
    assert "privacy_tier" in body
    assert "content_tier" in body
    assert "max_tokens" in body


# ---- Token events render inline -------------------------------


def test_token_events_render_inline_text():
    frames = [
        _sse_frame("token", {
            "sequence_index": 0, "text_delta": "Hello",
            "token_id": None, "finish_reason": None,
        }),
        _sse_frame("token", {
            "sequence_index": 1, "text_delta": " world",
            "token_id": None, "finish_reason": None,
        }),
        _sse_frame("result", {
            "success": True, "output": "Hello world",
            "ftns_charged": "0.2", "receipt": {},
        }),
    ]
    with patch(
        "httpx.stream", return_value=_stream_response(frames),
    ):
        result = _invoke([
            "--prompt", "Hi", "--stream", "--format", "text",
        ])
    assert result.exit_code == 0
    # Both token deltas surface in output
    assert "Hello" in result.output
    assert "world" in result.output


def test_result_event_surfaces_receipt():
    frames = [
        _sse_frame("token", {
            "sequence_index": 0, "text_delta": "X",
            "token_id": None, "finish_reason": None,
        }),
        _sse_frame("result", {
            "success": True, "output": "X",
            "ftns_charged": "0.5",
            "receipt": {
                "settler_signature": "0xabc",
                "settler_node_id": "node1",
            },
        }),
    ]
    with patch(
        "httpx.stream", return_value=_stream_response(frames),
    ):
        result = _invoke([
            "--prompt", "Hi", "--stream", "--format", "text",
        ])
    assert result.exit_code == 0
    # Cost + signed-receipt indicator surfaced after the stream
    assert "0.5" in result.output


# ---- Error paths ----------------------------------------------


def test_error_event_exits_1():
    frames = [
        _sse_frame("error", {
            "detail": "executor not initialized",
        }),
    ]
    with patch(
        "httpx.stream", return_value=_stream_response(frames),
    ):
        result = _invoke([
            "--prompt", "Hi", "--stream", "--format", "text",
        ])
    assert result.exit_code == 1
    assert "executor not initialized" in result.output or (
        "error" in result.output.lower()
    )


def test_initial_503_exits_1():
    frames = [b'{"detail": "rate limited"}']
    with patch(
        "httpx.stream",
        return_value=_stream_response(frames, status_code=503),
    ):
        result = _invoke([
            "--prompt", "Hi", "--stream", "--format", "text",
        ])
    assert result.exit_code == 1


def test_stream_unreachable_exits_2():
    with patch(
        "httpx.stream",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke([
            "--prompt", "Hi", "--stream", "--format", "text",
        ])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()


# ---- JSON mode accumulation ----------------------------------


def test_json_mode_accumulates_into_one_payload():
    """JSON mode emits ONE combined object so callers can pipe
    the output to `jq` without parsing event-stream framing."""
    frames = [
        _sse_frame("token", {
            "sequence_index": 0, "text_delta": "a",
            "token_id": None, "finish_reason": None,
        }),
        _sse_frame("token", {
            "sequence_index": 1, "text_delta": "b",
            "token_id": None, "finish_reason": None,
        }),
        _sse_frame("result", {
            "success": True, "output": "ab",
            "ftns_charged": "0.3",
            "receipt": {"settler_signature": "0xz"},
        }),
    ]
    with patch(
        "httpx.stream", return_value=_stream_response(frames),
    ):
        result = _invoke([
            "--prompt", "Hi", "--stream", "--format", "json",
        ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    # Should include the final result fields
    assert data.get("output") == "ab"
    # And a tokens array reconstructable from the stream
    assert "tokens" in data
    assert [t["text_delta"] for t in data["tokens"]] == ["a", "b"]

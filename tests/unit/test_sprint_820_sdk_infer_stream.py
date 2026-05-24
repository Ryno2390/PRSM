"""Sprint 820 — PRSMClient.infer_stream() SSE async generator.

Sprint 819 shipped `.infer()` for unary verifiable inference.
Sprint 820 adds the streaming counterpart, mirroring sprint
803's CLI `--stream` flag.

  async for event in client.infer_stream(prompt="..."):
      if event["type"] == "token":
          print(event["text_delta"], end="", flush=True)
      elif event["type"] == "result":
          # Terminal event with receipt
          ...

Events are dicts with a `type` discriminator:
  {"type": "token", "sequence_index": N, "text_delta": "...",
   "token_id": int|None, "finish_reason": str|None}
  {"type": "result", "success": True, "output": "...",
   "ftns_charged": "...", "receipt": {...}}
  {"type": "error", "detail": "..."}

Pin tests:
- .infer_stream method exists.
- POSTs to /compute/inference/stream.
- Body has canonical 6-field shape (same as sprint 819 .infer).
- Defaults match sprint 819 .infer.
- Parses SSE frames into token events.
- Parses terminal result event.
- Parses error event.
- Async-generator semantics: yields events; iteration terminates
  after result or error.
"""
from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


def _sse_frame(event_type: str, data: dict) -> str:
    body = json.dumps(data)
    return f"event: {event_type}\ndata: {body}\n\n"


class _FakeSSEResponse:
    """Mimics aiohttp Response with iter_chunked / content
    streaming."""

    def __init__(self, status: int, body_text: str):
        self.status = status
        self._body = body_text.encode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    @property
    def content(self):
        # aiohttp's resp.content has async iteration over lines.
        return _FakeContent(self._body)


class _FakeContent:
    def __init__(self, body: bytes):
        self._body = body

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        # Split on newlines, yield bytes per line + the trailing
        # blank lines that separate SSE frames.
        for line in self._body.split(b"\n"):
            yield line + b"\n"


class _FakeSession:
    """Mimics the aiohttp ClientSession surface the SDK uses."""

    def __init__(self, response: _FakeSSEResponse):
        self._response = response
        self.post_calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.post_calls.append({
            "url": url, "json": json, "headers": headers,
        })
        return self._response

    async def close(self):
        pass


# ---- Method exists --------------------------------------------


async def test_infer_stream_method_exists():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    assert hasattr(client, "infer_stream")
    assert callable(client.infer_stream)


# ---- POSTs to /compute/inference/stream -----------------------


async def test_infer_stream_posts_to_stream_endpoint():
    from prsm.sdk.client import PRSMClient

    body_text = _sse_frame("result", {
        "success": True, "output": "ok",
        "ftns_charged": "0.1", "receipt": {},
    })
    fake_response = _FakeSSEResponse(200, body_text)
    fake_session = _FakeSession(fake_response)

    client = PRSMClient("http://node:8000")
    client._session = fake_session

    async for _event in client.infer_stream(prompt="Hi"):
        pass

    assert len(fake_session.post_calls) == 1
    call = fake_session.post_calls[0]
    assert "/compute/inference/stream" in call["url"]


async def test_infer_stream_body_canonical_shape():
    from prsm.sdk.client import PRSMClient

    body_text = _sse_frame("result", {
        "success": True, "output": "", "ftns_charged": "0",
        "receipt": {},
    })
    fake_session = _FakeSession(_FakeSSEResponse(200, body_text))
    client = PRSMClient("http://node:8000")
    client._session = fake_session

    async for _e in client.infer_stream(prompt="Hi"):
        pass

    body = fake_session.post_calls[0]["json"]
    assert "prompt" in body
    assert "model_id" in body
    assert "budget_ftns" in body
    assert "privacy_tier" in body
    assert "content_tier" in body
    assert "max_tokens" in body


async def test_infer_stream_defaults_match_unary():
    from prsm.sdk.client import PRSMClient

    body_text = _sse_frame("result", {
        "success": True, "output": "", "ftns_charged": "0",
        "receipt": {},
    })
    fake_session = _FakeSession(_FakeSSEResponse(200, body_text))
    client = PRSMClient("http://node:8000")
    client._session = fake_session

    async for _e in client.infer_stream(prompt="Hi"):
        pass

    body = fake_session.post_calls[0]["json"]
    assert body["model_id"] == "gpt2"
    assert body["max_tokens"] == 8
    assert body["budget_ftns"] == 1.0
    assert body["privacy_tier"] == "none"
    assert body["content_tier"] == "A"


# ---- Event parsing --------------------------------------------


async def test_infer_stream_yields_token_then_result():
    from prsm.sdk.client import PRSMClient

    body_text = (
        _sse_frame("token", {
            "sequence_index": 0, "text_delta": "Hello",
            "token_id": None, "finish_reason": None,
        })
        + _sse_frame("token", {
            "sequence_index": 1, "text_delta": " world",
            "token_id": None, "finish_reason": None,
        })
        + _sse_frame("result", {
            "success": True, "output": "Hello world",
            "ftns_charged": "0.2", "receipt": {},
        })
    )
    fake_session = _FakeSession(_FakeSSEResponse(200, body_text))
    client = PRSMClient("http://node:8000")
    client._session = fake_session

    events = []
    async for ev in client.infer_stream(prompt="Hi"):
        events.append(ev)

    # 2 token events + 1 result event
    assert len(events) == 3
    assert events[0]["type"] == "token"
    assert events[0]["text_delta"] == "Hello"
    assert events[1]["type"] == "token"
    assert events[1]["text_delta"] == " world"
    assert events[2]["type"] == "result"
    assert events[2]["output"] == "Hello world"


async def test_infer_stream_yields_error_event():
    from prsm.sdk.client import PRSMClient

    body_text = _sse_frame("error", {
        "detail": "executor not initialized",
    })
    fake_session = _FakeSession(_FakeSSEResponse(200, body_text))
    client = PRSMClient("http://node:8000")
    client._session = fake_session

    events = []
    async for ev in client.infer_stream(prompt="Hi"):
        events.append(ev)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "executor not initialized" in events[0]["detail"]


# ---- Async-generator semantics -------------------------------


async def test_infer_stream_terminates_after_result():
    """After a result event, the async generator stops iterating
    even if more frames remain."""
    from prsm.sdk.client import PRSMClient

    body_text = (
        _sse_frame("result", {
            "success": True, "output": "done",
            "ftns_charged": "0", "receipt": {},
        })
        # Extra frames after result — SDK should ignore.
        + _sse_frame("token", {
            "sequence_index": 99, "text_delta": "ghost",
            "token_id": None, "finish_reason": None,
        })
    )
    fake_session = _FakeSession(_FakeSSEResponse(200, body_text))
    client = PRSMClient("http://node:8000")
    client._session = fake_session

    events = []
    async for ev in client.infer_stream(prompt="Hi"):
        events.append(ev)
    assert events[-1]["type"] == "result"
    # No ghost token after result
    assert all(
        ev.get("text_delta") != "ghost" for ev in events
    )

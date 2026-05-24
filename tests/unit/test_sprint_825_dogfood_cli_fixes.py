"""Sprint 825/826 — dogfood CLI fixes against live daemon.

Live walkthrough of the post-771 surface against a real
mock-executor daemon surfaced two more UX bugs:

  FINDING #6 (sprint 825 — cost field lookup): `compute infer`
  text mode rendered "Cost: ? FTNS" against a live parallax
  daemon because the cost lives at `receipt.cost_ftns`, not at
  top-level `ftns_charged` or `cost_ftns`. Sprint 802's CLI
  only checked the top-level keys + fell through to "?" against
  every real daemon — undermining the operator-facing economic
  receipt UX.

  FINDING #7 (sprint 826 — stream error body): sprint 803's
  `compute infer --stream` raised "Attempted to access streaming
  response content, without having called read()" when the mock
  executor returned HTTP 503 ("Inference executor does not
  support streaming"). The error-handling path accessed
  `resp.text` on an httpx streaming response without calling
  `resp.read()` first → the operator saw a confusing "Daemon
  unreachable" message even though the daemon was reachable +
  returning a clean 503 with actionable detail.

Sprint 825 fix: cost lookup chains
``data.get("ftns_charged", data.get("cost_ftns",
   receipt.get("cost_ftns", "?")))`` so the parallax canonical
location resolves correctly. Sprint 802's existing 10 tests
still pass (they covered top-level fields).

Sprint 826 fix: error-path calls ``resp.read()`` inside a
try/except before accessing ``resp.text``, with a fallback
diagnostic string for genuinely-unreadable bodies. Sprint 803's
existing 9 tests still pass (they exercised the success path).

Pin tests:
- Cost present only at receipt.cost_ftns → "Cost: <value> FTNS"
  rendered (FINDING #6 fix).
- Cost present at top-level still wins (back-compat).
- Cost absent everywhere → "Cost: ? FTNS" fallback preserved.
- Streaming 503 from server → exit 1 + server-detail surfaced
  (FINDING #7 fix).
- Streaming 503 → "Attempted to access" string NOT in output
  (regression guard against the httpx error leaking through).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke_infer(args):
    from prsm.cli import compute as _compute_group
    return CliRunner().invoke(
        _compute_group, ["infer"] + list(args),
    )


def _make_response(payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


# ---- Sprint 825 — cost-field lookup --------------------------


def test_cost_at_receipt_cost_ftns_renders():
    """FINDING #6 fix: when cost lives at receipt.cost_ftns
    (parallax canonical location), text mode must render it."""
    body = {
        "success": True,
        "output": "hi",
        "receipt": {
            "output_hash": "abc",
            "settler_signature": "sig",
            "cost_ftns": 0.10,
        },
    }
    with patch("httpx.post", return_value=_make_response(body)):
        result = _invoke_infer([
            "--prompt", "test",
            "--model", "mock-phi-3",
            "--max-tokens", "1",
        ])
    assert result.exit_code == 0, result.output
    assert "Cost: 0.10 FTNS" in result.output or "Cost: 0.1 FTNS" in result.output


def test_cost_at_top_level_still_wins_back_compat():
    """When ftns_charged is present at top level, it MUST win
    over receipt.cost_ftns — preserves sprint 802 contract."""
    body = {
        "success": True,
        "output": "hi",
        "ftns_charged": 0.42,
        "receipt": {
            "output_hash": "abc",
            "settler_signature": "sig",
            "cost_ftns": 0.99,
        },
    }
    with patch("httpx.post", return_value=_make_response(body)):
        result = _invoke_infer([
            "--prompt", "t",
            "--model", "mock-phi-3",
            "--max-tokens", "1",
        ])
    assert result.exit_code == 0
    assert "0.42" in result.output
    assert "0.99" not in result.output


def test_cost_absent_everywhere_falls_back_to_question_mark():
    """Sprint 802's fallback survives when no cost field exists
    anywhere — defensive against degenerate server responses."""
    body = {
        "success": True,
        "output": "hi",
        "receipt": {
            "output_hash": "abc",
            "settler_signature": "sig",
        },
    }
    with patch("httpx.post", return_value=_make_response(body)):
        result = _invoke_infer([
            "--prompt", "t",
            "--model", "mock-phi-3",
            "--max-tokens", "1",
        ])
    assert result.exit_code == 0
    assert "Cost: ? FTNS" in result.output


# ---- Sprint 826 — streaming error-body read ------------------


class _FakeStreamResponse:
    """Mimics httpx streaming response: .text raises until
    .read() is called. Used to live-fire the sprint 826 path."""

    def __init__(self, status_code, body_text):
        self.status_code = status_code
        self._body = body_text
        self._was_read = False

    @property
    def text(self):
        if not self._was_read:
            raise RuntimeError(
                "Attempted to access streaming response content, "
                "without having called read()."
            )
        return self._body

    def read(self):
        self._was_read = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_stream_503_surfaces_server_detail():
    """FINDING #7 fix: a 503 with a clean detail body must
    reach the operator's terminal, not the httpx error."""
    detail = (
        '{"detail":"Inference executor does not support '
        'streaming. Wire a ParallaxScheduledExecutor."}'
    )
    fake = _FakeStreamResponse(503, detail)
    with patch("httpx.stream", return_value=fake):
        result = _invoke_infer([
            "--stream",
            "--prompt", "t",
            "--model", "mock-phi-3",
            "--max-tokens", "1",
        ])
    assert result.exit_code == 1
    # Click may wrap long lines; normalize whitespace for match.
    flat = " ".join(result.output.split())
    assert "Inference executor does not support streaming" in flat


def test_stream_503_does_not_leak_httpx_error():
    """Regression guard: the sprint 803 path that raised
    "Attempted to access streaming response content" MUST NOT
    reach the operator's terminal."""
    fake = _FakeStreamResponse(
        503, '{"detail":"some operator-facing message"}',
    )
    with patch("httpx.stream", return_value=fake):
        result = _invoke_infer([
            "--stream",
            "--prompt", "t",
            "--model", "mock-phi-3",
            "--max-tokens", "1",
        ])
    assert "Attempted to access" not in result.output


def test_stream_503_json_mode_includes_detail():
    """JSON mode propagates the detail field so CI/scripts can
    chain on the actionable server message."""
    detail = '{"detail":"wire-the-executor"}'
    fake = _FakeStreamResponse(503, detail)
    with patch("httpx.stream", return_value=fake):
        result = _invoke_infer([
            "--stream",
            "--prompt", "t",
            "--model", "mock-phi-3",
            "--max-tokens", "1",
            "--format", "json",
        ])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["ok"] is False
    assert data["status"] == 503
    assert "wire-the-executor" in data["detail"]

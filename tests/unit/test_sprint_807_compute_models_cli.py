"""Sprint 807 — `prsm compute models` CLI lists available model_ids.

Pre-807 users discovered model_ids by reading docs or grepping
operator config. The /compute/models endpoint existed
(sprint 235+, F63 fix in sprint 535) but no first-class CLI
consumer. Result: every new user had to either guess "gpt2" or
ask in chat.

Sprint 807 ships:

  prsm compute models [--api-url URL] [--format text|json]

Wraps GET /compute/models. Text mode renders one model_id per
line + count footer. JSON mode emits the full server payload.

Exit codes:
  0 — listed successfully (even when models=[] — that's a valid
      empty state, not an error)
  1 — 503 (executor not initialized) → exit 1 with actionable
      hint pointing at PRSM_INFERENCE_EXECUTOR
  2 — daemon unreachable

Pin tests:
- Command registered under `compute` group.
- GET hits /compute/models.
- Text mode lists model_ids + count.
- Empty list → operator-readable "no models available" hint.
- JSON mode returns full payload.
- 503 → exit 1 with hint.
- Unreachable → exit 2.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args=None):
    from prsm.cli import compute as _compute_group
    return CliRunner().invoke(
        _compute_group, ["models"] + (args or []),
    )


def _success_response(models):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"models": list(models), "count": len(models)}
    return r


# ---- Command registration --------------------------------------


def test_models_command_registered():
    from prsm.cli import compute as _compute_group
    cmd_names = [c.name for c in _compute_group.commands.values()]
    assert "models" in cmd_names


# ---- URL + text mode ------------------------------------------


def test_get_url_is_compute_models():
    with patch(
        "httpx.get", return_value=_success_response(["gpt2"]),
    ) as mg:
        result = _invoke(["--format", "json"])
    assert result.exit_code == 0, result.output
    url = mg.call_args.args[0] if mg.call_args.args else (
        mg.call_args.kwargs.get("url")
    )
    assert url.endswith("/compute/models")


def test_text_mode_lists_models_and_count():
    with patch(
        "httpx.get",
        return_value=_success_response(["gpt2", "llama-7b"]),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0
    assert "gpt2" in result.output
    assert "llama-7b" in result.output
    # Count appears
    assert "2" in result.output


def test_empty_models_actionable_hint():
    """models=[] → text mode shows hint, not silent."""
    with patch("httpx.get", return_value=_success_response([])):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0
    out = result.output.lower()
    assert (
        "no models" in out
        or "0 models" in out
        or "empty" in out
    )


# ---- JSON mode ------------------------------------------------


def test_json_mode_returns_payload():
    with patch(
        "httpx.get",
        return_value=_success_response(["gpt2", "llama-7b"]),
    ):
        result = _invoke(["--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["models"] == ["gpt2", "llama-7b"]
    assert data["count"] == 2


# ---- Error paths ----------------------------------------------


def test_503_exits_1_with_hint():
    fake = MagicMock()
    fake.status_code = 503
    fake.text = '{"detail":"Inference executor not initialized"}'
    with patch("httpx.get", return_value=fake):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    out = result.output.lower()
    assert (
        "executor" in out
        or "503" in out
        or "prsm_inference_executor" in out
    )


def test_unreachable_exits_2():
    with patch(
        "httpx.get",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()

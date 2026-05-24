"""Sprint 824 — smoke-test surfaces unknown_model errors actionably.

Dogfood walkthrough surfaced a second real UX bug: when the
daemon returns HTTP 200 with `success: false` + `error:
"Unknown model_id: gpt2"` (the parallax executor's clean
application-error path), sprint-771's smoke-test fell through
to the "receipt missing settler_signature (§7 invariant
violated)" branch — a misleading message that suggests a
cryptography problem when the actual issue is just an
operator config mismatch (smoke-test's default --model is
"gpt2" but the executor's catalog doesn't include it,
e.g. mock-executor mode).

Sprint 824 fix: when `success` is False in the response body,
surface the `error` field directly + suggest the --model flag
when the error mentions "Unknown model_id". Falls through to
the §7-invariant message only when success is True but
signature is absent (a real cryptography problem).

Pin tests:
- Server returns 200 + success=false + error="Unknown
  model_id: gpt2" → exit 1 + error surfaced in output.
- Same case → "missing settler_signature" NOT in output
  (regression guard against the old confusing message).
- Same case → "--model" hint in output (operator
  discoverability).
- Server returns 200 + success=true + empty signature → still
  exits 1 with the §7-invariant message (preserves the real
  cryptography check).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args=None):
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["smoke-test"] + (args or []),
    )


def _success_pool_response():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "gpu_count": 2,
        "pool_kind": "dht-backed",
    }
    return r


# ---- Unknown-model error surfaces correctly ------------------


def test_unknown_model_exits_1_with_actionable_error():
    """200 + success=false + Unknown model_id → exit 1 with the
    error message in output."""
    bad = MagicMock()
    bad.status_code = 200
    bad.json.return_value = {
        "success": False,
        "error": "Unknown model_id: gpt2",
        "job_id": "j1",
        "request_id": "r1",
    }
    with patch("httpx.get", return_value=_success_pool_response()), \
         patch("httpx.post", return_value=bad):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    assert "Unknown model_id" in result.output


def test_unknown_model_does_not_misreport_signature():
    """Regression guard: the old confusing "missing
    settler_signature" message MUST NOT appear when the actual
    issue is application-level (success=false)."""
    bad = MagicMock()
    bad.status_code = 200
    bad.json.return_value = {
        "success": False,
        "error": "Unknown model_id: gpt2",
    }
    with patch("httpx.get", return_value=_success_pool_response()), \
         patch("httpx.post", return_value=bad):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    assert "settler_signature" not in result.output


def test_unknown_model_suggests_model_flag():
    """Hint nudges operator to use --model + run `compute
    models` to discover the catalog."""
    bad = MagicMock()
    bad.status_code = 200
    bad.json.return_value = {
        "success": False,
        "error": "Unknown model_id: gpt2",
    }
    with patch("httpx.get", return_value=_success_pool_response()), \
         patch("httpx.post", return_value=bad):
        result = _invoke(["--format", "text"])
    assert "--model" in result.output
    assert "compute models" in result.output


# ---- Real §7-invariant path still works ----------------------


def test_success_true_but_missing_sig_still_reports_invariant():
    """When success=true but the receipt's settler_signature is
    empty, the smoke-test must STILL flag the §7-invariant
    violation (the real cryptography check). Sprint 824 didn't
    change this path."""
    bad_receipt = MagicMock()
    bad_receipt.status_code = 200
    bad_receipt.json.return_value = {
        "success": True,
        "output": " the",
        "receipt": {"output_hash": "deadbeef"},  # no signature
    }
    with patch("httpx.get", return_value=_success_pool_response()), \
         patch("httpx.post", return_value=bad_receipt):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    assert "settler_signature" in result.output


# ---- JSON mode -----------------------------------------------


def test_unknown_model_json_mode_includes_error():
    """JSON mode propagates the error field so callers (CI,
    scripts) can chain."""
    import json
    bad = MagicMock()
    bad.status_code = 200
    bad.json.return_value = {
        "success": False,
        "error": "Unknown model_id: gpt2",
    }
    with patch("httpx.get", return_value=_success_pool_response()), \
         patch("httpx.post", return_value=bad):
        result = _invoke(["--format", "json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert (
        data["inference"]["error"] == "Unknown model_id: gpt2"
    )

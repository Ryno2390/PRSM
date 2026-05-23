"""Sprint 771 — `prsm node smoke-test` automates runbook §8.

The runbook tells operators to manually curl two endpoints after
deploy:
  1. GET /admin/parallax/pool/snapshot — ensure DHT has peers
  2. POST /compute/inference — ensure the inference roundtrip
     works end-to-end + the receipt has a settler_signature

Pre-771 every fleet operator copy-pastes the two curls + eyeballs
the JSON. Sprint 771 ships a single CLI: `prsm node smoke-test`.

Exit codes:
  0 — both checks passed
  1 — daemon answered but a check failed (pool empty, inference
      error, missing signature, etc.)
  2 — daemon unreachable

Pin tests:
- Command registered.
- Happy path (pool snapshot + inference both succeed → exit 0,
  text shows PASS for each check, signed-receipt confirmation).
- JSON output mode.
- Pool snapshot unreachable → exit 2 with actionable error.
- Inference returns non-200 → exit 1 with surfaced detail.
- Inference returns 200 but missing settler_signature → exit 1.
- --no-pool flag skips the pool snapshot check (single-node dev).
- --prompt override flag wired.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args=None):
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["smoke-test"] + (args or []),
    )


def test_smoke_test_command_registered():
    """Command exists."""
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "smoke-test" in cmd_names


def _pool_ok():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "gpu_count": 2,
        "pool_kind": "dht-backed",
        "peers": [
            {"node_id": "n1"},
            {"node_id": "n2"},
        ],
    }
    return r


def _inference_ok():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "success": True,
        "output": " the",
        "receipt": {
            "settler_signature": "abc123",
            "output_hash": "deadbeef",
        },
    }
    return r


def test_happy_path_both_checks_pass():
    """Pool + inference both succeed → exit 0, text shows PASS."""
    with patch("httpx.get", return_value=_pool_ok()), \
         patch("httpx.post", return_value=_inference_ok()):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    # Pool check surfaces gpu_count
    assert "2" in result.output
    # Inference check confirms signed receipt
    assert "settler_signature" in result.output.lower() or \
        "signed" in result.output.lower()


def test_json_output_returns_structured_payload():
    """JSON mode returns both check results."""
    with patch("httpx.get", return_value=_pool_ok()), \
         patch("httpx.post", return_value=_inference_ok()):
        result = _invoke(["--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["ok"] is True
    assert data["pool"]["ok"] is True
    assert data["pool"]["gpu_count"] == 2
    assert data["inference"]["ok"] is True
    assert data["inference"]["signed"] is True


def test_pool_unreachable_exits_2():
    """Network error on first probe → exit 2 with clear msg."""
    with patch(
        "httpx.get",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()


def test_inference_failure_exits_1():
    """Pool OK, inference returns 500 → exit 1, detail surfaced."""
    bad = MagicMock()
    bad.status_code = 500
    bad.text = '{"detail":"executor not initialized"}'
    with patch("httpx.get", return_value=_pool_ok()), \
         patch("httpx.post", return_value=bad):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    assert "executor not initialized" in result.output or \
        "500" in result.output


def test_missing_settler_signature_exits_1():
    """Inference returns 200 but receipt missing signature →
    exit 1 (the §7 verifiable-inference invariant FAILED)."""
    bad_receipt = MagicMock()
    bad_receipt.status_code = 200
    bad_receipt.json.return_value = {
        "success": True,
        "output": " the",
        "receipt": {"output_hash": "deadbeef"},
    }
    with patch("httpx.get", return_value=_pool_ok()), \
         patch("httpx.post", return_value=bad_receipt):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    assert "signature" in result.output.lower()


def test_no_pool_flag_skips_pool_check():
    """--no-pool skips snapshot probe; only inference is exercised.
    Useful on single-node dev where DHT is empty."""
    with patch("httpx.get") as mock_get, \
         patch("httpx.post", return_value=_inference_ok()):
        result = _invoke(["--no-pool", "--format", "text"])
    assert result.exit_code == 0, result.output
    mock_get.assert_not_called()


def test_prompt_override_flag_wired():
    """--prompt 'foo' passes 'foo' to the inference body."""
    with patch("httpx.get", return_value=_pool_ok()), \
         patch("httpx.post", return_value=_inference_ok()) as mp:
        result = _invoke(["--prompt", "Hello world"])
    assert result.exit_code == 0, result.output
    body = mp.call_args.kwargs.get("json")
    assert body is not None
    assert body.get("prompt") == "Hello world"

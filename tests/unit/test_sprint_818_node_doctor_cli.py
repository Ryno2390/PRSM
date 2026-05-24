"""Sprint 818 — `prsm node doctor` composite triage CLI.

Every CLI ecosystem ships a `doctor` (git doctor, brew doctor,
npm doctor, etc.) because operators with malfunctioning nodes
need ONE command to find the issue. Pre-818 PRSM had ~50
operator commands but no aggregate triage.

  prsm node doctor [--api-url URL] [--format text|json]

Composes 4 daemon checks:
  1. /health        — basic up/down + node_id
  2. /admin/preemption/status — preemption flag (sprint 776)
  3. /admin/output-cache-stats — cache hit-rate (sprint 815)
  4. /peers — connected peer count

Each check returns one of:
  PASS — green, working
  WARN — yellow, working but suboptimal (e.g. preempted, 0 peers)
  FAIL — red, broken (e.g. daemon unreachable, health 500)

Output (text):
  PRSM Node Doctor
  ────────────────
  [PASS] daemon: ok (node_id=abc123…)
  [PASS] preemption: clear (backend=aws)
  [WARN] output_cache: not configured
  [PASS] peers: 3 connected
  ────────────────
  Overall: WARN (1 warning, 0 failures)

Exit codes:
  0 — overall PASS or WARN
  1 — overall FAIL (one or more checks FAIL)
  2 — daemon unreachable (special case: /health itself failed)

Pin tests:
- Command registered.
- All-pass: text shows PASS lines, exit 0.
- /health fails (daemon unreachable) → exit 2.
- One subcheck FAIL → overall FAIL, exit 1.
- WARN-only → exit 0 (warnings are advisory).
- JSON mode returns structured payload {checks, overall}.
- Each subcheck failure is operator-readable (no traceback).
- Source-shape: doctor hits 4 endpoints (health, preemption,
  output-cache, peers).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args=None):
    from prsm.cli import node as _node_group
    return CliRunner().invoke(
        _node_group, ["doctor"] + (args or []),
    )


def _route_get_factory(responses):
    """Build a function suitable for `side_effect` on httpx.get
    that routes by URL substring to a response in `responses`
    dict. Default: a 200 MagicMock with {"status": "ok"}."""
    def _route(url, *args, **kwargs):
        for substring, resp in responses.items():
            if substring in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        # default 200 OK
        default = MagicMock()
        default.status_code = 200
        default.json.return_value = {"status": "ok"}
        return default
    return _route


def _ok(body):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = body
    return r


def _err(status, text="error"):
    r = MagicMock()
    r.status_code = status
    r.text = text
    return r


# ---- Command registration --------------------------------------


def test_doctor_command_registered():
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "doctor" in cmd_names


# ---- All-pass scenario -----------------------------------------


def _all_pass_routes():
    return {
        "/health": _ok({"status": "ok", "node_id": "a" * 32}),
        "/admin/preemption/status": _ok({
            "enabled": True, "backend": "aws",
            "preempted": False, "poll_interval_seconds": 10.0,
        }),
        "/admin/output-cache-stats": _ok({
            "hits": 100, "misses": 20, "puts": 120,
            "evictions": 0, "ttl_evictions": 0,
            "size": 120, "max_entries": 1024,
            "ttl_seconds": 3600.0,
        }),
        "/peers": _ok({
            "connected": [{"peer_id": "p1", "address": "x:1"}],
            "known": [],
        }),
    }


def test_all_pass_exit_0():
    with patch(
        "httpx.get",
        side_effect=_route_get_factory(_all_pass_routes()),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_all_pass_text_shows_each_check():
    with patch(
        "httpx.get",
        side_effect=_route_get_factory(_all_pass_routes()),
    ):
        result = _invoke(["--format", "text"])
    out = result.output.lower()
    # Each of the 4 check names surfaces
    assert "daemon" in out
    assert "preemption" in out
    assert "output_cache" in out or "cache" in out
    assert "peers" in out


# ---- Daemon unreachable → exit 2 -----------------------------


def test_health_unreachable_exits_2():
    with patch(
        "httpx.get",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 2
    out = result.output.lower()
    assert "unreachable" in out


# ---- Subcheck FAIL → overall FAIL → exit 1 -------------------


def test_subcheck_fail_overall_fail_exits_1():
    routes = _all_pass_routes()
    # Make /peers return 500
    routes["/peers"] = _err(500, "internal")
    with patch(
        "httpx.get",
        side_effect=_route_get_factory(routes),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 1
    assert "FAIL" in result.output


# ---- WARN doesn't trip exit code -----------------------------


def test_warn_only_exits_0():
    """Output cache returns 503 (not configured) → that's a
    WARN (operator hasn't opted in), not a FAIL. Overall stays
    0 because no FAIL-class issues."""
    routes = _all_pass_routes()
    routes["/admin/output-cache-stats"] = _err(503, "not configured")
    routes["/admin/preemption/status"] = _err(503, "not configured")
    with patch(
        "httpx.get",
        side_effect=_route_get_factory(routes),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0, result.output
    assert "WARN" in result.output


# ---- 0 peers is WARN, not FAIL -------------------------------


def test_zero_peers_is_warn():
    """Daemon up but isolated (0 peers) → WARN."""
    routes = _all_pass_routes()
    routes["/peers"] = _ok({"connected": [], "known": []})
    with patch(
        "httpx.get",
        side_effect=_route_get_factory(routes),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0
    assert "WARN" in result.output


# ---- Preempted is WARN, not FAIL -----------------------------


def test_preempted_is_warn():
    """Daemon up + preemption flag SET → WARN ("you are
    draining"), not FAIL."""
    routes = _all_pass_routes()
    routes["/admin/preemption/status"] = _ok({
        "enabled": True, "backend": "aws",
        "preempted": True, "poll_interval_seconds": 10.0,
    })
    with patch(
        "httpx.get",
        side_effect=_route_get_factory(routes),
    ):
        result = _invoke(["--format", "text"])
    assert result.exit_code == 0
    assert "WARN" in result.output


# ---- JSON output ----------------------------------------------


def test_json_returns_structured_payload():
    with patch(
        "httpx.get",
        side_effect=_route_get_factory(_all_pass_routes()),
    ):
        result = _invoke(["--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "checks" in data
    assert isinstance(data["checks"], list)
    # Each check has a status field
    for c in data["checks"]:
        assert c["status"] in ("PASS", "WARN", "FAIL")
    assert "overall" in data
    assert data["overall"] in ("PASS", "WARN", "FAIL")

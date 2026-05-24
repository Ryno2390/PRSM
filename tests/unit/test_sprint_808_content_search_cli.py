"""Sprint 808 — `prsm content search QUERY` keyword search CLI.

Content surface gains the third piece (after sprint 805 `fetch`
+ 806 `publish`): keyword-based discovery. Wraps GET
/content/search.

  prsm content search QUERY
                     [--limit N]
                     [--min-tier low|medium|high]
                     [--exclude-new]
                     [--api-url URL]
                     [--format text|json]

Text mode: one row per result with CID + filename + creator
tier. JSON mode: full server payload.

Exit codes:
  0 — searched successfully (zero hits is OK, not error)
  1 — server error
  2 — daemon unreachable

Pin tests:
- Command registered under `content` group.
- GET hits /content/search.
- Body uses ?q=QUERY query param (NOT path arg, NOT body field).
- Default limit 20 (server-side default).
- --limit threaded through.
- --min-tier accepts low/medium/high (validated by Click).
- --exclude-new sets exclude_new=true query param.
- Empty result list → actionable "no matches" hint.
- Each result row in text mode shows CID + filename + tier.
- JSON mode returns full server payload.
- Server 422 (invalid params) → exit 1.
- Server 413 (query too long) → exit 1.
- Unreachable → exit 2.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args):
    from prsm.cli import content as _content_group
    return CliRunner().invoke(
        _content_group, ["search"] + list(args),
    )


def _success_response(results):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "results": list(results),
        "count": len(results),
    }
    return r


# ---- Command registration --------------------------------------


def test_search_command_registered():
    from prsm.cli import content as _content_group
    cmd_names = [c.name for c in _content_group.commands.values()]
    assert "search" in cmd_names


# ---- URL + query param ----------------------------------------


def test_get_url_is_content_search():
    with patch(
        "httpx.get", return_value=_success_response([]),
    ) as mg:
        result = _invoke(["hello"])
    assert result.exit_code == 0, result.output
    url = mg.call_args.args[0] if mg.call_args.args else (
        mg.call_args.kwargs.get("url")
    )
    assert url.endswith("/content/search")


def test_query_passed_as_q_param():
    with patch(
        "httpx.get", return_value=_success_response([]),
    ) as mg:
        _invoke(["my search terms"])
    params = mg.call_args.kwargs.get("params") or {}
    assert params.get("q") == "my search terms"


def test_default_limit_20():
    with patch(
        "httpx.get", return_value=_success_response([]),
    ) as mg:
        _invoke(["x"])
    params = mg.call_args.kwargs.get("params") or {}
    assert params.get("limit") in (20, "20")


def test_limit_override():
    with patch(
        "httpx.get", return_value=_success_response([]),
    ) as mg:
        _invoke(["x", "--limit", "50"])
    params = mg.call_args.kwargs.get("params") or {}
    assert params.get("limit") in (50, "50")


def test_min_tier_threaded():
    with patch(
        "httpx.get", return_value=_success_response([]),
    ) as mg:
        _invoke(["x", "--min-tier", "medium"])
    params = mg.call_args.kwargs.get("params") or {}
    assert params.get("min_tier") == "medium"


def test_exclude_new_flag():
    with patch(
        "httpx.get", return_value=_success_response([]),
    ) as mg:
        _invoke(["x", "--exclude-new"])
    params = mg.call_args.kwargs.get("params") or {}
    val = params.get("exclude_new")
    assert val in (True, "true", "True", 1, "1")


# ---- Output ----------------------------------------------------


def test_empty_results_actionable_hint():
    with patch(
        "httpx.get", return_value=_success_response([]),
    ):
        result = _invoke(["nothing"])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "no matches" in out or "0 results" in out or "no results" in out


def test_text_mode_renders_row_per_result():
    results = [
        {
            "cid": "QmA",
            "filename": "doc-a.txt",
            "creator_tier": "medium",
        },
        {
            "cid": "QmB",
            "filename": "doc-b.md",
            "creator_tier": "high",
        },
    ]
    with patch(
        "httpx.get", return_value=_success_response(results),
    ):
        result = _invoke(["x", "--format", "text"])
    assert result.exit_code == 0
    assert "QmA" in result.output
    assert "doc-a.txt" in result.output
    assert "medium" in result.output
    assert "QmB" in result.output
    assert "doc-b.md" in result.output


def test_json_mode_returns_payload():
    results = [
        {"cid": "QmA", "filename": "a.txt"},
    ]
    with patch(
        "httpx.get", return_value=_success_response(results),
    ):
        result = _invoke(["x", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["count"] == 1
    assert data["results"][0]["cid"] == "QmA"


# ---- Error paths ----------------------------------------------


def test_server_422_exits_1():
    fake = MagicMock()
    fake.status_code = 422
    fake.text = '{"detail":"limit must be in [1, 100]"}'
    with patch("httpx.get", return_value=fake):
        result = _invoke(["x", "--format", "text"])
    assert result.exit_code == 1


def test_server_413_exits_1():
    fake = MagicMock()
    fake.status_code = 413
    fake.text = '{"detail":"q size exceeds cap"}'
    with patch("httpx.get", return_value=fake):
        result = _invoke(["x", "--format", "text"])
    assert result.exit_code == 1


def test_unreachable_exits_2():
    with patch(
        "httpx.get",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke(["x", "--format", "text"])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()

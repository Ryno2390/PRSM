"""Sprint 744 F72 — /openapi.json + /docs + /redoc hidden by default.

Pre-744, FastAPI's defaults exposed:

    /docs          — Swagger UI (interactive)
    /redoc         — ReDoc UI
    /openapi.json  — machine-readable API spec

These were unauthenticated AND unconditional. Any HTTP client
could fetch /openapi.json and get the complete API surface map:
every endpoint URL, every method, every path parameter, every
body schema — INCLUDING the /admin/* paths sprints 734-743
worked to defend.

For a peered network attacker this is a roadmap: scan
/openapi.json → see /admin/parallax/streams → know it returns
expected_sender peer IDs → probe with various Origin headers
+ proxy header combinations to find a bypass. Even with all the
F65-F71 defenses, the docs ENABLED reconnaissance.

Fix: hide /docs + /redoc + /openapi.json by default. Operators
who genuinely need interactive Swagger docs in dev set
`PRSM_API_DOCS_ENABLED=1`. Default = production-safe.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock


async def _invoke(app, path: str, client_host: str = "127.0.0.1"):
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": b"",
        "headers": [],
        "client": (client_host, 12345),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
        "root_path": "",
    }
    received = [{"type": "http.request", "body": b"", "more_body": False}]
    sent = []

    async def _receive():
        if received:
            return received.pop(0)
        return {"type": "http.disconnect"}

    async def _send(msg):
        sent.append(msg)

    await app(scope, _receive, _send)
    starts = [m for m in sent if m.get("type") == "http.response.start"]
    assert starts
    return starts[0]["status"]


def _build_app():
    """Build app — env state must be set BEFORE this because
    docs_url is captured at app-construction time."""
    from prsm.node.api import create_api_app
    node = MagicMock()
    node._chain_executor_pending_streams = {}
    return create_api_app(node, enable_security=False)


def test_openapi_json_hidden_by_default():
    """Pin: env unset → /openapi.json returns 404."""
    old = os.environ.pop("PRSM_API_DOCS_ENABLED", None)
    try:
        app = _build_app()
        status = asyncio.run(_invoke(app, "/openapi.json"))
        assert status == 404, (
            f"/openapi.json must be 404 by default; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_API_DOCS_ENABLED"] = old


def test_docs_hidden_by_default():
    """Pin: env unset → /docs returns 404."""
    old = os.environ.pop("PRSM_API_DOCS_ENABLED", None)
    try:
        app = _build_app()
        status = asyncio.run(_invoke(app, "/docs"))
        assert status == 404, (
            f"/docs must be 404 by default; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_API_DOCS_ENABLED"] = old


def test_redoc_hidden_by_default():
    """Pin: env unset → /redoc returns 404."""
    old = os.environ.pop("PRSM_API_DOCS_ENABLED", None)
    try:
        app = _build_app()
        status = asyncio.run(_invoke(app, "/redoc"))
        assert status == 404, (
            f"/redoc must be 404 by default; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_API_DOCS_ENABLED"] = old


def test_openapi_json_exposed_when_env_set():
    """Operator opt-in: PRSM_API_DOCS_ENABLED=1 → /openapi.json
    returns 200 with the spec."""
    os.environ["PRSM_API_DOCS_ENABLED"] = "1"
    try:
        app = _build_app()
        status = asyncio.run(_invoke(app, "/openapi.json"))
        assert status == 200, (
            f"/openapi.json must be 200 when env=1; got {status}"
        )
    finally:
        del os.environ["PRSM_API_DOCS_ENABLED"]


def test_docs_exposed_when_env_set():
    """Operator opt-in: env=1 → /docs accessible."""
    os.environ["PRSM_API_DOCS_ENABLED"] = "1"
    try:
        app = _build_app()
        status = asyncio.run(_invoke(app, "/docs"))
        assert status == 200, (
            f"/docs must be 200 when env=1; got {status}"
        )
    finally:
        del os.environ["PRSM_API_DOCS_ENABLED"]


def test_env_value_true_yes_also_enable():
    """The env accepts 1/true/yes as truthy (consistent with
    sprint 734's PRSM_ADMIN_REMOTE_ALLOWED parsing)."""
    for val in ("true", "yes"):
        os.environ["PRSM_API_DOCS_ENABLED"] = val
        try:
            app = _build_app()
            status = asyncio.run(_invoke(app, "/openapi.json"))
            assert status == 200, (
                f"env={val!r} must enable docs; got {status}"
            )
        finally:
            del os.environ["PRSM_API_DOCS_ENABLED"]


def test_env_zero_false_no_keep_hidden():
    """Explicit 0/false/no is treated same as unset (hidden)."""
    for val in ("0", "false", "no"):
        os.environ["PRSM_API_DOCS_ENABLED"] = val
        try:
            app = _build_app()
            status = asyncio.run(_invoke(app, "/openapi.json"))
            assert status == 404, (
                f"env={val!r} must keep docs hidden; got {status}"
            )
        finally:
            del os.environ["PRSM_API_DOCS_ENABLED"]

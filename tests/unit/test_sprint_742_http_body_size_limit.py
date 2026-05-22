"""Sprint 742 F70 — HTTP body-size limit (memory-DoS defense).

Pre-742, FastAPI accepted arbitrary JSON bodies on every
endpoint — a malicious POST with a 1GB body would allocate all
of it before any application-level size check ran. HTTP-side
analog of sprints 721/725 wire-protocol size limits.

Fix: middleware checks Content-Length header BEFORE the body is
read. Returns 413 Payload Too Large with actionable error if
the header exceeds PRSM_HTTP_MAX_BODY_BYTES (default 1 MiB).
Setting <=0 disables; non-int defaults to 1 MiB.

Limitations honestly:
- Only inspects Content-Length. A client sending chunked
  transfer-encoding could bypass the pre-read check. The
  fix bounds the common case (POSTed JSON); chunked attacks
  still need a separate defense (typically at the reverse-
  proxy layer, e.g., `client_max_body_size` in nginx).
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock


async def _invoke(app, path: str, content_length: int):
    """Helper: POST with synthetic Content-Length header."""
    headers = [(b"content-length", str(content_length).encode())]
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
        "root_path": "",
    }
    # Don't actually send a body — we want to test the
    # Content-Length pre-read gate. The handler shouldn't be
    # reached when content-length exceeds limit.
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
    from prsm.node.api import create_api_app
    node = MagicMock()
    node._chain_executor_pending_streams = {}
    return create_api_app(node, enable_security=False)


def test_body_under_limit_passes_through():
    """Sanity: a request with Content-Length WELL under the
    default 1 MiB limit reaches the inner handler (not 413).
    May be 4xx for other reasons (validation, auth) but NOT 413."""
    app = _build_app()
    old = os.environ.pop("PRSM_HTTP_MAX_BODY_BYTES", None)
    try:
        status = asyncio.run(_invoke(
            app, "/compute/inference", content_length=1024,
        ))
        assert status != 413, (
            f"1KB body must not be 413 with default 1 MiB limit; "
            f"got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_HTTP_MAX_BODY_BYTES"] = old


def test_body_over_default_limit_returns_413():
    """Behavioral: a request with Content-Length above 1 MiB
    (default) returns 413 with actionable error referencing the
    env var. The handler is NEVER invoked."""
    app = _build_app()
    old = os.environ.pop("PRSM_HTTP_MAX_BODY_BYTES", None)
    try:
        # 2 MiB > 1 MiB default
        status = asyncio.run(_invoke(
            app, "/compute/inference",
            content_length=2 * 1024 * 1024,
        ))
        assert status == 413, (
            f"2 MiB body must return 413 with default 1 MiB limit; "
            f"got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_HTTP_MAX_BODY_BYTES"] = old


def test_explicit_env_override_respected():
    """Setting PRSM_HTTP_MAX_BODY_BYTES=10485760 (10 MiB) allows
    a 5 MiB body that would be rejected at default 1 MiB."""
    app = _build_app()
    os.environ["PRSM_HTTP_MAX_BODY_BYTES"] = str(10 * 1024 * 1024)
    try:
        status = asyncio.run(_invoke(
            app, "/compute/inference",
            content_length=5 * 1024 * 1024,
        ))
        assert status != 413, (
            f"5 MiB body must not be 413 with 10 MiB override; "
            f"got {status}"
        )
    finally:
        del os.environ["PRSM_HTTP_MAX_BODY_BYTES"]


def test_zero_env_disables_limit():
    """PRSM_HTTP_MAX_BODY_BYTES=0 → no limit (pre-742 behavior).
    Operators who explicitly want unbounded bodies can opt-in."""
    app = _build_app()
    os.environ["PRSM_HTTP_MAX_BODY_BYTES"] = "0"
    try:
        # Massive body — should NOT be 413 because limit disabled
        status = asyncio.run(_invoke(
            app, "/compute/inference",
            content_length=100 * 1024 * 1024,
        ))
        assert status != 413, (
            f"100 MiB body must pass with limit=0; got {status}"
        )
    finally:
        del os.environ["PRSM_HTTP_MAX_BODY_BYTES"]


def test_non_int_env_safely_defaults():
    """Bad env value → safe-default 1 MiB rather than failing
    startup or disabling the gate."""
    app = _build_app()
    os.environ["PRSM_HTTP_MAX_BODY_BYTES"] = "one-megabyte"
    try:
        status = asyncio.run(_invoke(
            app, "/compute/inference",
            content_length=2 * 1024 * 1024,
        ))
        assert status == 413, (
            f"non-int env must fall back to 1 MiB default + reject 2 MiB; "
            f"got {status}"
        )
    finally:
        del os.environ["PRSM_HTTP_MAX_BODY_BYTES"]


def test_error_response_references_env_var():
    """Pin: the 413 response body must reference the env var so
    operators reading logs know what to tune."""
    import inspect
    from prsm.node import api as _api
    src = inspect.getsource(_api.create_api_app)
    assert "PRSM_HTTP_MAX_BODY_BYTES" in src
    assert "exceeds max bytes" in src


def test_middleware_applied_to_all_paths_not_just_admin():
    """Pin: the body-size gate is global, not /admin/* scoped.
    `/compute/*` endpoints (the primary production-traffic
    surface) MUST be protected."""
    app = _build_app()
    old = os.environ.pop("PRSM_HTTP_MAX_BODY_BYTES", None)
    try:
        for path in (
            "/compute/inference",
            "/compute/forge",
            "/compute/inference/stream",
        ):
            status = asyncio.run(_invoke(
                app, path, content_length=2 * 1024 * 1024,
            ))
            assert status == 413, (
                f"path {path} must be protected by body-size "
                f"gate; got {status}"
            )
    finally:
        if old is not None:
            os.environ["PRSM_HTTP_MAX_BODY_BYTES"] = old

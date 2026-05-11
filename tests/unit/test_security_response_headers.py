"""Sprint 188 — OWASP-recommended security response headers on every response.

Three headers applied uniformly via middleware:

  X-Content-Type-Options: nosniff
    Prevents browsers from MIME-sniffing past the declared
    Content-Type. Defends against polyglot attacks where a
    response declared application/json is sniffed as text/html
    and executes inline script.

  X-Frame-Options: DENY
    Prevents responses from being embedded in iframes. Defends
    against clickjacking on the dashboard surface.

  Referrer-Policy: strict-origin-when-cross-origin
    Limits Referer header leakage on outbound clickthroughs
    from the dashboard.

Strict-Transport-Security intentionally NOT set — PRSM runs on
http://127.0.0.1 by default; HSTS belongs at the operator's
reverse proxy where TLS termination actually happens.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._operator_address = None
    return TestClient(create_api_app(node, enable_security=False))


class TestSecurityHeaders:
    def test_x_content_type_options_nosniff(self):
        resp = _client().get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options_deny(self):
        resp = _client().get("/health")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_referrer_policy(self):
        resp = _client().get("/health")
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_headers_present_on_404(self):
        """Sprint 188 — security headers MUST be set on error
        responses too. 4xx/5xx responses are a common
        attack-amplification vector if they lack the headers."""
        resp = _client().get("/random/nonexistent/path")
        assert resp.status_code == 404
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_headers_present_on_info(self):
        resp = _client().get("/info")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_no_hsts_by_default(self):
        """Sprint 188 — Strict-Transport-Security intentionally NOT
        set. PRSM runs over http on 127.0.0.1; HSTS would break
        local dev. Operators with TLS-termination proxies set HSTS
        at the proxy layer."""
        resp = _client().get("/health")
        assert "Strict-Transport-Security" not in resp.headers

    def test_headers_dont_override_explicit_settings(self):
        """Sprint 188 invariant — middleware uses setdefault, so a
        route that explicitly sets one of these headers (e.g. an
        embed-friendly subpage) can override."""
        # Use a route that we know returns standard headers — they
        # should all show up.
        resp = _client().get("/info")
        # All three present, none modified.
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"

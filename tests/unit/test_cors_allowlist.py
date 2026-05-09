"""CORS allowlist via PRSM_ALLOWED_ORIGINS env var.

Production-hardening for nodes serving browser-based clients
(operator dashboards, prsm-ui, etc.). Operator declares the
explicit list of origins permitted to make cross-origin
requests; everything else gets blocked at the CORS layer
before reaching any endpoint.

Default behavior (env unset): permissive `*` allowlist preserves
v1 dev-friendly behavior bit-identically.

Comma-separated form:
  export PRSM_ALLOWED_ORIGINS="https://dash.example.com,https://ops.example.com"
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._royalty_distributor_client = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Default behavior — env unset
# ──────────────────────────────────────────────────────────────────────


class TestCORSDefault:
    def test_no_env_set_permits_any_origin(self):
        """Without PRSM_ALLOWED_ORIGINS, any origin is allowed
        (matches v1 dev-friendly behavior bit-identically)."""
        node = _node()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_ALLOWED_ORIGINS", None)
            client = _client(node)
            resp = client.get(
                "/health",
                headers={"Origin": "https://random-untrusted.example.com"},
            )
        assert resp.status_code == 200
        # Permissive default: ACAO header echoes back any origin.
        assert resp.headers.get("access-control-allow-origin") in {
            "*", "https://random-untrusted.example.com",
        }


# ──────────────────────────────────────────────────────────────────────
# Allowlist enforcement
# ──────────────────────────────────────────────────────────────────────


class TestCORSAllowlist:
    def test_listed_origin_gets_acao_header(self):
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_ALLOWED_ORIGINS": "https://dash.example.com,https://ops.example.com",
        }):
            client = _client(node)
            resp = client.get(
                "/health",
                headers={"Origin": "https://dash.example.com"},
            )
        assert resp.status_code == 200
        # Listed origin echoed in ACAO.
        assert resp.headers.get("access-control-allow-origin") == \
            "https://dash.example.com"

    def test_unlisted_origin_gets_no_acao(self):
        """Origin not in allowlist → no ACAO header on response.
        Browser blocks the response client-side."""
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_ALLOWED_ORIGINS": "https://dash.example.com",
        }):
            client = _client(node)
            resp = client.get(
                "/health",
                headers={"Origin": "https://attacker.example.com"},
            )
        # Response succeeds server-side, but no ACAO header → browser
        # blocks the response from reaching JS code.
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers

    def test_multiple_origins_each_handled_independently(self):
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_ALLOWED_ORIGINS": "https://a.example.com,https://b.example.com",
        }):
            client = _client(node)
            resp_a = client.get(
                "/health",
                headers={"Origin": "https://a.example.com"},
            )
            resp_b = client.get(
                "/health",
                headers={"Origin": "https://b.example.com"},
            )
            resp_c = client.get(
                "/health",
                headers={"Origin": "https://c.example.com"},
            )
        assert resp_a.headers.get("access-control-allow-origin") == \
            "https://a.example.com"
        assert resp_b.headers.get("access-control-allow-origin") == \
            "https://b.example.com"
        assert "access-control-allow-origin" not in resp_c.headers


# ──────────────────────────────────────────────────────────────────────
# CORS preflight (OPTIONS)
# ──────────────────────────────────────────────────────────────────────


class TestCORSPreflight:
    def test_preflight_allowed_origin_returns_200(self):
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_ALLOWED_ORIGINS": "https://dash.example.com",
        }):
            client = _client(node)
            resp = client.options(
                "/health",
                headers={
                    "Origin": "https://dash.example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
        # Preflight returns 200 with ACAO + ACAM headers.
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == \
            "https://dash.example.com"


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────


class TestCORSEdgeCases:
    def test_whitespace_in_csv_trimmed(self):
        """Operator pasting `"a , b , c"` works the same as
        `"a,b,c"` — origins are trimmed."""
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_ALLOWED_ORIGINS": " https://a.example.com , https://b.example.com ",
        }):
            client = _client(node)
            resp = client.get(
                "/health",
                headers={"Origin": "https://a.example.com"},
            )
        assert resp.headers.get("access-control-allow-origin") == \
            "https://a.example.com"

    def test_empty_csv_treated_as_unset(self):
        """Empty value (or all-whitespace) is treated as
        unset — falls back to permissive default."""
        node = _node()
        with patch.dict(os.environ, {"PRSM_ALLOWED_ORIGINS": "   "}):
            client = _client(node)
            resp = client.get(
                "/health",
                headers={"Origin": "https://anything.example.com"},
            )
        # Permissive fallback.
        assert resp.headers.get("access-control-allow-origin") in {
            "*", "https://anything.example.com",
        }

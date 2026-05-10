"""Sprint 160 — /content/upload Pydantic field constraints on
royalty_rate and replicas.

Pre-fix the request model had:
  royalty_rate: Optional[float] = Field(default=None,
      description="FTNS earned per access (0.001-0.1, default 0.01)")
  replicas: int = 3

— no actual numeric constraints. So a body with:
  royalty_rate: -1.0   — negative royalty
  royalty_rate: 100.0  — 10000% per access (above any sane cap)
  replicas: -5         — negative replication count

would all pass model validation, then either silently produce
wrong on-chain royalty splits (if the publisher path tolerated
the value) or hit a generic downstream error. Live dogfood:
  curl -d '{"text":"hi","royalty_rate":-1.0}' /content/upload
  → 503 (ContentPublisher not wired)  — but on a fully-wired
    node this would have leaked through.

Fix: add Pydantic Field(ge=..., le=...) constraints so the
422 fires at request validation time, before any downstream
subsystem check.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    """Node WITHOUT content_uploader wired so post-validation 503
    wouldn't mask Pydantic 422s. Sprint 160 invariant: Pydantic
    validation fires BEFORE the uploader-availability checks."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.content_uploader = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _post(node, body):
    return _client(node).post("/content/upload", json=body)


class TestContentUploadFieldValidation:
    def test_royalty_rate_below_min_returns_422(self):
        resp = _post(_node(), {
            "text": "hello", "royalty_rate": -1.0,
        })
        assert resp.status_code == 422
        assert "royalty_rate" in str(resp.json()).lower()

    def test_royalty_rate_above_max_returns_422(self):
        resp = _post(_node(), {
            "text": "hello", "royalty_rate": 100.0,
        })
        assert resp.status_code == 422
        assert "royalty_rate" in str(resp.json()).lower()

    def test_royalty_rate_at_min_passes(self):
        """Boundary — 0.001 should pass."""
        resp = _post(_node(), {
            "text": "hello", "royalty_rate": 0.001,
        })
        # content_uploader=None → 503; that's downstream of validation
        assert resp.status_code == 503

    def test_royalty_rate_at_max_passes(self):
        """Boundary — 0.1 should pass."""
        resp = _post(_node(), {
            "text": "hello", "royalty_rate": 0.1,
        })
        assert resp.status_code == 503

    def test_negative_replicas_returns_422(self):
        resp = _post(_node(), {
            "text": "hello", "replicas": -3,
        })
        assert resp.status_code == 422
        assert "replicas" in str(resp.json()).lower()

    def test_zero_replicas_accepted_local_only(self):
        """0 replicas = content stored locally only — preserved
        semantic from pre-sprint-160; NOT a validation error."""
        resp = _post(_node(), {
            "text": "hello", "replicas": 0,
        })
        assert resp.status_code == 503  # downstream of validation

    def test_excessive_replicas_returns_422(self):
        """Pydantic-level cap: 1000 covers any realistic scenario.
        env-tunable PRSM_MAX_REPLICAS still wins for tighter caps."""
        resp = _post(_node(), {
            "text": "hello", "replicas": 5000,
        })
        assert resp.status_code == 422

    def test_royalty_rate_omitted_passes(self):
        """Optional — None default still works."""
        resp = _post(_node(), {"text": "hello"})
        assert resp.status_code == 503

    def test_replicas_default_passes(self):
        """Default 3 still works."""
        resp = _post(_node(), {"text": "hello"})
        assert resp.status_code == 503

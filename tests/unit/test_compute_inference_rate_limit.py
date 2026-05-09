"""POST /compute/inference — per-requester rate limiting via
PRSM_INFERENCE_MAX_RPS_PER_REQUESTER. Independent bucket from
/compute/forge so operators can tune the two endpoints separately.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.rate_limiter import reset_global_bucket


def _node():
    """Bare node — inference_executor None so the rate-cap check fires
    BEFORE the 503 inference-not-configured check (same precedence
    as forge's rate-cap-before-agent_forge ordering)."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.inference_executor = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def setup_function():
    """Reset all rate-limit buckets between tests."""
    reset_global_bucket()


# ──────────────────────────────────────────────────────────────────────
# Cap enforcement
# ──────────────────────────────────────────────────────────────────────


class TestInferenceRateLimit:
    def test_burst_then_429(self):
        """cap=2/sec, burst=2: 3rd request rejects with 429."""
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER": "2",
        }):
            client = _client(node)
            for _ in range(2):
                resp = client.post(
                    "/compute/inference",
                    json={"prompt": "x", "model_id": "m"},
                )
                # Through to 503 (inference_executor None) — cap
                # not exceeded.
                assert resp.status_code == 503
            resp = client.post(
                "/compute/inference",
                json={"prompt": "x", "model_id": "m"},
            )
            assert resp.status_code == 429
            assert "/compute/inference" in resp.json()["detail"]

    def test_no_env_disables_limit(self):
        node = _node()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_INFERENCE_MAX_RPS_PER_REQUESTER", None)
            client = _client(node)
            for _ in range(10):
                resp = client.post(
                    "/compute/inference",
                    json={"prompt": "x", "model_id": "m"},
                )
                assert resp.status_code == 503  # never 429

    def test_invalid_env_disables_limit(self):
        node = _node()
        with patch.dict(os.environ, {
            "PRSM_INFERENCE_MAX_RPS_PER_REQUESTER": "not_a_number",
        }):
            client = _client(node)
            for _ in range(10):
                resp = client.post(
                    "/compute/inference",
                    json={"prompt": "x", "model_id": "m"},
                )
                assert resp.status_code == 503  # never 429


# ──────────────────────────────────────────────────────────────────────
# Independent buckets — forge vs inference
# ──────────────────────────────────────────────────────────────────────


class TestSeparateBuckets:
    def test_named_buckets_independent_at_module_level(self):
        """Direct test of the rate_limiter module: buckets keyed
        by name are independent. Drains one without affecting
        the other."""
        from prsm.node.rate_limiter import get_or_build_bucket

        reset_global_bucket()
        forge_bucket = get_or_build_bucket(2.0, name="forge")
        inference_bucket = get_or_build_bucket(2.0, name="inference")

        # Different objects.
        assert forge_bucket is not inference_bucket

        # Drain forge bucket fully.
        for _ in range(2):
            assert forge_bucket.try_consume("req-1") is True
        assert forge_bucket.try_consume("req-1") is False

        # Inference bucket is still full for the same requester.
        assert inference_bucket.try_consume("req-1") is True
        assert inference_bucket.try_consume("req-1") is True
        assert inference_bucket.try_consume("req-1") is False

    def test_default_name_preserves_legacy_aliases(self):
        """get_or_build_bucket(rate) without a name uses
        '_default' bucket and updates the legacy module-level
        _GLOBAL_BUCKET / _GLOBAL_RATE aliases for backwards
        compatibility."""
        import prsm.node.rate_limiter as rl

        reset_global_bucket()
        bucket = rl.get_or_build_bucket(3.0)
        assert bucket is rl._GLOBAL_BUCKET
        assert rl._GLOBAL_RATE == 3.0

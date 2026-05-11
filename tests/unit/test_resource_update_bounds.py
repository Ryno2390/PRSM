"""Sprint 207 — ResourceUpdateRequest bounds.

Pre-fix gaps in ResourceUpdateRequest:
  - storage_gb: gt=0 only (no upper bound, no allow_inf_nan)
  - max_concurrent_jobs: ge=1 only (no upper bound)
  - upload_mbps_limit / download_mbps_limit: ge=0 only (no upper,
                                              no allow_inf_nan)
  - active_days: List[int] no max-items cap

A million-element active_days list ties up the per-item loop.
max_concurrent_jobs=1e9 would crash schedulers downstream.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.config = MagicMock()
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_excessive_max_concurrent_jobs_rejected():
    resp = _client().put("/node/resources", json={
        "max_concurrent_jobs": 10_000_000,
    })
    assert resp.status_code == 422


def test_excessive_storage_gb_rejected():
    resp = _client().put("/node/resources", json={
        "storage_gb": 1e15,
    })
    assert resp.status_code == 422


def test_excessive_bandwidth_rejected():
    resp = _client().put("/node/resources", json={
        "upload_mbps_limit": 1e15,
    })
    assert resp.status_code == 422


def test_excessive_active_days_length_rejected():
    resp = _client().put("/node/resources", json={
        "active_days": [0] * 1000,
    })
    assert resp.status_code == 422


def test_typical_passes():
    resp = _client().put("/node/resources", json={
        "max_concurrent_jobs": 10,
        "storage_gb": 100.0,
        "upload_mbps_limit": 100.0,
        "active_days": [0, 1, 2, 3, 4],
    })
    assert resp.status_code != 422

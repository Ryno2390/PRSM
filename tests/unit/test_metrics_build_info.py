"""prsm_build_info gauge in /metrics (sprint 111)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._provenance_client = None
    node._royalty_distributor_client = None
    node._webhook_log = None
    node._slash_event_log = None
    node._heartbeat_log = None
    node._distribution_log = None
    return node


class TestBuildInfo:
    def test_emits_build_info_gauge(self):
        client = TestClient(create_api_app(_node(), enable_security=False))
        text = client.get("/metrics").text
        assert "# HELP prsm_build_info" in text
        assert "# TYPE prsm_build_info gauge" in text
        assert 'prsm_build_info{version="' in text
        assert text.rstrip().endswith("1") or "} 1" in text

    def test_version_matches_pyproject(self):
        from pathlib import Path
        repo_root = Path(__file__).parent.parent.parent
        pyproject = repo_root / "pyproject.toml"
        for line in pyproject.read_text().splitlines():
            if line.startswith("version = "):
                expected = line.split("=", 1)[1].strip().strip('"')
                break
        else:
            pytest.skip("Could not read version from pyproject.toml")
        client = TestClient(create_api_app(_node(), enable_security=False))
        text = client.get("/metrics").text
        # Either matches package metadata OR "unknown" (source-only)
        assert (
            f'version="{expected}"' in text
            or 'version="unknown"' in text
        )

    def test_value_is_one(self):
        # Standard Prometheus build_info pattern: gauge always 1
        client = TestClient(create_api_app(_node(), enable_security=False))
        text = client.get("/metrics").text
        # Find the build_info line specifically
        for line in text.splitlines():
            if line.startswith("prsm_build_info"):
                assert line.endswith(" 1"), (
                    f"Expected build_info to end with ' 1', got: {line}"
                )
                return
        pytest.fail("prsm_build_info line not found in /metrics output")

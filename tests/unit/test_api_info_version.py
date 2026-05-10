"""GET /api-info returns canonical version (sprint 110)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    return node


class TestApiInfoVersion:
    def test_version_matches_pyproject(self):
        from pathlib import Path
        # Read pyproject.toml version
        repo_root = Path(__file__).parent.parent.parent
        pyproject = repo_root / "pyproject.toml"
        text = pyproject.read_text()
        for line in text.splitlines():
            if line.startswith("version = "):
                expected = line.split("=", 1)[1].strip().strip('"')
                break
        else:
            pytest.skip("Could not read version from pyproject.toml")

        client = TestClient(create_api_app(_node(), enable_security=False))
        body = client.get("/api-info").json()
        # Either matches the package version OR "unknown" if
        # package not installed (source-only run).
        assert body["version"] in (expected, "unknown")

    def test_version_not_stale_0_2_0(self):
        """Regression guard: prevent regression to the hardcoded
        '0.2.0' that was wrong for years (sprint 110 fix)."""
        client = TestClient(create_api_app(_node(), enable_security=False))
        body = client.get("/api-info").json()
        assert body["version"] != "0.2.0", (
            "Version regressed to the stale 0.2.0 hardcode"
        )

    def test_other_fields_present(self):
        client = TestClient(create_api_app(_node(), enable_security=False))
        body = client.get("/api-info").json()
        assert body["name"] == "PRSM Node API"
        assert "docs" in body
        assert "openapi" in body

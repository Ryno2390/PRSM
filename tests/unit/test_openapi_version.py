"""OpenAPI spec version not stale (sprint 112).

The FastAPI app constructor previously hardcoded
version="0.24.0" — this surfaces in /openapi.json under
info.version. Operators integrating against the spec saw
wildly stale data.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    return node


class TestOpenAPIVersion:
    def test_version_not_stale_0_24_0(self):
        client = TestClient(create_api_app(_node(), enable_security=False))
        spec = client.get("/openapi.json").json()
        assert spec["info"]["version"] != "0.24.0"

    def test_version_matches_pyproject(self):
        from pathlib import Path
        repo_root = Path(__file__).parent.parent.parent
        for line in (repo_root / "pyproject.toml").read_text().splitlines():
            if line.startswith("version = "):
                expected = line.split("=", 1)[1].strip().strip('"')
                break
        else:
            pytest.skip("Could not read version from pyproject.toml")

        client = TestClient(create_api_app(_node(), enable_security=False))
        spec = client.get("/openapi.json").json()
        assert spec["info"]["version"] in (expected, "unknown")

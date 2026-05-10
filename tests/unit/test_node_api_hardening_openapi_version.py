"""api_hardening custom_openapi version not stale (sprint 131).

Pre-fix: api_hardening.custom_openapi() set the OpenAPI spec
version to a hardcoded "1.0.0", clobbering the
importlib-derived version that sprint 112 wired into the
FastAPI constructor. /openapi.json reported 1.0.0 even though
all other surfaces correctly reported 1.7.0.

Caught by user inspection during dogfood — eyeballing
/openapi.json showed 1.0.0 vs /api-info's 1.7.0.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.api_hardening import generate_openapi_schema


class TestOpenAPIVersion:
    def test_version_not_stale_1_0_0(self):
        # Build a minimal FastAPI-app-like stub
        app = MagicMock()
        app.openapi_schema = None
        app.routes = []
        app.title = "PRSM Node API"
        app.summary = None
        app.description = ""
        app.version = "0.0.0"  # constructor value (overridden by helper)

        schema = generate_openapi_schema(app)
        assert schema["info"]["version"] != "1.0.0"

    def test_version_matches_pyproject(self):
        from pathlib import Path
        repo = Path(__file__).parent.parent.parent
        expected = None
        for line in (repo / "pyproject.toml").read_text().splitlines():
            if line.startswith("version = "):
                expected = line.split("=", 1)[1].strip().strip('"')
                break
        if expected is None:
            pytest.skip("Could not read pyproject.toml")

        app = MagicMock()
        app.openapi_schema = None
        app.routes = []
        app.title = "PRSM Node API"
        app.summary = None
        app.description = ""
        app.version = "0.0.0"

        schema = generate_openapi_schema(app)
        # Either canonical OR "unknown" (source-only)
        assert schema["info"]["version"] in (expected, "unknown")

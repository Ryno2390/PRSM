"""MCP server version not stale (sprint 113)."""
from __future__ import annotations

import pytest

from prsm.mcp_server import create_server


class TestMcpServerVersion:
    def test_server_version_not_stale_0_39_0(self):
        server = create_server()
        # Server.version is the field passed at construction
        assert getattr(server, "version", None) != "0.39.0"

    def test_server_version_matches_pyproject(self):
        from pathlib import Path
        repo_root = Path(__file__).parent.parent.parent
        for line in (repo_root / "pyproject.toml").read_text().splitlines():
            if line.startswith("version = "):
                expected = line.split("=", 1)[1].strip().strip('"')
                break
        else:
            pytest.skip("Could not read version from pyproject.toml")

        server = create_server()
        actual = getattr(server, "version", None)
        assert actual in (expected, "unknown")

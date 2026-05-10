"""Interface API stale version surfaces fixed (sprint 114).

Two more locations carrying the stale '0.2.0':
  prsm/interface/api/main.py  (logger.info call)
  prsm/interface/api/openapi_config.py (custom_openapi_schema)
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _pyproject_version():
    repo_root = Path(__file__).parent.parent.parent
    for line in (repo_root / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    return None


class TestInterfaceMainNoStale:
    """The logger.info version arg should resolve at runtime via
    importlib.metadata, not be a hardcoded '0.2.0'."""

    def test_no_hardcoded_0_2_0_string_in_module(self):
        from pathlib import Path
        repo_root = Path(__file__).parent.parent.parent
        text = (repo_root / "prsm/interface/api/main.py").read_text()
        # Regression guard: the literal "0.2.0" should not
        # appear (the fix replaced it with importlib.metadata)
        assert '"0.2.0"' not in text


class TestOpenAPIConfigNoStale:
    def test_no_hardcoded_0_2_0_in_openapi_config(self):
        from pathlib import Path
        repo_root = Path(__file__).parent.parent.parent
        text = (
            repo_root / "prsm/interface/api/openapi_config.py"
        ).read_text()
        # The hardcoded version="0.2.0" parameter line should be gone
        for line in text.splitlines():
            if 'version="0.2.0"' in line:
                pytest.fail(
                    f"Stale version='0.2.0' still in openapi_config: "
                    f"{line!r}"
                )

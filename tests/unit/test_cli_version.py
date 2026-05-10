"""prsm version CLI command (sprint 115)."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from prsm.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def _pyproject_version():
    repo_root = Path(__file__).parent.parent.parent
    for line in (repo_root / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    return None


class TestVersion:
    def test_prints_version(self, runner):
        expected = _pyproject_version()
        if expected is None:
            pytest.skip("Could not read pyproject.toml")
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0
        assert expected in result.output
        assert "PRSM" in result.output

    def test_no_stale_0_2_0(self, runner):
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0
        assert "0.2.0" not in result.output

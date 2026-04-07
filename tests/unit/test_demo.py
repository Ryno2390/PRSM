"""Tests for the PRSM demo script."""

import pytest
from click.testing import CliRunner


class TestDemo:
    def test_demo_command_exists(self):
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "--help"])
        assert result.exit_code == 0
        assert "demo" in result.output.lower()

    def test_demo_runs_without_error(self):
        """Demo should complete without errors (no network needed)."""
        from prsm.demo import run_demo
        import asyncio
        # Run the demo -- it should not raise
        asyncio.run(run_demo())

    def test_getting_started_doc_exists(self):
        """GETTING_STARTED.md should exist in docs/."""
        from pathlib import Path
        doc = Path("docs/GETTING_STARTED.md")
        assert doc.exists(), "docs/GETTING_STARTED.md not found"
        content = doc.read_text()
        assert "pip install prsm-network" in content
        assert "OPENROUTER_API_KEY" in content
        assert "prsm node start" in content

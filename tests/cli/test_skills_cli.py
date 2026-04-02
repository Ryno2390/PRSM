"""Tests for prsm.cli_modules.skills_cli — Click commands via CliRunner."""
import json
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from prsm.cli_modules.skills_cli import skills, _get_registry


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_registry(registry):
    """Patch _get_registry to return our pre-loaded registry fixture."""
    with patch("prsm.cli_modules.skills_cli._get_registry", return_value=registry):
        yield registry


class TestSkillsList:
    def test_list_shows_all_skills(self, runner, mock_registry):
        result = runner.invoke(skills, ["list"])
        assert result.exit_code == 0
        assert "prsm-datasets" in result.output
        assert "prsm-compute" in result.output
        assert "prsm-network" in result.output

    def test_list_shows_counts(self, runner, mock_registry):
        result = runner.invoke(skills, ["list"])
        assert "3 packages" in result.output
        assert "12 tools" in result.output

    def test_list_empty_registry(self, runner):
        from prsm.skills.registry import SkillRegistry
        empty = SkillRegistry(load_builtins=False)
        with patch("prsm.cli_modules.skills_cli._get_registry", return_value=empty):
            result = runner.invoke(skills, ["list"])
            assert result.exit_code == 0
            assert "No skill packages" in result.output


class TestSkillsInfo:
    def test_info_existing_package(self, runner, mock_registry):
        result = runner.invoke(skills, ["info", "prsm-datasets"])
        assert result.exit_code == 0
        assert "prsm-datasets" in result.output
        assert "prsm_search_datasets" in result.output

    def test_info_nonexistent_package(self, runner, mock_registry):
        result = runner.invoke(skills, ["info", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "not found" in (result.stderr or "").lower()

    def test_info_shows_tools(self, runner, mock_registry):
        result = runner.invoke(skills, ["info", "prsm-compute"])
        assert result.exit_code == 0
        assert "prsm_submit_job" in result.output


class TestSkillsSearch:
    def test_search_by_name(self, runner, mock_registry):
        result = runner.invoke(skills, ["search", "datasets"])
        assert result.exit_code == 0
        assert "prsm-datasets" in result.output
        assert "1 result" in result.output

    def test_search_no_results(self, runner, mock_registry):
        result = runner.invoke(skills, ["search", "zzzznothing"])
        assert result.exit_code == 0
        assert "No skills matching" in result.output

    def test_search_broad(self, runner, mock_registry):
        result = runner.invoke(skills, ["search", "prsm"])
        assert result.exit_code == 0
        assert "3 results" in result.output


class TestSkillsExport:
    def test_export_valid_package(self, runner, mock_registry):
        result = runner.invoke(skills, ["export", "prsm-datasets"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "prsm-datasets"
        assert len(data["tools"]) == 4
        # Verify tool structure
        tool = data["tools"][0]
        assert "name" in tool
        assert "inputSchema" in tool
        assert "properties" in tool["inputSchema"]

    def test_export_nonexistent(self, runner, mock_registry):
        result = runner.invoke(skills, ["export", "nonexistent"])
        assert result.exit_code != 0

    def test_export_json_is_valid(self, runner, mock_registry):
        result = runner.invoke(skills, ["export", "prsm-compute"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["version"] == "1.0.0"
        assert "capabilities" in data

    def test_export_all_three_packages(self, runner, mock_registry):
        for pkg in ("prsm-datasets", "prsm-compute", "prsm-network"):
            result = runner.invoke(skills, ["export", pkg])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["name"] == pkg
            assert len(data["tools"]) == 4

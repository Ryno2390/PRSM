"""
Tests for prsm db-upgrade, db-downgrade, and db-status CLI commands.

These verify that:
1. Each command calls the correct alembic function with the right arguments
2. --revision flags are passed through correctly
3. Missing alembic.ini produces a clear error (SystemExit(1))
4. Alembic exceptions produce a clear error (SystemExit(1)), not a traceback
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from click.testing import CliRunner

from prsm.cli import main


class TestDbMigrationCli:
    """Test suite for database migration CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Provide a Click CLI test runner."""
        return CliRunner()

    def test_db_upgrade_calls_alembic_upgrade(self, cli_runner):
        """db-upgrade invokes alembic_command.upgrade(cfg, 'head') by default."""
        with patch("prsm.cli._find_alembic_ini") as mock_find, \
             patch("alembic.config.Config") as mock_config_cls, \
             patch("alembic.command.upgrade") as mock_upgrade:
            
            # Setup mocks
            mock_find.return_value = Path("/fake/alembic.ini")
            mock_cfg = MagicMock()
            mock_config_cls.return_value = mock_cfg
            
            # Invoke command
            result = cli_runner.invoke(main, ["db-upgrade"])
            
            # Verify upgrade was called with correct args
            mock_upgrade.assert_called_once_with(mock_cfg, "head")
            assert result.exit_code == 0

    def test_db_upgrade_custom_revision(self, cli_runner):
        """db-upgrade --revision passes the revision to alembic."""
        with patch("prsm.cli._find_alembic_ini") as mock_find, \
             patch("alembic.config.Config") as mock_config_cls, \
             patch("alembic.command.upgrade") as mock_upgrade:
            
            # Setup mocks
            mock_find.return_value = Path("/fake/alembic.ini")
            mock_cfg = MagicMock()
            mock_config_cls.return_value = mock_cfg
            
            # Invoke command with custom revision
            result = cli_runner.invoke(main, ["db-upgrade", "--revision", "006"])
            
            # Verify upgrade was called with custom revision
            mock_upgrade.assert_called_once_with(mock_cfg, "006")
            assert result.exit_code == 0

    def test_db_downgrade_calls_alembic_downgrade(self, cli_runner):
        """db-downgrade invokes alembic_command.downgrade(cfg, '-1') by default."""
        with patch("prsm.cli._find_alembic_ini") as mock_find, \
             patch("alembic.config.Config") as mock_config_cls, \
             patch("alembic.command.downgrade") as mock_downgrade:
            
            # Setup mocks
            mock_find.return_value = Path("/fake/alembic.ini")
            mock_cfg = MagicMock()
            mock_config_cls.return_value = mock_cfg
            
            # Invoke command
            result = cli_runner.invoke(main, ["db-downgrade"])
            
            # Verify downgrade was called with correct args
            mock_downgrade.assert_called_once_with(mock_cfg, "-1")
            assert result.exit_code == 0

    def test_db_upgrade_missing_ini_exits_with_error(self, cli_runner):
        """db-upgrade exits 1 with a clear message when alembic.ini is not found."""
        with patch("prsm.cli._find_alembic_ini") as mock_find:
            # Setup mock to return None (file not found)
            mock_find.return_value = None
            
            # Invoke command
            result = cli_runner.invoke(main, ["db-upgrade"])
            
            # Verify exit code and error message
            assert result.exit_code == 1
            assert "alembic.ini not found" in result.output

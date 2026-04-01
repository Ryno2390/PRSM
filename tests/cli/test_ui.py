"""Tests for prsm.cli_modules.ui — UI toolkit with mocked console output."""
import pytest
from unittest.mock import patch, MagicMock

from prsm.cli_modules import ui
from prsm.cli_modules.theme import ICONS


class TestShowBanner:
    def test_show_banner_runs(self, mock_console):
        ui.show_banner("1.2.3")
        assert len(mock_console) > 0

    def test_show_banner_compact(self, mock_console, monkeypatch):
        import prsm.cli_modules.ui as _ui_mod
        monkeypatch.setattr(_ui_mod.shutil, "get_terminal_size", lambda: MagicMock(columns=50))
        ui.show_banner()
        assert len(mock_console) > 0


class TestSection:
    def test_section_prints_icon_and_title(self, mock_console):
        ui.section("Test Section", icon="bullet")
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert "Test Section" in texts
        assert ICONS["bullet"] in texts

    def test_section_with_custom_icon(self, mock_console):
        ui.section("Net", icon="network")
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert ICONS["network"] in texts


class TestStepHeader:
    def test_step_header_prints_progress(self, mock_console):
        ui.step_header(3, 7, "Resources")
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert "3/7" in texts
        assert "Resources" in texts

    def test_step_header_bar_fill(self, mock_console):
        ui.step_header(5, 10, "Half")
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert "━" in texts


class TestStatusMessages:
    def test_success(self, mock_console):
        ui.success("done")
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert ICONS["success"] in texts
        assert "done" in texts

    def test_warning(self, mock_console):
        ui.warning("careful")
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert ICONS["warning"] in texts

    def test_error(self, mock_console):
        ui.error("fail")
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert ICONS["error"] in texts

    def test_info(self, mock_console):
        ui.info("note")
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert ICONS["info"] in texts

    def test_muted(self, mock_console):
        ui.muted("quiet")
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert "quiet" in texts


class TestResourceBar:
    def test_resource_bar_normal(self, mock_console):
        ui.resource_bar("CPU", 50)
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert "CPU" in texts
        assert "50%" in texts

    def test_resource_bar_high(self, mock_console):
        ui.resource_bar("MEM", 92)
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert "92%" in texts

    def test_resource_bar_warning_range(self, mock_console):
        ui.resource_bar("GPU", 80)
        texts = " ".join(str(a) for call in mock_console for a in call[0])
        assert "80%" in texts


class TestSummaryPanel:
    def test_summary_panel_renders(self, mock_console):
        ui.summary_panel("Test", {"key1": "val1", "key2": 42})
        assert len(mock_console) > 0


class TestPromptText:
    def test_prompt_text_calls_click(self, monkeypatch):
        monkeypatch.setattr("click.prompt", lambda *a, **kw: "hello")
        result = ui.prompt_text("Name", default="world")
        assert result == "hello"


class TestPromptChoice:
    def test_prompt_choice_calls_click(self, monkeypatch, mock_console):
        choices = [
            {"label": "A", "value": "a", "hint": "option a"},
            {"label": "B", "value": "b"},
        ]
        monkeypatch.setattr("click.prompt", lambda *a, **kw: "a")
        result = ui.prompt_choice("Pick one", choices, default=0)
        assert result == "a"


class TestPromptConfirm:
    def test_prompt_confirm_yes(self, monkeypatch):
        monkeypatch.setattr("click.confirm", lambda *a, **kw: True)
        assert ui.prompt_confirm("Sure?") is True

    def test_prompt_confirm_no(self, monkeypatch):
        monkeypatch.setattr("click.confirm", lambda *a, **kw: False)
        assert ui.prompt_confirm("Sure?") is False


class TestPromptNumber:
    def test_prompt_number_returns_int(self, monkeypatch):
        monkeypatch.setattr("click.prompt", lambda *a, **kw: 42)
        result = ui.prompt_number("CPU %", default=50, min_val=10, max_val=90)
        assert result == 42

"""Tests for prsm.cli_modules.theme — PRSMTheme dataclass and ICONS dict."""
from prsm.cli_modules.theme import PRSMTheme, THEME, ICONS


class TestPRSMTheme:
    """Verify the frozen dataclass has all expected fields."""

    def test_theme_is_frozen(self):
        import pytest
        with pytest.raises(AttributeError):
            THEME.primary = "blue"

    def test_primary_palette_fields(self):
        assert THEME.primary == "bold white"
        assert THEME.secondary == "white"
        assert THEME.accent == "bold"

    def test_text_hierarchy_fields(self):
        for attr in ("heading", "subheading", "body", "muted", "dim"):
            assert hasattr(THEME, attr)
            assert isinstance(getattr(THEME, attr), str)

    def test_semantic_fields(self):
        assert THEME.success == "green"
        assert THEME.warning == "yellow"
        assert THEME.error == "red"
        assert THEME.info == "dim"

    def test_interactive_fields(self):
        assert THEME.prompt == "bold"
        assert THEME.selected == "bold white"
        assert THEME.unselected == "dim"

    def test_border_fields(self):
        assert THEME.border == "dim white"
        assert THEME.rule == "dim"

    def test_default_construction(self):
        t = PRSMTheme()
        assert t == THEME

    def test_all_fields_are_strings(self):
        import dataclasses
        for f in dataclasses.fields(THEME):
            assert isinstance(getattr(THEME, f.name), str), f"{f.name} is not str"


class TestIcons:
    """Verify ICONS dict has all expected keys."""

    EXPECTED_KEYS = [
        "success", "warning", "error", "info", "bullet", "arrow",
        "check", "uncheck", "step", "network", "compute", "storage",
        "prism", "bar_full", "bar_empty", "line", "corner",
    ]

    def test_all_expected_keys_present(self):
        for key in self.EXPECTED_KEYS:
            assert key in ICONS, f"Missing icon key: {key}"

    def test_all_values_are_nonempty_strings(self):
        for key, val in ICONS.items():
            assert isinstance(val, str) and len(val) > 0, f"Bad icon: {key}={val!r}"

    def test_no_extra_keys(self):
        assert set(ICONS.keys()) == set(self.EXPECTED_KEYS)

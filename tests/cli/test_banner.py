"""Tests for prsm.cli_modules.banner — ASCII art, taglines, constants."""
from prsm.cli_modules.banner import (
    PRSM_BANNER, PRSM_ICON, PRSM_BANNER_COMPACT, PRSM_INLINE,
    TAGLINES, RULE,
)


class TestBannerStrings:
    def test_banner_is_multiline(self):
        assert "\n" in PRSM_BANNER
        assert len(PRSM_BANNER.strip().splitlines()) > 5

    def test_banner_contains_prsm_wordmark(self):
        # The block-letter wordmark uses half-block chars (▀▄█)
        assert "▄" in PRSM_BANNER  # half-block chars in wordmark

    def test_icon_is_multiline(self):
        assert "\n" in PRSM_ICON
        assert len(PRSM_ICON.strip().splitlines()) > 3

    def test_compact_banner_exists(self):
        # Compact should be shorter than full
        assert len(PRSM_BANNER_COMPACT) < len(PRSM_BANNER)

    def test_inline_banner(self):
        assert "PRSM" in PRSM_INLINE
        assert PRSM_INLINE == "◇ PRSM"


class TestTaglines:
    def test_taglines_is_list(self):
        assert isinstance(TAGLINES, list)

    def test_at_least_3_taglines(self):
        assert len(TAGLINES) >= 3

    def test_all_taglines_are_strings(self):
        for t in TAGLINES:
            assert isinstance(t, str) and len(t) > 0

    def test_taglines_end_with_period(self):
        for t in TAGLINES:
            assert t.endswith("."), f"Tagline missing period: {t!r}"


class TestRule:
    def test_rule_is_dashes(self):
        assert all(c == "─" for c in RULE)

    def test_rule_length(self):
        assert len(RULE) == 48

"""Unit tests for prsm.interface.display.

Covers Task 5 matrix from docs/2026-04-22-phase4-wallet-sdk-design-plan.md §6:

  - ftns_to_usd / usd_to_ftns conversion at various magnitudes
  - format_usd: negatives, zero, thousand separators
  - format_ftns: trailing-zero handling, precision
  - format_balance: usd mode shows both, ftns mode shows only FTNS
  - DisplayPreferenceStore default + set/get + persistence across reload
  - invalid mode rejected
"""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from prsm.interface.display import (
    DisplayMode,
    InMemoryDisplayPreferenceStore,
    SqliteDisplayPreferenceStore,
    StaticPriceSource,
    format_balance,
    format_ftns,
    format_usd,
    ftns_to_usd,
    usd_to_ftns,
)


# A deterministic price — $2.00 per FTNS — makes arithmetic trivial to verify.
PRICE = StaticPriceSource(Decimal("2.00"))


# --- conversion --------------------------------------------------------------


def test_ftns_to_usd_basic():
    assert ftns_to_usd(Decimal("1"), PRICE) == Decimal("2.00")
    assert ftns_to_usd(Decimal("10.5"), PRICE) == Decimal("21.00")


def test_ftns_to_usd_fractional():
    # 0.125 FTNS × $2.00 = $0.25
    assert ftns_to_usd(Decimal("0.125"), PRICE) == Decimal("0.25")


def test_usd_to_ftns_roundtrip():
    # $10 / $2 per FTNS = 5 FTNS
    assert usd_to_ftns(Decimal("10.00"), PRICE) == Decimal("5.000000")


def test_conversion_rounds_usd_to_cents():
    # 0.333... FTNS × $2.00 = $0.666... → rounds to $0.67 (half-to-even)
    src = StaticPriceSource(Decimal("2.00"))
    result = ftns_to_usd(Decimal("0.3333333"), src)
    assert result == Decimal("0.67")


# --- format_usd --------------------------------------------------------------


def test_format_usd_positive():
    assert format_usd(Decimal("2.50")) == "$2.50"


def test_format_usd_zero():
    assert format_usd(Decimal("0")) == "$0.00"


def test_format_usd_negative():
    assert format_usd(Decimal("-1.25")) == "-$1.25"


def test_format_usd_thousand_separator():
    assert format_usd(Decimal("1234567.89")) == "$1,234,567.89"


# --- format_ftns -------------------------------------------------------------


def test_format_ftns_has_unit_suffix():
    assert format_ftns(Decimal("1.5")) == "1.5000 FTNS"


def test_format_ftns_zero():
    assert format_ftns(Decimal("0")) == "0.0000 FTNS"


def test_format_ftns_rounds_to_precision():
    # 0.123456789 with precision=4 → 0.1235 (half-to-even)
    assert format_ftns(Decimal("0.123456789")) == "0.1235 FTNS"


# --- format_balance ----------------------------------------------------------


def test_format_balance_usd_mode_shows_both():
    # 0.125 FTNS at $2/FTNS → "$0.25 · 0.1250 FTNS"
    out = format_balance(Decimal("0.125"), PRICE, mode="usd")
    assert out == "$0.25 · 0.1250 FTNS"


def test_format_balance_ftns_mode_shows_only_ftns():
    out = format_balance(Decimal("0.125"), PRICE, mode="ftns")
    assert out == "0.1250 FTNS"


# --- DisplayPreferenceStore (in-memory) --------------------------------------


def test_preference_store_default_is_usd():
    store = InMemoryDisplayPreferenceStore()
    assert store.get_mode("alice") == "usd"


def test_preference_store_set_and_get():
    store = InMemoryDisplayPreferenceStore()
    store.set_mode("alice", "ftns")
    assert store.get_mode("alice") == "ftns"
    # Other users unaffected.
    assert store.get_mode("bob") == "usd"


def test_preference_store_rejects_invalid_mode():
    store = InMemoryDisplayPreferenceStore()
    with pytest.raises(ValueError):
        store.set_mode("alice", "eth")  # type: ignore[arg-type]


# --- DisplayPreferenceStore (sqlite) -----------------------------------------


def test_sqlite_preference_store_persists_across_reload():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "prefs.sqlite"

        store = SqliteDisplayPreferenceStore(db_path)
        store.set_mode("alice", "ftns")
        store.set_mode("bob", "usd")

        # New instance against same file.
        store2 = SqliteDisplayPreferenceStore(db_path)
        assert store2.get_mode("alice") == "ftns"
        assert store2.get_mode("bob") == "usd"
        assert store2.get_mode("carol") == "usd"  # default for unknown


def test_sqlite_preference_store_update_is_idempotent_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "prefs.sqlite"
        store = SqliteDisplayPreferenceStore(db_path)
        store.set_mode("alice", "ftns")
        store.set_mode("alice", "usd")  # overwrite
        assert store.get_mode("alice") == "usd"

"""FTNS↔USD conversion and display formatting for the PRSM UI.

Per docs/2026-04-22-phase4-wallet-sdk-design-plan.md §3.3 + §6 Task 5.

Per plan §3.3, all PRSM UI surfaces default to USD-denominated displays with
FTNS shown alongside ("$2.50 · 0.1250 FTNS"). Users can opt into FTNS-only
mode via a per-user preference.

The conversion layer is oracle-agnostic: callers pass in any object
implementing the `UsdPriceSource` protocol. The production path wires the
existing `FTNSOracle` (see `prsm.economy.blockchain.ftns_oracle`) — Phase 4
does not introduce a new oracle per plan §5.3.

Scope boundary: this module handles the *display* layer — string output + a
user-preference store. It does NOT touch settlement, royalties, or on-chain
accounting. Those stay in `prsm.economy.*` where FTNS is the unit of record.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Dict, Literal, Protocol


__all__ = [
    "DisplayMode",
    "DisplayPreferenceStore",
    "InMemoryDisplayPreferenceStore",
    "SqliteDisplayPreferenceStore",
    "StaticPriceSource",
    "UsdPriceSource",
    "format_balance",
    "format_ftns",
    "format_usd",
    "ftns_to_usd",
    "usd_to_ftns",
]


DisplayMode = Literal["usd", "ftns"]

_VALID_MODES: tuple[DisplayMode, ...] = ("usd", "ftns")

_CENT = Decimal("0.01")
_FTNS_PRECISION_DEFAULT = 4  # 4 decimal places — enough to represent small
                             # microtransactions without cluttering the UI.


class UsdPriceSource(Protocol):
    """Returns the current FTNS→USD price.

    Implementations:
      - `StaticPriceSource` for tests + UI fallback when the oracle is stale.
      - Production wrapper around `FTNSOracle.get_oracle_price()` — lives in
        the oracle integration layer, not here, so this module stays a leaf.
    """

    def ftns_price_usd(self) -> Decimal: ...


@dataclass(frozen=True)
class StaticPriceSource:
    """Fixed FTNS price. Use for tests or as a UI fallback."""

    price_usd: Decimal

    def ftns_price_usd(self) -> Decimal:
        return self.price_usd


def ftns_to_usd(ftns_amount: Decimal, price_source: UsdPriceSource) -> Decimal:
    """Convert FTNS → USD, quantized to cents with banker's rounding."""
    usd = ftns_amount * price_source.ftns_price_usd()
    return usd.quantize(_CENT, rounding=ROUND_HALF_EVEN)


def usd_to_ftns(
    usd_amount: Decimal,
    price_source: UsdPriceSource,
    *,
    decimals: int = 6,
) -> Decimal:
    """Convert USD → FTNS at the given precision (default 6 — subcent
    granularity for small micropayments)."""
    quantum = Decimal(10) ** -decimals
    return (usd_amount / price_source.ftns_price_usd()).quantize(
        quantum, rounding=ROUND_HALF_EVEN
    )


def format_usd(amount: Decimal) -> str:
    """Format a Decimal as `$X.XX`, with negatives surfaced as `-$X.XX` and
    thousand separators for readability."""
    quantized = amount.quantize(_CENT, rounding=ROUND_HALF_EVEN)
    if quantized < 0:
        return f"-${-quantized:,.2f}"
    return f"${quantized:,.2f}"


def format_ftns(amount: Decimal, *, decimals: int = _FTNS_PRECISION_DEFAULT) -> str:
    """Format a Decimal as `X.XXXX FTNS` at the given precision."""
    quantum = Decimal(10) ** -decimals
    quantized = amount.quantize(quantum, rounding=ROUND_HALF_EVEN)
    return f"{quantized:,.{decimals}f} FTNS"


def format_balance(
    ftns_amount: Decimal,
    price_source: UsdPriceSource,
    *,
    mode: DisplayMode = "usd",
) -> str:
    """Render a balance (or price) in the user's preferred mode.

    - `mode="usd"` → `"$2.50 · 0.1250 FTNS"` (default per plan §3.3).
    - `mode="ftns"` → `"0.1250 FTNS"`.
    """
    if mode == "ftns":
        return format_ftns(ftns_amount)
    usd = ftns_to_usd(ftns_amount, price_source)
    return f"{format_usd(usd)} · {format_ftns(ftns_amount)}"


# --- Preference persistence --------------------------------------------------


class DisplayPreferenceStore(Protocol):
    def get_mode(self, user_id: str) -> DisplayMode: ...
    def set_mode(self, user_id: str, mode: DisplayMode) -> None: ...


def _validate_mode(mode: str) -> DisplayMode:
    if mode not in _VALID_MODES:
        raise ValueError(
            f"invalid display mode {mode!r}; expected one of {_VALID_MODES}"
        )
    return mode  # type: ignore[return-value]


class InMemoryDisplayPreferenceStore:
    def __init__(self, default: DisplayMode = "usd") -> None:
        self._default: DisplayMode = _validate_mode(default)
        self._prefs: Dict[str, DisplayMode] = {}

    def get_mode(self, user_id: str) -> DisplayMode:
        return self._prefs.get(user_id, self._default)

    def set_mode(self, user_id: str, mode: DisplayMode) -> None:
        self._prefs[user_id] = _validate_mode(mode)


_PREF_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS display_preferences (
    user_id TEXT PRIMARY KEY,
    mode    TEXT NOT NULL CHECK (mode IN ('usd', 'ftns'))
);
"""


class SqliteDisplayPreferenceStore:
    """SQLite-backed per-user display preference.

    Keyed by a `user_id` string — the caller decides what that means (node_id
    hex, wallet address, session token hash, etc.). The table schema is
    Postgres-portable.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        default: DisplayMode = "usd",
    ) -> None:
        self._path = Path(db_path)
        self._default: DisplayMode = _validate_mode(default)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_PREF_SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def get_mode(self, user_id: str) -> DisplayMode:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT mode FROM display_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return self._default
        return row[0]  # type: ignore[no-any-return]

    def set_mode(self, user_id: str, mode: DisplayMode) -> None:
        mode = _validate_mode(mode)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO display_preferences (user_id, mode) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET mode = excluded.mode",
                (user_id, mode),
            )
            conn.commit()

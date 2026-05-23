"""Sprint 763 — power-source gate (AC vs battery).

Laptop operators want their PRSM daemon to refuse work while
on battery (so it doesn't drain the laptop while they're away
from a charger). Sprint 763 adds `PRSM_ACTIVE_ONLY_ON_AC=1` env.

Composition with sprint 755-758 active-window:
- Both gates AND together — daemon active iff window-ok AND
  power-ok.
- Either gate alone can refuse work.

Backward-compat: env unset → power gate disabled → always
returns True → no behavior change for existing operator fleet.

Fail-safe semantics: when psutil sensor is unavailable OR no
battery (desktop / server), treat as "on AC" → active. The
daemon should NOT mysteriously refuse work just because it
can't introspect power.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


def setup_function():
    from prsm.node.schedule import reset_cache_for_testing
    reset_cache_for_testing()
    os.environ.pop("PRSM_ACTIVE_ONLY_ON_AC", None)
    os.environ.pop("PRSM_ACTIVE_HOURS", None)


def teardown_function():
    from prsm.node.schedule import reset_cache_for_testing
    os.environ.pop("PRSM_ACTIVE_ONLY_ON_AC", None)
    os.environ.pop("PRSM_ACTIVE_HOURS", None)
    reset_cache_for_testing()


# ---- Default-disabled behavior ------------------------------------


def test_unset_env_passes_regardless_of_battery_state():
    """Backward-compat: env unset → power gate always returns
    True. Operators on laptop battery WITHOUT setting this env
    see no behavior change (daemon stays active)."""
    from prsm.node.schedule import is_currently_on_acceptable_power
    # Even if battery is dead, gate is disabled → True.
    with patch("psutil.sensors_battery") as mock_batt:
        mock_batt.return_value = type(
            "Battery", (), {"power_plugged": False},
        )()
        assert is_currently_on_acceptable_power() is True


def test_env_false_value_treated_as_disabled():
    """Explicit `PRSM_ACTIVE_ONLY_ON_AC=0` → disabled."""
    from prsm.node.schedule import is_currently_on_acceptable_power
    os.environ["PRSM_ACTIVE_ONLY_ON_AC"] = "0"
    assert is_currently_on_acceptable_power() is True


# ---- Enabled-feature behavior -------------------------------------


def test_enabled_and_on_battery_returns_false():
    """The actual feature: PRSM_ACTIVE_ONLY_ON_AC=1 + battery →
    gate returns False (daemon refuses work)."""
    from prsm.node.schedule import is_currently_on_acceptable_power
    os.environ["PRSM_ACTIVE_ONLY_ON_AC"] = "1"
    with patch("psutil.sensors_battery") as mock_batt:
        mock_batt.return_value = type(
            "Battery", (), {"power_plugged": False},
        )()
        assert is_currently_on_acceptable_power() is False


def test_enabled_and_on_ac_returns_true():
    """Feature on + plugged in → gate passes."""
    from prsm.node.schedule import is_currently_on_acceptable_power
    os.environ["PRSM_ACTIVE_ONLY_ON_AC"] = "1"
    with patch("psutil.sensors_battery") as mock_batt:
        mock_batt.return_value = type(
            "Battery", (), {"power_plugged": True},
        )()
        assert is_currently_on_acceptable_power() is True


def test_enabled_with_no_battery_treats_as_ac():
    """Desktop / server (psutil returns None) → treat as AC →
    gate passes. Otherwise enabling this env on a server would
    accidentally shut it down."""
    from prsm.node.schedule import is_currently_on_acceptable_power
    os.environ["PRSM_ACTIVE_ONLY_ON_AC"] = "1"
    with patch("psutil.sensors_battery", return_value=None):
        assert is_currently_on_acceptable_power() is True


def test_enabled_with_psutil_exception_failsafes_to_active():
    """Sensor introspection fails (perms / OS quirks) → fail-safe
    to True. The daemon should NOT mysteriously refuse work just
    because it can't read its own power state."""
    from prsm.node.schedule import is_currently_on_acceptable_power
    os.environ["PRSM_ACTIVE_ONLY_ON_AC"] = "1"
    with patch(
        "psutil.sensors_battery",
        side_effect=RuntimeError("sensor unavailable"),
    ):
        assert is_currently_on_acceptable_power() is True


# ---- Composition with active-window gate --------------------------


def test_composition_window_active_battery_off_means_inactive():
    """Active window says YES but on battery + feature on says NO
    → final is_currently_active() returns False."""
    from prsm.node.schedule import (
        is_currently_active, reset_cache_for_testing,
    )
    os.environ["PRSM_ACTIVE_HOURS"] = "00:00-23:59"  # ~always-active
    os.environ["PRSM_ACTIVE_ONLY_ON_AC"] = "1"
    reset_cache_for_testing()
    with patch("psutil.sensors_battery") as mock_batt:
        mock_batt.return_value = type(
            "Battery", (), {"power_plugged": False},
        )()
        assert is_currently_active() is False


def test_composition_window_inactive_overrides_power():
    """Active window says NO → gate returns False regardless of
    power. The window check fires first and short-circuits — we
    don't unnecessarily call psutil."""
    from prsm.node.schedule import (
        is_currently_active, reset_cache_for_testing,
    )
    # Schedule that's never active (cross-midnight ~zero-length
    # wouldn't parse — use a window that excludes most times).
    # Use mock instead: patch is_active to return False.
    os.environ["PRSM_ACTIVE_HOURS"] = "12:00-13:00"
    reset_cache_for_testing()
    with patch(
        "prsm.node.schedule.ActiveWindow.is_active",
        return_value=False,
    ):
        # On AC OR off AC — doesn't matter, window blocks.
        assert is_currently_active() is False


def test_composition_both_pass_means_active():
    """Window active AND on AC → daemon serves."""
    from prsm.node.schedule import (
        is_currently_active, reset_cache_for_testing,
    )
    os.environ["PRSM_ACTIVE_HOURS"] = "00:00-23:59"
    os.environ["PRSM_ACTIVE_ONLY_ON_AC"] = "1"
    reset_cache_for_testing()
    with patch("psutil.sensors_battery") as mock_batt:
        mock_batt.return_value = type(
            "Battery", (), {"power_plugged": True},
        )()
        assert is_currently_active() is True


def test_composition_no_gates_set_means_active():
    """Both envs unset → backward-compat always-active."""
    from prsm.node.schedule import is_currently_active
    # Battery state doesn't matter when feature off.
    with patch("psutil.sensors_battery") as mock_batt:
        mock_batt.return_value = type(
            "Battery", (), {"power_plugged": False},
        )()
        assert is_currently_active() is True

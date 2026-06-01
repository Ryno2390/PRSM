"""Sprint 905 (follow-up c) — staking UI view is utility-only, no APY.

After sp904 eliminated inflationary staking yield, the tokenomics
dashboard endpoint was still surfacing `apy = reward_rate_annual * 100`,
which now renders a misleading "APY: 0%" on the live dashboard
(`prsm/ui_assets/js/script.js`). The honest framing is that staking
confers UTILITY benefits (lock-based service discounts + priority
access), not a token yield.

These tests pin the `_staking_view` helper that builds the staking
block: it must declare `yield_model == "utility_only"`, carry a
human-readable benefits string, and keep `apy == 0.0` (back-compat for
the JS SDK type) — never a non-zero APY.
"""
from __future__ import annotations

from prsm.economy.tokenomics.staking_manager import StakingConfig
from prsm.interface.api.ui_api import _staking_view


def test_staking_view_is_utility_only():
    view = _staking_view(
        config=StakingConfig(),
        staked_amount=100.0,
        rewards_earned=0.0,
        next_reward_date="2026-06-02T00:00:00+00:00",
    )
    assert view["yield_model"] == "utility_only"
    assert view["apy"] == 0.0
    assert "discount" in view["benefits"].lower()
    assert "priority" in view["benefits"].lower()


def test_staking_view_apy_tracks_config_but_is_zero_in_v1():
    # The deployed v1 config has reward_rate_annual == 0.0 (sp904);
    # apy is derived from it, so it must read 0.0.
    cfg = StakingConfig()
    assert cfg.reward_rate_annual == 0.0
    view = _staking_view(
        config=cfg,
        staked_amount=0.0,
        rewards_earned=0.0,
        next_reward_date="2026-06-02T00:00:00+00:00",
    )
    assert view["apy"] == 0.0


def test_staking_view_preserves_core_fields():
    view = _staking_view(
        config=StakingConfig(),
        staked_amount=42.5,
        rewards_earned=0.0,
        next_reward_date="2026-06-02T00:00:00+00:00",
    )
    assert view["staked_amount"] == 42.5
    assert view["next_reward_date"] == "2026-06-02T00:00:00+00:00"
    assert "lock_period_days" in view

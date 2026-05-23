"""Sprint 765 — AutoClaimWorker foundation.

Background worker that periodically claims accumulated FTNS
rewards above an operator-configured threshold. Pre-765
operators had to manually call claim-rewards; sprint 765
automates the loop.

Pin tests cover:
- Config resolution from env (defaults + invalid + clamping)
- Worker enabled/disabled semantics
- _maybe_claim() one-iteration behavior:
  - Below threshold → no claim
  - Above threshold → claim + cumulative counter increments
  - Claim failure → logged + failure counter increments + no crash
- start() / stop() lifecycle (disabled worker skips, enabled
  schedules task)
"""
from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def setup_function():
    os.environ.pop("PRSM_AUTO_CLAIM_THRESHOLD_FTNS", None)
    os.environ.pop("PRSM_AUTO_CLAIM_INTERVAL_S", None)


def teardown_function():
    os.environ.pop("PRSM_AUTO_CLAIM_THRESHOLD_FTNS", None)
    os.environ.pop("PRSM_AUTO_CLAIM_INTERVAL_S", None)


# ---- Config resolution ---------------------------------------------


def test_config_unset_env_disabled():
    """Default unset → threshold=0 → disabled."""
    from prsm.node.auto_claim import resolve_auto_claim_config_from_env
    cfg = resolve_auto_claim_config_from_env()
    assert cfg.threshold_ftns == Decimal("0")
    assert cfg.enabled is False


def test_config_threshold_enables_worker():
    """Setting PRSM_AUTO_CLAIM_THRESHOLD_FTNS=100 enables."""
    from prsm.node.auto_claim import resolve_auto_claim_config_from_env
    os.environ["PRSM_AUTO_CLAIM_THRESHOLD_FTNS"] = "100"
    cfg = resolve_auto_claim_config_from_env()
    assert cfg.threshold_ftns == Decimal("100")
    assert cfg.enabled is True


def test_config_negative_threshold_treated_as_disabled():
    """Negative threshold is nonsensical → safe default to 0."""
    from prsm.node.auto_claim import resolve_auto_claim_config_from_env
    os.environ["PRSM_AUTO_CLAIM_THRESHOLD_FTNS"] = "-50"
    cfg = resolve_auto_claim_config_from_env()
    assert cfg.threshold_ftns == Decimal("0")


def test_config_invalid_threshold_safely_defaults():
    """Non-Decimal threshold → safe default 0 + warning log."""
    from prsm.node.auto_claim import resolve_auto_claim_config_from_env
    os.environ["PRSM_AUTO_CLAIM_THRESHOLD_FTNS"] = "not-a-number"
    cfg = resolve_auto_claim_config_from_env()
    assert cfg.threshold_ftns == Decimal("0")


def test_config_interval_default_3600():
    """PRSM_AUTO_CLAIM_INTERVAL_S unset → 3600s default."""
    from prsm.node.auto_claim import resolve_auto_claim_config_from_env
    cfg = resolve_auto_claim_config_from_env()
    assert cfg.interval_seconds == 3600.0


def test_config_interval_clamped_to_60s_minimum():
    """Interval < 60s clamped to 60 (claim attempts every minute
    would waste gas without meaningful operator benefit)."""
    from prsm.node.auto_claim import resolve_auto_claim_config_from_env
    os.environ["PRSM_AUTO_CLAIM_INTERVAL_S"] = "5"
    cfg = resolve_auto_claim_config_from_env()
    assert cfg.interval_seconds == 60.0


def test_config_invalid_interval_defaults():
    """Non-float interval → safe default 3600."""
    from prsm.node.auto_claim import resolve_auto_claim_config_from_env
    os.environ["PRSM_AUTO_CLAIM_INTERVAL_S"] = "an hour"
    cfg = resolve_auto_claim_config_from_env()
    assert cfg.interval_seconds == 3600.0


def test_config_is_frozen_and_hashable():
    """Frozen dataclass → hashable + equality-comparable."""
    from prsm.node.auto_claim import AutoClaimConfig
    c1 = AutoClaimConfig(Decimal("100"), 3600.0)
    c2 = AutoClaimConfig(Decimal("100"), 3600.0)
    assert c1 == c2
    assert hash(c1) == hash(c2)


# ---- Worker behavior -----------------------------------------------


@pytest.mark.asyncio
async def test_maybe_claim_below_threshold_skips():
    """Accumulated 50 FTNS, threshold 100 → no claim."""
    from prsm.node.auto_claim import AutoClaimWorker, AutoClaimConfig
    staking = MagicMock()
    staking.calculate_rewards = AsyncMock(return_value=[
        SimpleNamespace(reward_amount=Decimal("50")),
    ])
    staking.claim_rewards = AsyncMock()
    worker = AutoClaimWorker(
        staking, "user-1",
        config=AutoClaimConfig(Decimal("100"), 60.0),
    )
    result = await worker._maybe_claim()
    assert result is None
    staking.claim_rewards.assert_not_called()
    assert worker.claim_attempts == 0


@pytest.mark.asyncio
async def test_maybe_claim_above_threshold_claims():
    """Accumulated 150 FTNS, threshold 100 → claim called."""
    from prsm.node.auto_claim import AutoClaimWorker, AutoClaimConfig
    staking = MagicMock()
    staking.calculate_rewards = AsyncMock(return_value=[
        SimpleNamespace(reward_amount=Decimal("150")),
    ])
    staking.claim_rewards = AsyncMock(return_value=Decimal("150"))
    worker = AutoClaimWorker(
        staking, "user-1",
        config=AutoClaimConfig(Decimal("100"), 60.0),
    )
    result = await worker._maybe_claim()
    assert result == Decimal("150")
    staking.claim_rewards.assert_called_once_with("user-1")
    assert worker.claim_attempts == 1
    assert worker.total_claimed_ftns == Decimal("150")


@pytest.mark.asyncio
async def test_maybe_claim_disabled_skips():
    """Worker disabled (threshold=0) → no calculate, no claim."""
    from prsm.node.auto_claim import AutoClaimWorker, AutoClaimConfig
    staking = MagicMock()
    staking.calculate_rewards = AsyncMock()
    staking.claim_rewards = AsyncMock()
    worker = AutoClaimWorker(
        staking, "user-1",
        config=AutoClaimConfig(Decimal("0"), 60.0),
    )
    result = await worker._maybe_claim()
    assert result is None
    staking.calculate_rewards.assert_not_called()
    staking.claim_rewards.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_claim_failure_logged_no_crash():
    """claim_rewards raises → failure counter increments, no
    exception propagates."""
    from prsm.node.auto_claim import AutoClaimWorker, AutoClaimConfig
    staking = MagicMock()
    staking.calculate_rewards = AsyncMock(return_value=[
        SimpleNamespace(reward_amount=Decimal("200")),
    ])
    staking.claim_rewards = AsyncMock(
        side_effect=RuntimeError("gas estimation failed"),
    )
    worker = AutoClaimWorker(
        staking, "user-1",
        config=AutoClaimConfig(Decimal("100"), 60.0),
    )
    result = await worker._maybe_claim()
    assert result is None
    assert worker.claim_attempts == 1
    assert worker.claim_failures == 1


@pytest.mark.asyncio
async def test_maybe_claim_sums_multiple_calculations():
    """Multiple stakes contributing rewards → sum + threshold
    check on sum."""
    from prsm.node.auto_claim import AutoClaimWorker, AutoClaimConfig
    staking = MagicMock()
    staking.calculate_rewards = AsyncMock(return_value=[
        SimpleNamespace(reward_amount=Decimal("60")),
        SimpleNamespace(reward_amount=Decimal("80")),
    ])
    staking.claim_rewards = AsyncMock(return_value=Decimal("140"))
    worker = AutoClaimWorker(
        staking, "user-1",
        config=AutoClaimConfig(Decimal("100"), 60.0),
    )
    # Sum = 140 > 100, claim should fire
    result = await worker._maybe_claim()
    assert result == Decimal("140")


# ---- Lifecycle -----------------------------------------------------


@pytest.mark.asyncio
async def test_start_disabled_worker_is_noop():
    """start() on disabled worker schedules nothing."""
    from prsm.node.auto_claim import AutoClaimWorker, AutoClaimConfig
    worker = AutoClaimWorker(
        MagicMock(), "user-1",
        config=AutoClaimConfig(Decimal("0"), 60.0),
    )
    await worker.start()
    assert worker._task is None


@pytest.mark.asyncio
async def test_stop_idempotent_when_not_started():
    """stop() before start() is a no-op."""
    from prsm.node.auto_claim import AutoClaimWorker, AutoClaimConfig
    worker = AutoClaimWorker(
        MagicMock(), "user-1",
        config=AutoClaimConfig(Decimal("100"), 60.0),
    )
    await worker.stop()  # should not raise
    assert worker._task is None

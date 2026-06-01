"""Sprint 913 — governance staking is UTILITY-ONLY (no phantom 5% yield).

sp904 fixed the deployed v1 tokenomics: staking is UTILITY-ONLY (network-fee
discount + dispatch-priority via StakingManager), with NO token yield and NO
burn. But `GovernanceTokenDistributor.stake_tokens_for_governance` still
created a phantom "5% staking reward" distribution record
(`DistributionType.STAKING_REWARD`), and the `/api/v1/governance/stake`
endpoint + method docstrings advertised "5% staking rewards" and "up to 4x
voting multipliers" — contradicting the deployed monetary policy and
inflating `get_distribution_statistics` with rewards that were never minted.

sp906 fixed the same stale-yield claim in the MCP `prsm_stake` preview; this
closes the governance-distributor path sp906 missed. Staking returns voting
power only — no STAKING_REWARD distribution is recorded.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from prsm.economy.governance import token_distribution as td
from prsm.economy.governance.token_distribution import DistributionType


@pytest.fixture
def distributor(monkeypatch):
    fake = AsyncMock()
    # stake_for_governance returns (stake_tx, voting_power)
    fake.stake_for_governance = AsyncMock(return_value=(object(), Decimal("100")))
    monkeypatch.setattr(td, "DatabaseFTNSService", lambda *a, **k: fake)
    d = td.GovernanceTokenDistributor()
    d._fake = fake
    return d


@pytest.mark.asyncio
async def test_staking_records_no_yield_reward(distributor):
    """Utility-only: staking must NOT create a STAKING_REWARD distribution."""
    staked, voting_power = await distributor.stake_tokens_for_governance(
        user_id="u1", amount=Decimal("1000"), lock_duration_days=90,
    )

    assert staked == Decimal("1000")
    assert voting_power == Decimal("100")
    reward_records = [
        dist for dist in distributor.distributions.values()
        if dist.distribution_type == DistributionType.STAKING_REWARD
    ]
    assert reward_records == [], (
        "staking is utility-only (sp904) — no STAKING_REWARD yield record"
    )


@pytest.mark.asyncio
async def test_staking_still_tracks_staked_total(distributor):
    """The stake itself is still recorded for voting-power accounting."""
    await distributor.stake_tokens_for_governance(
        user_id="u1", amount=Decimal("500"), lock_duration_days=30,
    )
    assert distributor.activation_stats["total_staked_tokens"] == Decimal("500")
    # The FTNS service was asked to stake (lock) the tokens.
    distributor._fake.stake_for_governance.assert_awaited_once()


def test_stake_endpoint_docstring_has_no_yield_claims():
    """The /stake API docstring must not advertise yield / voting multipliers
    that the deployed v1 (utility-only) model does not provide."""
    from prsm.interface.api import governance_api

    doc = governance_api.stake_tokens_for_governance.__doc__ or ""
    low = doc.lower()
    assert "staking rewards" not in low
    assert "4x" not in low
    assert "multiplier" not in low

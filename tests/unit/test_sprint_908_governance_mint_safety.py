"""Sprint 908 — governance system-mint safety (money-path robustness).

The money-path review confirmed two governance defects:

1. SELF-SERVICE TIER ELEVATION (critical authz hole): POST /governance/
   activate took participant_tier from the REQUEST BODY and
   _validate_governance_eligibility was a literal `pass` that returned True
   for any tier — so any authenticated user could self-select CORE_TEAM and
   mint 100,000 FTNS (COMMUNITY is 1,000). Fixed: self-service activations
   are hard-capped at COMMUNITY; tier elevation requires an authorized
   system flow (self_service=False).

2. SHARED EVAL-ONCE UUID DEFAULT (prerequisite): TokenDistribution.
   distribution_id / GovernanceActivation.activation_id used `= uuid4()`
   (evaluated once at class definition) so every instance shared one id —
   breaking the audit trail, collapsing the self.distributions dict to a
   single entry, and making reference_id a constant. Fixed with
   Field(default_factory=uuid4).
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.economy.governance import token_distribution as td
from prsm.economy.governance.token_distribution import (
    TokenDistribution,
    GovernanceActivation,
    DistributionType,
    GovernanceParticipantTier,
)


# ── prerequisite: per-instance UUID defaults ─────────────


def test_distribution_id_is_unique_per_instance():
    a = TokenDistribution(
        recipient_user_id="u1",
        distribution_type=DistributionType.CONTRIBUTION_REWARD,
        amount=Decimal("1"),
    )
    b = TokenDistribution(
        recipient_user_id="u2",
        distribution_type=DistributionType.CONTRIBUTION_REWARD,
        amount=Decimal("1"),
    )
    assert a.distribution_id != b.distribution_id


def test_activation_id_is_unique_per_instance():
    a = GovernanceActivation(
        participant_user_id="u1",
        participant_tier=GovernanceParticipantTier.COMMUNITY,
        initial_token_allocation=Decimal("1000"),
    )
    b = GovernanceActivation(
        participant_user_id="u2",
        participant_tier=GovernanceParticipantTier.COMMUNITY,
        initial_token_allocation=Decimal("1000"),
    )
    assert a.activation_id != b.activation_id


# ── critical: self-service tier-elevation authz ──────────


@pytest.fixture
def distributor(monkeypatch):
    # Keep construction light + mock the user-exists lookup.
    monkeypatch.setattr(td, "DatabaseFTNSService", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        td.auth_manager, "get_user_by_id",
        AsyncMock(return_value={"id": "u1", "username": "u1"}),
        raising=False,
    )
    return td.GovernanceTokenDistributor()


@pytest.mark.asyncio
async def test_self_service_cannot_elevate_to_core_team(distributor):
    # The exploit: self-select CORE_TEAM (100k FTNS) must be REJECTED.
    ok = await distributor._validate_governance_eligibility(
        "u1", GovernanceParticipantTier.CORE_TEAM,
    )
    assert ok is False


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", [
    GovernanceParticipantTier.CONTRIBUTOR,
    GovernanceParticipantTier.EXPERT,
    GovernanceParticipantTier.DELEGATE,
    GovernanceParticipantTier.COUNCIL_MEMBER,
    GovernanceParticipantTier.CORE_TEAM,
])
async def test_self_service_rejects_all_elevated_tiers(distributor, tier):
    assert await distributor._validate_governance_eligibility("u1", tier) is False


@pytest.mark.asyncio
async def test_self_service_allows_community(distributor):
    ok = await distributor._validate_governance_eligibility(
        "u1", GovernanceParticipantTier.COMMUNITY,
    )
    assert ok is True


@pytest.mark.asyncio
async def test_authorized_flow_may_grant_elevated_tier(distributor):
    # An authorized system flow (early-adopter program / admin grant)
    # passes self_service=False and may grant an elevated tier.
    ok = await distributor._validate_governance_eligibility(
        "u1", GovernanceParticipantTier.CORE_TEAM, self_service=False,
    )
    assert ok is True


@pytest.mark.asyncio
async def test_unknown_user_rejected_even_at_community(monkeypatch):
    monkeypatch.setattr(td, "DatabaseFTNSService", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        td.auth_manager, "get_user_by_id", AsyncMock(return_value=None),
        raising=False,
    )
    dist = td.GovernanceTokenDistributor()
    ok = await dist._validate_governance_eligibility(
        "ghost", GovernanceParticipantTier.COMMUNITY,
    )
    assert ok is False

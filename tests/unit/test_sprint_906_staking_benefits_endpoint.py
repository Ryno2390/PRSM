"""Sprint 906 — GET /staking/benefits/{user_id} endpoint.

Exposes the live staking utility-benefit tier (service discount +
priority access) so the pricing and dispatch layers — and the dashboard
— consume a single source of truth. Delegates to
StakingManager.get_user_benefits (covered behaviorally in
test_staking_incentives.py); this pins the HTTP surface + shape.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from prsm.economy.tokenomics.staking_manager import StakingBenefits
from prsm.node.api import create_api_app


def _client(staking_manager=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.staking_manager = staking_manager
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_benefits_503_when_manager_unwired():
    resp = _client(staking_manager=None).get("/staking/benefits/alice")
    assert resp.status_code == 503


def test_benefits_active_tier_shape():
    sm = MagicMock()
    sm.get_user_benefits = AsyncMock(
        return_value=StakingBenefits(365, "365d", 0.10, 0.50)
    )
    resp = _client(staking_manager=sm).get("/staking/benefits/alice")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "alice"
    assert body["yield_model"] == "utility_only"
    assert body["tier_label"] == "365d"
    assert body["discount_fraction"] == 0.10
    assert body["priority_boost"] == 0.50
    assert body["is_active"] is True
    assert body["service_discount_pct"] == 10.0
    assert body["priority_boost_pct"] == 50.0


def test_benefits_none_tier_for_unstaked_user():
    sm = MagicMock()
    sm.get_user_benefits = AsyncMock(return_value=StakingBenefits.none())
    resp = _client(staking_manager=sm).get("/staking/benefits/nobody")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_active"] is False
    assert body["tier_label"] == "none"
    assert body["service_discount_pct"] == 0.0

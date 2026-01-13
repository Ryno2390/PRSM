
import pytest
import asyncio
from decimal import Decimal
from prsm.compute.nwtn.reasoning.autonomous_discovery import ChallengeManager, MoonshotFund, BreakthroughLevel
from prsm.sdks.nwtn_hardware_sdk import HardwareNWTNNode
from prsm.core.institutional.gateway import InstitutionalGateway, InstitutionalCapacity, ParticipationMode

@pytest.mark.asyncio
async def test_apm_genesis_challenge():
    """Verify that global challenges can be announced and met"""
    fund = MoonshotFund(treasury_balance=Decimal("1000000.0"))
    manager = ChallengeManager(fund)
    
    # 1. Announce Molecular Sorter Challenge
    challenge = manager.announce_challenge(
        title="Molecular Sorter",
        problem="Separate carbon atoms from air with 99% precision",
        domain="chemistry",
        reward=Decimal("50000.0")
    )
    
    assert challenge.active is True
    
    # 2. Simulate a node meeting the challenge
    node_id = "brilliant_researcher_01"
    mock_result = {"reward": 0.98} # Level 5 breakthrough
    
    success = await manager.evaluate_submission(challenge.challenge_id, node_id, mock_result)
    
    assert success is True
    assert challenge.active is False
    assert fund.treasury < Decimal("1000000.0")

@pytest.mark.asyncio
async def test_hardware_sdk_gating():
    """Verify that the hardware SDK correctly gates redundant data at the edge"""
    node = HardwareNWTNNode(hardware_id="sensor_01", secret_key="super_secret")
    
    # First reading - always surprising
    p1 = node.process_sensor_reading(100.0)
    assert p1 is not None
    assert p1["surprise"] == 1.0
    
    # Second reading - low surprise (1% change)
    p2 = node.process_sensor_reading(101.0)
    assert p2 is None # Gated!
    
    # Third reading - high surprise (100% change)
    p3 = node.process_sensor_reading(200.0)
    assert p3 is not None
    assert p3["surprise"] > 0.9 # High surprise
    assert "dt_hash" in p3

@pytest.mark.asyncio
async def test_institutional_production_federation():
    """Verify that institutions can upgrade to Production Tier"""
    gateway = InstitutionalGateway()
    capacity = InstitutionalCapacity(
        compute_tflops=10**6, storage_petabytes=500, 
        bandwidth_gbps=10000, model_parameters=10**13,
        research_personnel=1000, annual_ai_budget_usd=10**10
    )
    
    cern = await gateway.onboard_institution("CERN", capacity, [ParticipationMode.INFRASTRUCTURE])
    
    # Upgrade to Production
    token = await gateway.activate_production_federation(cern.participant_id)
    
    assert cern.is_production_ready is True
    assert token.startswith("FED-")
    assert len(cern.federation_token) > 10

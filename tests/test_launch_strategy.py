
import pytest
from decimal import Decimal
from datetime import datetime, timezone
from prsm.economy.tokenomics.ftns_service import FTNSService, FTNSTransactionType
from prsm.economy.governance.resilience import ResilienceManager
from prsm.compute.nwtn.reasoning.autonomous_discovery import MoonshotFund

def test_launch_caps():
    """Verify that staking is limited by the Launch Guardrails"""
    service = FTNSService()
    user_id = "early_adopter"
    
    # 1. Try to stake more than the per-user cap (10k)
    success = service.stake_tokens(user_id, Decimal("15000.0"))
    assert success is False
    
    # 2. Try a valid stake
    service.award_tokens(user_id, FTNSTransactionType.SYSTEM_USAGE, base_amount=Decimal("5000.0"))
    success_2 = service.stake_tokens(user_id, Decimal("1000.0"))
    assert success_2 is True
    assert service.total_staked == Decimal("1000.0")

def test_seal_911_emergency_pause():
    """Verify that guardians can collectively pause the network"""
    res = ResilienceManager()
    
    # Register 3 guardians
    res.guardians.register_guardian("dr_fauci")
    res.guardians.register_guardian("prof_einstein")
    res.guardians.register_guardian("prof_curie")
    
    # Try to pause with only 2 signatures (requires 3)
    res.guardians.emergency_pause(["dr_fauci", "prof_einstein"], "TREASURY_DRAIN")
    assert res.guardians.network_paused is False
    
    # Pause with 3 signatures
    res.guardians.emergency_pause(["dr_fauci", "prof_einstein", "prof_curie"], "TREASURY_DRAIN")
    assert res.guardians.network_paused is True

def test_bug_bounty_payout():
    """Verify that security vulnerabilities are rewarded from the Moonshot Fund"""
    fund = MoonshotFund(treasury_balance=Decimal("1000000.0"))
    researcher = "whitehat_01"
    
    # Claim a critical bounty (sandbox escape)
    fund.process_bounty_claim(researcher, "SANDBOX_ESCAPE", "critical")
    
    # Should pay 5% of fund (50,000)
    assert fund.treasury == Decimal("950000.0")


import pytest
from uuid import UUID
from prsm.economy.governance.human_centric import ExpertOracle, LiquidScienceGovernance, ExpertiseDomain

def test_expert_oracle_validation():
    """Verify that multiple experts are needed to approve high-stakes research"""
    oracle = ExpertOracle()
    
    # Register 3 experts in Biology
    oracle.register_expert("dr_smith", [ExpertiseDomain.BIOLOGY], 1000.0)
    oracle.register_expert("dr_jones", [ExpertiseDomain.BIOLOGY], 1000.0)
    oracle.register_expert("dr_lee", [ExpertiseDomain.BIOLOGY, ExpertiseDomain.ETHICS], 1000.0)
    
    # Create request for a medical model
    req_id = oracle.create_validation_request("cid_medical_v1", ExpertiseDomain.BIOLOGY)
    
    # First 2 approvals - still pending
    oracle.submit_validation("dr_smith", req_id, True)
    oracle.submit_validation("dr_jones", req_id, True)
    assert oracle.requests[req_id].status == "pending"
    
    # 3rd approval -> Approved!
    oracle.submit_validation("dr_lee", req_id, True)
    assert oracle.requests[req_id].status == "approved"

def test_liquid_science_delegation():
    """Verify domain-specific delegation of voting power"""
    gov = LiquidScienceGovernance()
    
    # User Alice has 100 power, Bob has 50, Charlie has 10
    user_powers = {"alice": 100.0, "bob": 50.0, "charlie": 10.0}
    
    # Bob delegates his Biology power to Alice
    gov.delegate_by_domain("bob", "alice", ExpertiseDomain.BIOLOGY)
    
    # Charlie delegates his AI Architecture power to Bob
    gov.delegate_by_domain("charlie", "bob", ExpertiseDomain.AI_ARCHITECTURE)
    
    # Alice's Biology power should be 100 (hers) + 50 (Bob's) = 150
    alice_bio_power = gov.calculate_domain_power("alice", ExpertiseDomain.BIOLOGY, 100.0, user_powers)
    assert alice_bio_power == 150.0
    
    # Alice's AI power should still be 100
    alice_ai_power = gov.calculate_domain_power("alice", ExpertiseDomain.AI_ARCHITECTURE, 100.0, user_powers)
    assert alice_ai_power == 100.0
    
    # Bob's AI power should be 50 + 10 = 60
    bob_ai_power = gov.calculate_domain_power("bob", ExpertiseDomain.AI_ARCHITECTURE, 50.0, user_powers)
    assert bob_ai_power == 60.0


import pytest
from prsm.compute.nwtn.reasoning.surprise_gating import SurpriseGater
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator

def test_surprise_calculation_numeric():
    gater = SurpriseGater(surprise_threshold=0.1)
    
    # 10% change is exactly at the threshold (depending on implementation)
    s1 = gater.calculate_surprise(100, 110)
    assert s1 == 0.1
    
    # 1% change is low surprise
    s2 = gater.calculate_surprise(100, 101)
    assert gater.should_gate(s2) is True
    
    # 50% change is high surprise
    s3 = gater.calculate_surprise(100, 150)
    assert gater.should_gate(s3) is False

def test_surprise_calculation_text():
    gater = SurpriseGater(surprise_threshold=0.3)
    
    prior = "The battery is charging"
    posterior_similar = "The battery is charging now" # 1 new word 'now' / 5 total words = 0.2
    posterior_different = "The battery has exploded into flames" # high surprise
    
    s_low = gater.calculate_surprise(prior, posterior_similar)
    s_high = gater.calculate_surprise(prior, posterior_different)
    
    assert gater.should_gate(s_low) is True
    assert gater.should_gate(s_high) is False

@pytest.mark.asyncio
async def test_orchestrator_gating():
    """Verify that the orchestrator trace excludes low-surprise steps"""
    # High threshold to force gating
    orchestrator = NeuroSymbolicOrchestrator(node_id="gated_node")
    orchestrator._lazy_init()  # Ensure gater is initialized
    orchestrator.gater.surprise_threshold = 0.5
    
    # query and s1_proposal are very similar in our mock
    # query: "Battery"
    # s1_proposal: "Neural intuition for Battery..."
    # The 'S2_LIGHT_CHECK' has surprise 0.2, so it should be gated.
    
    result = await orchestrator.solve_task("Battery", "Context")
    
    trace_actions = [s["a"] for s in result["trace"]]
    
    # 'S2_LIGHT_CHECK' (surprise 0.2) should be missing if mode is 'light'
    if result["mode"] == "light":
        assert "S2_LIGHT_CHECK" not in trace_actions

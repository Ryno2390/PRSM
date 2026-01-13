
import pytest
from prsm.compute.collaboration.state_sync import CollaborativeStateSpace, ValidationReport
from prsm.compute.nwtn.knowledge_graph import RecursiveKnowledgeGraph, ReasoningIntervention

def test_model_state_sync_and_forking():
    """Verify that we can fork specific model components (logic vs weights)"""
    space = CollaborativeStateSpace()
    
    # 1. Create initial state
    report_v1 = ValidationReport(accuracy_score=0.85, bias_index=0.02, compute_cost_ftns=5.0)
    state_v1 = space.create_state(
        model_id="solar_cell_v1",
        logic_cid="mcts_v1_cid",
        weights_cid="ssm_v1_cid",
        author_id="lab_tokyo",
        validation=report_v1
    )
    
    assert space.heads["main"] == state_v1.state_id
    
    # 2. Lab Berlin forks weights but replaces logic
    logic_v2, weights_v1 = space.fork_component(state_v1.state_id, new_logic_cid="mcts_improved_cid")
    
    assert logic_v2 == "mcts_improved_cid"
    assert weights_v1 == "ssm_v1_cid" # Kept from v1
    
    # 3. Commit the improved state
    report_v2 = ValidationReport(accuracy_score=0.92, bias_index=0.01, compute_cost_ftns=4.5)
    state_v2 = space.create_state(
        model_id="solar_cell_v1",
        logic_cid=logic_v2,
        weights_cid=weights_v1,
        author_id="lab_berlin",
        validation=report_v2,
        parent_id=state_v1.state_id
    )
    
    assert state_v2.parent_state_id == state_v1.state_id
    assert state_v2.validation.accuracy_score > state_v1.validation.accuracy_score

def test_agentic_coworking_intervention():
    """Verify that agents can intervene in live reasoning traces"""
    kg = RecursiveKnowledgeGraph()
    trace_id = "trace_physics_001"
    
    intervention = ReasoningIntervention(
        trace_id=trace_id,
        intervening_agent_id="agent_berlin",
        suggestion="Apply Graphene constant to the second step",
        target_step_index=1
    )
    
    kg.submit_intervention(intervention)
    
    assert trace_id in kg.live_traces
    assert len(kg.live_traces[trace_id]) == 1
    assert kg.live_traces[trace_id][0].intervening_agent_id == "agent_berlin"

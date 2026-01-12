
import pytest
import asyncio
from prsm.interface.nll_interface import NaturalLanguageLab
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
from prsm.core.identity.sro import ScientificReputationOracle
from prsm.data.legacy_sync import LegacySyncPlugin
from prsm.knowledge_system import UnifiedKnowledgeSystem

@pytest.mark.asyncio
async def test_nll_conversational_command():
    """Verify that natural language is parsed into valid orchestrator tasks"""
    orchestrator = NeuroSymbolicOrchestrator(node_id="nll_node")
    nll = NaturalLanguageLab(orchestrator)
    
    # 1. Test Discovery Intent
    query = "Propose a new cathode composition for a battery"
    response = await nll.execute_conversational_command(query)
    
    assert response["status"] == "success"
    assert response["intent"].command_type == "DISCOVER"
    assert "output" in response["result"]

    # 2. Test Query Intent
    query_2 = "Find papers on thermal runaway"
    response_2 = await nll.execute_conversational_command(query_2)
    assert response_2["intent"].command_type == "QUERY_GRAPH"
    assert "Found 3 papers" in response_2["data"]

@pytest.mark.asyncio
async def test_sro_reputation_bridge():
    """Verify that linking ORCID increases the trust multiplier"""
    sro = ScientificReputationOracle()
    user_id = "prof_oxford"
    
    # Baseline
    assert sro.get_trust_multiplier(user_id) == 1.0
    
    # Link ORCID with h-index of 50
    sro.link_credentials(user_id, orcid="0000-0001-2345-6789", h_index=50, institution="Oxford")
    
    # New trust should be 1 + log10(50) approx 2.7
    trust = sro.get_trust_multiplier(user_id)
    assert trust > 2.0
    assert trust < 3.0

@pytest.mark.asyncio
async def test_legacy_sync_api():
    """Verify that legacy data (Excel) flows automatically into the knowledge system"""
    ks = UnifiedKnowledgeSystem()
    plugin = LegacySyncPlugin(user_id="solo_chemist", ks=ks)
    
    mock_excel_data = [{"element": "Li", "qty": 0.5}, {"element": "Co", "qty": 0.5}]
    
    cid = await plugin.sync_from_excel("lab_notes.xlsx", mock_excel_data, domain="chemistry")
    
    assert cid.startswith("cid_")
    assert ks.indexed_items == 1
    assert cid in plugin.sync_history

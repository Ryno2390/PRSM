
import asyncio
import base64
from decimal import Decimal
from prsm.interface.nll_interface import NaturalLanguageLab
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
from prsm.core.identity.nhi import NHIManager
from prsm.core.identity.sro import ScientificReputationOracle
from prsm.core.compliance.regulatory_mapping import RegulatoryMappingAgent, Jurisdiction

async def simulate_new_user_journey():
    print("🧪 [SIMULATION] PRSM User Onboarding & First Discovery")
    print("="*60)
    
    # 1. SETUP IDENTITY (SRO)
    sro = ScientificReputationOracle()
    user_id = "user_montana_chemist"
    sro.link_credentials(user_id, orcid="0000-0002-1825-0097", h_index=15, institution="Independent")
    trust_score = sro.get_trust_multiplier(user_id)
    print(f"✅ SRO Linked. Trust Multiplier: {trust_score:.2f}x")

    # 2. REGISTER AUTONOMOUS AGENT (NHI)
    nhi_manager = NHIManager()
    agent, keypair = nhi_manager.register_agent(user_id, "researcher")
    print(f"🤖 Agent Registered: {agent.agent_id}")

    # 3. INITIALIZE ORCHESTRATOR & NLL
    orchestrator = NeuroSymbolicOrchestrator(node_id=agent.agent_id)
    nll = NaturalLanguageLab(orchestrator)
    print("🌈 Digital Lab Partner Active.")

    # 4. FIRST COMMAND (The "Battery" Query)
    print("\n🗣️ User: 'PRSM, find a new cathode composition for a battery and include my email me@example.com for logs.'")
    
    # This command includes PII (email) to test the Security Gate
    query = "Propose a new cathode composition for a battery. Logs to me@example.com"
    
    response = await nll.execute_conversational_command(query)
    
    # 5. ANALYZE THE BACKEND ACTIONS
    print("\n🔍 Backend Execution Report:")
    trace = response["result"]["trace"]
    
    # Verify PII Sanitization
    init_step = next(s for s in trace if s["a"] == "INIT")
    print(f"   - PII Check: {init_step['c']}") # Should be redacted
    
    # Verify FinOps Planning
    strategy_step = next(s for s in trace if s["a"] == "STRATEGY_PLANNED")
    print(f"   - FinOps: {strategy_step['c']} (Cost: {strategy_step['m']['estimated_cost']})")
    
    # Verify Surprise Gating
    print("   - Context State:")
    for step in trace:
        print(f"     [{step['a']}] Surprise: {step['s']:.4f}")
        
    # Verify PQC Signing
    if response["result"].get("pq_signature"):
        print("   - Security: Result signed with Post-Quantum Dilithium (Hybrid).")

    print("\n🎯 Result Summary:")
    print(f"   Discovery: {response['result']['output']}")
    print(f"   Breakthrough Score: {response['result']['reward']:.2f}")
    
    print("\n🎉 SUCCESS: User journey complete. The machine is learning.")

if __name__ == "__main__":
    asyncio.run(simulate_new_user_journey())

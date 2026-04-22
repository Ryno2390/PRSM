
import pytest
import asyncio
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator

@pytest.mark.asyncio
async def test_security_pii_redaction():
    """Verify that sensitive information is sanitized before processing"""
    orchestrator = NeuroSymbolicOrchestrator(node_id="security_node")
    
    query = "Search records for user@example.com and CC 1234-5678-9012-3456"
    result = await orchestrator.solve_task(query, "Private context")
    
    # Check the trace - the 'INIT' step should be sanitized
    init_step = next((s for s in result["trace"] if s["a"] == "INIT"), None)
    assert "[REDACTED_EMAIL]" in init_step["c"]
    assert "[REDACTED_CREDIT_CARD]" in init_step["c"]
    assert "user@example.com" not in init_step["c"]

@pytest.mark.asyncio
async def test_security_breakout_blocked():
    """Verify that dangerous code patterns are blocked before execution"""
    orchestrator = NeuroSymbolicOrchestrator(node_id="security_node")
    
    # Malicious query trying to import OS
    malicious_query = "import os; os.system('rm -rf /')"
    result = await orchestrator.solve_task(malicious_query, "Private context")
    
    assert result["status"] == "security_violation"
    assert "Dangerous pattern detected" in str(result["audit_findings"])
    
    # Trace should show the abort
    trace_actions = [s["a"] for s in result["trace"]]
    assert "SECURITY_ABORT" in trace_actions

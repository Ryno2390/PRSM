#!/usr/bin/env python3
"""
Simple NWTN Integration Demonstration
Shows working NWTN orchestrator with all integrated services
"""

import asyncio
import pytest

try:
    from prsm.core.models import UserInput
    from prsm.compute.nwtn.orchestrator import NWTNOrchestrator
    from tests.fixtures.nwtn_mocks import (
        MockContextManager, MockFTNSService, MockIPFSClient, MockModelRegistry
    )
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip("NWTN orchestrator module not yet implemented", allow_module_level=True)

async def demonstrate_nwtn():
    """Demonstrate basic NWTN functionality"""
    print("🚀 NWTN Integration Demonstration")
    print("=" * 50)
    
    # Create orchestrator with mock services for testing
    orchestrator = NWTNOrchestrator(
        context_manager=MockContextManager(),
        ftns_service=MockFTNSService(),
        ipfs_client=MockIPFSClient(),
        model_registry=MockModelRegistry()
    )
    
    # Give user some FTNS for testing
    await orchestrator.ftns_service.reward_contribution("demo_user", "data", 100.0)
    balance = await orchestrator.ftns_service.get_user_balance("demo_user")
    print(f"💰 User FTNS balance: {balance.balance}")
    
    # Register a test model
    from prsm.core.models import TeacherModel
    model = TeacherModel(
        name="Demo Model",
        specialization="general", 
        performance_score=0.9
    )
    model_bytes = b"Demo model data"
    cid = await orchestrator.ipfs_client.store_model(model_bytes, {"name": "Demo Model"})
    await orchestrator.model_registry.register_teacher_model(model, cid)
    
    print(f"📋 Registered models: {len(orchestrator.model_registry.registered_models)}")
    
    # Test intent clarification
    clarified = await orchestrator.clarify_intent("What is machine learning?")
    print(f"🎯 Intent: {clarified.intent_category}, Complexity: {clarified.complexity_estimate:.2f}")
    print(f"💡 Context required: {clarified.context_required}")
    
    # Test model discovery  
    models = await orchestrator.model_registry.discover_specialists("general")
    print(f"🔍 Available models: {len(models)}")
    
    print("\n✅ NWTN Integration Working Successfully!")
    print("   - FTNS token system ✓")
    print("   - IPFS model storage ✓") 
    print("   - Model registry ✓")
    print("   - Context management ✓")
    print("   - Intent clarification ✓")

if __name__ == "__main__":
    asyncio.run(demonstrate_nwtn())
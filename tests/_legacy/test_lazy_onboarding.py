
import pytest
import sys
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
from prsm.core.utils.dependency_manager import DependencyManager

@pytest.mark.asyncio
async def test_lazy_initialization():
    """Verify that orchestrator can be created in a skeleton environment"""
    # 1. Create orchestrator (should NOT trigger heavy imports yet)
    orchestrator = NeuroSymbolicOrchestrator(node_id="skeleton_node")
    assert orchestrator._initialized is False
    print("✅ Skeleton Orchestrator created successfully.")

@pytest.mark.asyncio
async def test_on_demand_hydration():
    """Verify that hydration is triggered only when a task starts"""
    orchestrator = NeuroSymbolicOrchestrator(node_id="test_hydration_node")
    
    # 1. Mock DependencyManager.is_installed to pretend some heavy deps are missing
    original_is_installed = DependencyManager.is_installed
    DependencyManager.is_installed = lambda name: False # Pretend everything is missing
    
    # 2. Mock hydrate_environment to avoid actual pip install in this test
    hydration_triggered = False
    def mock_hydrate():
        nonlocal hydration_triggered
        hydration_triggered = True
        return True
    
    DependencyManager.hydrate_environment = mock_hydrate
    
    # 3. Running a task should trigger hydration
    try:
        await orchestrator.solve_task("Test query", "Context")
    except Exception as e:
        # It might fail later due to other missing mocks, but we check the trigger
        pass
        
    assert hydration_triggered is True
    print("✅ Hydration trigger verified.")
    
    # Restore original methods
    DependencyManager.is_installed = original_is_installed

"""
Unit tests for NeuroSymbolicOrchestrator with BackendRegistry integration.

Tests verify that:
1. The orchestrator uses BackendRegistry for inference (not hardcoded responses)
2. Token counts are properly tracked and returned
3. Inference source is correctly reported
4. The reasoning trace maintains expected structure
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
from prsm.compute.nwtn.backends import BackendRegistry, BackendConfig, BackendType
from prsm.compute.nwtn.backends.base import GenerateResult, TokenUsage


class TestNeuroSymbolicOrchestratorWithBackend:
    """Test NeuroSymbolicOrchestrator with MockBackend injection."""
    
    @pytest.fixture
    def mock_backend_registry(self):
        """Create a mock BackendRegistry for testing."""
        # Create mock registry
        registry = MagicMock(spec=BackendRegistry)
        registry._initialized = True
        
        # Create mock inference result for System 1 (proposal)
        s1_result = MagicMock(spec=GenerateResult)
        s1_result.content = "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers."
        s1_result.model_id = "mock-model-v1"
        s1_result.provider = BackendType.MOCK
        s1_result.token_usage = MagicMock(spec=TokenUsage)
        s1_result.token_usage.total_tokens = 45
        
        # Create mock inference result for System 2 (verification)
        s2_result = MagicMock(spec=GenerateResult)
        s2_result.content = "Verified: Quantum computing leverages quantum bits (qubits) that can exist in superposition states, enabling parallel computation paths. Key applications include cryptography, optimization, and quantum simulation."
        s2_result.model_id = "mock-model-v1"
        s2_result.provider = BackendType.MOCK
        s2_result.token_usage = MagicMock(spec=TokenUsage)
        s2_result.token_usage.total_tokens = 62
        
        # Setup async mock to return different results based on call order
        call_count = [0]
        async def mock_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return s1_result
            return s2_result
        
        registry.execute_with_fallback = mock_execute
        return registry
    
    @pytest.mark.asyncio
    async def test_solve_task_uses_backend_registry(self, mock_backend_registry):
        """Test that solve_task uses BackendRegistry for inference, not hardcoded strings."""
        orchestrator = NeuroSymbolicOrchestrator(
            node_id="test_node",
            backend_registry=mock_backend_registry
        )
        
        result = await orchestrator.solve_task("What is quantum computing?", "")
        
        # Assert output is NOT the old hardcoded prefix
        assert not result["output"].startswith("Neural intuition for")
        assert not result["output"].startswith("Verified breakthrough:")
        
        # Assert output contains real mock content
        assert "quantum" in result["output"].lower()
        assert len(result["output"]) > 50  # Substantial content
    
    @pytest.mark.asyncio
    async def test_solve_task_returns_token_counts(self, mock_backend_registry):
        """Test that solve_task returns accumulated token counts."""
        orchestrator = NeuroSymbolicOrchestrator(
            node_id="test_node",
            backend_registry=mock_backend_registry
        )
        
        result = await orchestrator.solve_task("What is quantum computing?", "")
        
        # Assert tokens_used is present and positive
        assert "tokens_used" in result
        assert result["tokens_used"] > 0
        
        # In deep mode (default), should have tokens from both S1 and S2
        # S1: 45 tokens, S2: 62 tokens = 107 total
        assert result["tokens_used"] == 107
    
    @pytest.mark.asyncio
    async def test_solve_task_reports_inference_source(self, mock_backend_registry):
        """Test that solve_task reports the inference source correctly."""
        orchestrator = NeuroSymbolicOrchestrator(
            node_id="test_node",
            backend_registry=mock_backend_registry
        )
        
        result = await orchestrator.solve_task("What is quantum computing?", "")
        
        # Assert inference_source is present and correct
        assert "inference_source" in result
        assert result["inference_source"] == "mock"
    
    @pytest.mark.asyncio
    async def test_solve_task_maintains_reasoning_trace(self, mock_backend_registry):
        """Test that the reasoning trace still has all expected steps."""
        orchestrator = NeuroSymbolicOrchestrator(
            node_id="test_node",
            backend_registry=mock_backend_registry
        )
        
        result = await orchestrator.solve_task("What is quantum computing?", "")
        
        # Assert trace exists and has expected steps
        assert "trace" in result
        trace = result["trace"]
        
        # Check for expected trace steps
        # Trace format uses "a" key for action (step name)
        trace_steps = [step.get("a") for step in trace]
        assert "INIT" in trace_steps
        assert "S1_PROPOSAL" in trace_steps
        assert "S2_VERIFICATION" in trace_steps
    
    @pytest.mark.asyncio
    async def test_solve_task_light_mode_uses_only_s1(self, mock_backend_registry):
        """Test that light mode only uses System 1 (no S2 verification).
        
        Note: The verification mode is determined internally by a random roll.
        This test patches the random generator to force light mode.
        """
        orchestrator = NeuroSymbolicOrchestrator(
            node_id="test_node",
            backend_registry=mock_backend_registry
        )
        
        # Patch the deterministic generator to force light mode (roll <= 0.7)
        # Note: get_local_generator is imported locally inside solve_task, so we patch at source
        with patch('prsm.core.utils.deterministic.get_local_generator') as mock_gen:
            mock_rng = MagicMock()
            mock_rng.next_float.return_value = 0.5  # Forces light mode (roll <= 0.7)
            mock_gen.return_value = mock_rng
            
            result = await orchestrator.solve_task("What is quantum computing?", "")
        
        # In light mode, only S1 tokens should be counted
        assert result["tokens_used"] == 45  # Only S1 tokens
        
        # Trace should not have S2_VERIFICATION
        trace_steps = [step.get("a") for step in result["trace"]]
        assert "S2_VERIFICATION" not in trace_steps

"""
Unit tests for Compute Provider NWTN Orchestrator Integration

Tests the wiring of NWTN orchestrator to compute provider for real P2P inference.
Validates that:
- Inference jobs dispatch through NWTN orchestrator when wired
- Embedding jobs use backend registry when available
- Graceful fallback to mock when orchestrator not configured
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from prsm.node.compute_provider import ComputeProvider, ComputeJob, JobType


# ── Test Fixtures ───────────────────────────────────────────────────

@dataclass
class MockNWTNResponse:
    """Mock NWTN response matching NWTNResponse interface"""
    session_id: str
    response: str
    context_used: int
    ftns_charged: float
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    safety_validated: bool = True
    confidence_score: float = 0.85
    models_used: List[str] = field(default_factory=lambda: ["mock-model"])
    processing_time: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockEmbedResult:
    """Mock embedding result matching EmbedResult interface"""
    embedding: List[float]
    model_id: str
    provider: Any
    token_count: int


@dataclass
class MockProvider:
    """Mock provider enum value"""
    value: str = "mock"


class MockBackendRegistry:
    """Mock backend registry for testing"""
    
    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
        self.call_count = 0
        self.last_prompt = None
        self.last_text = None
    
    async def execute_with_fallback(self, prompt: str, **kwargs):
        """Mock execute that returns a mock result"""
        self.call_count += 1
        self.last_prompt = prompt
        
        if not self.should_succeed:
            from prsm.compute.nwtn.backends.exceptions import AllBackendsFailedError
            raise AllBackendsFailedError("Mock failure", errors=[("mock", "test error")])
        
        # Return mock GenerateResult
        @dataclass
        class MockGenerateResult:
            content: str = f"Mock response to: {prompt[:50]}..."
            model_id: str = "mock-model"
            provider: MockProvider = field(default_factory=MockProvider)
            token_usage: Any = None
            finish_reason: str = "stop"
        
        return MockGenerateResult()
    
    async def embed_with_fallback(self, text: str, model_id: str = None, dimensions: int = 1536):
        """Mock embedding generation"""
        self.call_count += 1
        self.last_text = text
        
        if not self.should_succeed:
            from prsm.compute.nwtn.backends.exceptions import AllBackendsFailedError
            raise AllBackendsFailedError("Mock embedding failure", errors=[("mock", "test error")])
        
        # Return normalized mock embedding
        import hashlib
        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(dimensions):
            byte_val = text_hash[i % len(text_hash)]
            embedding.append((byte_val - 128) / 128.0)
        
        # Normalize to unit vector
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return MockEmbedResult(
            embedding=embedding,
            model_id=model_id or "mock-embedding",
            provider=MockProvider("mock"),
            token_count=len(text.split()),
        )


class MockNWTNOrchestrator:
    """Mock NWTN orchestrator for testing"""
    
    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
        self.call_count = 0
        self.last_query = None
        self.last_user_id = None
        self.backend_registry = MockBackendRegistry(should_succeed)
    
    async def process_query(self, user_input):
        """Mock process_query that tracks calls and returns mock response"""
        self.call_count += 1
        self.last_query = user_input.prompt
        self.last_user_id = user_input.user_id
        
        if not self.should_succeed:
            raise Exception("Mock orchestrator failure")
        
        return MockNWTNResponse(
            session_id="test-session-123",
            response=f"Real AI response to: {user_input.prompt[:50]}...",
            context_used=100,
            ftns_charged=0.05,
            reasoning_trace=[
                {"agent_type": "architect", "output": {"intent": "research"}},
                {"agent_type": "executor", "output": {"analysis": "completed"}},
            ],
            models_used=["claude-3-5-sonnet"],
        )


def _make_identity(node_id: str = "test-node"):
    """Create a mock node identity"""
    identity = MagicMock()
    identity.node_id = node_id
    identity.sign = MagicMock(return_value="test-signature")
    identity.public_key_b64 = "dGVzdHB1YmtleQ=="
    return identity


def _make_transport(peer_count: int = 0):
    """Create a mock transport"""
    transport = MagicMock()
    transport.peer_count = peer_count
    return transport


def _make_gossip():
    """Create a mock gossip protocol"""
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock()
    return gossip


def _make_ledger():
    """Create a mock ledger"""
    import time
    ledger = MagicMock()
    ledger.credit = AsyncMock(return_value=MagicMock(
        tx_id="tx-123",
        from_wallet="system",
        to_wallet="test-node",
        amount=1.0,
        tx_type=MagicMock(value="compute_earning"),
        description="test",
        timestamp=time.time(),
    ))
    return ledger


def _make_compute_provider(orchestrator=None):
    """Create a compute provider with optional orchestrator"""
    provider = ComputeProvider(
        identity=_make_identity(),
        transport=_make_transport(),
        gossip=_make_gossip(),
        ledger=_make_ledger(),
    )
    provider.orchestrator = orchestrator
    provider._running = True
    return provider


# ── Inference Tests ───────────────────────────────────────────────────

class TestInferenceWithOrchestrator:
    """Tests for inference dispatch through NWTN orchestrator"""
    
    @pytest.mark.asyncio
    async def test_inference_with_orchestrator_dispatches_correctly(self):
        """When orchestrator is wired, inference dispatches through it"""
        mock_orchestrator = MockNWTNOrchestrator(should_succeed=True)
        provider = _make_compute_provider(orchestrator=mock_orchestrator)
        
        job = ComputeJob(
            job_id="job-inference-001",
            job_type=JobType.INFERENCE,
            requester_id="remote-node-456",
            payload={
                "prompt": "What is CRISPR and how does it work?",
                "model": "claude-3-5-sonnet",
            },
            ftns_budget=2.0,
        )
        
        result = await provider._run_inference(job)
        
        # Verify orchestrator was called
        assert mock_orchestrator.call_count == 1
        assert "CRISPR" in mock_orchestrator.last_query
        
        # Verify result structure
        assert result["source"] == "nwtn_orchestrator"
        assert "Real AI response" in result["response"]
        assert result["tokens_used"] == 100
        assert result["ftns_charged"] == 0.05
        assert "claude-3-5-sonnet" in result["models_used"]
    
    @pytest.mark.asyncio
    async def test_inference_without_orchestrator_returns_mock(self):
        """When orchestrator is not wired, inference returns mock response"""
        provider = _make_compute_provider(orchestrator=None)
        
        job = ComputeJob(
            job_id="job-inference-002",
            job_type=JobType.INFERENCE,
            requester_id="remote-node-456",
            payload={
                "prompt": "What is machine learning?",
                "model": "local",
            },
            ftns_budget=1.0,
        )
        
        result = await provider._run_inference(job)
        
        # Verify mock response
        assert result["source"] == "mock"
        assert "[PRSM node" in result["response"]
        assert "warning" in result
        assert "No LLM backend configured" in result["warning"]
    
    @pytest.mark.asyncio
    async def test_inference_orchestrator_failure_falls_back_gracefully(self):
        """When orchestrator fails, inference falls back to mock"""
        mock_orchestrator = MockNWTNOrchestrator(should_succeed=False)
        provider = _make_compute_provider(orchestrator=mock_orchestrator)
        
        job = ComputeJob(
            job_id="job-inference-003",
            job_type=JobType.INFERENCE,
            requester_id="remote-node-456",
            payload={
                "prompt": "Explain quantum computing",
                "model": "gpt-4",
            },
            ftns_budget=1.5,
        )
        
        result = await provider._run_inference(job)
        
        # Should fall back to mock on failure
        assert result["source"] == "mock"
        assert "[PRSM node" in result["response"]
    
    @pytest.mark.asyncio
    async def test_inference_preserves_prompt_in_result(self):
        """Inference result includes truncated prompt for traceability"""
        mock_orchestrator = MockNWTNOrchestrator(should_succeed=True)
        provider = _make_compute_provider(orchestrator=mock_orchestrator)
        
        long_prompt = "A" * 500  # Long prompt
        
        job = ComputeJob(
            job_id="job-inference-004",
            job_type=JobType.INFERENCE,
            requester_id="remote-node-456",
            payload={
                "prompt": long_prompt,
                "model": "claude-3-5-sonnet",
            },
            ftns_budget=2.0,
        )
        
        result = await provider._run_inference(job)
        
        # Prompt should be truncated to 200 chars in result
        assert len(result["prompt"]) <= 200
        assert result["prompt"] == long_prompt[:200]


# ── Embedding Tests ───────────────────────────────────────────────────

class TestEmbeddingWithBackendRegistry:
    """Tests for embedding dispatch through backend registry"""
    
    @pytest.mark.asyncio
    async def test_embedding_with_backend_registry_dispatches_correctly(self):
        """When orchestrator has backend_registry, embedding uses it"""
        mock_orchestrator = MockNWTNOrchestrator(should_succeed=True)
        provider = _make_compute_provider(orchestrator=mock_orchestrator)
        
        job = ComputeJob(
            job_id="job-embedding-001",
            job_type=JobType.EMBEDDING,
            requester_id="remote-node-789",
            payload={
                "text": "This is a test document for embedding",
                "dimensions": 768,
            },
            ftns_budget=0.5,
        )
        
        result = await provider._run_embedding(job)
        
        # Verify backend registry was called
        assert mock_orchestrator.backend_registry.call_count == 1
        assert "test document" in mock_orchestrator.backend_registry.last_text
        
        # Verify result structure
        assert result["source"] == "backend_registry"
        assert result["dimensions"] == 768
        assert len(result["embedding"]) == 768
        assert result["provider"] == "mock"
    
    @pytest.mark.asyncio
    async def test_embedding_without_orchestrator_returns_mock(self):
        """When orchestrator is not wired, embedding returns mock"""
        provider = _make_compute_provider(orchestrator=None)
        
        job = ComputeJob(
            job_id="job-embedding-002",
            job_type=JobType.EMBEDDING,
            requester_id="remote-node-789",
            payload={
                "text": "Another test document",
                "dimensions": 1536,
            },
            ftns_budget=0.3,
        )
        
        result = await provider._run_embedding(job)
        
        # Verify mock response
        assert result["source"] == "mock"
        assert result["provider"] == "mock"
        assert "warning" in result
        assert len(result["embedding"]) == 1536
    
    @pytest.mark.asyncio
    async def test_embedding_fallback_is_normalized(self):
        """Mock embedding fallback produces normalized unit vector"""
        provider = _make_compute_provider(orchestrator=None)
        
        job = ComputeJob(
            job_id="job-embedding-003",
            job_type=JobType.EMBEDDING,
            requester_id="remote-node-789",
            payload={
                "text": "Test for normalization",
                "dimensions": 128,
            },
            ftns_budget=0.2,
        )
        
        result = await provider._run_embedding(job)
        
        # Verify normalization (unit vector: magnitude should be ~1.0)
        embedding = result["embedding"]
        magnitude = sum(x * x for x in embedding) ** 0.5
        assert abs(magnitude - 1.0) < 0.01  # Allow small floating point error
    
    @pytest.mark.asyncio
    async def test_embedding_backend_failure_falls_back_gracefully(self):
        """When backend registry fails, embedding falls back to mock"""
        mock_orchestrator = MockNWTNOrchestrator(should_succeed=True)
        mock_orchestrator.backend_registry.should_succeed = False
        provider = _make_compute_provider(orchestrator=mock_orchestrator)
        
        job = ComputeJob(
            job_id="job-embedding-004",
            job_type=JobType.EMBEDDING,
            requester_id="remote-node-789",
            payload={
                "text": "Test for fallback",
                "dimensions": 512,
            },
            ftns_budget=0.4,
        )
        
        result = await provider._run_embedding(job)
        
        # Should fall back to mock on failure
        assert result["source"] == "mock"
        assert len(result["embedding"]) == 512


# ── Backward Compatibility Tests ─────────────────────────────────────

class TestBackwardCompatibility:
    """Tests for backward compatibility when orchestrator not configured"""
    
    @pytest.mark.asyncio
    async def test_node_without_orchestrator_still_works(self):
        """Node should function without orchestrator (mock mode)"""
        provider = _make_compute_provider(orchestrator=None)
        
        # Both inference and embedding should work
        inference_job = ComputeJob(
            job_id="job-compat-001",
            job_type=JobType.INFERENCE,
            requester_id="test-requester",
            payload={"prompt": "Test prompt"},
            ftns_budget=1.0,
        )
        
        embedding_job = ComputeJob(
            job_id="job-compat-002",
            job_type=JobType.EMBEDDING,
            requester_id="test-requester",
            payload={"text": "Test text"},
            ftns_budget=0.5,
        )
        
        inference_result = await provider._run_inference(inference_job)
        embedding_result = await provider._run_embedding(embedding_job)
        
        # Both should return mock results
        assert inference_result["source"] == "mock"
        assert embedding_result["source"] == "mock"
    
    @pytest.mark.asyncio
    async def test_benchmark_job_unaffected_by_orchestrator(self):
        """Benchmark jobs should not use orchestrator"""
        mock_orchestrator = MockNWTNOrchestrator(should_succeed=True)
        provider = _make_compute_provider(orchestrator=mock_orchestrator)
        
        job = ComputeJob(
            job_id="job-benchmark-001",
            job_type=JobType.BENCHMARK,
            requester_id="test-requester",
            payload={"iterations": 10000},
            ftns_budget=0.5,
        )
        
        result = await provider._run_benchmark(job)
        
        # Benchmark should not call orchestrator
        assert mock_orchestrator.call_count == 0
        
        # Should return benchmark results
        assert "benchmark_type" in result
        assert "primes_found" in result
        assert "elapsed_seconds" in result


# ── Integration Marker Tests ─────────────────────────────────────────

class TestIntegrationMarkers:
    """Tests that verify integration is properly marked in results"""
    
    @pytest.mark.asyncio
    async def test_inference_result_includes_provider_node(self):
        """Inference result should identify the providing node"""
        mock_orchestrator = MockNWTNOrchestrator(should_succeed=True)
        provider = _make_compute_provider(orchestrator=mock_orchestrator)
        
        job = ComputeJob(
            job_id="job-integration-001",
            job_type=JobType.INFERENCE,
            requester_id="remote-node",
            payload={"prompt": "Test"},
            ftns_budget=1.0,
        )
        
        result = await provider._run_inference(job)
        
        assert "provider_node" in result
        assert result["provider_node"] == provider.identity.node_id
    
    @pytest.mark.asyncio
    async def test_embedding_result_includes_provider_node(self):
        """Embedding result should identify the providing node"""
        mock_orchestrator = MockNWTNOrchestrator(should_succeed=True)
        provider = _make_compute_provider(orchestrator=mock_orchestrator)
        
        job = ComputeJob(
            job_id="job-integration-002",
            job_type=JobType.EMBEDDING,
            requester_id="remote-node",
            payload={"text": "Test"},
            ftns_budget=0.5,
        )
        
        result = await provider._run_embedding(job)
        
        assert "provider_node" in result
        assert result["provider_node"] == provider.identity.node_id

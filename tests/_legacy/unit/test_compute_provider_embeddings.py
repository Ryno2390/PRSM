"""
Unit tests for Compute Provider Embedding Integration

Tests the compute provider's embedding functionality which now uses
the backend registry for real embeddings when available.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib

from prsm.node.compute_provider import ComputeProvider, ComputeJob, JobType
from prsm.compute.nwtn.backends import (
    BackendType,
    BackendRegistry,
    BackendConfig,
    EmbedResult,
)


class TestComputeProviderEmbedding:
    """Tests for compute provider embedding functionality"""
    
    @pytest.fixture
    def mock_identity(self):
        """Create a mock node identity"""
        identity = MagicMock()
        identity.node_id = "test-node-123"
        identity.sign = MagicMock(return_value="signature")
        identity.public_key_b64 = "test-public-key"
        return identity
    
    @pytest.fixture
    def mock_transport(self):
        """Create a mock transport"""
        transport = MagicMock()
        transport.peer_count = 0
        return transport
    
    @pytest.fixture
    def mock_gossip(self):
        """Create a mock gossip protocol"""
        gossip = MagicMock()
        gossip.subscribe = MagicMock()
        gossip.publish = AsyncMock()
        return gossip
    
    @pytest.fixture
    def mock_ledger(self):
        """Create a mock ledger"""
        ledger = MagicMock()
        ledger.credit = AsyncMock()
        return ledger
    
    @pytest.fixture
    def compute_provider(self, mock_identity, mock_transport, mock_gossip, mock_ledger):
        """Create a compute provider with mocked dependencies"""
        return ComputeProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
        )
    
    @pytest.fixture
    def embedding_job(self):
        """Create a sample embedding job"""
        return ComputeJob(
            job_id="test-job-123",
            job_type=JobType.EMBEDDING,
            requester_id="test-requester",
            payload={
                "text": "This is test text for embedding",
                "dimensions": 1536
            },
            ftns_budget=1.0,
        )
    
    @pytest.mark.asyncio
    async def test_run_embedding_fallback_without_registry(self, compute_provider, embedding_job):
        """Test that embedding works with fallback when no registry is available"""
        # Ensure no orchestrator/backend_registry
        compute_provider.orchestrator = None
        
        result = await compute_provider._run_embedding(embedding_job)
        
        assert "embedding" in result
        assert result["dimensions"] == 1536
        assert len(result["embedding"]) == 1536
        assert result["source"] == "mock"
        assert result["provider"] == "mock"
    
    @pytest.mark.asyncio
    async def test_run_embedding_fallback_normalized(self, compute_provider, embedding_job):
        """Test that fallback embedding is normalized to unit vector"""
        compute_provider.orchestrator = None
        
        result = await compute_provider._run_embedding(embedding_job)
        
        # Check normalization
        magnitude = sum(x * x for x in result["embedding"]) ** 0.5
        assert 0.99 <= magnitude <= 1.01, f"Expected unit vector, got magnitude {magnitude}"
    
    @pytest.mark.asyncio
    async def test_run_embedding_fallback_deterministic(self, compute_provider):
        """Test that fallback embedding is deterministic"""
        compute_provider.orchestrator = None
        
        job1 = ComputeJob(
            job_id="job-1",
            job_type=JobType.EMBEDDING,
            requester_id="test",
            payload={"text": "same text", "dimensions": 1536},
            ftns_budget=1.0,
        )
        job2 = ComputeJob(
            job_id="job-2",
            job_type=JobType.EMBEDDING,
            requester_id="test",
            payload={"text": "same text", "dimensions": 1536},
            ftns_budget=1.0,
        )
        
        result1 = await compute_provider._run_embedding(job1)
        result2 = await compute_provider._run_embedding(job2)
        
        assert result1["embedding"] == result2["embedding"]
    
    @pytest.mark.asyncio
    async def test_run_embedding_custom_dimensions(self, compute_provider):
        """Test that custom dimensions are respected in fallback"""
        compute_provider.orchestrator = None
        
        job = ComputeJob(
            job_id="test-job",
            job_type=JobType.EMBEDDING,
            requester_id="test",
            payload={"text": "test text", "dimensions": 768},
            ftns_budget=1.0,
        )
        
        result = await compute_provider._run_embedding(job)
        
        assert len(result["embedding"]) == 768
        assert result["dimensions"] == 768
    
    @pytest.mark.asyncio
    async def test_run_embedding_with_backend_registry(self, compute_provider, embedding_job):
        """Test that embedding uses backend registry when available"""
        # Create a mock orchestrator with backend_registry
        mock_registry = MagicMock(spec=BackendRegistry)
        mock_registry.embed_with_fallback = AsyncMock(return_value=EmbedResult(
            embedding=[0.1] * 1536,
            model_id="text-embedding-3-small",
            provider=BackendType.OPENAI,
            token_count=10,
            metadata={"dimensions": 1536}
        ))
        
        mock_orchestrator = MagicMock()
        mock_orchestrator.backend_registry = mock_registry
        
        compute_provider.orchestrator = mock_orchestrator
        
        result = await compute_provider._run_embedding(embedding_job)
        
        # Verify registry was called
        mock_registry.embed_with_fallback.assert_called_once_with(
            text="This is test text for embedding",
            model_id=None,
            dimensions=1536
        )
        
        # Verify result structure
        assert result["source"] == "backend_registry"
        assert result["model_id"] == "text-embedding-3-small"
        assert result["provider"] == "openai"
        assert result["token_count"] == 10
        assert len(result["embedding"]) == 1536
    
    @pytest.mark.asyncio
    async def test_run_embedding_registry_failure_falls_back(self, compute_provider, embedding_job):
        """Test that embedding falls back when registry fails"""
        # Create a mock orchestrator with failing backend_registry
        mock_registry = MagicMock(spec=BackendRegistry)
        mock_registry.embed_with_fallback = AsyncMock(side_effect=Exception("API Error"))
        
        mock_orchestrator = MagicMock()
        mock_orchestrator.backend_registry = mock_registry
        
        compute_provider.orchestrator = mock_orchestrator
        
        result = await compute_provider._run_embedding(embedding_job)
        
        # Should fall back to hash-based embedding
        assert result["source"] == "mock"
        assert result["provider"] == "mock"
        assert len(result["embedding"]) == 1536
    
    @pytest.mark.asyncio
    async def test_run_embedding_with_model_id(self, compute_provider):
        """Test that model_id is passed to backend registry"""
        mock_registry = MagicMock(spec=BackendRegistry)
        mock_registry.embed_with_fallback = AsyncMock(return_value=EmbedResult(
            embedding=[0.1] * 1536,
            model_id="text-embedding-3-large",
            provider=BackendType.OPENAI,
            token_count=10,
            metadata={"dimensions": 1536}
        ))
        
        mock_orchestrator = MagicMock()
        mock_orchestrator.backend_registry = mock_registry
        
        compute_provider.orchestrator = mock_orchestrator
        
        job = ComputeJob(
            job_id="test-job",
            job_type=JobType.EMBEDDING,
            requester_id="test",
            payload={
                "text": "test text",
                "dimensions": 3072,
                "model_id": "text-embedding-3-large"
            },
            ftns_budget=1.0,
        )
        
        result = await compute_provider._run_embedding(job)
        
        # Verify model_id was passed
        mock_registry.embed_with_fallback.assert_called_once_with(
            text="test text",
            model_id="text-embedding-3-large",
            dimensions=3072
        )
        
        assert result["model_id"] == "text-embedding-3-large"
    
    @pytest.mark.asyncio
    async def test_run_embedding_default_dimensions(self, compute_provider):
        """Test that default dimensions is 1536 (OpenAI small)"""
        compute_provider.orchestrator = None
        
        job = ComputeJob(
            job_id="test-job",
            job_type=JobType.EMBEDDING,
            requester_id="test",
            payload={"text": "test text"},  # No dimensions specified
            ftns_budget=1.0,
        )
        
        result = await compute_provider._run_embedding(job)
        
        assert result["dimensions"] == 1536
        assert len(result["embedding"]) == 1536


class TestEmbeddingSimilarity:
    """Tests for embedding similarity behavior"""
    
    @pytest.fixture
    def compute_provider(self):
        """Create a minimal compute provider for testing"""
        return ComputeProvider(
            identity=MagicMock(node_id="test", sign=MagicMock(return_value="sig")),
            transport=MagicMock(peer_count=0),
            gossip=MagicMock(),
            ledger=MagicMock(),
        )
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
    
    @pytest.mark.asyncio
    async def test_fallback_embeddings_support_similarity(self, compute_provider):
        """Test that fallback embeddings can be used for similarity comparison"""
        compute_provider.orchestrator = None
        
        # Create embeddings for similar and different texts
        job1 = ComputeJob(
            job_id="job-1",
            job_type=JobType.EMBEDDING,
            requester_id="test",
            payload={"text": "machine learning algorithms", "dimensions": 1536},
            ftns_budget=1.0,
        )
        job2 = ComputeJob(
            job_id="job-2",
            job_type=JobType.EMBEDDING,
            requester_id="test",
            payload={"text": "machine learning algorithms", "dimensions": 1536},
            ftns_budget=1.0,
        )
        job3 = ComputeJob(
            job_id="job-3",
            job_type=JobType.EMBEDDING,
            requester_id="test",
            payload={"text": "completely different topic about cooking", "dimensions": 1536},
            ftns_budget=1.0,
        )
        
        result1 = await compute_provider._run_embedding(job1)
        result2 = await compute_provider._run_embedding(job2)
        result3 = await compute_provider._run_embedding(job3)
        
        # Same text should have identical embeddings
        assert result1["embedding"] == result2["embedding"]
        
        # Different texts should have different embeddings
        assert result1["embedding"] != result3["embedding"]
        
        # Verify vectors are normalized (cosine similarity with self = 1.0)
        sim_self = self._cosine_similarity(result1["embedding"], result1["embedding"])
        assert abs(sim_self - 1.0) < 0.01

"""Item 1 (2026-05-05) — unit tests for the local sentence-transformers
fallback in RealEmbeddingAPI.

Mocks the SentenceTransformer model so tests run fast and don't download
the ~80MB MiniLM model in CI. A separate slow-marked integration test
exercises the real model.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip env vars that would change provider availability."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PRSM_DISABLE_LOCAL_EMBEDDING", raising=False)
    monkeypatch.delenv("PRSM_LOCAL_EMBEDDING_MODEL", raising=False)


def test_local_provider_appears_in_fallback_order_when_lib_available():
    """sentence_transformers should sit between openai and mock."""
    from prsm.data.embeddings import real_embedding_api as mod
    with patch.object(mod, "HAS_SENTENCE_TRANSFORMERS", True):
        api = mod.RealEmbeddingAPI()
    assert "sentence_transformers" in api.providers
    # Mock should still exist (Item 2 will remove it later)
    assert "mock" in api.providers
    # Order: sentence_transformers comes before mock
    st_idx = api.fallback_order.index("sentence_transformers")
    mock_idx = api.fallback_order.index("mock")
    assert st_idx < mock_idx


def test_local_provider_excluded_when_disabled_via_env(monkeypatch):
    """PRSM_DISABLE_LOCAL_EMBEDDING=1 keeps the local provider out."""
    monkeypatch.setenv("PRSM_DISABLE_LOCAL_EMBEDDING", "1")
    from prsm.data.embeddings import real_embedding_api as mod
    with patch.object(mod, "HAS_SENTENCE_TRANSFORMERS", True):
        api = mod.RealEmbeddingAPI()
    assert "sentence_transformers" not in api.providers
    assert "sentence_transformers" not in api.fallback_order


def test_local_provider_skipped_when_lib_missing():
    """Without sentence-transformers installed, provider isn't registered."""
    from prsm.data.embeddings import real_embedding_api as mod
    with patch.object(mod, "HAS_SENTENCE_TRANSFORMERS", False):
        api = mod.RealEmbeddingAPI()
    assert "sentence_transformers" not in api.providers


def test_custom_model_name_via_env(monkeypatch):
    """PRSM_LOCAL_EMBEDDING_MODEL overrides the default."""
    monkeypatch.setenv(
        "PRSM_LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/all-mpnet-base-v2",
    )
    from prsm.data.embeddings import real_embedding_api as mod
    with patch.object(mod, "HAS_SENTENCE_TRANSFORMERS", True):
        api = mod.RealEmbeddingAPI()
    assert api.providers["sentence_transformers"].model == (
        "sentence-transformers/all-mpnet-base-v2"
    )


@pytest.mark.asyncio
async def test_st_generate_with_mocked_model():
    """Lazy-load + encode flow — verify model is loaded on first use,
    cached on second use, and returns L2-normalised 1D arrays."""
    from prsm.data.embeddings import real_embedding_api as mod

    # Fake numpy batch: 2 texts × 384 dim, L2-normalised.
    fake_batch = np.random.randn(2, 384).astype(np.float32)
    fake_batch /= np.linalg.norm(fake_batch, axis=1, keepdims=True)

    fake_model = MagicMock()
    fake_model.encode.return_value = fake_batch

    fake_st_module = MagicMock()
    fake_st_module.SentenceTransformer.return_value = fake_model

    with patch.object(mod, "HAS_SENTENCE_TRANSFORMERS", True), \
         patch.dict(
            "sys.modules",
            {"sentence_transformers": fake_st_module},
         ):
        api = mod.RealEmbeddingAPI()
        # Pre-condition: model not loaded yet
        assert api._st_model is None
        results = await api.generate_embeddings(
            ["hello world", "goodbye world"],
            preferred_provider="sentence_transformers",
        )

    assert len(results) == 2
    for emb in results:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)
        # L2-norm should be ~1.0 since the mock returned normalised vectors
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    # Model should now be loaded (cached for subsequent calls)
    assert api._st_model is not None
    fake_st_module.SentenceTransformer.assert_called_once()


@pytest.mark.asyncio
async def test_st_model_loaded_only_once_across_calls():
    """Lazy-load happens exactly once even after multiple calls."""
    from prsm.data.embeddings import real_embedding_api as mod

    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
    fake_st_module = MagicMock()
    fake_st_module.SentenceTransformer.return_value = fake_model

    with patch.object(mod, "HAS_SENTENCE_TRANSFORMERS", True), \
         patch.dict(
            "sys.modules",
            {"sentence_transformers": fake_st_module},
         ):
        api = mod.RealEmbeddingAPI()
        for _ in range(5):
            await api.generate_embeddings(
                ["t"], preferred_provider="sentence_transformers"
            )
    fake_st_module.SentenceTransformer.assert_called_once()
    assert fake_model.encode.call_count == 5


def test_provider_dimension_matches_minilm():
    """all-MiniLM-L6-v2 produces 384-dim embeddings — fallback should match."""
    from prsm.data.embeddings import real_embedding_api as mod
    with patch.object(mod, "HAS_SENTENCE_TRANSFORMERS", True):
        api = mod.RealEmbeddingAPI()
    assert api.providers["sentence_transformers"].embedding_dimension == 384


@pytest.mark.asyncio
async def test_fallback_order_no_openai_uses_local():
    """When OpenAI is missing, requesting an embedding falls through to
    the local provider BEFORE mock."""
    from prsm.data.embeddings import real_embedding_api as mod

    fake_model = MagicMock()
    fake_batch = np.random.randn(1, 384).astype(np.float32)
    fake_batch /= np.linalg.norm(fake_batch, axis=1, keepdims=True)
    fake_model.encode.return_value = fake_batch

    fake_st_module = MagicMock()
    fake_st_module.SentenceTransformer.return_value = fake_model

    with patch.object(mod, "HAS_SENTENCE_TRANSFORMERS", True), \
         patch.object(mod, "HAS_OPENAI", False), \
         patch.dict(
            "sys.modules",
            {"sentence_transformers": fake_st_module},
         ):
        api = mod.RealEmbeddingAPI()
        assert "openai" not in api.providers
        # No preferred — should pick first available (sentence_transformers)
        result = await api.generate_embedding("test")

    # Verify it was the local provider that ran (not mock)
    fake_model.encode.assert_called_once()
    assert isinstance(result, np.ndarray)
    assert result.shape == (384,)

"""Item 5 (2026-05-06) — unit tests for chunked embedding in
ContentUploader._get_embedding.

Verifies that long documents are split into overlapping chunks, each
embedded independently, and the chunk vectors mean-pooled +
L2-renormalized into a single document-level fingerprint that
_SemanticIndex can use for cosine-similarity dedup.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from prsm.node.content_uploader import (
    ContentUploader,
    _CHUNK_SIZE_CHARS,
    _CHUNK_OVERLAP_CHARS,
    _MAX_CHUNKS,
    _MIN_EMBEDDING_CHARS,
    _split_for_embedding,
)


# ─────────────────────────────────────────────────────────────────────
# _split_for_embedding pure-function tests
# ─────────────────────────────────────────────────────────────────────


def test_split_short_text_returns_single_chunk():
    """Text ≤ chunk-size returns [text] unchanged — no copying or splitting."""
    text = "x" * (_CHUNK_SIZE_CHARS - 100)
    chunks = _split_for_embedding(text)
    assert chunks == [text]


def test_split_exactly_chunk_size_returns_single_chunk():
    """Boundary case: text == chunk-size exactly stays as one chunk."""
    text = "x" * _CHUNK_SIZE_CHARS
    chunks = _split_for_embedding(text)
    assert chunks == [text]


def test_split_long_text_overlapping_chunks():
    """Text > chunk-size produces overlapping chunks."""
    # Make text 2.5x chunk size so we get 2-3 chunks
    text = "abcdefghij" * (int(_CHUNK_SIZE_CHARS * 2.5) // 10)
    chunks = _split_for_embedding(text)
    assert len(chunks) > 1
    # Each chunk is at most the configured size
    for c in chunks:
        assert len(c) <= _CHUNK_SIZE_CHARS
    # Consecutive chunks overlap by exactly _CHUNK_OVERLAP_CHARS chars
    for i in range(len(chunks) - 1):
        # The end of chunk i should be the start of chunk i+1
        overlap_from_first = chunks[i][-_CHUNK_OVERLAP_CHARS:]
        overlap_from_second = chunks[i + 1][:_CHUNK_OVERLAP_CHARS]
        assert overlap_from_first == overlap_from_second, (
            f"chunk {i}/{i+1} overlap mismatch"
        )


def test_split_caps_at_max_chunks():
    """Adversarial 100MB text doesn't fan out unboundedly."""
    text = "x" * (_CHUNK_SIZE_CHARS * 200)  # would naturally be ~200 chunks
    chunks = _split_for_embedding(text)
    assert len(chunks) == _MAX_CHUNKS


# ─────────────────────────────────────────────────────────────────────
# ContentUploader._get_embedding integration tests
# ─────────────────────────────────────────────────────────────────────


def _make_uploader_with_embed(embedding_fn):
    """Build a minimal ContentUploader with a mocked embedding_fn."""
    return ContentUploader(
        identity=MagicMock(),
        gossip=MagicMock(),
        ledger=MagicMock(),
        embedding_fn=embedding_fn,
    )


@pytest.mark.asyncio
async def test_embed_returns_none_below_min_chars():
    """Text shorter than _MIN_EMBEDDING_CHARS → no embedding."""
    embedding_fn = AsyncMock()
    uploader = _make_uploader_with_embed(embedding_fn)
    result = await uploader._get_embedding(b"short")
    assert result is None
    embedding_fn.assert_not_called()


@pytest.mark.asyncio
async def test_embed_short_text_single_call():
    """Text fits in one chunk → exactly one embedding call, no pooling."""
    fake_vec = np.random.randn(384).astype(np.float32)
    fake_vec /= np.linalg.norm(fake_vec)
    embedding_fn = AsyncMock(return_value=fake_vec)
    uploader = _make_uploader_with_embed(embedding_fn)

    text = "Lorem ipsum dolor sit amet. " * 20  # ~560 chars, well under chunk size
    result = await uploader._get_embedding(text.encode())

    assert result is not None
    assert np.array_equal(result, fake_vec)
    assert embedding_fn.call_count == 1


@pytest.mark.asyncio
async def test_embed_long_text_calls_per_chunk():
    """Text > chunk-size triggers multi-chunk embedding."""
    # Make embedding_fn return a deterministic vector per chunk so we
    # can assert on the pooling
    call_log = []

    async def fake_embed(text):
        call_log.append(text)
        # Different vector per chunk based on a hash of the chunk text
        seed = sum(ord(c) for c in text[:100]) % 1000
        rng = np.random.RandomState(seed)
        v = rng.randn(384).astype(np.float32)
        return v / np.linalg.norm(v)

    uploader = _make_uploader_with_embed(fake_embed)
    text = "abcdefghij" * (_CHUNK_SIZE_CHARS // 5)  # ~2x chunk size
    result = await uploader._get_embedding(text.encode())

    assert result is not None
    # Multiple chunks → multiple embedding calls
    assert len(call_log) > 1


@pytest.mark.asyncio
async def test_embed_long_text_pooled_result_is_l2_normalized():
    """Mean-pooled chunk vectors get L2-renormalized so _SemanticIndex
    cosine similarity stays in [-1, 1]."""
    # Use distinct fake vectors per chunk
    vectors = [np.random.randn(384).astype(np.float32) for _ in range(3)]
    vectors = [v / np.linalg.norm(v) for v in vectors]
    call_count = [0]

    async def fake_embed(text):
        v = vectors[call_count[0] % len(vectors)]
        call_count[0] += 1
        return v

    uploader = _make_uploader_with_embed(fake_embed)
    text = "x" * (_CHUNK_SIZE_CHARS * 2)
    result = await uploader._get_embedding(text.encode())

    assert result is not None
    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 1e-5, (
        f"pooled embedding not L2-normalized (norm={norm})"
    )


@pytest.mark.asyncio
async def test_embed_long_text_pooled_is_mean_then_normalized():
    """Pooled vector should equal normalize(mean(chunk_vectors))."""
    v1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)
    fake_returns = [v1, v2]
    idx = [0]

    async def fake_embed(text):
        v = fake_returns[idx[0]]
        idx[0] += 1
        return v

    uploader = _make_uploader_with_embed(fake_embed)
    text = "x" * (_CHUNK_SIZE_CHARS + _CHUNK_SIZE_CHARS // 2)  # 2 chunks
    result = await uploader._get_embedding(text.encode())

    assert idx[0] == 2  # Confirmed 2 chunks
    expected = (v1 + v2) / 2
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(result, expected, atol=1e-6)


@pytest.mark.asyncio
async def test_embed_handles_chunk_failure_gracefully():
    """If one chunk's embedding returns None, pool the rest."""
    v_ok = np.random.randn(384).astype(np.float32)
    v_ok /= np.linalg.norm(v_ok)
    call_count = [0]

    async def flaky_embed(text):
        call_count[0] += 1
        # Second chunk fails; others succeed
        if call_count[0] == 2:
            return None
        return v_ok

    uploader = _make_uploader_with_embed(flaky_embed)
    text = "x" * (_CHUNK_SIZE_CHARS * 2)
    result = await uploader._get_embedding(text.encode())

    # Should still get a result — pooled from the chunks that did succeed
    assert result is not None
    assert result.shape == (384,)


@pytest.mark.asyncio
async def test_embed_returns_none_when_all_chunks_fail():
    """If every chunk's embedding fails, return None (not zero-vec)."""
    async def broken_embed(text):
        return None
    uploader = _make_uploader_with_embed(broken_embed)
    text = "x" * (_CHUNK_SIZE_CHARS * 2)
    result = await uploader._get_embedding(text.encode())
    assert result is None


@pytest.mark.asyncio
async def test_embed_returns_none_on_exception():
    """Any exception in the embedding path returns None gracefully."""
    async def explosive_embed(text):
        raise RuntimeError("boom")
    uploader = _make_uploader_with_embed(explosive_embed)
    result = await uploader._get_embedding(b"a" * 1000)
    assert result is None


@pytest.mark.asyncio
async def test_embed_returns_none_when_no_fn_configured():
    """ContentUploader without embedding_fn returns None always."""
    uploader = _make_uploader_with_embed(None)
    result = await uploader._get_embedding(b"a" * 5000)
    assert result is None


# ─────────────────────────────────────────────────────────────────────
# Item 2 — mock removed from auto-fallback chain
# ─────────────────────────────────────────────────────────────────────


def test_mock_not_in_default_fallback_chain(monkeypatch):
    """Default config: mock is registered as a provider but NOT in the
    auto-fallback chain. Tests still get it via preferred_provider='mock'."""
    monkeypatch.delenv("PRSM_ALLOW_MOCK_EMBEDDING_FALLBACK", raising=False)
    from prsm.data.embeddings import real_embedding_api as mod
    api = mod.RealEmbeddingAPI()
    assert "mock" in api.providers, "mock must remain available for opt-in"
    assert "mock" not in api.fallback_order, (
        "mock must NOT be in auto-fallback (Item 2 fix)"
    )


def test_mock_re_enabled_via_env_override(monkeypatch):
    """PRSM_ALLOW_MOCK_EMBEDDING_FALLBACK=1 restores the old auto-fall behavior."""
    monkeypatch.setenv("PRSM_ALLOW_MOCK_EMBEDDING_FALLBACK", "1")
    from prsm.data.embeddings import real_embedding_api as mod
    api = mod.RealEmbeddingAPI()
    assert "mock" in api.fallback_order


@pytest.mark.asyncio
async def test_generate_raises_when_no_real_provider(monkeypatch):
    """No openai key, no sentence_transformers, no override → hard error
    instead of silent fallthrough to mock pseudo-vectors."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PRSM_ALLOW_MOCK_EMBEDDING_FALLBACK", raising=False)
    from prsm.data.embeddings import real_embedding_api as mod
    from unittest.mock import patch
    with patch.object(mod, "HAS_SENTENCE_TRANSFORMERS", False), \
         patch.object(mod, "HAS_OPENAI", False):
        api = mod.RealEmbeddingAPI()
        assert api.fallback_order == [], "fallback chain should be empty"
        with pytest.raises(RuntimeError, match="All embedding providers failed"):
            await api.generate_embedding("test")


@pytest.mark.asyncio
async def test_explicit_mock_request_still_works(monkeypatch):
    """Test code that needs deterministic output can still ask for mock."""
    monkeypatch.delenv("PRSM_ALLOW_MOCK_EMBEDDING_FALLBACK", raising=False)
    from prsm.data.embeddings import real_embedding_api as mod
    api = mod.RealEmbeddingAPI()
    # Explicit opt-in still works
    result = await api.generate_embedding("test", preferred_provider="mock")
    assert isinstance(result, np.ndarray)
    assert result.shape == (384,)

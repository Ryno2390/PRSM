"""
Semantic De-duplicator
======================

Second gate in the BSC pipeline.  Even if a chunk clears the KL filter
(high surprise score), it might be a rephrasing of information already on
the whiteboard.  This module catches that case.

Algorithm
---------
1. Embed the candidate chunk using a sentence-transformer model.
2. Compute cosine similarity against every embedding stored in the active
   whiteboard embedding index.
3. If max cosine similarity ≥ similarity_threshold, the chunk is semantically
   redundant and is discarded.
4. If the chunk is accepted, its embedding is added to the index so that
   future candidates are compared against it.

Implementation notes
--------------------
- Embeddings are stored in a plain numpy matrix (suitable for a single
  session's whiteboard, typically O(100s) of entries).
- For sessions with very large whiteboards (>10 000 entries) the coordinator
  can swap in an approximate-nearest-neighbour index (FAISS) by subclassing.
- Reuses ``prsm/data/embeddings/semantic_embedding_engine.py`` when
  available; falls back to HuggingFace ``sentence-transformers`` or
  ``transformers`` directly.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DedupResult:
    """Result of a semantic de-duplication check."""

    is_redundant: bool
    """True if the chunk is too similar to existing whiteboard content."""

    max_similarity: float
    """Cosine similarity to the most-similar existing whiteboard entry."""

    most_similar_index: Optional[int]
    """Index into the embedding store of the closest match, or None."""

    reason: str
    """Human-readable explanation for Nightly Synthesis / debugging."""


class SemanticDeduplicator:
    """
    Maintains an in-memory embedding index for the current session's
    whiteboard and checks new candidates for redundancy.

    Parameters
    ----------
    model_name : str
        Sentence-transformer or HuggingFace model ID for generating embeddings.
        Default: ``"all-MiniLM-L6-v2"`` (22 M params, 384-dim, very fast).
    similarity_threshold : float
        Cosine similarity ceiling.  Candidates at or above this value are
        considered redundant.  Default: 0.85.
    device : Optional[str]
        Torch device string.  None = auto-detect.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = similarity_threshold
        self._device = device

        self._encoder = None
        self._loaded = False

        # In-memory embedding store: shape (n_entries, embedding_dim)
        self._embeddings: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(self, chunk: str) -> DedupResult:
        """
        Check whether *chunk* is semantically redundant with the current
        whiteboard index.

        Parameters
        ----------
        chunk : str
            The surprise-filtered candidate chunk.

        Returns
        -------
        DedupResult
        """
        await self._ensure_loaded()

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed, chunk)

        if not self._embeddings:
            # Nothing on the whiteboard yet — always accept
            return DedupResult(
                is_redundant=False,
                max_similarity=0.0,
                most_similar_index=None,
                reason="whiteboard is empty — no comparison possible, accepting",
            )

        max_sim, best_idx = self._max_cosine(embedding)

        if max_sim >= self._threshold:
            return DedupResult(
                is_redundant=True,
                max_similarity=float(max_sim),
                most_similar_index=best_idx,
                reason=(
                    f"cosine similarity {max_sim:.3f} ≥ threshold {self._threshold:.3f} "
                    f"with whiteboard entry #{best_idx} — semantic duplicate, discarding"
                ),
            )

        return DedupResult(
            is_redundant=False,
            max_similarity=float(max_sim),
            most_similar_index=best_idx,
            reason=(
                f"max cosine similarity {max_sim:.3f} < threshold {self._threshold:.3f} "
                "— novel information, promoting"
            ),
        )

    async def add_to_index(self, chunk: str) -> int:
        """
        Embed *chunk* and add it to the whiteboard index.

        Call this only after a chunk has been accepted (i.e. ``DedupResult.is_redundant``
        is False) so the index reflects the actual whiteboard contents.

        Returns
        -------
        int
            Index of the newly added entry.
        """
        await self._ensure_loaded()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed, chunk)
        self._embeddings.append(embedding)
        return len(self._embeddings) - 1

    def clear(self) -> None:
        """Reset the index (call at the start of a new session)."""
        self._embeddings.clear()
        logger.debug("SemanticDeduplicator: index cleared")

    @property
    def index_size(self) -> int:
        """Number of embeddings currently in the index."""
        return len(self._embeddings)

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_encoder)

    def _load_encoder(self) -> None:
        """Try sentence-transformers first; fall back to transformers."""
        try:
            self._load_sentence_transformers()
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; falling back to "
                "transformers AutoModel for embeddings"
            )
            self._load_transformers_encoder()
        self._loaded = True

    def _load_sentence_transformers(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        device = self._resolve_device()
        logger.info(f"Loading embedding model {self._model_name} via sentence-transformers on {device}")
        self._encoder = SentenceTransformer(self._model_name, device=device)
        self._encoder_type = "sentence_transformers"

    def _load_transformers_encoder(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self._resolve_device()
        dtype = torch.float16 if device != "cpu" else torch.float32
        logger.info(f"Loading embedding model {self._model_name} via transformers on {device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._encoder = AutoModel.from_pretrained(
            self._model_name, torch_dtype=dtype
        ).to(device)
        self._encoder.eval()
        self._encoder_type = "transformers"

    # ------------------------------------------------------------------
    # Internal: embedding and similarity
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Return a unit-normalised embedding vector for *text*."""
        if self._encoder_type == "sentence_transformers":
            vec = self._encoder.encode(text, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)
        else:
            return self._embed_transformers(text)

    def _embed_transformers(self, text: str) -> np.ndarray:
        import torch

        device = self._resolve_device()
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = self._encoder(**inputs)
            # Mean-pool the last hidden state
            hidden = outputs.last_hidden_state  # [1, seq, dim]
            attention = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1)
            vec = pooled[0].cpu().float().numpy()

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def _max_cosine(self, query: np.ndarray) -> Tuple[float, int]:
        """
        Return (max_cosine_similarity, best_index) against the stored index.

        Embeddings are already L2-normalised so cosine similarity = dot product.
        """
        matrix = np.stack(self._embeddings, axis=0)  # [n, dim]
        sims = matrix @ query  # [n]
        best_idx = int(np.argmax(sims))
        return float(sims[best_idx]), best_idx

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

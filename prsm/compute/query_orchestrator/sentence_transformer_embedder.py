"""SentenceTransformerEmbedder — production-side Embedder backed by
the `sentence-transformers` library.

Satisfies the `Embedder` Protocol declared in
`prsm.compute.query_orchestrator.semantic_index_adapter`. The default
model name is pinned to `sentence-transformers/all-MiniLM-L6-v2` —
the same model `prsm/node/content_uploader.py` documents for the
upload-side `_SemanticIndex`. Matching the model on both sides keeps
query embeddings in the same vector space as stored shard embeddings;
otherwise top-k cosine-similarity scores would be meaningless.

Threat-model note (per `semantic_index_adapter` docstring): the
embedding model is part of the node's trusted compute base. A
swapped or poisoned local model could bias relevance scores on this
node only — cross-node poisoning is the R3 problem, covered by the
EmbeddingDHT signature path; local-model integrity is covered by
Phase 3.x.2 ModelRegistry's signed manifests.

Lazy-load rationale: the model weights are ~80MB on disk and several
hundred MB resident. node.py constructs the Embedder unconditionally
in the wiring path; deferring the load to first encode() keeps cold
start fast for nodes that never serve queries.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Satisfies the Embedder Protocol via the sentence-transformers
    library. Production wiring uses
    `sentence-transformers/all-MiniLM-L6-v2` by default — the same
    model as the content-upload path
    (`prsm/node/content_uploader.py`) so query embeddings live in
    the same vector space as stored embeddings.

    The model is loaded lazily on first encode() to keep node-start
    cost out of the critical path; constructing this object is free.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        cache_path: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Args:
            model_name: HF Hub model id. Default matches the
                content-upload path's pinned default. Override only
                when both upload and query lanes are migrated in
                lockstep — otherwise queries land in a different
                vector space than stored shards and similarity scores
                become meaningless.
            device: Torch device string (`"cuda"`, `"mps"`, `"cpu"`).
                None = let sentence-transformers auto-pick.
            cache_path: Local cache folder for the downloaded model.
                None = use the default HF cache (`~/.cache/huggingface`).
        """
        self._model_name = model_name
        self._device = device
        self._cache_path = cache_path
        self._model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Embedder Protocol
    # ------------------------------------------------------------------

    def encode(self, query: str) -> np.ndarray:
        """Encode a query string to a vector.

        Returns a 1-D float32 ndarray. The embedding is NOT
        L2-normalized — `_SemanticIndex.find_top_k` normalizes
        internally for cosine similarity.

        Raises:
            ValueError: If the query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query is empty")

        model = self._ensure_loaded()
        # convert_to_numpy=True returns numpy; normalize_embeddings is
        # left False so the caller (SemanticIndex) can normalize.
        vec = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        # Force float32 — some torch backends return float64 or
        # bfloat16 depending on device.
        return np.asarray(vec, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> SentenceTransformer:
        """Idempotent lazy-load. Subsequent calls reuse the same model
        object."""
        if self._model is None:
            kwargs = {}
            if self._device is not None:
                kwargs["device"] = self._device
            if self._cache_path is not None:
                kwargs["cache_folder"] = str(self._cache_path)
            self._model = SentenceTransformer(self._model_name, **kwargs)
        return self._model

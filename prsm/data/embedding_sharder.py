"""
Embedding-Based Semantic Sharder
=================================

Splits datasets into semantically meaningful shards using embedding
vectors. Records with similar meaning cluster into the same shard.

Embedding backends (tried in order):
1. sentence-transformers  — ``all-MiniLM-L6-v2`` (384-dim, fastest)
2. transformers AutoModel — same model via HuggingFace transformers
3. deterministic hash     — offline fallback, no ML libraries needed
"""

import hashlib
import json
import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prsm.data.shard_models import SemanticShard, SemanticShardManifest

logger = logging.getLogger(__name__)

# ── Default model ────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIM = 384  # MiniLM output dimension


# ── Embedding Provider ──────────────────────────────────────────────────

class EmbeddingProvider:
    """Loads the best available embedding backend and generates vectors.

    Tries sentence-transformers → transformers → hash fallback.
    Thread-safe after initial load.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self._model_name = model_name
        self._encoder: Any = None
        self._tokenizer: Any = None
        self._backend: str = "none"
        self._dim: int = 0
        self._loaded: bool = False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dimension(self) -> int:
        if not self._loaded:
            self._load()
        return self._dim

    def _load(self) -> None:
        """Try backends in priority order."""
        if self._loaded:
            return

        # 1. sentence-transformers (best: fast, normalized, correct)
        try:
            self._load_sentence_transformers()
            return
        except (ImportError, Exception) as exc:
            logger.debug("sentence-transformers unavailable: %s", exc)

        # 2. transformers AutoModel (slower, needs mean-pooling)
        try:
            self._load_transformers()
            return
        except (ImportError, Exception) as exc:
            logger.debug("transformers unavailable: %s", exc)

        # 3. Hash fallback (deterministic, works everywhere)
        logger.info(
            "No ML embedding library available — using deterministic hash "
            "projection. Install sentence-transformers for real semantic sharding: "
            "pip install prsm-network[ml]"
        )
        self._backend = "hash"
        self._dim = DEFAULT_EMBEDDING_DIM
        self._loaded = True

    def _load_sentence_transformers(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        device = self._resolve_device()
        logger.info(
            "Loading embedding model %s via sentence-transformers on %s",
            self._model_name, device,
        )
        self._encoder = SentenceTransformer(self._model_name, device=device)
        self._backend = "sentence_transformers"
        self._dim = self._encoder.get_sentence_embedding_dimension()
        self._loaded = True

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

        device = self._resolve_device()
        dtype = torch.float16 if device != "cpu" else torch.float32
        logger.info(
            "Loading embedding model %s via transformers on %s",
            self._model_name, device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._encoder = AutoModel.from_pretrained(
            self._model_name, torch_dtype=dtype,
        ).to(device)
        self._encoder.eval()
        self._backend = "transformers"
        # Infer dim from model config
        self._dim = self._encoder.config.hidden_size
        self._loaded = True

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ── Public API ───────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalised embedding vector for *text*."""
        if not self._loaded:
            self._load()

        if self._backend == "sentence_transformers":
            vec = self._encoder.encode(text, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)

        if self._backend == "transformers":
            return self._embed_transformers(text)

        # Hash fallback
        return self._hash_embed(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently. Returns (N, dim) array."""
        if not self._loaded:
            self._load()

        if self._backend == "sentence_transformers":
            vecs = self._encoder.encode(texts, normalize_embeddings=True, batch_size=64)
            return np.array(vecs, dtype=np.float32)

        # No native batching for transformers/hash — loop
        return np.array([self.embed(t) for t in texts], dtype=np.float32)

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
            hidden = outputs.last_hidden_state
            attention = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1)
            vec = pooled[0].cpu().float().numpy()

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic hash projection fallback."""
        h = hashlib.sha256(text.encode()).hexdigest()
        while len(h) < self._dim * 2:
            h += hashlib.sha256(h.encode()).hexdigest()
        values = np.array(
            [(int(h[i * 2 : i * 2 + 2], 16) - 128) / 128.0 for i in range(self._dim)],
            dtype=np.float32,
        )
        norm = np.linalg.norm(values)
        if norm > 0:
            values /= norm
        return values


# ── K-Means Clustering ──────────────────────────────────────────────────

def _kmeans(vectors: np.ndarray, k: int, max_iter: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means on L2-normalized vectors.

    Args:
        vectors: (N, dim) array of embeddings.
        k: number of clusters.
        max_iter: iteration cap.

    Returns:
        (labels, centroids) — labels is (N,) int array,
        centroids is (k, dim) float array.
    """
    n = vectors.shape[0]
    k = min(k, n)

    # Initialize centroids with k-means++ seeding
    centroids = _kmeanspp_init(vectors, k)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # Assign each vector to nearest centroid (cosine → dot product on normalized vecs)
        sims = vectors @ centroids.T  # (N, k)
        new_labels = sims.argmax(axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Recompute centroids
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroid = vectors[mask].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                centroids[j] = centroid

    return labels, centroids


def _kmeanspp_init(vectors: np.ndarray, k: int) -> np.ndarray:
    """K-means++ initialization for better starting centroids."""
    n, dim = vectors.shape
    rng = np.random.default_rng(42)  # Deterministic for reproducibility

    centroids = np.empty((k, dim), dtype=np.float32)
    # Pick first centroid randomly
    idx = rng.integers(n)
    centroids[0] = vectors[idx]

    for i in range(1, k):
        # Distance to nearest existing centroid (1 - cosine sim for normalized vecs)
        sims = vectors @ centroids[:i].T  # (N, i)
        min_sim = sims.max(axis=1)  # closest centroid similarity
        dists = 1.0 - min_sim  # convert to distance
        dists = np.maximum(dists, 0.0)

        # Weighted random selection
        total = dists.sum()
        if total > 0:
            probs = dists / total
        else:
            probs = np.ones(n) / n
        idx = rng.choice(n, p=probs)
        centroids[i] = vectors[idx]

    return centroids


# ── Legacy helpers (kept for backward compat) ───────────────────────────

# Singleton provider for lightweight usage
_default_provider: Optional[EmbeddingProvider] = None


def _simple_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> List[float]:
    """Generate an embedding for *text* using the best available backend.

    This is the public helper — callers get real embeddings when ML
    libraries are installed, hash fallback otherwise.
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = EmbeddingProvider()
    vec = _default_provider.embed(text)
    # If caller requests a different dim than the provider's native dim,
    # pad or truncate (rare — only if dim explicitly overridden)
    if dim != len(vec):
        if dim < len(vec):
            vec = vec[:dim]
        else:
            vec = np.pad(vec, (0, dim - len(vec)))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
    return vec.tolist()


# ── Sharder ──────────────────────────────────────────────────────────────

class EmbeddingSharder:
    """Splits datasets into semantic shards using real embeddings."""

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.provider = EmbeddingProvider(model_name=model_name)
        self.embedding_dim = embedding_dim

    def shard_records(
        self,
        records: List[Dict[str, Any]],
        dataset_id: str,
        n_shards: int = 4,
        text_field: str = "",
    ) -> SemanticShardManifest:
        """Shard a list of record dicts into semantic clusters.

        Args:
            records: List of record dictionaries.
            dataset_id: Identifier for the dataset.
            n_shards: Number of shards to create.
            text_field: Field to use for semantic clustering.

        Returns:
            SemanticShardManifest with shards and real centroid embeddings.
        """
        if not records:
            return SemanticShardManifest(
                dataset_id=dataset_id,
                total_records=0,
                total_size_bytes=0,
            )

        n_shards = min(n_shards, len(records))

        # 1. Extract text from each record
        texts = []
        for r in records:
            if text_field:
                texts.append(str(r.get(text_field, json.dumps(r, sort_keys=True))))
            else:
                texts.append(json.dumps(r, sort_keys=True))

        # 2. Embed all records
        embeddings = self.provider.embed_batch(texts)

        # 3. Cluster with k-means
        labels, centroids = _kmeans(embeddings, n_shards)

        # 4. Build shard objects
        shards = []
        total_size = 0

        for i in range(centroids.shape[0]):
            mask = labels == i
            cluster_indices = np.where(mask)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_records = [records[idx] for idx in cluster_indices]
            shard_data = json.dumps(cluster_records).encode()

            # Extract keywords from field values
            keywords = set()
            for r in cluster_records[:20]:
                for k, v in r.items():
                    if isinstance(v, str) and len(v) < 50:
                        keywords.add(v)
            keywords_list = sorted(keywords)[:10]

            shard = SemanticShard(
                shard_id=f"{dataset_id}-shard-{i:04d}",
                parent_dataset=dataset_id,
                cid=f"Qm{dataset_id}-{i:04d}",  # Placeholder until IPFS upload
                centroid=centroids[i].tolist(),
                record_count=len(cluster_records),
                size_bytes=len(shard_data),
                keywords=keywords_list,
            )
            shards.append(shard)
            total_size += len(shard_data)

        return SemanticShardManifest(
            dataset_id=dataset_id,
            total_records=len(records),
            total_size_bytes=total_size,
            shards=shards,
            embedding_dimension=self.provider.dimension,
        )

    def shard_file(
        self,
        data: bytes,
        dataset_id: str,
        n_shards: int = 4,
        text_field: str = "",
    ) -> SemanticShardManifest:
        """Shard raw file bytes (CSV/JSON) into semantic clusters."""
        from prsm.compute.agents.data_processor import DataProcessor
        processor = DataProcessor()
        records = processor._parse_data(data)
        return self.shard_records(records, dataset_id, n_shards, text_field)

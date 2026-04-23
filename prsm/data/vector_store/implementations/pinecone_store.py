"""
Pinecone vector store implementation for PRSM

Suitable for: Prototyping and early deployments where a fully managed
cloud vector database is preferred over self-hosted infrastructure.
Note: Cloud-only — no self-hosted option available.

Install: pip install pinecone-client
Setup: Create an index at https://app.pinecone.io
"""

# Defer annotation evaluation so module loads when the optional dep is missing —
# return-type annotations referencing the lazy-imported lib would otherwise fail
# at class-def time and mask the intended ImportError raised from __init__.
from __future__ import annotations

import json
import logging
import uuid
from typing import Dict, List, Optional, Any
import numpy as np
import os

try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

from ..base import (
    PRSMVectorStore, ContentMatch, SearchFilters,
    VectorStoreConfig, ContentType
)

logger = logging.getLogger(__name__)


class PineconeVectorStore(PRSMVectorStore):
    """
    Pinecone (cloud) implementation of PRSMVectorStore.

    Uses Pinecone's serverless index. Vector IDs are deterministic UUIDs
    derived from content_cid so upsert is idempotent.
    """

    def __init__(self, config: VectorStoreConfig):
        if not HAS_PINECONE:
            raise ImportError(
                "Pinecone support requires pinecone-client. "
                "Install with: pip install 'prsm[vectorstore]'"
            )
        super().__init__(config)
        self._pc = None
        self._index = None
        self._index_name = config.collection_name

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        try:
            api_key = self.config.password or os.getenv("PINECONE_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "PINECONE_API_KEY is required. "
                    "Set it via VectorStoreConfig.password or PINECONE_API_KEY env var."
                )
            self._pc = Pinecone(api_key=api_key)
            self._index = self._pc.Index(self._index_name)
            # Lightweight stats call to verify connectivity
            self._index.describe_index_stats()
            self.is_connected = True
            logger.info(f"✅ Connected to Pinecone index: {self._index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> bool:
        self._index = None
        self._pc = None
        self.is_connected = False
        return True

    # ── Collection management ─────────────────────────────────────────────────

    async def create_collection(self, collection_name: str,
                                vector_dimension: int,
                                metadata_schema: Dict[str, Any]) -> bool:
        try:
            existing = [i.name for i in self._pc.list_indexes()]
            if collection_name not in existing:
                environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
                cloud, region = (environment.rsplit("-", 1) + ["aws"])[:2]
                self._pc.create_index(
                    name=collection_name,
                    dimension=vector_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region)
                )
                logger.info(f"Created Pinecone index: {collection_name}")
            self._index = self._pc.Index(collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to create Pinecone index {collection_name}: {e}")
            return False

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def store_content_with_embeddings(self, content_cid: str,
                                            embeddings: np.ndarray,
                                            metadata: Dict[str, Any]) -> str:
        try:
            vector_id = str(uuid.uuid5(uuid.NAMESPACE_URL, content_cid))
            # Pinecone metadata values must be str / float / int / bool / list[str]
            pinecone_meta = {
                "content_cid": content_cid,
                "creator_id": metadata.get("creator_id", "") or "",
                "royalty_rate": float(metadata.get("royalty_rate", 0.08)),
                "content_type": metadata.get("content_type", ContentType.TEXT.value),
                "quality_score": float(metadata.get("quality_score") or 0.0),
                "metadata_json": json.dumps(metadata),
            }
            self._index.upsert(vectors=[(vector_id, embeddings.tolist(), pinecone_meta)])
            self._update_performance_metrics("storage", 0.0, True)
            return vector_id
        except Exception as e:
            self._update_performance_metrics("storage", 0.0, False)
            logger.error(f"Failed to upsert to Pinecone: {e}")
            raise

    async def search_similar_content(self, query_embedding: np.ndarray,
                                     filters: Optional[SearchFilters] = None,
                                     top_k: int = 10) -> List[ContentMatch]:
        try:
            pinecone_filter = self._build_pinecone_filter(filters)
            response = self._index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                filter=pinecone_filter,
                include_metadata=True
            )
            matches = []
            for match in response.matches:
                meta_raw = match.metadata or {}
                meta = json.loads(meta_raw.get("metadata_json", "{}"))
                matches.append(ContentMatch(
                    content_cid=meta_raw.get("content_cid", ""),
                    similarity_score=min(1.0, max(0.0, float(match.score))),
                    metadata=meta,
                    creator_id=meta_raw.get("creator_id") or None,
                    royalty_rate=float(meta_raw.get("royalty_rate", 0.08)),
                    content_type=ContentType(
                        meta_raw.get("content_type", ContentType.TEXT.value)
                    ),
                    quality_score=float(meta_raw["quality_score"])
                    if meta_raw.get("quality_score") else None,
                ))
            self._update_performance_metrics("query", 0.0, True)
            return matches
        except Exception as e:
            self._update_performance_metrics("query", 0.0, False)
            logger.error(f"Pinecone search failed: {e}")
            raise

    def _build_pinecone_filter(self, filters: Optional[SearchFilters]) -> Optional[Dict]:
        if not filters:
            return None
        f: Dict[str, Any] = {}
        if filters.content_types:
            f["content_type"] = {"$in": [ct.value for ct in filters.content_types]}
        if filters.creator_ids:
            f["creator_id"] = {"$in": filters.creator_ids}
        if filters.min_quality_score is not None:
            f["quality_score"] = {"$gte": filters.min_quality_score}
        return f or None

    async def update_content_metadata(self, content_cid: str,
                                      metadata_updates: Dict[str, Any]) -> bool:
        try:
            vector_id = str(uuid.uuid5(uuid.NAMESPACE_URL, content_cid))
            self._index.update(id=vector_id, set_metadata=metadata_updates)
            return True
        except Exception as e:
            logger.error(f"Failed to update Pinecone metadata: {e}")
            return False

    async def delete_content(self, content_cid: str) -> bool:
        try:
            vector_id = str(uuid.uuid5(uuid.NAMESPACE_URL, content_cid))
            self._index.delete(ids=[vector_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Pinecone: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        try:
            stats = self._index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count or 0,
                "collection": self._index_name,
                "backend": "pinecone",
                "dimension": stats.dimension,
            }
        except Exception as e:
            return {"error": str(e), "backend": "pinecone"}

"""
Qdrant vector store implementation for PRSM

Suitable for: Production deployments needing a modern standalone vector DB
with rich filtering, self-hosting capability, and a clean REST/gRPC API.

Install: pip install qdrant-client
Self-hosted: docker run -p 6333:6333 qdrant/qdrant
Cloud: https://cloud.qdrant.io
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

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchAny, MatchValue, Range
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

from ..base import (
    PRSMVectorStore, ContentMatch, SearchFilters,
    VectorStoreConfig, VectorStoreType, ContentType
)

logger = logging.getLogger(__name__)


class QdrantVectorStore(PRSMVectorStore):
    """
    Qdrant implementation of PRSMVectorStore.
 
    Each vector point carries the content_cid and full metadata as payload.
    Cosine similarity used throughout for consistency with pgvector.
    """

    def __init__(self, config: VectorStoreConfig):
        if not HAS_QDRANT:
            raise ImportError(
                "Qdrant support requires qdrant-client. "
                "Install with: pip install 'prsm[vectorstore]'"
            )
        super().__init__(config)
        self._client: Optional[QdrantClient] = None

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        try:
            import os
            # Qdrant Cloud uses url + api_key; self-hosted uses host + port
            url = os.getenv("QDRANT_URL", "")
            api_key = self.config.password or os.getenv("QDRANT_API_KEY", "")

            if url:
                self._client = QdrantClient(url=url, api_key=api_key or None)
            else:
                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    api_key=api_key or None,
                    timeout=self.config.connection_timeout
                )

            # Verify connectivity with a lightweight collections list call
            self._client.get_collections()
            self.is_connected = True
            logger.info(f"✅ Connected to Qdrant at {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> bool:
        try:
            if self._client:
                self._client.close()
                self._client = None
            self.is_connected = False
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Qdrant: {e}")
            return False

    # ── Collection management ─────────────────────────────────────────────────

    async def create_collection(self, collection_name: str,
                                vector_dimension: int,
                                metadata_schema: Dict[str, Any]) -> bool:
        try:
            existing = [c.name for c in self._client.get_collections().collections]
            if collection_name not in existing:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_dimension,
                        distance=Distance.COSINE
                    )
                )
                # Create payload indexes for filtering
                for field in ("content_type", "creator_id"):
                    self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name="quality_score",
                    field_schema=models.PayloadSchemaType.FLOAT
                )
            logger.info(f"Qdrant collection ready: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection {collection_name}: {e}")
            return False

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def store_content_with_embeddings(self, content_cid: str,
                                            embeddings: np.ndarray,
                                            metadata: Dict[str, Any]) -> str:
        try:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, content_cid))
            payload = {
                "content_cid": content_cid,
                "creator_id": metadata.get("creator_id", ""),
                "royalty_rate": float(metadata.get("royalty_rate", 0.08)),
                "content_type": metadata.get("content_type", ContentType.TEXT.value),
                "quality_score": float(metadata.get("quality_score") or 0.0),
                "metadata": json.dumps(metadata),
            }
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=[PointStruct(
                    id=point_id,
                    vector=embeddings.tolist(),
                    payload=payload
                )]
            )
            self._update_performance_metrics("storage", 0.0, True)
            logger.debug(f"Stored {content_cid} → Qdrant point {point_id}")
            return point_id
        except Exception as e:
            self._update_performance_metrics("storage", 0.0, False)
            logger.error(f"Failed to store to Qdrant: {e}")
            raise

    async def search_similar_content(self, query_embedding: np.ndarray,
                                     filters: Optional[SearchFilters] = None,
                                     top_k: int = 10) -> List[ContentMatch]:
        try:
            qdrant_filter = self._build_qdrant_filter(filters)
            hits = self._client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                score_threshold=0.0
            )
            matches = []
            for hit in hits:
                p = hit.payload or {}
                meta = json.loads(p.get("metadata", "{}"))
                matches.append(ContentMatch(
                    content_cid=p.get("content_cid", ""),
                    similarity_score=min(1.0, max(0.0, float(hit.score))),
                    metadata=meta,
                    creator_id=p.get("creator_id") or None,
                    royalty_rate=float(p.get("royalty_rate", 0.08)),
                    content_type=ContentType(
                        p.get("content_type", ContentType.TEXT.value)
                    ),
                    quality_score=float(p["quality_score"]) if p.get("quality_score") else None,
                ))
            self._update_performance_metrics("query", 0.0, True)
            return matches
        except Exception as e:
            self._update_performance_metrics("query", 0.0, False)
            logger.error(f"Qdrant search failed: {e}")
            raise

    def _build_qdrant_filter(self, filters: Optional[SearchFilters]) -> Optional[Filter]:
        if not filters:
            return None
        conditions = []
        if filters.content_types:
            conditions.append(FieldCondition(
                key="content_type",
                match=MatchAny(any=[ct.value for ct in filters.content_types])
            ))
        if filters.creator_ids:
            conditions.append(FieldCondition(
                key="creator_id",
                match=MatchAny(any=filters.creator_ids)
            ))
        if filters.min_quality_score is not None:
            conditions.append(FieldCondition(
                key="quality_score",
                range=Range(gte=filters.min_quality_score)
            ))
        if not conditions:
            return None
        return Filter(must=conditions)

    async def update_content_metadata(self, content_cid: str,
                                      metadata_updates: Dict[str, Any]) -> bool:
        try:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, content_cid))
            self._client.set_payload(
                collection_name=self.config.collection_name,
                payload=metadata_updates,
                points=[point_id]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update Qdrant metadata: {e}")
            return False

    async def delete_content(self, content_cid: str) -> bool:
        try:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, content_cid))
            self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.PointIdsList(points=[point_id])
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Qdrant: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        try:
            info = self._client.get_collection(self.config.collection_name)
            return {
                "total_vectors": info.vectors_count or 0,
                "indexed_vectors": info.indexed_vectors_count or 0,
                "collection": self.config.collection_name,
                "backend": "qdrant",
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e), "backend": "qdrant"}

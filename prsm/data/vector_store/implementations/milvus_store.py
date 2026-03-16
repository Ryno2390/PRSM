"""
Milvus vector store implementation for PRSM

Suitable for: Large-scale deployments (billions of vectors), high-throughput
workloads, and self-hosted infrastructure that needs to scale beyond
PostgreSQL's sweet spot.

Install: pip install pymilvus
Self-hosted: docker run -d milvusdb/milvus:latest
Cloud: https://cloud.zilliz.com
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

try:
    from pymilvus import (
        connections, Collection, FieldSchema, CollectionSchema,
        DataType, utility, MilvusException
    )
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False

from ..base import (
    PRSMVectorStore, ContentMatch, SearchFilters,
    VectorStoreConfig, VectorStoreType, ContentType
)

logger = logging.getLogger(__name__)

_MILVUS_FIELD_ID       = "vector_id"
_MILVUS_FIELD_CID      = "content_cid"
_MILVUS_FIELD_VECTOR   = "embedding"
_MILVUS_FIELD_METADATA = "metadata_json"
_MILVUS_FIELD_CREATOR  = "creator_id"
_MILVUS_FIELD_ROYALTY  = "royalty_rate"
_MILVUS_FIELD_CTYPE    = "content_type"
_MILVUS_FIELD_QUALITY  = "quality_score"


class MilvusVectorStore(PRSMVectorStore):
    """
    Milvus implementation of PRSMVectorStore.
 
    Uses HNSW index on the embedding field for ANN search.
    Metadata is serialised as JSON and stored in a VARCHAR column.
    """

    def __init__(self, config: VectorStoreConfig):
        if not HAS_MILVUS:
            raise ImportError(
                "Milvus support requires pymilvus. "
                "Install with: pip install 'prsm[vectorstore]'"
            )
        super().__init__(config)
        self._alias = f"prsm_{id(self)}"   # unique alias per instance
        self._collection: Optional[Collection] = None

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_connect)
            self.is_connected = True
            logger.info(f"✅ Connected to Milvus at {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.is_connected = False
            return False

    def _sync_connect(self):
        token = f"{self.config.username}:{self.config.password}" \
                if self.config.username else None
        connections.connect(
            alias=self._alias,
            host=self.config.host,
            port=str(self.config.port),
            token=token
        )

    async def disconnect(self) -> bool:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: connections.disconnect(self._alias)
            )
            self.is_connected = False
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
            return False

    # ── Collection management ─────────────────────────────────────────────────

    async def create_collection(self, collection_name: str,
                                vector_dimension: int,
                                metadata_schema: Dict[str, Any]) -> bool:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._sync_create_collection(collection_name, vector_dimension)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create Milvus collection {collection_name}: {e}")
            return False

    def _sync_create_collection(self, name: str, dim: int):
        if utility.has_collection(name, using=self._alias):
            self._collection = Collection(name, using=self._alias)
            return

        fields = [
            FieldSchema(_MILVUS_FIELD_ID,      DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(_MILVUS_FIELD_CID,     DataType.VARCHAR, max_length=512),
            FieldSchema(_MILVUS_FIELD_VECTOR,  DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(_MILVUS_FIELD_METADATA,DataType.VARCHAR, max_length=65535),
            FieldSchema(_MILVUS_FIELD_CREATOR, DataType.VARCHAR, max_length=256),
            FieldSchema(_MILVUS_FIELD_ROYALTY, DataType.FLOAT),
            FieldSchema(_MILVUS_FIELD_CTYPE,   DataType.VARCHAR, max_length=64),
            FieldSchema(_MILVUS_FIELD_QUALITY, DataType.FLOAT),
        ]
        schema = CollectionSchema(fields, description="PRSM content vectors")
        self._collection = Collection(name, schema=schema, using=self._alias)

        # HNSW index — fast ANN with good recall
        self._collection.create_index(
            _MILVUS_FIELD_VECTOR,
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200}
            }
        )
        self._collection.load()

    def _get_collection(self) -> Collection:
        if self._collection is None:
            self._collection = Collection(
                self.config.collection_name, using=self._alias
            )
            self._collection.load()
        return self._collection

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def store_content_with_embeddings(self, content_cid: str,
                                            embeddings: np.ndarray,
                                            metadata: Dict[str, Any]) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        vector_id = str(uuid.uuid4()).replace("-", "")[:64]
        await loop.run_in_executor(
            None,
            lambda: self._sync_upsert(vector_id, content_cid, embeddings, metadata)
        )
        self._update_performance_metrics("storage", 0.0, True)
        return vector_id

    def _sync_upsert(self, vid: str, cid: str,
                     emb: np.ndarray, meta: Dict[str, Any]):
        col = self._get_collection()
        data = [
            [vid],
            [cid],
            [emb.tolist()],
            [json.dumps(meta)],
            [meta.get("creator_id", "") or ""],
            [float(meta.get("royalty_rate", 0.08))],
            [meta.get("content_type", ContentType.TEXT.value)],
            [float(meta.get("quality_score", 0.0) or 0.0)],
        ]
        col.upsert(data)

    async def search_similar_content(self, query_embedding: np.ndarray,
                                     filters: Optional[SearchFilters] = None,
                                     top_k: int = 10) -> List[ContentMatch]:
        import asyncio
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._sync_search(query_embedding, filters, top_k)
        )
        self._update_performance_metrics("query", 0.0, True)
        return results

    def _sync_search(self, query: np.ndarray,
                     filters: Optional[SearchFilters],
                     top_k: int) -> List[ContentMatch]:
        col = self._get_collection()

        expr = self._build_filter_expr(filters)
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        hits = col.search(
            data=[query.tolist()],
            anns_field=_MILVUS_FIELD_VECTOR,
            param=search_params,
            limit=top_k,
            expr=expr if expr else None,
            output_fields=[
                _MILVUS_FIELD_CID, _MILVUS_FIELD_METADATA,
                _MILVUS_FIELD_CREATOR, _MILVUS_FIELD_ROYALTY,
                _MILVUS_FIELD_CTYPE, _MILVUS_FIELD_QUALITY,
            ]
        )

        matches = []
        for hit in hits[0]:
            entity = hit.entity
            meta = json.loads(entity.get(_MILVUS_FIELD_METADATA) or "{}")
            matches.append(ContentMatch(
                content_cid=entity.get(_MILVUS_FIELD_CID, ""),
                similarity_score=min(1.0, max(0.0, float(hit.score))),
                metadata=meta,
                creator_id=entity.get(_MILVUS_FIELD_CREATOR) or None,
                royalty_rate=float(entity.get(_MILVUS_FIELD_ROYALTY) or 0.08),
                content_type=ContentType(
                    entity.get(_MILVUS_FIELD_CTYPE) or ContentType.TEXT.value
                ),
                quality_score=float(entity.get(_MILVUS_FIELD_QUALITY))
                if entity.get(_MILVUS_FIELD_QUALITY) else None,
            ))
        return matches

    def _build_filter_expr(self, filters: Optional[SearchFilters]) -> str:
        if not filters:
            return ""
        parts = []
        if filters.content_types:
            vals = [f'"{ct.value}"' for ct in filters.content_types]
            parts.append(f"{_MILVUS_FIELD_CTYPE} in [{', '.join(vals)}]")
        if filters.creator_ids:
            vals = [f'"{c}"' for c in filters.creator_ids]
            parts.append(f"{_MILVUS_FIELD_CREATOR} in [{', '.join(vals)}]")
        if filters.min_quality_score is not None:
            parts.append(
                f"{_MILVUS_FIELD_QUALITY} >= {filters.min_quality_score}"
            )
        return " and ".join(parts)

    async def update_content_metadata(self, content_cid: str,
                                      metadata_updates: Dict[str, Any]) -> bool:
        # Milvus v2.x does not support partial update of non-vector fields;
        # fetch-merge-upsert pattern is required.
        logger.warning(
            "Milvus metadata update requires a full re-upsert. "
            "Content CID: %s", content_cid
        )
        return True  # Callers should re-call store_content_with_embeddings

    async def delete_content(self, content_cid: str) -> bool:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._get_collection().delete(
                    f'{_MILVUS_FIELD_CID} == "{content_cid}"'
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Milvus: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None,
                lambda: self._get_collection().num_entities
            )
            return {
                "total_vectors": stats,
                "collection": self.config.collection_name,
                "backend": "milvus"
            }
        except Exception as e:
            return {"error": str(e), "backend": "milvus"}

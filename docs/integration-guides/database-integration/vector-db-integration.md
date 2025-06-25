# Vector Database Integration Guide

Integrate PRSM with vector databases for advanced AI embeddings, semantic search, and retrieval-augmented generation (RAG).

## 🎯 Overview

This guide covers integrating PRSM with popular vector databases including Pinecone, Weaviate, Chroma, and Qdrant for efficient storage and retrieval of high-dimensional vectors.

## 📋 Prerequisites

- Vector database service configured
- PRSM instance with embedding capabilities
- Basic knowledge of embeddings and vector search
- Python development environment

## 🚀 Pinecone Integration

### 1. Pinecone Setup

```bash
# Install Pinecone SDK
pip install pinecone-client

# Set up environment variables
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_ENVIRONMENT="your-pinecone-environment"
```

### 2. Pinecone Configuration

```python
# prsm/integrations/vector_db/pinecone_client.py
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import pinecone
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from prsm.models.embeddings import EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)

class PineconeVectorDB:
    """Pinecone vector database integration for PRSM."""
    
    def __init__(
        self,
        api_key: str = None,
        environment: str = None,
        index_name: str = "prsm-embeddings"
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = index_name
        self.index = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        if not self.api_key or not self.environment:
            raise ValueError("Pinecone API key and environment must be provided")
    
    async def initialize(self, dimension: int = 1536, metric: str = "cosine"):
        """Initialize Pinecone connection and create index if needed."""
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    metadata_config={
                        "indexed": ["user_id", "document_type", "timestamp"]
                    }
                )
            
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = ""
    ) -> bool:
        """
        Upsert vectors to Pinecone.
        
        Args:
            vectors: List of vector dictionaries with id, values, and metadata
            namespace: Optional namespace for organizing vectors
        """
        try:
            # Run upsert in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.index.upsert(vectors=vectors, namespace=namespace)
            )
            logger.info(f"Successfully upserted {len(vectors)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False
    
    async def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Dict[str, Any] = None,
        namespace: str = "",
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query similar vectors from Pinecone.
        
        Args:
            query_vector: Query vector for similarity search
            top_k: Number of top results to return
            filter_metadata: Metadata filters
            namespace: Namespace to search in
            include_metadata: Whether to include metadata in results
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=filter_metadata,
                    namespace=namespace,
                    include_metadata=include_metadata,
                    include_values=False
                )
            )
            
            return response.get("matches", [])
            
        except Exception as e:
            logger.error(f"Failed to query vectors: {e}")
            return []
    
    async def delete_vectors(
        self,
        ids: List[str] = None,
        filter_metadata: Dict[str, Any] = None,
        namespace: str = ""
    ) -> bool:
        """Delete vectors from Pinecone."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.index.delete(
                    ids=ids,
                    filter=filter_metadata,
                    namespace=namespace
                )
            )
            logger.info(f"Successfully deleted vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    async def get_index_stats(self, namespace: str = "") -> Dict[str, Any]:
        """Get index statistics."""
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                self.executor,
                lambda: self.index.describe_index_stats()
            )
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    async def close(self):
        """Close connections and cleanup."""
        if self.executor:
            self.executor.shutdown(wait=True)
```

## 🔧 Weaviate Integration

### 1. Weaviate Setup

```bash
# Install Weaviate SDK
pip install weaviate-client

# Start Weaviate with Docker
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e DEFAULT_VECTORIZER_MODULE='none' \
  -e CLUSTER_HOSTNAME='node1' \
  semitechnologies/weaviate:latest
```

### 2. Weaviate Configuration

```python
# prsm/integrations/vector_db/weaviate_client.py
import asyncio
import logging
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.util import generate_uuid5
import numpy as np

logger = logging.getLogger(__name__)

class WeaviateVectorDB:
    """Weaviate vector database integration for PRSM."""
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        class_name: str = "PRSMDocument"
    ):
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self.client = None
    
    async def initialize(self):
        """Initialize Weaviate connection and schema."""
        try:
            # Initialize client
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)
                self.client = weaviate.Client(
                    url=self.url,
                    auth_client_secret=auth_config
                )
            else:
                self.client = weaviate.Client(url=self.url)
            
            # Create schema if it doesn't exist
            await self._create_schema()
            logger.info("Weaviate client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise
    
    async def _create_schema(self):
        """Create Weaviate schema for PRSM documents."""
        schema = {
            "class": self.class_name,
            "description": "PRSM document embeddings",
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Document content"
                },
                {
                    "name": "userId",
                    "dataType": ["string"],
                    "description": "User identifier"
                },
                {
                    "name": "documentType",
                    "dataType": ["string"],
                    "description": "Type of document"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "Creation timestamp"
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata"
                }
            ]
        }
        
        # Check if class exists
        try:
            existing_schema = self.client.schema.get()
            class_exists = any(
                cls["class"] == self.class_name 
                for cls in existing_schema.get("classes", [])
            )
            
            if not class_exists:
                self.client.schema.create_class(schema)
                logger.info(f"Created Weaviate class: {self.class_name}")
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        vectors: List[List[float]]
    ) -> List[str]:
        """Add documents with vectors to Weaviate."""
        try:
            added_ids = []
            
            for doc, vector in zip(documents, vectors):
                # Generate consistent UUID for the document
                doc_id = generate_uuid5(doc.get("content", ""))
                
                # Prepare data object
                data_object = {
                    "content": doc.get("content", ""),
                    "userId": doc.get("user_id", ""),
                    "documentType": doc.get("document_type", ""),
                    "timestamp": doc.get("timestamp", ""),
                    "metadata": doc.get("metadata", {})
                }
                
                # Add object with vector
                self.client.data_object.create(
                    data_object=data_object,
                    class_name=self.class_name,
                    uuid=doc_id,
                    vector=vector
                )
                
                added_ids.append(doc_id)
            
            logger.info(f"Added {len(added_ids)} documents to Weaviate")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return []
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        where_filter: Dict[str, Any] = None,
        certainty: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            query = (
                self.client.query
                .get(self.class_name, ["content", "userId", "documentType", "timestamp", "metadata"])
                .with_near_vector({
                    "vector": query_vector,
                    "certainty": certainty
                })
                .with_limit(limit)
            )
            
            if where_filter:
                query = query.with_where(where_filter)
            
            result = query.do()
            
            return result.get("data", {}).get("Get", {}).get(self.class_name, [])
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []
    
    async def delete_documents(
        self,
        where_filter: Dict[str, Any]
    ) -> bool:
        """Delete documents matching filter."""
        try:
            self.client.batch.delete_objects(
                class_name=self.class_name,
                where=where_filter
            )
            logger.info("Documents deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def get_object_count(self) -> int:
        """Get total number of objects in the class."""
        try:
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            return result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
            
        except Exception as e:
            logger.error(f"Failed to get object count: {e}")
            return 0
```

## 🔍 Chroma Integration

### 1. Chroma Setup

```bash
# Install Chroma
pip install chromadb

# Start Chroma server (optional)
docker run -d --name chroma -p 8000:8000 ghcr.io/chroma-core/chroma:latest
```

### 2. Chroma Configuration

```python
# prsm/integrations/vector_db/chroma_client.py
import asyncio
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import uuid

logger = logging.getLogger(__name__)

class ChromaVectorDB:
    """Chroma vector database integration for PRSM."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "prsm_embeddings"
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self.collection = None
    
    async def initialize(self):
        """Initialize Chroma client and collection."""
        try:
            # Initialize client
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            raise
    
    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ) -> List[str]:
        """Add documents to Chroma collection."""
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            )
            
            logger.info(f"Added {len(documents)} documents to Chroma")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return []
    
    async def query_similar(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None
    ) -> Dict[str, List]:
        """Query similar documents from Chroma."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=["documents", "metadatas", "distances"]
                )
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
            return {}
    
    async def update_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]] = None,
        metadatas: List[Dict[str, Any]] = None,
        documents: List[str] = None
    ) -> bool:
        """Update existing documents."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.update(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            )
            
            logger.info(f"Updated {len(ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            return False
    
    async def delete_documents(
        self,
        ids: List[str] = None,
        where: Dict[str, Any] = None
    ) -> bool:
        """Delete documents from collection."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.delete(
                    ids=ids,
                    where=where
                )
            )
            
            logger.info("Documents deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                None,
                lambda: self.collection.count()
            )
            
            return {
                "name": self.collection_name,
                "count": count,
                "metadata": self.collection.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
```

## 🔧 PRSM Vector Database Manager

### Unified Vector Database Interface

```python
# prsm/core/vector_store.py
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod
from prsm.integrations.vector_db.pinecone_client import PineconeVectorDB
from prsm.integrations.vector_db.weaviate_client import WeaviateVectorDB
from prsm.integrations.vector_db.chroma_client import ChromaVectorDB
from prsm.models.embeddings import EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)

class VectorDBType(Enum):
    """Supported vector database types."""
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"
    QDRANT = "qdrant"

class VectorDBInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the vector database connection."""
        pass
    
    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[Dict[str, Any]]
    ) -> List[str]:
        """Add vectors to the database."""
        pass
    
    @abstractmethod
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_vectors(
        self,
        filters: Dict[str, Any]
    ) -> bool:
        """Delete vectors matching filters."""
        pass

class VectorStoreManager:
    """Unified vector store manager for PRSM."""
    
    def __init__(
        self,
        db_type: VectorDBType,
        config: Dict[str, Any]
    ):
        self.db_type = db_type
        self.config = config
        self.vector_db = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the appropriate vector database client."""
        if self.db_type == VectorDBType.PINECONE:
            self.vector_db = PineconeVectorDB(**self.config)
        elif self.db_type == VectorDBType.WEAVIATE:
            self.vector_db = WeaviateVectorDB(**self.config)
        elif self.db_type == VectorDBType.CHROMA:
            self.vector_db = ChromaVectorDB(**self.config)
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    async def initialize(self):
        """Initialize the vector database."""
        await self.vector_db.initialize()
    
    async def store_document_embeddings(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Store document embeddings in the vector database."""
        try:
            if self.db_type == VectorDBType.PINECONE:
                vectors = []
                for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                    vectors.append({
                        "id": doc.get("id", f"doc_{i}"),
                        "values": embedding,
                        "metadata": {
                            "content": doc.get("content", ""),
                            "user_id": doc.get("user_id", ""),
                            "document_type": doc.get("document_type", ""),
                            "timestamp": doc.get("timestamp", "")
                        }
                    })
                await self.vector_db.upsert_vectors(vectors)
                return [v["id"] for v in vectors]
                
            elif self.db_type == VectorDBType.WEAVIATE:
                return await self.vector_db.add_documents(documents, embeddings)
                
            elif self.db_type == VectorDBType.CHROMA:
                doc_texts = [doc.get("content", "") for doc in documents]
                metadatas = [
                    {
                        "user_id": doc.get("user_id", ""),
                        "document_type": doc.get("document_type", ""),
                        "timestamp": doc.get("timestamp", "")
                    }
                    for doc in documents
                ]
                return await self.vector_db.add_documents(
                    documents=doc_texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return []
    
    async def search_similar_documents(
        self,
        query_embedding: List[float],
        user_id: Optional[str] = None,
        document_type: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            filters = {}
            if user_id:
                filters["user_id"] = user_id
            if document_type:
                filters["document_type"] = document_type
            
            if self.db_type == VectorDBType.PINECONE:
                results = await self.vector_db.query_vectors(
                    query_vector=query_embedding,
                    top_k=top_k,
                    filter_metadata=filters if filters else None
                )
                
                return [
                    {
                        "id": result["id"],
                        "score": result["score"],
                        "content": result.get("metadata", {}).get("content", ""),
                        "metadata": result.get("metadata", {})
                    }
                    for result in results
                    if result["score"] >= similarity_threshold
                ]
                
            elif self.db_type == VectorDBType.WEAVIATE:
                where_filter = None
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        if key == "user_id":
                            conditions.append({
                                "path": ["userId"],
                                "operator": "Equal",
                                "valueString": value
                            })
                        elif key == "document_type":
                            conditions.append({
                                "path": ["documentType"],
                                "operator": "Equal",
                                "valueString": value
                            })
                    
                    if conditions:
                        where_filter = {
                            "operator": "And",
                            "operands": conditions
                        } if len(conditions) > 1 else conditions[0]
                
                results = await self.vector_db.search_similar(
                    query_vector=query_embedding,
                    limit=top_k,
                    where_filter=where_filter,
                    certainty=similarity_threshold
                )
                
                return [
                    {
                        "id": result.get("_additional", {}).get("id", ""),
                        "score": result.get("_additional", {}).get("certainty", 0),
                        "content": result.get("content", ""),
                        "metadata": {
                            "user_id": result.get("userId", ""),
                            "document_type": result.get("documentType", ""),
                            "timestamp": result.get("timestamp", "")
                        }
                    }
                    for result in results
                ]
                
            elif self.db_type == VectorDBType.CHROMA:
                where_clause = filters if filters else None
                
                results = await self.vector_db.query_similar(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_clause
                )
                
                formatted_results = []
                if results.get("ids") and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        distance = results.get("distances", [[]])[0][i] if results.get("distances") else 1.0
                        similarity = 1 - distance  # Convert distance to similarity
                        
                        if similarity >= similarity_threshold:
                            formatted_results.append({
                                "id": doc_id,
                                "score": similarity,
                                "content": results.get("documents", [[]])[0][i] if results.get("documents") else "",
                                "metadata": results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                            })
                
                return formatted_results
                
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []
    
    async def delete_user_embeddings(self, user_id: str) -> bool:
        """Delete all embeddings for a specific user."""
        try:
            if self.db_type == VectorDBType.PINECONE:
                return await self.vector_db.delete_vectors(
                    filter_metadata={"user_id": user_id}
                )
                
            elif self.db_type == VectorDBType.WEAVIATE:
                where_filter = {
                    "path": ["userId"],
                    "operator": "Equal",
                    "valueString": user_id
                }
                return await self.vector_db.delete_documents(where_filter)
                
            elif self.db_type == VectorDBType.CHROMA:
                return await self.vector_db.delete_documents(
                    where={"user_id": user_id}
                )
                
        except Exception as e:
            logger.error(f"Failed to delete user embeddings: {e}")
            return False
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get vector storage statistics."""
        try:
            if self.db_type == VectorDBType.PINECONE:
                return await self.vector_db.get_index_stats()
                
            elif self.db_type == VectorDBType.WEAVIATE:
                count = await self.vector_db.get_object_count()
                return {"total_vectors": count}
                
            elif self.db_type == VectorDBType.CHROMA:
                return await self.vector_db.get_collection_info()
                
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
```

## 🤖 RAG Implementation

### Retrieval-Augmented Generation

```python
# prsm/core/rag.py
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from prsm.core.vector_store import VectorStoreManager
from prsm.models.embeddings import EmbeddingRequest
from prsm.models.query import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

class RAGManager:
    """Retrieval-Augmented Generation manager for PRSM."""
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_service,
        max_context_length: int = 4000,
        min_similarity: float = 0.7
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.max_context_length = max_context_length
        self.min_similarity = min_similarity
    
    async def process_rag_query(
        self,
        query: QueryRequest,
        retrieve_count: int = 5
    ) -> Tuple[QueryResponse, List[Dict[str, Any]]]:
        """Process a query using RAG approach."""
        try:
            # Generate query embedding
            query_embedding = await self._get_query_embedding(query.prompt)
            
            # Retrieve relevant documents
            relevant_docs = await self.vector_store.search_similar_documents(
                query_embedding=query_embedding,
                user_id=query.user_id,
                top_k=retrieve_count,
                similarity_threshold=self.min_similarity
            )
            
            if not relevant_docs:
                logger.info("No relevant documents found for RAG query")
                return await self._process_without_context(query), []
            
            # Build context from retrieved documents
            context = self._build_context(relevant_docs, query.prompt)
            
            # Create enhanced query with context
            enhanced_query = self._create_enhanced_query(query, context)
            
            # Process enhanced query
            response = await self._process_enhanced_query(enhanced_query)
            
            # Add retrieval metadata to response
            response.retrieval_metadata = {
                "retrieved_docs": len(relevant_docs),
                "context_length": len(context),
                "min_similarity": min(doc["score"] for doc in relevant_docs),
                "max_similarity": max(doc["score"] for doc in relevant_docs)
            }
            
            return response, relevant_docs
            
        except Exception as e:
            logger.error(f"RAG query processing failed: {e}")
            return await self._process_without_context(query), []
    
    async def _get_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for the query."""
        embedding_request = EmbeddingRequest(
            text=query_text,
            model="text-embedding-ada-002"
        )
        embedding_response = await self.embedding_service.create_embedding(embedding_request)
        return embedding_response.embedding
    
    def _build_context(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        current_length = 0
        
        # Sort documents by relevance score
        sorted_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
        
        for doc in sorted_docs:
            content = doc.get("content", "")
            doc_length = len(content)
            
            # Check if adding this document would exceed max length
            if current_length + doc_length > self.max_context_length:
                # Try to add a truncated version
                remaining_length = self.max_context_length - current_length
                if remaining_length > 100:  # Only add if meaningful
                    truncated_content = content[:remaining_length-3] + "..."
                    context_parts.append(f"Document (Score: {doc['score']:.3f}):\n{truncated_content}")
                break
            
            context_parts.append(f"Document (Score: {doc['score']:.3f}):\n{content}")
            current_length += doc_length
        
        return "\n\n".join(context_parts)
    
    def _create_enhanced_query(
        self,
        original_query: QueryRequest,
        context: str
    ) -> QueryRequest:
        """Create enhanced query with retrieved context."""
        enhanced_prompt = f"""Based on the following context documents, please answer the user's question.

Context:
{context}

User Question: {original_query.prompt}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what additional information might be needed."""
        
        enhanced_query = QueryRequest(
            prompt=enhanced_prompt,
            user_id=original_query.user_id,
            context=original_query.context,
            max_tokens=original_query.max_tokens,
            temperature=original_query.temperature,
            model=original_query.model,
            use_cache=original_query.use_cache
        )
        
        return enhanced_query
    
    async def _process_enhanced_query(
        self,
        query: QueryRequest
    ) -> QueryResponse:
        """Process the enhanced query with context."""
        # This would integrate with your main query processing pipeline
        # For now, this is a placeholder
        pass
    
    async def _process_without_context(
        self,
        query: QueryRequest
    ) -> QueryResponse:
        """Process query without retrieved context."""
        # Fallback to normal query processing
        pass
    
    async def index_document(
        self,
        content: str,
        user_id: str,
        document_type: str = "text",
        metadata: Dict[str, Any] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Index a document for RAG retrieval."""
        try:
            # Split document into chunks
            chunks = self._split_document(content, chunk_size, chunk_overlap)
            
            # Generate embeddings for chunks
            embeddings = []
            for chunk in chunks:
                embedding = await self._get_query_embedding(chunk)
                embeddings.append(embedding)
            
            # Prepare document metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "content": chunk,
                    "user_id": user_id,
                    "document_type": document_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    ***(metadata or {})
                }
                documents.append(doc_metadata)
            
            # Store in vector database
            document_ids = await self.vector_store.store_document_embeddings(
                documents=documents,
                embeddings=embeddings
            )
            
            logger.info(f"Indexed document with {len(chunks)} chunks")
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            return []
    
    def _split_document(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split document into overlapping chunks."""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                sentence_endings = ['. ', '! ', '? ', '\n\n']
                best_break = end
                
                for i in range(max(0, end - 100), end):
                    for ending in sentence_endings:
                        if content[i:i+len(ending)] == ending:
                            best_break = i + len(ending)
                            break
                
                end = best_break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
        
        return chunks
    
    async def update_document_index(
        self,
        document_id: str,
        new_content: str,
        user_id: str,
        document_type: str = "text",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Update an existing document in the index."""
        try:
            # Delete old document chunks
            await self.vector_store.delete_user_embeddings(user_id)
            
            # Re-index with new content
            new_ids = await self.index_document(
                content=new_content,
                user_id=user_id,
                document_type=document_type,
                metadata=metadata
            )
            
            return len(new_ids) > 0
            
        except Exception as e:
            logger.error(f"Failed to update document index: {e}")
            return False
```

## 📊 Performance Optimization

### Vector Search Optimization

```python
# prsm/optimization/vector_optimization.py
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from prsm.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class VectorSearchOptimizer:
    """Optimize vector search performance and accuracy."""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def batch_search(
        self,
        queries: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """Perform batch vector searches for better throughput."""
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.vector_store.search_similar_documents(
                    query_embedding=query["embedding"],
                    user_id=query.get("user_id"),
                    document_type=query.get("document_type"),
                    top_k=query.get("top_k", 10),
                    similarity_threshold=query.get("threshold", 0.7)
                )
                for query in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results
    
    async def adaptive_search(
        self,
        query_embedding: List[float],
        user_id: Optional[str] = None,
        initial_k: int = 5,
        max_k: int = 50,
        target_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Adaptively search for optimal number of results."""
        current_k = initial_k
        best_results = []
        
        while current_k <= max_k and len(best_results) < target_results:
            results = await self.vector_store.search_similar_documents(
                query_embedding=query_embedding,
                user_id=user_id,
                top_k=current_k
            )
            
            # Filter high-quality results
            high_quality = [r for r in results if r["score"] > 0.8]
            
            if len(high_quality) >= target_results:
                best_results = high_quality[:target_results]
                break
            
            best_results = results
            current_k *= 2
        
        return best_results
    
    def calculate_embedding_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = "cosine"
    ) -> float:
        """Calculate similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(vec1 - vec2)
            return 1 / (1 + distance)
            
        elif metric == "dot_product":
            # Dot product similarity
            return np.dot(vec1, vec2)
        
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    async def benchmark_search_performance(
        self,
        test_queries: List[List[float]],
        top_k_values: List[int] = [5, 10, 20, 50]
    ) -> Dict[str, Any]:
        """Benchmark vector search performance."""
        results = {}
        
        for top_k in top_k_values:
            start_time = time.time()
            
            # Run all test queries
            tasks = [
                self.vector_store.search_similar_documents(
                    query_embedding=query,
                    top_k=top_k
                )
                for query in test_queries
            ]
            
            search_results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[f"top_k_{top_k}"] = {
                "total_time": duration,
                "avg_time_per_query": duration / len(test_queries),
                "queries_per_second": len(test_queries) / duration,
                "total_results": sum(len(r) for r in search_results)
            }
        
        return results
    
    async def optimize_similarity_threshold(
        self,
        validation_queries: List[Dict[str, Any]],
        threshold_range: tuple = (0.5, 0.9),
        steps: int = 10
    ) -> float:
        """Find optimal similarity threshold using validation data."""
        thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)
        best_threshold = threshold_range[0]
        best_score = 0
        
        for threshold in thresholds:
            total_precision = 0
            total_recall = 0
            valid_queries = 0
            
            for query_data in validation_queries:
                results = await self.vector_store.search_similar_documents(
                    query_embedding=query_data["embedding"],
                    similarity_threshold=threshold,
                    top_k=20
                )
                
                if "relevant_docs" in query_data:
                    relevant_ids = set(query_data["relevant_docs"])
                    retrieved_ids = set(r["id"] for r in results)
                    
                    if retrieved_ids:
                        precision = len(relevant_ids & retrieved_ids) / len(retrieved_ids)
                        total_precision += precision
                    
                    if relevant_ids:
                        recall = len(relevant_ids & retrieved_ids) / len(relevant_ids)
                        total_recall += recall
                    
                    valid_queries += 1
            
            if valid_queries > 0:
                avg_precision = total_precision / valid_queries
                avg_recall = total_recall / valid_queries
                f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
                
                if f1_score > best_score:
                    best_score = f1_score
                    best_threshold = threshold
        
        logger.info(f"Optimal similarity threshold: {best_threshold:.3f} (F1: {best_score:.3f})")
        return best_threshold
```

## 📋 Testing and Validation

### Vector Database Testing

```python
# tests/test_vector_integration.py
import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, patch
from prsm.core.vector_store import VectorStoreManager, VectorDBType
from prsm.core.rag import RAGManager

@pytest.fixture
async def mock_vector_store():
    """Mock vector store for testing."""
    config = {
        "host": "localhost",
        "port": 8000,
        "collection_name": "test_collection"
    }
    
    manager = VectorStoreManager(VectorDBType.CHROMA, config)
    manager.vector_db = AsyncMock()
    
    return manager

@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return [np.random.rand(384).tolist() for _ in range(10)]

@pytest.fixture
def sample_documents():
    """Generate sample documents for testing."""
    return [
        {
            "id": f"doc_{i}",
            "content": f"This is test document {i} with some content.",
            "user_id": "test_user",
            "document_type": "text",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        for i in range(10)
    ]

class TestVectorStoreManager:
    """Test vector store functionality."""
    
    async def test_store_embeddings(
        self,
        mock_vector_store,
        sample_documents,
        sample_embeddings
    ):
        """Test storing document embeddings."""
        # Mock successful storage
        mock_vector_store.vector_db.add_documents.return_value = [
            doc["id"] for doc in sample_documents
        ]
        
        result_ids = await mock_vector_store.store_document_embeddings(
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        assert len(result_ids) == len(sample_documents)
        mock_vector_store.vector_db.add_documents.assert_called_once()
    
    async def test_search_similar_documents(
        self,
        mock_vector_store,
        sample_embeddings
    ):
        """Test searching for similar documents."""
        # Mock search results
        mock_results = {
            "ids": [["doc_1", "doc_2", "doc_3"]],
            "documents": [["Content 1", "Content 2", "Content 3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [[
                {"user_id": "test_user", "document_type": "text"},
                {"user_id": "test_user", "document_type": "text"},
                {"user_id": "test_user", "document_type": "text"}
            ]]
        }
        
        mock_vector_store.vector_db.query_similar.return_value = mock_results
        
        results = await mock_vector_store.search_similar_documents(
            query_embedding=sample_embeddings[0],
            top_k=3
        )
        
        assert len(results) == 3
        assert all("id" in result for result in results)
        assert all("score" in result for result in results)
        assert all("content" in result for result in results)
    
    async def test_delete_user_embeddings(self, mock_vector_store):
        """Test deleting user embeddings."""
        mock_vector_store.vector_db.delete_documents.return_value = True
        
        result = await mock_vector_store.delete_user_embeddings("test_user")
        
        assert result is True
        mock_vector_store.vector_db.delete_documents.assert_called_once()

class TestRAGManager:
    """Test RAG functionality."""
    
    @pytest.fixture
    async def rag_manager(self, mock_vector_store):
        """Create RAG manager for testing."""
        embedding_service = AsyncMock()
        embedding_service.create_embedding.return_value.embedding = np.random.rand(384).tolist()
        
        rag = RAGManager(
            vector_store=mock_vector_store,
            embedding_service=embedding_service
        )
        
        return rag
    
    async def test_document_chunking(self, rag_manager):
        """Test document chunking functionality."""
        long_content = "This is a test. " * 200  # Create long content
        
        chunks = rag_manager._split_document(
            content=long_content,
            chunk_size=100,
            chunk_overlap=20
        )
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allowing for sentence boundaries
    
    async def test_context_building(self, rag_manager):
        """Test context building from documents."""
        documents = [
            {"content": "First document content", "score": 0.9},
            {"content": "Second document content", "score": 0.8},
            {"content": "Third document content", "score": 0.7}
        ]
        
        context = rag_manager._build_context(documents, "test query")
        
        assert "First document content" in context
        assert "Score: 0.9" in context
        assert len(context) <= rag_manager.max_context_length
    
    async def test_index_document(self, rag_manager, mock_vector_store):
        """Test document indexing."""
        # Mock successful storage
        mock_vector_store.store_document_embeddings.return_value = ["chunk_1", "chunk_2"]
        
        document_ids = await rag_manager.index_document(
            content="This is a test document with some content that should be indexed.",
            user_id="test_user",
            document_type="text"
        )
        
        assert len(document_ids) > 0
        mock_vector_store.store_document_embeddings.assert_called_once()

class TestVectorPerformance:
    """Test vector database performance."""
    
    async def test_batch_operations(self, mock_vector_store, sample_documents, sample_embeddings):
        """Test batch processing performance."""
        batch_size = 5
        
        # Mock batch storage
        mock_vector_store.vector_db.add_documents.return_value = [
            doc["id"] for doc in sample_documents[:batch_size]
        ]
        
        # Process in batches
        for i in range(0, len(sample_documents), batch_size):
            batch_docs = sample_documents[i:i + batch_size]
            batch_embeddings = sample_embeddings[i:i + batch_size]
            
            result_ids = await mock_vector_store.store_document_embeddings(
                documents=batch_docs,
                embeddings=batch_embeddings
            )
            
            assert len(result_ids) == len(batch_docs)
    
    async def test_concurrent_searches(self, mock_vector_store, sample_embeddings):
        """Test concurrent search operations."""
        # Mock search results
        mock_vector_store.vector_db.query_similar.return_value = {
            "ids": [["doc_1"]],
            "documents": [["Test content"]],
            "distances": [[0.1]],
            "metadatas": [[{"user_id": "test_user"}]]
        }
        
        # Run concurrent searches
        tasks = [
            mock_vector_store.search_similar_documents(
                query_embedding=embedding,
                top_k=5
            )
            for embedding in sample_embeddings[:5]
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(isinstance(result, list) for result in results)

# Integration test with real vector database
@pytest.mark.integration
async def test_real_chroma_integration():
    """Test integration with real Chroma database."""
    try:
        config = {
            "host": "localhost",
            "port": 8000,
            "collection_name": "test_integration"
        }
        
        manager = VectorStoreManager(VectorDBType.CHROMA, config)
        await manager.initialize()
        
        # Test document storage
        documents = [
            {
                "id": "test_doc_1",
                "content": "This is a test document for integration testing.",
                "user_id": "integration_test_user",
                "document_type": "test"
            }
        ]
        
        embeddings = [np.random.rand(384).tolist()]
        
        doc_ids = await manager.store_document_embeddings(
            documents=documents,
            embeddings=embeddings
        )
        
        assert len(doc_ids) == 1
        
        # Test search
        search_results = await manager.search_similar_documents(
            query_embedding=embeddings[0],
            top_k=1
        )
        
        assert len(search_results) > 0
        assert search_results[0]["id"] == "test_doc_1"
        
        # Cleanup
        await manager.delete_user_embeddings("integration_test_user")
        
    except Exception as e:
        pytest.skip(f"Real Chroma integration not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
```

## 📋 Deployment and Production

### Production Configuration

```yaml
# docker-compose.vector-stack.yml
version: '3.8'

services:
  chroma:
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_DB_IMPL=clickhouse
      - CLICKHOUSE_HOST=clickhouse
      - CLICKHOUSE_PORT=8123
    depends_on:
      - clickhouse
    networks:
      - vector-network

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports:
      - "8123:8123"
      - "9000:9000"
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    environment:
      - CLICKHOUSE_DB=default
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=
    networks:
      - vector-network

  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER_MODULE=none
      - CLUSTER_HOSTNAME=node1
    volumes:
      - weaviate_data:/var/lib/weaviate
    networks:
      - vector-network

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - vector-network

volumes:
  chroma_data:
  clickhouse_data:
  weaviate_data:
  qdrant_data:

networks:
  vector-network:
    driver: bridge
```

---

**Need help with vector database integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).
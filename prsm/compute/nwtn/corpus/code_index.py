"""
Codebase Index - Persistent vector index over PRSM source code using ChromaDB.

Builds and persists a ChromaDB collection from CodeChunk objects.
Uses chromadb's default embedding function (all-MiniLM-L6-v2 via sentence-transformers).
"""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from .code_chunker import CodeChunk, CodeChunker


@dataclass
class SearchResult:
    """A search result from the codebase index."""
    
    filepath: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    docstring: str
    module_path: str
    score: float
    snippet: str           # first 300 chars of source


class CodebaseIndex:
    """
    Persistent vector index over PRSM source code using ChromaDB + local embeddings.
    
    Embeddings are generated from: "{symbol_name}: {docstring or first 200 chars of source}"
    Uses chromadb's default embedding function (all-MiniLM-L6-v2 via sentence-transformers,
    or falls back to chromadb's built-in if sentence-transformers not available).
    
    Index persists to: {repo_root}/.codebase_index/
    """
    
    COLLECTION_NAME = "prsm_codebase"
    EMBEDDING_TEXT_MAX_CHARS = 200
    SNIPPET_MAX_CHARS = 300
    
    def __init__(self, repo_root: str, persist_dir: Optional[str] = None):
        """
        Initialize the codebase index.
        
        Args:
            repo_root: Path to the repository root
            persist_dir: Optional custom persist directory (defaults to {repo_root}/.codebase_index/)
        """
        self.repo_root = Path(repo_root).resolve()
        self.persist_dir = Path(persist_dir) if persist_dir else self.repo_root / ".codebase_index"
        
        self._client = None
        self._collection = None
        self._embedding_function = None
        self._indexed_ids: set = set()
    
    def _get_embedding_text(self, chunk: CodeChunk) -> str:
        """Generate text for embedding from a code chunk."""
        if chunk.docstring:
            return f"{chunk.symbol_name}: {chunk.docstring}"
        else:
            # Use first 200 chars of source
            source_preview = chunk.source[:self.EMBEDDING_TEXT_MAX_CHARS].strip()
            return f"{chunk.symbol_name}: {source_preview}"
    
    def _get_snippet(self, source: str) -> str:
        """Get a snippet (first 300 chars) from source code."""
        return source[:self.SNIPPET_MAX_CHARS].strip()
    
    def _chunk_to_metadata(self, chunk: CodeChunk) -> Dict[str, Any]:
        """Convert a CodeChunk to ChromaDB metadata."""
        return {
            "filepath": chunk.filepath,
            "symbol_name": chunk.symbol_name,
            "symbol_type": chunk.symbol_type,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "docstring": chunk.docstring[:500] if chunk.docstring else "",  # Truncate for metadata limits
            "module_path": chunk.module_path,
        }
    
    def _result_from_metadata(self, metadata: Dict[str, Any], score: float, snippet: str) -> SearchResult:
        """Create a SearchResult from ChromaDB metadata."""
        return SearchResult(
            filepath=metadata.get("filepath", ""),
            symbol_name=metadata.get("symbol_name", ""),
            symbol_type=metadata.get("symbol_type", "function"),
            start_line=metadata.get("start_line", 0),
            end_line=metadata.get("end_line", 0),
            docstring=metadata.get("docstring", ""),
            module_path=metadata.get("module_path", ""),
            score=score,
            snippet=snippet,
        )
    
    async def _init_client(self) -> None:
        """Initialize ChromaDB client (lazy, called when needed)."""
        if self._client is not None:
            return
        
        def _init():
            import chromadb
            from chromadb.utils import embedding_functions
            
            # Create persist directory
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize persistent client
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            
            # Use default embedding function
            self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        await asyncio.to_thread(_init)
    
    async def _get_or_create_collection(self) -> None:
        """Get or create the ChromaDB collection."""
        if self._collection is not None:
            return
        
        await self._init_client()
        
        def _create():
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Load existing IDs
            existing = self._collection.get(include=[])
            self._indexed_ids = set(existing.get("ids", []))
        
        await asyncio.to_thread(_create)
    
    def is_built(self) -> bool:
        """
        Check if the index has been built and is non-empty.
        
        Returns:
            True if the index exists and contains at least one document
        """
        if self._collection is None:
            return False
        
        try:
            count = self._collection.count()
            return count > 0
        except Exception:
            return False
    
    async def build(self, force_rebuild: bool = False) -> int:
        """
        Build the index. Skips files that haven't changed (uses chunk_id for dedup).
        
        Args:
            force_rebuild: If True, rebuild the entire index from scratch
            
        Returns:
            Number of chunks indexed
        """
        await self._get_or_create_collection()
        
        # Clear existing index if force rebuild
        if force_rebuild and self._collection:
            def _clear():
                existing = self._collection.get(include=[])
                if existing and existing.get("ids"):
                    self._collection.delete(ids=existing["ids"])
                self._indexed_ids = set()
            
            await asyncio.to_thread(_clear)
        
        # Chunk the repo
        chunker = CodeChunker(str(self.repo_root))
        chunks = await asyncio.to_thread(chunker.chunk_repo)
        
        if not chunks:
            return 0
        
        # Filter out already-indexed chunks
        new_chunks = [c for c in chunks if c.chunk_id not in self._indexed_ids]
        
        if not new_chunks:
            return 0
        
        # Prepare data for ChromaDB
        ids = [c.chunk_id for c in new_chunks]
        documents = [self._get_embedding_text(c) for c in new_chunks]
        metadatas = [self._chunk_to_metadata(c) for c in new_chunks]
        
        # Add to collection in batches (ChromaDB has a batch size limit)
        BATCH_SIZE = 5000
        
        def _add_batch(batch_ids, batch_docs, batch_metas):
            self._collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            self._indexed_ids.update(batch_ids)
        
        total_added = 0
        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i:i + BATCH_SIZE]
            batch_docs = documents[i:i + BATCH_SIZE]
            batch_metas = metadatas[i:i + BATCH_SIZE]
            
            await asyncio.to_thread(_add_batch, batch_ids, batch_docs, batch_metas)
            total_added += len(batch_ids)
        
        return total_added
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        symbol_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Semantic search over the codebase.
        
        Args:
            query: Natural language search query
            top_k: Maximum number of results to return
            symbol_type: Optional filter: "class" or "function"
            
        Returns:
            List of SearchResult objects sorted by relevance score
        """
        await self._get_or_create_collection()
        
        if not self.is_built():
            return []
        
        # Build where clause for symbol_type filter
        where = None
        if symbol_type:
            where = {"symbol_type": symbol_type}
        
        def _query():
            return self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["metadatas", "distances", "documents"]
            )
        
        results = await asyncio.to_thread(_query)
        
        # Convert to SearchResult objects
        search_results: List[SearchResult] = []
        
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
        
        ids = results["ids"][0]
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        documents = results.get("documents", [[]])[0] or []
        
        for i, chunk_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            doc = documents[i] if i < len(documents) else ""
            
            # Convert distance to score (ChromaDB returns distances, lower is better)
            # For cosine distance, score = 1 - distance
            score = 1.0 - distance
            
            # Use document as snippet (it's the embedding text)
            snippet = doc[:self.SNIPPET_MAX_CHARS] if doc else ""
            
            result = self._result_from_metadata(metadata, score, snippet)
            search_results.append(result)
        
        return search_results
    
    async def rebuild(self) -> int:
        """
        Rebuild the entire index from scratch.
        
        Returns:
            Number of chunks indexed
        """
        return await self.build(force_rebuild=True)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        await self._get_or_create_collection()
        
        if not self._collection:
            return {"total_chunks": 0}
        
        count = await asyncio.to_thread(self._collection.count)
        
        return {
            "total_chunks": count,
            "persist_dir": str(self.persist_dir),
        }

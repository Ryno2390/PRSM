"""
Checkpoint → Long-Term Memory Pipeline
======================================

Commits checkpoint narratives to long-term searchable memory via embeddings.
This is the final piece of the NWTN Agent Team architecture — it ensures that
nothing is lost when agent context windows reset at checkpoints.

The pipeline bridges the Project Ledger (append-only Markdown) and the vector
store (searchable embeddings). After each checkpoint cycle, the narrative is
chunked, embedded, and stored so that:

1. Future agents can semantically search past checkpoint history
2. The MetaPlanUpdater can reference historical context
3. New agents joining mid-project can quickly find relevant past work
4. Nothing is lost when context windows reset

Components
----------
MemoryChunk / MemoryCommitResult / MemorySearchResult
    Data models for chunks, commit results, and search results.

NarrativeChunker
    Intelligently splits checkpoint narratives into semantically meaningful
    chunks for embedding. Understands Markdown structure and preserves
    section boundaries.

LocalMemoryStore
    Lightweight fallback when no vector store is available. Stores chunks
    as JSON files and uses TF-IDF for basic search. Ensures the system
    never fails just because external services aren't configured.

CheckpointMemoryCommitter
    Main orchestrator that coordinates chunking, embedding, and storage.
    Auto-detects whether to use real embeddings/vector stores or fall back
    to local storage.

Idempotency
-----------
The committer is idempotent — committing the same ledger entry twice skips
already-embedded chunks (dedup by content_hash). This makes retry logic
simple and safe.

Quick start
-----------
.. code-block:: python

    from prsm.compute.nwtn.synthesis import (
        CheckpointMemoryCommitter, NarrativeChunker,
    )

    # After a checkpoint cycle:
    committer = CheckpointMemoryCommitter(
        embedding_pipeline=pipeline,  # optional
        vector_store=store,           # optional
    )
    await committer.setup()
    
    result = await committer.commit_checkpoint(
        synthesis=result.synthesis,
        ledger_entry=result.ledger_entry,
    )

    # For agent onboarding:
    context = await committer.get_onboarding_context(
        query="work related to authentication",
        max_chunks=10,
    )

    # Bootstrapping from existing ledger:
    results = await committer.commit_full_ledger(ledger)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .ledger import LedgerEntry, ProjectLedger
    from .reconstructor import SynthesisResult

logger = logging.getLogger(__name__)


# ======================================================================
# Data models
# ======================================================================

@dataclass
class MemoryChunk:
    """
    A single chunk of a checkpoint narrative prepared for embedding.
    
    Each chunk represents a semantically meaningful portion of a checkpoint
    narrative, suitable for embedding and retrieval.
    """
    chunk_id: str
    """Unique ID: "{session_id}:{entry_index}:{chunk_index}" """
    
    session_id: str
    """Which session this chunk came from."""
    
    entry_index: int
    """Which ledger entry this came from."""
    
    chunk_index: int
    """Position within the entry's chunks."""
    
    content: str
    """The text chunk to be embedded."""
    
    metadata: Dict[str, Any]
    """Rich metadata: session_id, timestamp, agents, milestone refs, etc."""
    
    content_hash: str
    """SHA-256 of content for deduplication."""
    
    timestamp: datetime
    """When the original checkpoint was created."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryChunk":
        """Deserialize from a dict."""
        d = dict(d)
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


@dataclass
class MemoryCommitResult:
    """
    Result of committing a checkpoint to long-term memory.
    
    Provides detailed statistics about what was committed, skipped, or failed.
    """
    session_id: str
    """Session that was committed."""
    
    entry_index: int
    """Ledger entry index that was committed."""
    
    chunks_created: int
    """Total chunks generated from the narrative."""
    
    chunks_embedded: int
    """Chunks successfully embedded and stored."""
    
    chunks_skipped: int
    """Chunks skipped due to dedup (already existed)."""
    
    embedding_model: str
    """Which embedding model was used."""
    
    storage_backend: str
    """Which vector store was used (or 'local')."""
    
    total_tokens: int
    """Approximate token count for the embedded content."""
    
    commit_time_seconds: float
    """Time taken to commit."""
    
    success: bool
    """Whether the commit succeeded."""
    
    error: Optional[str] = None
    """Error message if success=False."""


@dataclass
class MemorySearchResult:
    """
    A single result from searching long-term checkpoint memory.
    
    Contains the matched chunk with similarity score and source attribution.
    """
    chunk: MemoryChunk
    """The matching chunk."""
    
    score: float
    """Similarity score 0-1."""
    
    session_id: str
    """Source session."""
    
    entry_index: int
    """Source ledger entry."""
    
    section_title: Optional[str] = None
    """Which section of the narrative this came from."""
    
    agents_involved: List[str] = field(default_factory=list)
    """Agents from the original checkpoint."""
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the original checkpoint was created."""


# ======================================================================
# Narrative Chunker
# ======================================================================

class NarrativeChunker:
    """
    Splits checkpoint narratives into semantically meaningful chunks.
    
    Unlike generic text splitters, this understands the structure of NWTN
    checkpoint narratives (Markdown with specific section headers) and
    preserves section boundaries.
    
    Chunking strategy
    -----------------
    1. Split by Markdown headers (##, ###) first — each section is a natural chunk
    2. If a section exceeds max_chunk_size, split by paragraphs
    3. If a paragraph exceeds max_chunk_size, split by sentences with overlap
    4. Each chunk gets a prefix with context: "Session: X | Checkpoint: Y | Section: Z"
    
    Parameters
    ----------
    max_chunk_size : int
        Maximum chunk size in approximate tokens (default: 512).
    chunk_overlap : int
        Overlap tokens for sentence-level splits (default: 50).
    include_context_prefix : bool
        Whether to add a context prefix to each chunk (default: True).
    """
    
    # Approximate characters per token (varies by model, but 4 is reasonable)
    CHARS_PER_TOKEN = 4
    
    # Markdown header patterns
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        chunk_overlap: int = 50,
        include_context_prefix: bool = True,
    ) -> None:
        self._max_chunk_size = max_chunk_size
        self._chunk_overlap = chunk_overlap
        self._include_context_prefix = include_context_prefix
    
    def chunk_narrative(
        self,
        narrative: str,
        session_id: str,
        entry_index: int,
        metadata: Optional[Dict] = None,
    ) -> List[MemoryChunk]:
        """
        Split a checkpoint narrative into chunks for embedding.
        
        Parameters
        ----------
        narrative : str
            The Markdown narrative to chunk.
        session_id : str
            Session identifier.
        entry_index : int
            Ledger entry index.
        metadata : dict, optional
            Additional metadata to include in each chunk.
        
        Returns
        -------
        List[MemoryChunk]
            List of chunks ready for embedding.
        """
        chunks: List[MemoryChunk] = []
        base_metadata = metadata or {}
        
        # Extract sections from the Markdown
        sections = self._extract_sections(narrative)
        
        chunk_index = 0
        for section_title, section_content in sections:
            section_chunks = self._chunk_section(
                content=section_content,
                section_title=section_title,
            )
            
            for chunk_content in section_chunks:
                chunk_id = f"{session_id}:{entry_index}:{chunk_index}"
                
                # Add context prefix if enabled
                if self._include_context_prefix:
                    prefix = self._build_context_prefix(
                        session_id=session_id,
                        entry_index=entry_index,
                        section_title=section_title,
                    )
                    chunk_content = f"{prefix}\n\n{chunk_content}"
                
                # Compute content hash
                content_hash = self._hash_content(chunk_content)
                
                # Build chunk metadata
                chunk_metadata = {
                    **base_metadata,
                    "session_id": session_id,
                    "entry_index": entry_index,
                    "chunk_index": chunk_index,
                    "section_title": section_title,
                    "content_length": len(chunk_content),
                }
                
                # Handle timestamp - may be datetime, string, or missing
                ts = base_metadata.get("timestamp", datetime.now(timezone.utc))
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                elif not isinstance(ts, datetime):
                    ts = datetime.now(timezone.utc)
                
                chunk = MemoryChunk(
                    chunk_id=chunk_id,
                    session_id=session_id,
                    entry_index=entry_index,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    metadata=chunk_metadata,
                    content_hash=content_hash,
                    timestamp=ts,
                )
                chunks.append(chunk)
                chunk_index += 1
        
        logger.debug(
            "NarrativeChunker: split narrative into %d chunks (session=%s, entry=%d)",
            len(chunks), session_id, entry_index,
        )
        return chunks
    
    def _extract_sections(self, narrative: str) -> List[Tuple[Optional[str], str]]:
        """
        Extract sections from Markdown narrative.
        
        Returns list of (section_title, section_content) tuples.
        Sections without headers get None as title.
        """
        sections: List[Tuple[Optional[str], str]] = []
        
        # Find all headers
        header_matches = list(self.HEADER_PATTERN.finditer(narrative))
        
        if not header_matches:
            # No headers — treat entire narrative as one section
            return [(None, narrative.strip())]
        
        # Add content before first header as intro section
        first_header_start = header_matches[0].start()
        if first_header_start > 0:
            intro = narrative[:first_header_start].strip()
            if intro:
                sections.append(("Introduction", intro))
        
        # Extract each section
        for i, match in enumerate(header_matches):
            header_level = len(match.group(1))
            section_title = match.group(2).strip()
            
            # Get content until next header (or end)
            content_start = match.end()
            content_end = (
                header_matches[i + 1].start()
                if i + 1 < len(header_matches)
                else len(narrative)
            )
            section_content = narrative[content_start:content_end].strip()
            
            if section_content:
                sections.append((section_title, section_content))
        
        return sections
    
    def _chunk_section(
        self,
        content: str,
        section_title: Optional[str],
    ) -> List[str]:
        """
        Chunk a single section, respecting size limits.
        """
        max_chars = self._max_chunk_size * self.CHARS_PER_TOKEN
        
        if len(content) <= max_chars:
            return [content]
        
        # Split by paragraphs first
        paragraphs = self._split_paragraphs(content)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size + 2 <= max_chars:
                # Add to current chunk
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for "\n\n"
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                
                if para_size > max_chars:
                    # Paragraph itself is too big — split by sentences
                    sentence_chunks = self._chunk_by_sentences(para, max_chars)
                    chunks.extend(sentence_chunks)
                    current_chunk = []
                    current_size = 0
                else:
                    current_chunk = [para]
                    current_size = para_size
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def _split_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs (double newlines)."""
        return [p.strip() for p in content.split("\n\n") if p.strip()]
    
    def _chunk_by_sentences(self, text: str, max_chars: int) -> List[str]:
        """
        Split text by sentences when paragraphs are too long.
        Includes overlap between chunks.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return [text[:max_chars]]
        
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0
        overlap_chars = self._chunk_overlap * self.CHARS_PER_TOKEN
        
        for sentence in sentences:
            sent_size = len(sentence)
            
            if current_size + sent_size + 1 <= max_chars:
                current_chunk.append(sentence)
                current_size += sent_size + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Add overlap from previous chunk if available
                if chunks and overlap_chars > 0:
                    prev_text = chunks[-1]
                    overlap_start = max(0, len(prev_text) - overlap_chars)
                    overlap_text = prev_text[overlap_start:]
                    current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                    current_size = len(overlap_text) + sent_size + 1 if overlap_text else sent_size
                else:
                    current_chunk = [sentence]
                    current_size = sent_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting — handles most cases
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _build_context_prefix(
        self,
        session_id: str,
        entry_index: int,
        section_title: Optional[str],
    ) -> str:
        """Build a context prefix for a chunk."""
        parts = [f"Session: {session_id}", f"Checkpoint: {entry_index}"]
        if section_title:
            parts.append(f"Section: {section_title}")
        return " | ".join(parts)
    
    def _hash_content(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ======================================================================
# Local Memory Store (Fallback)
# ======================================================================

class LocalMemoryStore:
    """
    Simple local fallback when no vector store is available.
    
    Stores chunks as JSON files and uses TF-IDF for basic search.
    Not as good as real embeddings, but ensures the system never fails
    just because a vector store isn't configured.
    
    Storage layout
    --------------
    - {memory_dir}/chunks/{session_id}/{entry_index}.json — chunk files
    - {memory_dir}/index.json — TF-IDF term frequencies index
    - {memory_dir}/hashes.json — content hash → chunk_id mapping for dedup
    
    Parameters
    ----------
    memory_dir : Path
        Directory for storing chunks and index.
    """
    
    def __init__(self, memory_dir: Path) -> None:
        self._dir = Path(memory_dir)
        self._chunks_dir = self._dir / "chunks"
        self._index_path = self._dir / "index.json"
        self._hashes_path = self._dir / "hashes.json"
        
        # In-memory caches
        self._chunk_cache: Dict[str, MemoryChunk] = {}
        self._hash_index: Dict[str, str] = {}  # content_hash -> chunk_id
        self._tf_index: Dict[str, Counter] = {}  # chunk_id -> term frequencies
        self._idf_cache: Dict[str, float] = {}  # term -> IDF value
        self._document_count: int = 0
        self._loaded = False
    
    async def setup(self) -> None:
        """Initialize the local store, loading existing data."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._chunks_dir.mkdir(parents=True, exist_ok=True)
        
        await self._load_index()
        self._loaded = True
        logger.info(
            "LocalMemoryStore: initialized with %d chunks in %s",
            len(self._chunk_cache), self._dir,
        )
    
    async def store_chunks(self, chunks: List[MemoryChunk]) -> int:
        """
        Store chunks to local files and update TF-IDF index.
        
        Returns number of chunks stored (excluding duplicates).
        """
        stored = 0
        
        for chunk in chunks:
            # Check for duplicates
            if await self.chunk_exists(chunk.content_hash):
                logger.debug(
                    "LocalMemoryStore: skipping duplicate chunk %s",
                    chunk.chunk_id,
                )
                continue
            
            # Store to file
            chunk_file = self._chunks_dir / chunk.session_id / f"{chunk.entry_index}_{chunk.chunk_index}.json"
            chunk_file.parent.mkdir(parents=True, exist_ok=True)
            chunk_file.write_text(
                json.dumps(chunk.to_dict(), indent=2),
                encoding="utf-8",
            )
            
            # Update in-memory cache
            self._chunk_cache[chunk.chunk_id] = chunk
            self._hash_index[chunk.content_hash] = chunk.chunk_id
            
            # Update TF-IDF index
            self._update_tf_index(chunk)
            
            stored += 1
        
        # Update IDF cache
        self._rebuild_idf_cache()
        
        # Persist index
        await self._save_index()
        
        logger.info(
            "LocalMemoryStore: stored %d chunks (%d total)",
            stored, len(self._chunk_cache),
        )
        return stored
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[MemoryChunk, float]]:
        """
        Search chunks using TF-IDF cosine similarity.
        
        Returns list of (chunk, score) tuples sorted by relevance.
        """
        if not self._loaded:
            await self._load_index()
        
        if not self._chunk_cache:
            return []
        
        # Tokenize query
        query_terms = self._tokenize(query)
        query_tf = Counter(query_terms)
        
        # Compute query TF-IDF vector
        query_tfidf: Dict[str, float] = {}
        query_norm = 0.0
        for term, tf in query_tf.items():
            idf = self._idf_cache.get(term, math.log(self._document_count + 1))
            tfidf = (1 + math.log(tf)) * idf
            query_tfidf[term] = tfidf
            query_norm += tfidf ** 2
        query_norm = math.sqrt(query_norm) if query_norm > 0 else 1.0
        
        # Compute similarity with each chunk
        scores: List[Tuple[MemoryChunk, float]] = []
        for chunk_id, chunk in self._chunk_cache.items():
            chunk_tf = self._tf_index.get(chunk_id, Counter())
            
            # Cosine similarity
            dot_product = 0.0
            chunk_norm = 0.0
            for term, tfidf in query_tfidf.items():
                if term in chunk_tf:
                    chunk_tfidf = (1 + math.log(chunk_tf[term])) * self._idf_cache.get(term, 1.0)
                    dot_product += tfidf * chunk_tfidf
            
            for term, tf in chunk_tf.items():
                idf = self._idf_cache.get(term, 1.0)
                chunk_tfidf = (1 + math.log(tf)) * idf
                chunk_norm += chunk_tfidf ** 2
            chunk_norm = math.sqrt(chunk_norm) if chunk_norm > 0 else 1.0
            
            similarity = dot_product / (query_norm * chunk_norm) if query_norm > 0 and chunk_norm > 0 else 0.0
            scores.append((chunk, similarity))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    async def chunk_exists(self, content_hash: str) -> bool:
        """Check if a chunk with this content hash already exists."""
        if not self._loaded:
            await self._load_index()
        return content_hash in self._hash_index
    
    async def get_chunk(self, chunk_id: str) -> Optional[MemoryChunk]:
        """Get a specific chunk by ID."""
        if not self._loaded:
            await self._load_index()
        return self._chunk_cache.get(chunk_id)
    
    async def stats(self) -> Dict[str, Any]:
        """Return statistics about the local store."""
        if not self._loaded:
            await self._load_index()
        
        # Count unique sessions and entries
        sessions: set = set()
        entries: set = set()
        for chunk in self._chunk_cache.values():
            sessions.add(chunk.session_id)
            entries.add(f"{chunk.session_id}:{chunk.entry_index}")
        
        return {
            "storage_backend": "local",
            "total_chunks": len(self._chunk_cache),
            "unique_sessions": len(sessions),
            "unique_entries": len(entries),
            "memory_dir": str(self._dir),
            "index_size_mb": (
                self._index_path.stat().st_size / (1024 * 1024)
                if self._index_path.exists() else 0
            ),
        }
    
    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    
    async def _load_index(self) -> None:
        """Load existing chunks and index from disk."""
        # Load hash index
        if self._hashes_path.exists():
            try:
                self._hash_index = json.loads(self._hashes_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load hash index: %s", exc)
                self._hash_index = {}
        
        # Load chunks
        for chunk_file in self._chunks_dir.rglob("*.json"):
            try:
                chunk_data = json.loads(chunk_file.read_text(encoding="utf-8"))
                chunk = MemoryChunk.from_dict(chunk_data)
                self._chunk_cache[chunk.chunk_id] = chunk
            except Exception as exc:
                logger.warning("Failed to load chunk %s: %s", chunk_file, exc)
        
        # Rebuild TF index from chunks
        for chunk in self._chunk_cache.values():
            self._update_tf_index(chunk)
        
        self._rebuild_idf_cache()
        logger.debug(
            "LocalMemoryStore: loaded %d chunks from disk",
            len(self._chunk_cache),
        )
    
    async def _save_index(self) -> None:
        """Save the hash index to disk."""
        self._hashes_path.write_text(
            json.dumps(self._hash_index, indent=2),
            encoding="utf-8",
        )
    
    def _update_tf_index(self, chunk: MemoryChunk) -> None:
        """Update the TF index for a chunk."""
        terms = self._tokenize(chunk.content)
        self._tf_index[chunk.chunk_id] = Counter(terms)
        self._document_count = len(self._tf_index)
    
    def _rebuild_idf_cache(self) -> None:
        """Rebuild the IDF cache from the TF index."""
        self._document_count = len(self._tf_index)
        if self._document_count == 0:
            self._idf_cache = {}
            return
        
        # Count document frequency for each term
        doc_freq: Counter = Counter()
        for tf in self._tf_index.values():
            for term in tf:
                doc_freq[term] += 1
        
        # Compute IDF for each term
        self._idf_cache = {
            term: math.log(self._document_count / (df + 1))
            for term, df in doc_freq.items()
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase terms.
        
        Simple tokenization: extract words (3+ chars), lowercase, no stopwords.
        """
        # Extract words (3+ characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Simple stopword list
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "been",
            "this", "that", "with", "from", "they", "will", "would", "there",
            "their", "what", "about", "which", "when", "make", "like", "into",
            "than", "them", "then", "some", "could", "other", "these", "those",
            "being", "after", "before", "through", "during", "between", "under",
            "again", "further", "once", "here", "where", "why", "how", "both",
            "each", "more", "most", "very", "just", "should", "now", "only",
            "also", "over", "such", "own", "same", "too", "very", "can",
        }
        
        return [w for w in words if w not in stopwords]


# ======================================================================
# Checkpoint Memory Committer
# ======================================================================

class CheckpointMemoryCommitter:
    """
    Commits checkpoint narratives to long-term searchable memory.
    
    This is the bridge between the Project Ledger (append-only Markdown)
    and the vector store (searchable embeddings). After each checkpoint
    cycle, the narrative is chunked, embedded, and stored so that:
    
    1. Future agents can semantically search past checkpoint history
    2. The MetaPlanUpdater can reference historical context
    3. New agents joining mid-project can quickly find relevant past work
    4. Nothing is lost when context windows reset
    
    The committer is idempotent — committing the same ledger entry twice
    skips already-embedded chunks (dedup by content_hash).
    
    Parameters
    ----------
    chunker : NarrativeChunker, optional
        Chunker for splitting narratives. Created with defaults if None.
    embedding_pipeline : EmbeddingPipeline, optional
        Pipeline for generating embeddings. Falls back to local storage if None.
    vector_store : PRSMVectorStore, optional
        Vector store for embeddings. Falls back to local storage if None.
    collection_name : str
        Name for the vector store collection (default: "nwtn_checkpoint_memory").
    backend_registry : optional
        LLM backend for generating search-optimized summaries.
    local_memory_dir : Path, optional
        Directory for local fallback storage. Default: ".prsm/memory".
    """
    
    def __init__(
        self,
        chunker: Optional[NarrativeChunker] = None,
        embedding_pipeline: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        collection_name: str = "nwtn_checkpoint_memory",
        backend_registry=None,
        local_memory_dir: Optional[Path] = None,
    ) -> None:
        self._chunker = chunker or NarrativeChunker()
        self._embedding_pipeline = embedding_pipeline
        self._vector_store = vector_store
        self._collection_name = collection_name
        self._backend = backend_registry
        
        # Local fallback storage
        self._local_store: Optional[LocalMemoryStore] = None
        self._local_dir = local_memory_dir or Path(".prsm/memory")
        
        # Track whether we're using real embeddings or local fallback
        self._use_embeddings = embedding_pipeline is not None and vector_store is not None
        
        # Track committed entries for dedup
        self._committed_entries: set = set()
    
    async def setup(self) -> None:
        """
        Initialize connections to embedding pipeline and vector store.
        
        If real embeddings are configured, tests the connection.
        Otherwise, initializes the local fallback store.
        """
        if self._use_embeddings:
            try:
                # Test embedding pipeline
                if hasattr(self._embedding_pipeline, "health_check"):
                    health = await self._embedding_pipeline.health_check()
                    if not health.get("overall_healthy", False):
                        logger.warning(
                            "CheckpointMemoryCommitter: embedding pipeline not healthy, "
                            "falling back to local storage"
                        )
                        self._use_embeddings = False
                logger.info(
                    "CheckpointMemoryCommitter: using embedding pipeline "
                    "(provider=%s)",
                    getattr(self._embedding_pipeline, "config", {}).get(
                        "preferred_embedding_provider", "unknown"
                    ) if hasattr(self._embedding_pipeline, "config") else "unknown",
                )
            except Exception as exc:
                logger.warning(
                    "CheckpointMemoryCommitter: embedding pipeline setup failed (%s), "
                    "falling back to local storage",
                    exc,
                )
                self._use_embeddings = False
        
        if not self._use_embeddings:
            # Initialize local fallback
            self._local_store = LocalMemoryStore(self._local_dir)
            await self._local_store.setup()
            logger.info(
                "CheckpointMemoryCommitter: using local storage fallback at %s",
                self._local_dir,
            )
    
    async def commit_checkpoint(
        self,
        synthesis: "SynthesisResult",
        ledger_entry: "LedgerEntry",
        meta_plan: Optional[Any] = None,
    ) -> MemoryCommitResult:
        """
        Commit a checkpoint narrative to long-term memory.
        
        Steps:
        1. Chunk the narrative using NarrativeChunker
        2. Dedup: check content_hash against already-committed chunks
        3. Generate embeddings for new chunks (or store locally)
        4. Store in vector store with rich metadata
        5. Return commit result
        
        Parameters
        ----------
        synthesis : SynthesisResult
            The synthesis from the checkpoint cycle.
        ledger_entry : LedgerEntry
            The ledger entry that was created.
        meta_plan : MetaPlan, optional
            The meta plan for milestone context.
        
        Returns
        -------
        MemoryCommitResult
            Detailed result of the commit operation.
        """
        start_time = time.time()
        
        try:
            # Check if already committed (idempotency)
            entry_key = f"{synthesis.session_id}:{ledger_entry.entry_index}"
            if entry_key in self._committed_entries:
                logger.info(
                    "CheckpointMemoryCommitter: entry %s already committed, skipping",
                    entry_key,
                )
                return MemoryCommitResult(
                    session_id=synthesis.session_id,
                    entry_index=ledger_entry.entry_index,
                    chunks_created=0,
                    chunks_embedded=0,
                    chunks_skipped=0,
                    embedding_model="none",
                    storage_backend="skipped",
                    total_tokens=0,
                    commit_time_seconds=time.time() - start_time,
                    success=True,
                )
            
            # Build metadata
            metadata = self._build_chunk_metadata(synthesis, ledger_entry, meta_plan)
            
            # Chunk the narrative
            chunks = self._chunker.chunk_narrative(
                narrative=synthesis.narrative,
                session_id=synthesis.session_id,
                entry_index=ledger_entry.entry_index,
                metadata=metadata,
            )
            
            if not chunks:
                return MemoryCommitResult(
                    session_id=synthesis.session_id,
                    entry_index=ledger_entry.entry_index,
                    chunks_created=0,
                    chunks_embedded=0,
                    chunks_skipped=0,
                    embedding_model="none",
                    storage_backend="none",
                    total_tokens=0,
                    commit_time_seconds=time.time() - start_time,
                    success=True,
                    error="No chunks created from narrative",
                )
            
            # Dedup: filter out already-existing chunks
            new_chunks: List[MemoryChunk] = []
            skipped = 0
            
            for chunk in chunks:
                exists = await self._chunk_exists(chunk.content_hash)
                if exists:
                    skipped += 1
                else:
                    new_chunks.append(chunk)
            
            if not new_chunks:
                logger.info(
                    "CheckpointMemoryCommitter: all %d chunks already exist, skipping",
                    len(chunks),
                )
                return MemoryCommitResult(
                    session_id=synthesis.session_id,
                    entry_index=ledger_entry.entry_index,
                    chunks_created=len(chunks),
                    chunks_embedded=0,
                    chunks_skipped=skipped,
                    embedding_model="none",
                    storage_backend="dedup",
                    total_tokens=0,
                    commit_time_seconds=time.time() - start_time,
                    success=True,
                )
            
            # Store chunks
            embedded = 0
            total_tokens = sum(len(c.content) // 4 for c in new_chunks)  # Approximate
            
            if self._use_embeddings:
                embedded = await self._store_with_embeddings(new_chunks)
            else:
                embedded = await self._store_locally(new_chunks)
            
            # Mark as committed
            self._committed_entries.add(entry_key)
            
            commit_time = time.time() - start_time
            logger.info(
                "CheckpointMemoryCommitter: committed %d chunks (%d skipped) in %.2fs "
                "(session=%s, entry=%d)",
                embedded, skipped, commit_time, synthesis.session_id, ledger_entry.entry_index,
            )
            
            return MemoryCommitResult(
                session_id=synthesis.session_id,
                entry_index=ledger_entry.entry_index,
                chunks_created=len(chunks),
                chunks_embedded=embedded,
                chunks_skipped=skipped,
                embedding_model=self._get_embedding_model(),
                storage_backend=self._get_storage_backend(),
                total_tokens=total_tokens,
                commit_time_seconds=commit_time,
                success=True,
            )
            
        except Exception as exc:
            logger.error(
                "CheckpointMemoryCommitter: commit failed for session %s: %s",
                synthesis.session_id, exc,
            )
            return MemoryCommitResult(
                session_id=synthesis.session_id,
                entry_index=ledger_entry.entry_index,
                chunks_created=0,
                chunks_embedded=0,
                chunks_skipped=0,
                embedding_model="error",
                storage_backend="error",
                total_tokens=0,
                commit_time_seconds=time.time() - start_time,
                success=False,
                error=str(exc),
            )
    
    async def commit_full_ledger(
        self,
        ledger: "ProjectLedger",
        meta_plan: Optional[Any] = None,
    ) -> List[MemoryCommitResult]:
        """
        Commit all entries in a ProjectLedger to memory.
        
        Useful for bootstrapping memory from an existing ledger.
        Idempotent — skips already-committed entries.
        
        Parameters
        ----------
        ledger : ProjectLedger
            The ledger to commit.
        meta_plan : MetaPlan, optional
            The meta plan for milestone context.
        
        Returns
        -------
        List[MemoryCommitResult]
            Results for each committed entry.
        """
        results: List[MemoryCommitResult] = []
        
        # Ensure ledger is loaded
        ledger.load()
        
        # Create synthetic SynthesisResults from ledger entries
        for entry in ledger.entries:
            # Build a SynthesisResult from the ledger entry
            synthesis = self._synthesis_from_entry(entry)
            
            result = await self.commit_checkpoint(
                synthesis=synthesis,
                ledger_entry=entry,
                meta_plan=meta_plan,
            )
            results.append(result)
        
        logger.info(
            "CheckpointMemoryCommitter: committed %d/%d ledger entries",
            sum(1 for r in results if r.success),
            len(results),
        )
        return results
    
    async def search_memory(
        self,
        query: str,
        top_k: int = 5,
        session_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[MemorySearchResult]:
        """
        Semantically search long-term checkpoint memory.
        
        Returns ranked results with source attribution.
        
        Parameters
        ----------
        query : str
            The search query.
        top_k : int
            Maximum results to return (default: 5).
        session_filter : str, optional
            Filter to a specific session.
        min_score : float
            Minimum similarity score (default: 0.0).
        
        Returns
        -------
        List[MemorySearchResult]
            Ranked search results.
        """
        if self._use_embeddings:
            return await self._search_with_embeddings(
                query=query,
                top_k=top_k,
                session_filter=session_filter,
                min_score=min_score,
            )
        else:
            return await self._search_locally(
                query=query,
                top_k=top_k,
                session_filter=session_filter,
                min_score=min_score,
            )
    
    async def get_onboarding_context(
        self,
        query: str,
        max_chunks: int = 10,
        max_chars: int = 4000,
    ) -> str:
        """
        Build an onboarding context string for a new agent joining mid-project.
        
        Searches memory for the most relevant chunks to the query,
        then formats them as a coherent context block.
        More useful than the ledger's to_onboarding_context() because
        it's semantic — the agent gets context relevant to their specific task.
        
        Parameters
        ----------
        query : str
            What the agent is working on.
        max_chunks : int
            Maximum chunks to include (default: 10).
        max_chars : int
            Maximum total characters (default: 4000).
        
        Returns
        -------
        str
            Formatted context string.
        """
        results = await self.search_memory(query=query, top_k=max_chunks)
        
        if not results:
            return "*(No relevant checkpoint history found)*\n"
        
        lines: List[str] = [
            "## Relevant Checkpoint History\n",
            f"*Found {len(results)} relevant checkpoint(s):*\n",
        ]
        
        total_chars = 0
        for result in results:
            chunk = result.chunk
            section_info = f" > {chunk.metadata.get('section_title', 'General')}" if chunk.metadata.get('section_title') else ""
            
            entry_line = (
                f"### Session {chunk.session_id} — Checkpoint {chunk.entry_index}{section_info}\n"
                f"*Relevance: {result.score:.2f} | Agents: {', '.join(result.agents_involved) or 'unknown'}*\n\n"
            )
            
            # Truncate chunk content if needed
            content_budget = max_chars - total_chars - len(entry_line) - 50
            if content_budget <= 0:
                break
            
            content = chunk.content
            if len(content) > content_budget:
                content = content[:content_budget] + "..."
            
            block = f"{entry_line}{content}\n\n---\n\n"
            total_chars += len(block)
            lines.append(block)
        
        return "".join(lines)
    
    async def status(self) -> Dict[str, Any]:
        """
        Return memory system status: chunks committed, storage backend, etc.
        
        Returns
        -------
        Dict[str, Any]
            Status information.
        """
        if self._use_embeddings:
            return {
                "mode": "embeddings",
                "collection_name": self._collection_name,
                "committed_entries": len(self._committed_entries),
                "embedding_model": self._get_embedding_model(),
            }
        elif self._local_store:
            stats = await self._local_store.stats()
            return {
                "mode": "local_fallback",
                "committed_entries": len(self._committed_entries),
                **stats,
            }
        else:
            return {
                "mode": "uninitialized",
                "committed_entries": len(self._committed_entries),
            }
    
    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    
    def _build_chunk_metadata(
        self,
        synthesis: "SynthesisResult",
        ledger_entry: "LedgerEntry",
        meta_plan: Optional[Any],
    ) -> Dict[str, Any]:
        """Build rich metadata for chunks."""
        # Handle timestamp - may be datetime or string
        ts = synthesis.timestamp
        if isinstance(ts, datetime):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)
        
        metadata: Dict[str, Any] = {
            "session_id": synthesis.session_id,
            "entry_index": ledger_entry.entry_index,
            "timestamp": ts_str,
            "agents_involved": synthesis.agents_involved,
            "pivot_count": synthesis.pivot_count,
            "whiteboard_entry_count": synthesis.whiteboard_entry_count,
            "llm_assisted": synthesis.llm_assisted,
            "content_hash": ledger_entry.content_hash,
            "chain_hash": ledger_entry.chain_hash,
        }
        
        # Add milestone references if meta_plan available
        if meta_plan and hasattr(meta_plan, "milestones"):
            milestone_titles = [m.title for m in meta_plan.milestones]
            metadata["milestones"] = milestone_titles
            
            # Current milestone if available
            if hasattr(meta_plan, "current_milestone"):
                metadata["current_milestone"] = meta_plan.current_milestone
        
        return metadata
    
    def _synthesis_from_entry(self, entry: "LedgerEntry") -> "SynthesisResult":
        """Create a SynthesisResult from a LedgerEntry."""
        # Import here to avoid circular import
        from .reconstructor import SynthesisResult
        
        return SynthesisResult(
            session_id=entry.session_id,
            narrative=entry.content,
            timestamp=entry.timestamp,
            whiteboard_entry_count=entry.whiteboard_entry_count,
            agents_involved=entry.agents_involved,
            pivot_count=0,  # Not stored in ledger entry
            llm_assisted=entry.llm_assisted,
        )
    
    async def _chunk_exists(self, content_hash: str) -> bool:
        """Check if a chunk with this content hash already exists."""
        if self._use_embeddings:
            # For real embeddings, we'd need to query the vector store
            # For now, rely on the committed entries tracking
            return False
        elif self._local_store:
            return await self._local_store.chunk_exists(content_hash)
        return False
    
    async def _store_with_embeddings(self, chunks: List[MemoryChunk]) -> int:
        """Store chunks using the embedding pipeline and vector store."""
        stored = 0
        
        for chunk in chunks:
            try:
                # Use embedding pipeline to process and store
                if hasattr(self._embedding_pipeline, "process_content"):
                    result = await self._embedding_pipeline.process_content(
                        text=chunk.content,
                        content_id=chunk.chunk_id,
                        metadata=chunk.metadata,
                    )
                    if result.success:
                        stored += 1
                    else:
                        logger.warning(
                            "CheckpointMemoryCommitter: failed to embed chunk %s: %s",
                            chunk.chunk_id, result.errors,
                        )
                elif hasattr(self._vector_store, "store_content_with_embeddings"):
                    # Fallback: direct vector store access (requires pre-computed embeddings)
                    logger.warning(
                        "CheckpointMemoryCommitter: embedding pipeline lacks process_content, "
                        "storing locally instead"
                    )
                    if self._local_store:
                        await self._local_store.store_chunks([chunk])
                        stored += 1
            except Exception as exc:
                logger.error(
                    "CheckpointMemoryCommitter: error storing chunk %s: %s",
                    chunk.chunk_id, exc,
                )
        
        return stored
    
    async def _store_locally(self, chunks: List[MemoryChunk]) -> int:
        """Store chunks using the local fallback store."""
        if self._local_store:
            return await self._local_store.store_chunks(chunks)
        return 0
    
    async def _search_with_embeddings(
        self,
        query: str,
        top_k: int,
        session_filter: Optional[str],
        min_score: float,
    ) -> List[MemorySearchResult]:
        """Search using the embedding pipeline."""
        results: List[MemorySearchResult] = []
        
        try:
            if hasattr(self._embedding_pipeline, "search_similar_content"):
                matches = await self._embedding_pipeline.search_similar_content(
                    query=query,
                    top_k=top_k,
                )
                
                for match in matches:
                    # Convert match to MemorySearchResult
                    chunk = self._match_to_chunk(match)
                    if chunk:
                        score = match.get("similarity_score", 0.0)
                        if score >= min_score:
                            if session_filter is None or chunk.session_id == session_filter:
                                results.append(MemorySearchResult(
                                    chunk=chunk,
                                    score=score,
                                    session_id=chunk.session_id,
                                    entry_index=chunk.entry_index,
                                    section_title=chunk.metadata.get("section_title"),
                                    agents_involved=chunk.metadata.get("agents_involved", []),
                                    timestamp=chunk.timestamp,
                                ))
        except Exception as exc:
            logger.error("CheckpointMemoryCommitter: embedding search failed: %s", exc)
        
        return results
    
    def _match_to_chunk(self, match: Dict[str, Any]) -> Optional[MemoryChunk]:
        """Convert a search match to a MemoryChunk."""
        metadata = match.get("metadata", {})
        content = match.get("text", "")
        
        if not content:
            return None
        
        return MemoryChunk(
            chunk_id=match.get("id", "unknown"),
            session_id=metadata.get("session_id", "unknown"),
            entry_index=metadata.get("entry_index", 0),
            chunk_index=metadata.get("chunk_index", 0),
            content=content,
            metadata=metadata,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            timestamp=datetime.fromisoformat(metadata["timestamp"]) if "timestamp" in metadata else datetime.now(timezone.utc),
        )
    
    async def _search_locally(
        self,
        query: str,
        top_k: int,
        session_filter: Optional[str],
        min_score: float,
    ) -> List[MemorySearchResult]:
        """Search using the local TF-IDF index."""
        if not self._local_store:
            return []
        
        matches = await self._local_store.search(query=query, top_k=top_k * 2)
        
        results: List[MemorySearchResult] = []
        for chunk, score in matches:
            if score < min_score:
                continue
            if session_filter is not None and chunk.session_id != session_filter:
                continue
            
            results.append(MemorySearchResult(
                chunk=chunk,
                score=score,
                session_id=chunk.session_id,
                entry_index=chunk.entry_index,
                section_title=chunk.metadata.get("section_title"),
                agents_involved=chunk.metadata.get("agents_involved", []),
                timestamp=chunk.timestamp,
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _get_embedding_model(self) -> str:
        """Get the name of the embedding model being used."""
        if self._use_embeddings and hasattr(self._embedding_pipeline, "config"):
            config = self._embedding_pipeline.config
            if hasattr(config, "embedding_model"):
                provider = getattr(config, "preferred_embedding_provider", "unknown")
                model = config.embedding_model
                return f"{provider}/{model}"
        return "local_tfidf"
    
    def _get_storage_backend(self) -> str:
        """Get the name of the storage backend being used."""
        if self._use_embeddings:
            if self._vector_store:
                return type(self._vector_store).__name__
            return "embedding_pipeline"
        return "local"


# ======================================================================
# Integration Documentation
# ======================================================================

# Integration with the checkpoint lifecycle:
#
# In live_scribe.py, after CheckpointLifecycleManager.initiate_checkpoint():
#
#     if result.success and result.synthesis and result.ledger_entry:
#         commit_result = await memory_committer.commit_checkpoint(
#             synthesis=result.synthesis,
#             ledger_entry=result.ledger_entry,
#             meta_plan=self._plan,
#         )
#
# For agent onboarding (in get_agent_context()):
#
#     context = await memory_committer.get_onboarding_context(
#         query=f"Work related to {checkpoint.title}",
#         max_chunks=10,
#     )
#
# For bootstrapping from existing ledger:
#
#     results = await memory_committer.commit_full_ledger(ledger)


# ======================================================================
# Import additions for synthesis/__init__.py
# ======================================================================

# Add these imports to prsm/compute/nwtn/synthesis/__init__.py:
#
# from .memory_commit import (
#     MemoryChunk,
#     MemoryCommitResult,
#     MemorySearchResult,
#     NarrativeChunker,
#     LocalMemoryStore,
#     CheckpointMemoryCommitter,
# )
#
# And add to __all__:
#     # Memory commit
#     "MemoryChunk",
#     "MemoryCommitResult",
#     "MemorySearchResult",
#     "NarrativeChunker",
#     "LocalMemoryStore",
#     "CheckpointMemoryCommitter",

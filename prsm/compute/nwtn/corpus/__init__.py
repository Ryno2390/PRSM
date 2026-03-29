"""NWTN Corpus — codebase vector index and semantic search."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from .code_chunker import CodeChunker, CodeChunk
from .code_index import CodebaseIndex, SearchResult

_default_index: Optional[CodebaseIndex] = None

def _find_repo_root() -> str:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return str(parent)
    return str(Path(__file__).resolve().parents[4])

async def get_codebase_index(repo_root: str = None) -> CodebaseIndex:
    global _default_index
    if _default_index is None:
        root = repo_root or _find_repo_root()
        _default_index = CodebaseIndex(root)
        if not _default_index.is_built():
            await _default_index.build()
    return _default_index

async def search_codebase(query: str, top_k: int = 5, symbol_type: str = None) -> List[SearchResult]:
    index = await get_codebase_index()
    return await index.search(query, top_k=top_k, symbol_type=symbol_type)

__all__ = ["CodeChunker", "CodeChunk", "CodebaseIndex", "SearchResult", "get_codebase_index", "search_codebase"]

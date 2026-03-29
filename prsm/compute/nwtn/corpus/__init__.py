"""
Codebase Vector Index Module

Provides persistent semantic search over PRSM's Python source code.
"""

from .code_index import CodebaseIndex, SearchResult
from .code_chunker import CodeChunker, CodeChunk

import os
from pathlib import Path
from typing import Optional, List

# Singleton index instance
_default_index: Optional[CodebaseIndex] = None


def _find_repo_root() -> str:
    """
    Walk up from this file's location to find the repo root (has pyproject.toml).
    
    Returns:
        Path to the repository root directory
    """
    current = Path(__file__).resolve()
    
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    
    # Fallback to current working directory
    return os.getcwd()


async def get_codebase_index(repo_root: Optional[str] = None) -> CodebaseIndex:
    """
    Get or lazily initialize the singleton codebase index.
    
    Args:
        repo_root: Optional path to repository root (auto-detected if not provided)
        
    Returns:
        Initialized CodebaseIndex instance
    """
    global _default_index
    
    if _default_index is None:
        repo_root = repo_root or _find_repo_root()
        _default_index = CodebaseIndex(repo_root)
        
        if not _default_index.is_built():
            await _default_index.build()
    
    return _default_index


async def search_codebase(
    query: str,
    top_k: int = 5,
    symbol_type: Optional[str] = None
) -> List[SearchResult]:
    """
    Convenience function: search the codebase index.
    
    Args:
        query: Natural language search query
        top_k: Maximum number of results to return
        symbol_type: Optional filter: "class" or "function"
        
    Returns:
        List of SearchResult objects sorted by relevance
        
    Example:
        results = await search_codebase("token payment handling")
        for r in results:
            print(f"{r.filepath}:{r.start_line} - {r.symbol_name}")
    """
    index = await get_codebase_index()
    return await index.search(query, top_k=top_k, symbol_type=symbol_type)


__all__ = [
    "CodebaseIndex",
    "SearchResult",
    "CodeChunker",
    "CodeChunk",
    "get_codebase_index",
    "search_codebase",
]

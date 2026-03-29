"""
Code Chunker - Parses Python source files into searchable chunks.

Each chunk represents one class or top-level function, extracted using
Python's ast module for reliable parsing.
"""

import ast
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CodeChunk:
    """A searchable chunk of code representing a single class or function."""
    
    chunk_id: str          # sha256 of filepath+name
    filepath: str          # relative to repo root
    symbol_name: str       # class or function name
    symbol_type: str       # "class" or "function"
    start_line: int
    end_line: int
    source: str            # raw source text of the symbol
    docstring: str         # extracted docstring or ""
    module_path: str       # e.g. "prsm.compute.nwtn.session"


class CodeChunker:
    """
    Parses Python source files into searchable chunks.
    
    Each chunk represents one class or top-level function, making it
    easy to search and retrieve relevant code snippets.
    """
    
    def __init__(self, repo_root: str):
        """
        Initialize the chunker.
        
        Args:
            repo_root: Path to the repository root directory
        """
        self.repo_root = Path(repo_root).resolve()
    
    def _compute_chunk_id(self, filepath: str, symbol_name: str, start_line: int) -> str:
        """Compute a unique chunk ID based on filepath, symbol name, and line number."""
        data = f"{filepath}:{symbol_name}:{start_line}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def _get_module_path(self, filepath: str) -> str:
        """Convert filepath to Python module path (e.g., 'prsm.compute.nwtn.session')."""
        rel_path = Path(filepath)
        if rel_path.is_absolute():
            try:
                rel_path = rel_path.relative_to(self.repo_root)
            except ValueError:
                pass
        
        # Remove .py extension and convert path separators to dots
        parts = list(rel_path.parts)
        if parts and parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        
        return '.'.join(parts)
    
    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """Extract source text for an AST node."""
        lines = source.splitlines()
        
        # Handle nodes with line numbers
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            start = node.lineno - 1  # 0-indexed
            end = node.end_lineno  # end_lineno is inclusive, so we use it directly
            return '\n'.join(lines[start:end])
        
        return ""
    
    def chunk_file(self, filepath: str) -> List[CodeChunk]:
        """
        Parse a .py file and return one CodeChunk per class/function.
        
        Uses Python's ast module for reliable parsing. Extracts docstrings
        with ast.get_docstring().
        
        Args:
            filepath: Path to the Python file (absolute or relative to repo root)
            
        Returns:
            List of CodeChunk objects, one per top-level class or function
        """
        # Resolve path
        path = Path(filepath)
        if not path.is_absolute():
            path = self.repo_root / path
        
        path = path.resolve()
        
        # Check if file exists
        if not path.exists():
            return []
        
        # Check if it's a Python file
        if not str(path).endswith('.py'):
            return []
        
        # Read source
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            return []
        
        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        
        # Get relative path for chunk
        try:
            rel_path = str(path.relative_to(self.repo_root))
        except ValueError:
            rel_path = str(path)
        
        module_path = self._get_module_path(rel_path)
        chunks: List[CodeChunk] = []
        
        # Extract top-level classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                # Determine symbol type
                if isinstance(node, ast.ClassDef):
                    symbol_type = "class"
                else:
                    symbol_type = "function"
                
                # Extract source
                symbol_source = self._get_source_segment(source, node)
                
                # Extract docstring
                docstring = ast.get_docstring(node) or ""
                
                # Create chunk
                chunk = CodeChunk(
                    chunk_id=self._compute_chunk_id(rel_path, node.name, node.lineno),
                    filepath=rel_path,
                    symbol_name=node.name,
                    symbol_type=symbol_type,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    source=symbol_source,
                    docstring=docstring,
                    module_path=module_path
                )
                chunks.append(chunk)
        
        return chunks
    
    def chunk_repo(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[CodeChunk]:
        """
        Walk the repo and chunk all .py files.
        
        Default exclude: __pycache__, .venv, migrations, tests, scripts
        
        Args:
            include_patterns: Optional list of glob patterns for files to include
            exclude_patterns: Optional list of glob patterns for files to exclude
            
        Returns:
            List of all CodeChunk objects from all matching files
        """
        # Default exclude patterns
        default_excludes = [
            '__pycache__',
            '.venv',
            'venv',
            'migrations',
            'tests',
            'scripts',
            '.git',
            'node_modules',
            '*.pyc',
            '*.pyo',
        ]
        
        exclude = exclude_patterns or default_excludes
        
        all_chunks: List[CodeChunk] = []
        
        # Walk the repo
        for root, dirs, files in os.walk(self.repo_root):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(
                self._match_pattern(d, pattern) for pattern in exclude
            )]
            
            # Process Python files
            for filename in files:
                if not filename.endswith('.py'):
                    continue
                
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, self.repo_root)
                
                # Check exclude patterns
                if any(self._match_pattern(rel_path, pattern) for pattern in exclude):
                    continue
                
                # Check include patterns (if specified)
                if include_patterns:
                    if not any(self._match_pattern(rel_path, pattern) for pattern in include_patterns):
                        continue
                
                # Chunk the file
                chunks = self.chunk_file(filepath)
                all_chunks.extend(chunks)
        
        return all_chunks
    
    def _match_pattern(self, path: str, pattern: str) -> bool:
        """
        Check if a path matches a glob-like pattern.
        
        Simple pattern matching that supports:
        - Exact match
        - * wildcard
        - Directory prefixes (e.g., 'tests/' matches 'tests/foo.py')
        """
        import fnmatch
        
        # Normalize path separators
        path = path.replace('\\', '/')
        pattern = pattern.replace('\\', '/')
        
        # Check for directory prefix match
        if pattern.endswith('/'):
            return path.startswith(pattern) or path.startswith(pattern.rstrip('/'))
        
        # Use fnmatch for glob patterns
        if '*' in pattern:
            return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern)
        
        # Exact match or prefix
        return path == pattern or path.startswith(pattern + '/') or os.path.basename(path) == pattern

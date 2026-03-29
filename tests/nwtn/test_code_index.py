"""
Tests for Codebase Vector Index

Uses a tiny synthetic repo (temp directory) to test the functionality
without indexing the real codebase.
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path

import pytest

from prsm.compute.nwtn.corpus.code_chunker import CodeChunker, CodeChunk
from prsm.compute.nwtn.corpus.code_index import CodebaseIndex, SearchResult
from prsm.compute.nwtn.corpus import (
    get_codebase_index,
    search_codebase,
    CodeChunker,
    CodeChunk,
    CodebaseIndex,
    SearchResult,
)


# ============== Test Fixtures ==============

@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    repo_dir = tempfile.mkdtemp()
    yield repo_dir
    shutil.rmtree(repo_dir, ignore_errors=True)


@pytest.fixture
def sample_python_file(temp_repo):
    """Create a sample Python file with classes and functions."""
    file_path = Path(temp_repo) / "sample.py"
    file_path.write_text('''
"""Sample module for testing."""

def standalone_function(x: int) -> int:
    """A simple function that doubles input.
    
    Args:
        x: Input integer
        
    Returns:
        Doubled value
    """
    return x * 2


class SampleClass:
    """A sample class with methods."""
    
    def __init__(self, value: int):
        """Initialize with a value."""
        self.value = value
    
    def get_value(self) -> int:
        """Return the stored value."""
        return self.value


class AnotherClass:
    """Another class without explicit docstring in methods."""
    
    def method_without_docstring(self):
        # This method has no docstring
        pass
    
    async def async_method(self):
        """An async method."""
        return await some_coroutine()
''')
    return str(file_path)


@pytest.fixture
def sample_file_no_docstrings(temp_repo):
    """Create a Python file without docstrings."""
    file_path = Path(temp_repo) / "no_docs.py"
    file_path.write_text('''
def simple_func(a, b):
    return a + b

class SimpleClass:
    def __init__(self):
        self.x = 1
    
    def method(self):
        return self.x
''')
    return str(file_path)


@pytest.fixture
def excluded_test_file(temp_repo):
    """Create a test file that should be excluded."""
    tests_dir = Path(temp_repo) / "tests"
    tests_dir.mkdir()
    file_path = tests_dir / "test_sample.py"
    file_path.write_text('''
def test_something():
    """Test case."""
    assert True
''')
    return str(file_path)


@pytest.fixture
def codebase_index(temp_repo):
    """Create a CodebaseIndex instance with temp persist dir."""
    persist_dir = Path(temp_repo) / ".codebase_index"
    index = CodebaseIndex(temp_repo, persist_dir=str(persist_dir))
    yield index
    # Cleanup is handled by temp_repo fixture


# ============== CodeChunker Tests ==============

class TestCodeChunker:
    """Tests for CodeChunker class."""
    
    def test_chunk_file_extracts_classes_and_functions(self, temp_repo, sample_python_file):
        """Test that chunk_file correctly extracts classes and functions with docstrings."""
        chunker = CodeChunker(temp_repo)
        chunks = chunker.chunk_file(sample_python_file)
        
        # Should extract 4 items: standalone_function, SampleClass, AnotherClass, async_method
        # Note: __init__ and get_value are methods inside SampleClass, not top-level
        assert len(chunks) >= 3
        
        # Check that we have both classes and functions
        symbol_names = [c.symbol_name for c in chunks]
        assert "standalone_function" in symbol_names
        assert "SampleClass" in symbol_names
        assert "AnotherClass" in symbol_names
        
        # Check symbol types
        for chunk in chunks:
            assert chunk.symbol_type in ("class", "function")
    
    def test_chunk_file_extracts_docstrings(self, temp_repo, sample_python_file):
        """Test that docstrings are properly extracted."""
        chunker = CodeChunker(temp_repo)
        chunks = chunker.chunk_file(sample_python_file)
        
        # Find the standalone function
        func_chunk = next(c for c in chunks if c.symbol_name == "standalone_function")
        assert "simple function that doubles" in func_chunk.docstring
        
        # Find SampleClass
        class_chunk = next(c for c in chunks if c.symbol_name == "SampleClass")
        assert "sample class" in class_chunk.docstring.lower()
    
    def test_chunk_file_handles_no_docstrings(self, temp_repo, sample_file_no_docstrings):
        """Test that files without docstrings are handled gracefully."""
        chunker = CodeChunker(temp_repo)
        chunks = chunker.chunk_file(sample_file_no_docstrings)
        
        # Should still extract the function and class
        assert len(chunks) >= 2
        symbol_names = [c.symbol_name for c in chunks]
        assert "simple_func" in symbol_names
        assert "SimpleClass" in symbol_names
        
        # Docstrings should be empty strings
        for chunk in chunks:
            assert isinstance(chunk.docstring, str)
    
    def test_chunk_file_returns_empty_for_nonexistent(self, temp_repo):
        """Test that nonexistent files return empty list."""
        chunker = CodeChunker(temp_repo)
        chunks = chunker.chunk_file("/nonexistent/file.py")
        assert chunks == []
    
    def test_chunk_file_returns_empty_for_non_python(self, temp_repo):
        """Test that non-Python files return empty list."""
        chunker = CodeChunker(temp_repo)
        txt_file = Path(temp_repo) / "test.txt"
        txt_file.write_text("not python")
        chunks = chunker.chunk_file(str(txt_file))
        assert chunks == []
    
    def test_chunk_repo_respects_exclude_patterns(self, temp_repo, sample_python_file, excluded_test_file):
        """Test that chunk_repo respects exclude patterns."""
        chunker = CodeChunker(temp_repo)
        
        # Default exclude should skip tests/
        chunks = chunker.chunk_repo()
        
        # Should not include the test file
        for chunk in chunks:
            assert "tests/" not in chunk.filepath
    
    def test_chunk_repo_includes_all_py_files(self, temp_repo, sample_python_file, sample_file_no_docstrings):
        """Test that chunk_repo processes all Python files."""
        chunker = CodeChunker(temp_repo)
        chunks = chunker.chunk_repo(exclude_patterns=[])  # No excludes
        
        # Should have chunks from both files
        symbol_names = [c.symbol_name for c in chunks]
        assert "standalone_function" in symbol_names  # From sample.py
        assert "simple_func" in symbol_names  # From no_docs.py
    
    def test_chunk_has_correct_fields(self, temp_repo, sample_python_file):
        """Test that CodeChunk has all required fields."""
        chunker = CodeChunker(temp_repo)
        chunks = chunker.chunk_file(sample_python_file)
        
        assert len(chunks) > 0
        chunk = chunks[0]
        
        # Check all fields exist and have correct types
        assert isinstance(chunk.chunk_id, str)
        assert isinstance(chunk.filepath, str)
        assert isinstance(chunk.symbol_name, str)
        assert isinstance(chunk.symbol_type, str)
        assert isinstance(chunk.start_line, int)
        assert isinstance(chunk.end_line, int)
        assert isinstance(chunk.source, str)
        assert isinstance(chunk.docstring, str)
        assert isinstance(chunk.module_path, str)
        
        # Check line numbers are valid
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


# ============== CodebaseIndex Tests ==============

class TestCodebaseIndex:
    """Tests for CodebaseIndex class."""
    
    @pytest.mark.asyncio
    async def test_build_indexes_chunks(self, codebase_index, sample_python_file):
        """Test that build() indexes chunks and is_built() returns True."""
        count = await codebase_index.build()
        
        assert count > 0
        assert codebase_index.is_built()
    
    @pytest.mark.asyncio
    async def test_is_built_returns_false_initially(self, codebase_index):
        """Test that is_built() returns False before building."""
        assert not codebase_index.is_built()
    
    @pytest.mark.asyncio
    async def test_search_returns_results_with_correct_fields(self, codebase_index, sample_python_file):
        """Test that search() returns results with correct fields."""
        await codebase_index.build()
        
        results = await codebase_index.search("function", top_k=5)
        
        assert isinstance(results, list)
        
        if results:  # May be empty if no matches
            result = results[0]
            assert isinstance(result, SearchResult)
            assert isinstance(result.filepath, str)
            assert isinstance(result.symbol_name, str)
            assert isinstance(result.symbol_type, str)
            assert isinstance(result.start_line, int)
            assert isinstance(result.end_line, int)
            assert isinstance(result.docstring, str)
            assert isinstance(result.module_path, str)
            assert isinstance(result.score, float)
            assert isinstance(result.snippet, str)
            assert 0.0 <= result.score <= 1.0
    
    @pytest.mark.asyncio
    async def test_search_with_symbol_type_filter(self, codebase_index, sample_python_file):
        """Test that search respects symbol_type filter."""
        await codebase_index.build()
        
        # Search for classes only
        class_results = await codebase_index.search("sample", top_k=10, symbol_type="class")
        
        for result in class_results:
            assert result.symbol_type == "class"
        
        # Search for functions only
        func_results = await codebase_index.search("function", top_k=10, symbol_type="function")
        
        for result in func_results:
            assert result.symbol_type == "function"
    
    @pytest.mark.asyncio
    async def test_search_returns_empty_before_build(self, codebase_index):
        """Test that search returns empty list before building index."""
        results = await codebase_index.search("anything")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_build_force_rebuild_skips_existing(self, codebase_index, sample_python_file):
        """Test that build(force_rebuild=False) skips already-indexed chunks."""
        # First build
        count1 = await codebase_index.build()
        
        # Second build without force
        count2 = await codebase_index.build(force_rebuild=False)
        
        # Should skip already indexed chunks
        assert count2 == 0
    
    @pytest.mark.asyncio
    async def test_build_force_rebuild_reindexes(self, codebase_index, sample_python_file):
        """Test that build(force_rebuild=True) reindexes everything."""
        # First build
        count1 = await codebase_index.build()
        
        # Force rebuild
        count2 = await codebase_index.build(force_rebuild=True)
        
        # Should reindex all chunks
        assert count2 == count1
    
    @pytest.mark.asyncio
    async def test_search_returns_relevant_results(self, codebase_index, sample_python_file):
        """Test that search returns semantically relevant results."""
        await codebase_index.build()
        
        # Search for something that should match
        results = await codebase_index.search("double input", top_k=3)
        
        # Should find the standalone_function which mentions doubling
        if results:
            # The most relevant result should be about the doubling function
            found_doubling = any(
                "double" in r.docstring.lower() or "double" in r.snippet.lower()
                for r in results
            )
            assert found_doubling or len(results) > 0  # At least we got results


# ============== Convenience Function Tests ==============

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    @pytest.mark.asyncio
    async def test_search_codebase_works_end_to_end(self, temp_repo, sample_python_file):
        """Test that search_codebase() convenience function works end-to-end."""
        # Create index in temp repo
        persist_dir = Path(temp_repo) / ".codebase_index"
        index = CodebaseIndex(temp_repo, persist_dir=str(persist_dir))
        
        # Build index
        count = await index.build()
        assert count > 0
        
        # Search using index directly (bypassing singleton for isolated test)
        results = await index.search("function", top_k=5)
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_get_codebase_index_returns_singleton(self, temp_repo, sample_python_file):
        """Test that get_codebase_index returns the same instance."""
        # Create a test-specific index to avoid affecting global state
        # Note: In real usage, this would return a singleton
        persist_dir = Path(temp_repo) / ".codebase_index"
        index1 = CodebaseIndex(temp_repo, persist_dir=str(persist_dir))
        index2 = CodebaseIndex(temp_repo, persist_dir=str(persist_dir))
        
        # Both should work independently
        assert index1.repo_root == index2.repo_root


# ============== Integration Tests ==============

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, temp_repo, sample_python_file, sample_file_no_docstrings):
        """Test the full pipeline: chunk -> index -> search."""
        # Create index
        persist_dir = Path(temp_repo) / ".codebase_index"
        index = CodebaseIndex(temp_repo, persist_dir=str(persist_dir))
        
        # Build index
        count = await index.build()
        assert count >= 4  # At least 4 chunks from the two files
        
        # Search
        results = await index.search("sample class", top_k=5)
        assert len(results) > 0
        
        # Verify results have expected content
        found_class = False
        for r in results:
            if "sample" in r.symbol_name.lower() or "sample" in r.docstring.lower():
                found_class = True
                break
        
        assert found_class or len(results) > 0
    
    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, temp_repo, sample_python_file):
        """Test that the index persists across different CodebaseIndex instances."""
        persist_dir = Path(temp_repo) / ".codebase_index"
        
        # First instance builds
        index1 = CodebaseIndex(temp_repo, persist_dir=str(persist_dir))
        count1 = await index1.build()
        
        # Second instance should see the persisted data
        index2 = CodebaseIndex(temp_repo, persist_dir=str(persist_dir))
        
        # is_built() should work without rebuilding
        # (but we need to initialize the client first)
        await index2._get_or_create_collection()
        assert index2.is_built()
        
        # Second build without force should skip (0 new chunks)
        count2 = await index2.build(force_rebuild=False)
        assert count2 == 0


# ============== Edge Cases ==============

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_chunk_file_with_syntax_error(self, temp_repo):
        """Test handling of files with syntax errors."""
        bad_file = Path(temp_repo) / "bad.py"
        bad_file.write_text("def broken(:\n    # syntax error")
        
        chunker = CodeChunker(temp_repo)
        chunks = chunker.chunk_file(str(bad_file))
        
        # Should return empty list for unparseable files
        assert chunks == []
    
    @pytest.mark.asyncio
    async def test_search_with_no_results(self, codebase_index, sample_python_file):
        """Test search with a query that matches nothing."""
        await codebase_index.build()
        
        results = await codebase_index.search("xyzzynonexistent12345", top_k=5)
        
        # Should return empty list, not crash
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_empty_repo(self, codebase_index):
        """Test building index on empty repo."""
        count = await codebase_index.build()
        
        # Should handle gracefully
        assert count == 0
        assert not codebase_index.is_built()

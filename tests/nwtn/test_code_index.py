"""Tests for FEAT-20260327-001: codebase vector index."""
import asyncio
import os
import tempfile
import textwrap
from pathlib import Path
import pytest
from prsm.compute.nwtn.corpus.code_chunker import CodeChunker, CodeChunk
from prsm.compute.nwtn.corpus.code_index import CodebaseIndex, SearchResult

SAMPLE_PY = textwrap.dedent('''
    """Sample module."""

    class PaymentHandler:
        """Handles FTNS token payments."""
        
        def process(self, amount: float) -> bool:
            """Process a payment."""
            return True
        
        def refund(self, amount: float) -> bool:
            return False

    def utility_function():
        """A standalone utility."""
        pass

    def no_docstring():
        x = 1
''')

@pytest.fixture
def tmp_repo(tmp_path):
    (tmp_path / "prsm").mkdir()
    f = tmp_path / "prsm" / "payments.py"
    f.write_text(SAMPLE_PY)
    f2 = tmp_path / "prsm" / "other.py"
    f2.write_text("class OtherClass:\n    pass\n")
    return tmp_path

def test_chunk_file_finds_classes(tmp_repo):
    chunker = CodeChunker(str(tmp_repo))
    chunks = chunker.chunk_file(str(tmp_repo / "prsm" / "payments.py"))
    names = [c.symbol_name for c in chunks]
    assert "PaymentHandler" in names

def test_chunk_file_finds_functions(tmp_repo):
    chunker = CodeChunker(str(tmp_repo))
    chunks = chunker.chunk_file(str(tmp_repo / "prsm" / "payments.py"))
    names = [c.symbol_name for c in chunks]
    assert "utility_function" in names

def test_chunk_extracts_docstring(tmp_repo):
    chunker = CodeChunker(str(tmp_repo))
    chunks = chunker.chunk_file(str(tmp_repo / "prsm" / "payments.py"))
    cls = next(c for c in chunks if c.symbol_name == "PaymentHandler")
    assert "FTNS token payments" in cls.docstring

def test_chunk_no_docstring(tmp_repo):
    chunker = CodeChunker(str(tmp_repo))
    chunks = chunker.chunk_file(str(tmp_repo / "prsm" / "payments.py"))
    fn = next((c for c in chunks if c.symbol_name == "no_docstring"), None)
    if fn:
        assert fn.docstring == ""

def test_chunk_symbol_types(tmp_repo):
    chunker = CodeChunker(str(tmp_repo))
    chunks = chunker.chunk_file(str(tmp_repo / "prsm" / "payments.py"))
    types = {c.symbol_name: c.symbol_type for c in chunks}
    assert types.get("PaymentHandler") == "class"
    assert types.get("utility_function") == "function"

def test_chunk_has_line_numbers(tmp_repo):
    chunker = CodeChunker(str(tmp_repo))
    chunks = chunker.chunk_file(str(tmp_repo / "prsm" / "payments.py"))
    for c in chunks:
        assert c.start_line > 0
        assert c.end_line >= c.start_line

def test_chunk_id_is_stable(tmp_repo):
    chunker = CodeChunker(str(tmp_repo))
    chunks1 = chunker.chunk_file(str(tmp_repo / "prsm" / "payments.py"))
    chunks2 = chunker.chunk_file(str(tmp_repo / "prsm" / "payments.py"))
    ids1 = {c.symbol_name: c.chunk_id for c in chunks1}
    ids2 = {c.symbol_name: c.chunk_id for c in chunks2}
    assert ids1 == ids2

def test_chunk_repo_finds_files(tmp_repo):
    chunker = CodeChunker(str(tmp_repo))
    chunks = chunker.chunk_repo()
    files = {c.filepath for c in chunks}
    assert any("payments.py" in f for f in files)

def test_chunk_repo_excludes_patterns(tmp_repo):
    (tmp_repo / "__pycache__").mkdir()
    (tmp_repo / "__pycache__" / "cached.py").write_text("class Cached: pass\n")
    chunker = CodeChunker(str(tmp_repo))
    chunks = chunker.chunk_repo()
    names = [c.symbol_name for c in chunks]
    assert "Cached" not in names

def test_index_build_and_is_built(tmp_repo):
    index = CodebaseIndex(str(tmp_repo), persist_dir=str(tmp_repo / ".idx"))
    assert not index.is_built()
    count = asyncio.run(index.build())
    assert count > 0
    assert index.is_built()

def test_index_search_returns_results(tmp_repo):
    index = CodebaseIndex(str(tmp_repo), persist_dir=str(tmp_repo / ".idx2"))
    asyncio.run(index.build())
    results = asyncio.run(index.search("payment token", top_k=3))
    assert len(results) > 0
    assert isinstance(results[0], SearchResult)

def test_search_result_fields(tmp_repo):
    index = CodebaseIndex(str(tmp_repo), persist_dir=str(tmp_repo / ".idx3"))
    asyncio.run(index.build())
    results = asyncio.run(index.search("payment", top_k=1))
    r = results[0]
    assert r.filepath
    assert r.symbol_name
    assert r.symbol_type in ("class", "function")
    assert r.start_line > 0
    assert 0.0 <= r.score <= 1.0

def test_index_dedup_on_rebuild(tmp_repo):
    index = CodebaseIndex(str(tmp_repo), persist_dir=str(tmp_repo / ".idx4"))
    count1 = asyncio.run(index.build())
    count2 = asyncio.run(index.build())  # second build should skip existing
    assert count2 == 0  # nothing new

def test_force_rebuild(tmp_repo):
    index = CodebaseIndex(str(tmp_repo), persist_dir=str(tmp_repo / ".idx5"))
    count1 = asyncio.run(index.build())
    count2 = asyncio.run(index.build(force_rebuild=True))
    assert count2 == count1  # same count after force rebuild

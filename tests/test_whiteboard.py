"""
Tests for Sub-phase 10.2: Active Whiteboard

Coverage:
  1. Schema validation (WhiteboardEntry, WhiteboardSnapshot)
  2. Store CRUD — write, read, session lifecycle, compressed_state strategies
  3. Monitor — file-change detection, debouncing, BSC integration (mocked)
  4. Query — all query methods, onboarding brief
  5. End-to-end — monitor → BSC (mocked) → store → query full flow
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.compute.nwtn.bsc import (
    BSCPromoter,
    ChunkMetadata,
    FilterDecision,
    KLFilterResult,
    PromotionDecision,
    SurpriseScore,
    DedupResult,
)
from prsm.compute.nwtn.whiteboard import (
    WhiteboardEntry,
    WhiteboardMonitor,
    WhiteboardQuery,
    WhiteboardSnapshot,
    WhiteboardStore,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
async def store():
    """In-memory whiteboard store, opened and closed around each test."""
    s = WhiteboardStore(":memory:")
    await s.open()
    await s.create_session("sess-test")
    yield s
    await s.close()


def _make_decision(
    chunk: str,
    source_agent: str = "agent/coder-20260326",
    session_id: str = "sess-test",
    surprise: float = 0.75,
    similarity: float = 0.10,
) -> PromotionDecision:
    """Create a promoted PromotionDecision for testing."""
    return PromotionDecision(
        promoted=True,
        chunk=chunk,
        metadata=ChunkMetadata(
            source_agent=source_agent,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
        ),
        surprise_score=surprise,
        raw_perplexity=surprise * 100,
        similarity_score=similarity,
        kl_result=KLFilterResult(
            decision=FilterDecision.PROMOTE,
            score=surprise,
            epsilon=0.55,
            reason="high surprise",
        ),
        dedup_result=DedupResult(
            is_redundant=False,
            max_similarity=similarity,
            most_similar_index=None,
            reason="novel",
        ),
        reason="PROMOTED",
    )


def _make_promoter_that_promotes(chunk_passthrough: bool = True) -> BSCPromoter:
    """Return a mocked BSCPromoter that always promotes."""
    promoter = MagicMock(spec=BSCPromoter)
    promoter.warmup = AsyncMock()
    promoter.stats = {"total_processed": 0, "total_promoted": 0, "promotion_rate": 0.0,
                      "whiteboard_entries": 0, "predictor_baseline_perplexity": 45.0,
                      "kl_epsilon": 0.55}

    async def _process(chunk, context, source_agent, session_id, extra_metadata=None):
        if chunk_passthrough:
            return _make_decision(chunk, source_agent=source_agent, session_id=session_id)
        else:
            return PromotionDecision(
                promoted=False, chunk=chunk,
                metadata=ChunkMetadata(source_agent=source_agent, session_id=session_id),
                surprise_score=0.2, raw_perplexity=10.0, similarity_score=0.0,
                kl_result=KLFilterResult(
                    decision=FilterDecision.DISCARD, score=0.2, epsilon=0.55,
                    reason="low surprise"
                ),
                dedup_result=None,
                reason="low surprise",
            )

    promoter.process_chunk = _process
    return promoter


# ======================================================================
# 1. Schema
# ======================================================================

class TestWhiteboardSchema:
    def test_entry_frozen(self):
        entry = WhiteboardEntry(
            session_id="s", source_agent="agent/test",
            chunk="hello", surprise_score=0.7,
            raw_perplexity=60.0, similarity_score=0.1,
        )
        with pytest.raises(Exception):
            entry.chunk = "mutated"  # type: ignore[misc]

    def test_agent_short_strips_prefix(self):
        e = WhiteboardEntry(
            session_id="s", source_agent="agent/security-20260326",
            chunk="x", surprise_score=0.5, raw_perplexity=40.0, similarity_score=0.0,
        )
        assert e.agent_short == "security-20260326"

    def test_agent_short_no_prefix(self):
        e = WhiteboardEntry(
            session_id="s", source_agent="bare-agent",
            chunk="x", surprise_score=0.5, raw_perplexity=40.0, similarity_score=0.0,
        )
        assert e.agent_short == "bare-agent"

    def test_timestamp_short_format(self):
        e = WhiteboardEntry(
            session_id="s", source_agent="a",
            chunk="x", surprise_score=0.5, raw_perplexity=40.0, similarity_score=0.0,
            timestamp=datetime(2026, 3, 26, 9, 15, 30, tzinfo=timezone.utc),
        )
        assert e.timestamp_short == "2026-03-26T09:15:30Z"

    def test_snapshot_entry_count_auto_computed(self):
        entries = [
            WhiteboardEntry(
                session_id="s", source_agent="a",
                chunk=f"chunk {i}", surprise_score=0.7,
                raw_perplexity=60.0, similarity_score=0.1,
            )
            for i in range(5)
        ]
        snap = WhiteboardSnapshot(session_id="s", entries=entries)
        assert snap.entry_count == 5

    def test_snapshot_agents_deduplicates(self):
        entries = [
            WhiteboardEntry(
                session_id="s", source_agent="agent/a",
                chunk="c", surprise_score=0.7, raw_perplexity=60.0, similarity_score=0.1,
            ),
            WhiteboardEntry(
                session_id="s", source_agent="agent/b",
                chunk="d", surprise_score=0.6, raw_perplexity=50.0, similarity_score=0.1,
            ),
            WhiteboardEntry(
                session_id="s", source_agent="agent/a",
                chunk="e", surprise_score=0.8, raw_perplexity=70.0, similarity_score=0.1,
            ),
        ]
        snap = WhiteboardSnapshot(session_id="s", entries=entries)
        assert set(snap.agents) == {"agent/a", "agent/b"}

    def test_snapshot_top_surprise(self):
        scores = [0.3, 0.9, 0.5, 0.8, 0.1]
        entries = [
            WhiteboardEntry(
                session_id="s", source_agent="a",
                chunk=f"c{i}", surprise_score=s,
                raw_perplexity=s * 100, similarity_score=0.0,
            )
            for i, s in enumerate(scores)
        ]
        snap = WhiteboardSnapshot(session_id="s", entries=entries)
        top2 = snap.top_surprise(2)
        assert [e.surprise_score for e in top2] == [0.9, 0.8]


# ======================================================================
# 2. Store
# ======================================================================

class TestWhiteboardStore:
    @pytest.mark.asyncio
    async def test_open_close_idempotent(self):
        s = WhiteboardStore(":memory:")
        await s.open()
        await s.close()
        await s.close()  # second close is a no-op

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with WhiteboardStore(":memory:") as s:
            await s.create_session("s")
            assert await s.entry_count("s") == 0

    @pytest.mark.asyncio
    async def test_write_and_read(self, store):
        d = _make_decision("Auth layer requires PostgreSQL.")
        entry = await store.write(d)

        assert entry.id is not None
        assert entry.chunk == "Auth layer requires PostgreSQL."
        assert entry.source_agent == "agent/coder-20260326"
        assert entry.surprise_score == pytest.approx(0.75)

        # decision.whiteboard_index should be back-filled
        assert d.whiteboard_index == entry.id

    @pytest.mark.asyncio
    async def test_write_rejected_for_non_promoted(self, store):
        d = _make_decision("routine update")
        d = PromotionDecision(
            promoted=False, chunk="routine",
            metadata=ChunkMetadata(source_agent="agent/x", session_id="sess-test"),
            surprise_score=0.2, raw_perplexity=10.0, similarity_score=0.0,
            kl_result=KLFilterResult(
                decision=FilterDecision.DISCARD, score=0.2, epsilon=0.55, reason="low"
            ),
            dedup_result=None,
            reason="discard",
        )
        with pytest.raises(ValueError, match="promoted"):
            await store.write(d)

    @pytest.mark.asyncio
    async def test_multiple_writes_and_get_all(self, store):
        chunks = ["Finding A", "Finding B", "Finding C"]
        for c in chunks:
            await store.write(_make_decision(c))

        entries = await store.get_all("sess-test")
        assert len(entries) == 3
        assert [e.chunk for e in entries] == chunks

    @pytest.mark.asyncio
    async def test_get_recent_ordering(self, store):
        for c in ["old", "middle", "newest"]:
            await store.write(_make_decision(c))

        recent = await store.get_recent("sess-test", 2)
        assert len(recent) == 2
        # Should be ordered timestamp ascending (oldest first in the returned slice)
        assert recent[-1].chunk == "newest"

    @pytest.mark.asyncio
    async def test_get_by_agent_filters_correctly(self, store):
        await store.write(_make_decision("coder chunk", source_agent="agent/coder-20260326"))
        await store.write(_make_decision("security chunk", source_agent="agent/security-20260326"))
        await store.write(_make_decision("coder chunk 2", source_agent="agent/coder-20260326"))

        coder_entries = await store.get_by_agent("sess-test", "agent/coder-20260326")
        assert len(coder_entries) == 2
        assert all(e.source_agent == "agent/coder-20260326" for e in coder_entries)

    @pytest.mark.asyncio
    async def test_get_top_surprise_ordering(self, store):
        scores = [0.6, 0.9, 0.4, 0.8, 0.55]
        for i, s in enumerate(scores):
            await store.write(_make_decision(f"chunk{i}", surprise=s))

        top3 = await store.get_top_surprise("sess-test", 3)
        assert [e.surprise_score for e in top3] == [0.9, 0.8, 0.6]

    @pytest.mark.asyncio
    async def test_entry_count(self, store):
        assert await store.entry_count("sess-test") == 0
        await store.write(_make_decision("a"))
        await store.write(_make_decision("b"))
        assert await store.entry_count("sess-test") == 2

    @pytest.mark.asyncio
    async def test_clear_session(self, store):
        await store.write(_make_decision("x"))
        await store.write(_make_decision("y"))
        n = await store.clear_session("sess-test")
        assert n == 2
        assert await store.entry_count("sess-test") == 0

    @pytest.mark.asyncio
    async def test_compressed_state_empty(self, store):
        state = await store.compressed_state("sess-test")
        assert "empty" in state.lower()

    @pytest.mark.asyncio
    async def test_compressed_state_contains_chunks(self, store):
        await store.write(_make_decision("PostgreSQL migration is required."))
        await store.write(_make_decision("JWT secret has been rotated."))

        state = await store.compressed_state("sess-test")
        assert "PostgreSQL" in state
        assert "JWT" in state

    @pytest.mark.asyncio
    async def test_compressed_state_respects_max_chars(self, store):
        # Write enough content to exceed the limit
        for i in range(20):
            await store.write(_make_decision("x" * 200, surprise=0.7 + i * 0.01))

        state = await store.compressed_state("sess-test", max_chars=500)
        assert len(state) <= 600  # some tolerance for header

    @pytest.mark.asyncio
    async def test_compressed_state_strategy_recent(self, store):
        for i in range(6):
            await store.write(_make_decision(f"entry-{i}", surprise=0.7))

        state = await store.compressed_state("sess-test", strategy="recent")
        assert "Whiteboard" in state

    @pytest.mark.asyncio
    async def test_compressed_state_strategy_top(self, store):
        await store.write(_make_decision("low", surprise=0.56))
        await store.write(_make_decision("high", surprise=0.95))

        state = await store.compressed_state("sess-test", strategy="top")
        # "high" entry appears before "low" in top strategy
        assert state.index("high") < state.index("low")

    @pytest.mark.asyncio
    async def test_snapshot_contains_all_entries(self, store):
        await store.write(_make_decision("a"))
        await store.write(_make_decision("b"))
        snap = await store.snapshot("sess-test")
        assert snap.session_id == "sess-test"
        assert snap.entry_count == 2
        assert {e.chunk for e in snap.entries} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        async with WhiteboardStore(":memory:") as s:
            await s.create_session("s1")
            await s.create_session("s2")
            await s.write(_make_decision("for-s1", session_id="s1"))
            await s.write(_make_decision("for-s2", session_id="s2"))

            assert await s.entry_count("s1") == 1
            assert await s.entry_count("s2") == 1

            s1_entries = await s.get_all("s1")
            assert s1_entries[0].chunk == "for-s1"

    @pytest.mark.asyncio
    async def test_file_backed_store(self, tmp_path):
        db_file = tmp_path / "whiteboard.db"
        async with WhiteboardStore(db_file) as s:
            await s.create_session("sess-file")
            await s.write(_make_decision("persisted", session_id="sess-file"))
            assert await s.entry_count("sess-file") == 1

        # Reopen and verify persistence
        async with WhiteboardStore(db_file) as s:
            assert await s.entry_count("sess-file") == 1
            entries = await s.get_all("sess-file")
            assert entries[0].chunk == "persisted"


# ======================================================================
# 3. Monitor
# ======================================================================

class TestWhiteboardMonitor:
    @pytest.mark.asyncio
    async def test_watch_agent_registration(self, tmp_path, store):
        promoter = _make_promoter_that_promotes()
        monitor = WhiteboardMonitor(promoter=promoter, store=store)
        f = tmp_path / "MEMORY.md"
        f.write_text("")

        monitor.watch_agent(str(f), "agent/coder", "sess-test")
        assert str(f.resolve()) in monitor.watched_paths

    @pytest.mark.asyncio
    async def test_unwatch_removes_agent(self, tmp_path, store):
        promoter = _make_promoter_that_promotes()
        monitor = WhiteboardMonitor(promoter=promoter, store=store)
        f = tmp_path / "MEMORY.md"
        f.write_text("")

        monitor.watch_agent(str(f), "agent/coder", "sess-test")
        monitor.unwatch_agent(str(f))
        assert str(f.resolve()) not in monitor.watched_paths

    @pytest.mark.asyncio
    async def test_read_delta_detects_new_content(self, tmp_path, store):
        promoter = _make_promoter_that_promotes(chunk_passthrough=True)
        monitor = WhiteboardMonitor(
            promoter=promoter, store=store, min_chunk_chars=10
        )
        f = tmp_path / "MEMORY.md"
        f.write_text("")

        monitor.watch_agent(str(f), "agent/coder", "sess-test")
        agent = list(monitor._agents.values())[0]

        # Simulate file append
        f.write_text("A" * 100)
        await monitor._read_delta(agent)

        # Buffer should contain the new content
        assert len(agent.buffer) > 0 or await store.entry_count("sess-test") > 0

    @pytest.mark.asyncio
    async def test_read_delta_does_not_reprocess_old_content(self, tmp_path, store):
        promoter = _make_promoter_that_promotes()
        monitor = WhiteboardMonitor(
            promoter=promoter, store=store, min_chunk_chars=10
        )
        f = tmp_path / "MEMORY.md"
        initial = "Initial content here — old.\n"
        f.write_text(initial)

        monitor.watch_agent(str(f), "agent/coder", "sess-test")
        agent = list(monitor._agents.values())[0]

        # First read: consume initial content
        await monitor._read_delta(agent)
        await monitor._submit_chunk(agent, force=True)
        count_after_first = await store.entry_count("sess-test")

        # Second read with no new content: nothing should change
        await monitor._read_delta(agent)
        await monitor._submit_chunk(agent, force=True)
        count_after_second = await store.entry_count("sess-test")

        assert count_after_second == count_after_first

    @pytest.mark.asyncio
    async def test_file_rotation_resets_offset(self, tmp_path, store):
        promoter = _make_promoter_that_promotes()
        monitor = WhiteboardMonitor(
            promoter=promoter, store=store, min_chunk_chars=10
        )
        f = tmp_path / "MEMORY.md"
        f.write_text("original content " * 5)

        monitor.watch_agent(str(f), "agent/coder", "sess-test")
        agent = list(monitor._agents.values())[0]

        # First read
        await monitor._read_delta(agent)
        assert agent.last_offset > 0
        original_offset = agent.last_offset

        # Simulate rotation: write new file with different inode by replacing
        f.unlink()
        f.write_text("rotated content " * 5)
        agent.last_inode = agent.last_inode + 999  # force mismatch

        await monitor._read_delta(agent)
        # Offset should reset to accommodate new file
        assert agent.last_offset <= len("rotated content " * 5) + 10

    @pytest.mark.asyncio
    async def test_submit_chunk_promotes_to_store(self, tmp_path, store):
        promoter = _make_promoter_that_promotes(chunk_passthrough=True)
        monitor = WhiteboardMonitor(
            promoter=promoter, store=store, min_chunk_chars=10
        )
        f = tmp_path / "MEMORY.md"
        f.write_text("")
        monitor.watch_agent(str(f), "agent/coder", "sess-test")
        agent = list(monitor._agents.values())[0]

        # Manually put content in buffer and submit
        agent.buffer = "This is an important new discovery about the system design."
        await monitor._submit_chunk(agent, force=True)

        assert await store.entry_count("sess-test") == 1

    @pytest.mark.asyncio
    async def test_submit_chunk_discarded_chunks_do_not_write(self, tmp_path, store):
        promoter = _make_promoter_that_promotes(chunk_passthrough=False)
        monitor = WhiteboardMonitor(
            promoter=promoter, store=store, min_chunk_chars=10
        )
        f = tmp_path / "MEMORY.md"
        f.write_text("")
        monitor.watch_agent(str(f), "agent/coder", "sess-test")
        agent = list(monitor._agents.values())[0]

        agent.buffer = "Routine progress update."
        await monitor._submit_chunk(agent, force=True)

        assert await store.entry_count("sess-test") == 0

    @pytest.mark.asyncio
    async def test_stop_force_flushes_buffers(self, tmp_path, store):
        promoter = _make_promoter_that_promotes(chunk_passthrough=True)
        monitor = WhiteboardMonitor(
            promoter=promoter, store=store, min_chunk_chars=10
        )
        f = tmp_path / "MEMORY.md"
        f.write_text("")
        monitor.watch_agent(str(f), "agent/coder", "sess-test")
        agent = list(monitor._agents.values())[0]

        # Put content in buffer but don't submit manually
        agent.buffer = "Something important discovered at session end."
        monitor._running = True  # pretend it's running

        await monitor.stop()

        # The buffer should have been force-flushed
        assert await store.entry_count("sess-test") == 1


# ======================================================================
# 4. Query
# ======================================================================

class TestWhiteboardQuery:
    @pytest.mark.asyncio
    async def test_compressed_state_delegates_to_store(self, store):
        await store.write(_make_decision("test chunk"))
        query = WhiteboardQuery(store)
        state = await query.compressed_state("sess-test")
        assert "test chunk" in state

    @pytest.mark.asyncio
    async def test_recent_returns_correct_count(self, store):
        for i in range(8):
            await store.write(_make_decision(f"chunk-{i}"))
        query = WhiteboardQuery(store)
        recent = await query.recent("sess-test", 3)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_by_agent_filters(self, store):
        await store.write(_make_decision("c", source_agent="agent/a"))
        await store.write(_make_decision("d", source_agent="agent/b"))
        query = WhiteboardQuery(store)
        results = await query.by_agent("sess-test", "agent/a")
        assert len(results) == 1
        assert results[0].chunk == "c"

    @pytest.mark.asyncio
    async def test_top_surprise_ordering(self, store):
        await store.write(_make_decision("low", surprise=0.57))
        await store.write(_make_decision("high", surprise=0.94))
        query = WhiteboardQuery(store)
        top = await query.top_surprise("sess-test", 1)
        assert top[0].chunk == "high"

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, store):
        await store.write(_make_decision("PostgreSQL requires migration"))
        await store.write(_make_decision("JWT auth rotated"))
        query = WhiteboardQuery(store)

        results = await query.search("sess-test", "postgresql")
        assert len(results) == 1
        assert "PostgreSQL" in results[0].chunk

    @pytest.mark.asyncio
    async def test_search_no_match_returns_empty(self, store):
        await store.write(_make_decision("some unrelated content"))
        query = WhiteboardQuery(store)
        results = await query.search("sess-test", "xyzzy_nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_entry_count(self, store):
        query = WhiteboardQuery(store)
        assert await query.entry_count("sess-test") == 0
        await store.write(_make_decision("a"))
        assert await query.entry_count("sess-test") == 1

    @pytest.mark.asyncio
    async def test_snapshot_returns_all_entries(self, store):
        await store.write(_make_decision("x"))
        await store.write(_make_decision("y"))
        query = WhiteboardQuery(store)
        snap = await query.snapshot("sess-test")
        assert snap.entry_count == 2

    @pytest.mark.asyncio
    async def test_onboarding_brief_empty_session(self, store):
        query = WhiteboardQuery(store)
        brief = await query.onboarding_brief("sess-test", "agent/new")
        assert "empty" in brief.lower() or "first" in brief.lower()

    @pytest.mark.asyncio
    async def test_onboarding_brief_contains_key_sections(self, store):
        await store.write(_make_decision("Security vuln found", source_agent="agent/security", surprise=0.92))
        await store.write(_make_decision("DB schema locked", source_agent="agent/coder", surprise=0.78))
        query = WhiteboardQuery(store)
        brief = await query.onboarding_brief("sess-test", "agent/new-reviewer")
        assert "agent/new-reviewer" in brief
        assert "Top Discoveries" in brief
        assert "Current Whiteboard State" in brief
        # High-surprise entry should appear in brief
        assert "Security vuln" in brief


# ======================================================================
# 5. End-to-end: monitor → BSC (mocked) → store → query
# ======================================================================

class TestWhiteboardEndToEnd:
    @pytest.mark.asyncio
    async def test_full_flow_file_write_to_queryable_whiteboard(self, tmp_path):
        """
        Write content to a MEMORY.md file, trigger monitor read, verify
        the content lands in the whiteboard and is queryable.
        """
        async with WhiteboardStore(":memory:") as store:
            await store.create_session("e2e")

            promoter = _make_promoter_that_promotes(chunk_passthrough=True)
            monitor = WhiteboardMonitor(
                promoter=promoter, store=store, min_chunk_chars=20
            )

            mem_file = tmp_path / "MEMORY.md"
            mem_file.write_text("")
            monitor.watch_agent(str(mem_file), "agent/coder", "e2e")

            # Simulate agent writing a discovery to its MEMORY.md
            content = (
                "DISCOVERY: The rate limiter middleware is applying per-IP limits "
                "before auth, which blocks legitimate load-balancer health checks."
            )
            mem_file.write_text(content)

            # Trigger manual read (replaces file-watch event in unit test context)
            agent = list(monitor._agents.values())[0]
            await monitor._read_delta(agent)
            await monitor._submit_chunk(agent, force=True)

            # The content should be on the whiteboard
            query = WhiteboardQuery(store)
            count = await query.entry_count("e2e")
            assert count == 1

            entries = await store.get_all("e2e")
            assert "rate limiter" in entries[0].chunk.lower()

            state = await query.compressed_state("e2e")
            assert "rate limiter" in state.lower()

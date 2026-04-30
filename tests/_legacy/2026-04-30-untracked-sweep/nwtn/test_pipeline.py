"""
Integration test for BSCPipeline — the complete event-driven orchestrator.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prsm.compute.nwtn.bsc.pipeline import BSCPipeline
from prsm.compute.nwtn.bsc.event_bus import EventBus, EventType


class TestBSCPipeline:
    """Test the complete BSC pipeline orchestrator."""

    @pytest.fixture
    def mock_components(self):
        """Mock all dependencies required by BSCPipeline."""
        whiteboard_store = MagicMock()
        meta_plan = MagicMock()
        team = MagicMock()
        convergence_tracker = MagicMock()
        narrative_synthesizer = MagicMock()
        project_ledger = MagicMock()
        backend_registry = MagicMock()

        return {
            "whiteboard_store": whiteboard_store,
            "meta_plan": meta_plan,
            "team": team,
            "convergence_tracker": convergence_tracker,
            "narrative_synthesizer": narrative_synthesizer,
            "project_ledger": project_ledger,
            "backend_registry": backend_registry,
        }

    async def test_pipeline_lifecycle(self, mock_components):
        """Pipeline can be started and stopped cleanly."""
        mock_scribe = AsyncMock()
        mock_scribe.setup = AsyncMock()

        with patch('prsm.compute.nwtn.bsc.pipeline.BSCPromoter.warmup', new_callable=AsyncMock):
            with patch('prsm.compute.nwtn.team.live_scribe.LiveScribe', return_value=mock_scribe):
                pipeline = BSCPipeline(**mock_components)

                assert not pipeline.running
                await pipeline.start()
                assert pipeline.running

                # Verify components were initialized
                pipeline.promoter.warmup.assert_called_once()  # type: ignore
                mock_scribe.setup.assert_called_once()

                await pipeline.stop()
                assert not pipeline.running

    async def test_process_chunk_calls_promoter_and_events(self, mock_components):
        """process_chunk() flows through promoter and triggers events."""
        mock_scribe = AsyncMock()
        mock_scribe.setup = AsyncMock()

        mock_promoter = AsyncMock()
        mock_promoter.warmup = AsyncMock()
        mock_promoter.process_chunk.return_value = MagicMock(promoted=True)

        with patch('prsm.compute.nwtn.bsc.pipeline.BSCPromoter', return_value=mock_promoter):
            with patch('prsm.compute.nwtn.team.live_scribe.LiveScribe', return_value=mock_scribe):
                pipeline = BSCPipeline(**mock_components)
                await pipeline.start()

                decision = await pipeline.process_chunk(
                    chunk="Test chunk",
                    source_agent="agent/coder",
                    session_id="sess-1",
                )

                mock_promoter.process_chunk.assert_called_once()
                assert decision.promoted is True

    async def test_advance_round(self, mock_components):
        """advance_round() delegates to promoter and returns stats."""
        mock_scribe = AsyncMock()
        mock_scribe.setup = AsyncMock()

        mock_promoter = MagicMock()
        mock_promoter.warmup = AsyncMock()
        mock_promoter.advance_round.return_value = {
            "round": 5,
            "epsilon": 0.15,
            "dedup_evicted": 3,
            "dedup_index_size": 42,
        }

        with patch('prsm.compute.nwtn.bsc.pipeline.BSCPromoter', return_value=mock_promoter):
            with patch('prsm.compute.nwtn.team.live_scribe.LiveScribe', return_value=mock_scribe):
                pipeline = BSCPipeline(**mock_components)
                await pipeline.start()

                result = pipeline.advance_round(
                    round_number=5,
                    session_id="sess-1",
                )

                mock_promoter.advance_round.assert_called_once_with(
                    round_number=5,
                    dedup_keep_last_n_rounds=2,
                    dedup_entries_per_round=10,
                    session_id="sess-1",
                )
                assert result["round"] == 5
                assert result["dedup_index_size"] == 42

    async def test_status_includes_all_components(self, mock_components):
        """status() returns promoter, push handler, event bus, and scribe stats."""
        mock_scribe = AsyncMock()
        mock_scribe.setup = AsyncMock()
        mock_scribe.status = AsyncMock(return_value={"inbox_counts": 7})

        mock_promoter = MagicMock()
        mock_promoter.warmup = AsyncMock()
        with patch('prsm.compute.nwtn.bsc.pipeline.BSCPromoter', return_value=mock_promoter):
            with patch('prsm.compute.nwtn.team.live_scribe.LiveScribe', return_value=mock_scribe):
                pipeline = BSCPipeline(**mock_components)
                await pipeline.start()

                status = await pipeline.status()

                assert "running" in status
                assert "config" in status
                assert "promoter" in status
                assert "push_handler" in status
                assert "event_bus" in status
                assert "scribe" in status
                assert status["scribe"]["inbox_counts"] == 7

    async def test_auto_start_on_process_chunk(self, mock_components):
        """process_chunk() automatically calls start() if pipeline not started."""
        mock_scribe = AsyncMock()
        mock_scribe.setup = AsyncMock()

        mock_promoter = AsyncMock()
        mock_promoter.warmup = AsyncMock()
        mock_promoter.process_chunk.return_value = MagicMock(promoted=False)

        with patch('prsm.compute.nwtn.bsc.pipeline.BSCPromoter', return_value=mock_promoter):
            with patch('prsm.compute.nwtn.team.live_scribe.LiveScribe', return_value=mock_scribe):
                pipeline = BSCPipeline(**mock_components)

                # Not started yet
                assert not pipeline.running

                # This call should auto‑start
                decision = await pipeline.process_chunk(
                    chunk="Late start",
                    source_agent="agent/qa",
                    session_id="sess-2",
                )

                # Should now be running
                assert pipeline.running
                mock_promoter.warmup.assert_called_once()
                mock_scribe.setup.assert_called_once()

                assert decision.promoted is False

    def test_accessors(self, mock_components):
        """Accessor properties return internal components."""
        mock_scribe = AsyncMock()
        with patch('prsm.compute.nwtn.bsc.pipeline.BSCPromoter'):
            with patch('prsm.compute.nwtn.team.live_scribe.LiveScribe', return_value=mock_scribe):
                pipeline = BSCPipeline(**mock_components)

                # Promoter is the same instance
                assert pipeline.promoter is pipeline._promoter

                # Event bus is available for custom subscribers
                assert isinstance(pipeline.event_bus, EventBus)

                # Scribe is the same instance
                assert pipeline.scribe is pipeline._scribe

    async def test_reset_session(self, mock_components):
        """reset_session() clears dedup index."""
        mock_scribe = AsyncMock()
        mock_scribe.setup = AsyncMock()

        mock_promoter = MagicMock()
        mock_promoter.warmup = AsyncMock()
        mock_promoter.reset_session = MagicMock()

        with patch('prsm.compute.nwtn.bsc.pipeline.BSCPromoter', return_value=mock_promoter):
            with patch('prsm.compute.nwtn.team.live_scribe.LiveScribe', return_value=mock_scribe):
                pipeline = BSCPipeline(**mock_components)
                await pipeline.start()

                await pipeline.reset_session()
                mock_promoter.reset_session.assert_called_once()
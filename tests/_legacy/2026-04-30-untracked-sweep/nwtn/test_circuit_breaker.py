"""
Tests for Circuit Breaker implementation.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.compute.nwtn.bsc.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from prsm.compute.nwtn.bsc.whiteboard_push import WhiteboardPushHandler


# ----------------------------------------------------------------------
# CircuitBreaker Unit Tests
# ----------------------------------------------------------------------


class TestCircuitBreakerState:
    """Tests for circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker()
        assert await breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_closed_to_open_after_threshold_failures(self):
        """Circuit opens after consecutive failures reach threshold."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def failing_call():
            raise RuntimeError("Service unavailable")

        for i in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(failing_call())
            assert await breaker.get_state() == CircuitState.CLOSED

        # Third failure should open the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        assert await breaker.get_state() == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_recovery_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        current_time = [0.0]

        def mock_clock():
            return current_time[0]

        breaker = CircuitBreaker(
            failure_threshold=1, recovery_timeout=60.0, clock=mock_clock
        )

        async def failing_call():
            raise RuntimeError("Service unavailable")

        # Open the circuit at time 0
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        assert await breaker.get_state() == CircuitState.OPEN

        # Advance time past recovery timeout
        current_time[0] = 61.0

        # Should now be in HALF_OPEN
        assert await breaker.get_state() == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_to_closed_after_success_threshold(self):
        """Circuit closes after enough successes in HALF_OPEN state."""
        current_time = [0.0]

        def mock_clock():
            return current_time[0]

        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=60.0,
            success_threshold=2,
            clock=mock_clock,
        )

        async def failing_call():
            raise RuntimeError("Service unavailable")

        async def success_call():
            return "ok"

        # Open the circuit at time 0
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        assert await breaker.get_state() == CircuitState.OPEN

        # Advance time past recovery timeout to get HALF_OPEN
        current_time[0] = 61.0
        assert await breaker.get_state() == CircuitState.HALF_OPEN

        # First success
        result = await breaker.call(success_call())
        assert result == "ok"
        assert await breaker.get_state() == CircuitState.HALF_OPEN

        # Second success should close the circuit
        result = await breaker.call(success_call())
        assert result == "ok"
        assert await breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self):
        """Any failure in HALF_OPEN immediately opens the circuit."""
        current_time = [0.0]

        def mock_clock():
            return current_time[0]

        breaker = CircuitBreaker(
            failure_threshold=1, recovery_timeout=60.0, clock=mock_clock
        )

        async def failing_call():
            raise RuntimeError("Service unavailable")

        async def success_call():
            return "ok"

        # Open the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        assert await breaker.get_state() == CircuitState.OPEN

        # Advance time to get HALF_OPEN
        current_time[0] = 61.0
        assert await breaker.get_state() == CircuitState.HALF_OPEN

        # One success
        await breaker.call(success_call())
        assert await breaker.get_state() == CircuitState.HALF_OPEN

        # Failure should immediately reopen
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        assert await breaker.get_state() == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_open_error_raised_when_open(self):
        """CircuitOpenError is raised when circuit is OPEN."""
        current_time = [0.0]

        def mock_clock():
            return current_time[0]

        breaker = CircuitBreaker(
            failure_threshold=1, recovery_timeout=60.0, clock=mock_clock
        )

        async def failing_call():
            raise RuntimeError("Service unavailable")

        async def success_call():
            return "ok"

        # Open the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())

        # Should raise CircuitOpenError
        with pytest.raises(CircuitOpenError) as exc_info:
            await breaker.call(success_call())
        assert "circuit breaker is open" in str(exc_info.value).lower()


class TestCircuitBreakerStats:
    """Tests for circuit breaker statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_current_state(self):
        """get_stats returns current state and counters."""
        breaker = CircuitBreaker(failure_threshold=3)

        stats = breaker.get_stats()
        assert stats["state"] == "closed"
        assert stats["consecutive_failures"] == 0
        assert stats["total_failures"] == 0
        assert stats["total_successes"] == 0

    @pytest.mark.asyncio
    async def test_stats_track_failures(self):
        """Statistics track failure counts."""
        breaker = CircuitBreaker(failure_threshold=5)

        async def failing_call():
            raise RuntimeError("fail")

        for i in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(failing_call())

        stats = breaker.get_stats()
        assert stats["consecutive_failures"] == 3
        assert stats["total_failures"] == 3
        assert stats["last_failure_at"] is not None

    @pytest.mark.asyncio
    async def test_stats_track_successes(self):
        """Statistics track success counts."""
        breaker = CircuitBreaker()

        async def success_call():
            return "ok"

        for i in range(3):
            await breaker.call(success_call())

        stats = breaker.get_stats()
        assert stats["total_successes"] == 3
        assert stats["consecutive_failures"] == 0

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """reset() returns circuit to CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=1)

        async def failing_call():
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        assert await breaker.get_state() == CircuitState.OPEN

        # Reset
        await breaker.reset()
        assert await breaker.get_state() == CircuitState.CLOSED
        stats = breaker.get_stats()
        assert stats["consecutive_failures"] == 0


class TestCircuitBreakerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_success_resets_consecutive_failures_in_closed(self):
        """A success in CLOSED state resets consecutive failure count."""
        breaker = CircuitBreaker(failure_threshold=5)

        async def failing_call():
            raise RuntimeError("fail")

        async def success_call():
            return "ok"

        # Two failures
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        assert breaker.get_stats()["consecutive_failures"] == 2

        # Success resets
        await breaker.call(success_call())
        assert breaker.get_stats()["consecutive_failures"] == 0

        # More failures - should not open yet
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        assert await breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_multiple_successes_needed_in_half_open(self):
        """HALF_OPEN requires success_threshold successes to close."""
        current_time = [0.0]

        def mock_clock():
            return current_time[0]

        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=60.0,
            success_threshold=3,
            clock=mock_clock,
        )

        async def failing_call():
            raise RuntimeError("fail")

        async def success_call():
            return "ok"

        # Open and transition to HALF_OPEN
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call())
        current_time[0] = 61.0
        assert await breaker.get_state() == CircuitState.HALF_OPEN

        # Two successes - still HALF_OPEN
        await breaker.call(success_call())
        await breaker.call(success_call())
        assert await breaker.get_state() == CircuitState.HALF_OPEN

        # Third success closes
        await breaker.call(success_call())
        assert await breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_safe(self):
        """Circuit breaker handles concurrent calls correctly."""
        breaker = CircuitBreaker(failure_threshold=10)

        async def success_call():
            await asyncio.sleep(0.01)
            return "ok"

        # Run multiple concurrent calls
        tasks = [breaker.call(success_call()) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert breaker.get_stats()["total_successes"] == 10


# ----------------------------------------------------------------------
# WhiteboardPushHandler Integration Tests
# ----------------------------------------------------------------------


class TestWhiteboardPushHandlerCircuitBreaker:
    """Tests for WhiteboardPushHandler circuit breaker integration."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock EventBus."""
        bus = MagicMock(spec=["subscribe", "unsubscribe"])
        bus.subscribe = AsyncMock()
        bus.unsubscribe = AsyncMock()
        return bus

    @pytest.fixture
    def mock_scribe(self):
        """Create a mock LiveScribe."""
        scribe = MagicMock()
        scribe.on_chunk_promoted = AsyncMock(return_value=MagicMock(conflict_detected=False))
        return scribe

    @pytest.fixture
    def decision(self):
        """Create a mock decision."""
        decision = MagicMock()
        decision.metadata = MagicMock()
        decision.metadata.source_agent = "test-agent"
        decision.surprise_score = 0.5
        return decision

    @pytest.mark.asyncio
    async def test_handler_with_no_circuit_breaker(self, mock_event_bus, mock_scribe, decision):
        """Handler works without circuit breaker (backward compatible)."""
        handler = WhiteboardPushHandler(event_bus=mock_event_bus, live_scribe=mock_scribe)

        event = MagicMock()
        event.session_id = "test-session"
        event.data = {"decision": decision}

        await handler._on_promoted(event)

        assert handler.get_stats()["pushed"] == 1
        mock_scribe.on_chunk_promoted.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_with_circuit_breaker_success(
        self, mock_event_bus, mock_scribe, decision
    ):
        """Handler uses circuit breaker on success."""
        breaker = CircuitBreaker()
        handler = WhiteboardPushHandler(
            event_bus=mock_event_bus, live_scribe=mock_scribe, circuit_breaker=breaker
        )

        event = MagicMock()
        event.session_id = "test-session"
        event.data = {"decision": decision}

        await handler._on_promoted(event)

        assert handler.get_stats()["pushed"] == 1
        assert "circuit_breaker" in handler.get_stats()

    @pytest.mark.asyncio
    async def test_handler_skips_when_circuit_open(
        self, mock_event_bus, mock_scribe, decision
    ):
        """Handler skips events and increments skipped when circuit is OPEN."""
        breaker = CircuitBreaker(failure_threshold=1)

        # Open the circuit by causing a failure
        failing_scribe = MagicMock()
        failing_scribe.on_chunk_promoted = AsyncMock(side_effect=RuntimeError("fail"))

        handler = WhiteboardPushHandler(
            event_bus=mock_event_bus, live_scribe=failing_scribe, circuit_breaker=breaker
        )

        # First call fails and opens circuit
        event = MagicMock()
        event.session_id = "test-session"
        event.data = {"decision": decision}
        await handler._on_promoted(event)
        assert handler.get_stats()["failed"] == 1

        # Create new handler with working scribe but same breaker (still open)
        handler2 = WhiteboardPushHandler(
            event_bus=mock_event_bus, live_scribe=mock_scribe, circuit_breaker=breaker
        )

        # This should be skipped due to open circuit
        await handler2._on_promoted(event)
        assert handler2.get_stats()["skipped"] == 1
        assert handler2.get_stats()["pushed"] == 0
        assert handler2.get_stats()["failed"] == 0

    @pytest.mark.asyncio
    async def test_handler_circuit_breaker_stats_included(
        self, mock_event_bus, mock_scribe, decision
    ):
        """Handler includes circuit breaker stats in get_stats()."""
        breaker = CircuitBreaker()
        handler = WhiteboardPushHandler(
            event_bus=mock_event_bus, live_scribe=mock_scribe, circuit_breaker=breaker
        )

        stats = handler.get_stats()
        assert "circuit_breaker" in stats
        assert stats["circuit_breaker"]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_handler_without_circuit_breaker_no_stats(
        self, mock_event_bus, mock_scribe, decision
    ):
        """Handler without circuit breaker doesn't include breaker stats."""
        handler = WhiteboardPushHandler(event_bus=mock_event_bus, live_scribe=mock_scribe)

        stats = handler.get_stats()
        assert "circuit_breaker" not in stats

    @pytest.mark.asyncio
    async def test_handler_counts_failures_through_circuit_breaker(
        self, mock_event_bus, decision
    ):
        """Handler counts failures when circuit breaker is configured."""
        breaker = CircuitBreaker(failure_threshold=3)

        failing_scribe = MagicMock()
        failing_scribe.on_chunk_promoted = AsyncMock(side_effect=RuntimeError("fail"))

        handler = WhiteboardPushHandler(
            event_bus=mock_event_bus, live_scribe=failing_scribe, circuit_breaker=breaker
        )

        event = MagicMock()
        event.session_id = "test-session"
        event.data = {"decision": decision}

        # Multiple failures
        for _ in range(3):
            await handler._on_promoted(event)

        assert handler.get_stats()["failed"] == 3
        # Circuit should be open now
        assert breaker.get_stats()["state"] == "open"

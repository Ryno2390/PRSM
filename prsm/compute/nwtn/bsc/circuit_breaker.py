"""
Circuit Breaker Pattern Implementation
=======================================

Protects external service calls from cascading failures by implementing
the circuit breaker pattern with three states: CLOSED, OPEN, and HALF_OPEN.

State Transitions
-----------------
- CLOSED → OPEN: After `failure_threshold` consecutive failures
- OPEN → HALF_OPEN: After `recovery_timeout` seconds
- HALF_OPEN → CLOSED: After `success_threshold` consecutive successes
- HALF_OPEN → OPEN: On any failure

Usage
-----
.. code-block:: python

    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

    try:
        result = await breaker.call(some_async_operation())
    except CircuitOpenError:
        # Circuit is open, fail-fast
        pass
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, Awaitable, Dict, Optional


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    """Normal operation — requests flow through."""

    OPEN = "open"
    """Fail-fast state — requests are rejected immediately."""

    HALF_OPEN = "half_open"
    """Probing state — testing if the service has recovered."""


class CircuitOpenError(Exception):
    """Raised when the circuit is OPEN and a call is attempted."""

    def __init__(self, message: str = "Circuit breaker is open") -> None:
        super().__init__(message)
        self.message = message


class CircuitBreaker:
    """
    Async-safe circuit breaker implementation.

    Parameters
    ----------
    failure_threshold : int
        Number of consecutive failures before opening the circuit.
        Default: 5
    recovery_timeout : float
        Seconds to wait in OPEN state before transitioning to HALF_OPEN.
        Default: 60.0
    success_threshold : int
        Number of consecutive successes in HALF_OPEN state required to
        close the circuit. Default: 2
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        clock: Optional[callable] = None,
    ) -> None:
        self._failure_threshold = max(1, failure_threshold)
        self._recovery_timeout = max(0.0, recovery_timeout)
        self._success_threshold = max(1, success_threshold)
        self._clock = clock if clock is not None else time.monotonic

        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_at: Optional[float] = None
        self._total_failures = 0
        self._total_successes = 0

    async def call(self, coro: Awaitable[Any]) -> Any:
        """
        Execute a coroutine through the circuit breaker.

        Parameters
        ----------
        coro : Awaitable[Any]
            The coroutine to execute.

        Returns
        -------
        Any
            The result of the coroutine.

        Raises
        ------
        CircuitOpenError
            If the circuit is OPEN and the call should be rejected.
        """
        async with self._lock:
            await self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker is open (failures={self._consecutive_failures})"
                )

        # Execute outside the lock to allow concurrent calls in CLOSED state
        try:
            result = await coro
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise

    async def get_state(self) -> CircuitState:
        """
        Get the current state of the circuit breaker.

        Returns
        -------
        CircuitState
            The current state, potentially transitioning from OPEN to HALF_OPEN
            if the recovery timeout has elapsed.
        """
        async with self._lock:
            await self._maybe_transition_to_half_open()
            return self._state

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current circuit breaker statistics.

        Returns
        -------
        dict
            Statistics including state, failure/success counts, and timestamps.
        """
        return {
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "failure_threshold": self._failure_threshold,
            "success_threshold": self._success_threshold,
            "recovery_timeout": self._recovery_timeout,
            "last_failure_at": self._last_failure_at,
        }

    async def reset(self) -> None:
        """
        Reset the circuit breaker to CLOSED state with cleared counters.
        """
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    async def _maybe_transition_to_half_open(self) -> None:
        """
        Check if we should transition from OPEN to HALF_OPEN.

        Must be called while holding self._lock.
        """
        if self._state != CircuitState.OPEN:
            return

        if self._last_failure_at is None:
            # Shouldn't happen, but be defensive
            self._state = CircuitState.HALF_OPEN
            self._consecutive_successes = 0
            return

        elapsed = self._clock() - self._last_failure_at
        if elapsed >= self._recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._consecutive_successes = 0

    async def _record_success(self) -> None:
        """
        Record a successful call and potentially close the circuit.
        """
        async with self._lock:
            self._total_successes += 1

            if self._state == CircuitState.CLOSED:
                # Reset consecutive failures on success in CLOSED state
                self._consecutive_failures = 0
            elif self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._consecutive_successes = 0
            # In OPEN state, we shouldn't reach here (CircuitOpenError raised)

    async def _record_failure(self) -> None:
        """
        Record a failed call and potentially open the circuit.
        """
        async with self._lock:
            self._total_failures += 1
            self._consecutive_failures += 1
            self._last_failure_at = self._clock()
            self._consecutive_successes = 0

            if self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self._failure_threshold:
                    self._state = CircuitState.OPEN
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN immediately opens
                self._state = CircuitState.OPEN

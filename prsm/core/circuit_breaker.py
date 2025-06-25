#!/usr/bin/env python3
"""
Circuit Breaker Implementation for PRSM
Advanced failure handling and system protection

🎯 PURPOSE IN PRSM:
Provides robust circuit breaker patterns for distributed AI orchestration,
protecting the system from cascading failures and ensuring graceful degradation
during high-load or fault conditions.

🔧 INTEGRATION POINTS:
- NWTN Agent Pipeline: Protect against agent failures
- FTNS Service: Prevent token calculation overload
- IPFS Client: Handle storage service disruptions
- External APIs: Manage third-party service failures
- Database Connections: Protect against DB overload

🚀 FEATURES:
- Multi-level circuit breaker hierarchy
- Adaptive failure thresholds based on load
- Real-time health monitoring and recovery
- Graceful degradation strategies
- Comprehensive failure analytics
"""

import asyncio
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import deque, defaultdict
import statistics

logger = structlog.get_logger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery

class FailureType(Enum):
    """Types of failures the circuit breaker handles"""
    TIMEOUT = "timeout"
    ERROR = "error"
    OVERLOAD = "overload"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    RATE_LIMIT = "rate_limit"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout_seconds: float = 30.0       # Request timeout
    sliding_window_size: int = 100      # Number of recent calls to consider
    failure_rate_threshold: float = 0.5 # Percentage of failures to trigger open
    min_calls_threshold: int = 10       # Minimum calls before calculating failure rate
    adaptive_threshold: bool = True     # Adapt thresholds based on load
    max_concurrent_calls: int = 1000    # Maximum concurrent requests

@dataclass
class CallResult:
    """Result of a circuit breaker protected call"""
    success: bool
    duration_ms: float
    error_type: Optional[FailureType] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    call_id: str = field(default_factory=lambda: str(uuid4()))

@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    current_state: CircuitState = CircuitState.CLOSED
    failure_rate: float = 0.0
    avg_response_time: float = 0.0
    last_failure_time: Optional[datetime] = None
    state_change_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0

class CircuitBreaker:
    """
    Advanced circuit breaker implementation with adaptive behavior
    
    Provides protection against cascading failures in distributed systems
    with intelligent failure detection and recovery mechanisms.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.state_change_count = 0
        
        # Call tracking
        self.call_history: deque = deque(maxlen=self.config.sliding_window_size)
        self.concurrent_calls = 0
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=100)
        self.failure_types: Dict[FailureType, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # State change callbacks
        self.state_change_callbacks: List[Callable[[CircuitState, CircuitState], None]] = []
        
        logger.info(f"Circuit breaker '{name}' initialized", 
                   state=self.state.value,
                   failure_threshold=self.config.failure_threshold,
                   recovery_timeout=self.config.recovery_timeout)
    
    async def call(self, func: Callable, *args, fallback: Callable = None, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            fallback: Fallback function if circuit is open
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
            
        Raises:
            CircuitBreakerOpenException: If circuit is open and no fallback
        """
        call_id = str(uuid4())
        
        # Check if circuit allows the call
        if not self._can_execute():
            self._record_rejected_call()
            
            if fallback:
                logger.debug(f"Circuit breaker '{self.name}' is open, using fallback",
                           call_id=call_id)
                return await self._execute_fallback(fallback, *args, **kwargs)
            else:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is open"
                )
        
        # Execute the call with monitoring
        return await self._execute_with_monitoring(func, call_id, *args, **kwargs)
    
    def _can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        with self._lock:
            # Check concurrent call limit
            if self.concurrent_calls >= self.config.max_concurrent_calls:
                logger.warning(f"Circuit breaker '{self.name}' rejecting call - concurrent limit reached",
                             concurrent_calls=self.concurrent_calls,
                             limit=self.config.max_concurrent_calls)
                return False
            
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset to half-open"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    async def _execute_with_monitoring(self, func: Callable, call_id: str, *args, **kwargs) -> Any:
        """Execute function with comprehensive monitoring"""
        start_time = time.perf_counter()
        
        with self._lock:
            self.concurrent_calls += 1
            self.total_calls += 1
        
        try:
            # Set timeout for the call
            result = await asyncio.wait_for(
                self._ensure_coroutine(func, *args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Record successful call
            call_result = CallResult(
                success=True,
                duration_ms=duration_ms,
                call_id=call_id
            )
            
            self._record_success(call_result)
            
            return result
            
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            call_result = CallResult(
                success=False,
                duration_ms=duration_ms,
                error_type=FailureType.TIMEOUT,
                error_message="Request timeout",
                call_id=call_id
            )
            
            self._record_failure(call_result)
            raise
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Classify error type
            error_type = self._classify_error(e)
            
            call_result = CallResult(
                success=False,
                duration_ms=duration_ms,
                error_type=error_type,
                error_message=str(e),
                call_id=call_id
            )
            
            self._record_failure(call_result)
            raise
            
        finally:
            with self._lock:
                self.concurrent_calls -= 1
    
    async def _ensure_coroutine(self, func: Callable, *args, **kwargs):
        """Ensure function is executed as coroutine"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run synchronous function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    async def _execute_fallback(self, fallback: Callable, *args, **kwargs) -> Any:
        """Execute fallback function"""
        try:
            return await self._ensure_coroutine(fallback, *args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback function failed for circuit '{self.name}'",
                        error=str(e))
            raise
    
    def _record_success(self, call_result: CallResult):
        """Record successful call"""
        with self._lock:
            self.successful_calls += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            
            self.call_history.append(call_result)
            self.response_times.append(call_result.duration_ms)
            
            # State transition logic for half-open state
            if (self.state == CircuitState.HALF_OPEN and 
                self.consecutive_successes >= self.config.success_threshold):
                self._transition_to_closed()
            
            logger.debug(f"Circuit breaker '{self.name}' recorded success",
                        duration_ms=call_result.duration_ms,
                        consecutive_successes=self.consecutive_successes,
                        state=self.state.value)
    
    def _record_failure(self, call_result: CallResult):
        """Record failed call"""
        with self._lock:
            self.failed_calls += 1
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_failure_time = call_result.timestamp
            
            self.call_history.append(call_result)
            self.response_times.append(call_result.duration_ms)
            
            if call_result.error_type:
                self.failure_types[call_result.error_type] += 1
            
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._transition_to_open()
            
            logger.warning(f"Circuit breaker '{self.name}' recorded failure",
                         error_type=call_result.error_type.value if call_result.error_type else "unknown",
                         consecutive_failures=self.consecutive_failures,
                         state=self.state.value)
    
    def _record_rejected_call(self):
        """Record rejected call"""
        with self._lock:
            self.rejected_calls += 1
            self.total_calls += 1
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        # Simple threshold check
        if self.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # Failure rate check (if we have enough data)
        if len(self.call_history) >= self.config.min_calls_threshold:
            recent_calls = list(self.call_history)[-self.config.min_calls_threshold:]
            failure_rate = sum(1 for call in recent_calls if not call.success) / len(recent_calls)
            
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit to open state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.state_change_count += 1
        
        logger.warning(f"Circuit breaker '{self.name}' opened",
                      consecutive_failures=self.consecutive_failures,
                      total_failures=self.failed_calls,
                      failure_rate=self.get_failure_rate())
        
        self._notify_state_change(old_state, self.state)
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.state_change_count += 1
        
        logger.info(f"Circuit breaker '{self.name}' half-opened (testing recovery)",
                   time_since_failure=(datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                   if self.last_failure_time else 0)
        
        self._notify_state_change(old_state, self.state)
    
    def _transition_to_closed(self):
        """Transition circuit to closed state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.state_change_count += 1
        self.consecutive_failures = 0
        
        logger.info(f"Circuit breaker '{self.name}' closed (recovered)",
                   consecutive_successes=self.consecutive_successes,
                   total_successes=self.successful_calls)
        
        self._notify_state_change(old_state, self.state)
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for failure tracking"""
        error_str = str(error).lower()
        
        if isinstance(error, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        elif "connection" in error_str or "network" in error_str:
            return FailureType.DEPENDENCY_FAILURE
        elif "rate limit" in error_str or "too many requests" in error_str:
            return FailureType.RATE_LIMIT
        elif "memory" in error_str or "resource" in error_str:
            return FailureType.RESOURCE_EXHAUSTION
        elif "overload" in error_str or "capacity" in error_str:
            return FailureType.OVERLOAD
        else:
            return FailureType.ERROR
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics"""
        with self._lock:
            return CircuitBreakerStats(
                total_calls=self.total_calls,
                successful_calls=self.successful_calls,
                failed_calls=self.failed_calls,
                rejected_calls=self.rejected_calls,
                current_state=self.state,
                failure_rate=self.get_failure_rate(),
                avg_response_time=self.get_avg_response_time(),
                last_failure_time=self.last_failure_time,
                state_change_count=self.state_change_count,
                consecutive_failures=self.consecutive_failures,
                consecutive_successes=self.consecutive_successes
            )
    
    def get_failure_rate(self) -> float:
        """Calculate current failure rate"""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    def get_avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    def add_state_change_callback(self, callback: Callable[[CircuitState, CircuitState], None]):
        """Add callback for state changes"""
        self.state_change_callbacks.append(callback)
    
    def _notify_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Notify all callbacks of state change"""
        for callback in self.state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback failed for circuit '{self.name}'",
                           error=str(e))
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.last_failure_time = None
            
            logger.info(f"Circuit breaker '{self.name}' manually reset")
            self._notify_state_change(old_state, self.state)


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers
    
    Provides centralized management and monitoring of circuit breakers
    across different components of the PRSM system.
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        
        # Global monitoring
        self.global_stats = {
            "total_breakers": 0,
            "open_breakers": 0,
            "half_open_breakers": 0,
            "closed_breakers": 0,
            "total_calls": 0,
            "total_failures": 0,
            "global_failure_rate": 0.0
        }
    
    def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one"""
        with self._lock:
            if name not in self._breakers:
                breaker = CircuitBreaker(name, config)
                breaker.add_state_change_callback(self._on_state_change)
                self._breakers[name] = breaker
                self.global_stats["total_breakers"] += 1
                self.global_stats["closed_breakers"] += 1
                
                logger.info(f"Created new circuit breaker '{name}'")
            
            return self._breakers[name]
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker"""
        return self._breakers.get(name)
    
    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers"""
        return self._breakers.copy()
    
    def _on_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Handle state change for global statistics"""
        with self._lock:
            # Update counters
            if old_state == CircuitState.CLOSED:
                self.global_stats["closed_breakers"] -= 1
            elif old_state == CircuitState.OPEN:
                self.global_stats["open_breakers"] -= 1
            elif old_state == CircuitState.HALF_OPEN:
                self.global_stats["half_open_breakers"] -= 1
            
            if new_state == CircuitState.CLOSED:
                self.global_stats["closed_breakers"] += 1
            elif new_state == CircuitState.OPEN:
                self.global_stats["open_breakers"] += 1
            elif new_state == CircuitState.HALF_OPEN:
                self.global_stats["half_open_breakers"] += 1
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global circuit breaker statistics"""
        with self._lock:
            # Calculate current stats
            total_calls = sum(b.total_calls for b in self._breakers.values())
            total_failures = sum(b.failed_calls for b in self._breakers.values())
            
            self.global_stats.update({
                "total_calls": total_calls,
                "total_failures": total_failures,
                "global_failure_rate": total_failures / total_calls if total_calls > 0 else 0.0
            })
            
            return self.global_stats.copy()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers"""
        return {
            "global": self.get_global_stats(),
            "breakers": {name: breaker.get_stats() for name, breaker in self._breakers.items()}
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            
            logger.info("All circuit breakers reset")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


# Global circuit breaker registry
circuit_registry = CircuitBreakerRegistry()


# === Convenience Functions ===

def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a circuit breaker"""
    return circuit_registry.get_or_create(name, config)


async def protected_call(
    name: str, 
    func: Callable, 
    *args, 
    config: CircuitBreakerConfig = None,
    fallback: Callable = None,
    **kwargs
) -> Any:
    """Execute function with circuit breaker protection"""
    breaker = get_circuit_breaker(name, config)
    return await breaker.call(func, *args, fallback=fallback, **kwargs)


def get_all_circuit_stats() -> Dict[str, Any]:
    """Get statistics for all circuit breakers"""
    return circuit_registry.get_all_stats()


# === PRSM-Specific Circuit Breaker Configurations ===

# Agent pipeline circuit breaker
AGENT_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    success_threshold=2,
    timeout_seconds=15.0,
    failure_rate_threshold=0.3
)

# FTNS service circuit breaker
FTNS_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    success_threshold=3,
    timeout_seconds=10.0,
    failure_rate_threshold=0.4
)

# IPFS client circuit breaker
IPFS_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=4,
    recovery_timeout=45.0,
    success_threshold=2,
    timeout_seconds=20.0,
    failure_rate_threshold=0.5
)

# Database circuit breaker
DATABASE_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=6,
    recovery_timeout=30.0,
    success_threshold=3,
    timeout_seconds=5.0,
    failure_rate_threshold=0.6
)

# External API circuit breaker
EXTERNAL_API_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=120.0,
    success_threshold=2,
    timeout_seconds=30.0,
    failure_rate_threshold=0.3
)
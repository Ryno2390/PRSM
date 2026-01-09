"""
CHRONOS Error Handling & Circuit Breakers

Production-grade error handling, retry logic, and circuit breakers for enterprise reliability.
"""

import asyncio
import logging
import functools
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, requests blocked
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


@dataclass
class ErrorEvent:
    """Represents an error event in the system."""
    timestamp: datetime
    component: str
    operation: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "context": self.context
        }


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        self.total_requests += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Execute the function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Success handling
            self._on_success()
            return result
            
        except self.expected_exception as e:
            # Expected failure handling
            self._on_failure()
            raise
        except Exception as e:
            # Unexpected failure
            self._on_failure()
            logger.error(f"Unexpected error in circuit breaker {self.name}: {e}")
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.total_successes += 1
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(f"Circuit breaker {self.name} reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed operation."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        success_rate = (self.total_successes / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate_percent": round(success_rate, 2),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryPolicy:
    """Configurable retry policy with exponential backoff."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


async def retry_with_policy(
    func: Callable,
    retry_policy: RetryPolicy,
    *args,
    **kwargs
):
    """Execute function with retry policy."""
    
    last_exception = None
    
    for attempt in range(retry_policy.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        except Exception as e:
            last_exception = e
            
            if attempt < retry_policy.max_attempts - 1:
                delay = retry_policy.calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {retry_policy.max_attempts} attempts failed")
    
    raise last_exception


class ErrorTracker:
    """Tracks and analyzes errors across the system."""
    
    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: List[ErrorEvent] = []
        self.component_stats = {}
        
    def record_error(
        self,
        component: str,
        operation: str,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record an error event."""
        
        event = ErrorEvent(
            timestamp=datetime.utcnow(),
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            context=context or {}
        )
        
        self.events.append(event)
        
        # Maintain size limit
        if len(self.events) > self.max_events:
            self.events.pop(0)
        
        # Update component stats
        if component not in self.component_stats:
            self.component_stats[component] = {
                "total_errors": 0,
                "error_types": {},
                "severity_breakdown": {s.value: 0 for s in ErrorSeverity}
            }
        
        stats = self.component_stats[component]
        stats["total_errors"] += 1
        stats["error_types"][event.error_type] = stats["error_types"].get(event.error_type, 0) + 1
        stats["severity_breakdown"][severity.value] += 1
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{component}.{operation}: {error}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"{component}.{operation}: {error}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"{component}.{operation}: {error}")
        else:
            logger.info(f"{component}.{operation}: {error}")
    
    def get_error_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get error summary statistics."""
        
        if since:
            relevant_events = [e for e in self.events if e.timestamp >= since]
        else:
            relevant_events = self.events
        
        if not relevant_events:
            return {
                "total_errors": 0,
                "components_affected": 0,
                "severity_breakdown": {s.value: 0 for s in ErrorSeverity},
                "top_error_types": [],
                "error_rate_trend": "stable"
            }
        
        # Calculate statistics
        severity_counts = {s.value: 0 for s in ErrorSeverity}
        error_types = {}
        components = set()
        
        for event in relevant_events:
            severity_counts[event.severity.value] += 1
            error_types[event.error_type] = error_types.get(event.error_type, 0) + 1
            components.add(event.component)
        
        # Top error types
        top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Error rate trend (simple)
        now = datetime.utcnow()
        recent_errors = [e for e in relevant_events if e.timestamp >= now - timedelta(minutes=30)]
        older_errors = [e for e in relevant_events if now - timedelta(hours=1) <= e.timestamp < now - timedelta(minutes=30)]
        
        if len(older_errors) == 0:
            trend = "stable"
        elif len(recent_errors) > len(older_errors) * 1.5:
            trend = "increasing"
        elif len(recent_errors) < len(older_errors) * 0.5:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "total_errors": len(relevant_events),
            "components_affected": len(components),
            "severity_breakdown": severity_counts,
            "top_error_types": [{"type": t, "count": c} for t, c in top_errors],
            "error_rate_trend": trend,
            "component_stats": self.component_stats
        }
    
    def get_recent_critical_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent critical errors."""
        
        since = datetime.utcnow() - timedelta(hours=hours)
        critical_events = [
            e for e in self.events 
            if e.timestamp >= since and e.severity == ErrorSeverity.CRITICAL
        ]
        
        return [event.to_dict() for event in critical_events]


# Global instances
error_tracker = ErrorTracker()
circuit_breakers = {}


def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60
) -> CircuitBreaker:
    """Create and register a circuit breaker."""
    
    if name in circuit_breakers:
        return circuit_breakers[name]
    
    breaker = CircuitBreaker(name, failure_threshold, recovery_timeout)
    circuit_breakers[name] = breaker
    return breaker


def with_circuit_breaker(
    breaker_name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60
):
    """Decorator to add circuit breaker protection to functions."""
    
    def decorator(func):
        breaker = create_circuit_breaker(breaker_name, failure_threshold, recovery_timeout)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await breaker.call(func, *args, **kwargs)
            except Exception as e:
                error_tracker.record_error(
                    component=breaker_name,
                    operation=func.__name__,
                    error=e,
                    severity=ErrorSeverity.HIGH
                )
                raise
        
        return wrapper
    return decorator


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Decorator to add retry logic to functions."""
    
    def decorator(func):
        retry_policy = RetryPolicy(max_attempts, base_delay, max_delay)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await retry_with_policy(func, retry_policy, *args, **kwargs)
            except Exception as e:
                error_tracker.record_error(
                    component=func.__module__,
                    operation=func.__name__,
                    error=e,
                    severity=ErrorSeverity.MEDIUM
                )
                raise
        
        return wrapper
    return decorator


async def get_system_health() -> Dict[str, Any]:
    """Get overall system health status."""
    
    # Circuit breaker health
    breaker_health = {}
    for name, breaker in circuit_breakers.items():
        metrics = breaker.get_metrics()
        breaker_health[name] = {
            "status": "healthy" if breaker.state == CircuitState.CLOSED else "degraded",
            "metrics": metrics
        }
    
    # Error summary
    error_summary = error_tracker.get_error_summary(since=datetime.utcnow() - timedelta(hours=1))
    
    # Overall health score
    total_breakers = len(circuit_breakers)
    healthy_breakers = sum(1 for b in circuit_breakers.values() if b.state == CircuitState.CLOSED)
    
    if total_breakers == 0:
        health_score = 100
    else:
        health_score = (healthy_breakers / total_breakers) * 100
        
        # Reduce score for high error rates
        critical_errors = error_summary["severity_breakdown"]["CRITICAL"]
        if critical_errors > 0:
            health_score *= 0.5
        elif error_summary["total_errors"] > 50:
            health_score *= 0.8
    
    return {
        "overall_health_score": round(health_score, 1),
        "status": "healthy" if health_score >= 90 else "degraded" if health_score >= 70 else "unhealthy",
        "circuit_breakers": breaker_health,
        "error_summary": error_summary,
        "timestamp": datetime.utcnow().isoformat()
    }
"""
Error Recovery Strategies
=========================

Advanced error recovery mechanisms including retry logic,
circuit breakers, fallback strategies, and self-healing systems.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, Optional, Callable, Union, Type, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from .exceptions import PRSMException, ErrorSeverity, ErrorCategory, ProcessingError

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    NONE = "none"
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SELF_HEALING = "self_healing"


@dataclass
class RecoveryConfig:
    """Configuration for error recovery"""
    strategy: RecoveryStrategy = RecoveryStrategy.NONE
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    jitter: bool = True
    timeout_seconds: float = 30.0
    fallback_function: Optional[Callable] = None
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    recovery_conditions: List[str] = field(default_factory=list)


class ErrorRecoveryStrategy(ABC):
    """Abstract base class for error recovery strategies"""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.recovery_stats = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "last_recovery_time": None
        }
    
    @abstractmethod
    async def attempt_recovery(
        self,
        error: PRSMException,
        original_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt to recover from error"""
        pass
    
    def should_attempt_recovery(self, error: PRSMException) -> bool:
        """Determine if recovery should be attempted for this error"""
        # Don't attempt recovery for security errors
        if error.category == ErrorCategory.SECURITY:
            return False
        
        # Don't attempt recovery for critical configuration errors
        if (error.category == ErrorCategory.CONFIGURATION and 
            error.severity == ErrorSeverity.CRITICAL):
            return False
        
        # Check if error is retryable
        if hasattr(error, 'is_retryable') and not error.is_retryable():
            return False
        
        return True
    
    def update_stats(self, success: bool):
        """Update recovery statistics"""
        self.recovery_stats["total_attempts"] += 1
        if success:
            self.recovery_stats["successful_recoveries"] += 1
        else:
            self.recovery_stats["failed_recoveries"] += 1
        self.recovery_stats["last_recovery_time"] = datetime.utcnow()
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        return {
            "strategy": self.config.strategy.value,
            **self.recovery_stats,
            "success_rate": (
                self.recovery_stats["successful_recoveries"] / 
                max(self.recovery_stats["total_attempts"], 1)
            )
        }


class RetryStrategy(ErrorRecoveryStrategy):
    """Retry-based error recovery with exponential backoff"""
    
    def __init__(self, config: RecoveryConfig):
        super().__init__(config)
        self.retry_counts: Dict[str, int] = {}
    
    async def attempt_recovery(
        self,
        error: PRSMException,
        original_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt recovery through retries"""
        if not self.should_attempt_recovery(error):
            raise error
        
        operation_key = f"{error.context.get('component', 'unknown')}:{error.context.get('operation', 'unknown')}"
        current_retry = self.retry_counts.get(operation_key, 0)
        
        if current_retry >= self.config.max_retries:
            logger.warning(f"Max retries ({self.config.max_retries}) exceeded for {operation_key}")
            self.retry_counts[operation_key] = 0  # Reset for future attempts
            self.update_stats(False)
            raise error
        
        # Calculate delay with exponential backoff and jitter
        delay = self._calculate_retry_delay(current_retry)
        
        logger.info(f"Retrying {operation_key} (attempt {current_retry + 1}/{self.config.max_retries}) after {delay:.2f}s")
        await asyncio.sleep(delay)
        
        # Increment retry count
        self.retry_counts[operation_key] = current_retry + 1
        
        try:
            # Attempt the operation again
            if asyncio.iscoroutinefunction(original_function):
                result = await original_function(*args, **kwargs)
            else:
                result = original_function(*args, **kwargs)
            
            # Success - reset retry count
            self.retry_counts[operation_key] = 0
            self.update_stats(True)
            return result
            
        except Exception as retry_error:
            # Convert to PRSM exception if needed
            if not isinstance(retry_error, PRSMException):
                retry_error = ProcessingError(
                    message=str(retry_error),
                    component=error.context.get("component", "unknown"),
                    operation=error.context.get("operation", "retry"),
                    original_exception=retry_error
                )
            
            # Recursive retry
            return await self.attempt_recovery(retry_error, original_function, *args, **kwargs)
    
    def _calculate_retry_delay(self, retry_attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        base_delay = self.config.retry_delay
        
        if self.config.exponential_backoff:
            delay = base_delay * (2 ** retry_attempt)
        else:
            delay = base_delay
        
        if self.config.jitter:
            # Add random jitter (±25%)
            jitter_factor = 0.25
            jitter = delay * jitter_factor * (2 * random.random() - 1)
            delay += jitter
        
        return max(delay, 0.1)  # Minimum 0.1 second delay


class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback to alternative implementation"""
    
    async def attempt_recovery(
        self,
        error: PRSMException,
        original_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt recovery using fallback function"""
        if not self.should_attempt_recovery(error):
            raise error
        
        if not self.config.fallback_function:
            logger.warning("No fallback function configured")
            self.update_stats(False)
            raise error
        
        try:
            logger.info(f"Attempting fallback recovery for {error.error_code}")
            
            if asyncio.iscoroutinefunction(self.config.fallback_function):
                result = await self.config.fallback_function(*args, **kwargs)
            else:
                result = self.config.fallback_function(*args, **kwargs)
            
            self.update_stats(True)
            return result
            
        except Exception as fallback_error:
            logger.error(f"Fallback strategy failed: {fallback_error}")
            self.update_stats(False)
            
            # Return the original error if fallback fails
            raise error


class CircuitBreakerStrategy(ErrorRecoveryStrategy):
    """Circuit breaker pattern for error recovery"""
    
    def __init__(self, config: RecoveryConfig):
        super().__init__(config)
        self.failure_counts: Dict[str, int] = {}
        self.circuit_open_times: Dict[str, datetime] = {}
        self.circuit_states: Dict[str, str] = {}  # closed, open, half-open
    
    async def attempt_recovery(
        self,
        error: PRSMException,
        original_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt recovery using circuit breaker pattern"""
        operation_key = f"{error.context.get('component', 'unknown')}:{error.context.get('operation', 'unknown')}"
        
        # Check circuit state
        circuit_state = self._get_circuit_state(operation_key)
        
        if circuit_state == "open":
            logger.warning(f"Circuit breaker is OPEN for {operation_key}")
            self.update_stats(False)
            raise ProcessingError(
                message="Circuit breaker is open - service temporarily unavailable",
                component=error.context.get("component", "unknown"),
                operation="circuit_breaker",
                context={"original_error": error.error_code}
            )
        
        try:
            # Attempt the operation
            if asyncio.iscoroutinefunction(original_function):
                result = await original_function(*args, **kwargs)
            else:
                result = original_function(*args, **kwargs)
            
            # Success - reset failure count and close circuit
            self._record_success(operation_key)
            self.update_stats(True)
            return result
            
        except Exception as circuit_error:
            # Record failure and potentially open circuit
            self._record_failure(operation_key)
            self.update_stats(False)
            
            if not isinstance(circuit_error, PRSMException):
                circuit_error = ProcessingError(
                    message=str(circuit_error),
                    component=error.context.get("component", "unknown"),
                    operation=error.context.get("operation", "circuit_breaker"),
                    original_exception=circuit_error
                )
            
            raise circuit_error
    
    def _get_circuit_state(self, operation_key: str) -> str:
        """Get current circuit breaker state"""
        current_state = self.circuit_states.get(operation_key, "closed")
        
        if current_state == "open":
            # Check if timeout has passed
            open_time = self.circuit_open_times.get(operation_key)
            if open_time and (datetime.utcnow() - open_time).total_seconds() > self.config.circuit_breaker_timeout:
                # Move to half-open state
                self.circuit_states[operation_key] = "half-open"
                logger.info(f"Circuit breaker for {operation_key} moved to HALF-OPEN state")
                return "half-open"
        
        return current_state
    
    def _record_success(self, operation_key: str):
        """Record successful operation"""
        self.failure_counts[operation_key] = 0
        self.circuit_states[operation_key] = "closed"
        if operation_key in self.circuit_open_times:
            del self.circuit_open_times[operation_key]
    
    def _record_failure(self, operation_key: str):
        """Record failed operation"""
        current_failures = self.failure_counts.get(operation_key, 0) + 1
        self.failure_counts[operation_key] = current_failures
        
        if current_failures >= self.config.circuit_breaker_threshold:
            # Open the circuit
            self.circuit_states[operation_key] = "open"
            self.circuit_open_times[operation_key] = datetime.utcnow()
            logger.warning(f"Circuit breaker OPENED for {operation_key} after {current_failures} failures")


class GracefulDegradationStrategy(ErrorRecoveryStrategy):
    """Graceful degradation with reduced functionality"""
    
    async def attempt_recovery(
        self,
        error: PRSMException,
        original_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt recovery through graceful degradation"""
        if not self.should_attempt_recovery(error):
            raise error
        
        logger.info(f"Attempting graceful degradation for {error.error_code}")
        
        # Return a degraded response based on the component
        component = error.context.get("component", "unknown")
        
        if component == "nwtn":
            # Return basic response for NWTN errors
            self.update_stats(True)
            return {
                "response": "Service temporarily degraded. Basic response provided.",
                "confidence_score": 0.5,
                "reasoning_engines_used": ["fallback"],
                "degraded_mode": True
            }
        
        elif component == "tokenomics":
            # Return estimated pricing for tokenomics errors
            self.update_stats(True)
            return {
                "estimated_cost": 1.0,  # Default cost
                "pricing_available": False,
                "degraded_mode": True
            }
        
        elif component == "marketplace":
            # Return empty results for marketplace errors
            self.update_stats(True)
            return {
                "results": [],
                "total_count": 0,
                "degraded_mode": True
            }
        
        else:
            # Generic degraded response
            self.update_stats(True)
            return {
                "status": "degraded",
                "message": "Service partially available",
                "degraded_mode": True
            }


class SelfHealingStrategy(ErrorRecoveryStrategy):
    """Self-healing strategy that attempts to fix underlying issues"""
    
    def __init__(self, config: RecoveryConfig):
        super().__init__(config)
        self.healing_attempts: Dict[str, int] = {}
    
    async def attempt_recovery(
        self,
        error: PRSMException,
        original_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt self-healing recovery"""
        if not self.should_attempt_recovery(error):
            raise error
        
        healing_key = f"{error.error_code}:{error.context.get('component', 'unknown')}"
        attempts = self.healing_attempts.get(healing_key, 0)
        
        if attempts >= 3:  # Max healing attempts
            logger.warning(f"Max self-healing attempts reached for {healing_key}")
            self.update_stats(False)
            raise error
        
        logger.info(f"Attempting self-healing for {error.error_code}")
        
        # Attempt specific healing based on error type
        try:
            success = await self._attempt_healing(error)
            
            if success:
                # Try the original operation again
                if asyncio.iscoroutinefunction(original_function):
                    result = await original_function(*args, **kwargs)
                else:
                    result = original_function(*args, **kwargs)
                
                # Reset healing attempts on success
                self.healing_attempts[healing_key] = 0
                self.update_stats(True)
                return result
            else:
                self.healing_attempts[healing_key] = attempts + 1
                self.update_stats(False)
                raise error
                
        except Exception as healing_error:
            self.healing_attempts[healing_key] = attempts + 1
            logger.error(f"Self-healing failed: {healing_error}")
            self.update_stats(False)
            raise error
    
    async def _attempt_healing(self, error: PRSMException) -> bool:
        """Attempt to heal the underlying issue"""
        if error.category == ErrorCategory.RESOURCE:
            # Try to free up resources
            return await self._heal_resource_issue(error)
        
        elif error.category == ErrorCategory.NETWORK:
            # Try to reconnect or reset connections
            return await self._heal_network_issue(error)
        
        elif error.category == ErrorCategory.CONFIGURATION:
            # Try to reset configuration to defaults
            return await self._heal_configuration_issue(error)
        
        return False
    
    async def _heal_resource_issue(self, error: PRSMException) -> bool:
        """Attempt to heal resource-related issues"""
        logger.info("Attempting to heal resource issue")
        
        # Simulate resource cleanup
        await asyncio.sleep(0.1)
        
        # In a real implementation, this would:
        # - Free up memory
        # - Close unused connections
        # - Clear caches
        # - Restart failed services
        
        return True
    
    async def _heal_network_issue(self, error: PRSMException) -> bool:
        """Attempt to heal network-related issues"""
        logger.info("Attempting to heal network issue")
        
        # Simulate network healing
        await asyncio.sleep(0.2)
        
        # In a real implementation, this would:
        # - Reset connection pools
        # - Clear DNS cache
        # - Retry with different endpoints
        # - Restart network services
        
        return True
    
    async def _heal_configuration_issue(self, error: PRSMException) -> bool:
        """Attempt to heal configuration-related issues"""
        logger.info("Attempting to heal configuration issue")
        
        # Simulate configuration healing
        await asyncio.sleep(0.1)
        
        # In a real implementation, this would:
        # - Reset to default configuration
        # - Reload configuration from file
        # - Validate and fix configuration
        # - Restart affected components
        
        return True


class CompositeRecoveryStrategy(ErrorRecoveryStrategy):
    """Composite strategy that combines multiple recovery approaches"""
    
    def __init__(self, strategies: List[ErrorRecoveryStrategy]):
        # Use config from first strategy
        super().__init__(strategies[0].config if strategies else RecoveryConfig())
        self.strategies = strategies
    
    async def attempt_recovery(
        self,
        error: PRSMException,
        original_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt recovery using multiple strategies in sequence"""
        last_error = error
        
        for i, strategy in enumerate(self.strategies):
            try:
                logger.info(f"Attempting recovery strategy {i+1}/{len(self.strategies)}: {strategy.__class__.__name__}")
                result = await strategy.attempt_recovery(last_error, original_function, *args, **kwargs)
                self.update_stats(True)
                return result
            
            except Exception as strategy_error:
                logger.warning(f"Recovery strategy {i+1} failed: {strategy_error}")
                if isinstance(strategy_error, PRSMException):
                    last_error = strategy_error
                continue
        
        # All strategies failed
        self.update_stats(False)
        raise last_error


# Factory function for creating recovery strategies
def create_recovery_strategy(config: RecoveryConfig) -> ErrorRecoveryStrategy:
    """Create recovery strategy based on configuration"""
    if config.strategy == RecoveryStrategy.RETRY:
        return RetryStrategy(config)
    elif config.strategy == RecoveryStrategy.FALLBACK:
        return FallbackStrategy(config)
    elif config.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
        return CircuitBreakerStrategy(config)
    elif config.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
        return GracefulDegradationStrategy(config)
    elif config.strategy == RecoveryStrategy.SELF_HEALING:
        return SelfHealingStrategy(config)
    else:
        # Default to retry strategy
        return RetryStrategy(config)


# Decorator for automatic error recovery
def with_error_recovery(
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **recovery_kwargs
):
    """Decorator to add error recovery to functions"""
    def decorator(func: Callable) -> Callable:
        config = RecoveryConfig(
            strategy=strategy,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **recovery_kwargs
        )
        recovery_strategy = create_recovery_strategy(config)
        
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except PRSMException as e:
                return await recovery_strategy.attempt_recovery(e, func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except PRSMException as e:
                # Convert to async for recovery
                return asyncio.run(recovery_strategy.attempt_recovery(e, func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
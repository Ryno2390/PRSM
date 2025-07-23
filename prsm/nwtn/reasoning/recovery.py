#!/usr/bin/env python3
"""
NWTN Meta-Reasoning Engine Failure Recovery and Circuit Breakers
===============================================================

Failure detection, recovery strategies, and circuit breaker implementations
for the NWTN meta-reasoning system.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

from .types import (
    ReasoningEngine,
    FailureType,
    RecoveryAction,
    FailureDetectionMode,
    FailureEvent,
    RecoveryStrategy
)

# Setup logging
logger = logging.getLogger(__name__)


class FailureDetector:
    """Advanced failure detection system"""
    
    def __init__(self):
        self.failure_events: List[FailureEvent] = []
        self.detection_mode = FailureDetectionMode.HYBRID
        self.failure_thresholds = {
            FailureType.TIMEOUT: 3,
            FailureType.API_ERROR: 5,
            FailureType.RESOURCE_EXHAUSTION: 2,
            FailureType.LOGIC_ERROR: 10,
            FailureType.NETWORK: 5
        }
        self.time_window_hours = 1  # Consider failures within last hour
        self.enabled = True
        
        # Pattern detection
        self.failure_patterns = {}
        self.pattern_detection_enabled = True
        
        # Failure prediction
        self.prediction_enabled = False
        self.prediction_models = {}
    
    def detect_failure(self, engine_type: ReasoningEngine, execution_time: float, 
                      success: bool, error: str = None, context: Dict[str, Any] = None) -> Optional[FailureEvent]:
        """Detect if a failure has occurred"""
        if not self.enabled:
            return None
        
        failure_event = None
        
        # Check for timeout failure
        if execution_time > 30.0:  # Timeout threshold
            failure_event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                engine_id=f"{engine_type.value}_engine",
                failure_type=FailureType.TIMEOUT,
                severity=8,
                error_message=f"Execution time {execution_time:.2f}s exceeded timeout threshold",
                error_details=context or {},
                operation="reasoning_execution"
            )
        
        # Check for API error failure
        elif not success and error:
            # Determine severity based on error type
            severity = 5
            if "rate limit" in error.lower():
                failure_type = FailureType.RATE_LIMIT
                severity = 6
            elif "auth" in error.lower():
                failure_type = FailureType.AUTHENTICATION
                severity = 7
            elif "network" in error.lower():
                failure_type = FailureType.NETWORK
                severity = 6
            else:
                failure_type = FailureType.API_ERROR
                severity = 5
            
            failure_event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                engine_id=f"{engine_type.value}_engine",
                failure_type=failure_type,
                severity=severity,
                error_message=error,
                error_details=context or {},
                operation="api_call"
            )
        
        # Check for resource exhaustion
        elif context and context.get("memory_usage", 0) > 1000:  # 1GB threshold
            failure_event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                engine_id=f"{engine_type.value}_engine",
                failure_type=FailureType.RESOURCE_EXHAUSTION,
                severity=9,
                error_message=f"Memory usage {context['memory_usage']:.2f}MB exceeded threshold",
                error_details=context or {},
                operation="resource_management"
            )
        
        # Check for quality degradation (logic error)
        elif context and context.get("quality_score", 1.0) < 0.3:
            failure_event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                engine_id=f"{engine_type.value}_engine",
                failure_type=FailureType.LOGIC_ERROR,
                severity=4,
                error_message=f"Quality score {context['quality_score']:.2f} below threshold",
                error_details=context or {},
                operation="quality_assessment"
            )
        
        if failure_event:
            self.failure_events.append(failure_event)
            
            # Keep only recent failures (last 1000)
            if len(self.failure_events) > 1000:
                self.failure_events = self.failure_events[-1000:]
            
            # Check for repeated failures pattern
            if self._detect_repeated_failures(engine_type):
                repeated_failure = FailureEvent(
                    timestamp=datetime.now(timezone.utc),
                    engine_id=f"{engine_type.value}_engine",
                    failure_type=FailureType.SYSTEM_ERROR,
                    severity=10,
                    error_message="Repeated failures detected - system instability",
                    error_details={"original_failure": failure_event.failure_type.value},
                    operation="pattern_detection"
                )
                self.failure_events.append(repeated_failure)
                return repeated_failure
            
            return failure_event
        
        return None
    
    def _detect_repeated_failures(self, engine_type: ReasoningEngine) -> bool:
        """Detect if an engine has repeated failures"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.time_window_hours)
        
        recent_failures = [
            event for event in self.failure_events
            if event.engine_id == f"{engine_type.value}_engine" and 
               event.timestamp >= cutoff_time and
               event.failure_type != FailureType.SYSTEM_ERROR
        ]
        
        return len(recent_failures) >= self.failure_thresholds.get(FailureType.SYSTEM_ERROR, 5)
    
    def get_failure_history(self, engine_type: ReasoningEngine = None, 
                           hours: int = 24) -> List[FailureEvent]:
        """Get failure history for an engine or all engines"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        failures = [
            event for event in self.failure_events
            if event.timestamp >= cutoff_time
        ]
        
        if engine_type:
            failures = [f for f in failures if f.engine_id == f"{engine_type.value}_engine"]
        
        return sorted(failures, key=lambda f: f.timestamp, reverse=True)
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics"""
        total_failures = len(self.failure_events)
        
        if total_failures == 0:
            return {
                "total_failures": 0,
                "failure_rate": 0.0,
                "most_common_failure": None,
                "engine_failure_counts": {},
                "recent_failures": 0
            }
        
        # Count failures by type
        failure_type_counts = {}
        for event in self.failure_events:
            failure_type = event.failure_type.value
            failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
        
        # Count failures by engine
        engine_failure_counts = {}
        for event in self.failure_events:
            engine = event.engine_id
            engine_failure_counts[engine] = engine_failure_counts.get(engine, 0) + 1
        
        # Recent failures (last hour)
        recent_failures = len(self.get_failure_history(hours=1))
        
        return {
            "total_failures": total_failures,
            "failure_types": failure_type_counts,
            "engine_failure_counts": engine_failure_counts,
            "most_common_failure": max(failure_type_counts.items(), key=lambda x: x[1])[0] if failure_type_counts else None,
            "recent_failures": recent_failures,
            "detection_mode": self.detection_mode.value,
            "enabled": self.enabled
        }
    
    def reset_failure_history(self, engine_type: ReasoningEngine = None):
        """Reset failure history for specific engine or all engines"""
        if engine_type:
            self.failure_events = [
                event for event in self.failure_events
                if event.engine_id != f"{engine_type.value}_engine"
            ]
        else:
            self.failure_events = []
    
    def enable_detection(self):
        """Enable failure detection"""
        self.enabled = True
    
    def disable_detection(self):
        """Disable failure detection"""
        self.enabled = False


class CircuitBreaker:
    """Circuit breaker implementation for engine protection"""
    
    def __init__(self, engine_type: ReasoningEngine, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, half_open_max_calls: int = 3):
        self.engine_type = engine_type
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout  # seconds
        self.half_open_max_calls = half_open_max_calls
        
        # Circuit breaker state
        self.state = "closed"  # closed, open, half_open
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.successful_calls = 0
        
        # Statistics
        self.total_calls = 0
        self.blocked_calls = 0
        self.state_changes = []
        
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        self.total_calls += 1
        
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self._should_attempt_reset():
                self._transition_to_half_open()
                return True
            else:
                self.blocked_calls += 1
                return False
        elif self.state == "half_open":
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            else:
                self.blocked_calls += 1
                return False
        
        return False
    
    def record_success(self):
        """Record a successful execution"""
        self.successful_calls += 1
        
        if self.state == "half_open":
            if self.successful_calls >= self.half_open_max_calls:
                self._transition_to_closed()
        else:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record a failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            self._transition_to_open()
        elif self.state == "half_open":
            self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from open state"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        self.state = "closed"
        self.failure_count = 0
        self.half_open_calls = 0
        self.successful_calls = 0
        self._record_state_change("closed")
    
    def _transition_to_open(self):
        """Transition to open state"""
        self.state = "open"
        self.half_open_calls = 0
        self.successful_calls = 0
        self._record_state_change("open")
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self.state = "half_open"
        self.half_open_calls = 0
        self.successful_calls = 0
        self._record_state_change("half_open")
    
    def _record_state_change(self, new_state: str):
        """Record state change for monitoring"""
        self.state_changes.append({
            "timestamp": datetime.now(timezone.utc),
            "from_state": getattr(self, '_previous_state', 'unknown'),
            "to_state": new_state,
            "failure_count": self.failure_count
        })
        self._previous_state = new_state
        
        logger.info(f"Circuit breaker for {self.engine_type.value} transitioned to {new_state}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "engine_type": self.engine_type.value,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "total_calls": self.total_calls,
            "blocked_calls": self.blocked_calls,
            "success_rate": self.successful_calls / max(1, self.total_calls - self.blocked_calls),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "state_changes": len(self.state_changes)
        }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self._transition_to_closed()


class FailureRecoveryManager:
    """Manages failure recovery strategies and execution"""
    
    def __init__(self, meta_reasoning_engine):
        self.meta_reasoning_engine = meta_reasoning_engine
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.recovery_history: List[Dict[str, Any]] = []
        self.circuit_breakers: Dict[ReasoningEngine, CircuitBreaker] = {}
        self.enabled = True
        
        # Initialize circuit breakers for all engines
        for engine_type in ReasoningEngine:
            self.circuit_breakers[engine_type] = CircuitBreaker(engine_type)
        
        # Recovery statistics
        self.recovery_stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_success_rate": 0.0
        }
    
    def _initialize_recovery_strategies(self) -> Dict[FailureType, RecoveryStrategy]:
        """Initialize recovery strategies for different failure types"""
        return {
            FailureType.TIMEOUT: RecoveryStrategy(
                strategy_id="timeout_recovery",
                name="Timeout Recovery Strategy",
                description="Handle timeout failures with retry and fallback",
                failure_types=[FailureType.TIMEOUT],
                severity_threshold=5,
                frequency_threshold=2,
                time_window_seconds=300,
                primary_action=RecoveryAction.RETRY,
                fallback_actions=[RecoveryAction.FALLBACK_ENGINE, RecoveryAction.CIRCUIT_BREAKER],
                max_retries=2,
                retry_delay_seconds=2.0,
                exponential_backoff=True,
                enabled=True,
                priority=8
            ),
            FailureType.API_ERROR: RecoveryStrategy(
                strategy_id="api_error_recovery",
                name="API Error Recovery Strategy", 
                description="Handle API errors with retry and engine switching",
                failure_types=[FailureType.API_ERROR],
                severity_threshold=4,
                frequency_threshold=3,
                time_window_seconds=300,
                primary_action=RecoveryAction.RETRY,
                fallback_actions=[RecoveryAction.FALLBACK_ENGINE, RecoveryAction.REDUCE_LOAD],
                max_retries=3,
                retry_delay_seconds=1.0,
                exponential_backoff=True,
                enabled=True,
                priority=6
            ),
            FailureType.RATE_LIMIT: RecoveryStrategy(
                strategy_id="rate_limit_recovery",
                name="Rate Limit Recovery Strategy",
                description="Handle rate limiting with delays and load reduction",
                failure_types=[FailureType.RATE_LIMIT],
                severity_threshold=3,
                frequency_threshold=1,
                time_window_seconds=60,
                primary_action=RecoveryAction.REDUCE_LOAD,
                fallback_actions=[RecoveryAction.FALLBACK_ENGINE, RecoveryAction.CIRCUIT_BREAKER],
                max_retries=1,
                retry_delay_seconds=5.0,
                exponential_backoff=False,
                enabled=True,
                priority=7
            ),
            FailureType.RESOURCE_EXHAUSTION: RecoveryStrategy(
                strategy_id="resource_recovery",
                name="Resource Exhaustion Recovery Strategy",
                description="Handle resource exhaustion with restart and load reduction",
                failure_types=[FailureType.RESOURCE_EXHAUSTION],
                severity_threshold=7,
                frequency_threshold=1,
                time_window_seconds=120,
                primary_action=RecoveryAction.RESTART,
                fallback_actions=[RecoveryAction.REDUCE_LOAD, RecoveryAction.MAINTENANCE_MODE],
                max_retries=1,
                retry_delay_seconds=10.0,
                exponential_backoff=False,
                enabled=True,
                priority=9
            ),
            FailureType.SYSTEM_ERROR: RecoveryStrategy(
                strategy_id="system_error_recovery",
                name="System Error Recovery Strategy",
                description="Handle repeated failures with circuit breaking and isolation",
                failure_types=[FailureType.SYSTEM_ERROR],
                severity_threshold=8,
                frequency_threshold=1,
                time_window_seconds=60,
                primary_action=RecoveryAction.CIRCUIT_BREAKER,
                fallback_actions=[RecoveryAction.MAINTENANCE_MODE, RecoveryAction.ESCALATE],
                max_retries=1,
                retry_delay_seconds=30.0,
                exponential_backoff=False,
                enabled=True,
                priority=10
            )
        }
    
    async def attempt_recovery(self, failure_event: FailureEvent) -> bool:
        """Attempt to recover from a failure"""
        if not self.enabled:
            return False
        
        # Extract engine type from engine_id
        engine_type = self._extract_engine_type(failure_event.engine_id)
        if not engine_type:
            return False
        
        strategy = self.recovery_strategies.get(failure_event.failure_type)
        if not strategy:
            logger.warning(f"No recovery strategy for failure type: {failure_event.failure_type}")
            return False
        
        # Check if strategy should trigger
        recent_failures = self._get_recent_failures(engine_type, strategy.time_window_seconds)
        if not strategy.should_trigger(failure_event, recent_failures):
            return False
        
        recovery_record = {
            "timestamp": datetime.now(timezone.utc),
            "failure_event": failure_event.to_dict(),
            "strategy": strategy.strategy_id,
            "attempts": [],
            "success": False
        }
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(engine_type)
        if circuit_breaker and not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker preventing recovery for {engine_type.value}")
            return False
        
        # Attempt recovery actions
        success = False
        for attempt in range(strategy.max_retries + 1):
            action = strategy.primary_action if attempt == 0 else (
                strategy.fallback_actions[min(attempt-1, len(strategy.fallback_actions)-1)] 
                if strategy.fallback_actions else strategy.primary_action
            )
            
            logger.info(f"Attempting recovery action {action.value} for {engine_type.value}")
            
            try:
                action_success = await self._execute_recovery_action(
                    engine_type, action, failure_event, strategy
                )
                
                recovery_record["attempts"].append({
                    "attempt": attempt,
                    "action": action.value,
                    "success": action_success,
                    "timestamp": datetime.now(timezone.utc)
                })
                
                if action_success:
                    recovery_record["success"] = True
                    failure_event.recovery_attempted = True
                    failure_event.recovery_action = action
                    failure_event.recovery_successful = True
                    
                    self.recovery_stats["successful_recoveries"] += 1
                    self._update_recovery_success_rate()
                    
                    # Record success in circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    logger.info(f"Recovery successful for {engine_type.value} using {action.value}")
                    success = True
                    break
                
                # Wait before next attempt with exponential backoff
                if attempt < strategy.max_retries:
                    delay = strategy.retry_delay_seconds
                    if strategy.exponential_backoff:
                        delay *= (2 ** attempt)
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Recovery action {action.value} failed: {str(e)}")
                recovery_record["attempts"].append({
                    "attempt": attempt,
                    "action": action.value,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc)
                })
        
        if not success:
            self.recovery_stats["failed_recoveries"] += 1
            self._update_recovery_success_rate()
            
            # Record failure in circuit breaker
            if circuit_breaker:
                circuit_breaker.record_failure()
        
        self.recovery_stats["total_recoveries"] += 1
        self.recovery_history.append(recovery_record)
        
        # Keep only recent history (last 1000 records)
        if len(self.recovery_history) > 1000:
            self.recovery_history = self.recovery_history[-1000:]
        
        return success
    
    def _extract_engine_type(self, engine_id: str) -> Optional[ReasoningEngine]:
        """Extract engine type from engine ID"""
        for engine_type in ReasoningEngine:
            if engine_type.value in engine_id:
                return engine_type
        return None
    
    def _get_recent_failures(self, engine_type: ReasoningEngine, time_window_seconds: int) -> List[FailureEvent]:
        """Get recent failures for an engine"""
        if not hasattr(self.meta_reasoning_engine, 'failure_detector'):
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=time_window_seconds)
        return [
            event for event in self.meta_reasoning_engine.failure_detector.failure_events
            if event.engine_id == f"{engine_type.value}_engine" and event.timestamp >= cutoff_time
        ]
    
    async def _execute_recovery_action(self, engine_type: ReasoningEngine, 
                                     action: RecoveryAction, failure_event: FailureEvent,
                                     strategy: RecoveryStrategy) -> bool:
        """Execute a specific recovery action"""
        
        if action == RecoveryAction.RETRY:
            # Simple retry - just return True to indicate we can try again
            return True
        
        elif action == RecoveryAction.FALLBACK_ENGINE:
            # Switch to a different engine temporarily
            return await self._fallback_engine_strategy(engine_type, failure_event)
        
        elif action == RecoveryAction.REDUCE_LOAD:
            # Reduce load on the engine
            return self._reduce_engine_load(engine_type)
        
        elif action == RecoveryAction.CIRCUIT_BREAKER:
            # Activate circuit breaker
            circuit_breaker = self.circuit_breakers.get(engine_type)
            if circuit_breaker:
                circuit_breaker._transition_to_open()
                return True
            return False
        
        elif action == RecoveryAction.RESTART:
            # Restart the engine (simulated)
            return await self._restart_engine(engine_type)
        
        elif action == RecoveryAction.MAINTENANCE_MODE:
            # Put engine in maintenance mode
            return self._enable_maintenance_mode(engine_type)
        
        elif action == RecoveryAction.ESCALATE:
            # Escalate to human operators
            return self._escalate_failure(engine_type, failure_event)
        
        elif action == RecoveryAction.IGNORE:
            # Ignore the failure
            return True
        
        else:
            logger.warning(f"Unknown recovery action: {action}")
            return False
    
    async def _fallback_engine_strategy(self, engine_type: ReasoningEngine, failure_event: FailureEvent) -> bool:
        """Implement fallback to alternative engine"""
        # Find healthy alternative engines
        available_engines = []
        for alt_engine in ReasoningEngine:
            if alt_engine != engine_type:
                circuit_breaker = self.circuit_breakers.get(alt_engine)
                if not circuit_breaker or circuit_breaker.can_execute():
                    available_engines.append(alt_engine)
        
        if available_engines:
            logger.info(f"Fallback engines available for {engine_type.value}: {[e.value for e in available_engines]}")
            return True
        else:
            logger.warning(f"No fallback engines available for {engine_type.value}")
            return False
    
    def _reduce_engine_load(self, engine_type: ReasoningEngine) -> bool:
        """Reduce load on an engine"""
        try:
            # If load balancer exists, reduce weight for this engine
            if hasattr(self.meta_reasoning_engine, 'load_balancer'):
                current_weight = self.meta_reasoning_engine.load_balancer.engine_weights.get(engine_type, 1.0)
                self.meta_reasoning_engine.load_balancer.engine_weights[engine_type] = current_weight * 0.5
                logger.info(f"Reduced load weight for {engine_type.value} to {current_weight * 0.5}")
            return True
        except Exception as e:
            logger.error(f"Failed to reduce load for {engine_type.value}: {str(e)}")
            return False
    
    async def _restart_engine(self, engine_type: ReasoningEngine) -> bool:
        """Restart a reasoning engine (simulated)"""
        try:
            logger.info(f"Simulating restart for engine {engine_type.value}")
            
            # Reset circuit breaker
            circuit_breaker = self.circuit_breakers.get(engine_type)
            if circuit_breaker:
                circuit_breaker.reset()
            
            # Reset health metrics if available
            if hasattr(self.meta_reasoning_engine, 'health_monitor'):
                self.meta_reasoning_engine.health_monitor.reset_engine_metrics(engine_type)
            
            # Simulate restart delay
            await asyncio.sleep(1.0)
            
            logger.info(f"Engine {engine_type.value} restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart engine {engine_type.value}: {str(e)}")
            return False
    
    def _enable_maintenance_mode(self, engine_type: ReasoningEngine) -> bool:
        """Enable maintenance mode for an engine"""
        try:
            # Mark engine as in maintenance
            if hasattr(self.meta_reasoning_engine, 'health_monitor'):
                # Set engine health status to maintenance
                if engine_type in self.meta_reasoning_engine.health_monitor.engine_metrics:
                    metrics = self.meta_reasoning_engine.health_monitor.engine_metrics[engine_type]
                    metrics.health_status = "maintenance"
            
            logger.info(f"Engine {engine_type.value} set to maintenance mode")
            return True
        except Exception as e:
            logger.error(f"Failed to set maintenance mode for {engine_type.value}: {str(e)}")
            return False
    
    def _escalate_failure(self, engine_type: ReasoningEngine, failure_event: FailureEvent) -> bool:
        """Escalate failure to human operators"""
        logger.critical(f"ESCALATED: Engine {engine_type.value} failure requires human intervention")
        logger.critical(f"Failure details: {failure_event.error_message}")
        
        # In a real system, this would notify operations team
        # For now, just log the escalation
        return True
    
    def _update_recovery_success_rate(self):
        """Update recovery success rate"""
        if self.recovery_stats["total_recoveries"] > 0:
            self.recovery_stats["recovery_success_rate"] = (
                self.recovery_stats["successful_recoveries"] / 
                self.recovery_stats["total_recoveries"]
            )
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        circuit_breaker_stats = {
            engine.value: breaker.get_status()
            for engine, breaker in self.circuit_breakers.items()
        }
        
        return {
            **self.recovery_stats,
            "circuit_breakers": circuit_breaker_stats,
            "recovery_history_count": len(self.recovery_history),
            "enabled": self.enabled,
            "strategies": {
                failure_type.value: {
                    "name": strategy.name,
                    "enabled": strategy.enabled,
                    "priority": strategy.priority
                }
                for failure_type, strategy in self.recovery_strategies.items()
            }
        }
    
    def get_circuit_breaker_status(self, engine_type: ReasoningEngine) -> Dict[str, Any]:
        """Get circuit breaker status for specific engine"""
        circuit_breaker = self.circuit_breakers.get(engine_type)
        if circuit_breaker:
            return circuit_breaker.get_status()
        return {"error": "Circuit breaker not found"}
    
    def reset_recovery_history(self):
        """Reset recovery history"""
        self.recovery_history = []
        self.recovery_stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_success_rate": 0.0
        }
    
    def reset_circuit_breaker(self, engine_type: ReasoningEngine):
        """Reset circuit breaker for specific engine"""
        circuit_breaker = self.circuit_breakers.get(engine_type)
        if circuit_breaker:
            circuit_breaker.reset()
    
    def enable_recovery(self):
        """Enable failure recovery"""
        self.enabled = True
    
    def disable_recovery(self):
        """Disable failure recovery"""
        self.enabled = False


# Export classes for use in other modules
__all__ = [
    'FailureDetector',
    'CircuitBreaker',
    'FailureRecoveryManager'
]
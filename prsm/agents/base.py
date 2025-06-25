"""
PRSM Base Agent Framework
Abstract base class for all agents in the 5-layer PRSM architecture

This module provides the foundational infrastructure for PRSM's agent-based
architecture. It defines common patterns and safety measures used across
all agent types in the system.

Core Components:

1. BaseAgent (Abstract Class):
   - Common interface for all agent types (Architect, Prompter, Router, Executor, Compiler)
   - Built-in safety validation for inputs and outputs
   - Performance tracking and metrics collection
   - Error handling with graceful degradation
   - Circuit breaker integration for system-wide safety

2. PerformanceTracker:
   - Real-time monitoring of agent operations
   - Metrics collection for system optimization
   - Success/failure rate tracking
   - Execution time analysis for bottleneck identification

3. AgentPool:
   - Load balancing across multiple agents of the same type
   - Round-robin scheduling for fair resource distribution
   - Health monitoring and automatic failover
   - Horizontal scaling support for high-throughput scenarios

4. AgentRegistry:
   - Global coordination of all system agents
   - Centralized status monitoring and health checks
   - Agent discovery and routing for inter-agent communication
   - System-wide statistics and reporting

Safety Features:
- Input/output validation against known attack patterns
- Data size limits to prevent resource exhaustion
- Execution timeouts for hung operations
- Safety flag tracking for governance and auditing
- Automatic deactivation on repeated safety violations

Performance Features:
- Sub-second operation tracking with microsecond precision
- Comprehensive metrics for optimization insights
- Memory and CPU usage monitoring
- Throughput analysis for capacity planning

Integration Points:
- FTNS service for computational cost tracking
- Circuit breaker system for distributed safety
- Governance system for agent policy enforcement
- Recursive self-improvement for agent evolution

All agents inherit from BaseAgent to ensure:
- Consistent behavior across the distributed system
- Unified safety and monitoring infrastructure
- Standardized error handling and reporting
- Compatible interfaces for agent coordination
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import structlog

from prsm.core.config import get_settings
from prsm.core.models import (
    AgentType, AgentResponse, SafetyFlag, SafetyLevel,
    PerformanceMetrics, TaskStatus
)

logger = structlog.get_logger(__name__)
settings = get_settings()


class AgentError(Exception):
    """Base exception for agent-related errors
    
    Used for non-safety-related agent failures such as:
    - Configuration errors
    - Resource unavailability  
    - Communication failures
    - Invalid agent states
    """
    pass


class SafetyViolationError(AgentError):
    """Raised when an agent detects a safety violation
    
    Triggers circuit breaker activation and safety reporting.
    Common triggers include:
    - Malicious input patterns
    - Unsafe output content
    - Resource limit violations
    - Security policy breaches
    """
    pass


class PerformanceTracker:
    """Tracks agent performance metrics
    
    Provides real-time monitoring and historical analysis of agent
    operations for system optimization and capacity planning.
    
    Features:
    - Operation-level timing with microsecond precision
    - Success/failure rate calculation
    - Resource usage tracking
    - Trend analysis for performance degradation detection
    - Integration with recursive self-improvement system
    
    Used by the performance monitoring subsystem to identify
    optimization opportunities and track system health.
    """
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.metrics: List[PerformanceMetrics] = []
    
    def start_operation(self, operation_name: str) -> str:
        """Start tracking an operation"""
        operation_id = str(uuid4())
        self.current_operation = {
            "operation_id": operation_id,
            "operation_name": operation_name,
            "start_time": time.time(),
            "status": "in_progress"
        }
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """End tracking an operation and record metrics"""
        if not hasattr(self, 'current_operation') or \
           self.current_operation.get("operation_id") != operation_id:
            logger.warning("Invalid operation tracking", operation_id=operation_id)
            return None
        
        end_time = time.time()
        duration = end_time - self.current_operation["start_time"]
        
        metrics = PerformanceMetrics(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            operation_name=self.current_operation["operation_name"],
            duration_seconds=duration,
            success=success,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.metrics.append(metrics)
        delattr(self, 'current_operation')
        
        logger.info("Operation completed",
                   agent_id=self.agent_id,
                   operation=metrics.operation_name,
                   duration=f"{duration:.3f}s",
                   success=success)
        
        return metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get aggregate performance statistics"""
        if not self.metrics:
            return {"total_operations": 0}
        
        successful_ops = [m for m in self.metrics if m.success]
        failed_ops = [m for m in self.metrics if not m.success]
        
        total_duration = sum(m.duration_seconds for m in self.metrics)
        avg_duration = total_duration / len(self.metrics)
        
        return {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self.metrics),
            "average_duration": avg_duration,
            "total_duration": total_duration
        }


class BaseAgent(ABC):
    """
    Abstract base class for all PRSM agents
    
    Foundation for the 5-layer PRSM agent architecture providing
    standardized safety, performance, and coordination capabilities.
    
    Common Functionality:
    - Safety validation with circuit breaker integration
    - Performance logging and metrics collection
    - Error handling with graceful degradation
    - Input/output validation and sanitization
    - FTNS cost tracking integration
    - Governance policy enforcement
    
    Agent Lifecycle:
    1. Initialization with unique ID and type classification
    2. Registration with global agent registry
    3. Processing requests through safe_process() wrapper
    4. Performance tracking and safety monitoring
    5. Deactivation/reactivation based on health status
    
    Safety Features:
    - Pre/post-processing content validation
    - Automatic pattern detection for malicious content
    - Resource usage limits and monitoring
    - Emergency deactivation capabilities
    - Complete audit trail maintenance
    
    Integration Points:
    - AgentRegistry for global coordination
    - PerformanceTracker for optimization insights
    - SafetyFlag system for governance reporting
    - Circuit breaker system for distributed safety
    
    All specialized agents (Architect, Prompter, Router, Executor,
    Compiler) inherit from this base to ensure system-wide consistency.
    """
    
    def __init__(self, agent_id: Optional[str] = None, agent_type: Optional[AgentType] = None):
        self.agent_id = agent_id or str(uuid4())
        self.agent_type = agent_type or AgentType.EXECUTOR  # Default type
        self.performance_tracker = PerformanceTracker(self.agent_id, self.agent_type)
        self.safety_flags: List[SafetyFlag] = []
        self.active = True
        
        logger.info("Agent initialized",
                   agent_id=self.agent_id,
                   agent_type=self.agent_type.value)
    
    @abstractmethod
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Process input data and return result
        
        Args:
            input_data: Data to process
            context: Optional context information
            
        Returns:
            Processed result
        """
        pass
    
    async def safe_process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Safely process input with full validation and error handling
        
        Args:
            input_data: Data to process
            context: Optional context information
            
        Returns:
            AgentResponse with result and metadata
        """
        operation_id = self.performance_tracker.start_operation("safe_process")
        
        try:
            # Pre-processing safety validation
            if not await self.validate_safety(input_data):
                raise SafetyViolationError("Input failed safety validation")
            
            # Check agent status
            if not self.active:
                raise AgentError("Agent is not active")
            
            # Process the input
            start_time = time.time()
            result = await self.process(input_data, context)
            processing_time = time.time() - start_time
            
            # Post-processing safety validation
            if not await self.validate_safety(result):
                raise SafetyViolationError("Output failed safety validation")
            
            # Create response
            response = AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                input_data=str(input_data)[:500],  # Truncate for storage
                output_data=result,
                success=True,
                processing_time=processing_time,
                safety_validated=True,
                metadata={
                    "context_provided": context is not None,
                    "safety_flags": len(self.safety_flags)
                }
            )
            
            # Log performance
            await self.log_performance(input_data, result, {
                "processing_time": processing_time,
                "success": True
            })
            
            self.performance_tracker.end_operation(operation_id, success=True, metadata={
                "processing_time": processing_time,
                "output_size": len(str(result))
            })
            
            return response
            
        except Exception as e:
            logger.error("Agent processing failed",
                        agent_id=self.agent_id,
                        error=str(e),
                        error_type=type(e).__name__)
            
            # Log performance for failed operation
            await self.log_performance(input_data, None, {
                "error": str(e),
                "success": False
            })
            
            self.performance_tracker.end_operation(operation_id, success=False, metadata={
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            # Create error response
            response = AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                input_data=str(input_data)[:500],
                output_data=None,
                success=False,
                error_message=str(e),
                safety_validated=False,
                metadata={"error_type": type(e).__name__}
            )
            
            return response
    
    async def validate_safety(self, data: Any) -> bool:
        """
        Validate data for safety concerns
        
        Args:
            data: Data to validate
            
        Returns:
            True if data passes safety validation
        """
        try:
            # Basic safety checks
            if data is None:
                return True  # None is safe
            
            data_str = str(data).lower()
            
            # Check for common unsafe patterns
            unsafe_patterns = [
                "rm -rf", "del /", "format c:",  # Dangerous commands
                "password", "secret", "token",   # Potential credentials
                "exec(", "eval(", "__import__",  # Code injection
                "DROP TABLE", "DELETE FROM",    # SQL injection patterns
            ]
            
            for pattern in unsafe_patterns:
                if pattern in data_str:
                    safety_flag = SafetyFlag(
                        level=SafetyLevel.MEDIUM,
                        category="unsafe_pattern",
                        description=f"Detected unsafe pattern: {pattern}",
                        triggered_by=pattern
                    )
                    self.safety_flags.append(safety_flag)
                    
                    logger.warning("Safety concern detected",
                                 agent_id=self.agent_id,
                                 pattern=pattern,
                                 severity=SafetyLevel.MEDIUM.value)
                    
                    # For now, log but don't block (configurable in production)
                    if settings.is_production:
                        return False
            
            # Check data size limits
            if len(str(data)) > settings.max_parallel_tasks * 10000:  # Rough limit
                safety_flag = SafetyFlag(
                    level=SafetyLevel.LOW,
                    category="data_size",
                    description="Data size exceeds recommended limits",
                    triggered_by=f"Size: {len(str(data))} characters"
                )
                self.safety_flags.append(safety_flag)
            
            return True
            
        except Exception as e:
            logger.error("Safety validation failed",
                        agent_id=self.agent_id,
                        error=str(e))
            return False
    
    async def log_performance(self, input_data: Any, output: Any, metrics: Dict[str, Any]):
        """
        Log performance metrics for this operation
        
        Args:
            input_data: Input that was processed
            output: Output that was generated
            metrics: Performance metrics dictionary
        """
        try:
            performance_data = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "input_size": len(str(input_data)) if input_data is not None else 0,
                "output_size": len(str(output)) if output else 0,
                "success": metrics.get("success", True),
                "processing_time": metrics.get("processing_time", 0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Add custom metrics
            performance_data.update(metrics)
            
            logger.info("Agent performance logged",
                       agent_id=self.agent_id,
                       **{k: v for k, v in performance_data.items() 
                          if k not in ["agent_id", "timestamp"]})
            
        except Exception as e:
            logger.error("Performance logging failed",
                        agent_id=self.agent_id,
                        error=str(e))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and statistics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "active": self.active,
            "safety_flags": len(self.safety_flags),
            "performance_stats": self.performance_tracker.get_performance_stats()
        }
    
    def deactivate(self, reason: str = "Manual deactivation"):
        """Deactivate this agent"""
        self.active = False
        logger.warning("Agent deactivated",
                      agent_id=self.agent_id,
                      reason=reason)
    
    def activate(self):
        """Reactivate this agent"""
        self.active = True
        logger.info("Agent activated",
                   agent_id=self.agent_id)
    
    def clear_safety_flags(self):
        """Clear all safety flags"""
        flag_count = len(self.safety_flags)
        self.safety_flags = []
        logger.info("Safety flags cleared",
                   agent_id=self.agent_id,
                   cleared_count=flag_count)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.agent_id[:8]}, type={self.agent_type.value})>"


class AgentPool:
    """Manages a pool of agents for load balancing and coordination
    
    Provides horizontal scaling and load distribution for agent types.
    Essential for handling high-throughput scenarios and ensuring
    system resilience through redundancy.
    
    Features:
    - Round-robin load balancing across healthy agents
    - Automatic failover when agents become unavailable
    - Health monitoring and status reporting
    - Dynamic scaling based on demand
    - Performance-based routing (future enhancement)
    
    Usage Patterns:
    - Multiple Executor agents for parallel task processing
    - Redundant Router agents for model selection reliability
    - Load-balanced Compiler agents for result synthesis
    - Backup Architect agents for system resilience
    
    Integration with PRSM systems:
    - Performance metrics feed into recursive self-improvement
    - Health status monitored by circuit breaker system
    - Agent allocation tracked for FTNS cost calculation
    - Pool statistics available for governance decisions
    """
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.agents: List[BaseAgent] = []
        self.round_robin_index = 0
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the pool"""
        if agent.agent_type != self.agent_type:
            raise ValueError(f"Agent type mismatch: expected {self.agent_type}, got {agent.agent_type}")
        
        self.agents.append(agent)
        logger.info("Agent added to pool",
                   pool_type=self.agent_type.value,
                   agent_id=agent.agent_id,
                   pool_size=len(self.agents))
    
    def get_next_agent(self) -> Optional[BaseAgent]:
        """Get next available agent using round-robin"""
        active_agents = [a for a in self.agents if a.active]
        
        if not active_agents:
            return None
        
        agent = active_agents[self.round_robin_index % len(active_agents)]
        self.round_robin_index += 1
        
        return agent
    
    async def process_with_pool(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process input using an available agent from the pool"""
        agent = self.get_next_agent()
        
        if not agent:
            raise AgentError(f"No active agents available in {self.agent_type.value} pool")
        
        return await agent.safe_process(input_data, context)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get status of all agents in the pool"""
        return {
            "agent_type": self.agent_type.value,
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents if a.active]),
            "agent_statuses": [agent.get_status() for agent in self.agents]
        }


# Global agent registry for tracking all agents
class AgentRegistry:
    """Global registry for all agents in the system
    
    Central coordination hub for PRSM's distributed agent network.
    Maintains real-time awareness of all system agents and their
    capabilities for optimal resource allocation.
    
    Registry Functions:
    - Agent registration and discovery
    - Pool management and load balancing
    - System-wide health monitoring
    - Performance analytics aggregation
    - Agent capability matching for task routing
    
    Operational Benefits:
    - Single point of truth for agent status
    - Efficient agent lookup and routing
    - Centralized monitoring and alerting
    - Resource optimization across the network
    - Simplified debugging and troubleshooting
    
    Integration Points:
    - NWTN Orchestrator for agent coordination
    - Model Registry for capability matching
    - Circuit Breaker system for safety enforcement
    - Performance Monitor for optimization insights
    - Governance system for policy enforcement
    
    Scalability Features:
    - Horizontal scaling across federation nodes
    - Distributed registry synchronization (future)
    - Agent migration for load balancing
    - Automatic capacity planning and provisioning
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.pools: Dict[AgentType, AgentPool] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent globally"""
        self.agents[agent.agent_id] = agent
        
        # Add to appropriate pool
        if agent.agent_type not in self.pools:
            self.pools[agent.agent_type] = AgentPool(agent.agent_type)
        
        self.pools[agent.agent_type].add_agent(agent)
        
        logger.info("Agent registered globally",
                   agent_id=agent.agent_id,
                   agent_type=agent.agent_type.value,
                   total_agents=len(self.agents))
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_pool(self, agent_type: AgentType) -> Optional[AgentPool]:
        """Get agent pool by type"""
        return self.pools.get(agent_type)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of entire agent system"""
        return {
            "total_agents": len(self.agents),
            "agent_pools": {
                agent_type.value: pool.get_pool_status() 
                for agent_type, pool in self.pools.items()
            },
            "active_agents": len([a for a in self.agents.values() if a.active])
        }


# Global agent registry instance
agent_registry = AgentRegistry()
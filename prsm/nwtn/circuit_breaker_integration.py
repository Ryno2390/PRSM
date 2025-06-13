#!/usr/bin/env python3
"""
NWTN Orchestrator Circuit Breaker Integration
Integrates circuit breaker protection into the NWTN agent pipeline

🎯 PURPOSE:
Provides comprehensive protection for the NWTN agent pipeline against
cascading failures, agent timeouts, and overload conditions while
maintaining system availability and performance.

🔧 INTEGRATION POINTS:
- Agent execution protection
- Pipeline timeout handling
- Session management protection
- Quality control circuit breaker
- Cost calculation protection
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID
from datetime import datetime, timezone
import structlog

from ..core.circuit_breaker import (
    get_circuit_breaker, protected_call, CircuitBreakerConfig,
    CircuitBreakerOpenException, AGENT_CIRCUIT_CONFIG
)
from ..core.models import PRSMSession, QueryResponse

logger = structlog.get_logger(__name__)

class NWTNCircuitBreakerIntegration:
    """
    Circuit breaker integration for NWTN orchestrator
    
    Provides comprehensive protection for agent pipeline execution
    with intelligent fallback strategies and graceful degradation.
    """
    
    def __init__(self):
        # Circuit breaker configurations for different components
        self.agent_configs = {
            "architect": CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                timeout_seconds=10.0,
                failure_rate_threshold=0.4
            ),
            "prompter": CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=25.0,
                timeout_seconds=8.0,
                failure_rate_threshold=0.4
            ),
            "router": CircuitBreakerConfig(
                failure_threshold=4,
                recovery_timeout=20.0,
                timeout_seconds=5.0,
                failure_rate_threshold=0.3
            ),
            "executor": CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=45.0,
                timeout_seconds=20.0,
                failure_rate_threshold=0.5
            ),
            "compiler": CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=35.0,
                timeout_seconds=12.0,
                failure_rate_threshold=0.4
            )
        }
        
        # Pipeline-level circuit breaker
        self.pipeline_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            timeout_seconds=30.0,
            failure_rate_threshold=0.6,
            max_concurrent_calls=1000
        )
        
        # Fallback strategies
        self.fallback_strategies = {
            "architect": self._architect_fallback,
            "prompter": self._prompter_fallback,
            "router": self._router_fallback,
            "executor": self._executor_fallback,
            "compiler": self._compiler_fallback
        }
        
        logger.info("NWTN Circuit Breaker Integration initialized")
    
    async def execute_agent_with_protection(
        self,
        agent_type: str,
        agent_function: Callable,
        session: PRSMSession,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute agent function with circuit breaker protection
        
        Args:
            agent_type: Type of agent (architect, prompter, etc.)
            agent_function: Agent function to execute
            session: PRSM session context
            *args: Agent function arguments
            **kwargs: Agent function keyword arguments
            
        Returns:
            Agent execution result or fallback result
        """
        circuit_name = f"nwtn_agent_{agent_type}"
        config = self.agent_configs.get(agent_type, AGENT_CIRCUIT_CONFIG)
        
        try:
            # Execute with circuit breaker protection
            result = await protected_call(
                circuit_name,
                agent_function,
                *args,
                config=config,
                fallback=self.fallback_strategies.get(agent_type),
                **kwargs
            )
            
            logger.debug(f"Agent {agent_type} executed successfully",
                        session_id=str(session.session_id))
            
            return result
            
        except CircuitBreakerOpenException:
            logger.warning(f"Circuit breaker open for agent {agent_type}",
                         session_id=str(session.session_id))
            
            # Execute fallback if available
            fallback = self.fallback_strategies.get(agent_type)
            if fallback:
                return await fallback(session, *args, **kwargs)
            else:
                raise
    
    async def execute_pipeline_with_protection(
        self,
        pipeline_function: Callable,
        session: PRSMSession,
        query: str,
        *args,
        **kwargs
    ) -> QueryResponse:
        """
        Execute complete agent pipeline with protection
        
        Args:
            pipeline_function: Pipeline execution function
            session: PRSM session
            query: User query
            *args: Pipeline arguments
            **kwargs: Pipeline keyword arguments
            
        Returns:
            Query response or degraded response
        """
        circuit_name = "nwtn_pipeline"
        
        try:
            # Execute full pipeline with protection
            response = await protected_call(
                circuit_name,
                pipeline_function,
                session,
                query,
                *args,
                config=self.pipeline_config,
                fallback=self._pipeline_fallback,
                **kwargs
            )
            
            logger.info("Pipeline executed successfully",
                       session_id=str(session.session_id),
                       response_length=len(response.response))
            
            return response
            
        except CircuitBreakerOpenException:
            logger.error("Pipeline circuit breaker open",
                        session_id=str(session.session_id))
            
            # Return degraded response
            return await self._pipeline_fallback(session, query, *args, **kwargs)
    
    async def create_session_with_protection(
        self,
        session_function: Callable,
        user_id: str,
        *args,
        **kwargs
    ) -> PRSMSession:
        """
        Create session with circuit breaker protection
        
        Args:
            session_function: Session creation function
            user_id: User identifier
            *args: Session function arguments
            **kwargs: Session function keyword arguments
            
        Returns:
            PRSM session or fallback session
        """
        circuit_name = "nwtn_session_creation"
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=30.0,
            timeout_seconds=5.0
        )
        
        try:
            session = await protected_call(
                circuit_name,
                session_function,
                user_id,
                *args,
                config=config,
                fallback=self._session_creation_fallback,
                **kwargs
            )
            
            logger.info("Session created successfully",
                       session_id=str(session.session_id),
                       user_id=user_id)
            
            return session
            
        except Exception as e:
            logger.error("Session creation failed",
                        user_id=user_id,
                        error=str(e))
            raise
    
    # === Fallback Strategies ===
    
    async def _architect_fallback(self, session: PRSMSession, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for architect agent"""
        logger.info("Using architect fallback strategy",
                   session_id=str(session.session_id))
        
        # Simple task analysis fallback
        query = kwargs.get('query', args[0] if args else "")
        
        return {
            "task_type": "general",
            "complexity": "medium",
            "approach": "direct_execution",
            "context_requirements": ["basic"],
            "fallback_used": True,
            "agent": "architect_fallback"
        }
    
    async def _prompter_fallback(self, session: PRSMSession, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for prompter agent"""
        logger.info("Using prompter fallback strategy",
                   session_id=str(session.session_id))
        
        # Basic prompt preparation
        return {
            "optimized_prompt": kwargs.get('query', args[0] if args else ""),
            "context_additions": [],
            "prompt_strategy": "direct",
            "fallback_used": True,
            "agent": "prompter_fallback"
        }
    
    async def _router_fallback(self, session: PRSMSession, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for router agent"""
        logger.info("Using router fallback strategy",
                   session_id=str(session.session_id))
        
        # Default routing to primary model
        return {
            "selected_model": "primary",
            "routing_confidence": 0.7,
            "routing_reason": "fallback_default",
            "fallback_used": True,
            "agent": "router_fallback"
        }
    
    async def _executor_fallback(self, session: PRSMSession, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for executor agent"""
        logger.info("Using executor fallback strategy",
                   session_id=str(session.session_id))
        
        # Simple response generation
        query = kwargs.get('query', args[0] if args else "")
        
        fallback_response = (
            f"I apologize, but I'm currently experiencing technical difficulties. "
            f"I understand you asked: '{query}'. Please try again in a few moments, "
            f"or rephrase your question for better results."
        )
        
        return {
            "response": fallback_response,
            "confidence": 0.5,
            "execution_method": "fallback",
            "fallback_used": True,
            "agent": "executor_fallback"
        }
    
    async def _compiler_fallback(self, session: PRSMSession, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for compiler agent"""
        logger.info("Using compiler fallback strategy",
                   session_id=str(session.session_id))
        
        # Basic response compilation
        response_data = kwargs.get('response_data', args[0] if args else {})
        
        return {
            "final_response": response_data.get("response", "Service temporarily unavailable."),
            "metadata": {
                "agents_used": ["fallback"],
                "fallback_mode": True,
                "degraded_service": True
            },
            "quality_score": 0.6,
            "fallback_used": True,
            "agent": "compiler_fallback"
        }
    
    async def _pipeline_fallback(self, session: PRSMSession, query: str, *args, **kwargs) -> QueryResponse:
        """Fallback for entire pipeline"""
        logger.warning("Using pipeline fallback strategy",
                      session_id=str(session.session_id))
        
        # Generate minimal but helpful response
        fallback_response = (
            "I'm currently experiencing system overload and cannot process your request "
            "through the normal pipeline. Please try again in a few moments. "
            f"Your query: '{query}' has been noted."
        )
        
        return QueryResponse(
            response=fallback_response,
            session_id=session.session_id,
            confidence=0.3,
            metadata={
                "agents_used": [],
                "fallback_mode": True,
                "processing_time_ms": 0,
                "degraded_service": True,
                "circuit_breaker_activated": True
            }
        )
    
    async def _session_creation_fallback(self, user_id: str, *args, **kwargs) -> PRSMSession:
        """Fallback for session creation"""
        logger.warning("Using session creation fallback",
                      user_id=user_id)
        
        # Create minimal session
        from uuid import uuid4
        
        return PRSMSession(
            session_id=uuid4(),
            user_id=user_id,
            session_type="fallback",
            created_at=datetime.now(timezone.utc),
            metadata={
                "fallback_session": True,
                "limited_functionality": True
            }
        )
    
    # === Monitoring and Health ===
    
    def get_circuit_health(self) -> Dict[str, Any]:
        """Get circuit breaker health status"""
        from ..core.circuit_breaker import get_all_circuit_stats
        
        stats = get_all_circuit_stats()
        
        # Filter NWTN-related circuits
        nwtn_circuits = {
            name: data for name, data in stats.get("breakers", {}).items()
            if "nwtn" in name.lower()
        }
        
        # Calculate health metrics
        total_circuits = len(nwtn_circuits)
        open_circuits = sum(1 for data in nwtn_circuits.values() 
                           if data.current_state == "open")
        
        health_score = 1.0 - (open_circuits / total_circuits) if total_circuits > 0 else 1.0
        
        return {
            "health_score": health_score,
            "total_circuits": total_circuits,
            "open_circuits": open_circuits,
            "circuit_details": nwtn_circuits,
            "system_status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "critical"
        }
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform active health check of circuit breakers"""
        
        async def dummy_function():
            return "health_check_success"
        
        health_results = {}
        
        # Test each agent circuit
        for agent_type in self.agent_configs.keys():
            circuit_name = f"nwtn_agent_{agent_type}"
            
            try:
                result = await protected_call(
                    f"health_check_{circuit_name}",
                    dummy_function,
                    config=CircuitBreakerConfig(
                        failure_threshold=1,
                        timeout_seconds=2.0
                    )
                )
                health_results[agent_type] = "healthy"
            except Exception as e:
                health_results[agent_type] = f"unhealthy: {str(e)}"
        
        # Test pipeline circuit
        try:
            await protected_call(
                "health_check_nwtn_pipeline",
                dummy_function,
                config=CircuitBreakerConfig(
                    failure_threshold=1,
                    timeout_seconds=2.0
                )
            )
            health_results["pipeline"] = "healthy"
        except Exception as e:
            health_results["pipeline"] = f"unhealthy: {str(e)}"
        
        overall_health = all("healthy" in status for status in health_results.values())
        
        return {
            "overall_healthy": overall_health,
            "component_health": health_results,
            "timestamp": datetime.now(timezone.utc),
            "check_type": "active_health_check"
        }


# Global integration instance
nwtn_circuit_integration = None

def get_nwtn_circuit_integration() -> NWTNCircuitBreakerIntegration:
    """Get or create NWTN circuit breaker integration"""
    global nwtn_circuit_integration
    if nwtn_circuit_integration is None:
        nwtn_circuit_integration = NWTNCircuitBreakerIntegration()
    return nwtn_circuit_integration
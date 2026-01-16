import asyncio
import time
import structlog
from typing import Dict, Any
from .providers.base import AbstractReasoningProvider
from .providers.cloud import CloudProvider
from .providers.edge import EdgeProvider
from prsm.core.redis_client import redis_manager

logger = structlog.get_logger(__name__)

class System1CandidateGenerator:
    """
    Generates candidate reasoning chains using the Provider Pattern.
    Implements a Circuit Breaker to fallback from Cloud to Edge.
    Circuit state is now externalized to Redis for cluster-wide coordination.
    """
    def __init__(self, use_edge_only: bool = False):
        self.cloud_provider = CloudProvider()
        self.edge_provider = EdgeProvider()
        self.use_edge_only = use_edge_only
        
        # Circuit Breaker Config
        self.failure_threshold = 3
        self.reset_timeout = 60 # seconds
        
        # Redis Keys
        self.key_status = "circuit:cloud:status" # closed, open, half-open
        self.key_failures = "circuit:cloud:failures"
        self.key_last_failure = "circuit:cloud:last_failure"

        # Local Fallback State (if Redis fails)
        self.local_circuit_open = False
        self.local_failure_count = 0
        self.local_last_failure_time = 0

    async def generate_proposal(self, prompt: str, context: str) -> Dict[str, Any]:
        """
        Orchestrates generation, handling fallback logic.
        """
        # 1. Check Circuit Breaker
        if await self._should_use_edge():
            return await self._generate_with_fallback(prompt, context)

        # 2. Try Cloud Provider
        try:
            result = await self.cloud_provider.generate(prompt, context)
            await self._reset_circuit()
            return result
        except Exception as e:
            logger.warning(f"Cloud provider failed: {e}. Activating fallback.")
            await self._record_failure()
            return await self._generate_with_fallback(prompt, context)

    async def _generate_with_fallback(self, prompt: str, context: str) -> Dict[str, Any]:
        """Executes generation using the Edge provider."""
        logger.info("Routing to Edge Provider (Circuit Breaker/Fallback)")
        return await self.edge_provider.generate(prompt, context)

    async def _should_use_edge(self) -> bool:
        """Determines if we should skip cloud and go straight to edge."""
        if self.use_edge_only:
            return True
            
        try:
            if redis_manager.client and redis_manager.client.connected:
                status = await redis_manager.client.redis_client.get(self.key_status)
                if status == "open":
                    last_fail = await redis_manager.client.redis_client.get(self.key_last_failure)
                    if last_fail and (time.time() - float(last_fail) > self.reset_timeout):
                        logger.info("Circuit Breaker Half-Open: Trying Cloud...")
                        return False # Try cloud again (Half-Open)
                    return True # Still open
                return False
        except Exception:
            # Fallback to local state if Redis fails
            pass

        if self.local_circuit_open:
            if time.time() - self.local_last_failure_time > self.reset_timeout:
                logger.info("Circuit Breaker Half-Open (Local): Trying Cloud...")
                return False
            return True
            
        return False

    async def _record_failure(self):
        """Updates failure metrics and trips circuit if needed."""
        now = time.time()
        
        # Redis Update
        try:
            if redis_manager.client and redis_manager.client.connected:
                await redis_manager.client.redis_client.incr(self.key_failures)
                await redis_manager.client.redis_client.set(self.key_last_failure, now)
                
                failures = int(await redis_manager.client.redis_client.get(self.key_failures) or 0)
                if failures >= self.failure_threshold:
                    await redis_manager.client.redis_client.set(self.key_status, "open")
                    logger.error("Circuit Breaker TRIPPED (Redis). All traffic routed to Edge.")
        except Exception:
            pass

        # Local Update
        self.local_failure_count += 1
        self.local_last_failure_time = now
        if self.local_failure_count >= self.failure_threshold:
            self.local_circuit_open = True
            logger.error("Circuit Breaker TRIPPED (Local). All traffic routed to Edge.")

    async def _reset_circuit(self):
        """Resets the circuit breaker on success."""
        # Redis Reset
        try:
            if redis_manager.client and redis_manager.client.connected:
                status = await redis_manager.client.redis_client.get(self.key_status)
                if status == "open":
                    logger.info("Circuit Breaker CLOSED (Redis). Cloud service restored.")
                await redis_manager.client.redis_client.set(self.key_status, "closed")
                await redis_manager.client.redis_client.set(self.key_failures, 0)
        except Exception:
            pass

        # Local Reset
        if self.local_circuit_open:
            logger.info("Circuit Breaker CLOSED (Local). Cloud service restored.")
        self.local_circuit_open = False
        self.local_failure_count = 0

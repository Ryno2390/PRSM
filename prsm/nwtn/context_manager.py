"""
NWTN Context Manager
Manages context allocation, cost calculation, and usage tracking for PRSM
Integrates with FTNS token system for transparent resource management
"""

from typing import Dict, List, Optional, Tuple
from uuid import UUID
import asyncio
import structlog
from datetime import datetime, timezone
import math

from prsm.core.models import PRSMSession, ContextUsage, TaskStatus
from prsm.core.config import get_settings
from prsm.tokenomics.ftns_service import FTNSService

logger = structlog.get_logger(__name__)
settings = get_settings()


class ContextManager:
    """
    NWTN Context Management System
    
    Handles:
    - Context cost calculation based on complexity and depth
    - Usage tracking per session
    - FTNS integration for resource allocation
    - Context optimization recommendations
    """
    
    def __init__(self, ftns_service: Optional[FTNSService] = None):
        self.ftns_service = ftns_service or FTNSService()
        self.session_usage: Dict[UUID, List[ContextUsage]] = {}
        self.cost_multipliers = {
            "simple": 1.0,
            "medium": 2.5,
            "complex": 5.0,
            "research": 8.0,
            "recursive": 12.0
        }
        
    async def calculate_context_cost(
        self, 
        prompt_complexity: float, 
        depth: int,
        intent_category: str = "general",
        estimated_agents: int = 3
    ) -> int:
        """
        Calculate FTNS context cost for a query
        
        Args:
            prompt_complexity: Complexity score 0.0-1.0
            depth: Expected recursion depth
            intent_category: Category for complexity multiplier
            estimated_agents: Number of agents likely to be involved
            
        Returns:
            int: FTNS tokens required for context
        """
        try:
            # Base cost calculation
            base_cost = math.ceil(prompt_complexity * 100)
            
            # Depth multiplier (exponential scaling)
            depth_multiplier = 1.5 ** max(0, depth - 1)
            
            # Category multiplier
            category_multiplier = self.cost_multipliers.get(intent_category, 1.0)
            
            # Agent coordination overhead
            agent_multiplier = 1.0 + (estimated_agents - 1) * 0.3
            
            # Calculate final cost
            total_cost = int(base_cost * depth_multiplier * category_multiplier * agent_multiplier)
            
            # Minimum cost threshold
            total_cost = max(total_cost, settings.nwtn_min_context_cost or 10)
            
            logger.info("Context cost calculated",
                       complexity=prompt_complexity,
                       depth=depth,
                       category=intent_category,
                       agents=estimated_agents,
                       base_cost=base_cost,
                       total_cost=total_cost)
            
            return total_cost
            
        except Exception as e:
            logger.error("Context cost calculation failed", error=str(e))
            return settings.nwtn_min_context_cost or 10
    
    async def allocate_context(
        self, 
        session: PRSMSession, 
        required_context: int
    ) -> bool:
        """
        Allocate context for a session using FTNS
        
        Args:
            session: PRSM session requiring context
            required_context: Amount of context needed
            
        Returns:
            bool: True if allocation successful
        """
        try:
            # Check if user has sufficient allocation
            if session.nwtn_context_allocation < required_context:
                logger.warning("Insufficient context allocation",
                             session_id=session.session_id,
                             allocated=session.nwtn_context_allocation,
                             required=required_context)
                return False
            
            # Check user's FTNS balance if enabled
            if settings.ftns_enabled:
                ftns_cost = await self._calculate_ftns_cost(required_context)
                balance_obj = await self.ftns_service.get_user_balance(session.user_id)
                balance = balance_obj.balance
                
                if balance < ftns_cost:
                    logger.warning("Insufficient FTNS balance",
                                 user_id=session.user_id,
                                 balance=balance,
                                 required=ftns_cost)
                    return False
            
            # Record allocation
            usage = ContextUsage(
                session_id=session.session_id,
                user_id=session.user_id,
                context_allocated=required_context,
                context_used=0,
                ftns_cost=ftns_cost if settings.ftns_enabled else 0.0,
                allocation_time=datetime.now(timezone.utc)
            )
            
            if session.session_id not in self.session_usage:
                self.session_usage[session.session_id] = []
            self.session_usage[session.session_id].append(usage)
            
            logger.info("Context allocated successfully",
                       session_id=session.session_id,
                       context=required_context,
                       ftns_cost=ftns_cost if settings.ftns_enabled else 0.0)
            
            return True
            
        except Exception as e:
            logger.error("Context allocation failed",
                        session_id=session.session_id,
                        error=str(e))
            return False
    
    async def track_context_usage(
        self, 
        session_id: UUID, 
        context_used: int,
        stage: str = "processing"
    ) -> bool:
        """
        Track actual context usage during processing
        
        Args:
            session_id: Session using context
            context_used: Amount of context consumed
            stage: Processing stage for tracking
            
        Returns:
            bool: True if tracking successful
        """
        try:
            if session_id not in self.session_usage:
                logger.warning("No allocation found for session", session_id=session_id)
                return False
            
            # Find the most recent allocation
            current_usage = self.session_usage[session_id][-1]
            current_usage.context_used += context_used
            current_usage.usage_stages.append({
                "stage": stage,
                "context_used": context_used,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Check if we're exceeding allocation
            if current_usage.context_used > current_usage.context_allocated:
                logger.warning("Context usage exceeding allocation",
                             session_id=session_id,
                             used=current_usage.context_used,
                             allocated=current_usage.context_allocated)
            
            logger.debug("Context usage tracked",
                        session_id=session_id,
                        stage=stage,
                        used=context_used,
                        total_used=current_usage.context_used)
            
            return True
            
        except Exception as e:
            logger.error("Context usage tracking failed",
                        session_id=session_id,
                        error=str(e))
            return False
    
    async def finalize_usage(
        self, 
        session_id: UUID
    ) -> Optional[float]:
        """
        Finalize context usage and charge FTNS
        
        Args:
            session_id: Session to finalize
            
        Returns:
            Optional[float]: FTNS amount charged, None if failed
        """
        try:
            if session_id not in self.session_usage:
                logger.warning("No usage record found", session_id=session_id)
                return None
            
            usage = self.session_usage[session_id][-1]
            usage.completion_time = datetime.now(timezone.utc)
            
            # Calculate actual FTNS charge based on usage
            if settings.ftns_enabled:
                actual_cost = await self._calculate_ftns_cost(usage.context_used)
                
                # Charge the user
                success = await self.ftns_service.charge_context_access(
                    usage.user_id, 
                    usage.context_used
                )
                
                if success:
                    usage.ftns_charged = actual_cost
                    logger.info("Context usage finalized",
                               session_id=session_id,
                               context_used=usage.context_used,
                               ftns_charged=actual_cost)
                    return actual_cost
                else:
                    logger.error("Failed to charge FTNS for context usage",
                               session_id=session_id)
                    return None
            
            logger.info("Context usage finalized (FTNS disabled)",
                       session_id=session_id,
                       context_used=usage.context_used)
            return 0.0
            
        except Exception as e:
            logger.error("Context usage finalization failed",
                        session_id=session_id,
                        error=str(e))
            return None
    
    async def optimize_context_allocation(
        self, 
        historical_data: List[Dict]
    ) -> Dict[str, float]:
        """
        Analyze historical usage to optimize context allocation
        
        Args:
            historical_data: Past usage data for analysis
            
        Returns:
            Dict with optimization recommendations
        """
        try:
            if not historical_data:
                return {"recommendation": "insufficient_data"}
            
            # Calculate usage patterns
            total_sessions = len(historical_data)
            over_allocated = sum(1 for d in historical_data 
                               if d.get("allocated", 0) > d.get("used", 0) * 1.5)
            under_allocated = sum(1 for d in historical_data 
                                if d.get("used", 0) > d.get("allocated", 0))
            
            over_allocation_rate = over_allocated / total_sessions
            under_allocation_rate = under_allocated / total_sessions
            
            # Calculate average efficiency
            efficiency_scores = []
            for d in historical_data:
                if d.get("allocated", 0) > 0:
                    efficiency = min(d.get("used", 0) / d.get("allocated", 1), 1.0)
                    efficiency_scores.append(efficiency)
            
            avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.5
            
            recommendations = {
                "avg_efficiency": avg_efficiency,
                "over_allocation_rate": over_allocation_rate,
                "under_allocation_rate": under_allocation_rate,
                "recommended_buffer": 0.2 if over_allocation_rate > 0.3 else 0.4,
                "optimization_potential": (1.0 - avg_efficiency) * 100
            }
            
            logger.info("Context optimization analysis completed",
                       sessions_analyzed=total_sessions,
                       avg_efficiency=avg_efficiency,
                       recommendations=recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error("Context optimization failed", error=str(e))
            return {"recommendation": "analysis_failed"}
    
    async def get_session_usage(self, session_id: UUID) -> Optional[ContextUsage]:
        """Get context usage for a session"""
        if session_id in self.session_usage:
            return self.session_usage[session_id][-1]
        return None
    
    async def get_user_usage_stats(self, user_id: str) -> Dict[str, any]:
        """Get usage statistics for a user"""
        user_sessions = []
        for usages in self.session_usage.values():
            for usage in usages:
                if usage.user_id == user_id:
                    user_sessions.append(usage)
        
        if not user_sessions:
            return {"total_sessions": 0}
        
        total_context = sum(u.context_used for u in user_sessions)
        total_ftns = sum(u.ftns_charged for u in user_sessions)
        
        return {
            "total_sessions": len(user_sessions),
            "total_context_used": total_context,
            "total_ftns_charged": total_ftns,
            "avg_context_per_session": total_context / len(user_sessions),
            "avg_ftns_per_session": total_ftns / len(user_sessions)
        }
    
    async def _calculate_ftns_cost(self, context_units: int) -> float:
        """Calculate FTNS cost for context units"""
        # Base rate: 0.1 FTNS per context unit
        base_rate = getattr(settings, 'ftns_context_rate', 0.1)
        return context_units * base_rate


# Global context manager instance
context_manager = ContextManager()
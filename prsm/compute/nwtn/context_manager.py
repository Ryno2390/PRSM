#!/usr/bin/env python3
"""
NWTN Context Manager
====================

Manages context allocation, usage tracking, and optimization for
NWTN query processing sessions.

Provides:
- Session-based context tracking
- Usage history and optimization
- Context allocation recommendations
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ContextUsage:
    """Records context usage for a session"""
    session_id: str
    context_used: int
    context_allocated: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def efficiency(self) -> float:
        """Calculate context usage efficiency"""
        if self.context_allocated == 0:
            return 0.0
        return self.context_used / self.context_allocated


class ContextManager:
    """
    Manages context allocation and tracking for NWTN sessions.
    
    Features:
    - Session context tracking
    - Usage history collection
    - Optimization recommendations
    """
    
    _HISTORY_LIMIT = 1000  # cap in-memory history to prevent unbounded growth
    
    def __init__(self):
        self.session_usage: Dict[str, ContextUsage] = {}
        self.usage_history: List[Dict[str, Any]] = []
        self._optimization_cache: Optional[Dict[str, Any]] = None
        
        logger.info("ContextManager initialized")
    
    async def _persist_usage(self, session_id: str, usage: ContextUsage) -> None:
        """Persist a ContextUsage record to Redis and update PRSMSessionModel."""
        # ── Redis: per-session usage (24h TTL) ───────────────────────────────
        try:
            from prsm.core.redis_client import get_redis_client
            redis = get_redis_client()
            if redis.connected and redis.redis_client:
                key = f"prsm:context:usage:{session_id}"
                payload = json.dumps({
                    "session_id": usage.session_id,
                    "context_used": usage.context_used,
                    "context_allocated": usage.context_allocated,
                    "timestamp": usage.timestamp,
                })
                await redis.redis_client.setex(key, 86400, payload)
        except Exception as e:
            logger.debug("Redis context persist failed", session_id=session_id, error=str(e))

        # ── DB: update PRSMSessionModel if the session row exists ─────────────
        try:
            from prsm.core.database import get_async_session, PRSMSessionModel
            from sqlalchemy import update
            from uuid import UUID
            async with get_async_session() as db:
                await db.execute(
                    update(PRSMSessionModel)
                    .where(PRSMSessionModel.session_id == UUID(session_id))
                    .values(
                        context_used=usage.context_used,
                        nwtn_context_allocation=usage.context_allocated,
                    )
                )
                await db.commit()
        except Exception as e:
            logger.debug("DB context persist failed", session_id=session_id, error=str(e))

    async def _delete_redis_key(self, key: str) -> None:
        """Delete a Redis key, ignoring errors."""
        try:
            from prsm.core.redis_client import get_redis_client
            redis = get_redis_client()
            if redis.connected and redis.redis_client:
                await redis.redis_client.delete(key)
        except Exception as e:
            logger.debug("Redis key delete failed", key=key, error=str(e))

    async def _load_history_from_db(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Load recent context usage records from PRSMSessionModel.
        Used by optimize_context_allocation() when in-memory history is empty.
        """
        try:
            from prsm.core.database import get_async_session, PRSMSessionModel
            from sqlalchemy import select, desc
            async with get_async_session() as db:
                result = await db.execute(
                    select(PRSMSessionModel)
                    .where(PRSMSessionModel.context_used > 0)
                    .order_by(desc(PRSMSessionModel.created_at))
                    .limit(limit)
                )
                rows = result.scalars().all()
            return [
                {
                    "session_id": str(row.session_id),
                    "used":       row.context_used,
                    "allocated":  row.nwtn_context_allocation or row.context_used,
                    "efficiency": (row.context_used / row.nwtn_context_allocation)
                                  if row.nwtn_context_allocation else 1.0,
                    "timestamp":  row.created_at.isoformat() if row.created_at else "",
                }
                for row in rows
                if row.nwtn_context_allocation and row.nwtn_context_allocation > 0
            ]
        except Exception as e:
            logger.debug("DB history load failed", error=str(e))
            return []

    def record_usage(
        self,
        session_id: str,
        context_used: int,
        context_allocated: int
    ) -> ContextUsage:
        """Record context usage for a session."""
        usage = ContextUsage(
            session_id=session_id,
            context_used=context_used,
            context_allocated=context_allocated
        )
        self.session_usage[session_id] = usage
        self.usage_history.append({
            "session_id": session_id,
            "used": context_used,
            "allocated": context_allocated,
            "efficiency": usage.efficiency,
            "timestamp": usage.timestamp
        })

        # Cap in-memory history to prevent unbounded growth
        if len(self.usage_history) > self._HISTORY_LIMIT:
            self.usage_history = self.usage_history[-self._HISTORY_LIMIT:]

        self._optimization_cache = None

        # Fire-and-forget persistence (non-blocking, non-critical)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist_usage(session_id, usage))
        except RuntimeError:
            pass  # No event loop running (e.g., synchronous test context)

        logger.debug(
            "Context usage recorded",
            session_id=session_id,
            used=context_used,
            allocated=context_allocated
        )
        return usage
    
    async def get_session_usage(
        self,
        session_id: Union[str, UUID]
    ) -> Optional[ContextUsage]:
        """Get context usage for a specific session (in-memory → Redis fallback)."""
        sid = str(session_id)

        # Fast path: in-memory
        if sid in self.session_usage:
            return self.session_usage[sid]

        # Redis fallback: recovers data across restarts
        try:
            from prsm.core.redis_client import get_redis_client
            redis = get_redis_client()
            if redis.connected and redis.redis_client:
                key = f"prsm:context:usage:{sid}"
                data = await redis.redis_client.get(key)
                if data:
                    d = json.loads(data)
                    usage = ContextUsage(
                        session_id=d["session_id"],
                        context_used=d["context_used"],
                        context_allocated=d["context_allocated"],
                    )
                    usage.timestamp = d["timestamp"]
                    self.session_usage[sid] = usage  # hydrate in-memory
                    return usage
        except Exception as e:
            logger.debug("Redis context lookup failed", session_id=sid, error=str(e))

        return None
    
    async def optimize_context_allocation(
        self,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze historical usage and provide optimization recommendations.
        
        Args:
            historical_data: Optional list of historical usage data.
                           If not provided, uses internal history.
                           
        Returns:
            Dict with optimization recommendations and statistics
        """
        
        if self._optimization_cache is not None:
            return self._optimization_cache
        
        data = historical_data or self.usage_history

        # Hydrate from DB if no in-memory history
        if not data:
            data = await self._load_history_from_db()
        
        if not data:
            return {
                "avg_efficiency": 0.7,
                "over_allocation_rate": 0.1,
                "under_allocation_rate": 0.1,
                "optimization_potential": 10.0,
                "recommended_allocation": 100,
                "sample_size": 0
            }
        
        efficiencies = []
        over_allocated = 0
        under_allocated = 0
        
        for entry in data:
            used = entry.get("used", 0)
            allocated = entry.get("allocated", 1)
            
            if allocated > 0:
                eff = used / allocated
                efficiencies.append(eff)
                
                if eff < 0.7:
                    over_allocated += 1
                elif eff > 0.9:
                    under_allocated += 1
        
        avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0.7
        over_rate = over_allocated / len(data) if data else 0.1
        under_rate = under_allocated / len(data) if data else 0.1
        
        avg_used = sum(d.get("used", 0) for d in data) / len(data) if data else 50
        recommended_allocation = int(avg_used / max(avg_efficiency, 0.5))
        
        result = {
            "avg_efficiency": round(avg_efficiency, 3),
            "over_allocation_rate": round(over_rate, 3),
            "under_allocation_rate": round(under_rate, 3),
            "optimization_potential": round((1 - avg_efficiency) * 100, 1),
            "recommended_allocation": recommended_allocation,
            "sample_size": len(data),
            "total_context_used": sum(d.get("used", 0) for d in data),
            "total_context_allocated": sum(d.get("allocated", 0) for d in data)
        }
        
        self._optimization_cache = result
        
        logger.info(
            "Context optimization analysis complete",
            avg_efficiency=avg_efficiency,
            sample_size=len(data)
        )
        
        return result
    
    def clear_session(self, session_id: str) -> bool:
        """Clear context usage for a session (in-memory + Redis)."""
        removed = session_id in self.session_usage
        if removed:
            del self.session_usage[session_id]

        # Fire-and-forget Redis cleanup
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._delete_redis_key(f"prsm:context:usage:{session_id}")
            )
        except RuntimeError:
            pass

        return removed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall context management statistics."""
        total_sessions = len(self.session_usage)
        total_history = len(self.usage_history)
        
        if self.usage_history:
            avg_used = sum(h["used"] for h in self.usage_history) / total_history
            avg_allocated = sum(h["allocated"] for h in self.usage_history) / total_history
        else:
            avg_used = 0
            avg_allocated = 0
        
        return {
            "active_sessions": total_sessions,
            "total_history_entries": total_history,
            "average_context_used": avg_used,
            "average_context_allocated": avg_allocated
        }


_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get or create the singleton ContextManager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager

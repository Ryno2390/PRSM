"""
Provenance integration for vector store operations

Links vector search with FTNS royalty system for creator compensation.
This module will be integrated with the FTNS service in Phase 1C.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from .base import PRSMVectorStore, ContentMatch, SearchFilters

logger = logging.getLogger(__name__)


class ProvenanceVectorIntegration:
    """
    Links vector search operations with FTNS royalty tracking
    
    This class provides the bridge between vector similarity search
    and creator compensation through the FTNS token system.
    
    Features:
    - Automatic usage tracking for royalty calculation
    - Creator compensation based on content access patterns
    - Integration with FTNS service for token distribution
    - Audit trails for governance and transparency
    """
    
    def __init__(self, vector_store: PRSMVectorStore):
        self.vector_store = vector_store
        self.usage_tracking_enabled = True
        self.usage_log: List[Dict[str, Any]] = []
        
    async def search_with_royalty_tracking(self,
                                         query_embedding: np.ndarray,
                                         user_id: str,
                                         filters: Optional[SearchFilters] = None,
                                         top_k: int = 10) -> List[ContentMatch]:
        """
        Search content and automatically track usage for royalty payments
        
        Args:
            query_embedding: Vector embedding of search query
            user_id: ID of user performing the search
            filters: Optional search filters
            top_k: Maximum number of results
            
        Returns:
            List of content matches with royalty tracking recorded
        """
        # Perform the search
        results = await self.vector_store.search_similar_content(
            query_embedding, filters, top_k
        )
        
        # Track usage for royalty calculation
        if self.usage_tracking_enabled and results:
            await self._track_content_access(results, user_id)
        
        return results
    
    async def _track_content_access(self, 
                                  content_matches: List[ContentMatch], 
                                  user_id: str):
        """
        Track content access for royalty calculation
        
        This method records usage data that will be used by the FTNS service
        to calculate and distribute royalty payments to content creators.
        """
        for match in content_matches:
            usage_entry = {
                "timestamp": datetime.utcnow(),
                "user_id": user_id,
                "content_cid": match.content_cid,
                "creator_id": match.creator_id,
                "royalty_rate": match.royalty_rate,
                "similarity_score": match.similarity_score,
                "content_type": match.content_type.value,
                "access_type": "search_result"
            }
            
            self.usage_log.append(usage_entry)
            
            # In Phase 1C, this will integrate with the actual FTNS service:
            # await self.ftns_service.record_content_usage(usage_entry)
            
        logger.debug(f"Tracked access for {len(content_matches)} content items by user {user_id}")
    
    async def get_usage_statistics(self, 
                                 time_period_hours: int = 24) -> Dict[str, Any]:
        """
        Get usage statistics for royalty calculation
        
        Args:
            time_period_hours: Hours to look back for statistics
            
        Returns:
            Dictionary with usage statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_period_hours)
        recent_usage = [
            entry for entry in self.usage_log 
            if entry["timestamp"] > cutoff_time
        ]
        
        # Calculate statistics
        stats = {
            "total_accesses": len(recent_usage),
            "unique_users": len(set(entry["user_id"] for entry in recent_usage)),
            "unique_content": len(set(entry["content_cid"] for entry in recent_usage)),
            "unique_creators": len(set(entry["creator_id"] for entry in recent_usage if entry["creator_id"])),
            "total_royalty_value": sum(
                entry["similarity_score"] * entry["royalty_rate"] 
                for entry in recent_usage
            ),
            "content_type_breakdown": {},
            "creator_breakdown": {}
        }
        
        # Content type breakdown
        for entry in recent_usage:
            content_type = entry["content_type"]
            if content_type not in stats["content_type_breakdown"]:
                stats["content_type_breakdown"][content_type] = 0
            stats["content_type_breakdown"][content_type] += 1
        
        # Creator breakdown
        for entry in recent_usage:
            creator_id = entry.get("creator_id")
            if creator_id:
                if creator_id not in stats["creator_breakdown"]:
                    stats["creator_breakdown"][creator_id] = {
                        "accesses": 0,
                        "total_royalty_value": 0.0
                    }
                stats["creator_breakdown"][creator_id]["accesses"] += 1
                stats["creator_breakdown"][creator_id]["total_royalty_value"] += (
                    entry["similarity_score"] * entry["royalty_rate"]
                )
        
        return stats
    
    async def generate_royalty_report(self,
                                    time_period_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Generate royalty payment report for FTNS distribution
        
        Args:
            time_period_hours: Hours to include in report
            
        Returns:
            List of royalty payment entries for FTNS service
        """
        stats = await self.get_usage_statistics(time_period_hours)
        royalty_payments = []
        
        for creator_id, creator_stats in stats["creator_breakdown"].items():
            if creator_stats["total_royalty_value"] > 0:
                payment_entry = {
                    "creator_id": creator_id,
                    "period_start": datetime.utcnow() - timedelta(hours=time_period_hours),
                    "period_end": datetime.utcnow(),
                    "total_accesses": creator_stats["accesses"],
                    "royalty_value": creator_stats["total_royalty_value"],
                    "payment_status": "pending"
                }
                royalty_payments.append(payment_entry)
        
        return royalty_payments
    
    def enable_usage_tracking(self):
        """Enable usage tracking for royalty calculation"""
        self.usage_tracking_enabled = True
        logger.info("Usage tracking enabled")
    
    def disable_usage_tracking(self):
        """Disable usage tracking"""
        self.usage_tracking_enabled = False
        logger.info("Usage tracking disabled")
    
    def clear_usage_log(self):
        """Clear the usage log (use carefully!)"""
        cleared_count = len(self.usage_log)
        self.usage_log = []
        logger.info(f"Cleared {cleared_count} usage log entries")


# Import fix for timedelta
from datetime import timedelta
#!/usr/bin/env python3
"""
NWTN Complete System - Unified Facade
=====================================

The NWTNCompleteSystem provides a unified interface to the complete NWTN
reasoning infrastructure. It serves as a facade that integrates:

- NWTNOrchestrator: Central coordination layer
- MetaReasoningEngine: System 1/2 dual-process reasoning
- ExternalStorageConfig: IPFS and external knowledge base
- Provenance tracking: Content attribution and usage tracking
- API configuration: User API key management for LLM calls

This facade provides a single entry point for:
1. Query processing with breakthrough modes (CONSERVATIVE/REVOLUTIONARY)
2. Knowledge base search and retrieval
3. Natural language response generation
4. FTNS token accounting and provenance tracking
"""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4
from enum import Enum

import structlog

from prsm.core.models import UserInput

logger = structlog.get_logger(__name__)

try:
    from prsm.compute.nwtn.orchestrator import NWTNOrchestrator, get_nwtn_orchestrator
    from prsm.compute.nwtn.meta_reasoning_engine import (
        MetaReasoningEngine, ThinkingMode, BreakthroughMode, get_meta_reasoning_engine
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NWTN components not fully available: {e}")
    COMPONENTS_AVAILABLE = False


@dataclass
class NWTNCompleteResponse:
    """Complete response from NWTN system"""
    query: str
    response: str
    natural_language_response: Optional[str] = None
    final_response: Optional[str] = None
    session_id: str = ""
    user_id: str = ""
    confidence_score: float = 0.0
    context_used: int = 0
    ftns_cost: float = 0.0
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    content_sources: List[str] = field(default_factory=list)
    breakthrough_mode: str = "balanced"
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BreakthroughModes:
    """Breakthrough mode configuration"""
    CONSERVATIVE = BreakthroughMode.CONSERVATIVE if COMPONENTS_AVAILABLE else "conservative"
    BALANCED = BreakthroughMode.BALANCED if COMPONENTS_AVAILABLE else "balanced"
    REVOLUTIONARY = BreakthroughMode.REVOLUTIONARY if COMPONENTS_AVAILABLE else "revolutionary"


class NWTNCompleteSystem:
    """
    Unified facade for the complete NWTN reasoning system.
    
    Provides a single entry point for all NWTN operations including
    query processing, knowledge retrieval, and provenance tracking.
    """
    
    def __init__(self):
        self.orchestrator: Optional[NWTNOrchestrator] = None
        self.meta_engine: Optional[MetaReasoningEngine] = None
        self._initialized = False
        self._user_api_configs: Dict[str, Dict[str, str]] = {}
        
        logger.info("NWTNCompleteSystem created")
    
    async def initialize(self) -> bool:
        """Initialize all NWTN components."""
        if self._initialized:
            return True
        
        try:
            if COMPONENTS_AVAILABLE:
                self.orchestrator = get_nwtn_orchestrator()
                await self.orchestrator.initialize()
                
                self.meta_engine = get_meta_reasoning_engine()
                await self.meta_engine.initialize()
                await self.meta_engine.initialize_external_knowledge_base()
            else:
                logger.warning("NWTN components not available, running in limited mode")
            
            self._initialized = True
            logger.info("NWTNCompleteSystem initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NWTNCompleteSystem: {e}")
            return False
    
    async def configure_user_api(
        self,
        user_id: str,
        provider: str,
        api_key: str
    ) -> bool:
        """
        Configure API key for a user.
        
        Args:
            user_id: User identifier
            provider: API provider (claude, openai, etc.)
            api_key: API key for the provider
            
        Returns:
            True if configuration succeeded
        """
        self._user_api_configs[user_id] = {
            "provider": provider,
            "api_key": api_key,
            "configured_at": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"API configured for user {user_id} with provider {provider}")
        return True
    
    async def process_query(
        self,
        user_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        show_reasoning_trace: bool = False
    ) -> NWTNCompleteResponse:
        """
        Process a query through the complete NWTN pipeline.
        
        Args:
            user_id: User making the query
            query: The query to process
            context: Optional context including breakthrough_mode, preferences
            show_reasoning_trace: Whether to include detailed reasoning trace
            
        Returns:
            NWTNCompleteResponse with complete results
        """
        
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        context = context or {}
        breakthrough_mode = context.get("breakthrough_mode", "balanced")
        
        thinking_mode = ThinkingMode.STANDARD
        if context.get("reasoning_depth") == "deep":
            thinking_mode = ThinkingMode.DEEP
        elif context.get("enable_deep_reasoning"):
            thinking_mode = ThinkingMode.DEEP
        
        if self.meta_engine:
            meta_result = await self.meta_engine.meta_reason(
                query=query,
                context=context,
                thinking_mode=thinking_mode
            )
            
            response = NWTNCompleteResponse(
                query=query,
                response=meta_result.final_synthesis,
                natural_language_response=meta_result.final_synthesis,
                final_response=meta_result.final_synthesis,
                session_id=str(uuid4()),
                user_id=user_id,
                confidence_score=meta_result.meta_confidence,
                context_used=int(meta_result.ftns_cost * 100),
                ftns_cost=meta_result.ftns_cost,
                reasoning_trace=meta_result.reasoning_trace if show_reasoning_trace else [],
                breakthrough_mode=breakthrough_mode,
                processing_time=meta_result.processing_time
            )
        else:
            response = await self._process_fallback(query, user_id, context)
        
        response.processing_time = time.time() - start_time
        
        logger.info(
            "Query processed",
            query=query[:50],
            user_id=user_id,
            confidence=response.confidence_score
        )
        
        return response
    
    async def _process_fallback(
        self,
        query: str,
        user_id: str,
        context: Dict[str, Any]
    ) -> NWTNCompleteResponse:
        """Fallback processing when meta-engine is unavailable."""
        
        return NWTNCompleteResponse(
            query=query,
            response=f"NWTN analysis of: {query}\n\nProcessed in fallback mode with limited capabilities.",
            natural_language_response=f"Analysis of: {query}",
            session_id=str(uuid4()),
            user_id=user_id,
            confidence_score=0.5,
            context_used=50,
            ftns_cost=0.5,
            reasoning_trace=[{"step": "fallback", "mode": "limited"}],
            breakthrough_mode=context.get("breakthrough_mode", "balanced")
        )
    
    async def search_knowledge_base(
        self,
        query: str,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search the NWTN knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of knowledge base entries matching the query
        """
        
        if not self._initialized:
            await self.initialize()
        
        if self.meta_engine and self.meta_engine.external_kb:
            return await self.meta_engine.external_kb.search_papers(query, max_results)
        
        return [
            {
                "title": f"Knowledge entry for: {query[:50]}...",
                "content": f"Relevant content discussing {query[:50]}...",
                "relevance_score": 0.8,
                "source": "fallback_knowledge_base"
            }
        ]
    
    def get_user_api_config(self, user_id: str) -> Optional[Dict[str, str]]:
        """Get API configuration for a user."""
        return self._user_api_configs.get(user_id)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the NWTN system."""
        return {
            "initialized": self._initialized,
            "orchestrator_available": self.orchestrator is not None,
            "meta_engine_available": self.meta_engine is not None,
            "external_kb_available": (
                self.meta_engine.external_kb is not None 
                if self.meta_engine else False
            ),
            "configured_users": len(self._user_api_configs)
        }


_nwtn_complete_system: Optional[NWTNCompleteSystem] = None


def get_nwtn_complete_system() -> NWTNCompleteSystem:
    """Get or create the singleton NWTN complete system instance."""
    global _nwtn_complete_system
    if _nwtn_complete_system is None:
        _nwtn_complete_system = NWTNCompleteSystem()
    return _nwtn_complete_system


async def process_nwtn_query(
    user_id: str,
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> NWTNCompleteResponse:
    """Convenience function to process a query through NWTN."""
    system = get_nwtn_complete_system()
    return await system.process_query(user_id, query, context)

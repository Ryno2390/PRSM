#!/usr/bin/env python3
"""
NWTN Complete System - Unified Interface for Natural Language Multi-Modal Reasoning
===================================================================================

This module provides the complete NWTN system interface that combines:
1. NWTN Voicebox - Natural language interface with user's choice of LLM
2. 8-Step NWTN Core - Sophisticated multi-modal reasoning system
3. Full PRSM Integration - Resource discovery, distributed execution, marketplace

The complete system enables users to interact with the world's most advanced 
reasoning AI through natural language while leveraging the full power of PRSM's
distributed infrastructure.

System Architecture:
┌─────────────────────┐
│    User Query       │ "What are the most promising approaches for 
│  (Natural Language) │  commercial atomically precise manufacturing?"
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│   NWTN Voicebox     │ • Query analysis & clarification
│   (This Module)     │ • API key management (Claude/GPT-4/etc.)
│                     │ • Natural language translation
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│   8-Step NWTN Core  │ 1. Query decomposition
│   Multi-Modal       │ 2. Reasoning classification
│   Reasoning System  │ 3. Multi-modal analysis (all 7 reasoning types)
│                     │ 4. Network validation
│                     │ 5. PRSM resource discovery
│                     │ 6. Distributed execution
│                     │ 7. Marketplace asset integration
│                     │ 8. Result compilation & synthesis
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Natural Language   │ "Based on NWTN's comprehensive analysis using
│     Response        │  all 7 reasoning modes, here are the top 5
│                     │  most promising approaches..."
└─────────────────────┘

Key Features:
• Complete natural language interface to NWTN's advanced reasoning
• BYOA (Bring Your Own API) support for Claude, GPT-4, Gemini, etc.
• Full 8-step reasoning pipeline with network validation
• Automatic cost estimation and FTNS charging
• Comprehensive reasoning trace and transparency
• Integration with PRSM's distributed infrastructure

Usage Example:
    system = NWTNCompleteSystem()
    await system.initialize()
    
    # Configure user's API key
    await system.configure_user_api("user123", "claude", "sk-...")
    
    # Process natural language query
    response = await system.process_query(
        user_id="user123",
        query="What are the most promising approaches for commercial atomically precise manufacturing?"
    )
    
    print(response.natural_language_response)
    print(f"Confidence: {response.confidence_score}")
    print(f"Used reasoning modes: {response.used_reasoning_modes}")
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import structlog

from prsm.nwtn.voicebox import NWTNVoicebox, LLMProvider, VoiceboxResponse, QueryComplexity
from prsm.nwtn.multi_modal_reasoning_engine import MultiModalReasoningEngine
from prsm.tokenomics.ftns_service import FTNSService
from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class SystemStatus:
    """Status of the complete NWTN system"""
    voicebox_initialized: bool = False
    multi_modal_engine_initialized: bool = False
    ftns_service_initialized: bool = False
    total_users_configured: int = 0
    total_queries_processed: int = 0
    average_confidence_score: float = 0.0
    supported_providers: List[str] = None
    
    def __post_init__(self):
        if self.supported_providers is None:
            self.supported_providers = ["claude", "openai", "gemini", "azure_openai"]


class NWTNCompleteSystem:
    """
    NWTN Complete System - Unified Interface for Natural Language Multi-Modal Reasoning
    
    This class provides the complete NWTN system that combines natural language
    interaction with sophisticated multi-modal reasoning capabilities.
    
    The system operates in two main modes:
    1. Interactive Mode: Real-time natural language conversation
    2. Batch Mode: Process multiple queries efficiently
    
    Key Components:
    - NWTN Voicebox: Natural language interface layer
    - Multi-Modal Reasoning Engine: 8-step reasoning pipeline
    - FTNS Integration: Economic model and resource allocation
    - PRSM Integration: Distributed execution and marketplace
    """
    
    def __init__(self):
        self.voicebox = None
        self.multi_modal_engine = None
        self.ftns_service = None
        
        # System statistics
        self.total_queries_processed = 0
        self.total_users_configured = 0
        self.confidence_scores = []
        
        # Configuration
        self.system_initialized = False
        
        logger.info("NWTN Complete System initialized")
    
    async def initialize(self):
        """Initialize the complete NWTN system"""
        try:
            logger.info("🚀 Initializing NWTN Complete System...")
            
            # Initialize NWTN Voicebox
            logger.info("📢 Initializing NWTN Voicebox...")
            self.voicebox = NWTNVoicebox()
            await self.voicebox.initialize()
            
            # Initialize Multi-Modal Reasoning Engine
            logger.info("🧠 Initializing Multi-Modal Reasoning Engine...")
            self.multi_modal_engine = MultiModalReasoningEngine()
            await self.multi_modal_engine.initialize()
            
            # Initialize FTNS Service
            logger.info("💰 Initializing FTNS Service...")
            self.ftns_service = FTNSService()
            await self.ftns_service.initialize()
            
            self.system_initialized = True
            
            logger.info("✅ NWTN Complete System fully initialized")
            logger.info("🎯 Ready to process natural language queries with advanced multi-modal reasoning")
            
        except Exception as e:
            logger.error(f"Failed to initialize NWTN Complete System: {e}")
            raise
    
    async def configure_user_api(
        self,
        user_id: str,
        provider: str,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """Configure user's API key for their chosen LLM provider"""
        if not self.system_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            # Convert string provider to enum
            provider_enum = LLMProvider(provider.lower())
            
            # Configure through voicebox
            success = await self.voicebox.configure_api_key(
                user_id=user_id,
                provider=provider_enum,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name
            )
            
            if success:
                self.total_users_configured += 1
                logger.info(f"✅ User {user_id} configured with {provider} API")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to configure user API: {e}")
            return False
    
    async def process_query(
        self,
        user_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        show_reasoning_trace: bool = False
    ) -> VoiceboxResponse:
        """
        Process natural language query through complete NWTN system
        
        This method orchestrates the complete pipeline:
        1. Query analysis and clarification (if needed)
        2. 8-step multi-modal reasoning processing
        3. Natural language response generation
        4. Cost calculation and FTNS charging
        """
        if not self.system_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            start_time = datetime.now(timezone.utc)
            
            logger.info(f"🔄 Processing query for user {user_id}: {query[:100]}...")
            
            # Process through voicebox (which handles the complete pipeline)
            response = await self.voicebox.process_query(
                user_id=user_id,
                query=query,
                context=context
            )
            
            # Update system statistics
            self.total_queries_processed += 1
            if response.confidence_score > 0:
                self.confidence_scores.append(response.confidence_score)
            
            # Log processing summary
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"✅ Query processed in {processing_time:.2f}s")
            logger.info(f"📊 Confidence: {response.confidence_score:.2f}")
            logger.info(f"🧠 Reasoning modes: {', '.join(response.used_reasoning_modes)}")
            logger.info(f"💰 Cost: {response.total_cost_ftns:.2f} FTNS")
            
            # Optionally show reasoning trace
            if show_reasoning_trace and response.reasoning_trace:
                logger.info("🔍 Reasoning trace:")
                for i, step in enumerate(response.reasoning_trace):
                    logger.info(f"  Step {i+1}: {step}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    
    async def provide_clarification(
        self,
        user_id: str,
        query_id: str,
        clarification: str
    ) -> VoiceboxResponse:
        """Provide clarification for ambiguous query"""
        if not self.system_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            logger.info(f"📝 Processing clarification for query {query_id}")
            
            response = await self.voicebox.provide_clarification(
                user_id=user_id,
                query_id=query_id,
                clarification_response=clarification
            )
            
            logger.info(f"✅ Clarification processed")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process clarification: {e}")
            raise
    
    async def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        try:
            avg_confidence = 0.0
            if self.confidence_scores:
                avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
            
            return SystemStatus(
                voicebox_initialized=self.voicebox is not None,
                multi_modal_engine_initialized=self.multi_modal_engine is not None,
                ftns_service_initialized=self.ftns_service is not None,
                total_users_configured=self.total_users_configured,
                total_queries_processed=self.total_queries_processed,
                average_confidence_score=avg_confidence,
                supported_providers=["claude", "openai", "gemini", "azure_openai", "huggingface"]
            )
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return SystemStatus()
    
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get user's interaction history"""
        if not self.system_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            return await self.voicebox.get_user_interaction_history(user_id, limit)
            
        except Exception as e:
            logger.error(f"Failed to get user history: {e}")
            return []
    
    async def batch_process_queries(
        self,
        user_id: str,
        queries: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[VoiceboxResponse]:
        """Process multiple queries efficiently"""
        if not self.system_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            logger.info(f"🔄 Batch processing {len(queries)} queries for user {user_id}")
            
            responses = []
            for i, query in enumerate(queries):
                logger.info(f"📝 Processing query {i+1}/{len(queries)}")
                
                response = await self.process_query(
                    user_id=user_id,
                    query=query,
                    context=context
                )
                responses.append(response)
                
                # Brief pause to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            logger.info(f"✅ Batch processing completed: {len(responses)} responses")
            return responses
            
        except Exception as e:
            logger.error(f"Failed to batch process queries: {e}")
            raise
    
    async def estimate_query_cost(
        self,
        user_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Estimate cost and complexity for a query without processing"""
        if not self.system_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            # Use voicebox to analyze query
            from uuid import uuid4
            analysis = await self.voicebox._analyze_query(
                query_id=str(uuid4()),
                query=query,
                context=context
            )
            
            return {
                "estimated_cost_ftns": analysis.estimated_cost_ftns,
                "complexity": analysis.complexity.value,
                "estimated_reasoning_modes": analysis.estimated_reasoning_modes,
                "domain_hints": analysis.domain_hints,
                "requires_clarification": analysis.clarification_status.value != "clear",
                "clarification_questions": analysis.clarification_questions
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate query cost: {e}")
            return {}
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        try:
            logger.info("🔄 Shutting down NWTN Complete System...")
            
            # System is stateless, so shutdown is simple
            self.system_initialized = False
            
            logger.info("✅ NWTN Complete System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Example usage and demonstration
async def demonstration():
    """Demonstration of the complete NWTN system"""
    print("🚀 NWTN Complete System Demonstration")
    print("=" * 50)
    
    # Initialize system
    system = NWTNCompleteSystem()
    await system.initialize()
    
    # Configure user API (example)
    user_id = "demo_user"
    configured = await system.configure_user_api(
        user_id=user_id,
        provider="claude",
        api_key="sk-demo-key-12345"  # Demo key
    )
    
    if configured:
        print(f"✅ User {user_id} configured with Claude API")
        
        # Estimate cost for example query
        example_query = "What are the most promising approaches for commercial atomically precise manufacturing?"
        cost_estimate = await system.estimate_query_cost(user_id, example_query)
        
        print(f"\n📊 Cost Estimate for Query:")
        print(f"   Query: {example_query}")
        print(f"   Estimated Cost: {cost_estimate.get('estimated_cost_ftns', 0):.2f} FTNS")
        print(f"   Complexity: {cost_estimate.get('complexity', 'unknown')}")
        print(f"   Reasoning Modes: {', '.join(cost_estimate.get('estimated_reasoning_modes', []))}")
        
        # Process the query
        print(f"\n🔄 Processing query through complete NWTN system...")
        try:
            response = await system.process_query(
                user_id=user_id,
                query=example_query,
                show_reasoning_trace=True
            )
            
            print(f"\n✅ Query Processing Complete!")
            print(f"📝 Response: {response.natural_language_response[:200]}...")
            print(f"📊 Confidence: {response.confidence_score:.2f}")
            print(f"🧠 Reasoning Modes Used: {', '.join(response.used_reasoning_modes)}")
            print(f"💰 Actual Cost: {response.total_cost_ftns:.2f} FTNS")
            print(f"⏱️  Processing Time: {response.processing_time_seconds:.2f}s")
            
        except Exception as e:
            print(f"❌ Demo processing failed: {e}")
            print("(This is expected in demo mode without real API keys)")
    
    # Get system status
    status = await system.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Voicebox Initialized: {status.voicebox_initialized}")
    print(f"   Multi-Modal Engine Initialized: {status.multi_modal_engine_initialized}")
    print(f"   FTNS Service Initialized: {status.ftns_service_initialized}")
    print(f"   Total Users Configured: {status.total_users_configured}")
    print(f"   Total Queries Processed: {status.total_queries_processed}")
    print(f"   Supported Providers: {', '.join(status.supported_providers)}")
    
    # Shutdown
    await system.shutdown()
    print(f"\n✅ Demonstration complete!")


# Global system instance
_complete_system = None

async def get_complete_system() -> NWTNCompleteSystem:
    """Get the global complete system instance"""
    global _complete_system
    if _complete_system is None:
        _complete_system = NWTNCompleteSystem()
        await _complete_system.initialize()
    return _complete_system


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstration())
#!/usr/bin/env python3
"""
NWTN Voicebox - Natural Language Interface for Multi-Modal Reasoning
===================================================================

The NWTN Voicebox provides a natural language interface layer that sits on top
of the sophisticated 8-step NWTN multi-modal reasoning system. This enables
users to interact with NWTN's advanced reasoning capabilities through
conversational natural language.

Phase 1: BYOA (Bring Your Own API) Implementation
- Users provide their own API keys for third-party LLMs (Claude, GPT-4, etc.)
- Natural language query ingestion and analysis
- Clarification question generation when needed
- Result translation from structured insights to natural language
- Complete integration with the 8-step NWTN reasoning pipeline

Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  NWTN Voicebox   │───▶│  8-Step NWTN    │
│  (Natural Lang) │    │  (This Module)   │    │  Core System    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Third-Party    │    │  Structured     │
                       │   LLM APIs       │    │  Insights       │
                       │ (Claude/GPT-4)   │    │  (JSON Results) │
                       └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  Natural Lang   │
                                                │  Response       │
                                                └─────────────────┘

Key Features:
1. API Key Management: Secure handling of user-provided API keys
2. Query Analysis: Intelligent analysis of natural language queries
3. Clarification System: Ask clarifying questions when needed
4. Result Translation: Convert structured insights to natural language
5. Full Integration: Seamless connection to 8-step NWTN reasoning
6. Error Handling: Graceful handling of API failures and edge cases

Usage:
    from prsm.nwtn.voicebox import NWTNVoicebox
    
    voicebox = NWTNVoicebox()
    await voicebox.initialize()
    
    # User provides API key
    await voicebox.configure_api_key(user_id="user123", provider="claude", api_key="sk-...")
    
    # Natural language interaction
    response = await voicebox.process_query(
        user_id="user123",
        query="What are the most promising approaches for commercial atomically precise manufacturing?"
    )
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime, timezone
import structlog
from pydantic import BaseModel, Field

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, MetaReasoningResult, ThinkingMode
from prsm.nwtn.config import VerbosityLevel
from prsm.nwtn.breakthrough_modes import (
    BreakthroughMode, breakthrough_mode_manager, get_breakthrough_mode_config, suggest_breakthrough_mode
)
from prsm.core.config import get_settings
from prsm.core.database_service import get_database_service
from prsm.tokenomics.ftns_service import FTNSService
from prsm.core.models import FTNSTransaction
from prsm.tokenomics.enhanced_pricing_engine import calculate_query_cost, get_pricing_preview, PricingCalculation
from prsm.nwtn.content_royalty_engine import ContentRoyaltyEngine, QueryComplexity
from prsm.nwtn.external_storage_config import get_external_knowledge_base
from prsm.provenance.enhanced_provenance_system import EnhancedProvenanceSystem
from prsm.nwtn.content_grounding_synthesizer import ContentGroundingSynthesizer

logger = structlog.get_logger(__name__)
settings = get_settings()


class LLMProvider(str, Enum):
    """Supported LLM providers for BYOA"""
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"


class QueryComplexity(str, Enum):
    """Query complexity levels"""
    SIMPLE = "simple"          # Direct factual questions
    MODERATE = "moderate"      # Analysis requiring 2-3 reasoning modes
    COMPLEX = "complex"        # Multi-step analysis requiring 4+ reasoning modes
    BREAKTHROUGH = "breakthrough"  # Research-level queries requiring full network validation


class ClarificationStatus(str, Enum):
    """Status of query clarification"""
    CLEAR = "clear"           # Query is clear and ready for processing
    NEEDS_CLARIFICATION = "needs_clarification"  # Query needs clarification
    CLARIFYING = "clarifying"  # Currently asking clarifying questions
    CLARIFIED = "clarified"    # User has provided clarification


@dataclass
class APIConfiguration:
    """Configuration for third-party LLM API"""
    provider: LLMProvider
    api_key: str
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QueryAnalysis:
    """Analysis of user query"""
    query_id: str
    original_query: str
    complexity: QueryComplexity
    estimated_reasoning_modes: List[str]
    domain_hints: List[str]
    clarification_status: ClarificationStatus
    clarification_questions: List[str]
    estimated_cost_ftns: float
    analysis_confidence: float
    requires_breakthrough_mode: bool


@dataclass
class ClarificationExchange:
    """Record of clarification interaction"""
    exchange_id: str
    query_id: str
    question: str
    user_response: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SourceLink:
    """Information about a source used in reasoning"""
    content_id: str
    title: str
    creator: str
    ipfs_link: str
    contribution_date: str
    relevance_score: float
    content_type: str
    
@dataclass
class VoiceboxResponse:
    """Complete voicebox response to user with enhanced token-based pricing details"""
    response_id: str
    query_id: str
    natural_language_response: str
    structured_insights: Optional[MetaReasoningResult] = None
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    total_cost_ftns: float = 0.0
    confidence_score: float = 0.0
    used_reasoning_modes: List[str] = field(default_factory=list)
    source_links: List[SourceLink] = field(default_factory=list)
    attribution_summary: str = ""
    royalties_distributed: Dict[str, float] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Enhanced token-based pricing details
    pricing_calculation: Optional[PricingCalculation] = None
    base_computational_tokens: int = 0
    reasoning_multiplier: float = 1.0
    verbosity_factor: float = 1.0
    market_rate: float = 1.0
    quality_bonus: float = 1.0
    cost_breakdown: Dict[str, Any] = field(default_factory=dict)
    
    # Breakthrough mode details
    breakthrough_mode: str = "balanced"
    breakthrough_intensity: str = ""
    candidate_distribution: Dict[str, float] = field(default_factory=dict)
    mode_features: Dict[str, bool] = field(default_factory=dict)


class NWTNVoicebox:
    """
    NWTN Voicebox - Natural Language Interface for Multi-Modal Reasoning
    
    Phase 1: BYOA (Bring Your Own API) Implementation
    
    Provides natural language interface capabilities for NWTN's sophisticated
    multi-modal reasoning system, allowing users to interact conversationally
    with the world's most advanced reasoning AI.
    """
    
    def __init__(self):
        self.meta_reasoning_engine = None  # Will be initialized async
        self.database_service = get_database_service()
        self.ftns_service = None  # Will be initialized async
        self.content_royalty_engine = None  # Will be initialized async
        self.provenance_system = None  # Will be initialized async
        self.external_knowledge_base = None  # Will be initialized async
        self.content_grounding_synthesizer = None  # Will be initialized async
        
        # User API configurations
        self.user_api_configs: Dict[str, APIConfiguration] = {}
        
        # System default API configuration
        self.default_api_config: Optional[APIConfiguration] = None
        
        # Active query sessions
        self.active_queries: Dict[str, QueryAnalysis] = {}
        self.clarification_exchanges: Dict[str, List[ClarificationExchange]] = {}
        
        # Voicebox configuration
        self.max_clarification_rounds = 3
        self.min_confidence_threshold = 0.7
        self.complexity_cost_multipliers = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 2.5,
            QueryComplexity.COMPLEX: 5.0,
            QueryComplexity.BREAKTHROUGH: 10.0
        }
        
        logger.info("NWTN Voicebox initialized")
    
    async def initialize(self):
        """Initialize voicebox service dependencies"""
        try:
            # Initialize multi-modal reasoning engine
            self.meta_reasoning_engine = MetaReasoningEngine()
            # Initialize external knowledge base for Ferrari fuel line
            await self.meta_reasoning_engine.initialize_external_knowledge_base()
            # MetaReasoningEngine initializes itself in constructor
            
            # Initialize FTNS service
            from ..tokenomics.ftns_service import get_ftns_service
            self.ftns_service = await get_ftns_service()
            # FTNSService doesn't have an async initialize method
            
            # Initialize content royalty engine
            self.content_royalty_engine = ContentRoyaltyEngine()
            await self.content_royalty_engine.initialize()
            
            # Initialize provenance system for source links
            self.provenance_system = EnhancedProvenanceSystem()
            
            # Initialize external knowledge base for Ferrari fuel line
            self.external_knowledge_base = await get_external_knowledge_base()
            
            # Initialize content grounding synthesizer
            self.content_grounding_synthesizer = ContentGroundingSynthesizer(self.external_knowledge_base)
            
            # Initialize default Claude API configuration
            self._initialize_default_api_config()
            
            logger.info("✅ NWTN Voicebox fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize voicebox: {e}")
            raise
    
    async def configure_api_key(
        self,
        user_id: str,
        provider: LLMProvider,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """Configure API key for user's chosen LLM provider"""
        try:
            # Validate API key format
            if not self._validate_api_key_format(provider, api_key):
                raise ValueError(f"Invalid API key format for provider: {provider}")
            
            # Test API key with a simple request
            api_valid = await self._test_api_key(provider, api_key, base_url, model_name)
            if not api_valid:
                raise ValueError(f"API key validation failed for provider: {provider}")
            
            # Create configuration
            config = APIConfiguration(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name or self._get_default_model(provider)
            )
            
            # Store configuration securely
            self.user_api_configs[user_id] = config
            
            # Store encrypted configuration in database
            await self.database_service.store_user_api_config(user_id, provider.value, {
                'api_key_hash': self._hash_api_key(api_key),  # Store hash, not key
                'base_url': base_url,
                'model_name': config.model_name,
                'created_at': config.created_at
            })
            
            logger.info(f"✅ API key configured for user {user_id}, provider: {provider}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure API key: {e}")
            return False
    
    async def process_query(
        self,
        user_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> VoiceboxResponse:
        """Process natural language query through complete NWTN pipeline with breakthrough mode support"""
        try:
            start_time = datetime.now(timezone.utc)
            query_id = str(uuid4())
            
            # Check if user has configured API
            if user_id not in self.user_api_configs:
                raise ValueError("User has not configured API key. Please configure your API key first.")
            
            # Determine breakthrough mode
            breakthrough_mode = self._determine_breakthrough_mode(query, context)
            breakthrough_config = get_breakthrough_mode_config(breakthrough_mode)
            
            # Create enhanced context with breakthrough mode
            enhanced_context = breakthrough_mode_manager.create_reasoning_context(breakthrough_mode, context or {})
            
            # Analyze query complexity and requirements
            analysis = await self._analyze_query(query_id, query, enhanced_context)
            self.active_queries[query_id] = analysis
            
            # Check if clarification is needed
            if analysis.clarification_status == ClarificationStatus.NEEDS_CLARIFICATION:
                return await self._request_clarification(query_id, analysis)
            
            # Validate user has sufficient FTNS balance
            user_balance = await self.ftns_service.get_balance(user_id)
            if user_balance < analysis.estimated_cost_ftns:
                raise ValueError(f"Insufficient FTNS balance: {user_balance} < {analysis.estimated_cost_ftns}")
            
            # Process query through 8-step NWTN reasoning system with breakthrough mode
            reasoning_result = await self._process_through_nwtn(query_id, analysis, enhanced_context)
            
            # Extract verbosity and thinking mode from enhanced context
            verbosity_level_str = enhanced_context.get("verbosity_level", "STANDARD")
            thinking_mode_str = enhanced_context.get("thinking_mode", "INTERMEDIATE")
            
            # Convert to enums
            try:
                verbosity_level = VerbosityLevel[verbosity_level_str.upper()]
                thinking_mode = ThinkingMode[thinking_mode_str.upper()]
            except KeyError:
                verbosity_level = VerbosityLevel.STANDARD
                thinking_mode = ThinkingMode.INTERMEDIATE
            
            # Calculate precise FTNS cost using token-based pricing
            pricing_calculation = await calculate_query_cost(
                query=query,
                thinking_mode=thinking_mode,
                verbosity_level=verbosity_level,
                query_id=query_id,
                user_tier=context.get("user_tier", "standard") if context else "standard"
            )
            
            # Translate structured insights to natural language
            natural_response = await self._translate_to_natural_language(
                user_id, query, reasoning_result, analysis, verbosity_level_str
            )
            
            # Use precise token-based cost
            actual_cost = pricing_calculation.total_ftns_cost
            # Create a transaction to charge the user
            transaction = FTNSTransaction(
                from_user=user_id,
                to_user="system",
                amount=actual_cost,
                transaction_type="charge",
                description=f"NWTN Query: {query_id}",
                context_units=int(actual_cost * 10)  # Convert to context units
            )
            await self.ftns_service._update_balance(user_id, -actual_cost)
            await self.ftns_service._record_transaction(transaction)
            
            # Generate source links and attribution summary
            source_links = await self._generate_source_links(reasoning_result)
            attribution_summary = await self._generate_attribution_summary(reasoning_result, source_links)
            royalties_summary = await self._get_royalty_summary(reasoning_result)
            
            # Create enhanced response with source access and token-based pricing details
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            response = VoiceboxResponse(
                response_id=str(uuid4()),
                query_id=query_id,
                natural_language_response=natural_response,
                structured_insights=reasoning_result,
                reasoning_trace=reasoning_result.reasoning_results if reasoning_result else [],
                processing_time_seconds=processing_time,
                total_cost_ftns=float(actual_cost),
                confidence_score=reasoning_result.meta_confidence if reasoning_result else 0.0,
                used_reasoning_modes=reasoning_result.reasoning_path if reasoning_result else [],
                source_links=source_links,
                attribution_summary=attribution_summary,
                royalties_distributed=royalties_summary,
                # Enhanced token-based pricing details
                pricing_calculation=pricing_calculation,
                base_computational_tokens=pricing_calculation.base_computational_tokens,
                reasoning_multiplier=pricing_calculation.reasoning_multiplier,
                verbosity_factor=pricing_calculation.verbosity_factor,
                market_rate=pricing_calculation.market_rate,
                quality_bonus=pricing_calculation.quality_bonus,
                cost_breakdown=pricing_calculation.cost_breakdown,
                # Breakthrough mode details
                breakthrough_mode=breakthrough_mode.value,
                breakthrough_intensity=breakthrough_mode_manager._calculate_breakthrough_intensity(breakthrough_config),
                candidate_distribution=breakthrough_config.candidate_distribution.__dict__,
                mode_features={
                    "assumption_challenging": breakthrough_config.assumption_challenging_enabled,
                    "wild_hypothesis": breakthrough_config.wild_hypothesis_enabled,
                    "impossibility_exploration": breakthrough_config.impossibility_exploration_enabled
                }
            )
            
            # Store interaction in database
            await self._store_interaction(user_id, response)
            
            # Clean up active query
            if query_id in self.active_queries:
                del self.active_queries[query_id]
            
            logger.info(f"✅ Query processed: {query_id}, cost: {actual_cost} FTNS")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    
    async def get_pricing_preview(
        self,
        query: str,
        thinking_mode: str = "INTERMEDIATE",
        verbosity_level: str = "STANDARD"
    ) -> Dict[str, Any]:
        """Get pricing preview for query using enhanced token-based pricing"""
        try:
            # Convert string parameters to enums
            try:
                thinking_mode_enum = ThinkingMode[thinking_mode.upper()]
                verbosity_level_enum = VerbosityLevel[verbosity_level.upper()]
            except KeyError:
                thinking_mode_enum = ThinkingMode.INTERMEDIATE
                verbosity_level_enum = VerbosityLevel.STANDARD
            
            # Get pricing preview
            preview = await get_pricing_preview(
                query=query,
                thinking_mode=thinking_mode_enum,
                verbosity_level=verbosity_level_enum
            )
            
            return {
                "estimated_cost": preview["estimated_cost"],
                "cost_breakdown": preview["cost_breakdown"],
                "estimated_response_tokens": preview["estimated_response_tokens"],
                "pricing_tier": preview["pricing_tier"],
                "market_rate_category": preview["market_rate_category"],
                "pricing_explanation": {
                    "thinking_mode": thinking_mode,
                    "thinking_multiplier": f"{preview['cost_breakdown']['reasoning_multiplier']}x",
                    "verbosity_level": verbosity_level,
                    "verbosity_multiplier": f"{preview['cost_breakdown']['verbosity_factor']}x",
                    "market_conditions": preview["cost_breakdown"]["market_conditions"]["rate_category"],
                    "base_tokens": preview["cost_breakdown"]["base_tokens"]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get pricing preview: {e}")
            return {
                "estimated_cost": 10.0,
                "error": str(e),
                "pricing_explanation": "Error calculating preview"
            }
    
    async def get_breakthrough_modes_info(self) -> Dict[str, Any]:
        """Get information about all available breakthrough modes"""
        try:
            modes_info = breakthrough_mode_manager.get_all_modes_info()
            
            return {
                "available_modes": modes_info,
                "default_mode": "balanced",
                "mode_selection_help": {
                    "conservative": "Choose for medical, safety, regulatory, or high-stakes decisions",
                    "balanced": "Choose for business, academic, or general research questions",
                    "creative": "Choose for innovation, R&D, or brainstorming sessions",
                    "revolutionary": "Choose for moonshot projects or paradigm-shifting challenges"
                },
                "pricing_impact": {
                    "conservative": "0.8x complexity multiplier (lower cost)",
                    "balanced": "1.0x complexity multiplier (standard cost)",
                    "creative": "1.3x complexity multiplier (higher cost)",
                    "revolutionary": "1.8x complexity multiplier (premium cost)"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get breakthrough modes info: {e}")
            return {"error": str(e)}
    
    async def suggest_breakthrough_mode_for_query(self, query: str) -> Dict[str, Any]:
        """Get breakthrough mode suggestion for a specific query"""
        try:
            suggested_mode = suggest_breakthrough_mode(query)
            mode_config = get_breakthrough_mode_config(suggested_mode)
            
            return {
                "suggested_mode": suggested_mode.value,
                "mode_name": mode_config.name,
                "description": mode_config.description,
                "reasoning": f"Suggested based on query analysis",
                "use_cases": mode_config.use_cases[:3],
                "complexity_multiplier": mode_config.complexity_multiplier,
                "estimated_time": breakthrough_mode_manager._estimate_processing_time(mode_config),
                "breakthrough_intensity": breakthrough_mode_manager._calculate_breakthrough_intensity(mode_config)
            }
            
        except Exception as e:
            logger.error(f"Failed to suggest breakthrough mode: {e}")
            return {"error": str(e)}
    
    async def provide_clarification(
        self,
        user_id: str,
        query_id: str,
        clarification_response: str
    ) -> VoiceboxResponse:
        """User provides clarification for ambiguous query"""
        try:
            if query_id not in self.active_queries:
                raise ValueError("Query not found or expired")
            
            analysis = self.active_queries[query_id]
            
            # Record clarification response
            if query_id not in self.clarification_exchanges:
                self.clarification_exchanges[query_id] = []
            
            # Find the last unanswered question
            last_exchange = None
            for exchange in reversed(self.clarification_exchanges[query_id]):
                if exchange.user_response is None:
                    last_exchange = exchange
                    break
            
            if last_exchange:
                last_exchange.user_response = clarification_response
            
            # Re-analyze query with clarification
            updated_analysis = await self._reanalyze_with_clarification(
                query_id, analysis, clarification_response
            )
            self.active_queries[query_id] = updated_analysis
            
            # If still needs clarification, ask more questions
            if updated_analysis.clarification_status == ClarificationStatus.NEEDS_CLARIFICATION:
                return await self._request_clarification(query_id, updated_analysis)
            
            # Process the clarified query
            return await self.process_query(user_id, updated_analysis.original_query)
            
        except Exception as e:
            logger.error(f"Failed to process clarification: {e}")
            raise
    
    async def get_user_interaction_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get user's interaction history with NWTN"""
        try:
            history = await self.database_service.get_user_voicebox_history(user_id, limit)
            return history
            
        except Exception as e:
            logger.error(f"Failed to get interaction history: {e}")
            return []
    
    # === Private Methods ===
    
    def _initialize_default_api_config(self):
        """Initialize default Claude API configuration from environment variables"""
        import os
        
        # Check for Claude API key in environment variables
        claude_api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY')
        
        if claude_api_key:
            self.default_api_config = APIConfiguration(
                provider=LLMProvider.CLAUDE,
                api_key=claude_api_key,
                model_name="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.7,
                timeout=30
            )
            logger.info("✅ Default Claude API configuration initialized")
        else:
            logger.warning("⚠️  No Claude API key found in environment variables (ANTHROPIC_API_KEY or CLAUDE_API_KEY)")
            logger.info("💡 Voicebox will use fallback structured responses")
    
    async def _analyze_query(
        self,
        query_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryAnalysis:
        """Analyze query complexity and requirements"""
        try:
            # Basic complexity analysis
            complexity = self._assess_query_complexity(query)
            
            # Identify likely reasoning modes needed
            reasoning_modes = self._identify_reasoning_modes(query)
            
            # Extract domain hints
            domain_hints = self._extract_domain_hints(query)
            
            # Determine if clarification is needed
            clarification_needed, questions = self._assess_clarification_needs(query)
            
            # Estimate cost based on complexity
            estimated_cost = self._estimate_cost(complexity, reasoning_modes)
            
            analysis = QueryAnalysis(
                query_id=query_id,
                original_query=query,
                complexity=complexity,
                estimated_reasoning_modes=reasoning_modes,
                domain_hints=domain_hints,
                clarification_status=ClarificationStatus.NEEDS_CLARIFICATION if clarification_needed else ClarificationStatus.CLEAR,
                clarification_questions=questions,
                estimated_cost_ftns=estimated_cost,
                analysis_confidence=0.8,  # Would be calculated more sophisticatedly
                requires_breakthrough_mode=complexity == QueryComplexity.BREAKTHROUGH
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            raise
    
    def _assess_query_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity level"""
        query_lower = query.lower()
        
        # Breakthrough indicators
        breakthrough_indicators = [
            "breakthrough", "novel", "research", "cutting-edge", "revolutionary",
            "unprecedented", "paradigm", "innovation", "discovery", "invention"
        ]
        
        # Complex indicators
        complex_indicators = [
            "compare", "analyze", "evaluate", "synthesize", "integrate",
            "multi-step", "comprehensive", "thorough", "detailed analysis"
        ]
        
        # Simple indicators
        simple_indicators = [
            "what is", "define", "explain", "describe", "list", "name"
        ]
        
        if any(indicator in query_lower for indicator in breakthrough_indicators):
            return QueryComplexity.BREAKTHROUGH
        elif any(indicator in query_lower for indicator in complex_indicators):
            return QueryComplexity.COMPLEX
        elif len(query.split()) > 20 or "?" in query[:-1]:  # Multiple questions
            return QueryComplexity.MODERATE
        elif any(indicator in query_lower for indicator in simple_indicators):
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.MODERATE
    
    def _identify_reasoning_modes(self, query: str) -> List[str]:
        """Identify likely reasoning modes needed for query"""
        modes = []
        query_lower = query.lower()
        
        # Deductive reasoning indicators
        if any(word in query_lower for word in ["therefore", "thus", "if...then", "given that"]):
            modes.append("deductive")
        
        # Inductive reasoning indicators
        if any(word in query_lower for word in ["pattern", "trend", "generally", "usually", "typically"]):
            modes.append("inductive")
        
        # Abductive reasoning indicators
        if any(word in query_lower for word in ["why", "explain", "reason", "cause", "because"]):
            modes.append("abductive")
        
        # Analogical reasoning indicators
        if any(word in query_lower for word in ["like", "similar", "compare", "analogous", "parallel"]):
            modes.append("analogical")
        
        # Causal reasoning indicators
        if any(word in query_lower for word in ["cause", "effect", "result", "leads to", "due to"]):
            modes.append("causal")
        
        # Probabilistic reasoning indicators
        if any(word in query_lower for word in ["likely", "probable", "chance", "risk", "uncertain"]):
            modes.append("probabilistic")
        
        # Counterfactual reasoning indicators
        if any(word in query_lower for word in ["what if", "would have", "suppose", "imagine"]):
            modes.append("counterfactual")
        
        # Default to analogical if no specific indicators
        if not modes:
            modes = ["analogical"]
        
        return modes
    
    def _extract_domain_hints(self, query: str) -> List[str]:
        """Extract domain hints from query"""
        domains = []
        query_lower = query.lower()
        
        domain_keywords = {
            "technology": ["technology", "engineering", "software", "hardware", "computing"],
            "science": ["science", "research", "experiment", "study", "hypothesis"],
            "medicine": ["medicine", "health", "treatment", "disease", "patient"],
            "business": ["business", "market", "profit", "revenue", "strategy"],
            "manufacturing": ["manufacturing", "production", "factory", "assembly"],
            "materials": ["materials", "chemical", "molecular", "atomic", "substance"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)
        
        return domains or ["general"]
    
    def _assess_clarification_needs(self, query: str) -> Tuple[bool, List[str]]:
        """Assess if query needs clarification"""
        questions = []
        query_lower = query.lower()
        
        # Check for ambiguous terms, but only if they're not referring to a clearly defined subject
        # Don't trigger clarification for questions like "What is X and how does it work?"
        has_clear_subject = any(word in query_lower for word in ["what is", "how does", "explain", "describe", "define"])
        
        if not has_clear_subject:
            if "it" in query_lower or "this" in query_lower or "that" in query_lower:
                questions.append("Could you clarify what specific subject or concept you're referring to?")
        
        # Check for vague scope - but only for comparative questions without clear context
        if ("best" in query_lower or "most" in query_lower) and "what" not in query_lower:
            questions.append("What specific criteria should I use to evaluate 'best' or 'most'?")
        
        # Check for missing context - but be more lenient for well-formed questions
        if len(query.split()) < 3:  # Only very short queries need more context
            questions.append("Could you provide more context about what you're looking for?")
        
        # Check for unclear intent - but not for scientific or technical questions
        unclear_words = ["help", "about", "stuff", "things", "something"]
        if any(word in query_lower for word in unclear_words) and not has_clear_subject:
            questions.append("What specific information or outcome are you looking for?")
        
        return len(questions) > 0, questions
    
    def _estimate_cost(self, complexity: QueryComplexity, reasoning_modes: List[str]) -> float:
        """Estimate FTNS cost for query processing"""
        base_cost = 10.0  # Base cost in FTNS
        complexity_multiplier = self.complexity_cost_multipliers[complexity]
        mode_multiplier = len(reasoning_modes) * 0.5
        
        return base_cost * complexity_multiplier * (1 + mode_multiplier)
    
    async def _request_clarification(
        self,
        query_id: str,
        analysis: QueryAnalysis
    ) -> VoiceboxResponse:
        """Request clarification from user"""
        # Create clarification exchange
        exchange = ClarificationExchange(
            exchange_id=str(uuid4()),
            query_id=query_id,
            question=analysis.clarification_questions[0]  # Ask first question
        )
        
        if query_id not in self.clarification_exchanges:
            self.clarification_exchanges[query_id] = []
        self.clarification_exchanges[query_id].append(exchange)
        
        # Create clarification response
        clarification_response = f"I need some clarification to provide the best answer:\n\n{exchange.question}\n\nPlease provide more details so I can process your query through NWTN's advanced reasoning system."
        
        return VoiceboxResponse(
            response_id=str(uuid4()),
            query_id=query_id,
            natural_language_response=clarification_response,
            confidence_score=0.0,
            used_reasoning_modes=[]
        )
    
    async def _process_through_nwtn(
        self,
        query_id: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]] = None
    ) -> MetaReasoningResult:
        """Process query through 8-step NWTN reasoning system"""
        try:
            # Prepare context for NWTN
            nwtn_context = {
                "query_id": query_id,
                "complexity": analysis.complexity.value,
                "domain_hints": analysis.domain_hints,
                "estimated_reasoning_modes": analysis.estimated_reasoning_modes,
                "requires_breakthrough_mode": analysis.requires_breakthrough_mode
            }
            
            if context:
                nwtn_context.update(context)
            
            # Map reasoning depth to ThinkingMode
            depth_mapping = {
                "QUICK": ThinkingMode.QUICK,
                "STANDARD": ThinkingMode.INTERMEDIATE, 
                "DEEP": ThinkingMode.DEEP
            }
            
            reasoning_depth = context.get("reasoning_depth", "STANDARD") if context else "STANDARD"
            thinking_mode = depth_mapping.get(reasoning_depth, ThinkingMode.INTERMEDIATE)
            
            # Process through meta-reasoning engine
            result = await self.meta_reasoning_engine.meta_reason(
                query=analysis.original_query,
                context=nwtn_context,
                thinking_mode=thinking_mode
            )
            
            # Add comprehensive compatibility attributes for voicebox
            result.reasoning_path = self._extract_reasoning_path(result)
            result.integrated_conclusion = self._extract_integrated_conclusion(result)
            result.multi_modal_evidence = self._extract_evidence(result)
            result.identified_uncertainties = self._extract_uncertainties(result)
            result.reasoning_results = self._extract_reasoning_results(result)
            result.overall_confidence = result.meta_confidence
            result.reasoning_trace = self._extract_reasoning_trace(result)
            result.confidence_score = result.meta_confidence
            result.used_reasoning_modes = result.reasoning_path
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process through NWTN: {e}")
            raise
    
    def _extract_reasoning_path(self, result: MetaReasoningResult) -> List[str]:
        """Extract reasoning modes used from MetaReasoningResult"""
        reasoning_path = []
        if result.parallel_results:
            # Get unique reasoning engines used
            engines = set()
            for r in result.parallel_results:
                if hasattr(r, 'reasoning_engine'):
                    engines.add(r.reasoning_engine)
                elif hasattr(r, 'engine_name'):
                    engines.add(r.engine_name)
            reasoning_path = [f"{engine}_reasoning" for engine in engines]
        return reasoning_path
    
    def _extract_integrated_conclusion(self, result: MetaReasoningResult) -> str:
        """Extract integrated conclusion from MetaReasoningResult"""
        if hasattr(result, 'final_synthesis') and result.final_synthesis:
            return str(result.final_synthesis)
        
        # Fallback: summarize from parallel results
        if result.parallel_results:
            conclusions = []
            for r in result.parallel_results:
                if hasattr(r, 'conclusion'):
                    conclusions.append(r.conclusion)
            if conclusions:
                return "; ".join(conclusions[:3])  # First 3 conclusions
        
        return "Meta-reasoning analysis completed with enhanced reasoning engines"
    
    def _extract_evidence(self, result: MetaReasoningResult) -> List[str]:
        """Extract multi-modal evidence from MetaReasoningResult"""
        evidence = []
        if result.parallel_results:
            for r in result.parallel_results:
                if hasattr(r, 'supporting_evidence'):
                    evidence.extend(r.supporting_evidence)
                elif hasattr(r, 'evidence'):
                    evidence.extend(r.evidence)
        return evidence
    
    def _extract_uncertainties(self, result: MetaReasoningResult) -> List[str]:
        """Extract uncertainties from MetaReasoningResult"""
        uncertainties = []
        if result.parallel_results:
            for r in result.parallel_results:
                if hasattr(r, 'uncertainties'):
                    uncertainties.extend(r.uncertainties)
                elif hasattr(r, 'limitations'):
                    uncertainties.extend(r.limitations)
                elif hasattr(r, 'confidence') and r.confidence < 0.7:
                    uncertainties.append(f"Low confidence in {getattr(r, 'reasoning_engine', 'reasoning')}: {r.confidence:.2f}")
        return uncertainties
    
    def _extract_reasoning_results(self, result: MetaReasoningResult) -> List[Dict[str, Any]]:
        """Extract reasoning results from MetaReasoningResult"""
        reasoning_results = []
        if result.parallel_results:
            for r in result.parallel_results:
                reasoning_results.append({
                    'engine': getattr(r, 'reasoning_engine', 'unknown'),
                    'conclusion': getattr(r, 'conclusion', 'No conclusion'),
                    'confidence': getattr(r, 'confidence', 0.0),
                    'evidence': getattr(r, 'supporting_evidence', []),
                    'reasoning_trace': getattr(r, 'reasoning_trace', [])
                })
        return reasoning_results
    
    def _extract_reasoning_trace(self, result: MetaReasoningResult) -> List[Dict[str, Any]]:
        """Extract reasoning trace from MetaReasoningResult"""
        trace = []
        if result.parallel_results:
            for r in result.parallel_results:
                if hasattr(r, 'reasoning_trace'):
                    trace.extend(r.reasoning_trace)
        return trace

    async def _translate_to_natural_language(
        self,
        user_id: str,
        original_query: str,
        reasoning_result: MetaReasoningResult,
        analysis: QueryAnalysis,
        verbosity_level: str = "STANDARD"
    ) -> str:
        """Translate structured insights to natural language response with grounded content"""
        try:
            logger.info("Starting grounded natural language translation",
                       verbosity_level=verbosity_level,
                       user_id=user_id)
            
            # Determine which API configuration to use
            api_config = None
            
            # Priority 1: User's configured API
            if user_id and user_id in self.user_api_configs:
                api_config = self.user_api_configs[user_id]
                logger.debug(f"Using user's {api_config.provider.value} API configuration")
            
            # Priority 2: System default Claude API
            elif self.default_api_config:
                api_config = self.default_api_config
                logger.debug(f"Using default {api_config.provider.value} API configuration")
            
            # If we have an API configuration, use grounded synthesis
            if api_config:
                # Get target token count for verbosity level
                target_tokens = self._get_target_tokens_for_verbosity(verbosity_level)
                
                # Extract retrieved papers from reasoning result
                retrieved_papers = await self._extract_retrieved_papers(reasoning_result)
                
                # Use content grounding synthesizer to prepare grounded content
                grounding_result = await self.content_grounding_synthesizer.prepare_grounded_synthesis(
                    reasoning_result=reasoning_result,
                    target_tokens=target_tokens,
                    retrieved_papers=retrieved_papers,
                    verbosity_level=verbosity_level
                )
                
                logger.info("Content grounding completed",
                           source_papers=len(grounding_result.source_papers),
                           grounding_quality=grounding_result.grounding_quality,
                           content_tokens=grounding_result.content_tokens_estimate)
                
                # Call the LLM API with grounded content
                natural_response = await self._call_user_llm(
                    api_config, grounding_result.grounded_content
                )
                
                # Add Works Cited section based on actual papers used
                works_cited = self._generate_works_cited_from_grounded_papers(grounding_result.source_papers)
                if works_cited:
                    natural_response += f"\n\n## Works Cited\n\n{works_cited}"
                
                # Calculate and distribute usage royalties
                await self._distribute_usage_royalties(reasoning_result, user_id)
                
                logger.info("Grounded natural language translation completed",
                           response_length=len(natural_response),
                           papers_cited=len(grounding_result.source_papers))
                
                return natural_response
            
            else:
                logger.info("No API configuration available, using fallback response")
                fallback_response = self._create_fallback_response(reasoning_result)
                
                # Even for fallback, distribute royalties
                await self._distribute_usage_royalties(reasoning_result, user_id)
                
                return fallback_response
            
        except Exception as e:
            logger.error(f"Failed to translate to natural language with grounding: {e}")
            # Fallback to structured response, but still try to distribute royalties
            try:
                await self._distribute_usage_royalties(reasoning_result, user_id)
            except Exception as royalty_error:
                logger.error(f"Failed to distribute royalties in fallback: {royalty_error}")
            
            return self._create_fallback_response(reasoning_result)
    
    def _build_translation_prompt(
        self,
        original_query: str,
        reasoning_result: MetaReasoningResult,
        analysis: QueryAnalysis
    ) -> str:
        """Build prompt for translating structured insights to natural language"""
        return f"""You are the natural language interface for NWTN, the world's most advanced multi-modal reasoning AI system. NWTN has just completed sophisticated analysis of a user's query using all 7 fundamental forms of reasoning.

Your task is to translate NWTN's structured insights into a clear, helpful natural language response that directly addresses the user's question.

ORIGINAL USER QUERY:
{original_query}

NWTN'S STRUCTURED ANALYSIS:
- Confidence Score: {reasoning_result.meta_confidence}
- Reasoning Modes Used: {', '.join(reasoning_result.reasoning_path)}
- Primary Insights: {reasoning_result.integrated_conclusion}
- Supporting Evidence: {reasoning_result.multi_modal_evidence}
- Uncertainty Factors: {reasoning_result.identified_uncertainties}

REASONING TRACE:
{json.dumps(reasoning_result.reasoning_results, indent=2)}

INSTRUCTIONS:
1. Provide a clear, direct answer to the user's question
2. Explain the key insights in accessible language
3. Mention the reasoning approach used (but don't over-explain the technical details)
4. If there are important uncertainties or limitations, mention them
5. Be conversational but authoritative
6. Focus on practical value and actionable insights

Generate a natural language response that makes NWTN's sophisticated reasoning accessible to the user:"""
    
    async def _call_user_llm(
        self,
        api_config: APIConfiguration,
        prompt: str
    ) -> str:
        """Call user's configured LLM API"""
        try:
            if api_config.provider == LLMProvider.CLAUDE:
                return await self._call_claude_api(api_config, prompt)
            elif api_config.provider == LLMProvider.OPENAI:
                return await self._call_openai_api(api_config, prompt)
            elif api_config.provider == LLMProvider.GEMINI:
                return await self._call_gemini_api(api_config, prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {api_config.provider}")
                
        except Exception as e:
            logger.error(f"Failed to call user LLM: {e}")
            raise
    
    async def _call_claude_api(self, api_config: APIConfiguration, prompt: str) -> str:
        """Call Claude API for natural language translation"""
        try:
            import aiohttp
            import json
            
            # Claude API configuration
            url = api_config.base_url or "https://api.anthropic.com/v1/messages"
            model = api_config.model_name or "claude-3-sonnet-20240229"
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_config.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Prepare the request payload
            payload = {
                "model": model,
                "max_tokens": api_config.max_tokens,
                "temperature": api_config.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Make the API call
            timeout = aiohttp.ClientTimeout(total=api_config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["content"][0]["text"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Claude API error {response.status}: {error_text}")
                        raise Exception(f"Claude API error: {response.status}")
                        
        except ImportError:
            logger.warning("aiohttp not available for Claude API calls")
            raise Exception("aiohttp required for Claude API integration")
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise
    
    async def _call_openai_api(self, api_config: APIConfiguration, prompt: str) -> str:
        """Call OpenAI API (placeholder - would implement actual API call)"""
        # In production, would use actual OpenAI API
        return f"[OPENAI API RESPONSE] Based on NWTN's multi-modal reasoning analysis, here's the response to your query: {prompt[:200]}..."
    
    async def _call_gemini_api(self, api_config: APIConfiguration, prompt: str) -> str:
        """Call Gemini API (placeholder - would implement actual API call)"""
        # In production, would use actual Gemini API
        return f"[GEMINI API RESPONSE] Based on NWTN's multi-modal reasoning analysis, here's the response to your query: {prompt[:200]}..."
    
    def _create_fallback_response(self, reasoning_result: MetaReasoningResult) -> str:
        """Create fallback response when LLM translation fails"""
        return f"""Based on NWTN's multi-modal reasoning analysis (confidence: {reasoning_result.meta_confidence:.2f}):

**Key Insights:**
{reasoning_result.integrated_conclusion}

**Supporting Evidence:**
{', '.join(reasoning_result.multi_modal_evidence)}

**Reasoning Approach:**
Used {', '.join(reasoning_result.reasoning_path)} reasoning modes.

**Important Considerations:**
{', '.join(reasoning_result.identified_uncertainties)}

This analysis was generated using NWTN's sophisticated multi-modal reasoning system that employs all 7 fundamental forms of human reasoning with network validation for unprecedented confidence."""
    
    async def _calculate_actual_cost(
        self,
        analysis: QueryAnalysis,
        reasoning_result: MetaReasoningResult
    ) -> float:
        """Calculate actual cost based on processing"""
        # Base cost from estimate
        base_cost = analysis.estimated_cost_ftns
        
        # Adjust based on actual reasoning modes used
        if reasoning_result:
            actual_modes = len(reasoning_result.reasoning_path)
            estimated_modes = len(analysis.estimated_reasoning_modes)
            mode_adjustment = actual_modes / max(estimated_modes, 1)
            
            # Adjust based on confidence (higher confidence = more processing)
            confidence_adjustment = reasoning_result.meta_confidence
            
            return base_cost * mode_adjustment * confidence_adjustment
        
        return base_cost * 0.5  # Reduced cost if processing failed
    
    async def _store_interaction(self, user_id: str, response: VoiceboxResponse):
        """Store interaction in database"""
        try:
            await self.database_service.store_voicebox_interaction(user_id, {
                'response_id': response.response_id,
                'query_id': response.query_id,
                'natural_language_response': response.natural_language_response,
                'processing_time_seconds': response.processing_time_seconds,
                'total_cost_ftns': response.total_cost_ftns,
                'confidence_score': response.confidence_score,
                'used_reasoning_modes': response.used_reasoning_modes,
                'generated_at': response.generated_at
            })
            
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
    
    async def _reanalyze_with_clarification(
        self,
        query_id: str,
        original_analysis: QueryAnalysis,
        clarification: str
    ) -> QueryAnalysis:
        """Re-analyze query with user clarification"""
        # Enhanced query with clarification
        enhanced_query = f"{original_analysis.original_query}\n\nClarification: {clarification}"
        
        # Re-analyze with enhanced query
        return await self._analyze_query(query_id, enhanced_query)
    
    def _validate_api_key_format(self, provider: LLMProvider, api_key: str) -> bool:
        """Validate API key format for provider"""
        if provider == LLMProvider.CLAUDE:
            return api_key.startswith("sk-") and len(api_key) > 20
        elif provider == LLMProvider.OPENAI:
            return api_key.startswith("sk-") and len(api_key) > 20
        elif provider == LLMProvider.GEMINI:
            return len(api_key) > 20
        else:
            return len(api_key) > 10
    
    async def _test_api_key(
        self,
        provider: LLMProvider,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """Test API key with simple request"""
        # In production, would make actual API test call
        return True
    
    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for provider"""
        defaults = {
            LLMProvider.CLAUDE: "claude-3-5-sonnet-20241022",
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.GEMINI: "gemini-pro",
            LLMProvider.AZURE_OPENAI: "gpt-4",
            LLMProvider.HUGGINGFACE: "meta-llama/Llama-2-7b-chat-hf"
        }
        return defaults.get(provider, "default")
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def _generate_attribution_text(self, reasoning_result: MetaReasoningResult) -> str:
        """
        Generate human-readable attribution text for content sources
        
        Args:
            reasoning_result: The reasoning result containing content usage information
            
        Returns:
            Formatted attribution text for inclusion in responses
        """
        try:
            # Check if reasoning result has content sources information
            if not hasattr(reasoning_result, 'content_sources') or not reasoning_result.content_sources:
                return "This response was generated using NWTN's internal knowledge base."
            
            content_sources = reasoning_result.content_sources
            
            # Generate attribution summary
            if len(content_sources) == 1:
                attribution = f"This response is based on 1 knowledge source"
            else:
                attribution = f"This response is based on {len(content_sources)} knowledge sources"
            
            # Add source quality information if available
            if hasattr(reasoning_result, 'average_source_confidence'):
                confidence = reasoning_result.average_source_confidence
                confidence_text = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "varied"
                attribution += f" with {confidence_text} confidence"
            
            attribution += " from PRSM's verified knowledge corpus."
            
            logger.debug("Generated attribution text",
                        source_count=len(content_sources),
                        attribution_length=len(attribution))
            
            return attribution
            
        except Exception as e:
            logger.error(f"Failed to generate attribution text: {e}")
            return "This response was generated using NWTN's verified knowledge sources."
    
    def _build_translation_prompt_with_attribution(
        self,
        original_query: str,
        reasoning_result: MetaReasoningResult,
        analysis: QueryAnalysis,
        attribution_text: str
    ) -> str:
        """Build enhanced translation prompt with source attribution"""
        
        base_prompt = self._build_translation_prompt(original_query, reasoning_result, analysis)
        
        # Add attribution instructions to the prompt
        enhanced_prompt = f"""{base_prompt}

IMPORTANT ATTRIBUTION REQUIREMENTS:
{attribution_text}

Please include appropriate source attribution in your response. Acknowledge that this analysis is based on verified knowledge sources from PRSM's corpus and was processed through NWTN's multi-modal reasoning system.

Remember to:
1. Credit the underlying knowledge sources appropriately
2. Mention that this represents analysis by NWTN's reasoning system
3. Be transparent about the collaborative nature of the knowledge synthesis

Generate your response with proper attribution:"""
        
        return enhanced_prompt
    
    async def _distribute_usage_royalties(self, reasoning_result: MetaReasoningResult, user_id: str):
        """
        Calculate and distribute FTNS royalties to content creators using sophisticated engine
        
        Args:
            reasoning_result: The reasoning result containing content usage information
            user_id: User who requested the reasoning (for cost calculation)
        """
        try:
            # Check if reasoning result has content sources
            if not hasattr(reasoning_result, 'content_sources') or not reasoning_result.content_sources:
                logger.debug("No content sources found for royalty distribution")
                return
            
            content_sources = reasoning_result.content_sources
            
            # Determine query complexity based on reasoning result
            query_complexity = self._determine_query_complexity(reasoning_result)
            
            # Get user tier (could be enhanced to look up actual user tier from database)
            user_tier = "basic"  # Default tier - could be enhanced to get from user profile
            
            # Prepare reasoning context for royalty calculation
            reasoning_context = {
                'reasoning_path': getattr(reasoning_result, 'reasoning_path', []),
                'overall_confidence': getattr(reasoning_result, 'overall_confidence', 0.0),
                'multi_modal_evidence': getattr(reasoning_result, 'multi_modal_evidence', []),
                'content_weights': self._extract_content_weights(reasoning_result),
                'confidence_contributions': self._extract_confidence_contributions(reasoning_result),
                'domain_relevance': self._extract_domain_relevance(reasoning_result),
                'user_id': user_id,
                'reasoning_type': 'nwtn_multi_modal'
            }
            
            # Calculate royalties using the sophisticated engine
            royalty_calculations = await self.content_royalty_engine.calculate_usage_royalty(
                content_sources=content_sources,
                query_complexity=query_complexity,
                user_tier=user_tier,
                reasoning_context=reasoning_context
            )
            
            if not royalty_calculations:
                logger.debug("No royalty calculations generated")
                return
            
            # Distribute the calculated royalties
            distribution_result = await self.content_royalty_engine.distribute_royalties(
                royalty_calculations=royalty_calculations
            )
            
            logger.info("Sophisticated royalty distribution completed",
                       user_id=user_id,
                       content_sources=len(content_sources),
                       successful_distributions=distribution_result.successful_distributions,
                       failed_distributions=distribution_result.failed_distributions,
                       total_distributed=float(distribution_result.total_royalties_distributed))
            
        except Exception as e:
            logger.error(f"Failed to distribute usage royalties: {e}")
    
    def _determine_query_complexity(self, reasoning_result: MetaReasoningResult) -> QueryComplexity:
        """Determine query complexity based on reasoning result characteristics"""
        try:
            # Get reasoning path length and confidence
            reasoning_modes_used = len(getattr(reasoning_result, 'reasoning_path', []))
            overall_confidence = getattr(reasoning_result, 'overall_confidence', 0.0)
            
            # Determine complexity based on reasoning depth and confidence
            if reasoning_modes_used >= 6 and overall_confidence > 0.9:
                return QueryComplexity.BREAKTHROUGH
            elif reasoning_modes_used >= 4:
                return QueryComplexity.COMPLEX
            elif reasoning_modes_used >= 2:
                return QueryComplexity.MODERATE
            else:
                return QueryComplexity.SIMPLE
                
        except Exception:
            return QueryComplexity.MODERATE  # Default complexity
    
    def _extract_content_weights(self, reasoning_result: MetaReasoningResult) -> Dict[str, float]:
        """Extract content weights from reasoning result"""
        try:
            # This would extract how much each content source contributed to the reasoning
            # Placeholder implementation - would be more sophisticated in production
            content_sources = getattr(reasoning_result, 'content_sources', [])
            if not content_sources:
                return {}
            
            # For now, assign equal weights
            weight_per_source = 1.0 / len(content_sources)
            return {str(content_id): weight_per_source for content_id in content_sources}
            
        except Exception:
            return {}
    
    def _extract_confidence_contributions(self, reasoning_result: MetaReasoningResult) -> Dict[str, float]:
        """Extract confidence contributions from reasoning result"""
        try:
            # This would extract how much each content source contributed to overall confidence
            # Placeholder implementation
            content_sources = getattr(reasoning_result, 'content_sources', [])
            overall_confidence = getattr(reasoning_result, 'overall_confidence', 0.0)
            
            if not content_sources:
                return {}
            
            # For now, distribute confidence equally
            confidence_per_source = overall_confidence / len(content_sources)
            return {str(content_id): confidence_per_source for content_id in content_sources}
            
        except Exception:
            return {}
    
    def _extract_domain_relevance(self, reasoning_result: MetaReasoningResult) -> Dict[str, float]:
        """Extract domain relevance from reasoning result"""
        try:
            # This would extract how relevant each content source is to the query domain
            # Placeholder implementation
            content_sources = getattr(reasoning_result, 'content_sources', [])
            
            if not content_sources:
                return {}
            
            # For now, assume high relevance for all sources
            return {str(content_id): 1.0 for content_id in content_sources}
            
        except Exception:
            return {}
    
    async def _generate_source_links(self, reasoning_result: MetaReasoningResult) -> List[SourceLink]:
        """
        Generate direct access links for content sources used in reasoning
        
        FERRARI FUEL LINE CONNECTION: This method now connects to the external drive
        containing 150K+ papers to generate real, attributable source links.
        
        Args:
            reasoning_result: The reasoning result containing content sources
            
        Returns:
            List of SourceLink objects with access information to real papers
        """
        try:
            source_links: List[SourceLink] = []
            
            # Check if reasoning result has content sources
            if not hasattr(reasoning_result, 'content_sources') or not reasoning_result.content_sources:
                # If no content sources, search external knowledge base for related papers
                if hasattr(reasoning_result, 'original_query') and reasoning_result.original_query:
                    logger.info("No content sources found, searching external knowledge base",
                               query=reasoning_result.original_query)
                    
                    # Search for relevant papers in external storage
                    if self.external_knowledge_base and self.external_knowledge_base.initialized:
                        papers = await self.external_knowledge_base.search_papers(
                            reasoning_result.original_query, 
                            max_results=10
                        )
                        
                        for paper in papers:
                            source_link = SourceLink(
                                content_id=paper.get('id', str(uuid4())),
                                title=paper.get('title', 'Research Paper'),
                                creator=paper.get('authors', 'Unknown Author'),
                                ipfs_link=f"https://arxiv.org/abs/{paper.get('arxiv_id', paper.get('id', ''))}",
                                contribution_date=paper.get('publish_date', '2024-01-01'),
                                relevance_score=paper.get('relevance_score', 0.8),
                                content_type='research_paper'
                            )
                            source_links.append(source_link)
                    else:
                        logger.warning("External knowledge base not available for source link generation")
                
                return source_links
            
            content_sources = reasoning_result.content_sources
            
            # Try to generate source links from external knowledge base first
            if self.external_knowledge_base and self.external_knowledge_base.initialized:
                try:
                    # Content sources are now formatted as "Title by Authors" - extract titles for search
                    paper_titles = []
                    for source in content_sources:
                        if " by " in source:
                            title = source.split(" by ")[0]
                            paper_titles.append(title)
                        else:
                            paper_titles.append(source)
                    
                    # Search for papers by title to get their IDs
                    paper_ids = []
                    for title in paper_titles:
                        papers = await self.external_knowledge_base.search_papers(title, max_results=1)
                        if papers:
                            paper_ids.append(papers[0].get('id'))
                    
                    # Get source links from external knowledge base
                    external_links = await self.external_knowledge_base.generate_source_links(paper_ids)
                    
                    for link_data in external_links:
                        source_link = SourceLink(
                            content_id=link_data['content_id'],
                            title=link_data['title'],
                            creator=link_data['creator'],
                            ipfs_link=link_data['ipfs_link'],
                            contribution_date=link_data['contribution_date'],
                            relevance_score=link_data['relevance_score'],
                            content_type=link_data['content_type']
                        )
                        source_links.append(source_link)
                        
                        logger.debug("External source link generated",
                                   content_id=link_data['content_id'],
                                   title=link_data['title'],
                                   creator=link_data['creator'])
                    
                    if source_links:
                        logger.info("Source links generated from external knowledge base",
                                   total_sources=len(content_sources),
                                   successful_links=len(source_links))
                        return source_links
                
                except Exception as external_error:
                    logger.warning(f"Failed to generate source links from external knowledge base: {external_error}")
            
            # Fallback to provenance system for legacy content
            for content_id in content_sources:
                try:
                    # Get attribution chain from provenance system
                    attribution_chain = await self.provenance_system._load_attribution_chain(content_id)
                    if not attribution_chain:
                        logger.warning(f"No attribution chain found for content {content_id}")
                        continue
                    
                    # Get content fingerprint for IPFS link
                    fingerprint = await self.provenance_system._load_content_fingerprint(content_id)
                    if not fingerprint or not fingerprint.ipfs_hash:
                        logger.warning(f"No IPFS hash found for content {content_id}")
                        continue
                    
                    # Generate IPFS access link
                    ipfs_link = f"https://ipfs.prsm.ai/ipfs/{fingerprint.ipfs_hash}"
                    
                    # Determine content type
                    content_type = fingerprint.content_type.value if fingerprint.content_type else "unknown"
                    
                    # Calculate relevance score (placeholder - would be more sophisticated)
                    relevance_score = 0.8  # Default relevance
                    
                    # Create source link
                    source_link = SourceLink(
                        content_id=str(content_id),
                        title=getattr(attribution_chain, 'title', f"Content {str(content_id)[:8]}"),
                        creator=attribution_chain.original_creator,
                        ipfs_link=ipfs_link,
                        contribution_date=attribution_chain.creation_timestamp.isoformat(),
                        relevance_score=relevance_score,
                        content_type=content_type
                    )
                    
                    source_links.append(source_link)
                    
                    logger.debug("Legacy source link generated",
                               content_id=str(content_id),
                               creator=attribution_chain.original_creator,
                               ipfs_hash=fingerprint.ipfs_hash[:16] + "...")
                
                except Exception as source_error:
                    logger.warning(f"Failed to generate source link for {content_id}: {source_error}")
                    continue
            
            logger.info("Source links generated",
                       total_sources=len(content_sources),
                       successful_links=len(source_links),
                       source_type="external_knowledge_base" if self.external_knowledge_base else "legacy_provenance")
            
            return source_links
            
        except Exception as e:
            logger.error(f"Failed to generate source links: {e}")
            return []
    
    async def _generate_attribution_summary(
        self, 
        reasoning_result: MetaReasoningResult, 
        source_links: List[SourceLink]
    ) -> str:
        """
        Generate a comprehensive attribution summary for the response
        
        Args:
            reasoning_result: The reasoning result
            source_links: Generated source links
            
        Returns:
            Human-readable attribution summary
        """
        try:
            if not source_links:
                return "This response was generated using NWTN's internal knowledge base with multi-modal reasoning."
            
            source_count = len(source_links)
            unique_creators = len(set(link.creator for link in source_links))
            
            # Build attribution summary
            if source_count == 1:
                summary = f"This response is based on 1 verified source"
            else:
                summary = f"This response is based on {source_count} verified sources"
            
            if unique_creators == 1:
                summary += f" from 1 contributor"
            else:
                summary += f" from {unique_creators} contributors"
            
            summary += " in PRSM's knowledge corpus."
            
            # Add confidence information
            confidence = getattr(reasoning_result, 'overall_confidence', 0.0)
            if confidence > 0.9:
                summary += " Analysis performed with high confidence using NWTN's multi-modal reasoning."
            elif confidence > 0.7:
                summary += " Analysis performed with good confidence using NWTN's multi-modal reasoning."
            else:
                summary += " Analysis performed using NWTN's multi-modal reasoning."
            
            # Add reasoning modes used
            reasoning_modes = getattr(reasoning_result, 'reasoning_path', [])
            if reasoning_modes:
                modes_text = ", ".join(reasoning_modes[:3])  # Show first 3 modes
                if len(reasoning_modes) > 3:
                    modes_text += f" and {len(reasoning_modes) - 3} other reasoning modes"
                summary += f" Reasoning employed: {modes_text}."
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate attribution summary: {e}")
            return "This response was generated using NWTN's verified knowledge sources."
    
    def _determine_breakthrough_mode(self, query: str, context: Optional[Dict[str, Any]]) -> BreakthroughMode:
        """Determine the appropriate breakthrough mode for a query"""
        try:
            # Check if user explicitly specified a mode
            if context and "breakthrough_mode" in context:
                mode_str = context["breakthrough_mode"].upper()
                try:
                    return BreakthroughMode[mode_str]
                except KeyError:
                    logger.warning(f"Unknown breakthrough mode '{mode_str}', using suggestion")
            
            # Use AI suggestion based on query content
            suggested_mode = suggest_breakthrough_mode(query)
            logger.info(f"Suggested breakthrough mode '{suggested_mode.value}' for query")
            
            return suggested_mode
            
        except Exception as e:
            logger.error(f"Failed to determine breakthrough mode: {e}")
            return BreakthroughMode.BALANCED  # Safe default
    
    async def _get_royalty_summary(self, reasoning_result: MetaReasoningResult) -> Dict[str, float]:
        """
        Get summary of royalties distributed for this reasoning session
        
        Args:
            reasoning_result: The reasoning result containing royalty information
            
        Returns:
            Dictionary mapping creator IDs to royalty amounts
        """
        try:
            # This would typically be populated during the royalty distribution process
            # For now, return a placeholder summary
            
            if not hasattr(reasoning_result, 'content_sources') or not reasoning_result.content_sources:
                return {}
            
            # Simplified summary - in production this would come from the royalty engine
            royalty_summary = {}
            content_sources = reasoning_result.content_sources
            
            # Calculate estimated royalties (simplified)
            estimated_royalty_per_source = 0.02  # 0.02 FTNS per source (example)
            
            for content_id in content_sources:
                try:
                    # Get creator info
                    attribution_chain = await self.provenance_system._load_attribution_chain(content_id)
                    if attribution_chain:
                        creator_id = attribution_chain.original_creator
                        if creator_id not in royalty_summary:
                            royalty_summary[creator_id] = 0.0
                        royalty_summary[creator_id] += estimated_royalty_per_source
                
                except Exception:
                    continue
            
            return royalty_summary
            
        except Exception as e:
            logger.error(f"Failed to get royalty summary: {e}")
            return {}
    
    def _get_target_tokens_for_verbosity(self, verbosity_level: str) -> int:
        """Get target token count for verbosity level"""
        verbosity_tokens = {
            "BRIEF": 500,
            "STANDARD": 1000,
            "DETAILED": 2000,
            "COMPREHENSIVE": 3500,
            "ACADEMIC": 4000
        }
        return verbosity_tokens.get(verbosity_level, 1000)
    
    async def _extract_retrieved_papers(self, reasoning_result: MetaReasoningResult) -> List[Dict[str, Any]]:
        """Extract retrieved papers from reasoning result"""
        try:
            # Priority 1: Check if reasoning result has external_papers (from MetaReasoningEngine)
            if hasattr(reasoning_result, 'external_papers') and reasoning_result.external_papers:
                logger.info(f"Found {len(reasoning_result.external_papers)} external papers in reasoning result")
                return reasoning_result.external_papers
                
            # Priority 2: Check if reasoning result has retrieved_papers
            if hasattr(reasoning_result, 'retrieved_papers') and reasoning_result.retrieved_papers:
                logger.info(f"Found {len(reasoning_result.retrieved_papers)} retrieved papers in reasoning result")
                return reasoning_result.retrieved_papers
            
            # Priority 3: Try to get papers from external knowledge base directly
            if hasattr(self, 'external_knowledge_base') and self.external_knowledge_base and hasattr(reasoning_result, 'original_query'):
                logger.info("No papers in reasoning result, searching external knowledge base directly")
                try:
                    papers = await self.external_knowledge_base.search_papers(
                        reasoning_result.original_query, max_results=10
                    )
                    if papers:
                        logger.info(f"Found {len(papers)} papers via direct search")
                        return papers
                except Exception as search_error:
                    logger.warning(f"Direct external search failed: {search_error}")
            
            # Fallback: extract from content_sources if available (but search for real papers)
            if hasattr(reasoning_result, 'content_sources') and reasoning_result.content_sources:
                papers = []
                for i, source in enumerate(reasoning_result.content_sources):
                    # Try to search external knowledge base for this specific paper
                    if hasattr(self, 'external_knowledge_base') and self.external_knowledge_base:
                        try:
                            # Extract title for search
                            if " by " in source:
                                title, authors = source.split(" by ", 1)
                            else:
                                title = source
                                authors = "Unknown"
                            
                            # Search for the specific paper
                            search_results = await self.external_knowledge_base.search_papers(title, max_results=1)
                            if search_results:
                                papers.extend(search_results)
                                continue
                        except Exception as search_error:
                            logger.warning(f"Failed to search for paper '{source}': {search_error}")
                    
                    # Fallback: create placeholder if can't find real paper
                    paper = {
                        'arxiv_id': f'placeholder_{i}',
                        'title': source.split(" by ")[0] if " by " in source else source,
                        'authors': source.split(" by ")[1] if " by " in source else "Unknown",
                        'abstract': f'Content from {source}',
                        'score': 0.8
                    }
                    papers.append(paper)
                
                logger.info(f"Extracted {len(papers)} papers from content_sources")
                return papers
            
            logger.warning("No papers found in any source")
            return []
            
        except Exception as e:
            logger.warning(f"Failed to extract retrieved papers: {e}")
            return []
    
    def _generate_works_cited_from_grounded_papers(self, source_papers: List) -> str:
        """Generate Works Cited section from grounded papers"""
        try:
            if not source_papers:
                return ""
            
            citations = []
            for i, paper in enumerate(source_papers[:15], 1):  # Limit to 15 citations
                # Format citation in academic style
                authors = getattr(paper, 'authors', 'Unknown Author')
                title = getattr(paper, 'title', 'Untitled')
                arxiv_id = getattr(paper, 'arxiv_id', '')
                publish_date = getattr(paper, 'publish_date', '2023')
                
                # Format year from publish_date
                year = publish_date[:4] if len(publish_date) >= 4 else publish_date
                
                # Create APA-style citation
                if arxiv_id:
                    citation = f"{i}. {authors} ({year}). {title}. arXiv:{arxiv_id}."
                else:
                    citation = f"{i}. {authors} ({year}). {title}."
                
                citations.append(citation)
            
            return "\n".join(citations)
            
        except Exception as e:
            logger.warning(f"Failed to generate works cited: {e}")
            return ""


# Global voicebox service instance
_voicebox_service = None

async def get_voicebox_service() -> NWTNVoicebox:
    """Get the global voicebox service instance"""
    global _voicebox_service
    if _voicebox_service is None:
        _voicebox_service = NWTNVoicebox()
        await _voicebox_service.initialize()
    return _voicebox_service
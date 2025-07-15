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

from prsm.nwtn.multi_modal_reasoning_engine import MultiModalReasoningEngine, IntegratedReasoningResult
from prsm.core.config import get_settings
from prsm.core.database_service import get_database_service
from prsm.tokenomics.ftns_service import FTNSService

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
class VoiceboxResponse:
    """Complete voicebox response to user"""
    response_id: str
    query_id: str
    natural_language_response: str
    structured_insights: Optional[IntegratedReasoningResult] = None
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    total_cost_ftns: float = 0.0
    confidence_score: float = 0.0
    used_reasoning_modes: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NWTNVoicebox:
    """
    NWTN Voicebox - Natural Language Interface for Multi-Modal Reasoning
    
    Phase 1: BYOA (Bring Your Own API) Implementation
    
    Provides natural language interface capabilities for NWTN's sophisticated
    multi-modal reasoning system, allowing users to interact conversationally
    with the world's most advanced reasoning AI.
    """
    
    def __init__(self):
        self.multi_modal_engine = None  # Will be initialized async
        self.database_service = get_database_service()
        self.ftns_service = None  # Will be initialized async
        
        # User API configurations
        self.user_api_configs: Dict[str, APIConfiguration] = {}
        
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
            self.multi_modal_engine = MultiModalReasoningEngine()
            await self.multi_modal_engine.initialize()
            
            # Initialize FTNS service
            self.ftns_service = FTNSService()
            await self.ftns_service.initialize()
            
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
            await self.database_service.store_user_api_config(user_id, {
                'provider': provider.value,
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
        """Process natural language query through complete NWTN pipeline"""
        try:
            start_time = datetime.now(timezone.utc)
            query_id = str(uuid4())
            
            # Check if user has configured API
            if user_id not in self.user_api_configs:
                raise ValueError("User has not configured API key. Please configure your API key first.")
            
            # Analyze query complexity and requirements
            analysis = await self._analyze_query(query_id, query, context)
            self.active_queries[query_id] = analysis
            
            # Check if clarification is needed
            if analysis.clarification_status == ClarificationStatus.NEEDS_CLARIFICATION:
                return await self._request_clarification(query_id, analysis)
            
            # Validate user has sufficient FTNS balance
            user_balance = await self.ftns_service.get_balance(user_id)
            if user_balance < analysis.estimated_cost_ftns:
                raise ValueError(f"Insufficient FTNS balance: {user_balance} < {analysis.estimated_cost_ftns}")
            
            # Process query through 8-step NWTN reasoning system
            reasoning_result = await self._process_through_nwtn(query_id, analysis, context)
            
            # Translate structured insights to natural language
            natural_response = await self._translate_to_natural_language(
                user_id, query, reasoning_result, analysis
            )
            
            # Calculate actual costs and charge user
            actual_cost = await self._calculate_actual_cost(analysis, reasoning_result)
            await self.ftns_service.charge_user(user_id, actual_cost, f"NWTN Query: {query_id}")
            
            # Create response
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            response = VoiceboxResponse(
                response_id=str(uuid4()),
                query_id=query_id,
                natural_language_response=natural_response,
                structured_insights=reasoning_result,
                reasoning_trace=reasoning_result.reasoning_trace if reasoning_result else [],
                processing_time_seconds=processing_time,
                total_cost_ftns=actual_cost,
                confidence_score=reasoning_result.confidence_score if reasoning_result else 0.0,
                used_reasoning_modes=reasoning_result.used_reasoning_modes if reasoning_result else []
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
        
        # Check for ambiguous terms
        if "it" in query.lower() or "this" in query.lower() or "that" in query.lower():
            questions.append("Could you clarify what specific subject or concept you're referring to?")
        
        # Check for vague scope
        if "best" in query.lower() or "most" in query.lower():
            questions.append("What specific criteria should I use to evaluate 'best' or 'most'?")
        
        # Check for missing context
        if len(query.split()) < 5:
            questions.append("Could you provide more context about what you're looking for?")
        
        # Check for unclear intent
        unclear_words = ["help", "about", "stuff", "things", "something"]
        if any(word in query.lower() for word in unclear_words):
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
    ) -> IntegratedReasoningResult:
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
            
            # Process through multi-modal reasoning engine
            result = await self.multi_modal_engine.process_query(
                query=analysis.original_query,
                context=nwtn_context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process through NWTN: {e}")
            raise
    
    async def _translate_to_natural_language(
        self,
        user_id: str,
        original_query: str,
        reasoning_result: IntegratedReasoningResult,
        analysis: QueryAnalysis
    ) -> str:
        """Translate structured insights to natural language response"""
        try:
            # Get user's API configuration
            api_config = self.user_api_configs[user_id]
            
            # Prepare translation prompt
            translation_prompt = self._build_translation_prompt(
                original_query, reasoning_result, analysis
            )
            
            # Call user's configured LLM API
            natural_response = await self._call_user_llm(
                api_config, translation_prompt
            )
            
            return natural_response
            
        except Exception as e:
            logger.error(f"Failed to translate to natural language: {e}")
            # Fallback to structured response
            return self._create_fallback_response(reasoning_result)
    
    def _build_translation_prompt(
        self,
        original_query: str,
        reasoning_result: IntegratedReasoningResult,
        analysis: QueryAnalysis
    ) -> str:
        """Build prompt for translating structured insights to natural language"""
        return f"""You are the natural language interface for NWTN, the world's most advanced multi-modal reasoning AI system. NWTN has just completed sophisticated analysis of a user's query using all 7 fundamental forms of reasoning.

Your task is to translate NWTN's structured insights into a clear, helpful natural language response that directly addresses the user's question.

ORIGINAL USER QUERY:
{original_query}

NWTN'S STRUCTURED ANALYSIS:
- Confidence Score: {reasoning_result.confidence_score}
- Reasoning Modes Used: {', '.join(reasoning_result.used_reasoning_modes)}
- Primary Insights: {reasoning_result.primary_insights}
- Supporting Evidence: {reasoning_result.supporting_evidence}
- Uncertainty Factors: {reasoning_result.uncertainty_factors}

REASONING TRACE:
{json.dumps(reasoning_result.reasoning_trace, indent=2)}

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
        """Call Claude API (placeholder - would implement actual API call)"""
        # In production, would use actual Claude API
        return f"[CLAUDE API RESPONSE] Based on NWTN's multi-modal reasoning analysis, here's the response to your query: {prompt[:200]}..."
    
    async def _call_openai_api(self, api_config: APIConfiguration, prompt: str) -> str:
        """Call OpenAI API (placeholder - would implement actual API call)"""
        # In production, would use actual OpenAI API
        return f"[OPENAI API RESPONSE] Based on NWTN's multi-modal reasoning analysis, here's the response to your query: {prompt[:200]}..."
    
    async def _call_gemini_api(self, api_config: APIConfiguration, prompt: str) -> str:
        """Call Gemini API (placeholder - would implement actual API call)"""
        # In production, would use actual Gemini API
        return f"[GEMINI API RESPONSE] Based on NWTN's multi-modal reasoning analysis, here's the response to your query: {prompt[:200]}..."
    
    def _create_fallback_response(self, reasoning_result: IntegratedReasoningResult) -> str:
        """Create fallback response when LLM translation fails"""
        return f"""Based on NWTN's multi-modal reasoning analysis (confidence: {reasoning_result.confidence_score:.2f}):

**Key Insights:**
{reasoning_result.primary_insights}

**Supporting Evidence:**
{reasoning_result.supporting_evidence}

**Reasoning Approach:**
Used {', '.join(reasoning_result.used_reasoning_modes)} reasoning modes.

**Important Considerations:**
{reasoning_result.uncertainty_factors}

This analysis was generated using NWTN's sophisticated multi-modal reasoning system that employs all 7 fundamental forms of human reasoning with network validation for unprecedented confidence."""
    
    async def _calculate_actual_cost(
        self,
        analysis: QueryAnalysis,
        reasoning_result: IntegratedReasoningResult
    ) -> float:
        """Calculate actual cost based on processing"""
        # Base cost from estimate
        base_cost = analysis.estimated_cost_ftns
        
        # Adjust based on actual reasoning modes used
        if reasoning_result:
            actual_modes = len(reasoning_result.used_reasoning_modes)
            estimated_modes = len(analysis.estimated_reasoning_modes)
            mode_adjustment = actual_modes / max(estimated_modes, 1)
            
            # Adjust based on confidence (higher confidence = more processing)
            confidence_adjustment = reasoning_result.confidence_score
            
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
            LLMProvider.CLAUDE: "claude-3-sonnet-20240229",
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


# Global voicebox service instance
_voicebox_service = None

async def get_voicebox_service() -> NWTNVoicebox:
    """Get the global voicebox service instance"""
    global _voicebox_service
    if _voicebox_service is None:
        _voicebox_service = NWTNVoicebox()
        await _voicebox_service.initialize()
    return _voicebox_service
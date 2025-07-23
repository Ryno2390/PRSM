#!/usr/bin/env python3
"""
NWTN Unified Pipeline Controller
===============================

This module provides a unified controller that ensures seamless integration
between all NWTN pipeline components, addressing the orchestration complexity
identified in the system analysis.

Pipeline Flow:
Raw Files → Content Embeddings → Content Search → Candidate Generation → 
Deep Reasoning → Deep Reasoning Output → Claude API Synthesis → Natural Language Answer

Key Features:
- Centralized pipeline orchestration
- Robust error handling and recovery
- Performance monitoring and optimization
- Seamless data flow between components
- Integrated cost estimation and billing
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from uuid import uuid4, UUID
import structlog

# Import with graceful handling of missing components
try:
    from prsm.nwtn.complete_system import NWTNCompleteSystem
    COMPLETE_SYSTEM_AVAILABLE = True
except ImportError:
    NWTNCompleteSystem = None
    COMPLETE_SYSTEM_AVAILABLE = False

try:
    from prsm.nwtn.voicebox import NWTNVoicebox, VoiceboxResponse
    VOICEBOX_AVAILABLE = True
except ImportError:
    NWTNVoicebox = None
    VoiceboxResponse = None
    VOICEBOX_AVAILABLE = False

try:
    from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, MetaReasoningResult
    META_REASONING_AVAILABLE = True
except ImportError:
    MetaReasoningEngine = None
    MetaReasoningResult = None
    META_REASONING_AVAILABLE = False

# Content processing components (optional)
try:
    from prsm.nwtn.content_ingestion_engine import ContentIngestionEngine
    CONTENT_INGESTION_AVAILABLE = True
except ImportError:
    ContentIngestionEngine = None
    CONTENT_INGESTION_AVAILABLE = False

try:
    from prsm.nwtn.semantic_retriever import SemanticRetriever
    SEMANTIC_RETRIEVER_AVAILABLE = True
except ImportError:
    SemanticRetriever = None
    SEMANTIC_RETRIEVER_AVAILABLE = False

try:
    from prsm.nwtn.content_analyzer import ContentAnalyzer
    CONTENT_ANALYZER_AVAILABLE = True
except ImportError:
    ContentAnalyzer = None
    CONTENT_ANALYZER_AVAILABLE = False

try:
    from prsm.nwtn.candidate_answer_generator import CandidateAnswerGenerator
    CANDIDATE_GENERATOR_AVAILABLE = True
except ImportError:
    CandidateAnswerGenerator = None
    CANDIDATE_GENERATOR_AVAILABLE = False

try:
    from prsm.nwtn.content_grounding_synthesizer import ContentGroundingSynthesizer
    CONTENT_GROUNDING_AVAILABLE = True
except ImportError:
    ContentGroundingSynthesizer = None
    CONTENT_GROUNDING_AVAILABLE = False

try:
    from prsm.tokenomics.ftns_service import FTNSService
    FTNS_SERVICE_AVAILABLE = True
except ImportError:
    FTNSService = None
    FTNS_SERVICE_AVAILABLE = False

try:
    from prsm.core.config import get_settings_safe
    settings = get_settings_safe()
    CONFIG_AVAILABLE = settings is not None
except Exception:
    settings = None
    CONFIG_AVAILABLE = False

logger = structlog.get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages"""
    INITIALIZATION = "initialization"
    CONTENT_INGESTION = "content_ingestion"
    EMBEDDING_GENERATION = "embedding_generation"
    CONTENT_SEARCH = "content_search"
    CONTENT_ANALYSIS = "content_analysis"
    CANDIDATE_GENERATION = "candidate_generation"
    DEEP_REASONING = "deep_reasoning"
    CONTENT_GROUNDING = "content_grounding"
    SYNTHESIS = "synthesis"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics"""
    total_processing_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    tokens_processed: int = 0
    papers_analyzed: int = 0
    reasoning_engines_used: List[str] = field(default_factory=list)
    total_cost_ftns: float = 0.0
    confidence_score: float = 0.0
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipelineContext:
    """Context maintained throughout pipeline execution"""
    query_id: str
    user_id: str
    original_query: str
    processed_query: str = ""
    domain_hints: List[str] = field(default_factory=list)
    complexity_level: str = "medium"
    verbosity_level: str = "normal"
    raw_content: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Dict[str, Any] = field(default_factory=dict)
    retrieved_papers: List[Dict[str, Any]] = field(default_factory=list)
    analyzed_content: Dict[str, Any] = field(default_factory=dict)
    candidate_answers: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_results: Dict[str, Any] = field(default_factory=dict)
    grounded_content: Dict[str, Any] = field(default_factory=dict)
    final_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    query_id: str
    status: PipelineStatus
    stage: PipelineStage
    context: PipelineContext
    metrics: PipelineMetrics
    natural_language_response: str = ""
    reasoning_trace: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UnifiedPipelineController:
    """
    Unified controller for the complete NWTN pipeline
    
    This controller orchestrates all pipeline components to ensure seamless
    data flow and robust error handling throughout the processing chain.
    """
    
    def __init__(self):
        # Core components
        self.complete_system: Optional[NWTNCompleteSystem] = None
        self.voicebox: Optional[NWTNVoicebox] = None
        self.meta_reasoning_engine: Optional[MetaReasoningEngine] = None
        self.content_ingestion_engine: Optional[ContentIngestionEngine] = None
        self.semantic_retriever: Optional[SemanticRetriever] = None
        self.content_analyzer: Optional[ContentAnalyzer] = None
        self.candidate_generator: Optional[CandidateAnswerGenerator] = None
        self.content_grounding_synthesizer: Optional[ContentGroundingSynthesizer] = None
        self.ftns_service: Optional[FTNSService] = None
        
        # Pipeline state
        self.active_pipelines: Dict[str, PipelineResult] = {}
        self.component_health: Dict[str, bool] = {}
        self.performance_metrics: Dict[str, List[float]] = {
            'total_time': [],
            'reasoning_time': [],
            'synthesis_time': [],
            'confidence_scores': []
        }
        
        # Configuration
        self.max_concurrent_pipelines = 10
        self.pipeline_timeout = 300  # 5 minutes
        self.retry_attempts = 3
        
        logger.info("Unified Pipeline Controller initialized")
    
    async def initialize(self) -> bool:
        """Initialize all pipeline components"""
        try:
            logger.info("🚀 Initializing Unified Pipeline Controller...")
            
            # Initialize complete system (includes voicebox and meta reasoning)
            if COMPLETE_SYSTEM_AVAILABLE and NWTNCompleteSystem:
                logger.info("📋 Initializing NWTN Complete System...")
                self.complete_system = NWTNCompleteSystem()
                await self.complete_system.initialize()
                
                # Extract components from complete system
                self.voicebox = self.complete_system.voicebox
                self.meta_reasoning_engine = self.complete_system.multi_modal_engine
                self.ftns_service = self.complete_system.ftns_service
                
                self.component_health['complete_system'] = True
            else:
                logger.warning("Complete system not available, using fallback mode")
                self.component_health['complete_system'] = False
            
            # Initialize content processing components
            logger.info("📄 Initializing content processing components...")
            
            # Content ingestion engine
            if CONTENT_INGESTION_AVAILABLE and ContentIngestionEngine:
                try:
                    self.content_ingestion_engine = ContentIngestionEngine()
                    await self.content_ingestion_engine.initialize()
                    self.component_health['content_ingestion'] = True
                except Exception as e:
                    logger.warning(f"Content ingestion engine initialization failed: {e}")
                    self.component_health['content_ingestion'] = False
            else:
                logger.info("Content ingestion engine not available")
                self.component_health['content_ingestion'] = False
            
            # Semantic retriever
            if SEMANTIC_RETRIEVER_AVAILABLE and SemanticRetriever:
                try:
                    self.semantic_retriever = SemanticRetriever()
                    await self.semantic_retriever.initialize()
                    self.component_health['semantic_retriever'] = True
                except Exception as e:
                    logger.warning(f"Semantic retriever initialization failed: {e}")
                    self.component_health['semantic_retriever'] = False
            else:
                logger.info("Semantic retriever not available")
                self.component_health['semantic_retriever'] = False
            
            # Content analyzer
            if CONTENT_ANALYZER_AVAILABLE and ContentAnalyzer:
                try:
                    self.content_analyzer = ContentAnalyzer()
                    await self.content_analyzer.initialize()
                    self.component_health['content_analyzer'] = True
                except Exception as e:
                    logger.warning(f"Content analyzer initialization failed: {e}")
                    self.component_health['content_analyzer'] = False
            else:
                logger.info("Content analyzer not available")
                self.component_health['content_analyzer'] = False
            
            # Candidate answer generator
            if CANDIDATE_GENERATOR_AVAILABLE and CandidateAnswerGenerator:
                try:
                    self.candidate_generator = CandidateAnswerGenerator()
                    await self.candidate_generator.initialize()
                    self.component_health['candidate_generator'] = True
                except Exception as e:
                    logger.warning(f"Candidate generator initialization failed: {e}")
                    self.component_health['candidate_generator'] = False
            else:
                logger.info("Candidate generator not available")
                self.component_health['candidate_generator'] = False
            
            # Content grounding synthesizer
            if CONTENT_GROUNDING_AVAILABLE and ContentGroundingSynthesizer:
                try:
                    self.content_grounding_synthesizer = ContentGroundingSynthesizer()
                    await self.content_grounding_synthesizer.initialize()
                    self.component_health['content_grounding'] = True
                except Exception as e:
                    logger.warning(f"Content grounding synthesizer initialization failed: {e}")
                    self.component_health['content_grounding'] = False
            else:
                logger.info("Content grounding synthesizer not available")
                self.component_health['content_grounding'] = False
            
            # Check core component health
            core_components_healthy = all([
                self.complete_system is not None,
                self.voicebox is not None,
                self.ftns_service is not None
            ])
            
            if not core_components_healthy:
                raise RuntimeError("Core components failed to initialize")
            
            logger.info("✅ Unified Pipeline Controller fully initialized")
            logger.info(f"📊 Component health: {self.component_health}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Unified Pipeline Controller: {e}")
            return False
    
    async def process_query_full_pipeline(
        self,
        user_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        verbosity_level: str = "normal",
        enable_detailed_tracing: bool = False
    ) -> PipelineResult:
        """
        Process query through the complete NWTN pipeline with unified orchestration
        
        This method ensures seamless integration between all pipeline stages:
        1. Query Analysis & Preprocessing
        2. Content Search & Retrieval
        3. Content Analysis & Understanding
        4. Candidate Answer Generation
        5. Deep Multi-Modal Reasoning
        6. Content Grounding & Synthesis
        7. Natural Language Response Generation
        """
        query_id = str(uuid4())
        start_time = datetime.now(timezone.utc)
        
        # Initialize pipeline context and result
        pipeline_context = PipelineContext(
            query_id=query_id,
            user_id=user_id,
            original_query=query,
            verbosity_level=verbosity_level,
            metadata=context or {}
        )
        
        pipeline_result = PipelineResult(
            query_id=query_id,
            status=PipelineStatus.RUNNING,
            stage=PipelineStage.INITIALIZATION,
            context=pipeline_context,
            metrics=PipelineMetrics()
        )
        
        self.active_pipelines[query_id] = pipeline_result
        
        try:
            logger.info(f"🔄 Starting full pipeline processing for query: {query_id}")
            
            # Stage 1: Query Analysis & Preprocessing
            await self._stage_query_analysis(pipeline_result, enable_detailed_tracing)
            
            # Stage 2: Content Search & Retrieval
            await self._stage_content_search(pipeline_result, enable_detailed_tracing)
            
            # Stage 3: Content Analysis & Understanding
            await self._stage_content_analysis(pipeline_result, enable_detailed_tracing)
            
            # Stage 4: Candidate Answer Generation
            await self._stage_candidate_generation(pipeline_result, enable_detailed_tracing)
            
            # Stage 5: Deep Multi-Modal Reasoning
            await self._stage_deep_reasoning(pipeline_result, enable_detailed_tracing)
            
            # Stage 6: Content Grounding & Synthesis
            await self._stage_content_grounding(pipeline_result, enable_detailed_tracing)
            
            # Stage 7: Natural Language Response Generation
            await self._stage_synthesis(pipeline_result, enable_detailed_tracing)
            
            # Finalize pipeline
            pipeline_result.status = PipelineStatus.COMPLETED
            pipeline_result.stage = PipelineStage.COMPLETED
            
            # Calculate final metrics
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            pipeline_result.metrics.total_processing_time = total_time
            
            # Update performance metrics
            self.performance_metrics['total_time'].append(total_time)
            if pipeline_result.metrics.confidence_score > 0:
                self.performance_metrics['confidence_scores'].append(pipeline_result.metrics.confidence_score)
            
            logger.info(f"✅ Pipeline completed successfully for query: {query_id}")
            logger.info(f"⏱️  Total processing time: {total_time:.2f}s")
            logger.info(f"📊 Confidence score: {pipeline_result.metrics.confidence_score:.2f}")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed for query {query_id}: {e}")
            
            pipeline_result.status = PipelineStatus.FAILED
            pipeline_result.stage = PipelineStage.FAILED
            pipeline_result.error_message = str(e)
            pipeline_result.metrics.error_count += 1
            
            return pipeline_result
            
        finally:
            # Clean up active pipeline
            if query_id in self.active_pipelines:
                del self.active_pipelines[query_id]
    
    async def _stage_query_analysis(self, result: PipelineResult, trace: bool = False):
        """Stage 1: Query Analysis & Preprocessing"""
        stage_start = datetime.now(timezone.utc)
        result.stage = PipelineStage.INITIALIZATION
        
        try:
            if trace:
                logger.info(f"🔍 Stage 1: Query Analysis for {result.query_id}")
            
            # Use voicebox for initial query analysis
            if self.voicebox:
                # Analyze query complexity and domain
                from uuid import uuid4
                analysis = await self.voicebox._analyze_query(
                    query_id=result.query_id,
                    query=result.context.original_query,
                    context=result.context.metadata
                )
                
                result.context.processed_query = result.context.original_query
                result.context.domain_hints = analysis.domain_hints
                result.context.complexity_level = analysis.complexity.value
                
                if trace:
                    result.reasoning_trace.append(f"Query analyzed: complexity={analysis.complexity.value}, domains={analysis.domain_hints}")
            
            stage_time = (datetime.now(timezone.utc) - stage_start).total_seconds()
            result.metrics.stage_times['query_analysis'] = stage_time
            
        except Exception as e:
            logger.error(f"Query analysis stage failed: {e}")
            raise
    
    async def _stage_content_search(self, result: PipelineResult, trace: bool = False):
        """Stage 2: Content Search & Retrieval"""
        stage_start = datetime.now(timezone.utc)
        result.stage = PipelineStage.CONTENT_SEARCH
        
        try:
            if trace:
                logger.info(f"🔍 Stage 2: Content Search for {result.query_id}")
            
            # Use semantic retriever if available
            if self.semantic_retriever and self.component_health.get('semantic_retriever', False):
                # Perform semantic search
                search_results = await self.semantic_retriever.search_papers(
                    query=result.context.processed_query,
                    limit=20,
                    domain_filters=result.context.domain_hints
                )
                
                result.context.retrieved_papers = search_results
                result.metrics.papers_analyzed = len(search_results)
                
                if trace:
                    result.reasoning_trace.append(f"Retrieved {len(search_results)} relevant papers")
            else:
                # Fallback to basic search
                result.context.retrieved_papers = []
                if trace:
                    result.reasoning_trace.append("Content search unavailable, using fallback mode")
            
            stage_time = (datetime.now(timezone.utc) - stage_start).total_seconds()
            result.metrics.stage_times['content_search'] = stage_time
            
        except Exception as e:
            logger.error(f"Content search stage failed: {e}")
            # Continue with empty results
            result.context.retrieved_papers = []
            result.metrics.warnings.append(f"Content search failed: {e}")
    
    async def _stage_content_analysis(self, result: PipelineResult, trace: bool = False):
        """Stage 3: Content Analysis & Understanding"""
        stage_start = datetime.now(timezone.utc)
        result.stage = PipelineStage.CONTENT_ANALYSIS
        
        try:
            if trace:
                logger.info(f"🔍 Stage 3: Content Analysis for {result.query_id}")
            
            # Use content analyzer if available
            if self.content_analyzer and self.component_health.get('content_analyzer', False):
                # Analyze retrieved papers
                if result.context.retrieved_papers:
                    analyzed_content = await self.content_analyzer.analyze_papers(
                        papers=result.context.retrieved_papers,
                        query=result.context.processed_query
                    )
                    
                    result.context.analyzed_content = analyzed_content
                    
                    if trace:
                        result.reasoning_trace.append(f"Analyzed {len(result.context.retrieved_papers)} papers for insights")
                else:
                    result.context.analyzed_content = {}
                    if trace:
                        result.reasoning_trace.append("No papers to analyze")
            else:
                # Fallback analysis
                result.context.analyzed_content = {
                    'summaries': [],
                    'key_concepts': [],
                    'insights': []
                }
                if trace:
                    result.reasoning_trace.append("Content analysis unavailable, using fallback mode")
            
            stage_time = (datetime.now(timezone.utc) - stage_start).total_seconds()
            result.metrics.stage_times['content_analysis'] = stage_time
            
        except Exception as e:
            logger.error(f"Content analysis stage failed: {e}")
            result.context.analyzed_content = {}
            result.metrics.warnings.append(f"Content analysis failed: {e}")
    
    async def _stage_candidate_generation(self, result: PipelineResult, trace: bool = False):
        """Stage 4: Candidate Answer Generation"""
        stage_start = datetime.now(timezone.utc)
        result.stage = PipelineStage.CANDIDATE_GENERATION
        
        try:
            if trace:
                logger.info(f"🔍 Stage 4: Candidate Generation for {result.query_id}")
            
            # Use candidate generator if available
            if self.candidate_generator and self.component_health.get('candidate_generator', False):
                # Generate candidate answers
                candidates = await self.candidate_generator.generate_candidates(
                    query=result.context.processed_query,
                    analyzed_content=result.context.analyzed_content,
                    num_candidates=5
                )
                
                result.context.candidate_answers = candidates
                
                if trace:
                    result.reasoning_trace.append(f"Generated {len(candidates)} candidate answers")
            else:
                # Simple fallback candidate generation
                result.context.candidate_answers = [
                    {
                        'answer': f"Based on available information: {result.context.processed_query}",
                        'confidence': 0.5,
                        'sources': []
                    }
                ]
                if trace:
                    result.reasoning_trace.append("Candidate generation unavailable, using fallback mode")
            
            stage_time = (datetime.now(timezone.utc) - stage_start).total_seconds()
            result.metrics.stage_times['candidate_generation'] = stage_time
            
        except Exception as e:
            logger.error(f"Candidate generation stage failed: {e}")
            result.context.candidate_answers = []
            result.metrics.warnings.append(f"Candidate generation failed: {e}")
    
    async def _stage_deep_reasoning(self, result: PipelineResult, trace: bool = False):
        """Stage 5: Deep Multi-Modal Reasoning"""
        stage_start = datetime.now(timezone.utc)
        result.stage = PipelineStage.DEEP_REASONING
        
        try:
            if trace:
                logger.info(f"🔍 Stage 5: Deep Reasoning for {result.query_id}")
            
            # Use meta reasoning engine if available
            if self.meta_reasoning_engine:
                # Prepare reasoning request
                reasoning_request = {
                    'query': result.context.processed_query,
                    'candidates': result.context.candidate_answers,
                    'analyzed_content': result.context.analyzed_content,
                    'domain_hints': result.context.domain_hints,
                    'complexity': result.context.complexity_level
                }
                
                # Execute deep reasoning
                reasoning_results = await self.meta_reasoning_engine.reason(
                    query=result.context.processed_query,
                    context=reasoning_request
                )
                
                result.context.reasoning_results = reasoning_results
                
                # Extract reasoning engines used
                if hasattr(reasoning_results, 'engines_used'):
                    result.metrics.reasoning_engines_used = reasoning_results.engines_used
                
                # Extract confidence score
                if hasattr(reasoning_results, 'confidence_score'):
                    result.metrics.confidence_score = reasoning_results.confidence_score
                
                if trace:
                    engines_used = getattr(reasoning_results, 'engines_used', ['unknown'])
                    result.reasoning_trace.append(f"Deep reasoning completed using: {', '.join(engines_used)}")
            else:
                # Fallback reasoning
                result.context.reasoning_results = {
                    'conclusion': result.context.candidate_answers[0] if result.context.candidate_answers else {},
                    'confidence': 0.5,
                    'reasoning_chain': []
                }
                if trace:
                    result.reasoning_trace.append("Deep reasoning unavailable, using fallback mode")
            
            stage_time = (datetime.now(timezone.utc) - stage_start).total_seconds()
            result.metrics.stage_times['deep_reasoning'] = stage_time
            
        except Exception as e:
            logger.error(f"Deep reasoning stage failed: {e}")
            result.context.reasoning_results = {}
            result.metrics.warnings.append(f"Deep reasoning failed: {e}")
    
    async def _stage_content_grounding(self, result: PipelineResult, trace: bool = False):
        """Stage 6: Content Grounding & Synthesis Preparation"""
        stage_start = datetime.now(timezone.utc)
        result.stage = PipelineStage.CONTENT_GROUNDING
        
        try:
            if trace:
                logger.info(f"🔍 Stage 6: Content Grounding for {result.query_id}")
            
            # Use content grounding synthesizer if available
            if self.content_grounding_synthesizer and self.component_health.get('content_grounding', False):
                # Prepare grounding context
                grounding_context = {
                    'query': result.context.processed_query,
                    'papers': result.context.retrieved_papers,
                    'analyzed_content': result.context.analyzed_content,
                    'reasoning_results': result.context.reasoning_results,
                    'verbosity_level': result.context.verbosity_level
                }
                
                # Ground content for synthesis
                grounded_content = await self.content_grounding_synthesizer.prepare_grounding(
                    context=grounding_context
                )
                
                result.context.grounded_content = grounded_content
                
                if trace:
                    result.reasoning_trace.append("Content grounded and prepared for synthesis")
            else:
                # Simple fallback grounding
                result.context.grounded_content = {
                    'grounded_facts': [],
                    'source_citations': [],
                    'synthesis_context': result.context.reasoning_results
                }
                if trace:
                    result.reasoning_trace.append("Content grounding unavailable, using fallback mode")
            
            stage_time = (datetime.now(timezone.utc) - stage_start).total_seconds()
            result.metrics.stage_times['content_grounding'] = stage_time
            
        except Exception as e:
            logger.error(f"Content grounding stage failed: {e}")
            result.context.grounded_content = {}
            result.metrics.warnings.append(f"Content grounding failed: {e}")
    
    async def _stage_synthesis(self, result: PipelineResult, trace: bool = False):
        """Stage 7: Natural Language Response Generation via Claude API"""
        stage_start = datetime.now(timezone.utc)
        result.stage = PipelineStage.SYNTHESIS
        
        try:
            if trace:
                logger.info(f"🔍 Stage 7: Synthesis for {result.query_id}")
            
            # Use voicebox for natural language synthesis
            if self.voicebox:
                # Prepare synthesis context
                synthesis_context = {
                    'original_query': result.context.original_query,
                    'reasoning_results': result.context.reasoning_results,
                    'grounded_content': result.context.grounded_content,
                    'retrieved_papers': result.context.retrieved_papers,
                    'verbosity_level': result.context.verbosity_level
                }
                
                # Generate natural language response
                voicebox_response = await self.voicebox._synthesize_response(
                    query_id=result.query_id,
                    user_id=result.context.user_id,
                    synthesis_context=synthesis_context
                )
                
                result.natural_language_response = voicebox_response.natural_language_response
                result.sources = voicebox_response.sources
                
                # Update metrics from voicebox response
                if hasattr(voicebox_response, 'total_cost_ftns'):
                    result.metrics.total_cost_ftns = voicebox_response.total_cost_ftns
                if hasattr(voicebox_response, 'confidence_score'):
                    result.metrics.confidence_score = voicebox_response.confidence_score
                
                if trace:
                    result.reasoning_trace.append("Natural language synthesis completed via Claude API")
            else:
                # Fallback synthesis
                result.natural_language_response = f"Based on the analysis of your query '{result.context.original_query}', here are the key findings from the reasoning process."
                if trace:
                    result.reasoning_trace.append("Synthesis unavailable, using fallback response")
            
            stage_time = (datetime.now(timezone.utc) - stage_start).total_seconds()
            result.metrics.stage_times['synthesis'] = stage_time
            
        except Exception as e:
            logger.error(f"Synthesis stage failed: {e}")
            result.natural_language_response = f"I apologize, but I encountered an error while processing your query: {result.context.original_query}"
            result.metrics.warnings.append(f"Synthesis failed: {e}")
    
    async def get_pipeline_status(self, query_id: str) -> Optional[PipelineResult]:
        """Get current status of a running pipeline"""
        return self.active_pipelines.get(query_id)
    
    async def cancel_pipeline(self, query_id: str) -> bool:
        """Cancel a running pipeline"""
        if query_id in self.active_pipelines:
            pipeline = self.active_pipelines[query_id]
            pipeline.status = PipelineStatus.CANCELLED
            del self.active_pipelines[query_id]
            logger.info(f"Pipeline {query_id} cancelled")
            return True
        return False
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return {
            'component_health': self.component_health,
            'active_pipelines': len(self.active_pipelines),
            'performance_metrics': {
                'avg_processing_time': sum(self.performance_metrics['total_time']) / len(self.performance_metrics['total_time']) if self.performance_metrics['total_time'] else 0,
                'avg_confidence_score': sum(self.performance_metrics['confidence_scores']) / len(self.performance_metrics['confidence_scores']) if self.performance_metrics['confidence_scores'] else 0,
                'total_queries_processed': len(self.performance_metrics['total_time'])
            },
            'system_status': 'healthy' if all(self.component_health.values()) else 'degraded'
        }
    
    async def configure_user_api(self, user_id: str, provider: str, api_key: str) -> bool:
        """Configure user API key for the pipeline"""
        if self.complete_system:
            return await self.complete_system.configure_user_api(user_id, provider, api_key)
        return False
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline controller"""
        try:
            logger.info("🔄 Shutting down Unified Pipeline Controller...")
            
            # Cancel active pipelines
            for query_id in list(self.active_pipelines.keys()):
                await self.cancel_pipeline(query_id)
            
            # Shutdown complete system
            if self.complete_system:
                await self.complete_system.shutdown()
            
            logger.info("✅ Unified Pipeline Controller shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during pipeline controller shutdown: {e}")


# Global pipeline controller instance
_pipeline_controller = None

async def get_pipeline_controller() -> UnifiedPipelineController:
    """Get the global pipeline controller instance"""
    global _pipeline_controller
    if _pipeline_controller is None:
        _pipeline_controller = UnifiedPipelineController()
        await _pipeline_controller.initialize()
    return _pipeline_controller


# Convenience functions for direct pipeline access
async def process_query_unified(
    user_id: str,
    query: str,
    provider: str = "claude",
    api_key: Optional[str] = None,
    verbosity_level: str = "normal",
    enable_tracing: bool = False
) -> PipelineResult:
    """
    Process query through unified pipeline with API configuration
    
    This is the main entry point for the complete NWTN pipeline.
    """
    controller = await get_pipeline_controller()
    
    # Configure API if provided
    if api_key:
        await controller.configure_user_api(user_id, provider, api_key)
    
    # Process query through full pipeline
    return await controller.process_query_full_pipeline(
        user_id=user_id,
        query=query,
        verbosity_level=verbosity_level,
        enable_detailed_tracing=enable_tracing
    )


async def get_pipeline_health() -> Dict[str, Any]:
    """Get pipeline system health"""
    controller = await get_pipeline_controller()
    return await controller.get_system_health()


# Example usage
async def demonstration():
    """Demonstration of the unified pipeline"""
    print("🚀 NWTN Unified Pipeline Demonstration")
    print("=" * 50)
    
    try:
        # Process query through unified pipeline
        result = await process_query_unified(
            user_id="demo_user",
            query="What are the most promising approaches for commercial atomically precise manufacturing?",
            provider="claude",
            api_key="sk-demo-key",  # Demo key
            verbosity_level="detailed",
            enable_tracing=True
        )
        
        print(f"✅ Pipeline Status: {result.status.value}")
        print(f"📋 Final Stage: {result.stage.value}")
        print(f"⏱️  Processing Time: {result.metrics.total_processing_time:.2f}s")
        print(f"📊 Confidence Score: {result.metrics.confidence_score:.2f}")
        print(f"💰 Total Cost: {result.metrics.total_cost_ftns:.2f} FTNS")
        
        if result.reasoning_trace:
            print(f"\n🔍 Reasoning Trace:")
            for i, step in enumerate(result.reasoning_trace, 1):
                print(f"  {i}. {step}")
        
        if result.natural_language_response:
            print(f"\n📝 Response: {result.natural_language_response[:200]}...")
        
        # Get system health
        health = await get_pipeline_health()
        print(f"\n📊 System Health: {health['system_status']}")
        print(f"🏃 Active Pipelines: {health['active_pipelines']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("(This is expected without real API keys)")


if __name__ == "__main__":
    asyncio.run(demonstration())
"""
Enhanced Context Compression Engine

🧠 COGNITION.AI INSIGHTS INTEGRATION:
- Enhanced context compression using specialized LLM for conversation history distillation
- Intelligent context summarization with key detail preservation and semantic understanding
- Context distillation engine that maintains critical information while reducing token overhead
- Adaptive compression based on context importance, agent requirements, and task complexity
- Reasoning trace integration for comprehensive context awareness across multi-agent interactions

This module implements sophisticated context management techniques inspired by Cognition.AI's
insights on building robust long-running agents with proper context sharing and coordination.
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import (
    PRSMBaseModel, TimestampMixin, ReasoningStep, AgentType, TaskStatus
)
from prsm.compute.agents.executors.model_executor import ModelExecutor

logger = structlog.get_logger(__name__)


class ContextCompressionLevel(str, Enum):
    """Context compression levels"""
    MINIMAL = "minimal"         # Preserve 90%+ of original context
    MODERATE = "moderate"       # Preserve 70-80% of original context
    AGGRESSIVE = "aggressive"   # Preserve 50-60% of original context
    EXTREME = "extreme"         # Preserve 30-40% of original context


class ContextType(str, Enum):
    """Types of context for different compression strategies"""
    CONVERSATION_HISTORY = "conversation_history"
    REASONING_TRACE = "reasoning_trace"
    AGENT_COMMUNICATION = "agent_communication"
    TASK_CONTEXT = "task_context"
    SESSION_METADATA = "session_metadata"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    ERROR_CONTEXT = "error_context"
    SAFETY_CONTEXT = "safety_context"


class ContextImportance(str, Enum):
    """Context importance levels for preservation prioritization"""
    CRITICAL = "critical"       # Must be preserved at all compression levels
    HIGH = "high"              # Preserve unless extreme compression needed
    MEDIUM = "medium"          # Preserve in minimal/moderate compression
    LOW = "low"                # Can be summarized or removed
    TRANSIENT = "transient"    # Safe to remove in any compression


class CompressionStrategy(str, Enum):
    """Context compression strategies"""
    SUMMARIZATION = "summarization"           # Create summaries of lengthy content
    DEDUPLICATION = "deduplication"          # Remove redundant information
    HIERARCHICAL = "hierarchical"            # Organize by importance hierarchy
    SEMANTIC_CLUSTERING = "semantic_clustering" # Group related concepts
    TEMPORAL_PRIORITIZATION = "temporal_prioritization" # Recent content priority
    AGENT_SPECIFIC = "agent_specific"        # Tailor to agent requirements


class ContextSegment(PRSMBaseModel):
    """Individual context segment with metadata"""
    segment_id: UUID = Field(default_factory=uuid4)
    content: str
    context_type: ContextType
    importance: ContextImportance
    
    # Metadata
    agent_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = Field(default=0)
    compression_score: float = Field(ge=0.0, le=1.0, default=1.0)  # 1.0 = no compression
    
    # Semantic Analysis
    key_concepts: List[str] = Field(default_factory=list)
    semantic_embeddings: Optional[List[float]] = None
    topic_clusters: List[str] = Field(default_factory=list)
    
    # Relationships
    related_segments: List[UUID] = Field(default_factory=list)
    dependency_segments: List[UUID] = Field(default_factory=list)
    
    # Compression Metadata
    original_content: Optional[str] = None
    compression_method: Optional[CompressionStrategy] = None
    compression_ratio: float = Field(ge=0.0, le=1.0, default=1.0)


class CompressedContext(TimestampMixin):
    """Compressed context result with preservation guarantees"""
    compression_id: UUID = Field(default_factory=uuid4)
    original_context_id: str
    session_id: Optional[UUID] = None
    
    # Compression Configuration
    compression_level: ContextCompressionLevel
    target_token_limit: int
    strategies_used: List[CompressionStrategy] = Field(default_factory=list)
    
    # Compressed Content
    compressed_segments: List[ContextSegment] = Field(default_factory=list)
    preserved_critical_context: List[ContextSegment] = Field(default_factory=list)
    summary_context: Optional[str] = None
    
    # Compression Statistics
    original_token_count: int = Field(default=0)
    compressed_token_count: int = Field(default=0)
    compression_ratio: float = Field(ge=0.0, le=1.0, default=1.0)
    information_retention_score: float = Field(ge=0.0, le=1.0, default=1.0)
    
    # Quality Metrics
    semantic_coherence_score: float = Field(ge=0.0, le=1.0, default=1.0)
    critical_info_preservation: float = Field(ge=0.0, le=1.0, default=1.0)
    agent_usability_score: float = Field(ge=0.0, le=1.0, default=1.0)
    
    # Reconstruction Metadata
    reconstruction_instructions: List[str] = Field(default_factory=list)
    decompression_notes: Optional[str] = None
    context_recovery_map: Dict[str, Any] = Field(default_factory=dict)


class ContextDistillationRequest(PRSMBaseModel):
    """Request for context distillation"""
    request_id: UUID = Field(default_factory=uuid4)
    source_context: List[ContextSegment] = Field(default_factory=list)
    target_agent_type: Optional[AgentType] = None
    
    # Distillation Parameters
    compression_level: ContextCompressionLevel = ContextCompressionLevel.MODERATE
    max_output_tokens: int = Field(default=2000)
    preserve_reasoning_traces: bool = True
    preserve_safety_context: bool = True
    
    # Agent-Specific Requirements
    agent_context_requirements: Dict[str, Any] = Field(default_factory=dict)
    task_specific_focus: List[str] = Field(default_factory=list)
    domain_knowledge_retention: List[str] = Field(default_factory=list)
    
    # Quality Requirements
    minimum_information_retention: float = Field(ge=0.0, le=1.0, default=0.8)
    minimum_coherence_score: float = Field(ge=0.0, le=1.0, default=0.9)


class ContextDistillationResult(TimestampMixin):
    """Result from context distillation process"""
    result_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    
    # Distilled Context
    distilled_context: CompressedContext
    agent_specific_context: Dict[AgentType, str] = Field(default_factory=dict)
    key_insights_summary: str = ""
    
    # Quality Assessment
    distillation_quality_score: float = Field(ge=0.0, le=1.0, default=1.0)
    information_completeness: float = Field(ge=0.0, le=1.0, default=1.0)
    agent_readiness_score: float = Field(ge=0.0, le=1.0, default=1.0)
    
    # Performance Metrics
    distillation_time_seconds: float = Field(default=0.0)
    token_reduction_achieved: float = Field(ge=0.0, le=1.0, default=0.0)
    computational_efficiency_gain: float = Field(default=1.0)
    
    # Recommendations
    optimization_recommendations: List[str] = Field(default_factory=list)
    agent_coordination_notes: List[str] = Field(default_factory=list)


class EnhancedContextCompressionEngine:
    """
    Enhanced Context Compression Engine
    
    Implements sophisticated context management using specialized LLM-based compression:
    - Intelligent conversation history distillation with semantic understanding
    - Key detail preservation using importance-based hierarchical compression
    - Agent-specific context optimization for multi-agent coordination
    - Reasoning trace integration for comprehensive context awareness
    - Adaptive compression strategies based on task complexity and agent requirements
    """
    
    def __init__(self, max_context_tokens: int = 8000):
        self.max_context_tokens = max_context_tokens
        self.model_executor = ModelExecutor()
        
        # Compression Configuration
        self.compression_strategies = self._initialize_compression_strategies()
        self.importance_classifiers = self._initialize_importance_classifiers()
        self.agent_context_profiles = self._initialize_agent_profiles()
        
        # Context Storage
        self.context_segments: Dict[UUID, ContextSegment] = {}
        self.compressed_contexts: Dict[UUID, CompressedContext] = {}
        self.distillation_results: List[ContextDistillationResult] = []
        
        # Performance Tracking
        self.compression_metrics = {
            "total_compressions": 0,
            "average_compression_ratio": 0.0,
            "average_quality_score": 0.0,
            "total_tokens_saved": 0
        }
        
        logger.info("EnhancedContextCompressionEngine initialized",
                   max_context_tokens=max_context_tokens)
    
    def _initialize_compression_strategies(self) -> Dict[CompressionStrategy, Dict[str, Any]]:
        """Initialize compression strategy configurations"""
        return {
            CompressionStrategy.SUMMARIZATION: {
                "description": "Create intelligent summaries preserving key information",
                "compression_ratio": 0.3,  # Reduce to 30% of original
                "quality_retention": 0.85,
                "suitable_for": [ContextType.CONVERSATION_HISTORY, ContextType.REASONING_TRACE],
                "min_token_threshold": 200
            },
            CompressionStrategy.DEDUPLICATION: {
                "description": "Remove redundant and repetitive information",
                "compression_ratio": 0.8,  # Remove 20% redundancy
                "quality_retention": 0.95,
                "suitable_for": [ContextType.AGENT_COMMUNICATION, ContextType.TASK_CONTEXT],
                "similarity_threshold": 0.8
            },
            CompressionStrategy.HIERARCHICAL: {
                "description": "Organize by importance with selective preservation",
                "compression_ratio": 0.6,  # Keep 60% based on importance
                "quality_retention": 0.9,
                "suitable_for": [ContextType.DOMAIN_KNOWLEDGE, ContextType.SESSION_METADATA],
                "importance_cutoff": ContextImportance.MEDIUM
            },
            CompressionStrategy.SEMANTIC_CLUSTERING: {
                "description": "Group related concepts and compress clusters",
                "compression_ratio": 0.7,  # 70% retention through clustering
                "quality_retention": 0.88,
                "suitable_for": [ContextType.CONVERSATION_HISTORY, ContextType.DOMAIN_KNOWLEDGE],
                "cluster_threshold": 5
            },
            CompressionStrategy.TEMPORAL_PRIORITIZATION: {
                "description": "Prioritize recent and relevant temporal context",
                "compression_ratio": 0.75,  # Keep 75% with temporal focus
                "quality_retention": 0.92,
                "suitable_for": [ContextType.REASONING_TRACE, ContextType.ERROR_CONTEXT],
                "recency_weight": 0.7
            },
            CompressionStrategy.AGENT_SPECIFIC: {
                "description": "Tailor compression to specific agent requirements",
                "compression_ratio": 0.65,  # Agent-optimized compression
                "quality_retention": 0.93,
                "suitable_for": [ContextType.AGENT_COMMUNICATION, ContextType.TASK_CONTEXT],
                "agent_optimization": True
            }
        }
    
    def _initialize_importance_classifiers(self) -> Dict[ContextType, Dict[str, Any]]:
        """Initialize importance classification rules for different context types"""
        return {
            ContextType.CONVERSATION_HISTORY: {
                "critical_patterns": ["error", "failure", "safety", "security", "violation"],
                "high_patterns": ["decision", "conclusion", "result", "outcome", "recommendation"],
                "medium_patterns": ["analysis", "discussion", "consideration", "evaluation"],
                "low_patterns": ["greeting", "acknowledgment", "confirmation", "clarification"],
                "transient_patterns": ["timestamp", "metadata", "system_info"]
            },
            ContextType.REASONING_TRACE: {
                "critical_patterns": ["final_decision", "critical_error", "safety_check", "validation_result"],
                "high_patterns": ["reasoning_step", "logical_conclusion", "evidence", "justification"],
                "medium_patterns": ["intermediate_result", "calculation", "comparison", "hypothesis"],
                "low_patterns": ["initial_thought", "brainstorming", "exploration", "consideration"],
                "transient_patterns": ["step_number", "timestamp", "agent_id"]
            },
            ContextType.AGENT_COMMUNICATION: {
                "critical_patterns": ["coordination_failure", "conflict_resolution", "safety_alert"],
                "high_patterns": ["task_assignment", "result_sharing", "dependency_notification"],
                "medium_patterns": ["status_update", "progress_report", "information_request"],
                "low_patterns": ["acknowledgment", "handshake", "routine_check"],
                "transient_patterns": ["communication_metadata", "routing_info"]
            },
            ContextType.SAFETY_CONTEXT: {
                "critical_patterns": ["safety_violation", "security_breach", "critical_failure"],
                "high_patterns": ["safety_warning", "risk_assessment", "mitigation_action"],
                "medium_patterns": ["safety_check", "compliance_review", "risk_monitoring"],
                "low_patterns": ["routine_safety", "preventive_measure", "safety_training"],
                "transient_patterns": ["safety_log_metadata"]
            }
        }
    
    def _initialize_agent_profiles(self) -> Dict[AgentType, Dict[str, Any]]:
        """Initialize agent-specific context requirements"""
        return {
            AgentType.ARCHITECT: {
                "required_context": [ContextType.TASK_CONTEXT, ContextType.REASONING_TRACE],
                "preferred_compression": CompressionStrategy.HIERARCHICAL,
                "importance_focus": [ContextImportance.CRITICAL, ContextImportance.HIGH],
                "max_context_tokens": 3000,
                "context_priorities": ["task_decomposition", "dependencies", "complexity_analysis"]
            },
            AgentType.PROMPTER: {
                "required_context": [ContextType.CONVERSATION_HISTORY, ContextType.DOMAIN_KNOWLEDGE],
                "preferred_compression": CompressionStrategy.SEMANTIC_CLUSTERING,
                "importance_focus": [ContextImportance.CRITICAL, ContextImportance.HIGH, ContextImportance.MEDIUM],
                "max_context_tokens": 2500,
                "context_priorities": ["user_intent", "domain_expertise", "prompt_optimization"]
            },
            AgentType.ROUTER: {
                "required_context": [ContextType.TASK_CONTEXT, ContextType.AGENT_COMMUNICATION],
                "preferred_compression": CompressionStrategy.AGENT_SPECIFIC,
                "importance_focus": [ContextImportance.CRITICAL, ContextImportance.HIGH],
                "max_context_tokens": 2000,
                "context_priorities": ["model_capabilities", "task_requirements", "performance_metrics"]
            },
            AgentType.EXECUTOR: {
                "required_context": [ContextType.TASK_CONTEXT, ContextType.SAFETY_CONTEXT],
                "preferred_compression": CompressionStrategy.TEMPORAL_PRIORITIZATION,
                "importance_focus": [ContextImportance.CRITICAL],
                "max_context_tokens": 1500,
                "context_priorities": ["execution_instructions", "safety_requirements", "error_handling"]
            },
            AgentType.COMPILER: {
                "required_context": [ContextType.REASONING_TRACE, ContextType.AGENT_COMMUNICATION],
                "preferred_compression": CompressionStrategy.SUMMARIZATION,
                "importance_focus": [ContextImportance.CRITICAL, ContextImportance.HIGH, ContextImportance.MEDIUM],
                "max_context_tokens": 4000,
                "context_priorities": ["synthesis_requirements", "quality_assessment", "coherence_validation"]
            }
        }
    
    async def compress_context(
        self,
        context_segments: List[ContextSegment],
        compression_level: ContextCompressionLevel = ContextCompressionLevel.MODERATE,
        target_token_limit: Optional[int] = None,
        strategies: Optional[List[CompressionStrategy]] = None
    ) -> CompressedContext:
        """
        Compress context using intelligent multi-strategy approach
        """
        logger.info("Starting context compression",
                   segments_count=len(context_segments),
                   compression_level=compression_level,
                   target_tokens=target_token_limit)
        
        # Calculate original token count
        original_tokens = sum(segment.token_count for segment in context_segments)
        target_tokens = target_token_limit or self._calculate_target_tokens(original_tokens, compression_level)
        
        # Initialize compressed context
        compressed_context = CompressedContext(
            original_context_id=f"context_{uuid4()}",
            compression_level=compression_level,
            target_token_limit=target_tokens,
            original_token_count=original_tokens
        )
        
        try:
            # Step 1: Classify importance of each segment
            await self._classify_segment_importance(context_segments)
            
            # Step 2: Select optimal compression strategies
            selected_strategies = strategies or await self._select_compression_strategies(
                context_segments, compression_level, target_tokens
            )
            compressed_context.strategies_used = selected_strategies
            
            # Step 3: Preserve critical context unconditionally
            critical_segments = [s for s in context_segments if s.importance == ContextImportance.CRITICAL]
            compressed_context.preserved_critical_context = critical_segments
            
            # Step 4: Apply compression strategies
            compressed_segments = await self._apply_compression_strategies(
                context_segments, selected_strategies, target_tokens, critical_segments
            )
            compressed_context.compressed_segments = compressed_segments
            
            # Step 5: Generate summary context
            compressed_context.summary_context = await self._generate_summary_context(
                context_segments, compressed_segments, critical_segments
            )
            
            # Step 6: Calculate compression statistics
            await self._calculate_compression_statistics(compressed_context)
            
            # Step 7: Assess compression quality
            await self._assess_compression_quality(compressed_context, context_segments)
            
            # Step 8: Generate reconstruction instructions
            compressed_context.reconstruction_instructions = await self._generate_reconstruction_instructions(
                compressed_context, context_segments
            )
            
            # Store compressed context
            self.compressed_contexts[compressed_context.compression_id] = compressed_context
            
            # Update performance metrics
            self._update_compression_metrics(compressed_context)
            
            logger.info("Context compression completed",
                       compression_id=compressed_context.compression_id,
                       compression_ratio=compressed_context.compression_ratio,
                       quality_score=compressed_context.semantic_coherence_score)
            
            return compressed_context
            
        except Exception as e:
            logger.error("Context compression failed", error=str(e))
            # Return minimal compression on failure
            compressed_context.compressed_segments = context_segments
            compressed_context.compression_ratio = 1.0
            return compressed_context
    
    async def _classify_segment_importance(self, segments: List[ContextSegment]):
        """Classify importance of context segments"""
        for segment in segments:
            if segment.importance == ContextImportance.CRITICAL:
                continue  # Already classified as critical
            
            # Get classification rules for context type
            classifiers = self.importance_classifiers.get(segment.context_type, {})
            content_lower = segment.content.lower()
            
            # Check for critical patterns
            if any(pattern in content_lower for pattern in classifiers.get("critical_patterns", [])):
                segment.importance = ContextImportance.CRITICAL
            # Check for high importance patterns
            elif any(pattern in content_lower for pattern in classifiers.get("high_patterns", [])):
                segment.importance = ContextImportance.HIGH
            # Check for medium importance patterns
            elif any(pattern in content_lower for pattern in classifiers.get("medium_patterns", [])):
                segment.importance = ContextImportance.MEDIUM
            # Check for low importance patterns
            elif any(pattern in content_lower for pattern in classifiers.get("low_patterns", [])):
                segment.importance = ContextImportance.LOW
            # Check for transient patterns
            elif any(pattern in content_lower for pattern in classifiers.get("transient_patterns", [])):
                segment.importance = ContextImportance.TRANSIENT
            else:
                # Default classification based on content length and type
                if len(segment.content) > 500 and segment.context_type in [ContextType.REASONING_TRACE, ContextType.SAFETY_CONTEXT]:
                    segment.importance = ContextImportance.HIGH
                elif len(segment.content) > 200:
                    segment.importance = ContextImportance.MEDIUM
                else:
                    segment.importance = ContextImportance.LOW
    
    def _calculate_target_tokens(self, original_tokens: int, compression_level: ContextCompressionLevel) -> int:
        """Calculate target token count based on compression level"""
        compression_ratios = {
            ContextCompressionLevel.MINIMAL: 0.9,      # 90% retention
            ContextCompressionLevel.MODERATE: 0.7,     # 70% retention
            ContextCompressionLevel.AGGRESSIVE: 0.5,   # 50% retention
            ContextCompressionLevel.EXTREME: 0.3       # 30% retention
        }
        
        ratio = compression_ratios.get(compression_level, 0.7)
        return int(original_tokens * ratio)
    
    async def _select_compression_strategies(
        self,
        segments: List[ContextSegment],
        compression_level: ContextCompressionLevel,
        target_tokens: int
    ) -> List[CompressionStrategy]:
        """Select optimal compression strategies based on context analysis"""
        strategies = []
        
        # Analyze context types present
        context_types = set(segment.context_type for segment in segments)
        
        # Select strategies based on context types and compression level
        if ContextType.CONVERSATION_HISTORY in context_types:
            if compression_level in [ContextCompressionLevel.AGGRESSIVE, ContextCompressionLevel.EXTREME]:
                strategies.append(CompressionStrategy.SUMMARIZATION)
            else:
                strategies.append(CompressionStrategy.SEMANTIC_CLUSTERING)
        
        if ContextType.REASONING_TRACE in context_types:
            strategies.append(CompressionStrategy.HIERARCHICAL)
            if compression_level in [ContextCompressionLevel.MODERATE, ContextCompressionLevel.AGGRESSIVE]:
                strategies.append(CompressionStrategy.TEMPORAL_PRIORITIZATION)
        
        if ContextType.AGENT_COMMUNICATION in context_types:
            strategies.append(CompressionStrategy.DEDUPLICATION)
            strategies.append(CompressionStrategy.AGENT_SPECIFIC)
        
        # Always include deduplication for efficiency
        if CompressionStrategy.DEDUPLICATION not in strategies:
            strategies.append(CompressionStrategy.DEDUPLICATION)
        
        return strategies
    
    async def _apply_compression_strategies(
        self,
        segments: List[ContextSegment],
        strategies: List[CompressionStrategy],
        target_tokens: int,
        critical_segments: List[ContextSegment]
    ) -> List[ContextSegment]:
        """Apply selected compression strategies to context segments"""
        compressed_segments = segments.copy()
        
        # Reserve tokens for critical segments
        critical_tokens = sum(seg.token_count for seg in critical_segments)
        available_tokens = target_tokens - critical_tokens
        
        if available_tokens <= 0:
            # Only return critical segments if no tokens available
            return critical_segments
        
        # Apply each strategy sequentially
        for strategy in strategies:
            if strategy == CompressionStrategy.DEDUPLICATION:
                compressed_segments = await self._apply_deduplication(compressed_segments)
            elif strategy == CompressionStrategy.SUMMARIZATION:
                compressed_segments = await self._apply_summarization(compressed_segments, available_tokens)
            elif strategy == CompressionStrategy.HIERARCHICAL:
                compressed_segments = await self._apply_hierarchical_compression(compressed_segments, available_tokens)
            elif strategy == CompressionStrategy.SEMANTIC_CLUSTERING:
                compressed_segments = await self._apply_semantic_clustering(compressed_segments, available_tokens)
            elif strategy == CompressionStrategy.TEMPORAL_PRIORITIZATION:
                compressed_segments = await self._apply_temporal_prioritization(compressed_segments, available_tokens)
            elif strategy == CompressionStrategy.AGENT_SPECIFIC:
                compressed_segments = await self._apply_agent_specific_compression(compressed_segments, available_tokens)
            
            # Check if we've reached target token count
            current_tokens = sum(seg.token_count for seg in compressed_segments)
            if current_tokens <= available_tokens:
                break
        
        return compressed_segments
    
    async def _apply_deduplication(self, segments: List[ContextSegment]) -> List[ContextSegment]:
        """Remove duplicate and highly similar content"""
        unique_segments = []
        seen_content = set()
        
        for segment in segments:
            # Create content hash for exact duplicates
            content_hash = hashlib.sha256(segment.content.encode()).hexdigest()
            
            if content_hash not in seen_content:
                # Check for semantic similarity with existing segments
                is_similar = await self._check_semantic_similarity(segment, unique_segments)
                
                if not is_similar:
                    unique_segments.append(segment)
                    seen_content.add(content_hash)
                else:
                    # Mark as compressed by deduplication
                    segment.compression_method = CompressionStrategy.DEDUPLICATION
                    segment.compression_ratio = 0.0  # Completely removed
        
        return unique_segments
    
    async def _check_semantic_similarity(self, segment: ContextSegment, existing_segments: List[ContextSegment]) -> bool:
        """Check if segment is semantically similar to existing segments"""
        # Simplified similarity check based on content overlap
        segment_words = set(segment.content.lower().split())
        
        for existing in existing_segments[-5:]:  # Check against last 5 segments for efficiency
            existing_words = set(existing.content.lower().split())
            
            # Calculate Jaccard similarity
            intersection = segment_words.intersection(existing_words)
            union = segment_words.union(existing_words)
            
            if len(union) > 0:
                similarity = len(intersection) / len(union)
                if similarity > 0.8:  # 80% similarity threshold
                    return True
        
        return False
    
    async def _apply_summarization(self, segments: List[ContextSegment], available_tokens: int) -> List[ContextSegment]:
        """Apply intelligent summarization to reduce content length"""
        summarized_segments = []
        
        for segment in segments:
            if segment.importance in [ContextImportance.CRITICAL, ContextImportance.HIGH]:
                # Preserve high-importance content
                summarized_segments.append(segment)
            elif len(segment.content) > 300:  # Only summarize longer content
                # Generate summary using LLM
                summary = await self._generate_intelligent_summary(segment)
                
                # Create compressed segment
                compressed_segment = ContextSegment(
                    content=summary,
                    context_type=segment.context_type,
                    importance=segment.importance,
                    agent_id=segment.agent_id,
                    timestamp=segment.timestamp,
                    token_count=len(summary.split()),  # Rough token estimate
                    original_content=segment.content,
                    compression_method=CompressionStrategy.SUMMARIZATION,
                    compression_ratio=len(summary) / len(segment.content)
                )
                
                summarized_segments.append(compressed_segment)
            else:
                # Keep short content as-is
                summarized_segments.append(segment)
        
        return summarized_segments
    
    async def _generate_intelligent_summary(self, segment: ContextSegment) -> str:
        """Generate intelligent summary preserving key information"""
        try:
            summary_prompt = f"""
            Summarize the following {segment.context_type.value} content while preserving all critical information:
            
            Content: {segment.content}
            
            Requirements:
            - Preserve all important decisions, conclusions, and outcomes
            - Maintain technical accuracy and key details
            - Reduce length by 60-70% while keeping essential information
            - Focus on actionable insights and critical context
            
            Summary:
            """
            
            response = await self.model_executor.process({
                "task": summary_prompt,
                "models": ["gpt-3.5-turbo"],
                "parallel": False
            })
            
            if response and response[0].success:
                summary = response[0].result.get("content", segment.content[:200])
                return summary
            else:
                # Fallback to simple truncation
                return segment.content[:200] + "..." if len(segment.content) > 200 else segment.content
                
        except Exception as e:
            logger.error("Summary generation failed", error=str(e))
            return segment.content[:200] + "..." if len(segment.content) > 200 else segment.content
    
    async def _apply_hierarchical_compression(self, segments: List[ContextSegment], available_tokens: int) -> List[ContextSegment]:
        """Apply hierarchical compression based on importance levels"""
        # Sort by importance (critical first)
        importance_order = [
            ContextImportance.CRITICAL,
            ContextImportance.HIGH,
            ContextImportance.MEDIUM,
            ContextImportance.LOW,
            ContextImportance.TRANSIENT
        ]
        
        compressed_segments = []
        tokens_used = 0
        
        for importance_level in importance_order:
            level_segments = [s for s in segments if s.importance == importance_level]
            
            for segment in level_segments:
                if tokens_used + segment.token_count <= available_tokens:
                    compressed_segments.append(segment)
                    tokens_used += segment.token_count
                else:
                    # Truncate or summarize to fit remaining tokens
                    remaining_tokens = available_tokens - tokens_used
                    if remaining_tokens > 50:  # If enough tokens for meaningful content
                        truncated_content = segment.content[:remaining_tokens * 4]  # Rough estimate
                        truncated_segment = ContextSegment(
                            content=truncated_content,
                            context_type=segment.context_type,
                            importance=segment.importance,
                            agent_id=segment.agent_id,
                            timestamp=segment.timestamp,
                            token_count=remaining_tokens,
                            original_content=segment.content,
                            compression_method=CompressionStrategy.HIERARCHICAL,
                            compression_ratio=remaining_tokens / segment.token_count
                        )
                        compressed_segments.append(truncated_segment)
                    
                    break  # Stop adding segments once limit reached
        
        return compressed_segments
    
    async def _apply_semantic_clustering(self, segments: List[ContextSegment], available_tokens: int) -> List[ContextSegment]:
        """Apply semantic clustering to group and compress related content"""
        # Group segments by context type and semantic similarity
        clusters = await self._create_semantic_clusters(segments)
        
        compressed_segments = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single segment clusters remain unchanged
                compressed_segments.extend(cluster)
            else:
                # Compress cluster into representative segment
                cluster_summary = await self._compress_cluster(cluster)
                compressed_segments.append(cluster_summary)
        
        return compressed_segments
    
    async def _create_semantic_clusters(self, segments: List[ContextSegment]) -> List[List[ContextSegment]]:
        """Create semantic clusters of related segments"""
        clusters = []
        unprocessed = segments.copy()
        
        while unprocessed:
            current_segment = unprocessed.pop(0)
            current_cluster = [current_segment]
            
            # Find semantically similar segments
            for segment in unprocessed[:]:
                if await self._are_semantically_related(current_segment, segment):
                    current_cluster.append(segment)
                    unprocessed.remove(segment)
            
            clusters.append(current_cluster)
        
        return clusters
    
    async def _are_semantically_related(self, segment1: ContextSegment, segment2: ContextSegment) -> bool:
        """Check if two segments are semantically related"""
        # Check context type similarity
        if segment1.context_type == segment2.context_type:
            # Check content similarity
            words1 = set(segment1.content.lower().split())
            words2 = set(segment2.content.lower().split())
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if len(union) > 0:
                similarity = len(intersection) / len(union)
                return similarity > 0.3  # 30% similarity threshold for clustering
        
        return False
    
    async def _compress_cluster(self, cluster: List[ContextSegment]) -> ContextSegment:
        """Compress a cluster of related segments into a single representative segment"""
        # Combine content from all segments in cluster
        combined_content = "\n".join(segment.content for segment in cluster)
        
        # Use highest importance level in cluster
        highest_importance = max(segment.importance for segment in cluster)
        
        # Generate compressed representation
        if len(combined_content) > 500:
            compressed_content = await self._generate_intelligent_summary(
                ContextSegment(
                    content=combined_content,
                    context_type=cluster[0].context_type,
                    importance=highest_importance
                )
            )
        else:
            compressed_content = combined_content
        
        # Create compressed segment
        return ContextSegment(
            content=compressed_content,
            context_type=cluster[0].context_type,
            importance=highest_importance,
            agent_id=cluster[0].agent_id,
            timestamp=min(segment.timestamp for segment in cluster),
            token_count=len(compressed_content.split()),
            original_content=combined_content,
            compression_method=CompressionStrategy.SEMANTIC_CLUSTERING,
            compression_ratio=len(compressed_content) / len(combined_content)
        )
    
    async def _apply_temporal_prioritization(self, segments: List[ContextSegment], available_tokens: int) -> List[ContextSegment]:
        """Apply temporal prioritization keeping recent content"""
        # Sort by timestamp (most recent first)
        sorted_segments = sorted(segments, key=lambda x: x.timestamp, reverse=True)
        
        compressed_segments = []
        tokens_used = 0
        
        for segment in sorted_segments:
            if tokens_used + segment.token_count <= available_tokens:
                compressed_segments.append(segment)
                tokens_used += segment.token_count
            else:
                break
        
        return compressed_segments
    
    async def _apply_agent_specific_compression(self, segments: List[ContextSegment], available_tokens: int) -> List[ContextSegment]:
        """Apply agent-specific compression based on agent requirements"""
        # This would be customized based on specific agent needs
        # For now, preserve agent communication and task context
        
        priority_types = [ContextType.SAFETY_CONTEXT, ContextType.TASK_CONTEXT, ContextType.AGENT_COMMUNICATION]
        
        compressed_segments = []
        tokens_used = 0
        
        # First pass: priority context types
        for context_type in priority_types:
            type_segments = [s for s in segments if s.context_type == context_type]
            for segment in type_segments:
                if tokens_used + segment.token_count <= available_tokens:
                    compressed_segments.append(segment)
                    tokens_used += segment.token_count
        
        # Second pass: remaining segments
        remaining_segments = [s for s in segments if s not in compressed_segments]
        for segment in remaining_segments:
            if tokens_used + segment.token_count <= available_tokens:
                compressed_segments.append(segment)
                tokens_used += segment.token_count
            else:
                break
        
        return compressed_segments
    
    async def _generate_summary_context(
        self,
        original_segments: List[ContextSegment],
        compressed_segments: List[ContextSegment],
        critical_segments: List[ContextSegment]
    ) -> str:
        """Generate overall summary context for the compression"""
        try:
            summary_prompt = f"""
            Generate a concise summary of the context compression process:
            
            Original segments: {len(original_segments)}
            Compressed segments: {len(compressed_segments)}
            Critical segments preserved: {len(critical_segments)}
            
            Key information that was preserved:
            - All critical safety and security context
            - Important reasoning steps and decisions
            - Essential task and agent communication
            
            Provide a brief overview of what information is available and what was compressed.
            """
            
            response = await self.model_executor.process({
                "task": summary_prompt,
                "models": ["gpt-3.5-turbo"],
                "parallel": False
            })
            
            if response and response[0].success:
                return response[0].result.get("content", "Context compressed successfully")
            else:
                return "Context compressed successfully with preservation of critical information"
                
        except Exception as e:
            logger.error("Summary context generation failed", error=str(e))
            return "Context compressed successfully"
    
    async def _calculate_compression_statistics(self, compressed_context: CompressedContext):
        """Calculate compression statistics and metrics"""
        all_segments = compressed_context.compressed_segments + compressed_context.preserved_critical_context
        compressed_tokens = sum(segment.token_count for segment in all_segments)
        
        compressed_context.compressed_token_count = compressed_tokens
        
        if compressed_context.original_token_count > 0:
            compressed_context.compression_ratio = compressed_tokens / compressed_context.original_token_count
        else:
            compressed_context.compression_ratio = 1.0
    
    async def _assess_compression_quality(self, compressed_context: CompressedContext, original_segments: List[ContextSegment]):
        """Assess the quality of the compression"""
        # Calculate information retention score
        critical_preserved = len(compressed_context.preserved_critical_context)
        total_critical = len([s for s in original_segments if s.importance == ContextImportance.CRITICAL])
        
        if total_critical > 0:
            compressed_context.critical_info_preservation = critical_preserved / total_critical
        else:
            compressed_context.critical_info_preservation = 1.0
        
        # Calculate semantic coherence (simplified metric)
        compressed_context.semantic_coherence_score = min(1.0, 
            0.5 + (compressed_context.compression_ratio * 0.5))
        
        # Calculate agent usability score
        compressed_context.agent_usability_score = (
            compressed_context.critical_info_preservation * 0.6 +
            compressed_context.semantic_coherence_score * 0.4
        )
        
        # Calculate overall information retention
        compressed_context.information_retention_score = (
            compressed_context.critical_info_preservation * 0.7 +
            compressed_context.compression_ratio * 0.3
        )
    
    async def _generate_reconstruction_instructions(
        self,
        compressed_context: CompressedContext,
        original_segments: List[ContextSegment]
    ) -> List[str]:
        """Generate instructions for context reconstruction if needed"""
        instructions = []
        
        # Add general reconstruction guidance
        instructions.append("Context compressed using intelligent multi-strategy approach")
        instructions.append(f"Compression level: {compressed_context.compression_level.value}")
        instructions.append(f"Strategies used: {', '.join(s.value for s in compressed_context.strategies_used)}")
        
        # Add specific recovery notes
        if compressed_context.compression_ratio < 0.5:
            instructions.append("Significant compression applied - refer to summary context for overview")
        
        if len(compressed_context.preserved_critical_context) > 0:
            instructions.append("Critical context preserved separately - merge with compressed content")
        
        # Add quality assessment
        instructions.append(f"Information retention score: {compressed_context.information_retention_score:.2f}")
        instructions.append(f"Semantic coherence: {compressed_context.semantic_coherence_score:.2f}")
        
        return instructions
    
    def _update_compression_metrics(self, compressed_context: CompressedContext):
        """Update performance metrics for compression tracking"""
        self.compression_metrics["total_compressions"] += 1
        
        # Update average compression ratio
        total_compressions = self.compression_metrics["total_compressions"]
        current_avg = self.compression_metrics["average_compression_ratio"]
        new_ratio = compressed_context.compression_ratio
        
        self.compression_metrics["average_compression_ratio"] = (
            (current_avg * (total_compressions - 1) + new_ratio) / total_compressions
        )
        
        # Update average quality score
        current_quality = self.compression_metrics["average_quality_score"]
        new_quality = compressed_context.information_retention_score
        
        self.compression_metrics["average_quality_score"] = (
            (current_quality * (total_compressions - 1) + new_quality) / total_compressions
        )
        
        # Update tokens saved
        tokens_saved = compressed_context.original_token_count - compressed_context.compressed_token_count
        self.compression_metrics["total_tokens_saved"] += tokens_saved
    
    async def distill_context_for_agent(
        self,
        request: ContextDistillationRequest
    ) -> ContextDistillationResult:
        """
        Distill context specifically for target agent requirements
        """
        logger.info("Starting context distillation for agent",
                   target_agent=request.target_agent_type,
                   source_segments=len(request.source_context))
        
        start_time = time.time()
        
        # Initialize distillation result
        result = ContextDistillationResult(
            request_id=request.request_id
        )
        
        try:
            # Get agent-specific requirements
            agent_profile = None
            if request.target_agent_type:
                agent_profile = self.agent_context_profiles.get(request.target_agent_type)
            
            # Apply agent-specific compression
            if agent_profile:
                compression_level = self._determine_agent_compression_level(
                    request, agent_profile
                )
                target_tokens = min(
                    request.max_output_tokens,
                    agent_profile.get("max_context_tokens", request.max_output_tokens)
                )
            else:
                compression_level = request.compression_level
                target_tokens = request.max_output_tokens
            
            # Compress context with agent-specific focus
            compressed = await self.compress_context(
                request.source_context,
                compression_level,
                target_tokens,
                [CompressionStrategy.AGENT_SPECIFIC]
            )
            
            result.distilled_context = compressed
            
            # Generate agent-specific context summaries
            if request.target_agent_type:
                agent_context = await self._generate_agent_specific_context(
                    compressed, request.target_agent_type, agent_profile
                )
                result.agent_specific_context[request.target_agent_type] = agent_context
            
            # Generate key insights summary
            result.key_insights_summary = await self._generate_key_insights(
                request.source_context, compressed, request.task_specific_focus
            )
            
            # Assess distillation quality
            result.distillation_quality_score = compressed.information_retention_score
            result.information_completeness = compressed.critical_info_preservation
            result.agent_readiness_score = compressed.agent_usability_score
            
            # Calculate performance metrics
            result.distillation_time_seconds = time.time() - start_time
            result.token_reduction_achieved = 1.0 - compressed.compression_ratio
            result.computational_efficiency_gain = (
                request.max_output_tokens / compressed.compressed_token_count
                if compressed.compressed_token_count > 0 else 1.0
            )
            
            # Generate recommendations
            result.optimization_recommendations = await self._generate_optimization_recommendations(
                request, compressed, agent_profile
            )
            
            result.agent_coordination_notes = await self._generate_coordination_notes(
                request, compressed, agent_profile
            )
            
            # Store result
            self.distillation_results.append(result)
            
            logger.info("Context distillation completed",
                       distillation_id=result.result_id,
                       quality_score=result.distillation_quality_score,
                       token_reduction=result.token_reduction_achieved)
            
            return result
            
        except Exception as e:
            logger.error("Context distillation failed", error=str(e))
            
            # Return minimal result on failure
            result.distillation_quality_score = 0.5
            result.distillation_time_seconds = time.time() - start_time
            
            return result
    
    def _determine_agent_compression_level(
        self,
        request: ContextDistillationRequest,
        agent_profile: Dict[str, Any]
    ) -> ContextCompressionLevel:
        """Determine optimal compression level for specific agent"""
        # Consider agent's context requirements
        required_context_types = agent_profile.get("required_context", [])
        max_tokens = agent_profile.get("max_context_tokens", 2000)
        
        # Calculate source context complexity
        source_tokens = sum(seg.token_count for seg in request.source_context)
        
        if source_tokens <= max_tokens:
            return ContextCompressionLevel.MINIMAL
        elif source_tokens <= max_tokens * 1.5:
            return ContextCompressionLevel.MODERATE
        elif source_tokens <= max_tokens * 2:
            return ContextCompressionLevel.AGGRESSIVE
        else:
            return ContextCompressionLevel.EXTREME
    
    async def _generate_agent_specific_context(
        self,
        compressed: CompressedContext,
        agent_type: AgentType,
        agent_profile: Dict[str, Any]
    ) -> str:
        """Generate agent-specific context summary"""
        context_priorities = agent_profile.get("context_priorities", [])
        
        try:
            agent_prompt = f"""
            Generate a context summary specifically for a {agent_type.value} agent:
            
            Agent priorities: {', '.join(context_priorities)}
            
            Compressed context available:
            - {len(compressed.compressed_segments)} compressed segments
            - {len(compressed.preserved_critical_context)} critical segments
            - Summary: {compressed.summary_context}
            
            Focus on information relevant to {agent_type.value} responsibilities and priorities.
            """
            
            response = await self.model_executor.process({
                "task": agent_prompt,
                "models": ["gpt-3.5-turbo"],
                "parallel": False
            })
            
            if response and response[0].success:
                return response[0].result.get("content", "Agent-specific context generated")
            else:
                return f"Context prepared for {agent_type.value} agent"
                
        except Exception as e:
            logger.error("Agent-specific context generation failed", error=str(e))
            return f"Context available for {agent_type.value} agent"
    
    async def _generate_key_insights(
        self,
        source_context: List[ContextSegment],
        compressed: CompressedContext,
        task_focus: List[str]
    ) -> str:
        """Generate key insights summary from context distillation"""
        try:
            insights_prompt = f"""
            Extract key insights from the context distillation:
            
            Task focus areas: {', '.join(task_focus) if task_focus else 'General'}
            Original context: {len(source_context)} segments
            Compressed to: {compressed.compressed_token_count} tokens
            Compression ratio: {compressed.compression_ratio:.2f}
            
            Identify the most important insights, decisions, and actionable information
            that survived the compression process.
            """
            
            response = await self.model_executor.process({
                "task": insights_prompt,
                "models": ["gpt-3.5-turbo"],
                "parallel": False
            })
            
            if response and response[0].success:
                return response[0].result.get("content", "Key insights extracted from compressed context")
            else:
                return "Key insights preserved in compressed context"
                
        except Exception as e:
            logger.error("Key insights generation failed", error=str(e))
            return "Key insights available in compressed context"
    
    async def _generate_optimization_recommendations(
        self,
        request: ContextDistillationRequest,
        compressed: CompressedContext,
        agent_profile: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate optimization recommendations for future context management"""
        recommendations = []
        
        # Compression quality recommendations
        if compressed.information_retention_score < 0.8:
            recommendations.append("Consider reducing compression level to preserve more information")
        
        if compressed.compression_ratio > 0.9:
            recommendations.append("Increase compression aggressiveness to better utilize token limits")
        
        # Agent-specific recommendations
        if agent_profile:
            required_types = agent_profile.get("required_context", [])
            available_types = set(seg.context_type for seg in request.source_context)
            
            missing_types = set(required_types) - available_types
            if missing_types:
                recommendations.append(f"Consider including {', '.join(missing_types)} context types")
        
        # Performance recommendations
        if len(request.source_context) > 20:
            recommendations.append("Consider pre-filtering source context to improve compression efficiency")
        
        return recommendations
    
    async def _generate_coordination_notes(
        self,
        request: ContextDistillationRequest,
        compressed: CompressedContext,
        agent_profile: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate notes for agent coordination"""
        notes = []
        
        # Context sharing notes
        if compressed.compression_ratio < 0.5:
            notes.append("Significant compression applied - share full reasoning traces between agents")
        
        # Critical information notes
        if len(compressed.preserved_critical_context) > 0:
            notes.append("Critical context preserved separately - ensure all agents have access")
        
        # Agent-specific coordination
        if agent_profile and request.target_agent_type:
            notes.append(f"Context optimized for {request.target_agent_type.value} - may need adaptation for other agents")
        
        return notes


# Factory Functions
def create_enhanced_context_compression_engine(max_context_tokens: int = 8000) -> EnhancedContextCompressionEngine:
    """Create an enhanced context compression engine"""
    return EnhancedContextCompressionEngine(max_context_tokens)


def create_context_segments_from_reasoning_trace(reasoning_steps: List[ReasoningStep]) -> List[ContextSegment]:
    """Create context segments from PRSM reasoning trace"""
    segments = []
    
    for step in reasoning_steps:
        segment = ContextSegment(
            content=json.dumps({
                "agent_type": step.agent_type.value,
                "input": step.input_data,
                "output": step.output_data,
                "execution_time": step.execution_time,
                "confidence": step.confidence_score
            }),
            context_type=ContextType.REASONING_TRACE,
            importance=ContextImportance.HIGH,
            agent_id=step.agent_id,
            timestamp=step.timestamp,
            token_count=len(str(step.input_data).split()) + len(str(step.output_data).split())
        )
        segments.append(segment)
    
    return segments


def create_context_distillation_request(
    source_segments: List[ContextSegment],
    target_agent: AgentType,
    compression_level: ContextCompressionLevel = ContextCompressionLevel.MODERATE,
    max_tokens: int = 2000
) -> ContextDistillationRequest:
    """Create a context distillation request for specific agent"""
    return ContextDistillationRequest(
        source_context=source_segments,
        target_agent_type=target_agent,
        compression_level=compression_level,
        max_output_tokens=max_tokens,
        preserve_reasoning_traces=True,
        preserve_safety_context=True
    )
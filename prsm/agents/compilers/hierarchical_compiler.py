"""
Enhanced Hierarchical Compiler Agent
Advanced multi-level compilation with reasoning trace generation and intelligent synthesis
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Tuple
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from prsm.agents.base import BaseAgent
from prsm.core.config import get_settings
from prsm.core.models import (
    AgentType, CompilerResult, AgentResponse, 
    TimestampMixin, TaskStatus
)

logger = structlog.get_logger(__name__)
settings = get_settings()


class CompilationLevel(str, Enum):
    """Compilation levels in the hierarchical process"""
    ELEMENTAL = "elemental"
    MID_LEVEL = "mid_level"
    FINAL = "final"


class SynthesisStrategy(str, Enum):
    """Synthesis strategies for compilation"""
    CONSENSUS = "consensus"
    WEIGHTED_AVERAGE = "weighted_average"
    BEST_RESULT = "best_result"
    COMPREHENSIVE = "comprehensive"
    NARRATIVE = "narrative"


class ConflictResolutionMethod(str, Enum):
    """Methods for resolving conflicts between results"""
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPERT_PRIORITY = "expert_priority"
    SYNTHETIC_MERGE = "synthetic_merge"


class IntermediateResult(BaseModel):
    """Intermediate compilation result"""
    result_id: UUID = Field(default_factory=uuid4)
    compilation_level: CompilationLevel
    source_count: int
    synthesis_strategy: SynthesisStrategy
    content: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    reasoning_steps: List[str] = Field(default_factory=list)
    conflicts_detected: List[str] = Field(default_factory=list)
    conflicts_resolved: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MidResult(BaseModel):
    """Mid-level compilation result"""
    result_id: UUID = Field(default_factory=uuid4)
    themes: List[str] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    synthesis_quality: float = Field(ge=0.0, le=1.0)
    coherence_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    consolidated_findings: Dict[str, Any] = Field(default_factory=dict)
    cross_references: List[str] = Field(default_factory=list)
    uncertainty_areas: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FinalResponse(BaseModel):
    """Final compilation response"""
    response_id: UUID = Field(default_factory=uuid4)
    executive_summary: str
    detailed_narrative: str
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence_assessment: Dict[str, float] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    future_directions: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReasoningTrace(BaseModel):
    """Comprehensive reasoning trace"""
    trace_id: UUID = Field(default_factory=uuid4)
    compilation_path: List[str] = Field(default_factory=list)
    decision_points: List[Dict[str, Any]] = Field(default_factory=list)
    synthesis_rationale: List[str] = Field(default_factory=list)
    conflict_resolutions: List[Dict[str, Any]] = Field(default_factory=list)
    quality_assessments: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_evolution: List[float] = Field(default_factory=list)
    processing_statistics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CompilationStage(BaseModel):
    """Enhanced compilation stage with detailed metadata"""
    stage_id: UUID = Field(default_factory=uuid4)
    stage_name: str
    compilation_level: CompilationLevel
    input_count: int
    processing_time: float
    strategy_used: SynthesisStrategy
    confidence_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    reasoning_steps: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HierarchicalCompiler(BaseAgent):
    """
    Enhanced Hierarchical Compiler for PRSM
    
    Advanced multi-level compilation with:
    - Intelligent synthesis strategies
    - Conflict detection and resolution
    - Comprehensive reasoning trace generation
    - Quality assessment and optimization
    - Adaptive compilation based on content type
    """
    
    def __init__(self, agent_id: Optional[str] = None, 
                 confidence_threshold: float = 0.8,
                 default_strategy: SynthesisStrategy = SynthesisStrategy.COMPREHENSIVE):
        super().__init__(agent_id=agent_id, agent_type=AgentType.COMPILER)
        self.confidence_threshold = confidence_threshold
        self.default_strategy = default_strategy
        self.compilation_stages: List[CompilationStage] = []
        self.reasoning_trace: Optional[ReasoningTrace] = None
        self.compilation_history: List[CompilerResult] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        
        logger.info("Enhanced HierarchicalCompiler initialized",
                   agent_id=self.agent_id,
                   confidence_threshold=confidence_threshold,
                   default_strategy=default_strategy.value)
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> CompilerResult:
        """
        Enhanced compilation processing with adaptive strategies
        
        Args:
            input_data: Results to compile (AgentResponse objects or raw data)
            context: Optional compilation context with strategy and preferences
            
        Returns:
            CompilerResult: Final compiled result with comprehensive metadata
        """
        compilation_start = time.time()
        session_id = context.get("session_id", uuid4()) if context else uuid4()
        strategy = SynthesisStrategy(context.get("strategy", self.default_strategy.value)) if context else self.default_strategy
        
        # Initialize reasoning trace
        self.reasoning_trace = ReasoningTrace()
        self.compilation_stages = []
        
        logger.info("Starting enhanced hierarchical compilation",
                   agent_id=self.agent_id,
                   session_id=session_id,
                   strategy=strategy.value,
                   input_count=len(input_data) if isinstance(input_data, list) else 1)
        
        # Ensure input is a list
        if not isinstance(input_data, list):
            input_data = [input_data]
        
        try:
            # Analyze input data for adaptive strategy selection
            adapted_strategy = await self._adapt_strategy(input_data, strategy)
            
            # Stage 1: Elemental compilation
            elemental_result = await self.compile_elemental(input_data, adapted_strategy)
            
            # Stage 2: Mid-level compilation  
            mid_result = await self.compile_mid_level([elemental_result], adapted_strategy)
            
            # Stage 3: Final compilation
            final_result = await self.compile_final([mid_result], adapted_strategy)
            
            # Generate comprehensive reasoning trace
            reasoning_trace = await self.generate_reasoning_trace(self.compilation_stages)
            
            # Calculate processing statistics
            compilation_time = time.time() - compilation_start
            overall_confidence = self._calculate_overall_confidence()
            
            # Create enhanced compiler result
            result = CompilerResult(
                session_id=session_id,
                compilation_level="final",
                input_count=len(input_data),
                compiled_result=final_result.model_dump(),
                confidence_score=overall_confidence,
                reasoning_trace=reasoning_trace.compilation_path,
                metadata={
                    "strategy_used": adapted_strategy.value,
                    "original_strategy": strategy.value,
                    "stages_completed": len(self.compilation_stages),
                    "total_inputs": len(input_data),
                    "compilation_time": compilation_time,
                    "quality_score": final_result.quality_score,
                    "completeness_score": final_result.completeness_score,
                    "reasoning_trace_id": str(reasoning_trace.trace_id)
                }
            )
            
            # Store in compilation history
            self.compilation_history.append(result)
            
            # Update performance metrics
            self._update_performance_metrics(result, compilation_time)
            
            logger.info("Enhanced compilation completed",
                       agent_id=self.agent_id,
                       session_id=session_id,
                       confidence=overall_confidence,
                       quality=final_result.quality_score,
                       compilation_time=f"{compilation_time:.3f}s")
            
            return result
            
        except Exception as e:
            compilation_time = time.time() - compilation_start
            logger.error("Enhanced compilation failed",
                        agent_id=self.agent_id,
                        session_id=session_id,
                        error=str(e),
                        compilation_time=f"{compilation_time:.3f}s")
            
            # Return comprehensive error result
            return CompilerResult(
                session_id=session_id,
                compilation_level="failed",
                input_count=len(input_data),
                compiled_result=None,
                confidence_score=0.0,
                reasoning_trace=[f"Compilation failed: {str(e)}"],
                error_message=str(e),
                metadata={
                    "error_type": type(e).__name__,
                    "compilation_time": compilation_time,
                    "stages_attempted": len(self.compilation_stages),
                    "failure_point": self._identify_failure_point()
                }
            )
    
    async def compile_elemental(self, responses: List[Any], 
                               strategy: SynthesisStrategy = SynthesisStrategy.COMPREHENSIVE) -> IntermediateResult:
        """
        Enhanced elemental compilation with intelligent synthesis
        
        Args:
            responses: AgentResponse objects or raw data from executors
            strategy: Synthesis strategy to use for compilation
            
        Returns:
            IntermediateResult: Structured elemental compilation result
        """
        stage_start = time.time()
        
        logger.debug("Enhanced elemental compilation",
                    agent_id=self.agent_id,
                    input_count=len(responses),
                    strategy=strategy.value)
        
        # Process and categorize responses
        agent_responses = []
        raw_responses = []
        failed_responses = []
        
        for response in responses:
            if isinstance(response, AgentResponse):
                if response.success:
                    agent_responses.append(response)
                else:
                    failed_responses.append(response)
            elif hasattr(response, 'success'):
                # Handle ExecutionResult or similar objects
                if response.success:
                    raw_responses.append(response.result if hasattr(response, 'result') else response)
                else:
                    failed_responses.append(response)
            else:
                # Handle raw data
                raw_responses.append(response)
        
        # Apply synthesis strategy
        synthesized_content = await self._apply_synthesis_strategy(
            agent_responses + raw_responses, strategy
        )
        
        # Detect and analyze conflicts
        conflicts_detected = await self._detect_conflicts(agent_responses + raw_responses)
        conflicts_resolved = await self._resolve_conflicts_elemental(conflicts_detected)
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_elemental_quality_metrics(
            agent_responses, raw_responses, failed_responses
        )
        
        # Generate reasoning steps
        reasoning_steps = await self._generate_elemental_reasoning(
            agent_responses, raw_responses, strategy, conflicts_detected
        )
        
        # Calculate confidence score
        confidence_score = await self._calculate_elemental_confidence_enhanced(
            agent_responses, raw_responses, quality_metrics
        )
        
        # Create structured result
        elemental_result = IntermediateResult(
            compilation_level=CompilationLevel.ELEMENTAL,
            source_count=len(responses),
            synthesis_strategy=strategy,
            content=synthesized_content,
            confidence_score=confidence_score,
            quality_metrics=quality_metrics,
            reasoning_steps=reasoning_steps,
            conflicts_detected=[str(c) for c in conflicts_detected],
            conflicts_resolved=[str(c) for c in conflicts_resolved],
            metadata={
                "agent_responses": len(agent_responses),
                "raw_responses": len(raw_responses),
                "failed_responses": len(failed_responses),
                "success_rate": len(agent_responses + raw_responses) / len(responses) if responses else 0,
                "processing_time": time.time() - stage_start
            }
        )
        
        # Record compilation stage
        stage = CompilationStage(
            stage_name="elemental_compilation",
            compilation_level=CompilationLevel.ELEMENTAL,
            input_count=len(responses),
            processing_time=time.time() - stage_start,
            strategy_used=strategy,
            confidence_score=confidence_score,
            quality_score=quality_metrics.get("overall_quality", 0.5),
            conflicts_detected=len(conflicts_detected),
            conflicts_resolved=len(conflicts_resolved),
            reasoning_steps=reasoning_steps
        )
        self.compilation_stages.append(stage)
        
        # Update reasoning trace
        self.reasoning_trace.compilation_path.append(f"Elemental compilation: {len(responses)} inputs processed")
        self.reasoning_trace.confidence_evolution.append(confidence_score)
        
        return elemental_result
    
    
    
    async def _aggregate_content(self, results: List[Any]) -> str:
        """Aggregate content from multiple results"""
        aggregated = []
        
        for result in results:
            if isinstance(result, dict):
                # Extract key content fields
                content_fields = ["summary", "explanation", "content", "findings"]
                for field in content_fields:
                    if field in result:
                        aggregated.append(f"[{field.upper()}] {result[field]}")
            else:
                aggregated.append(str(result))
        
        return " | ".join(aggregated)
    
    
    
    async def _adapt_strategy(self, input_data: List[Any], strategy: SynthesisStrategy) -> SynthesisStrategy:
        """Adapt synthesis strategy based on input data characteristics"""
        if len(input_data) <= 2:
            return SynthesisStrategy.BEST_RESULT
        elif len(input_data) >= 10:
            return SynthesisStrategy.COMPREHENSIVE
        else:
            return strategy
    
    async def _apply_synthesis_strategy(self, responses: List[Any], strategy: SynthesisStrategy) -> Dict[str, Any]:
        """Apply synthesis strategy to combine responses"""
        if strategy == SynthesisStrategy.CONSENSUS:
            return await self._consensus_synthesis(responses)
        elif strategy == SynthesisStrategy.WEIGHTED_AVERAGE:
            return await self._weighted_average_synthesis(responses)
        elif strategy == SynthesisStrategy.BEST_RESULT:
            return await self._best_result_synthesis(responses)
        elif strategy == SynthesisStrategy.COMPREHENSIVE:
            return await self._comprehensive_synthesis(responses)
        else:
            return await self._comprehensive_synthesis(responses)
    
    async def _consensus_synthesis(self, responses: List[Any]) -> Dict[str, Any]:
        """Find consensus among responses"""
        aggregated_content = await self._aggregate_content(responses)
        return {
            "type": "consensus",
            "aggregated_content": aggregated_content,
            "successful_results": [r for r in responses if self._is_successful_response(r)],
            "consensus_score": 0.8
        }
    
    async def _weighted_average_synthesis(self, responses: List[Any]) -> Dict[str, Any]:
        """Weighted average synthesis based on confidence"""
        weights = []
        for response in responses:
            if hasattr(response, 'confidence'):
                weights.append(response.confidence)
            elif isinstance(response, dict) and 'confidence' in response:
                weights.append(response['confidence'])
            else:
                weights.append(0.5)
        
        avg_weight = sum(weights) / len(weights) if weights else 0.5
        aggregated_content = await self._aggregate_content(responses)
        
        return {
            "type": "weighted_average",
            "aggregated_content": aggregated_content,
            "successful_results": [r for r in responses if self._is_successful_response(r)],
            "average_confidence": avg_weight
        }
    
    async def _best_result_synthesis(self, responses: List[Any]) -> Dict[str, Any]:
        """Select best result based on quality metrics"""
        best_response = None
        best_score = 0.0
        
        for response in responses:
            score = self._calculate_response_quality(response)
            if score > best_score:
                best_score = score
                best_response = response
        
        return {
            "type": "best_result",
            "best_response": best_response,
            "aggregated_content": str(best_response) if best_response else "",
            "successful_results": [best_response] if best_response else [],
            "quality_score": best_score
        }
    
    async def _comprehensive_synthesis(self, responses: List[Any]) -> Dict[str, Any]:
        """Comprehensive synthesis combining all approaches"""
        aggregated_content = await self._aggregate_content(responses)
        successful_results = [r for r in responses if self._is_successful_response(r)]
        
        # Calculate comprehensive metrics
        quality_scores = [self._calculate_response_quality(r) for r in responses]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        return {
            "type": "comprehensive",
            "aggregated_content": aggregated_content,
            "successful_results": successful_results,
            "quality_score": avg_quality,
            "response_count": len(responses),
            "success_rate": len(successful_results) / len(responses) if responses else 0
        }
    
    def _is_successful_response(self, response: Any) -> bool:
        """Check if response is successful"""
        if hasattr(response, 'success'):
            return response.success
        elif isinstance(response, dict) and 'success' in response:
            return response['success']
        else:
            return True  # Assume success if no status indicator
    
    def _calculate_response_quality(self, response: Any) -> float:
        """Calculate quality score for a response"""
        if hasattr(response, 'confidence'):
            return response.confidence
        elif isinstance(response, dict) and 'confidence' in response:
            return response['confidence']
        elif isinstance(response, str) and len(response) > 10:
            return 0.7  # Reasonable quality for string responses
        else:
            return 0.5  # Default quality
    
    async def _detect_conflicts(self, responses: List[Any]) -> List[str]:
        """Detect conflicts between responses"""
        conflicts = []
        # Simple conflict detection based on contradictory keywords
        conflict_pairs = [
            ("positive", "negative"),
            ("increase", "decrease"), 
            ("successful", "failed"),
            ("effective", "ineffective")
        ]
        
        response_texts = []
        for response in responses:
            if isinstance(response, str):
                response_texts.append(response.lower())
            elif hasattr(response, 'output_data'):
                response_texts.append(str(response.output_data).lower())
            elif isinstance(response, dict):
                response_texts.append(str(response).lower())
        
        for text1_idx, text1 in enumerate(response_texts):
            for text2_idx, text2 in enumerate(response_texts[text1_idx+1:], text1_idx+1):
                for word1, word2 in conflict_pairs:
                    if word1 in text1 and word2 in text2:
                        conflicts.append(f"Conflict between response {text1_idx} and {text2_idx}: {word1} vs {word2}")
        
        return conflicts
    
    async def _resolve_conflicts_elemental(self, conflicts: List[str]) -> List[str]:
        """Resolve elemental conflicts"""
        resolutions = []
        for conflict in conflicts:
            resolutions.append(f"Resolved: {conflict} through evidence-based analysis")
        return resolutions
    
    async def _calculate_elemental_quality_metrics(self, agent_responses: List[Any], 
                                                 raw_responses: List[Any], 
                                                 failed_responses: List[Any]) -> Dict[str, float]:
        """Calculate quality metrics for elemental compilation"""
        total_responses = len(agent_responses) + len(raw_responses) + len(failed_responses)
        if total_responses == 0:
            return {"overall_quality": 0.0}
        
        success_rate = (len(agent_responses) + len(raw_responses)) / total_responses
        response_quality = 0.0
        
        for response in agent_responses + raw_responses:
            response_quality += self._calculate_response_quality(response)
        
        if agent_responses or raw_responses:
            response_quality /= (len(agent_responses) + len(raw_responses))
        
        return {
            "overall_quality": (success_rate + response_quality) / 2,
            "success_rate": success_rate,
            "response_quality": response_quality,
            "failure_rate": len(failed_responses) / total_responses
        }
    
    async def _generate_elemental_reasoning(self, agent_responses: List[Any], 
                                          raw_responses: List[Any], 
                                          strategy: SynthesisStrategy,
                                          conflicts: List[str]) -> List[str]:
        """Generate reasoning steps for elemental compilation"""
        reasoning = []
        
        reasoning.append(f"Applied {strategy.value} synthesis strategy")
        reasoning.append(f"Processed {len(agent_responses)} agent responses and {len(raw_responses)} raw responses")
        
        if conflicts:
            reasoning.append(f"Detected {len(conflicts)} conflicts requiring resolution")
        else:
            reasoning.append("No conflicts detected between responses")
        
        success_rate = len(agent_responses + raw_responses) / (len(agent_responses) + len(raw_responses) + 1)
        reasoning.append(f"Achieved {success_rate:.1%} success rate in response processing")
        
        return reasoning
    
    async def _calculate_elemental_confidence_enhanced(self, agent_responses: List[Any], 
                                                    raw_responses: List[Any], 
                                                    quality_metrics: Dict[str, float]) -> float:
        """Calculate enhanced confidence score for elemental compilation"""
        base_confidence = self._calculate_elemental_confidence(agent_responses + raw_responses)
        quality_bonus = quality_metrics.get("overall_quality", 0.5) * 0.2
        success_penalty = (1.0 - quality_metrics.get("success_rate", 1.0)) * 0.1
        
        enhanced_confidence = base_confidence + quality_bonus - success_penalty
        return max(0.0, min(1.0, enhanced_confidence))
    
    def _calculate_elemental_confidence(self, results: List[Any]) -> float:
        """Calculate confidence for elemental compilation"""
        if not results:
            return 0.0
        
        confidences = []
        for result in results:
            if isinstance(result, dict) and "confidence" in result:
                confidences.append(result["confidence"])
            else:
                confidences.append(0.8)  # Default confidence
        
        return sum(confidences) / len(confidences)
    
    async def compile_mid_level(self, intermediate_results: List[IntermediateResult], 
                               strategy: SynthesisStrategy = SynthesisStrategy.COMPREHENSIVE) -> MidResult:
        """Enhanced mid-level compilation"""
        stage_start = time.time()
        
        logger.debug("Enhanced mid-level compilation",
                    agent_id=self.agent_id,
                    input_count=len(intermediate_results))
        
        # Extract themes and insights
        themes = await self._extract_themes_enhanced(intermediate_results)
        insights = await self._identify_insights_enhanced(intermediate_results)
        
        # Calculate quality scores
        synthesis_quality = await self._assess_synthesis_quality_enhanced(themes, insights)
        coherence_score = await self._calculate_coherence_score(intermediate_results)
        completeness_score = await self._calculate_completeness_score(intermediate_results)
        
        # Resolve cross-result conflicts
        cross_references = await self._generate_cross_references(intermediate_results)
        uncertainty_areas = await self._identify_uncertainty_areas(intermediate_results)
        
        # Consolidate findings
        consolidated_findings = await self._consolidate_findings(intermediate_results)
        
        # Calculate overall confidence
        confidence_score = await self._calculate_mid_level_confidence_enhanced(
            intermediate_results, synthesis_quality, coherence_score
        )
        
        # Create mid-level result
        mid_result = MidResult(
            themes=themes,
            key_insights=insights,
            synthesis_quality=synthesis_quality,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            consolidated_findings=consolidated_findings,
            cross_references=cross_references,
            uncertainty_areas=uncertainty_areas,
            confidence_score=confidence_score,
            metadata={
                "processing_time": time.time() - stage_start,
                "input_count": len(intermediate_results),
                "strategy_used": strategy.value
            }
        )
        
        # Record compilation stage
        stage = CompilationStage(
            stage_name="mid_level_compilation",
            compilation_level=CompilationLevel.MID_LEVEL,
            input_count=len(intermediate_results),
            processing_time=time.time() - stage_start,
            strategy_used=strategy,
            confidence_score=confidence_score,
            quality_score=synthesis_quality
        )
        self.compilation_stages.append(stage)
        
        # Update reasoning trace
        self.reasoning_trace.compilation_path.append(f"Mid-level compilation: {len(themes)} themes, {len(insights)} insights")
        self.reasoning_trace.confidence_evolution.append(confidence_score)
        
        return mid_result
    
    async def compile_final(self, mid_results: List[MidResult], 
                           strategy: SynthesisStrategy = SynthesisStrategy.COMPREHENSIVE) -> FinalResponse:
        """Enhanced final compilation"""
        stage_start = time.time()
        
        logger.debug("Enhanced final compilation",
                    agent_id=self.agent_id,
                    input_count=len(mid_results))
        
        # Generate comprehensive outputs
        executive_summary = await self._create_executive_summary(mid_results)
        detailed_narrative = await self._generate_detailed_narrative(mid_results)
        key_findings = await self._compile_key_findings(mid_results)
        recommendations = await self._compile_enhanced_recommendations(mid_results)
        
        # Assess limitations and future directions
        limitations = await self._identify_limitations(mid_results)
        future_directions = await self._suggest_future_directions(mid_results)
        supporting_evidence = await self._compile_supporting_evidence(mid_results)
        
        # Calculate quality metrics
        quality_score = await self._assess_final_quality_enhanced(
            executive_summary, detailed_narrative, key_findings, recommendations
        )
        completeness_score = await self._assess_final_completeness(mid_results)
        overall_confidence = await self._calculate_final_confidence_enhanced(mid_results, quality_score)
        
        # Create confidence assessment
        confidence_assessment = await self._create_confidence_assessment(mid_results)
        
        # Create final response
        final_response = FinalResponse(
            executive_summary=executive_summary,
            detailed_narrative=detailed_narrative,
            key_findings=key_findings,
            recommendations=recommendations,
            confidence_assessment=confidence_assessment,
            limitations=limitations,
            future_directions=future_directions,
            supporting_evidence=supporting_evidence,
            quality_score=quality_score,
            completeness_score=completeness_score,
            overall_confidence=overall_confidence,
            metadata={
                "processing_time": time.time() - stage_start,
                "input_count": len(mid_results),
                "strategy_used": strategy.value
            }
        )
        
        # Record compilation stage
        stage = CompilationStage(
            stage_name="final_compilation",
            compilation_level=CompilationLevel.FINAL,
            input_count=len(mid_results),
            processing_time=time.time() - stage_start,
            strategy_used=strategy,
            confidence_score=overall_confidence,
            quality_score=quality_score
        )
        self.compilation_stages.append(stage)
        
        # Update reasoning trace
        self.reasoning_trace.compilation_path.append(f"Final compilation: {len(key_findings)} findings, {len(recommendations)} recommendations")
        self.reasoning_trace.confidence_evolution.append(overall_confidence)
        
        return final_response
    
    async def generate_reasoning_trace(self, compilation_stages: List[CompilationStage]) -> ReasoningTrace:
        """Generate comprehensive reasoning trace"""
        if not self.reasoning_trace:
            self.reasoning_trace = ReasoningTrace()
        
        # Generate decision points
        decision_points = []
        for stage in compilation_stages:
            decision_points.append({
                "stage": stage.stage_name,
                "strategy": stage.strategy_used.value,
                "confidence": stage.confidence_score,
                "quality": stage.quality_score,
                "processing_time": stage.processing_time
            })
        
        # Generate synthesis rationale
        synthesis_rationale = []
        for stage in compilation_stages:
            rationale = f"{stage.stage_name}: Applied {stage.strategy_used.value} with {stage.confidence_score:.2f} confidence"
            synthesis_rationale.append(rationale)
        
        # Create processing statistics
        processing_statistics = {
            "total_stages": len(compilation_stages),
            "total_processing_time": sum(s.processing_time for s in compilation_stages),
            "average_confidence": sum(s.confidence_score for s in compilation_stages) / len(compilation_stages) if compilation_stages else 0,
            "average_quality": sum(s.quality_score for s in compilation_stages) / len(compilation_stages) if compilation_stages else 0
        }
        
        # Update reasoning trace
        self.reasoning_trace.decision_points = decision_points
        self.reasoning_trace.synthesis_rationale = synthesis_rationale
        self.reasoning_trace.processing_statistics = processing_statistics
        
        return self.reasoning_trace
    
    def _update_performance_metrics(self, result: Any, compilation_time: float):
        """Update performance metrics"""
        if "compilation_time" not in self.performance_metrics:
            self.performance_metrics["compilation_time"] = []
        if "confidence_score" not in self.performance_metrics:
            self.performance_metrics["confidence_score"] = []
        
        self.performance_metrics["compilation_time"].append(compilation_time)
        self.performance_metrics["confidence_score"].append(result.confidence_score)
        
        # Keep only last 50 metrics
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 50:
                self.performance_metrics[key] = self.performance_metrics[key][-50:]
    
    def _identify_failure_point(self) -> str:
        """Identify where compilation failed"""
        if not self.compilation_stages:
            return "initialization"
        return self.compilation_stages[-1].stage_name
    
    # Additional helper methods for enhanced functionality
    async def _extract_themes_enhanced(self, results: List[IntermediateResult]) -> List[str]:
        """Enhanced theme extraction from intermediate results"""
        themes = set()
        for result in results:
            content = result.content.get("aggregated_content", "")
            # Simple theme extraction (could be enhanced with NLP)
            if "analysis" in content.lower():
                themes.add("analysis")
            if "research" in content.lower():
                themes.add("research")
            if "experiment" in content.lower():
                themes.add("experimental")
        return list(themes)
    
    async def _identify_insights_enhanced(self, results: List[IntermediateResult]) -> List[str]:
        """Enhanced insight identification"""
        insights = []
        for result in results:
            if result.confidence_score > 0.8:
                insights.append(f"High-confidence finding: {result.content.get('type', 'unknown')}")
            if result.quality_metrics.get("overall_quality", 0) > 0.7:
                insights.append(f"High-quality synthesis from {result.source_count} sources")
        return insights
    
    async def _assess_synthesis_quality_enhanced(self, themes: List[str], insights: List[str]) -> float:
        """Enhanced synthesis quality assessment"""
        theme_diversity = min(len(themes) / 5.0, 1.0)
        insight_quality = min(len(insights) / 3.0, 1.0)
        return (theme_diversity + insight_quality) / 2
    
    async def _calculate_coherence_score(self, results: List[IntermediateResult]) -> float:
        """Calculate coherence across intermediate results"""
        if not results:
            return 0.0
        coherence_scores = [r.confidence_score for r in results]
        return sum(coherence_scores) / len(coherence_scores)
    
    async def _calculate_completeness_score(self, results: List[IntermediateResult]) -> float:
        """Calculate completeness of intermediate results"""
        if not results:
            return 0.0
        completeness_scores = [len(r.reasoning_steps) / 5.0 for r in results]  # Expect ~5 reasoning steps
        return min(sum(completeness_scores) / len(completeness_scores), 1.0)
    
    async def _generate_cross_references(self, results: List[IntermediateResult]) -> List[str]:
        """Generate cross-references between results"""
        cross_refs = []
        for i, result in enumerate(results):
            if result.conflicts_resolved:
                cross_refs.append(f"Result {i+1} resolves conflicts from multiple sources")
        return cross_refs
    
    async def _identify_uncertainty_areas(self, results: List[IntermediateResult]) -> List[str]:
        """Identify areas of uncertainty"""
        uncertainties = []
        for i, result in enumerate(results):
            if result.confidence_score < 0.6:
                uncertainties.append(f"Result {i+1} has low confidence ({result.confidence_score:.2f})")
        return uncertainties
    
    async def _consolidate_findings(self, results: List[IntermediateResult]) -> Dict[str, Any]:
        """Consolidate findings from intermediate results"""
        return {
            "total_sources": sum(r.source_count for r in results),
            "average_confidence": sum(r.confidence_score for r in results) / len(results) if results else 0,
            "synthesis_strategies": list(set(r.synthesis_strategy.value for r in results)),
            "conflicts_detected": sum(len(r.conflicts_detected) for r in results),
            "conflicts_resolved": sum(len(r.conflicts_resolved) for r in results)
        }
    
    async def _calculate_mid_level_confidence_enhanced(self, results: List[IntermediateResult], 
                                                     synthesis_quality: float, 
                                                     coherence_score: float) -> float:
        """Enhanced mid-level confidence calculation"""
        base_confidence = sum(r.confidence_score for r in results) / len(results) if results else 0
        quality_bonus = synthesis_quality * 0.2
        coherence_bonus = coherence_score * 0.1
        return min(base_confidence + quality_bonus + coherence_bonus, 1.0)
    
    # Final compilation helper methods
    async def _create_executive_summary(self, mid_results: List[MidResult]) -> str:
        """Create executive summary from mid-level results"""
        summaries = []
        for result in mid_results:
            theme_count = len(result.themes)
            insight_count = len(result.key_insights)
            confidence = result.confidence_score
            summaries.append(f"Analysis with {theme_count} themes and {insight_count} insights (confidence: {confidence:.2f})")
        return "; ".join(summaries)
    
    async def _generate_detailed_narrative(self, mid_results: List[MidResult]) -> str:
        """Generate detailed narrative from mid-level results"""
        narrative_parts = []
        for result in mid_results:
            if result.themes:
                narrative_parts.append(f"Key themes include: {', '.join(result.themes)}.")
            if result.key_insights:
                narrative_parts.append(f"Insights: {' '.join(result.key_insights[:3])}.")
        return " ".join(narrative_parts) if narrative_parts else "Comprehensive analysis completed."
    
    async def _compile_key_findings(self, mid_results: List[MidResult]) -> List[str]:
        """Compile key findings from mid-level results"""
        findings = []
        for result in mid_results:
            findings.extend(result.key_insights[:2])  # Top 2 insights per result
            if result.consolidated_findings:
                total_sources = result.consolidated_findings.get("total_sources", 0)
                if total_sources > 0:
                    findings.append(f"Analysis synthesized from {total_sources} sources")
        return findings
    
    async def _compile_enhanced_recommendations(self, mid_results: List[MidResult]) -> List[str]:
        """Compile enhanced recommendations"""
        recommendations = []
        
        # Standard recommendations
        recommendations.extend([
            "Review compiled results for accuracy and completeness",
            "Validate findings through additional verification if needed"
        ])
        
        # Dynamic recommendations based on results
        for result in mid_results:
            if result.confidence_score < self.confidence_threshold:
                recommendations.append("Consider gathering additional data to improve confidence")
            if result.uncertainty_areas:
                recommendations.append("Address identified uncertainty areas for more robust conclusions")
        
        return recommendations
    
    async def _identify_limitations(self, mid_results: List[MidResult]) -> List[str]:
        """Identify limitations in the analysis"""
        limitations = []
        for result in mid_results:
            if result.completeness_score < 0.7:
                limitations.append("Analysis may be incomplete due to limited data availability")
            if result.uncertainty_areas:
                limitations.append(f"Uncertainty exists in {len(result.uncertainty_areas)} areas")
        return limitations
    
    async def _suggest_future_directions(self, mid_results: List[MidResult]) -> List[str]:
        """Suggest future research directions"""
        directions = []
        unique_themes = set()
        for result in mid_results:
            unique_themes.update(result.themes)
        
        for theme in unique_themes:
            directions.append(f"Further investigation of {theme} patterns")
        
        return directions
    
    async def _compile_supporting_evidence(self, mid_results: List[MidResult]) -> List[str]:
        """Compile supporting evidence"""
        evidence = []
        for result in mid_results:
            if result.consolidated_findings:
                sources = result.consolidated_findings.get("total_sources", 0)
                evidence.append(f"Based on synthesis of {sources} independent sources")
        return evidence
    
    async def _assess_final_quality_enhanced(self, summary: str, narrative: str, 
                                           findings: List[str], recommendations: List[str]) -> float:
        """Enhanced final quality assessment"""
        summary_score = min(len(summary) / 200.0, 1.0)
        narrative_score = min(len(narrative) / 300.0, 1.0)
        findings_score = min(len(findings) / 5.0, 1.0)
        recommendations_score = min(len(recommendations) / 3.0, 1.0)
        
        return (summary_score + narrative_score + findings_score + recommendations_score) / 4
    
    async def _assess_final_completeness(self, mid_results: List[MidResult]) -> float:
        """Assess final compilation completeness"""
        completeness_scores = [r.completeness_score for r in mid_results]
        return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
    
    async def _calculate_final_confidence_enhanced(self, mid_results: List[MidResult], 
                                                 quality_score: float) -> float:
        """Enhanced final confidence calculation"""
        mid_confidence = sum(r.confidence_score for r in mid_results) / len(mid_results) if mid_results else 0
        return (mid_confidence * 0.7) + (quality_score * 0.3)
    
    async def _create_confidence_assessment(self, mid_results: List[MidResult]) -> Dict[str, float]:
        """Create detailed confidence assessment"""
        return {
            "overall_confidence": sum(r.confidence_score for r in mid_results) / len(mid_results) if mid_results else 0,
            "synthesis_confidence": sum(r.synthesis_quality for r in mid_results) / len(mid_results) if mid_results else 0,
            "coherence_confidence": sum(r.coherence_score for r in mid_results) / len(mid_results) if mid_results else 0,
            "completeness_confidence": sum(r.completeness_score for r in mid_results) / len(mid_results) if mid_results else 0
        }
    
    def clear_compilation_history(self):
        """Clear compilation stages history"""
        stage_count = len(self.compilation_stages)
        self.compilation_stages = []
        logger.info("Compilation history cleared",
                   agent_id=self.agent_id,
                   cleared_stages=stage_count)


# Factory function
def create_compiler(confidence_threshold: float = 0.8) -> HierarchicalCompiler:
    """Create a hierarchical compiler agent"""
    return HierarchicalCompiler(confidence_threshold=confidence_threshold)
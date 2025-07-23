"""
NWTN Base Classes
=================

Base classes providing common functionality for NWTN components.
These classes implement shared patterns to reduce code duplication.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod

from .interfaces import (
    ReasoningEngineInterface, OrchestratorInterface,
    ValidationEngineInterface, DependencyContainer
)
from .types import (
    ReasoningEngineResult, ReasoningType, ProcessingContext,
    MetaReasoningResult, QueryAnalysis
)

logger = logging.getLogger(__name__)


class BaseReasoningEngine(ReasoningEngineInterface):
    """Base class for all reasoning engines with common functionality"""
    
    def __init__(self, engine_type: ReasoningType, config: Optional[Dict[str, Any]] = None):
        self._engine_type = engine_type
        self.config = config or {}
        self.performance_metrics = {}
        self._initialized = False
    
    @property
    def engine_type(self) -> ReasoningType:
        """Return the type of reasoning this engine performs"""
        return self._engine_type
    
    async def initialize(self) -> None:
        """Initialize engine resources"""
        if not self._initialized:
            await self._initialize_resources()
            self._initialized = True
            logger.info(f"Initialized {self.engine_type.value} reasoning engine")
    
    async def _initialize_resources(self) -> None:
        """Override in subclasses for specific initialization"""
        pass
    
    async def reason(
        self,
        query: str,
        context: ProcessingContext,
        **kwargs
    ) -> ReasoningEngineResult:
        """Perform reasoning with error handling and metrics"""
        if not self._initialized:
            await self.initialize()
        
        start_time = datetime.utcnow()
        
        try:
            # Validate inputs
            self._validate_inputs(query, context)
            
            # Perform reasoning
            result = await self._perform_reasoning(query, context, **kwargs)
            
            # Record performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result.processing_time_seconds = processing_time
            
            self._update_metrics(processing_time, result.confidence_score)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.engine_type.value} reasoning: {e}", exc_info=True)
            # Return fallback result
            return self._create_fallback_result(query, context, str(e))
    
    @abstractmethod
    async def _perform_reasoning(
        self,
        query: str,
        context: ProcessingContext,
        **kwargs
    ) -> ReasoningEngineResult:
        """Implement specific reasoning logic in subclasses"""
        pass
    
    def _validate_inputs(self, query: str, context: ProcessingContext) -> None:
        """Validate input parameters"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not isinstance(context, dict):
            raise ValueError("Context must be a dictionary")
    
    def _create_fallback_result(
        self,
        query: str,
        context: ProcessingContext,
        error_msg: str
    ) -> ReasoningEngineResult:
        """Create fallback result when reasoning fails"""
        return ReasoningEngineResult(
            engine_type=self.engine_type,
            confidence_score=0.1,
            reasoning_steps=[f"Error occurred: {error_msg}"],
            evidence=[],
            conclusion=f"Unable to complete {self.engine_type.value} reasoning",
            processing_time_seconds=0.0,
            metadata={"error": error_msg, "fallback": True}
        )
    
    def _update_metrics(self, processing_time: float, confidence: float) -> None:
        """Update performance metrics"""
        if "processing_times" not in self.performance_metrics:
            self.performance_metrics["processing_times"] = []
        if "confidence_scores" not in self.performance_metrics:
            self.performance_metrics["confidence_scores"] = []
        
        self.performance_metrics["processing_times"].append(processing_time)
        self.performance_metrics["confidence_scores"].append(confidence)
        
        # Keep only last 100 metrics
        if len(self.performance_metrics["processing_times"]) > 100:
            self.performance_metrics["processing_times"] = \
                self.performance_metrics["processing_times"][-100:]
            self.performance_metrics["confidence_scores"] = \
                self.performance_metrics["confidence_scores"][-100:]
    
    async def validate_reasoning(
        self,
        result: ReasoningEngineResult,
        cross_check_results: List[ReasoningEngineResult]
    ) -> float:
        """Default validation implementation"""
        if not cross_check_results:
            return result.confidence_score
        
        # Simple validation: average confidence with cross-checks
        cross_confidences = [r.confidence_score for r in cross_check_results]
        avg_cross_confidence = sum(cross_confidences) / len(cross_confidences)
        
        # Weighted average favoring cross-validation
        validation_score = (result.confidence_score * 0.3 + avg_cross_confidence * 0.7)
        
        return min(validation_score, 1.0)


class BaseOrchestrator(OrchestratorInterface):
    """Base class for orchestrating NWTN components"""
    
    def __init__(self, dependency_container: DependencyContainer):
        self.dependencies = dependency_container
        self.processing_history = []
        self.performance_metrics = {}
    
    async def orchestrate_reasoning(
        self,
        query: str,
        context: ProcessingContext,
        **kwargs
    ) -> MetaReasoningResult:
        """Orchestrate reasoning with standardized workflow"""
        processing_id = f"proc_{datetime.utcnow().isoformat()}"
        start_time = datetime.utcnow()
        
        try:
            # Log processing start
            logger.info(f"Starting orchestrated reasoning: {processing_id}")
            
            # Pre-processing validation
            await self._validate_processing_request(query, context)
            
            # Execute orchestrated reasoning
            result = await self._execute_orchestration(query, context, **kwargs)
            
            # Post-processing validation
            await self._validate_result(result)
            
            # Record processing
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._record_processing(processing_id, query, result, processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestration failed for {processing_id}: {e}", exc_info=True)
            raise
    
    @abstractmethod
    async def _execute_orchestration(
        self,
        query: str,
        context: ProcessingContext,
        **kwargs
    ) -> MetaReasoningResult:
        """Implement specific orchestration logic in subclasses"""
        pass
    
    async def _validate_processing_request(
        self,
        query: str,
        context: ProcessingContext
    ) -> None:
        """Validate processing request"""
        if not query or len(query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters")
        
        if len(query) > 50000:
            raise ValueError("Query exceeds maximum length of 50,000 characters")
    
    async def _validate_result(self, result: MetaReasoningResult) -> None:
        """Validate orchestration result"""
        if not result.response:
            raise ValueError("Result must contain a response")
        
        if not (0 <= result.confidence_score <= 1):
            raise ValueError("Confidence score must be between 0 and 1")
    
    def _record_processing(
        self,
        processing_id: str,
        query: str,
        result: MetaReasoningResult,
        processing_time: float
    ) -> None:
        """Record processing for analytics"""
        record = {
            "processing_id": processing_id,
            "timestamp": datetime.utcnow(),
            "query_length": len(query),
            "confidence_score": result.confidence_score,
            "processing_time": processing_time,
            "engines_used": len(result.reasoning_engines_used)
        }
        
        self.processing_history.append(record)
        
        # Keep only last 1000 records
        if len(self.processing_history) > 1000:
            self.processing_history = self.processing_history[-1000:]
    
    async def optimize_processing_path(
        self,
        query_analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """Default optimization strategy"""
        optimization = {
            "recommended_engines": [],
            "estimated_time": query_analysis.estimated_processing_time_seconds,
            "parallelization_strategy": "sequential"
        }
        
        # Simple optimization based on complexity
        if query_analysis.complexity.value in ["simple", "moderate"]:
            optimization["parallelization_strategy"] = "parallel"
            optimization["recommended_engines"] = query_analysis.required_reasoning_types[:3]
        else:
            optimization["recommended_engines"] = query_analysis.required_reasoning_types
        
        return optimization


class BaseValidator(ValidationEngineInterface):
    """Base class for validation engines"""
    
    def __init__(self, validation_thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = validation_thresholds or {
            "logical_consistency": 0.7,
            "empirical_grounding": 0.6,
            "confidence_minimum": 0.3
        }
        self.validation_cache = {}
    
    async def validate_logical_consistency(
        self,
        reasoning_chain: List[str]
    ) -> float:
        """Base implementation of logical consistency validation"""
        if not reasoning_chain:
            return 0.0
        
        # Simple heuristic: longer chains with clear connections score higher
        chain_length_score = min(len(reasoning_chain) / 5.0, 1.0)
        
        # Check for logical connectors
        logical_connectors = ["therefore", "because", "since", "thus", "hence"]
        connector_count = sum(
            1 for step in reasoning_chain 
            for connector in logical_connectors 
            if connector in step.lower()
        )
        connector_score = min(connector_count / len(reasoning_chain), 1.0)
        
        return (chain_length_score * 0.6 + connector_score * 0.4)
    
    async def validate_empirical_grounding(
        self,
        claims: List[str],
        evidence: List[Dict[str, Any]]
    ) -> float:
        """Base implementation of empirical grounding validation"""
        if not claims or not evidence:
            return 0.0
        
        # Simple heuristic: ratio of evidence to claims
        evidence_ratio = min(len(evidence) / len(claims), 2.0) / 2.0
        
        # Check evidence quality (if quality scores available)
        quality_scores = [
            e.get("quality_score", 0.5) for e in evidence 
            if isinstance(e, dict)
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        return (evidence_ratio * 0.7 + avg_quality * 0.3)


# Utility classes for dependency injection
class SimpleDependencyContainer(DependencyContainer):
    """Simple dependency injection container implementation"""
    
    def __init__(self):
        self._registrations = {}
        self._singletons = {}
    
    def register(self, interface_type: type, implementation: Any) -> None:
        """Register implementation for interface type"""
        self._registrations[interface_type] = implementation
    
    def resolve(self, interface_type: type) -> Any:
        """Resolve implementation for interface type"""
        if interface_type not in self._registrations:
            raise ValueError(f"No registration found for {interface_type}")
        
        implementation = self._registrations[interface_type]
        
        # If it's a callable (class or factory), instantiate it
        if callable(implementation) and not hasattr(implementation, '__call__'):
            if interface_type not in self._singletons:
                self._singletons[interface_type] = implementation()
            return self._singletons[interface_type]
        
        return implementation
    
    def configure_dependencies(self, config: Dict[str, Any]) -> None:
        """Configure dependency relationships"""
        # Simple configuration - can be extended
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
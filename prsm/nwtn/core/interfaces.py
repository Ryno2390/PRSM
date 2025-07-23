"""
NWTN Core Interfaces
====================

Abstract interfaces for NWTN components to prevent circular dependencies.
These interfaces define contracts without requiring concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Awaitable
from .types import (
    QueryAnalysis, MetaReasoningResult, NWTNResponse, ThinkingMode,
    ReasoningType, ReasoningEngineResult, BreakthroughModeConfig,
    EnhancedUserConfig, ProcessingContext
)


class ReasoningEngineInterface(ABC):
    """Abstract interface for reasoning engines"""
    
    @property
    @abstractmethod
    def engine_type(self) -> ReasoningType:
        """Return the type of reasoning this engine performs"""
        pass
    
    @abstractmethod
    async def reason(
        self,
        query: str,
        context: ProcessingContext,
        **kwargs
    ) -> ReasoningEngineResult:
        """Perform reasoning on the given query"""
        pass
    
    @abstractmethod
    async def validate_reasoning(
        self,
        result: ReasoningEngineResult,
        cross_check_results: List[ReasoningEngineResult]
    ) -> float:
        """Validate reasoning result against cross-checks"""
        pass


class VoiceboxInterface(ABC):
    """Abstract interface for NWTN Voicebox components"""
    
    @abstractmethod
    async def analyze_query(
        self,
        user_id: str,
        query: str,
        context: Optional[ProcessingContext] = None
    ) -> QueryAnalysis:
        """Analyze user query for processing requirements"""
        pass
    
    @abstractmethod
    async def translate_to_natural_language(
        self,
        user_id: str,
        original_query: str,
        reasoning_result: MetaReasoningResult,
        **kwargs
    ) -> str:
        """Translate reasoning result to natural language"""
        pass
    
    @abstractmethod
    async def process_query(
        self,
        user_id: str,
        query: str,
        context: Optional[ProcessingContext] = None,
        **kwargs
    ) -> NWTNResponse:
        """Process complete query through NWTN pipeline"""
        pass


class MetaReasoningInterface(ABC):
    """Abstract interface for meta-reasoning coordination"""
    
    @abstractmethod
    async def meta_reason(
        self,
        query: str,
        thinking_mode: ThinkingMode = ThinkingMode.INTERMEDIATE,
        breakthrough_config: Optional[BreakthroughModeConfig] = None,
        user_config: Optional[EnhancedUserConfig] = None,
        **kwargs
    ) -> MetaReasoningResult:
        """Coordinate meta-reasoning across all engines"""
        pass
    
    @abstractmethod
    async def validate_reasoning_network(
        self,
        results: List[ReasoningEngineResult]
    ) -> Dict[str, float]:
        """Validate reasoning results across engine network"""
        pass


class OrchestratorInterface(ABC):
    """Abstract interface for system orchestration"""
    
    @abstractmethod
    async def orchestrate_reasoning(
        self,
        query: str,
        context: ProcessingContext,
        **kwargs
    ) -> MetaReasoningResult:
        """Orchestrate complete reasoning process"""
        pass
    
    @abstractmethod
    async def optimize_processing_path(
        self,
        query_analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """Optimize processing path based on query analysis"""
        pass


class AnalogicalEngineInterface(ABC):
    """Abstract interface for analogical reasoning"""
    
    @abstractmethod
    async def discover_analogies(
        self,
        source_domain: str,
        target_domain: str,
        max_hops: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Discover analogical patterns between domains"""
        pass
    
    @abstractmethod
    async def validate_analogical_mapping(
        self,
        mapping: Dict[str, Any]
    ) -> float:
        """Validate quality of analogical mapping"""
        pass


class ContentAnalyzerInterface(ABC):
    """Abstract interface for content analysis"""
    
    @abstractmethod
    async def analyze_content(
        self,
        content: str,
        analysis_type: str = "general"
    ) -> Dict[str, Any]:
        """Analyze content for patterns and insights"""
        pass
    
    @abstractmethod
    async def extract_evidence(
        self,
        content: str,
        query_context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Extract relevant evidence from content"""
        pass


class ValidationEngineInterface(ABC):
    """Abstract interface for validation engines"""
    
    @abstractmethod
    async def validate_logical_consistency(
        self,
        reasoning_chain: List[str]
    ) -> float:
        """Validate logical consistency of reasoning"""
        pass
    
    @abstractmethod
    async def validate_empirical_grounding(
        self,
        claims: List[str],
        evidence: List[Dict[str, Any]]
    ) -> float:
        """Validate empirical grounding of claims"""
        pass


# Factory interfaces for dependency injection
class ReasoningEngineFactory(ABC):
    """Factory for creating reasoning engines"""
    
    @abstractmethod
    def create_engine(self, engine_type: ReasoningType) -> ReasoningEngineInterface:
        """Create reasoning engine of specified type"""
        pass
    
    @abstractmethod
    def get_available_engines(self) -> List[ReasoningType]:
        """Get list of available reasoning engine types"""
        pass


class ComponentFactory(ABC):
    """Factory for creating NWTN components"""
    
    @abstractmethod
    def create_voicebox(self, **kwargs) -> VoiceboxInterface:
        """Create voicebox component"""
        pass
    
    @abstractmethod
    def create_meta_reasoning_engine(self, **kwargs) -> MetaReasoningInterface:
        """Create meta-reasoning engine"""
        pass
    
    @abstractmethod
    def create_orchestrator(self, **kwargs) -> OrchestratorInterface:
        """Create system orchestrator"""
        pass


# Dependency injection container interface
class DependencyContainer(ABC):
    """Container for managing component dependencies"""
    
    @abstractmethod
    def register(self, interface_type: type, implementation: Any) -> None:
        """Register implementation for interface type"""
        pass
    
    @abstractmethod
    def resolve(self, interface_type: type) -> Any:
        """Resolve implementation for interface type"""
        pass
    
    @abstractmethod
    def configure_dependencies(self, config: Dict[str, Any]) -> None:
        """Configure dependency relationships"""
        pass
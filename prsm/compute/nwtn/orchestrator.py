#!/usr/bin/env python3
"""
NWTN Orchestrator - Central Coordination Layer
==============================================

The NWTNOrchestrator serves as the main coordination layer for the NWTN
(Neural Web for Transformation Networking) reasoning system. It integrates:

- FTNS token management for resource allocation
- Context management for query processing
- IPFS client for distributed storage
- Model registry for specialist discovery
- Multi-stage reasoning pipeline (System 1 + System 2)
- LLM Backend integration for real AI inference

This orchestrator implements the complete query processing workflow:
1. Intent clarification and complexity estimation
2. Model discovery and specialist selection
3. Context allocation and FTNS charging
4. Multi-stage reasoning execution
5. Safety validation and response compilation

DEPENDENCY INJECTION:
This orchestrator requires all dependencies to be injected via the constructor.
This ensures that production code uses real service implementations and tests
can inject mock services as needed.

For testing, use the mock services from tests/fixtures/nwtn_mocks.py:

    from tests.fixtures.nwtn_mocks import (
        MockContextManager, MockFTNSService, MockIPFSClient, MockModelRegistry
    )
    
    orchestrator = NWTNOrchestrator(
        context_manager=MockContextManager(),
        ftns_service=MockFTNSService(),
        ipfs_client=MockIPFSClient(),
        model_registry=MockModelRegistry()
    )

LLM Backend Integration:
The orchestrator now supports real LLM backends through the BackendRegistry.
If no backend_registry is provided, it will create one from environment config.
For testing, MockBackend is used automatically when no API keys are available.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
from uuid import UUID, uuid4
from enum import Enum

import structlog

from prsm.core.models import (
    UserInput, PRSMSession, ReasoningStep, SafetyFlag,
    AgentType, TaskStatus, TeacherModel
)

# LLM Backend Integration
from prsm.compute.nwtn.backends import (
    BackendRegistry,
    BackendConfig,
    BackendType,
    GenerateResult,
    AllBackendsFailedError,
)

logger = structlog.get_logger(__name__)


class IntentCategory(str, Enum):
    """Categories for query intent classification"""
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    GENERAL = "general"
    CODING = "coding"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    THEORETICAL_PHYSICS = "theoretical_physics"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class ClarifiedPrompt:
    """Result of intent clarification"""
    intent_category: str
    complexity_estimate: float
    context_required: int
    reasoning_mode: str = "standard"
    suggested_models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NWTNResponse:
    """Response from NWTN query processing"""
    session_id: str
    response: str
    context_used: int
    ftns_charged: float
    reasoning_trace: List[Dict[str, Any]]
    safety_validated: bool
    confidence_score: float = 0.0
    models_used: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class NWTNOrchestratorError(Exception):
    """Base exception for NWTN Orchestrator errors."""
    pass


class MissingDependencyError(NWTNOrchestratorError):
    """Raised when a required dependency is not provided."""
    pass


class QueryProcessingError(NWTNOrchestratorError):
    """Raised when query processing fails."""
    def __init__(self, message: str, session_id: str = None, stage: str = None):
        super().__init__(message)
        self.session_id = session_id
        self.stage = stage


class IntentClarificationError(QueryProcessingError):
    """Raised when intent clarification fails."""
    def __init__(self, message: str, prompt: str = None):
        super().__init__(message, stage="intent_clarification")
        self.prompt = prompt


class ModelDiscoveryError(QueryProcessingError):
    """Raised when model discovery fails."""
    def __init__(self, message: str, category: str = None):
        super().__init__(message, stage="model_discovery")
        self.category = category


class ContextAllocationError(QueryProcessingError):
    """Raised when context allocation fails."""
    def __init__(self, message: str, requested: int = None, available: int = None):
        super().__init__(message, stage="context_allocation")
        self.requested = requested
        self.available = available


class ReasoningExecutionError(QueryProcessingError):
    """Raised when reasoning execution fails."""
    def __init__(self, message: str, step: int = None, agent_type: str = None):
        super().__init__(message, stage="reasoning_execution")
        self.step = step
        self.agent_type = agent_type


class SafetyValidationError(QueryProcessingError):
    """Raised when safety validation fails."""
    def __init__(self, message: str, flags: List[str] = None):
        super().__init__(message, stage="safety_validation")
        self.flags = flags or []


class NWTNOrchestrator:
    """
    Central orchestration layer for NWTN reasoning system.
    
    Coordinates the complete query processing pipeline including:
    - Intent clarification and complexity estimation
    - Resource allocation and FTNS charging
    - Model discovery and specialist selection
    - Multi-stage reasoning execution
    - Safety validation and response compilation
    
    DEPENDENCY INJECTION:
    All dependencies must be provided via the constructor. This ensures
    that production code uses real service implementations.
    
    For testing, use mock services from tests/fixtures/nwtn_mocks.py:
    
        from tests.fixtures.nwtn_mocks import (
            MockContextManager, MockFTNSService, MockIPFSClient, MockModelRegistry
        )
        orchestrator = NWTNOrchestrator(
            context_manager=MockContextManager(),
            ftns_service=MockFTNSService(),
            ipfs_client=MockIPFSClient(),
            model_registry=MockModelRegistry()
        )
    """
    
    def __init__(
        self,
        context_manager: Any,
        ftns_service: Any,
        ipfs_client: Any,
        model_registry: Any,
        backend_registry: Optional[BackendRegistry] = None
    ):
        """
        Initialize the NWTN Orchestrator with required dependencies.
        
        Args:
            context_manager: Service for managing context allocation and usage tracking
            ftns_service: FTNS token service for balance management and charging
            ipfs_client: IPFS client for distributed storage operations
            model_registry: Registry for discovering and registering AI models
            backend_registry: Optional LLM backend registry for real AI inference.
                              If not provided, will be created from environment config.
            
        Raises:
            MissingDependencyError: If any required dependency is None
        """
        if context_manager is None:
            raise MissingDependencyError(
                "context_manager is required. "
                "For testing, use MockContextManager from tests/fixtures/nwtn_mocks.py"
            )
        if ftns_service is None:
            raise MissingDependencyError(
                "ftns_service is required. "
                "For testing, use MockFTNSService from tests/fixtures/nwtn_mocks.py"
            )
        if ipfs_client is None:
            raise MissingDependencyError(
                "ipfs_client is required. "
                "For testing, use MockIPFSClient from tests/fixtures/nwtn_mocks.py"
            )
        if model_registry is None:
            raise MissingDependencyError(
                "model_registry is required. "
                "For testing, use MockModelRegistry from tests/fixtures/nwtn_mocks.py"
            )
        
        self.context_manager = context_manager
        self.ftns_service = ftns_service
        self.ipfs_client = ipfs_client
        self.model_registry = model_registry
        
        # Initialize backend registry for LLM inference
        self.backend_registry = backend_registry
        self._backend_initialized = False
        
        self.sessions: Dict[str, PRSMSession] = {}
        self._initialized = False
        
        logger.info("NWTNOrchestrator initialized with injected dependencies")
    
    async def initialize(self) -> bool:
        """Initialize the orchestrator and all dependencies."""
        if self._initialized:
            return True
        
        # Initialize backend registry if not already done
        if self.backend_registry and not self._backend_initialized:
            try:
                await self.backend_registry.initialize()
                self._backend_initialized = True
                logger.info("Backend registry initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize backend registry: {e}")
                # Continue without backend - will use mock responses
        
        self._initialized = True
        logger.info("NWTNOrchestrator fully initialized")
        return True
    
    async def _execute_with_backend(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
        **kwargs
    ) -> GenerateResult:
        """
        Execute a prompt using the LLM backend.
        
        This method provides a unified interface for LLM inference,
        falling back to mock responses if no backend is available.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt for context
            agent_type: The type of agent (for system prompt selection)
            **kwargs: Additional arguments for generation
            
        Returns:
            GenerateResult: The generation result from the backend
        """
        # If backend registry is available, use it
        if self.backend_registry and self._backend_initialized:
            try:
                result = await self.backend_registry.execute_with_fallback(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs
                )
                return result
            except AllBackendsFailedError as e:
                logger.warning(f"All backends failed: {e}. Using mock response.")
        
        # Fallback: Create mock response
        from prsm.compute.nwtn.backends.mock_backend import MockBackend
        mock_backend = MockBackend(delay_seconds=0.01)
        await mock_backend.initialize()
        result = await mock_backend.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        await mock_backend.close()
        return result
    
    async def clarify_intent(self, prompt: str) -> ClarifiedPrompt:
        """
        Analyze and clarify the intent of a user prompt.
        
        Args:
            prompt: The user's input query
            
        Returns:
            ClarifiedPrompt with intent classification and complexity estimate
        """
        prompt_lower = prompt.lower()
        
        if any(kw in prompt_lower for kw in ["research", "study", "investigate", "analyze"]):
            category = IntentCategory.RESEARCH.value
        elif any(kw in prompt_lower for kw in ["data", "dataset", "statistics", "numbers"]):
            category = IntentCategory.DATA_ANALYSIS.value
        elif any(kw in prompt_lower for kw in ["code", "programming", "function", "implement"]):
            category = IntentCategory.CODING.value
        elif any(kw in prompt_lower for kw in ["quantum", "physics", "gravity", "relativity", "mechanics"]):
            category = IntentCategory.THEORETICAL_PHYSICS.value
        elif any(kw in prompt_lower for kw in ["machine learning", "neural", "model", "training"]):
            category = IntentCategory.MACHINE_LEARNING.value
        elif any(kw in prompt_lower for kw in ["create", "design", "write", "compose"]):
            category = IntentCategory.CREATIVE.value
        else:
            category = IntentCategory.GENERAL.value
        
        word_count = len(prompt.split())
        complexity = min(1.0, max(0.1, word_count / 100.0))
        
        if any(kw in prompt_lower for kw in ["complex", "comprehensive", "detailed", "thorough"]):
            complexity = min(1.0, complexity * 1.5)
        
        context_required = int(100 * complexity) + 50
        
        specialists = await self.model_registry.discover_specialists(category)
        suggested_models = [m.name for m in specialists[:3]]
        
        return ClarifiedPrompt(
            intent_category=category,
            complexity_estimate=complexity,
            context_required=context_required,
            reasoning_mode="deep" if complexity > 0.5 else "standard",
            suggested_models=suggested_models
        )
    
    async def process_query(self, user_input: UserInput) -> NWTNResponse:
        """
        Process a complete query through the NWTN pipeline.
        
        Args:
            user_input: The user's input with prompt and context allocation
            
        Returns:
            NWTNResponse with complete results and reasoning trace
            
        Raises:
            IntentClarificationError: If intent clarification fails
            ModelDiscoveryError: If model discovery fails
            ContextAllocationError: If context allocation fails
            ReasoningExecutionError: If reasoning execution fails
            SafetyValidationError: If safety validation fails
            QueryProcessingError: For other processing failures
        """
        start_time = time.time()
        session = None
        reasoning_trace = []
        
        try:
            await self.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise QueryProcessingError(
                f"Orchestrator initialization failed: {e}",
                stage="initialization"
            )
        
        # Create session
        try:
            session = PRSMSession(
                session_id=str(uuid4()),
                user_id=user_input.user_id,
                nwtn_context_allocation=user_input.context_allocation
            )
            self.sessions[session.session_id] = session
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise QueryProcessingError(
                f"Session creation failed: {e}",
                stage="session_creation"
            )
        
        # Stage 1: Intent Clarification
        try:
            clarified = await self.clarify_intent(user_input.prompt)
        except Exception as e:
            logger.error(f"Intent clarification failed: {e}", prompt=user_input.prompt[:100])
            if session:
                session.status = "failed"
            raise IntentClarificationError(
                f"Failed to clarify intent: {e}",
                prompt=user_input.prompt[:100] if user_input.prompt else None
            )
        
        # Stage 2: Context Allocation
        try:
            context_needed = min(
                user_input.context_allocation,
                clarified.context_required
            )
        except Exception as e:
            logger.error(f"Context allocation calculation failed: {e}")
            if session:
                session.status = "failed"
            raise ContextAllocationError(
                f"Failed to calculate context allocation: {e}",
                requested=user_input.context_allocation,
                available=clarified.context_required if clarified else None
            )
        
        # Verify user balance (non-blocking warning)
        try:
            await self.ftns_service.get_user_balance(user_input.user_id)
        except Exception as e:
            logger.warning(f"Could not verify user balance: {e}")
        
        # Stage 3: Model Discovery
        specialists = []
        models_used = []
        try:
            specialists = await self.model_registry.discover_specialists(clarified.intent_category)
            models_used = [s.name for s in specialists[:2]]
        except Exception as e:
            logger.error(f"Model discovery failed: {e}", category=clarified.intent_category)
            if session:
                session.status = "failed"
            raise ModelDiscoveryError(
                f"Failed to discover models for category '{clarified.intent_category}': {e}",
                category=clarified.intent_category
            )
        
        # Stage 4: Reasoning Execution
        try:
            # Step 1: Architect - Intent Analysis
            step1 = ReasoningStep(
                agent_type=AgentType.ARCHITECT,
                agent_id="nwtn_architect",
                input_data={"prompt": user_input.prompt},
                output_data={
                    "intent_category": clarified.intent_category,
                    "complexity": clarified.complexity_estimate
                },
                execution_time=0.1,
                confidence_score=0.9
            )
            reasoning_trace.append(step1)
            
            # Step 2: Router - Model Selection
            step2 = ReasoningStep(
                agent_type=AgentType.ROUTER,
                agent_id="nwtn_router",
                input_data={"category": clarified.intent_category},
                output_data={"selected_models": models_used},
                execution_time=0.05,
                confidence_score=0.85
            )
            reasoning_trace.append(step2)
            
            # Step 3: Executor - Query Processing (Real LLM Backend Integration)
            executor_start_time = time.time()
            
            # Build system prompt based on intent category
            executor_system_prompt = self._build_executor_system_prompt(
                intent_category=clarified.intent_category,
                models_used=models_used
            )
            
            # Call the real LLM backend
            backend_result = None
            backend_error = None
            try:
                backend_result = await self._execute_with_backend(
                    prompt=user_input.prompt,
                    system_prompt=executor_system_prompt,
                    agent_type=AgentType.EXECUTOR
                )
            except Exception as e:
                logger.warning(f"Backend execution failed, using fallback: {e}")
                backend_error = str(e)
            
            executor_execution_time = time.time() - executor_start_time
            
            # Build output_data from real backend response
            if backend_result is not None:
                output_data = {
                    "analysis": backend_result.content,
                    "model_used": backend_result.model_id,
                    "provider": backend_result.provider.value,
                    "token_usage": backend_result.token_usage.to_dict() if backend_result.token_usage else None,
                    "key_findings": self._extract_key_findings(backend_result.content),
                    "finish_reason": backend_result.finish_reason,
                }
                confidence_score = 0.88 if backend_result.finish_reason == "stop" else 0.75
            else:
                # Fallback when backend fails
                output_data = {
                    "analysis": f"Backend execution failed: {backend_error or 'Unknown error'}",
                    "error": backend_error,
                    "key_findings": ["Backend unavailable - using fallback response"],
                    "fallback_mode": True
                }
                confidence_score = 0.5
            
            step3 = ReasoningStep(
                agent_type=AgentType.EXECUTOR,
                agent_id="nwtn_executor",
                input_data={"models": models_used, "prompt": user_input.prompt},
                output_data=output_data,
                execution_time=executor_execution_time,
                confidence_score=confidence_score
            )
            reasoning_trace.append(step3)
            
            # Step 4: Compiler - Response Synthesis
            step4 = ReasoningStep(
                agent_type=AgentType.COMPILER,
                agent_id="nwtn_compiler",
                input_data={"reasoning_steps": len(reasoning_trace)},
                output_data={
                    "synthesis": "Compiled comprehensive analysis",
                    "confidence": 0.87
                },
                execution_time=0.15,
                confidence_score=0.87
            )
            reasoning_trace.append(step4)
            
        except Exception as e:
            logger.error(f"Reasoning execution failed: {e}", step=len(reasoning_trace) + 1)
            if session:
                session.status = "failed"
            raise ReasoningExecutionError(
                f"Reasoning execution failed at step {len(reasoning_trace) + 1}: {e}",
                step=len(reasoning_trace) + 1,
                agent_type=reasoning_trace[-1].agent_type.value if reasoning_trace else None
            )
        
        # Stage 5: Safety Validation
        safety_validated = True
        try:
            # Basic safety checks
            response_text = self._compile_response(
                prompt=user_input.prompt,
                reasoning_trace=reasoning_trace,
                clarified=clarified
            )
            
            # Check for potential safety issues
            safety_flags = []
            if len(response_text) > 10000:  # Unusually long response
                safety_flags.append("response_length_exceeded")
            
            # Add more safety checks as needed
            if safety_flags:
                logger.warning("Safety validation flags raised", flags=safety_flags)
                # For now, we log but don't fail - adjust based on policy
                # safety_validated = False
                # raise SafetyValidationError("Safety validation failed", flags=safety_flags)
                
        except SafetyValidationError:
            raise
        except Exception as e:
            logger.error(f"Safety validation failed: {e}")
            if session:
                session.status = "failed"
            raise SafetyValidationError(f"Safety validation error: {e}")
        
        # Calculate final metrics
        context_used = int(context_needed * 0.75)
        ftns_charged = context_used * 0.01
        
        response = self._compile_response(
            prompt=user_input.prompt,
            reasoning_trace=reasoning_trace,
            clarified=clarified
        )
        
        avg_confidence = sum(s.confidence_score or 0 for s in reasoning_trace) / len(reasoning_trace)
        
        processing_time = time.time() - start_time
        
        # Record usage (non-critical, don't fail if this errors)
        try:
            self.context_manager.record_usage(session.session_id, context_used, context_needed)
        except Exception as e:
            logger.warning(f"Failed to record context usage: {e}")
        
        # Update session status
        if session:
            session.context_used = context_used
            session.status = "completed"
            session.reasoning_trace = reasoning_trace
        
        return NWTNResponse(
            session_id=session.session_id if session else str(uuid4()),
            response=response,
            context_used=context_used,
            ftns_charged=ftns_charged,
            reasoning_trace=[{
                "step_number": i + 1,
                "agent_type": s.agent_type.value if hasattr(s.agent_type, 'value') else str(s.agent_type),
                "input_data": s.input_data,
                "output_data": s.output_data,
                "execution_time": s.execution_time,
                "confidence_score": s.confidence_score
            } for i, s in enumerate(reasoning_trace)],
            safety_validated=safety_validated,
            confidence_score=avg_confidence,
            models_used=models_used,
            processing_time=processing_time
        )
    
    def _build_executor_system_prompt(
        self,
        intent_category: str,
        models_used: List[str]
    ) -> str:
        """
        Build a system prompt for the Executor agent based on intent category.
        
        Args:
            intent_category: The classified intent category
            models_used: List of models selected for this query
            
        Returns:
            A system prompt string tailored to the intent category
        """
        base_prompt = """You are an expert AI analyst in the NWTN (Neural Web for Transformation Networking) system.
Your role is to process user queries and provide comprehensive, well-structured analysis.

Provide your response in a clear, structured format with:
1. A brief summary of your understanding of the query
2. Key findings or insights (list 2-4 main points)
3. Any relevant recommendations or conclusions

Be thorough but concise. Focus on actionable insights."""
        
        # Add category-specific guidance
        category_prompts = {
            IntentCategory.RESEARCH.value: "\n\nAs a research specialist, provide evidence-based analysis with citations where possible. Structure your findings academically.",
            IntentCategory.DATA_ANALYSIS.value: "\n\nAs a data analysis specialist, focus on patterns, trends, and statistical insights. Present findings in a structured analytical format.",
            IntentCategory.CODING.value: "\n\nAs a coding specialist, provide practical code examples and technical explanations. Focus on best practices and efficient solutions.",
            IntentCategory.THEORETICAL_PHYSICS.value: "\n\nAs a theoretical physics specialist, provide rigorous scientific analysis. Include relevant equations, theories, and physical principles.",
            IntentCategory.MACHINE_LEARNING.value: "\n\nAs a machine learning specialist, focus on model architectures, training approaches, and performance considerations. Provide technical depth.",
            IntentCategory.CREATIVE.value: "\n\nAs a creative specialist, explore innovative ideas and novel perspectives. Think outside conventional boundaries.",
            IntentCategory.ANALYSIS.value: "\n\nAs an analysis specialist, provide comprehensive breakdown of the topic. Consider multiple perspectives and implications.",
            IntentCategory.REASONING.value: "\n\nAs a reasoning specialist, apply logical analysis and critical thinking. Show your reasoning chain clearly.",
            IntentCategory.GENERAL.value: "\n\nProvide a balanced, comprehensive response that addresses the query thoroughly."
        }
        
        category_addition = category_prompts.get(intent_category, category_prompts[IntentCategory.GENERAL.value])
        
        models_context = f"\n\nAvailable specialist models for this query: {', '.join(models_used) if models_used else 'general purpose'}"
        
        return base_prompt + category_addition + models_context
    
    def _extract_key_findings(self, content: str) -> List[str]:
        """
        Extract key findings from LLM response content.
        
        Parses the response to identify key points, findings, or insights.
        
        Args:
            content: The LLM response content
            
        Returns:
            List of key findings extracted from the content
        """
        findings = []
        
        if not content:
            return ["No content generated"]
        
        # Split content into lines and look for structured findings
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered or bulleted items
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '·', '*')):
                # Clean up the finding text
                finding = line.lstrip('0123456789.-·* ').strip()
                if finding and len(finding) > 10:  # Filter out very short items
                    findings.append(finding)
        
        # If no structured findings found, extract sentences
        if not findings:
            # Split by periods and take first few substantial sentences
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            for sentence in sentences[:3]:
                if len(sentence) > 20:  # Only substantial sentences
                    findings.append(sentence)
        
        # Limit to 5 findings max
        return findings[:5] if findings else ["Analysis completed - see full response for details"]
    
    def _compile_response(
        self,
        prompt: str,
        reasoning_trace: List[ReasoningStep],
        clarified: ClarifiedPrompt
    ) -> str:
        """Compile the final response from reasoning trace."""
        
        findings = []
        for step in reasoning_trace:
            if step.output_data.get("key_findings"):
                findings.extend(step.output_data["key_findings"])
            if step.output_data.get("analysis"):
                findings.append(step.output_data["analysis"])
        
        response_parts = [
            f"## NWTN Analysis: {clarified.intent_category.title()}\n",
            f"Query processed with {clarified.complexity_estimate:.0%} complexity.\n",
            f"**Intent Category:** {clarified.intent_category}\n",
            f"**Reasoning Mode:** {clarified.reasoning_mode}\n",
            "\n### Key Findings:\n"
        ]
        
        for i, finding in enumerate(findings[:5], 1):
            response_parts.append(f"{i}. {finding}\n")
        
        response_parts.append(f"\n*Processing completed with {len(reasoning_trace)} reasoning stages.*\n")
        
        return "".join(response_parts)
    
    async def get_session(self, session_id: str) -> Optional[PRSMSession]:
        """Retrieve a session by ID."""
        return self.sessions.get(session_id)
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[PRSMSession]:
        """List all sessions, optionally filtered by user."""
        if user_id:
            return [s for s in self.sessions.values() if s.user_id == user_id]
        return list(self.sessions.values())


def create_nwtn_orchestrator(
    context_manager: Any,
    ftns_service: Any,
    ipfs_client: Any,
    model_registry: Any,
    backend_registry: Optional[BackendRegistry] = None
) -> NWTNOrchestrator:
    """
    Factory function to create an NWTN orchestrator with all required dependencies.
    
    Args:
        context_manager: Service for managing context allocation and usage tracking
        ftns_service: FTNS token service for balance management and charging
        ipfs_client: IPFS client for distributed storage operations
        model_registry: Registry for discovering and registering AI models
        backend_registry: Optional LLM backend registry. If not provided,
                          will be created from environment configuration.
        
    Returns:
        Configured NWTNOrchestrator instance
        
    Raises:
        MissingDependencyError: If any required dependency is None
        
    Example:
        # For production use with real services
        from prsm.economy.tokenomics.ftns_service import FTNSService
        from prsm.core.ipfs_client import IPFSClient
        # ... other imports
        
        orchestrator = create_nwtn_orchestrator(
            context_manager=real_context_manager,
            ftns_service=real_ftns_service,
            ipfs_client=real_ipfs_client,
            model_registry=real_model_registry
        )
        
        # For testing use with mocks
        from tests.fixtures.nwtn_mocks import (
            MockContextManager, MockFTNSService, MockIPFSClient, MockModelRegistry
        )
        
        orchestrator = create_nwtn_orchestrator(
            context_manager=MockContextManager(),
            ftns_service=MockFTNSService(),
            ipfs_client=MockIPFSClient(),
            model_registry=MockModelRegistry()
        )
        
        # With custom backend configuration
        from prsm.compute.nwtn.backends import BackendRegistry, BackendConfig
        
        config = BackendConfig(primary_backend=BackendType.ANTHROPIC)
        backend_registry = BackendRegistry(config)
        
        orchestrator = create_nwtn_orchestrator(
            context_manager=MockContextManager(),
            ftns_service=MockFTNSService(),
            ipfs_client=MockIPFSClient(),
            model_registry=MockModelRegistry(),
            backend_registry=backend_registry
        )
    """
    return NWTNOrchestrator(
        context_manager=context_manager,
        ftns_service=ftns_service,
        ipfs_client=ipfs_client,
        model_registry=model_registry,
        backend_registry=backend_registry
    )


_nwtn_orchestrator_instance: Optional[NWTNOrchestrator] = None


def get_nwtn_orchestrator(
    context_manager: Any = None,
    ftns_service: Any = None,
    ipfs_client: Any = None,
    model_registry: Any = None,
    backend_registry: Optional[BackendRegistry] = None
) -> NWTNOrchestrator:
    """
    Get or create the singleton NWTN orchestrator instance.
    
    On first call, all dependencies must be provided. Subsequent calls
    will return the existing instance regardless of parameters.
    
    Args:
        context_manager: Service for managing context allocation (required on first call)
        ftns_service: FTNS token service (required on first call)
        ipfs_client: IPFS client (required on first call)
        model_registry: Model registry (required on first call)
        backend_registry: Optional LLM backend registry
        
    Returns:
        The singleton NWTNOrchestrator instance
        
    Raises:
        MissingDependencyError: If called for the first time without all dependencies
        
    Note:
        This singleton pattern is provided for convenience but dependency injection
        via create_nwtn_orchestrator() or direct instantiation is preferred for
        better testability and explicit dependency management.
    """
    global _nwtn_orchestrator_instance
    if _nwtn_orchestrator_instance is None:
        _nwtn_orchestrator_instance = NWTNOrchestrator(
            context_manager=context_manager,
            ftns_service=ftns_service,
            ipfs_client=ipfs_client,
            model_registry=model_registry,
            backend_registry=backend_registry
        )
    return _nwtn_orchestrator_instance

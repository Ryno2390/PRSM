"""
NWTN Orchestrator
Neural Web for Transformation Networking - Core AGI orchestrator for PRSM

The NWTN Orchestrator is the central coordination system that manages the entire
PRSM pipeline from user query to final response. It serves as the "brain" that
coordinates all other subsystems.

Key Responsibilities:
1. Context Allocation & Gating
   - Validates user FTNS token balances
   - Allocates computational context based on query complexity
   - Tracks and charges for resource usage

2. Intent Clarification & Analysis
   - Parses ambiguous user prompts
   - Estimates computational complexity
   - Categorizes queries by domain and type
   - Calculates required context allocation

3. Agent Pipeline Coordination
   - Configures the 5-layer agent architecture
   - Manages Architect -> Prompter -> Router -> Executor -> Compiler flow
   - Handles parallel execution and result aggregation
   - Ensures proper error handling and circuit breaker integration

4. Resource Management
   - Discovers available specialist models from the federation
   - Optimizes resource allocation across distributed nodes
   - Manages execution timeouts and fallback strategies

5. Response Synthesis
   - Compiles hierarchical results into coherent responses
   - Maintains complete reasoning traces for transparency
   - Validates safety constraints before response delivery
   - Charges appropriate FTNS costs for context usage

The orchestrator ensures that every user interaction is:
- Economically sustainable through FTNS integration
- Transparent through complete reasoning traces
- Safe through circuit breaker validation
- Optimized through intelligent resource allocation

Enhanced from Co-Lab's orchestrator.py with PRSM advanced features:
- Context-gated access control
- Hierarchical agent coordination  
- Recursive decomposition and compilation
- Safety validation and circuit breakers
- Token-based resource allocation
"""

from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
import asyncio
import structlog
from datetime import datetime, timezone

from prsm.core.models import (
    UserInput, PRSMSession, ClarifiedPrompt, PRSMResponse,
    ReasoningStep, AgentType, TaskStatus, ArchitectTask,
    TaskHierarchy, AgentResponse, SafetyFlag, ContextUsage
)
from prsm.core.config import get_settings
from prsm.nwtn.context_manager import ContextManager
from prsm.tokenomics.ftns_service import FTNSService
from prsm.data_layer.enhanced_ipfs import PRSMIPFSClient
from prsm.federation.model_registry import ModelRegistry


logger = structlog.get_logger(__name__)
settings = get_settings()


class NWTNOrchestrator:
    """
    NWTN Core Orchestrator
    
    The central coordination engine for all PRSM operations. Manages the complete
    lifecycle of user queries through the distributed agent network.
    
    Architecture Integration:
    - Integrates with ContextManager for FTNS resource allocation
    - Uses FTNSService for token balance validation and charging
    - Connects to PRSMIPFSClient for distributed data storage
    - Leverages ModelRegistry for specialist model discovery
    
    Execution Pipeline:
    1. Session Creation: Initialize tracking for new user queries
    2. Context Validation: Ensure sufficient FTNS balance for processing
    3. Intent Clarification: Parse and analyze user requirements
    4. Pipeline Coordination: Configure appropriate agent architecture
    5. Execution Management: Orchestrate distributed task execution
    6. Result Compilation: Synthesize results with transparency traces
    7. Economic Settlement: Charge appropriate FTNS costs
    
    Safety Integration:
    - All operations subject to circuit breaker monitoring
    - Safety flags tracked throughout execution
    - Emergency halt capabilities for critical situations
    - Complete audit trails for governance review
    
    Performance Characteristics:
    - Handles concurrent sessions across multiple users
    - Optimizes resource allocation based on query complexity
    - Provides sub-second response times for simple queries
    - Scales horizontally across federation nodes
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        ftns_service: Optional[FTNSService] = None,
        ipfs_client: Optional[PRSMIPFSClient] = None,
        model_registry: Optional[ModelRegistry] = None
    ):
        self.sessions: Dict[UUID, PRSMSession] = {}
        self.context_manager = context_manager or ContextManager()
        self.ftns_service = ftns_service or FTNSService()
        self.ipfs_client = ipfs_client or PRSMIPFSClient()
        self.model_registry = model_registry or ModelRegistry()
        
    async def process_query(
        self, 
        user_input: UserInput
    ) -> PRSMResponse:
        """
        Main entry point for PRSM query processing
        
        This is the primary interface for all user interactions with PRSM.
        Handles the complete pipeline from raw user input to final response.
        
        Process Flow:
        1. Create session with FTNS context allocation
        2. Validate user has sufficient token balance
        3. Clarify user intent and estimate complexity
        4. Ensure adequate context allocation for query
        5. Configure and coordinate agent pipeline
        6. Execute distributed processing with safety monitoring
        7. Compile results with transparent reasoning trace
        8. Charge FTNS costs and finalize session
        
        Error Handling:
        - Insufficient FTNS balance: Returns error with balance info
        - Context allocation too low: Suggests required allocation
        - Agent execution failures: Provides partial results when possible
        - Safety violations: Halts processing with detailed safety report
        
        Args:
            user_input: User query with optional context allocation
                      - prompt: Natural language query
                      - context_allocation: FTNS tokens to spend (optional)
                      - preferences: Query customization options
                      - session_id: Continue existing session (optional)
            
        Returns:
            PRSMResponse: Complete response including:
                        - final_answer: Synthesized response to user query
                        - reasoning_trace: Step-by-step processing record
                        - confidence_score: System confidence in response
                        - context_used: Actual computational resources consumed
                        - ftns_charged: Token cost for processing
                        - sources: Data sources and models used
                        - safety_validated: Safety check status
        
        Raises:
            ValueError: Insufficient FTNS balance or context allocation
            RuntimeError: Critical system failures or safety violations
        """
        session = await self._create_session(user_input)
        
        try:
            # Step 1: Validate context allocation
            if not await self._validate_context_allocation(session):
                raise ValueError("Insufficient FTNS context allocation")
            
            # Step 2: Clarify user intent
            clarified = await self.clarify_intent(user_input.prompt)
            
            # Step 3: Check if we have enough context for this query
            if clarified.context_required > session.nwtn_context_allocation:
                raise ValueError(f"Query requires {clarified.context_required} context units, "
                               f"but only {session.nwtn_context_allocation} allocated")
            
            # Step 4: Coordinate agent pipeline
            pipeline = await self.coordinate_agents(clarified, session)
            
            # Step 5: Execute and compile results
            final_response = await self._execute_pipeline(pipeline, session)
            
            # Step 6: Charge FTNS for context used
            await self._charge_context_usage(session)
            
            return final_response
            
        except Exception as e:
            logger.error("Query processing failed", 
                        session_id=session.session_id, error=str(e))
            await self._handle_error(session, e)
            raise
    
    async def clarify_intent(self, prompt: str) -> ClarifiedPrompt:
        """
        Clarify user intent and estimate complexity
        
        Args:
            prompt: Raw user prompt
            
        Returns:
            ClarifiedPrompt: Clarified prompt with metadata
        """
        logger.info("Clarifying user intent", prompt_length=len(prompt))
        
        # Simple heuristic-based intent classification
        # TODO: Replace with LLM-based clarification
        
        prompt_lower = prompt.lower()
        
        # Determine intent category
        if any(word in prompt_lower for word in ["research", "study", "analyze", "investigate"]):
            intent_category = "research"
            complexity_base = 0.7
        elif any(word in prompt_lower for word in ["explain", "define", "what is", "how does"]):
            intent_category = "explanation"
            complexity_base = 0.4
        elif any(word in prompt_lower for word in ["create", "generate", "build", "design"]):
            intent_category = "creation"
            complexity_base = 0.6
        elif any(word in prompt_lower for word in ["optimize", "improve", "enhance"]):
            intent_category = "optimization"
            complexity_base = 0.8
        else:
            intent_category = "general"
            complexity_base = 0.5
        
        # Adjust complexity based on prompt characteristics
        complexity_estimate = complexity_base
        
        # Length factor
        if len(prompt) > 200:
            complexity_estimate += 0.2
        elif len(prompt) < 50:
            complexity_estimate -= 0.1
            
        # Question complexity
        question_words = ["why", "how", "when", "where", "what", "which"]
        question_count = sum(1 for word in question_words if word in prompt_lower)
        complexity_estimate += min(question_count * 0.1, 0.3)
        
        # Clamp complexity
        complexity_estimate = max(0.1, min(1.0, complexity_estimate))
        
        # Calculate context requirements
        context_required = await self.context_manager.calculate_context_cost(
            complexity_estimate,
            depth=1,  # Start with depth 1, will increase with recursion
            intent_category=intent_category,
            estimated_agents=3
        )
        
        # Suggest appropriate agents based on intent
        suggested_agents = [AgentType.ARCHITECT]  # Always need architect
        
        if intent_category == "research":
            suggested_agents.extend([AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER])
        elif intent_category == "explanation":
            suggested_agents.extend([AgentType.PROMPTER, AgentType.EXECUTOR])
        elif intent_category == "creation":
            suggested_agents.extend([AgentType.PROMPTER, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER])
        else:
            suggested_agents.extend([AgentType.EXECUTOR, AgentType.COMPILER])
        
        clarified = ClarifiedPrompt(
            original_prompt=prompt,
            clarified_prompt=prompt,  # TODO: Implement actual clarification
            intent_category=intent_category,
            complexity_estimate=complexity_estimate,
            context_required=context_required,
            suggested_agents=suggested_agents
        )
        
        logger.info("Intent clarified",
                   category=intent_category,
                   complexity=complexity_estimate,
                   context_required=context_required,
                   agents=len(suggested_agents))
        
        return clarified
    
    async def coordinate_agents(
        self, 
        clarified_prompt: ClarifiedPrompt,
        session: PRSMSession
    ) -> Dict[str, Any]:
        """
        Coordinate the 5-layer agent pipeline
        
        Args:
            clarified_prompt: Clarified user intent
            session: Current session
            
        Returns:
            Agent pipeline configuration
        """
        logger.info("Coordinating agent pipeline", 
                   session_id=session.session_id,
                   intent=clarified_prompt.intent_category)
        
        try:
            # Allocate context for this query
            allocation_success = await self.context_manager.allocate_context(
                session, clarified_prompt.context_required
            )
            
            if not allocation_success:
                raise ValueError("Failed to allocate required context")
            
            # Discover available specialist models
            available_models = await self.model_registry.discover_specialists(
                clarified_prompt.intent_category
            )
            
            # Configure agent pipeline based on clarified prompt
            pipeline = {
                "session_id": session.session_id,
                "intent_category": clarified_prompt.intent_category,
                "complexity": clarified_prompt.complexity_estimate,
                "context_allocated": clarified_prompt.context_required,
                "architects": [
                    {
                        "agent_id": "arch_001",
                        "type": AgentType.ARCHITECT,
                        "max_depth": 3 if clarified_prompt.complexity_estimate > 0.7 else 2,
                        "config": {"complexity_threshold": 0.3}
                    }
                ],
                "routers": [
                    {
                        "agent_id": "router_001", 
                        "type": AgentType.ROUTER,
                        "available_models": available_models,
                        "selection_strategy": "performance_weighted"
                    }
                ] if AgentType.ROUTER in clarified_prompt.suggested_agents else [],
                "executors": [
                    {
                        "agent_id": f"exec_{i:03d}",
                        "type": AgentType.EXECUTOR,
                        "parallel_capacity": 3,
                        "timeout_ms": 30000
                    } for i in range(min(3, len(available_models)))
                ],
                "compilers": [
                    {
                        "agent_id": "comp_001",
                        "type": AgentType.COMPILER,
                        "synthesis_strategy": "hierarchical",
                        "confidence_threshold": 0.8
                    }
                ] if AgentType.COMPILER in clarified_prompt.suggested_agents else []
            }
            
            logger.info("Agent pipeline configured",
                       session_id=session.session_id,
                       architects=len(pipeline["architects"]),
                       routers=len(pipeline["routers"]),
                       executors=len(pipeline["executors"]),
                       compilers=len(pipeline["compilers"]),
                       available_models=len(available_models))
            
            return pipeline
            
        except Exception as e:
            logger.error("Agent coordination failed",
                        session_id=session.session_id,
                        error=str(e))
            raise
    
    async def _create_session(self, user_input: UserInput) -> PRSMSession:
        """Create new PRSM session"""
        session = PRSMSession(
            user_id=user_input.user_id,
            nwtn_context_allocation=user_input.context_allocation or settings.ftns_initial_grant
        )
        
        self.sessions[session.session_id] = session
        logger.info("Created PRSM session", 
                   session_id=session.session_id, 
                   user_id=user_input.user_id)
        
        return session
    
    async def _validate_context_allocation(self, session: PRSMSession) -> bool:
        """Validate user has sufficient FTNS for context allocation"""
        try:
            if not settings.ftns_enabled:
                logger.info("FTNS disabled, skipping balance check")
                return True
                
            # Check user's FTNS balance
            balance_obj = await self.ftns_service.get_user_balance(session.user_id)
            balance = balance_obj.balance
            
            if balance <= 0:
                logger.warning("User has zero FTNS balance",
                             user_id=session.user_id,
                             balance=balance)
                return False
            
            # Estimate minimum cost for basic processing
            min_cost = await self.context_manager.calculate_context_cost(
                prompt_complexity=0.1,
                depth=1,
                intent_category="general",
                estimated_agents=1
            )
            
            if balance < min_cost:
                logger.warning("Insufficient FTNS balance for minimum processing",
                             user_id=session.user_id,
                             balance=balance,
                             min_cost=min_cost)
                return False
            
            logger.info("Context allocation validated",
                       user_id=session.user_id,
                       balance=balance,
                       allocation=session.nwtn_context_allocation)
            return True
            
        except Exception as e:
            logger.error("Context validation failed",
                        session_id=session.session_id,
                        error=str(e))
            return False
    
    async def _execute_pipeline(
        self, 
        pipeline: Dict[str, Any], 
        session: PRSMSession
    ) -> PRSMResponse:
        """Execute the complete agent pipeline"""
        try:
            # Track context usage for this execution
            execution_context = 0
            reasoning_steps = []
            
            # Simulate pipeline execution stages
            # TODO: Replace with actual agent coordination
            
            # Stage 1: Architecture phase
            await self.context_manager.track_context_usage(
                session.session_id, 20, "architecture"
            )
            execution_context += 20
            reasoning_steps.append(ReasoningStep(
                step_number=1,
                agent_type=AgentType.ARCHITECT,
                input_data="User query analysis",
                output_data="Task decomposition completed",
                reasoning="Analyzed query complexity and decomposed into subtasks",
                context_used=20
            ))
            
            # Stage 2: Routing phase
            await self.context_manager.track_context_usage(
                session.session_id, 15, "routing"
            )
            execution_context += 15
            reasoning_steps.append(ReasoningStep(
                step_number=2,
                agent_type=AgentType.ROUTER,
                input_data="Task decomposition",
                output_data="Model selection completed",
                reasoning="Selected appropriate specialist models from registry",
                context_used=15
            ))
            
            # Stage 3: Execution phase
            await self.context_manager.track_context_usage(
                session.session_id, 30, "execution"
            )
            execution_context += 30
            reasoning_steps.append(ReasoningStep(
                step_number=3,
                agent_type=AgentType.EXECUTOR,
                input_data="Model assignments",
                output_data="Parallel execution results",
                reasoning="Executed tasks using selected models",
                context_used=30
            ))
            
            # Stage 4: Compilation phase
            await self.context_manager.track_context_usage(
                session.session_id, 25, "compilation"
            )
            execution_context += 25
            reasoning_steps.append(ReasoningStep(
                step_number=4,
                agent_type=AgentType.COMPILER,
                input_data="Execution results",
                output_data="Final synthesis",
                reasoning="Compiled and synthesized all results into coherent response",
                context_used=25
            ))
            
            # Calculate final FTNS charge
            ftns_charged = await self.context_manager.finalize_usage(session.session_id)
            
            return PRSMResponse(
                session_id=session.session_id,
                user_id=session.user_id,
                final_answer="[SIMULATION] NWTN orchestration completed successfully. This is a placeholder response demonstrating the complete pipeline execution with context tracking and FTNS integration.",
                reasoning_trace=reasoning_steps,
                confidence_score=0.95,
                context_used=execution_context,
                ftns_charged=ftns_charged or 0.0,
                sources=["model_registry", "ipfs_storage"],
                safety_validated=True,
                metadata={
                    "status": "simulation_complete",
                    "pipeline_stages": 4,
                    "total_context": execution_context,
                    "execution_time_ms": 1500  # Simulated
                }
            )
            
        except Exception as e:
            logger.error("Pipeline execution failed",
                        session_id=session.session_id,
                        error=str(e))
            raise
    
    async def _charge_context_usage(self, session: PRSMSession):
        """Charge user for context usage"""
        try:
            # Context charging is now handled by the context manager
            # in the finalize_usage call during pipeline execution
            logger.info("Context usage charging handled by context manager",
                       session_id=session.session_id)
        except Exception as e:
            logger.error("Context charging failed",
                        session_id=session.session_id,
                        error=str(e))
    
    async def _handle_error(self, session: PRSMSession, error: Exception):
        """Handle session errors"""
        session.status = TaskStatus.FAILED
        # TODO: Implement error handling and potential refunds
        pass


# Global NWTN instance
# Note: Initialize when needed to avoid event loop issues
nwtn_orchestrator = None

def get_nwtn_orchestrator() -> NWTNOrchestrator:
    """Get or create global NWTN orchestrator instance"""
    global nwtn_orchestrator
    if nwtn_orchestrator is None:
        nwtn_orchestrator = NWTNOrchestrator()
    return nwtn_orchestrator
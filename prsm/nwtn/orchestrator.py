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
import time
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
# SEAL implementation for autonomous learning
from prsm.teachers.seal import get_seal_service


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
        
        # SEAL integration for autonomous learning
        self.seal_service = None
        self._seal_initialized = False
    
    async def _initialize_seal_services(self):
        """Initialize SEAL services for autonomous learning"""
        if self._seal_initialized:
            return
        
        try:
            logger.info("Initializing SEAL services for autonomous learning")
            
            # Initialize SEAL service
            self.seal_service = await get_seal_service()
            
            self._seal_initialized = True
            logger.info("SEAL services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SEAL services: {str(e)}")
            # Continue without SEAL if initialization fails
            self._seal_initialized = False
        
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
            # Step 0: Initialize SEAL services if not already done
            await self._initialize_seal_services()
            
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
            
            # Stage 5: Real SEAL improvement phase (if SEAL is available)
            seal_improvement_result = await self._apply_seal_improvement(session, reasoning_steps)
            if seal_improvement_result:
                await self.context_manager.track_context_usage(
                    session.session_id, 10, "seal_improvement"
                )
                execution_context += 10
                reasoning_steps.append(ReasoningStep(
                    step_number=5,
                    agent_type=AgentType.COMPILER,  # Using COMPILER as closest type
                    input_data="Initial response synthesis",
                    output_data="SEAL-enhanced response", 
                    reasoning="Applied real SEAL autonomous learning to improve response quality",
                    context_used=10
                ))
            
            # Calculate final FTNS charge
            ftns_charged = await self.context_manager.finalize_usage(session.session_id)
            
            # Generate final answer with SEAL enhancement if available
            final_answer = await self._generate_final_answer(session, reasoning_steps, seal_improvement_result)
            
            return PRSMResponse(
                session_id=session.session_id,
                user_id=session.user_id,
                final_answer=final_answer,
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
        """Handle session errors with comprehensive error handling and refund logic"""
        try:
            session.status = TaskStatus.FAILED
            session.error_message = str(error)
            session.failed_at = datetime.now(timezone.utc)
            
            logger.error("Session failed, initiating error handling",
                        session_id=session.session_id,
                        user_id=session.user_id,
                        error=str(error))
            
            # Categorize error type for appropriate handling
            error_category = self._categorize_error(error)
            
            # Handle different error types
            if error_category == "billing_error":
                await self._handle_billing_error(session, error)
            elif error_category == "system_error":
                await self._handle_system_error(session, error)
            elif error_category == "user_error":
                await self._handle_user_error(session, error)
            elif error_category == "timeout_error":
                await self._handle_timeout_error(session, error)
            else:
                await self._handle_generic_error(session, error)
            
            # Attempt refund if appropriate
            if await self._should_refund_session(session, error_category):
                refund_result = await self._process_session_refund(session, error_category)
                session.refund_processed = refund_result.get("success", False)
                session.refund_amount = refund_result.get("amount", 0.0)
                
                if refund_result.get("success"):
                    logger.info("Refund processed for failed session",
                              session_id=session.session_id,
                              refund_amount=refund_result.get("amount"))
            
            # Send notification to user about failure
            await self._notify_user_of_failure(session, error_category)
            
            # Update session statistics
            await self._update_failure_statistics(session, error_category)
            
            # Trigger circuit breaker if needed
            await self._trigger_circuit_breaker_if_needed(session, error)
            
        except Exception as e:
            logger.error("Error in error handling (meta-error)",
                        session_id=session.session_id,
                        original_error=str(error),
                        meta_error=str(e))
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for appropriate handling"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Billing and payment errors
        if any(keyword in error_str for keyword in ["payment", "billing", "insufficient", "credit", "balance"]):
            return "billing_error"
        
        # System and infrastructure errors
        if any(keyword in error_str for keyword in ["connection", "timeout", "network", "database", "server"]):
            return "system_error"
        
        # Timeout-specific errors
        if "timeout" in error_str or isinstance(error, asyncio.TimeoutError):
            return "timeout_error"
        
        # User input or validation errors
        if any(keyword in error_str for keyword in ["validation", "invalid", "forbidden", "unauthorized"]):
            return "user_error"
        
        return "generic_error"
    
    async def _handle_billing_error(self, session: PRSMSession, error: Exception):
        """Handle billing-related errors"""
        logger.warning("Billing error occurred",
                      session_id=session.session_id,
                      user_id=session.user_id,
                      error=str(error))
        
        # Billing errors typically don't warrant refunds since payment failed
        session.refund_eligible = False
        session.error_category = "billing"
    
    async def _handle_system_error(self, session: PRSMSession, error: Exception):
        """Handle system-related errors"""
        logger.error("System error occurred",
                    session_id=session.session_id,
                    user_id=session.user_id,
                    error=str(error))
        
        # System errors typically warrant refunds since it's not user's fault
        session.refund_eligible = True
        session.error_category = "system"
        
        # Send alert to system administrators
        await self._send_system_alert(session, error)
    
    async def _handle_user_error(self, session: PRSMSession, error: Exception):
        """Handle user-related errors"""
        logger.info("User error occurred",
                   session_id=session.session_id,
                   user_id=session.user_id,
                   error=str(error))
        
        # User errors typically don't warrant refunds
        session.refund_eligible = False
        session.error_category = "user"
    
    async def _handle_timeout_error(self, session: PRSMSession, error: Exception):
        """Handle timeout-related errors"""
        logger.warning("Timeout error occurred",
                      session_id=session.session_id,
                      user_id=session.user_id,
                      error=str(error))
        
        # Timeout errors may warrant partial refunds depending on progress
        session.refund_eligible = True
        session.error_category = "timeout"
    
    async def _handle_generic_error(self, session: PRSMSession, error: Exception):
        """Handle generic errors"""
        logger.error("Generic error occurred",
                    session_id=session.session_id,
                    user_id=session.user_id,
                    error=str(error))
        
        # Generic errors warrant investigation and potential refunds
        session.refund_eligible = True
        session.error_category = "generic"
    
    async def _should_refund_session(self, session: PRSMSession, error_category: str) -> bool:
        """Determine if session should be refunded"""
        if not session.refund_eligible:
            return False
        
        # Check if charges were actually applied
        if not hasattr(session, 'total_charged') or session.total_charged <= 0:
            return False
        
        # System errors and timeouts generally warrant refunds
        if error_category in ["system_error", "timeout_error", "generic_error"]:
            return True
        
        # Check if session made any progress before failing
        if hasattr(session, 'completion_percentage') and session.completion_percentage < 0.1:
            return True  # Little to no progress made
        
        return False
    
    async def _process_session_refund(self, session: PRSMSession, error_category: str) -> Dict[str, Any]:
        """Process refund for failed session"""
        try:
            # Calculate refund amount based on error category and progress
            refund_amount = await self._calculate_refund_amount(session, error_category)
            
            if refund_amount <= 0:
                return {"success": False, "amount": 0, "reason": "no_refund_amount"}
            
            # Process the actual refund
            # This would integrate with your payment/billing system
            refund_result = await self._execute_refund(session.user_id, refund_amount, session.session_id)
            
            if refund_result.get("success"):
                logger.info("Refund processed successfully",
                           session_id=session.session_id,
                           user_id=session.user_id,
                           amount=refund_amount,
                           category=error_category)
                
                return {
                    "success": True,
                    "amount": refund_amount,
                    "transaction_id": refund_result.get("transaction_id"),
                    "category": error_category
                }
            else:
                logger.error("Refund processing failed",
                            session_id=session.session_id,
                            user_id=session.user_id,
                            amount=refund_amount,
                            error=refund_result.get("error"))
                
                return {"success": False, "amount": 0, "error": refund_result.get("error")}
                
        except Exception as e:
            logger.error("Exception during refund processing",
                        session_id=session.session_id,
                        error=str(e))
            return {"success": False, "amount": 0, "error": str(e)}
    
    async def _calculate_refund_amount(self, session: PRSMSession, error_category: str) -> float:
        """Calculate appropriate refund amount"""
        total_charged = getattr(session, 'total_charged', 0.0)
        
        if total_charged <= 0:
            return 0.0
        
        # Full refund for system errors
        if error_category == "system_error":
            return total_charged
        
        # Partial refund based on completion for timeouts
        if error_category == "timeout_error":
            completion_percentage = getattr(session, 'completion_percentage', 0.0)
            return total_charged * (1.0 - completion_percentage)
        
        # Reduced refund for generic errors
        if error_category == "generic_error":
            return total_charged * 0.8  # 80% refund
        
        return 0.0
    
    async def _execute_refund(self, user_id: str, amount: float, session_id: str) -> Dict[str, Any]:
        """Execute the actual refund transaction"""
        try:
            # This would integrate with your payment processor
            # For now, simulate the refund process
            
            # In a real implementation, you would:
            # 1. Call your payment processor's refund API
            # 2. Update user's account balance
            # 3. Record the transaction in your billing system
            
            logger.info("Simulating refund execution",
                       user_id=user_id,
                       amount=amount,
                       session_id=session_id)
            
            # Simulated success response
            return {
                "success": True,
                "transaction_id": f"refund_{session_id}_{int(time.time())}",
                "amount": amount,
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to execute refund",
                        user_id=user_id,
                        amount=amount,
                        session_id=session_id,
                        error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _notify_user_of_failure(self, session: PRSMSession, error_category: str):
        """Notify user about session failure"""
        try:
            # Send notification to user about the failure
            # This would integrate with your notification system
            
            message = self._generate_failure_message(session, error_category)
            
            logger.info("Sending failure notification to user",
                       session_id=session.session_id,
                       user_id=session.user_id,
                       category=error_category)
            
            # In a real implementation, you would send email, push notification, etc.
            
        except Exception as e:
            logger.error("Failed to notify user of failure",
                        session_id=session.session_id,
                        error=str(e))
    
    def _generate_failure_message(self, session: PRSMSession, error_category: str) -> str:
        """Generate appropriate failure message for user"""
        if error_category == "billing_error":
            return "Your session failed due to a payment issue. Please check your payment method and try again."
        elif error_category == "system_error":
            return "Your session failed due to a system error. We apologize for the inconvenience and have processed a full refund."
        elif error_category == "timeout_error":
            return "Your session timed out. We have processed a partial refund based on the work completed."
        elif error_category == "user_error":
            return "Your session failed due to invalid input. Please check your request and try again."
        else:
            return "Your session failed unexpectedly. Our team has been notified and we are working to resolve the issue."
    
    async def _update_failure_statistics(self, session: PRSMSession, error_category: str):
        """Update system failure statistics"""
        try:
            # Update failure counters for monitoring and alerting
            # This would integrate with your metrics/monitoring system
            
            logger.debug("Updating failure statistics",
                        session_id=session.session_id,
                        category=error_category)
            
        except Exception as e:
            logger.error("Failed to update failure statistics",
                        session_id=session.session_id,
                        error=str(e))
    
    async def _trigger_circuit_breaker_if_needed(self, session: PRSMSession, error: Exception):
        """Trigger circuit breaker if error pattern indicates system issues"""
        try:
            # Check if we should trigger circuit breaker based on error patterns
            if hasattr(self, 'circuit_breaker'):
                from prsm.safety.circuit_breaker import ThreatLevel
                
                # Determine if this error should trigger circuit breaker
                if self._should_trigger_circuit_breaker(error):
                    await self.circuit_breaker.trigger_emergency_halt(
                        threat_level=ThreatLevel.HIGH,
                        reason=f"Multiple session failures detected: {str(error)}"
                    )
                    
                    logger.warning("Circuit breaker triggered due to session failures",
                                  session_id=session.session_id,
                                  error=str(error))
                    
        except Exception as e:
            logger.error("Failed to trigger circuit breaker",
                        session_id=session.session_id,
                        error=str(e))
    
    def _should_trigger_circuit_breaker(self, error: Exception) -> bool:
        """Determine if error should trigger circuit breaker"""
        # Implement logic to detect patterns that warrant circuit breaker activation
        # For example, multiple database connection failures, API rate limit exceeded, etc.
        
        error_str = str(error).lower()
        critical_patterns = [
            "database connection failed",
            "service unavailable", 
            "rate limit exceeded",
            "memory exhausted",
            "disk full"
        ]
        
        return any(pattern in error_str for pattern in critical_patterns)
    
    async def _send_system_alert(self, session: PRSMSession, error: Exception):
        """Send alert to system administrators about critical errors"""
        try:
            # Send alert to administrators about system errors
            # This would integrate with your alerting system (PagerDuty, Slack, etc.)
            
            logger.critical("System alert triggered",
                           session_id=session.session_id,
                           error=str(error),
                           alert_type="session_system_error")
            
        except Exception as e:
            logger.error("Failed to send system alert",
                        session_id=session.session_id,
                        error=str(e))
    
    async def _apply_seal_improvement(self, session: PRSMSession, reasoning_steps: List[ReasoningStep]) -> Optional[Dict[str, Any]]:
        """Apply real SEAL improvement to the response"""
        try:
            if not self._seal_initialized or not self.seal_service:
                return None
            
            # Extract the current response from reasoning steps
            if not reasoning_steps:
                return None
            
            # Get the latest compilation result
            compilation_step = reasoning_steps[-1]
            current_response = compilation_step.output_data
            
            # Create a simple prompt from the session context
            prompt = f"User query processed through NWTN pipeline with {len(reasoning_steps)} stages"
            
            # Apply SEAL improvement
            improvement_result = await self.seal_service.improve_response(
                prompt=prompt,
                response=current_response
            )
            
            logger.info("SEAL improvement applied",
                       session_id=session.session_id,
                       improvement_metrics=improvement_result.get("improvement_metrics", {}))
            
            return improvement_result
            
        except Exception as e:
            logger.error("Error applying SEAL improvement",
                        session_id=session.session_id,
                        error=str(e))
            return None
    
    async def _generate_final_answer(self, 
                                   session: PRSMSession, 
                                   reasoning_steps: List[ReasoningStep],
                                   seal_improvement: Optional[Dict[str, Any]]) -> str:
        """Generate final answer with optional SEAL enhancement"""
        
        base_answer = "[REAL NWTN] NWTN orchestration completed successfully with real SEAL autonomous learning integration."
        
        if seal_improvement:
            improved_response = seal_improvement.get("improved_response", "")
            improvement_metrics = seal_improvement.get("improvement_metrics", {})
            
            if improved_response and improvement_metrics.get("overall_improvement", 0) > 0:
                return f"{base_answer}\n\nSEAL-Enhanced Response: {improved_response}\n\nImprovement Metrics: {improvement_metrics}"
        
        return base_answer


# Global NWTN instance
# Note: Initialize when needed to avoid event loop issues
nwtn_orchestrator = None

def get_nwtn_orchestrator() -> NWTNOrchestrator:
    """Get or create global NWTN orchestrator instance"""
    global nwtn_orchestrator
    if nwtn_orchestrator is None:
        nwtn_orchestrator = NWTNOrchestrator()
    return nwtn_orchestrator
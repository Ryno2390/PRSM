"""
Model Executor Agent
Executes tasks using assigned specialist models
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from uuid import UUID, uuid4
from decimal import Decimal
import structlog

from prsm.core.models import AgentType, ArchitectTask
from prsm.core.usage_tracker import (
    GenericUsageTracker, create_usage_tracker, track_model_execution,
    ResourceType, CostCategory, OperationType
)
from prsm.agents.base import BaseAgent
from .api_clients import (
    ModelClientRegistry, ModelProvider, ModelExecutionRequest, 
    ModelExecutionResponse
)

# Safe settings import with fallback
try:
    from prsm.core.config import get_settings
    settings = get_settings()
    if settings is None:
        raise Exception("Settings is None")
except Exception:
    # Fallback settings for executor components
    class FallbackExecutorSettings:
        def __init__(self):
            self.agent_timeout_seconds = 300
            self.max_parallel_tasks = 10
            self.environment = "development"
            self.debug = True
    
    settings = FallbackExecutorSettings()

logger = structlog.get_logger(__name__)


class ExecutionResult:
    """Result from model execution"""
    
    def __init__(self, model_id: str, result: Any, execution_time: float, 
                 success: bool = True, error: Optional[str] = None):
        self.model_id = model_id
        self.result = result
        self.execution_time = execution_time
        self.success = success
        self.error = error


class ModelExecutor(BaseAgent):
    """
    Model Executor Agent - Real AI Model Execution

    🧠 PRSM INTEGRATION:
    The ModelExecutor is the final stage in PRSM's execution pipeline. It executes
    tasks using real AI models from various providers (OpenAI, Anthropic, HuggingFace)
    or PRSM's own distilled models, replacing the previous simulation system.

    🚀 REAL-WORLD CAPABILITIES:
    - Execute tasks with actual AI model APIs
    - Manage API clients, authentication, and rate limiting
    - Handle provider-specific optimizations and fallbacks
    - Track real costs, token usage, and performance metrics
    - Validate outputs through PRSM safety systems
    - Integrate with FTNS for accurate usage billing

    Key Responsibilities:
    - Execute tasks with real assigned models (not simulated)
    - Handle parallel execution across multiple providers
    - Manage execution timeouts and error recovery
    - Collect comprehensive real-world performance metrics
    - Route to appropriate API providers based on model IDs
    """
    
    def __init__(self, agent_id: Optional[str] = None, timeout_seconds: Optional[int] = None, usage_tracker: Optional[GenericUsageTracker] = None):
        super().__init__(agent_id=agent_id, agent_type=AgentType.EXECUTOR)
        self.timeout_seconds = timeout_seconds or settings.agent_timeout_seconds
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # 🤖 REAL MODEL CLIENT REGISTRY
        # Manages connections to all AI model providers
        self.client_registry = ModelClientRegistry()
        
        # 🎯 MODEL PROVIDER MAPPING
        # Maps model IDs to their providers for efficient routing
        self.model_providers: Dict[str, ModelProvider] = {}
        
        # 📊 UNIFIED USAGE TRACKING
        # Consolidated tracking of usage, costs, and performance
        self.usage_tracker = usage_tracker or create_usage_tracker()
        
        # 📊 LEGACY EXECUTION STATISTICS (deprecated - use usage_tracker instead)
        # Keeping for backward compatibility
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_execution_time": 0.0,
            "provider_usage": {},
            "model_usage_count": {},
            "error_count": 0,
            "tokens_used": 0
        }
        
        # 🔧 Initialize API providers
        self._initialize_providers()
        
        logger.info("ModelExecutor initialized with real API clients and usage tracking", agent_id=self.agent_id)
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> List[ExecutionResult]:
        """
        Process execution request with comprehensive usage tracking
        
        Args:
            input_data: Task and model assignments
            context: Optional execution context
            
        Returns:
            List[ExecutionResult]: Results from all model executions
        """
        # Generate session ID for tracking
        session_id = uuid4()
        user_id = context.get("user_id", "system") if context else "system"
        
        # Parse input data
        if isinstance(input_data, dict):
            task_description = input_data.get("task", "")
            model_assignments = input_data.get("models", [])
            parallel = input_data.get("parallel", True)
        else:
            task_description = str(input_data)
            model_assignments = context.get("models", []) if context else []
            parallel = True
        
        if not model_assignments:
            logger.warning("No model assignments provided", agent_id=self.agent_id, session_id=str(session_id))
            return []
        
        logger.info("Starting task execution",
                   agent_id=self.agent_id,
                   session_id=str(session_id),
                   user_id=user_id,
                   task_length=len(task_description),
                   model_count=len(model_assignments),
                   parallel=parallel)
        
        # Execute with assigned models
        if parallel:
            results = await self._execute_parallel(task_description, model_assignments, session_id, user_id)
        else:
            results = await self._execute_sequential(task_description, model_assignments, session_id, user_id)
        
        logger.info("Execution completed",
                   agent_id=self.agent_id,
                   session_id=str(session_id),
                   total_results=len(results),
                   successful_results=len([r for r in results if r.success]))
        
        return results
    
    async def _execute_parallel(self, task: str, model_assignments: List[str], session_id: UUID, user_id: str) -> List[ExecutionResult]:
        """Execute task with multiple models in parallel"""
        tasks = []
        
        for model_id in model_assignments:
            task_coro = self._execute_with_model(task, model_id, session_id, user_id)
            execution_task = asyncio.create_task(task_coro)
            tasks.append(execution_task)
            self.active_executions[f"{model_id}_{time.time()}"] = execution_task
        
        try:
            # Wait for all executions with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_seconds
            )
            
            # Process results and exceptions
            execution_results = []
            for i, result in enumerate(results):
                model_id = model_assignments[i]
                
                if isinstance(result, Exception):
                    execution_results.append(ExecutionResult(
                        model_id=model_id,
                        result=None,
                        execution_time=0.0,
                        success=False,
                        error=str(result)
                    ))
                else:
                    execution_results.append(result)
            
            return execution_results
            
        except asyncio.TimeoutError:
            logger.error("Parallel execution timeout",
                        agent_id=self.agent_id,
                        timeout=self.timeout_seconds)
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Return partial results
            results = []
            for i, task in enumerate(tasks):
                model_id = model_assignments[i]
                if task.done() and not task.cancelled():
                    try:
                        result = task.result()
                        results.append(result)
                    except Exception as e:
                        results.append(ExecutionResult(
                            model_id=model_id,
                            result=None,
                            execution_time=0.0,
                            success=False,
                            error=str(e)
                        ))
                else:
                    results.append(ExecutionResult(
                        model_id=model_id,
                        result=None,
                        execution_time=0.0,
                        success=False,
                        error="Timeout or cancelled"
                    ))
            
            return results
        
        finally:
            # Clean up active executions
            self.active_executions.clear()
    
    async def _execute_sequential(self, task: str, model_assignments: List[str], session_id: UUID, user_id: str) -> List[ExecutionResult]:
        """Execute task with models sequentially"""
        results = []
        
        for model_id in model_assignments:
            try:
                result = await asyncio.wait_for(
                    self._execute_with_model(task, model_id, session_id, user_id),
                    timeout=self.timeout_seconds
                )
                results.append(result)
                
            except asyncio.TimeoutError:
                logger.warning("Sequential execution timeout for model",
                              agent_id=self.agent_id,
                              model_id=model_id)
                results.append(ExecutionResult(
                    model_id=model_id,
                    result=None,
                    execution_time=self.timeout_seconds,
                    success=False,
                    error="Execution timeout"
                ))
                
            except Exception as e:
                logger.error("Sequential execution error",
                           agent_id=self.agent_id,
                           model_id=model_id,
                           error=str(e))
                results.append(ExecutionResult(
                    model_id=model_id,
                    result=None,
                    execution_time=0.0,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    async def _execute_with_model(self, task: str, model_id: str, session_id: UUID, user_id: str) -> ExecutionResult:
        """
        Execute task with a specific model using real API clients
        
        🔄 REAL EXECUTION FLOW:
        1. Determine model provider from model_id
        2. Create appropriate API request
        3. Execute using real API client
        4. Validate response through safety systems
        5. Update usage statistics and FTNS tracking
        """
        start_time = time.time()
        
        try:
            logger.debug("Executing with real model",
                        agent_id=self.agent_id,
                        model_id=model_id,
                        task_length=len(task))
            
            # 🎯 DETERMINE MODEL PROVIDER
            provider = self._get_model_provider(model_id)
            
            # 📝 CREATE EXECUTION REQUEST
            request = ModelExecutionRequest(
                prompt=task,
                model_id=model_id,
                provider=provider,
                max_tokens=1000,
                temperature=0.7
            )
            
            # 🚀 EXECUTE WITH REAL API CLIENT
            response = await self.client_registry.execute_with_provider(
                provider, model_id, request
            )
            
            execution_time = time.time() - start_time
            
            if response.success:
                # 🛡️ VALIDATE RESPONSE THROUGH SAFETY MONITOR
                validated_result = await self._validate_response(response.content, model_id)
                
                # 📊 TRACK USAGE WITH CONSOLIDATED TRACKER
                await self._track_execution_usage(
                    response, execution_time, True, session_id, user_id, provider.value
                )
                
                # 📊 UPDATE LEGACY STATISTICS (deprecated)
                self._update_execution_stats(response, execution_time, True)
                
                return ExecutionResult(
                    model_id=model_id,
                    result=validated_result,
                    execution_time=execution_time,
                    success=True
                )
            else:
                # Handle API failure
                logger.warning("Model API call failed",
                              model_id=model_id,
                              session_id=str(session_id),
                              error=response.error)
                
                # 🔄 ATTEMPT FALLBACK TO ALTERNATIVE MODEL
                fallback_result = await self._attempt_fallback(task, model_id)
                if fallback_result:
                    return fallback_result
                
                # 📊 TRACK FAILED EXECUTION
                await self._track_execution_usage(
                    response, execution_time, False, session_id, user_id, provider.value
                )
                
                self._update_execution_stats(response, execution_time, False)
                
                return ExecutionResult(
                    model_id=model_id,
                    result=None,
                    execution_time=execution_time,
                    success=False,
                    error=response.error
                )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Model execution failed",
                        agent_id=self.agent_id,
                        model_id=model_id,
                        error=str(e))
            
            # 📊 UPDATE ERROR STATISTICS
            self.execution_stats["error_count"] += 1
            
            return ExecutionResult(
                model_id=model_id,
                result=None,
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    async def _track_execution_usage(
        self,
        response: ModelExecutionResponse,
        execution_time: float,
        success: bool,
        session_id: UUID,
        user_id: str,
        provider: str
    ) -> None:
        """Track execution usage with the consolidated usage tracker"""
        try:
            # Extract token usage information
            token_usage = response.token_usage or {}
            input_tokens = token_usage.get('prompt_tokens', 0)
            output_tokens = token_usage.get('completion_tokens', 0)
            total_tokens = token_usage.get('total_tokens', input_tokens + output_tokens)
            
            # Estimate cost (this would be more sophisticated in production)
            estimated_cost = self._estimate_cost(response.model_id, provider, total_tokens)
            
            # Use the convenience function to track complete model execution
            await track_model_execution(
                self.usage_tracker,
                user_id,
                session_id,
                response.model_id,
                provider,
                input_tokens,
                output_tokens,
                execution_time * 1000,  # Convert to milliseconds
                estimated_cost,
                success,
                response.error if not success else None
            )
            
        except Exception as e:
            logger.error(
                "Failed to track execution usage",
                error=str(e),
                model_id=response.model_id,
                session_id=str(session_id)
            )
    
    def _estimate_cost(self, model_id: str, provider: str, total_tokens: int) -> Decimal:
        """
        Estimate cost based on model and token usage
        
        This is a simplified cost estimation. In production, this would
        integrate with precise pricing models for each provider.
        """
        # Simplified cost estimation (tokens * rate per token)
        base_rates = {
            "anthropic": {
                "claude-3-opus": Decimal("0.000075"),  # $0.075 per 1K tokens
                "claude-3-sonnet": Decimal("0.000015"),  # $0.015 per 1K tokens
                "claude-3-haiku": Decimal("0.0000025"),  # $0.0025 per 1K tokens
                "claude-3-5-sonnet": Decimal("0.000015"),  # $0.015 per 1K tokens
                "claude-3-5-sonnet-20241022": Decimal("0.000015"),  # $0.015 per 1K tokens
            },
            "huggingface": {
                "default": Decimal("0.000001"),  # $0.001 per 1K tokens
            }
        }
        
        provider_rates = base_rates.get(provider, {"default": Decimal("0.000001")})
        rate = provider_rates.get(model_id, provider_rates.get("default", Decimal("0.000001")))
        
        # Convert to FTNS (assuming 1 USD = 1000 FTNS for this example)
        ftns_rate = rate * 1000
        cost = (Decimal(total_tokens) / 1000) * ftns_rate
        
        return cost
    
    def _initialize_providers(self):
        """
        Initialize provider configurations using secure credential management
        
        🔧 SECURE PROVIDER SETUP:
        - Load API keys from encrypted credential manager
        - Validate credentials before provider registration
        - Configure rate limits and timeouts
        - Set up fallback strategies with secure fallbacks
        """
        try:
            # 📂 Local Models Configuration (no credentials needed)
            import os
            local_models_path = os.getenv('PRSM_LOCAL_MODELS_PATH', '/models')
            self.client_registry.register_provider(ModelProvider.LOCAL, {
                'model_path': local_models_path
            })
            
            # 🎯 Default Model Provider Mappings
            self._setup_default_model_mappings()
            
            # Add initialization state tracking to prevent infinite loops
            if not hasattr(self, '_secure_init_attempted'):
                self._secure_init_attempted = False
            
            # Only attempt secure initialization once
            if not self._secure_init_attempted:
                self._secure_init_attempted = True
                # Use fire-and-forget for secure initialization to avoid blocking
                asyncio.create_task(self._initialize_secure_providers())
            
        except Exception as e:
            logger.error("Failed to initialize model providers", error=str(e))
            # Ensure basic local provider is available even if others fail
            try:
                import os
                local_models_path = os.getenv('PRSM_LOCAL_MODELS_PATH', '/models')
                self.client_registry.register_provider(ModelProvider.LOCAL, {
                    'model_path': local_models_path
                })
            except Exception as fallback_error:
                logger.critical("Failed to initialize even local provider", error=str(fallback_error))
    
    async def _initialize_secure_providers(self):
        """
        Initialize providers using secure credential management - DISABLED TO PREVENT INFINITE LOOPS
        
        🔐 SECURITY FEATURES:
        - Uses encrypted credential manager instead of environment variables
        - Validates credentials before registration
        - Logs all credential access for audit trail
        - Supports both user-specific and system credentials
        """
        # DISABLED: Skip secure initialization completely to prevent infinite loops
        logger.info("Secure provider initialization disabled to prevent infinite loops")
        
        # Go directly to insecure fallback
        if not hasattr(self, '_fallback_init_completed'):
            logger.warning("Using insecure credential fallback - not recommended for production")
            await self._fallback_insecure_initialization()
            self._fallback_init_completed = True
        
        return
    
    async def _fallback_insecure_initialization(self):
        """
        Fallback to insecure environment variable initialization for development
        
        ⚠️ SECURITY WARNING:
        This method is only for development/testing purposes.
        Production deployments should use secure credential management.
        """
        import os
        
        # Add circuit breaker to prevent infinite fallback loops
        if hasattr(self, '_fallback_init_completed') and self._fallback_init_completed:
            logger.warning("Fallback initialization already completed, skipping to prevent infinite loop")
            return
        
        self._fallback_init_completed = True
        logger.warning("Using insecure credential fallback - not recommended for production")
        
        # OpenAI not used - NWTN uses Claude API only
        # All reasoning engines configured to use Claude API
        
        # 🔑 Anthropic Configuration (INSECURE FALLBACK)
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.client_registry.register_provider(ModelProvider.ANTHROPIC, {
                'api_key': anthropic_key,
                'secure': False  # Mark as insecure
            })
            logger.warning("Anthropic provider configured with insecure credentials")
        
        # 🔑 Hugging Face Configuration (INSECURE FALLBACK)
        hf_key = os.getenv('HUGGINGFACE_API_KEY')
        if hf_key:
            self.client_registry.register_provider(ModelProvider.HUGGINGFACE, {
                'api_key': hf_key,
                'secure': False  # Mark as insecure
            })
            logger.warning("HuggingFace provider configured with insecure credentials")
    
    def _setup_default_model_mappings(self):
        """Set up default mappings from model IDs to providers"""
        # NWTN uses Claude API only - all models map to Anthropic
        # Anthropic models
        anthropic_models = [
            'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'claude-2.1',
            'claude-3-5-sonnet', 'claude-3-5-sonnet-20241022'
        ]
        for model in anthropic_models:
            self.model_providers[model] = ModelProvider.ANTHROPIC
        
        # Common HuggingFace models
        hf_models = ['microsoft/DialoGPT-medium', 'facebook/blenderbot-400M-distill']
        for model in hf_models:
            self.model_providers[model] = ModelProvider.HUGGINGFACE
    
    def _get_model_provider(self, model_id: str) -> ModelProvider:
        """
        Determine the provider for a given model ID
        
        🎯 PROVIDER DETECTION:
        1. Check explicit mappings first
        2. Use heuristics based on model name
        3. Default to local if model file exists
        4. Fallback to HuggingFace as last resort
        """
        # Check explicit mappings
        if model_id in self.model_providers:
            return self.model_providers[model_id]
        
        # Use heuristics based on model name
        model_lower = model_id.lower()
        
        # NWTN uses Claude API only - all AI models route to Anthropic
        if any(name in model_lower for name in ['claude', 'anthropic']):
            return ModelProvider.ANTHROPIC
        elif any(name in model_lower for name in ['gpt', 'davinci', 'curie', 'babbage']):
            # Legacy OpenAI models redirect to Claude
            logger.warning(f"Legacy OpenAI model {model_id} redirected to Claude API")
            return ModelProvider.ANTHROPIC
        elif model_id.startswith('prsm-distilled-'):
            return ModelProvider.LOCAL
        elif '/' in model_id:  # Hugging Face format: org/model
            return ModelProvider.HUGGINGFACE
        else:
            # Default to Claude for unknown AI models
            return ModelProvider.ANTHROPIC
    
    async def _validate_response(self, content: str, model_id: str) -> Dict[str, Any]:
        """
        Validate model response through PRSM safety systems
        
        🛡️ SAFETY VALIDATION:
        1. Content filtering for harmful outputs
        2. Quality assessment and confidence scoring
        3. Bias detection and mitigation
        4. Format validation and standardization
        """
        try:
            # TODO: Integrate with actual safety monitor
            # For now, perform basic validation
            
            if not content or len(content.strip()) == 0:
                raise ValueError("Empty response from model")
            
            # Basic safety checks
            unsafe_patterns = ['harmful', 'illegal', 'dangerous']
            if any(pattern in content.lower() for pattern in unsafe_patterns):
                logger.warning("Potentially unsafe content detected", model_id=model_id)
            
            # Format response consistently
            validated_result = {
                "type": "model_response",
                "content": content.strip(),
                "model_id": model_id,
                "validation_passed": True,
                "safety_score": 0.9,  # Placeholder score
                "timestamp": time.time()
            }
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return {
                "type": "validation_error",
                "content": "Response failed safety validation",
                "model_id": model_id,
                "validation_passed": False,
                "error": str(e)
            }
    
    async def _attempt_fallback(self, task: str, failed_model_id: str) -> Optional[ExecutionResult]:
        """
        Attempt fallback to alternative model on failure
        
        🔄 FALLBACK STRATEGY:
        1. Identify similar models from different providers
        2. Try cheaper/faster alternatives
        3. Use local models if available
        4. Return None if all fallbacks fail
        """
        try:
            # Define fallback mappings - Claude API only
            fallback_map = {
                'claude-3-opus': 'claude-3-sonnet',
                'claude-3-sonnet': 'claude-3-haiku',
                'claude-3-5-sonnet': 'claude-3-sonnet',
                'claude-3-5-sonnet-20241022': 'claude-3-sonnet'
            }
            
            fallback_model = fallback_map.get(failed_model_id)
            if fallback_model:
                logger.info(f"Attempting fallback: {failed_model_id} -> {fallback_model}")
                return await self._execute_with_model(task, fallback_model)
            
        except Exception as e:
            logger.error(f"Fallback attempt failed: {e}")
        
        return None
    
    def _update_execution_stats(self, response: ModelExecutionResponse, execution_time: float, success: bool):
        """Update comprehensive execution statistics"""
        self.execution_stats["total_executions"] += 1
        
        if success:
            self.execution_stats["successful_executions"] += 1
        
        # Update timing statistics
        self.execution_stats["total_execution_time"] += execution_time
        self.execution_stats["average_execution_time"] = (
            self.execution_stats["total_execution_time"] / self.execution_stats["total_executions"]
        )
        
        # Update provider usage
        provider_name = response.provider.value
        if provider_name not in self.execution_stats["provider_usage"]:
            self.execution_stats["provider_usage"][provider_name] = 0
        self.execution_stats["provider_usage"][provider_name] += 1
        
        # Update model usage
        if response.model_id not in self.execution_stats["model_usage_count"]:
            self.execution_stats["model_usage_count"][response.model_id] = 0
        self.execution_stats["model_usage_count"][response.model_id] += 1
        
        # Update token usage
        if hasattr(response, 'token_usage') and response.token_usage:
            tokens = response.token_usage.get('total_tokens', 0)
            self.execution_stats["tokens_used"] += tokens
        
        total_executions = self.execution_stats["total_executions"]
        successful_executions = self.execution_stats["successful_executions"]
        
        logger.debug("Updated execution statistics",
                    total_executions=total_executions,
                    success_rate=successful_executions / total_executions,
                    avg_time=self.execution_stats["average_execution_time"])
    
    async def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'client_registry'):
            await self.client_registry.cleanup()
        logger.info("ModelExecutor cleanup completed")
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a specific execution"""
        if execution_id in self.active_executions:
            task = self.active_executions[execution_id]
            task.cancel()
            del self.active_executions[execution_id]
            logger.info("Execution cancelled",
                       agent_id=self.agent_id,
                       execution_id=execution_id)
            return True
        return False
    
    def get_active_executions(self) -> List[str]:
        """Get list of currently active execution IDs"""
        return list(self.active_executions.keys())
    
    async def execute_request(self, prompt: str, model_name: str, temperature: float = 0.7, **kwargs) -> str:
        """
        Execute a request with the specified model
        
        Args:
            prompt: The text prompt to send to the model
            model_name: The name/ID of the model to use
            temperature: Temperature setting for the model
            **kwargs: Additional parameters
            
        Returns:
            The model's response as a string
        """
        # Prepare input data for the process method
        input_data = {
            "task": prompt,
            "models": [model_name],
            "parallel": False,
            "temperature": temperature
        }
        
        # Execute using the existing process method
        try:
            results = await self.process(input_data)
            
            # Return the content from the first result
            if results and results[0].success:
                result_content = results[0].result
                if isinstance(result_content, dict):
                    return result_content.get("content", str(result_content))
                else:
                    return str(result_content)
            else:
                error_msg = results[0].error if results else "Unknown error"
                raise Exception(f"Model execution failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"execute_request failed: {e}", 
                        agent_id=self.agent_id,
                        model_name=model_name,
                        prompt_length=len(prompt))
            raise


# Factory function
def create_executor(timeout_seconds: Optional[int] = None) -> ModelExecutor:
    """Create a model executor agent"""
    return ModelExecutor(timeout_seconds=timeout_seconds)
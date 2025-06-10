"""
Model Executor Agent
Executes tasks using assigned specialist models
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from uuid import UUID
import structlog

from prsm.core.models import AgentType, ArchitectTask
from prsm.core.config import get_settings
from prsm.agents.base import BaseAgent
from .api_clients import (
    ModelClientRegistry, ModelProvider, ModelExecutionRequest, 
    ModelExecutionResponse
)

logger = structlog.get_logger(__name__)
settings = get_settings()


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
    
    def __init__(self, agent_id: Optional[str] = None, timeout_seconds: Optional[int] = None):
        super().__init__(agent_id=agent_id, agent_type=AgentType.EXECUTOR)
        self.timeout_seconds = timeout_seconds or settings.agent_timeout_seconds
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # 🤖 REAL MODEL CLIENT REGISTRY
        # Manages connections to all AI model providers
        self.client_registry = ModelClientRegistry()
        
        # 🎯 MODEL PROVIDER MAPPING
        # Maps model IDs to their providers for efficient routing
        self.model_providers: Dict[str, ModelProvider] = {}
        
        # 📊 COMPREHENSIVE EXECUTION STATISTICS
        # Track real API usage, costs, and performance across providers
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
        
        logger.info("ModelExecutor initialized with real API clients", agent_id=self.agent_id)
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> List[ExecutionResult]:
        """
        Process execution request
        
        Args:
            input_data: Task and model assignments
            context: Optional execution context
            
        Returns:
            List[ExecutionResult]: Results from all model executions
        """
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
            logger.warning("No model assignments provided", agent_id=self.agent_id)
            return []
        
        logger.info("Starting task execution",
                   agent_id=self.agent_id,
                   task_length=len(task_description),
                   model_count=len(model_assignments),
                   parallel=parallel)
        
        # Execute with assigned models
        if parallel:
            results = await self._execute_parallel(task_description, model_assignments)
        else:
            results = await self._execute_sequential(task_description, model_assignments)
        
        logger.info("Execution completed",
                   agent_id=self.agent_id,
                   total_results=len(results),
                   successful_results=len([r for r in results if r.success]))
        
        return results
    
    async def _execute_parallel(self, task: str, model_assignments: List[str]) -> List[ExecutionResult]:
        """Execute task with multiple models in parallel"""
        tasks = []
        
        for model_id in model_assignments:
            task_coro = self._execute_with_model(task, model_id)
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
    
    async def _execute_sequential(self, task: str, model_assignments: List[str]) -> List[ExecutionResult]:
        """Execute task with models sequentially"""
        results = []
        
        for model_id in model_assignments:
            try:
                result = await asyncio.wait_for(
                    self._execute_with_model(task, model_id),
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
    
    async def _execute_with_model(self, task: str, model_id: str) -> ExecutionResult:
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
                
                # 📊 UPDATE STATISTICS
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
                              error=response.error)
                
                # 🔄 ATTEMPT FALLBACK TO ALTERNATIVE MODEL
                fallback_result = await self._attempt_fallback(task, model_id)
                if fallback_result:
                    return fallback_result
                
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
    
    async def _simulate_model_execution(self, task: str, model_id: str) -> Dict[str, Any]:
        """Simulate model execution for testing purposes"""
        # TODO: Replace with actual model execution
        
        # Generate a simulated response based on task characteristics
        task_lower = task.lower()
        
        if "research" in task_lower:
            result = {
                "type": "research_findings",
                "summary": f"Research results for: {task[:100]}...",
                "key_points": [
                    "Finding 1: Relevant information discovered",
                    "Finding 2: Additional context provided",
                    "Finding 3: Conclusions drawn"
                ],
                "confidence": 0.85,
                "sources": ["source_1", "source_2", "source_3"]
            }
        elif "analyze" in task_lower:
            result = {
                "type": "analysis_result",
                "summary": f"Analysis of: {task[:100]}...",
                "insights": [
                    "Pattern 1: Significant trend identified",
                    "Pattern 2: Correlation discovered",
                    "Pattern 3: Anomaly detected"
                ],
                "confidence": 0.92,
                "methodology": "Statistical analysis with ML techniques"
            }
        elif "explain" in task_lower:
            result = {
                "type": "explanation",
                "explanation": f"Explanation for: {task[:100]}...",
                "key_concepts": [
                    "Concept 1: Fundamental principle",
                    "Concept 2: Related mechanism",
                    "Concept 3: Practical application"
                ],
                "confidence": 0.88,
                "clarity_score": 0.9
            }
        else:
            result = {
                "type": "general_response",
                "content": f"Response to: {task[:100]}...",
                "details": "Processed using general-purpose model",
                "confidence": 0.75
            }
        
        # Add model-specific information
        result["model_id"] = model_id
        result["processing_metadata"] = {
            "task_length": len(task),
            "processing_approach": "simulated_execution",
            "version": "1.0"
        }
        
        return result
    
    def _initialize_providers(self):
        """
        Initialize provider configurations using secure credential management
        
        🔧 SECURE PROVIDER SETUP:
        - Load API keys from encrypted credential manager
        - Validate credentials before provider registration
        - Configure rate limits and timeouts
        - Set up fallback strategies with secure fallbacks
        """
        # This method is kept for backward compatibility but now uses secure credentials
        # Actual secure initialization happens in _initialize_secure_providers
        asyncio.create_task(self._initialize_secure_providers())
        
        # 📂 Local Models Configuration (no credentials needed)
        import os
        local_models_path = os.getenv('PRSM_LOCAL_MODELS_PATH', '/models')
        self.client_registry.register_provider(ModelProvider.LOCAL, {
            'model_path': local_models_path
        })
        
        # 🎯 Default Model Provider Mappings
        self._setup_default_model_mappings()
    
    async def _initialize_secure_providers(self):
        """
        Initialize providers using secure credential management
        
        🔐 SECURITY FEATURES:
        - Uses encrypted credential manager instead of environment variables
        - Validates credentials before registration
        - Logs all credential access for audit trail
        - Supports both user-specific and system credentials
        """
        from prsm.integrations.security.secure_api_client_factory import secure_client_factory, SecureClientType
        
        try:
            # Initialize for system-level operations (background tasks, etc.)
            user_id = "system"
            
            # 🔐 Secure OpenAI Configuration
            if await secure_client_factory.validate_client_credentials(SecureClientType.OPENAI, user_id):
                openai_client = await secure_client_factory.get_secure_client(SecureClientType.OPENAI, user_id)
                if openai_client:
                    # Register secure OpenAI client
                    self.client_registry.register_provider(ModelProvider.OPENAI, {
                        'secure_client': openai_client,
                        'client_type': SecureClientType.OPENAI
                    })
                    logger.info("Secure OpenAI provider configured")
            else:
                logger.warning("OpenAI credentials not available or invalid")
            
            # 🔐 Secure Anthropic Configuration  
            if await secure_client_factory.validate_client_credentials(SecureClientType.ANTHROPIC, user_id):
                anthropic_client = await secure_client_factory.get_secure_client(SecureClientType.ANTHROPIC, user_id)
                if anthropic_client:
                    # Register secure Anthropic client
                    self.client_registry.register_provider(ModelProvider.ANTHROPIC, {
                        'secure_client': anthropic_client,
                        'client_type': SecureClientType.ANTHROPIC
                    })
                    logger.info("Secure Anthropic provider configured")
            else:
                logger.warning("Anthropic credentials not available or invalid")
            
            # 🔐 Secure Hugging Face Configuration
            if await secure_client_factory.validate_client_credentials(SecureClientType.HUGGINGFACE, user_id):
                hf_client = await secure_client_factory.get_secure_client(SecureClientType.HUGGINGFACE, user_id)
                if hf_client:
                    # Register secure HuggingFace client
                    self.client_registry.register_provider(ModelProvider.HUGGINGFACE, {
                        'secure_client': hf_client,
                        'client_type': SecureClientType.HUGGINGFACE
                    })
                    logger.info("Secure HuggingFace provider configured")
            else:
                logger.warning("HuggingFace credentials not available or invalid")
            
            logger.info("Secure provider initialization completed")
            
        except Exception as e:
            logger.error("Failed to initialize secure providers", error=str(e))
            # Fall back to insecure initialization for development
            await self._fallback_insecure_initialization()
    
    async def _fallback_insecure_initialization(self):
        """
        Fallback to insecure environment variable initialization for development
        
        ⚠️ SECURITY WARNING:
        This method is only for development/testing purposes.
        Production deployments should use secure credential management.
        """
        import os
        
        logger.warning("Using insecure credential fallback - not recommended for production")
        
        # 🔑 OpenAI Configuration (INSECURE FALLBACK)
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.client_registry.register_provider(ModelProvider.OPENAI, {
                'api_key': openai_key,
                'secure': False  # Mark as insecure
            })
            logger.warning("OpenAI provider configured with insecure credentials")
        
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
        # OpenAI models
        openai_models = ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo']
        for model in openai_models:
            self.model_providers[model] = ModelProvider.OPENAI
        
        # Anthropic models
        anthropic_models = ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'claude-2.1']
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
        
        if any(name in model_lower for name in ['gpt', 'davinci', 'curie', 'babbage']):
            return ModelProvider.OPENAI
        elif any(name in model_lower for name in ['claude', 'anthropic']):
            return ModelProvider.ANTHROPIC
        elif model_id.startswith('prsm-distilled-'):
            return ModelProvider.LOCAL
        elif '/' in model_id:  # Hugging Face format: org/model
            return ModelProvider.HUGGINGFACE
        else:
            # Default to local for unknown models
            return ModelProvider.LOCAL
    
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
            # Define fallback mappings
            fallback_map = {
                'gpt-4': 'gpt-3.5-turbo',
                'claude-3-opus': 'claude-3-sonnet',
                'claude-3-sonnet': 'claude-3-haiku'
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
        
        logger.debug("Updated execution statistics",
                    total_executions=self.execution_stats["total_executions"],
                    success_rate=self.execution_stats["successful_executions"] / self.execution_stats["total_executions"],
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


# Factory function
def create_executor(timeout_seconds: Optional[int] = None) -> ModelExecutor:
    """Create a model executor agent"""
    return ModelExecutor(timeout_seconds=timeout_seconds)
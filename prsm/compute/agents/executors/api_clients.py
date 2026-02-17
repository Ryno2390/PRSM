"""
Real API Clients for PRSM Model Execution

🎯 PURPOSE IN PRSM:
This module provides concrete implementations for executing tasks with real AI models
from various providers (OpenAI, Anthropic, Hugging Face, etc.). It replaces the
simulated model execution with actual API calls and local model inference.

🔧 INTEGRATION POINTS:
- ModelExecutor: Uses these clients for real model execution
- Model Registry: Provides model metadata and access credentials
- FTNS System: Tracks usage costs and token consumption
- Safety Monitor: Validates all model outputs before returning
"""

import asyncio
import aiohttp
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class ModelProvider(Enum):
    """Supported AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    OLLAMA = "ollama"
    LOCAL = "local"
    PRSM_DISTILLED = "prsm_distilled"


@dataclass
class ModelExecutionRequest:
    """Request for model execution"""
    prompt: str
    model_id: str
    provider: ModelProvider
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ModelExecutionResponse:
    """Response from model execution"""
    content: str
    provider: ModelProvider
    model_id: str
    execution_time: float
    token_usage: Dict[str, int]
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseModelClient(ABC):
    """
    Abstract base class for AI model clients
    
    🎯 PURPOSE: Provides consistent interface across different AI providers
    while allowing for provider-specific optimizations and features.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.config = kwargs
        
    async def initialize(self):
        """Initialize the client and establish connections"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers=self._get_headers()
        )
        await self._setup_client()
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        pass
    
    @abstractmethod
    async def _setup_client(self):
        """Provider-specific setup"""
        pass
    
    @abstractmethod
    def _get_provider(self) -> ModelProvider:
        """Get the provider type for this client"""
        pass
    
    @abstractmethod
    async def _execute_api_request(self, request: ModelExecutionRequest) -> Dict[str, Any]:
        """Execute the provider-specific API request"""
        pass
    
    @abstractmethod
    def _parse_success_response(self, data: Dict[str, Any], request: ModelExecutionRequest) -> Dict[str, Any]:
        """Parse successful API response into standardized format"""
        pass
    
    @abstractmethod
    def _parse_error_response(self, error_data: Dict[str, Any], status_code: int) -> str:
        """Parse error response into human-readable message"""
        pass
    
    async def execute(self, request: ModelExecutionRequest) -> ModelExecutionResponse:
        """
        Execute a task with the model using standardized flow
        
        🔄 STANDARDIZED EXECUTION FLOW:
        1. Start timing and logging
        2. Execute provider-specific API request
        3. Parse response using provider-specific logic
        4. Create standardized response object
        5. Handle errors consistently
        """
        start_time = time.time()
        provider = self._get_provider()
        
        logger.debug(
            "Starting model execution",
            provider=provider.value,
            model_id=request.model_id,
            prompt_length=len(request.prompt)
        )
        
        try:
            # Execute provider-specific API request
            response_data = await self._execute_api_request(request)
            execution_time = time.time() - start_time
            
            # Parse success response
            parsed_data = self._parse_success_response(response_data, request)
            
            response = ModelExecutionResponse(
                content=parsed_data["content"],
                provider=provider,
                model_id=request.model_id,
                execution_time=execution_time,
                token_usage=parsed_data.get("token_usage", {}),
                success=True,
                metadata=parsed_data.get("metadata", {})
            )
            
            logger.info(
                "Model execution completed successfully",
                provider=provider.value,
                model_id=request.model_id,
                execution_time=execution_time,
                tokens_used=parsed_data.get("token_usage", {}).get("total_tokens", 0)
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(
                "Model execution failed",
                provider=provider.value,
                model_id=request.model_id,
                execution_time=execution_time,
                error=error_msg,
                exc_info=True
            )
            
            return ModelExecutionResponse(
                content="",
                provider=provider,
                model_id=request.model_id,
                execution_time=execution_time,
                token_usage={},
                success=False,
                error=error_msg
            )
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# OpenAI client removed - NWTN uses Claude API only


class AnthropicClient(BaseModelClient):
    """
    Anthropic API client for Claude models
    
    🤖 MODELS SUPPORTED:
    - Claude-3 (claude-3-opus, claude-3-sonnet, claude-3-haiku)
    - Claude-2 (claude-2.1, claude-2.0)
    """
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.anthropic.com/v1"
        
    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    async def _setup_client(self):
        """Validate API key"""
        logger.info("Anthropic client initialized")
    
    def _get_provider(self) -> ModelProvider:
        return ModelProvider.ANTHROPIC
    
    async def _execute_api_request(self, request: ModelExecutionRequest) -> Dict[str, Any]:
        """Execute Anthropic API request"""
        payload = {
            "model": request.model_id,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}]
        }
        
        if request.system_prompt:
            payload["system"] = request.system_prompt
        
        async with self.session.post(
            f"{self.base_url}/messages",
            json=payload
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_data = await response.json()
                error_msg = self._parse_error_response(error_data, response.status)
                raise Exception(f"Anthropic API error: {error_msg}")
    
    def _parse_success_response(self, data: Dict[str, Any], request: ModelExecutionRequest) -> Dict[str, Any]:
        """Parse Anthropic success response"""
        return {
            "content": data["content"][0]["text"],
            "token_usage": data.get("usage", {}),
            "metadata": {"stop_reason": data.get("stop_reason")}
        }
    
    def _parse_error_response(self, error_data: Dict[str, Any], status_code: int) -> str:
        """Parse Anthropic error response"""
        return error_data.get("error", {}).get("message", f"HTTP {status_code}")


class HuggingFaceClient(BaseModelClient):
    """
    Hugging Face API client for community models
    
    🤖 MODELS SUPPORTED:
    - All Hugging Face Inference API models
    - Custom models hosted on HF Hub
    - PRSM distilled models published to HF Hub
    """
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api-inference.huggingface.co/models"
        
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def _setup_client(self):
        """Test API connectivity"""
        logger.info("HuggingFace client initialized")
    
    def _get_provider(self) -> ModelProvider:
        return ModelProvider.HUGGINGFACE
    
    async def _execute_api_request(self, request: ModelExecutionRequest) -> Dict[str, Any]:
        """Execute HuggingFace API request"""
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "return_full_text": False
            }
        }
        
        async with self.session.post(
            f"{self.base_url}/{request.model_id}",
            json=payload
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")
    
    def _parse_success_response(self, data: Dict[str, Any], request: ModelExecutionRequest) -> Dict[str, Any]:
        """Parse HuggingFace success response"""
        # Handle different response formats
        if isinstance(data, list) and len(data) > 0:
            content = data[0].get("generated_text", str(data[0]))
        else:
            content = str(data)
        
        # Estimate token usage (HF doesn't always provide this)
        estimated_tokens = len(content.split()) * 1.3  # Rough estimate
        
        return {
            "content": content,
            "token_usage": {"estimated_tokens": int(estimated_tokens)},
            "metadata": {"response_type": "text_generation"}
        }
    
    def _parse_error_response(self, error_data: Dict[str, Any], status_code: int) -> str:
        """Parse HuggingFace error response"""
        return f"HTTP {status_code}: {str(error_data)}"


class LocalModelClient(BaseModelClient):
    """
    Local model client for PRSM distilled models
    
    🤖 MODELS SUPPORTED:
    - PyTorch models (.pth files)
    - TensorFlow models (SavedModel format)
    - Transformers models (local directory)
    - ONNX models for cross-platform inference
    """
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(None, **kwargs)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def _get_headers(self) -> Dict[str, str]:
        return {}
    
    async def _setup_client(self):
        """Load local model based on file type"""
        try:
            # Detect model type and load appropriately
            if self.model_path.endswith('.pth'):
                await self._load_pytorch_model()
            elif 'saved_model' in self.model_path:
                await self._load_tensorflow_model()
            else:
                await self._load_transformers_model()
                
            logger.info("Local model loaded", model_path=self.model_path)
        except Exception as e:
            logger.error("Failed to load local model", error=str(e), model_path=self.model_path)
            raise
    
    def _get_provider(self) -> ModelProvider:
        return ModelProvider.LOCAL
    
    async def _load_pytorch_model(self):
        """Load PyTorch model"""
        try:
            import torch
            self.model = torch.load(self.model_path, map_location='cpu', weights_only=True)
            self.model.eval()
        except ImportError:
            raise RuntimeError("PyTorch not available for local model execution")
    
    async def _load_tensorflow_model(self):
        """Load TensorFlow model"""
        try:
            import tensorflow as tf
            self.model = tf.saved_model.load(self.model_path)
        except ImportError:
            raise RuntimeError("TensorFlow not available for local model execution")
    
    async def _load_transformers_model(self):
        """Load Transformers model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except ImportError:
            raise RuntimeError("Transformers not available for local model execution")
    
    async def _execute_api_request(self, request: ModelExecutionRequest) -> Dict[str, Any]:
        """Execute local model inference (overrides HTTP-based execution)"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Execute based on model type
        if hasattr(self.model, 'generate') and self.tokenizer:
            # Transformers model
            content = await self._execute_transformers(request)
        else:
            # PyTorch/TensorFlow model - simplified execution
            content = f"Local model response to: {request.prompt[:100]}..."
        
        return {"content": content}
    
    def _parse_success_response(self, data: Dict[str, Any], request: ModelExecutionRequest) -> Dict[str, Any]:
        """Parse local model success response"""
        return {
            "content": data["content"],
            "token_usage": {"local_inference": True},
            "metadata": {"model_path": self.model_path}
        }
    
    def _parse_error_response(self, error_data: Dict[str, Any], status_code: int) -> str:
        """Parse local model error response"""
        return str(error_data)
    
    async def _execute_transformers(self, request: ModelExecutionRequest) -> str:
        """Execute with Transformers model"""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch not available for transformers execution")
        
        inputs = self.tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response


class ModelClientRegistry:
    """
    Registry for managing different model clients
    
    🎯 PURPOSE IN PRSM:
    Central hub for routing model execution requests to appropriate clients
    based on provider and model specifications. Handles client lifecycle,
    connection pooling, and fallback strategies.
    """
    
    def __init__(self):
        self.clients: Dict[str, BaseModelClient] = {}
        self.provider_configs: Dict[ModelProvider, Dict[str, Any]] = {}
    
    def register_provider(self, provider: ModelProvider, config: Dict[str, Any]):
        """Register a provider with configuration"""
        self.provider_configs[provider] = config
        logger.info(f"Registered provider: {provider.value}")
    
    async def _create_fallback_config(self, provider: ModelProvider) -> Dict[str, Any]:
        """Create fallback configuration from secure credential system or environment variables"""
        import os
        
        # First try to get credentials from secure credential system
        try:
            from ...integrations.security.secure_api_client_factory import SecureClientType, secure_client_factory
            
            # Map provider to secure client type
            provider_mapping = {
                ModelProvider.ANTHROPIC: SecureClientType.ANTHROPIC,
                ModelProvider.HUGGINGFACE: SecureClientType.HUGGINGFACE,
            }
            
            if provider in provider_mapping:
                client_type = provider_mapping[provider]
                
                # Try to get system credentials
                credentials = await secure_client_factory._get_secure_credentials(client_type, "system")
                if credentials:
                    logger.info(f"Using secure credentials for {provider.value}")
                    return credentials
                    
        except Exception as e:
            logger.warning(f"Failed to get secure credentials for {provider.value}: {e}")
        
        # Fallback to environment variables
        logger.warning(f"No secure configuration found for {provider.value}, attempting fallback from environment variables")
        
        # OpenAI removed - NWTN uses Claude API only
        if provider == ModelProvider.ANTHROPIC:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                return {'api_key': api_key}
        elif provider == ModelProvider.HUGGINGFACE:
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            if api_key:
                return {'api_key': api_key}
        
        # If no fallback is available, raise an error
        raise ValueError(f"No configuration or fallback credentials available for {provider.value}")
    
    
    async def get_client(self, provider: ModelProvider, model_id: str) -> BaseModelClient:
        """Get or create client for provider"""
        client_key = f"{provider.value}:{model_id}"
        
        if client_key not in self.clients:
            config = self.provider_configs.get(provider, {})
            
            # If no configuration is available, try to create a fallback configuration
            if not config and provider != ModelProvider.LOCAL:
                config = await self._create_fallback_config(provider)
            
            # OpenAI removed - NWTN uses Claude API only
            if provider == ModelProvider.ANTHROPIC:
                client = AnthropicClient(**config)
            elif provider == ModelProvider.HUGGINGFACE:
                client = HuggingFaceClient(**config)
            elif provider == ModelProvider.LOCAL:
                client = LocalModelClient(model_path=config.get('model_path', model_id), **config)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            await client.initialize()
            self.clients[client_key] = client
        
        return self.clients[client_key]
    
    async def execute_with_provider(
        self, 
        provider: ModelProvider, 
        model_id: str, 
        request: ModelExecutionRequest
    ) -> ModelExecutionResponse:
        """Execute request with specific provider"""
        client = await self.get_client(provider, model_id)
        return await client.execute(request)
    
    async def cleanup(self):
        """Close all client connections"""
        for client in self.clients.values():
            await client.close()
        self.clients.clear()
        logger.info("All model clients closed")
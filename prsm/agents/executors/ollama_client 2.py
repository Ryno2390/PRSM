#!/usr/bin/env python3
"""
Ollama Local Model Client for PRSM
=================================

Integrates local AI models via Ollama for:
- Zero-cost development and testing
- Privacy-first sensitive data processing  
- Offline capability and rate limit elimination
- Hybrid cloud/local architecture demonstration

🔒 PRIVACY BENEFITS:
- Sensitive data never leaves local machine
- No external API calls for confidential queries
- Complete control over model execution
- GDPR/compliance-friendly processing

⚡ PERFORMANCE BENEFITS:
- No API rate limits during development
- Predictable latency (no network dependency)
- Model persistence eliminates cold starts
- Parallel processing capability

🏗️ PRSM INTEGRATION:
- Compatible with ModelExecutionRequest/Response
- Unified interface with cloud providers
- Cost tracking (CPU/GPU usage estimation)
- Seamless routing between local/cloud
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import structlog

# Import PRSM base classes
from .api_clients import (
    BaseModelClient,
    ModelExecutionRequest,
    ModelExecutionResponse,
    ModelProvider
)

logger = structlog.get_logger(__name__)


@dataclass
class OllamaModel:
    """Ollama model metadata"""
    name: str
    size: str
    family: str
    parameter_count: str
    quantization: str
    capabilities: List[str]
    context_length: int = 4096
    estimated_vram_gb: float = 0.0
    

class OllamaClient(BaseModelClient):
    """Local AI model client via Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.total_requests = 0
        self.total_generation_time = 0.0
        self.available_models: Dict[str, OllamaModel] = {}
        
        # Popular model configurations
        self.model_configs = {
            "llama3.2:1b": OllamaModel(
                name="llama3.2:1b",
                size="1.3GB",
                family="llama",
                parameter_count="1.24B",
                quantization="Q8_0",
                capabilities=["chat", "instruct", "multilingual"],
                context_length=131072,
                estimated_vram_gb=2.6
            ),
            "llama3.2:3b": OllamaModel(
                name="llama3.2:3b",
                size="2.0GB",
                family="llama",
                parameter_count="3.21B",
                quantization="Q4_K_M",
                capabilities=["chat", "instruct", "multilingual", "reasoning"],
                context_length=131072,
                estimated_vram_gb=4.2
            ),
            "mistral:7b": OllamaModel(
                name="mistral:7b",
                size="4.1GB",
                family="mistral",
                parameter_count="7.24B",
                quantization="Q4_0",
                capabilities=["chat", "instruct", "code"],
                context_length=32768,
                estimated_vram_gb=6.8
            ),
            "codellama:7b": OllamaModel(
                name="codellama:7b",
                size="3.8GB",
                family="llama",
                parameter_count="6.74B",
                quantization="Q4_0",
                capabilities=["code", "instruct", "completion"],
                context_length=16384,
                estimated_vram_gb=6.2
            )
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Ollama API requests"""
        return {
            "Content-Type": "application/json",
            "User-Agent": "PRSM/1.0 (Local Ollama Client)"
        }
    
    async def _setup_client(self) -> None:
        """Ollama-specific setup and model discovery"""
        try:
            # Test server connectivity
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    models_data = await response.json()
                    installed_models = [model['name'] for model in models_data.get('models', [])]
                    
                    # Update available models with installed ones
                    for model_name in installed_models:
                        if model_name in self.model_configs:
                            self.available_models[model_name] = self.model_configs[model_name]
                    
                    logger.info("Ollama client initialized",
                               installed_models=len(installed_models),
                               available_models=len(self.available_models))
                else:
                    logger.error("Failed to connect to Ollama server", status=response.status)
        except Exception as e:
            logger.error("Ollama setup failed", error=str(e))
    
    async def list_available_models(self) -> List[str]:
        """List locally available models"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model['name'] for model in data.get('models', [])]
                return []
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return []
    
    def get_model_info(self, model_name: str) -> Optional[OllamaModel]:
        """Get model information"""
        return self.model_configs.get(model_name)
    
    def estimate_local_cost(self, generation_time: float, model_name: str) -> Dict[str, float]:
        """Estimate local compute cost (electricity, hardware depreciation)"""
        model_info = self.get_model_info(model_name)
        if not model_info:
            return {"compute_cost_usd": 0.0, "electricity_cost_usd": 0.0}
        
        # Rough estimates for local compute costs
        # GPU power consumption: ~200W for inference
        # Electricity cost: ~$0.15/kWh average US rate
        electricity_cost = (200 * generation_time / 3600) * 0.15 / 1000  # Convert to USD
        
        # Hardware depreciation: ~$0.001 per inference for consumer hardware
        compute_cost = 0.001
        
        return {
            "compute_cost_usd": compute_cost,
            "electricity_cost_usd": electricity_cost,
            "total_local_cost_usd": compute_cost + electricity_cost
        }
    
    async def execute(self, request: ModelExecutionRequest) -> ModelExecutionResponse:
        """Execute a model request through Ollama"""
        start_time = time.time()
        
        try:
            # Check if model is available
            if request.model_id not in self.available_models:
                return ModelExecutionResponse(
                    content="",
                    provider=ModelProvider.LOCAL,
                    model_id=request.model_id,
                    execution_time=time.time() - start_time,
                    token_usage={},
                    success=False,
                    error=f"Model '{request.model_id}' not available locally. Available: {list(self.available_models.keys())}"
                )
            
            # Build request payload for Ollama
            messages = []
            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": request.prompt
            })
            
            payload = {
                "model": request.model_id,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            }
            
            logger.info("Sending request to Ollama",
                       model=request.model_id,
                       tokens_requested=request.max_tokens)
            
            # Make API request to Ollama
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    return ModelExecutionResponse(
                        content="",
                        provider=ModelProvider.LOCAL,
                        model_id=request.model_id,
                        execution_time=time.time() - start_time,
                        token_usage={},
                        success=False,
                        error=f"Ollama error {response.status}: {error_text}"
                    )
                
                response_data = await response.json()
                
                # Extract response content
                content = response_data['message']['content']
                
                # Parse token usage (Ollama provides some timing info)
                eval_count = response_data.get('eval_count', 0)
                prompt_eval_count = response_data.get('prompt_eval_count', 0)
                total_tokens = eval_count + prompt_eval_count
                
                # Calculate costs
                execution_time = time.time() - start_time
                generation_time = response_data.get('total_duration', 0) / 1e9  # Convert nanoseconds to seconds
                cost_breakdown = self.estimate_local_cost(generation_time, request.model_id)
                
                # Update session stats
                self.total_requests += 1
                self.total_generation_time += generation_time
                
                logger.info("Ollama request completed",
                           model=request.model_id,
                           tokens=total_tokens,
                           generation_time_ms=generation_time * 1000,
                           local_cost_usd=cost_breakdown['total_local_cost_usd'])
                
                return ModelExecutionResponse(
                    content=content,
                    provider=ModelProvider.LOCAL,
                    model_id=request.model_id,
                    execution_time=execution_time,
                    token_usage={
                        'prompt_tokens': prompt_eval_count,
                        'completion_tokens': eval_count,
                        'total_tokens': total_tokens
                    },
                    success=True,
                    metadata={
                        'local_execution': True,
                        'generation_time_s': generation_time,
                        'cost_breakdown': cost_breakdown,
                        'model_family': self.available_models[request.model_id].family,
                        'parameter_count': self.available_models[request.model_id].parameter_count,
                        'quantization': self.available_models[request.model_id].quantization
                    }
                )
                
        except Exception as e:
            logger.error("Ollama request failed", error=str(e))
            return ModelExecutionResponse(
                content="",
                provider=ModelProvider.LOCAL,
                model_id=request.model_id,
                execution_time=time.time() - start_time,
                token_usage={},
                success=False,
                error=f"Local execution failed: {str(e)}"
            )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        avg_generation_time = (
            self.total_generation_time / max(self.total_requests, 1)
        )
        
        return {
            'total_requests': self.total_requests,
            'total_generation_time_s': self.total_generation_time,
            'avg_generation_time_s': avg_generation_time,
            'available_models': len(self.available_models),
            'local_execution': True,
            'privacy_mode': True
        }
    
    async def download_model(self, model_name: str) -> bool:
        """Download a model to local storage"""
        try:
            logger.info("Downloading model", model=model_name)
            
            payload = {"name": model_name}
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=payload
            ) as response:
                
                if response.status == 200:
                    # Refresh available models
                    await self._setup_client()
                    logger.info("Model downloaded successfully", model=model_name)
                    return True
                else:
                    error_text = await response.text()
                    logger.error("Model download failed", 
                               model=model_name, 
                               status=response.status,
                               error=error_text)
                    return False
                    
        except Exception as e:
            logger.error("Model download error", model=model_name, error=str(e))
            return False

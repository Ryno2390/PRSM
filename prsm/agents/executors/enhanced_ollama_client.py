"""
Enhanced Ollama Client for PRSM Local Model Support
Comprehensive local model integration with performance benchmarking

🎯 PURPOSE IN PRSM:
Advanced local model support through Ollama with:
- Expanded model library with configurations
- Performance benchmarking for local models  
- Model auto-downloading and management
- Local model comparison tools
- Integration with PRSM routing and cost optimization
"""

import asyncio
import aiohttp
import json
import time
import hashlib
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

class OllamaModel(Enum):
    """Comprehensive Ollama model library"""
    # General purpose models
    LLAMA2_7B = "llama2:7b"
    LLAMA2_13B = "llama2:13b"
    LLAMA2_70B = "llama2:70b"
    
    # Code-specialized models
    CODELLAMA_7B = "codellama:7b"
    CODELLAMA_13B = "codellama:13b"
    CODELLAMA_34B = "codellama:34b"
    
    # Instruction-tuned models
    LLAMA2_7B_CHAT = "llama2:7b-chat"
    LLAMA2_13B_CHAT = "llama2:13b-chat"
    LLAMA2_70B_CHAT = "llama2:70b-chat"
    
    # Specialized models
    VICUNA_7B = "vicuna:7b"
    VICUNA_13B = "vicuna:13b"
    MISTRAL_7B = "mistral:7b"
    MIXTRAL_8X7B = "mixtral:8x7b"
    
    # Lightweight models for edge deployment
    TINYLLAMA = "tinyllama:1.1b"
    PHI_2 = "phi:2.7b"
    NEURAL_CHAT = "neural-chat:7b"
    
    # Multimodal models
    LLAVA = "llava:7b"
    LLAVA_13B = "llava:13b"
    
    # Fine-tuned variants
    OPENHERMES = "openhermes:7b"
    WIZARD_VICUNA = "wizard-vicuna:13b"
    ORCA_MINI = "orca-mini:3b"

@dataclass
class ModelConfiguration:
    """Configuration for specific model deployment"""
    model: OllamaModel
    context_length: int
    memory_requirement: int  # MB
    disk_requirement: int    # MB
    cpu_threads: int
    use_gpu: bool
    recommended_use_cases: List[str]
    performance_tier: str    # "fast", "balanced", "quality"
    cost_tier: str          # "free", "low", "medium", "high"

@dataclass
class OllamaPerformanceMetrics:
    """Performance metrics for local model inference"""
    model: str
    tokens_per_second: float
    memory_usage: int       # MB
    cpu_usage: float       # Percentage
    gpu_usage: float       # Percentage  
    latency: float         # Seconds
    quality_score: float   # 0-1
    benchmark_date: float  # Timestamp

@dataclass
class OllamaUsageStats:
    """Track local model usage statistics"""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_inference_time: float = 0.0
    average_tokens_per_second: float = 0.0
    memory_peak: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    model_usage: Dict[str, int] = field(default_factory=dict)

class EnhancedOllamaClient:
    """
    Enhanced Ollama client with comprehensive local model support
    
    🚀 FEATURES:
    - Expanded model library with 20+ models
    - Automatic model downloading and management
    - Performance benchmarking and comparison
    - Resource usage optimization
    - Model recommendation engine
    - Integration with PRSM cost optimization
    """
    
    # Model configurations optimized for different use cases
    MODEL_CONFIGS = {
        OllamaModel.TINYLLAMA: ModelConfiguration(
            model=OllamaModel.TINYLLAMA,
            context_length=2048,
            memory_requirement=1024,
            disk_requirement=700,
            cpu_threads=2,
            use_gpu=False,
            recommended_use_cases=["simple chat", "edge deployment", "testing"],
            performance_tier="fast",
            cost_tier="free"
        ),
        OllamaModel.LLAMA2_7B: ModelConfiguration(
            model=OllamaModel.LLAMA2_7B,
            context_length=4096,
            memory_requirement=8192,
            disk_requirement=3800,
            cpu_threads=4,
            use_gpu=True,
            recommended_use_cases=["general purpose", "reasoning", "content generation"],
            performance_tier="balanced",
            cost_tier="free"
        ),
        OllamaModel.CODELLAMA_7B: ModelConfiguration(
            model=OllamaModel.CODELLAMA_7B,
            context_length=16384,
            memory_requirement=8192,
            disk_requirement=3800,
            cpu_threads=4,
            use_gpu=True,
            recommended_use_cases=["code generation", "debugging", "technical writing"],
            performance_tier="balanced",
            cost_tier="free"
        ),
        OllamaModel.MIXTRAL_8X7B: ModelConfiguration(
            model=OllamaModel.MIXTRAL_8X7B,
            context_length=32768,
            memory_requirement=32768,
            disk_requirement=26000,
            cpu_threads=8,
            use_gpu=True,
            recommended_use_cases=["complex reasoning", "research", "high-quality content"],
            performance_tier="quality",
            cost_tier="free"
        ),
        OllamaModel.LLAVA: ModelConfiguration(
            model=OllamaModel.LLAVA,
            context_length=4096,
            memory_requirement=12288,
            disk_requirement=4500,
            cpu_threads=4,
            use_gpu=True,
            recommended_use_cases=["image analysis", "multimodal tasks", "visual QA"],
            performance_tier="balanced",
            cost_tier="free"
        )
    }
    
    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 auto_download: bool = True,
                 performance_monitoring: bool = True,
                 benchmark_models: bool = False):
        """
        Initialize enhanced Ollama client
        
        Args:
            base_url: Ollama server URL
            auto_download: Automatically download missing models
            performance_monitoring: Track performance metrics
            benchmark_models: Run performance benchmarks
        """
        self.base_url = base_url
        self.auto_download = auto_download
        self.performance_monitoring = performance_monitoring
        self.benchmark_models = benchmark_models
        
        # Usage tracking
        self.usage_stats = OllamaUsageStats()
        self.performance_metrics: Dict[str, OllamaPerformanceMetrics] = {}
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self._available_models: List[str] = []
    
    async def initialize(self):
        """Initialize client and discover available models"""
        if self._initialized:
            return
            
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)  # Local inference can be slow
        )
        
        # Test connection and get available models
        await self._discover_models()
        
        # Run benchmarks if requested
        if self.benchmark_models:
            await self._benchmark_available_models()
            
        self._initialized = True
        logger.info(f"Enhanced Ollama client initialized with {len(self._available_models)} models")
    
    async def close(self):
        """Close client session"""
        if self.session:
            await self.session.close()
        self._initialized = False
    
    async def _discover_models(self):
        """Discover available models on Ollama server"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self._available_models = [model["name"] for model in data.get("models", [])]
                    logger.info(f"Discovered {len(self._available_models)} available models")
                else:
                    logger.warning(f"Failed to discover models: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
    
    async def ensure_model_available(self, model: OllamaModel) -> bool:
        """Ensure model is downloaded and available"""
        model_name = model.value
        
        if model_name in self._available_models:
            return True
            
        if not self.auto_download:
            logger.warning(f"Model {model_name} not available and auto-download disabled")
            return False
        
        logger.info(f"Downloading model {model_name}...")
        try:
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name}
            ) as response:
                if response.status == 200:
                    # Stream download progress
                    async for line in response.content:
                        if line:
                            progress = json.loads(line.decode())
                            if "status" in progress:
                                logger.debug(f"Download progress: {progress['status']}")
                    
                    # Refresh model list
                    await self._discover_models()
                    return model_name in self._available_models
                else:
                    logger.error(f"Failed to download model {model_name}: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False
    
    async def generate(self,
                      model: OllamaModel,
                      prompt: str,
                      system: Optional[str] = None,
                      context: Optional[List[int]] = None,
                      stream: bool = False,
                      **kwargs) -> Dict[str, Any]:
        """
        Generate text using specified model
        
        Args:
            model: Model to use for generation
            prompt: Input prompt
            system: System prompt
            context: Conversation context
            stream: Stream response
            **kwargs: Additional generation parameters
        """
        if not self._initialized:
            await self.initialize()
        
        # Ensure model is available
        if not await self.ensure_model_available(model):
            raise RuntimeError(f"Model {model.value} not available")
        
        # Prepare request
        request_data = {
            "model": model.value,
            "prompt": prompt,
            "stream": stream
        }
        
        if system:
            request_data["system"] = system
        if context:
            request_data["context"] = context
            
        # Add additional parameters
        request_data.update(kwargs)
        
        # Track performance
        start_time = time.time()
        memory_before = await self._get_memory_usage()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=request_data
            ) as response:
                
                if response.status == 200:
                    if stream:
                        return await self._handle_streaming_response(response, model, start_time)
                    else:
                        data = await response.json()
                        return await self._handle_complete_response(data, model, start_time, memory_before)
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Generation failed: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            # Update failed request stats
            self.usage_stats.failed_requests += 1
            logger.error(f"Generation error: {e}")
            raise
    
    async def _handle_complete_response(self, 
                                       data: Dict[str, Any], 
                                       model: OllamaModel,
                                       start_time: float,
                                       memory_before: int) -> Dict[str, Any]:
        """Handle complete (non-streaming) response"""
        response_time = time.time() - start_time
        memory_after = await self._get_memory_usage()
        
        # Extract metrics
        response_text = data.get("response", "")
        tokens_generated = len(response_text.split())  # Rough estimate
        
        # Update usage stats
        self.usage_stats.total_requests += 1
        self.usage_stats.successful_requests += 1
        self.usage_stats.total_tokens_generated += tokens_generated
        self.usage_stats.total_inference_time += response_time
        
        if response_time > 0:
            tokens_per_second = tokens_generated / response_time
            self.usage_stats.average_tokens_per_second = (
                self.usage_stats.total_tokens_generated / self.usage_stats.total_inference_time
            )
        else:
            tokens_per_second = 0
        
        # Update model-specific usage
        model_name = model.value
        self.usage_stats.model_usage[model_name] = self.usage_stats.model_usage.get(model_name, 0) + 1
        
        # Track performance metrics
        if self.performance_monitoring:
            self.performance_metrics[model_name] = OllamaPerformanceMetrics(
                model=model_name,
                tokens_per_second=tokens_per_second,
                memory_usage=memory_after - memory_before,
                cpu_usage=0.0,  # Would need system monitoring
                gpu_usage=0.0,  # Would need GPU monitoring
                latency=response_time,
                quality_score=0.0,  # Would need quality assessment
                benchmark_date=time.time()
            )
        
        return {
            "response": response_text,
            "model": model_name,
            "context": data.get("context", []),
            "done": data.get("done", True),
            "performance": {
                "response_time": response_time,
                "tokens_per_second": tokens_per_second,
                "memory_usage": memory_after - memory_before
            }
        }
    
    async def _handle_streaming_response(self, 
                                        response: aiohttp.ClientResponse,
                                        model: OllamaModel,
                                        start_time: float) -> Dict[str, Any]:
        """Handle streaming response"""
        full_response = ""
        
        async for line in response.content:
            if line:
                try:
                    chunk = json.loads(line.decode())
                    if "response" in chunk:
                        full_response += chunk["response"]
                        
                    if chunk.get("done", False):
                        response_time = time.time() - start_time
                        tokens_generated = len(full_response.split())
                        
                        # Update stats
                        self.usage_stats.total_requests += 1
                        self.usage_stats.successful_requests += 1
                        self.usage_stats.total_tokens_generated += tokens_generated
                        
                        return {
                            "response": full_response,
                            "model": model.value,
                            "context": chunk.get("context", []),
                            "done": True,
                            "performance": {
                                "response_time": response_time,
                                "tokens_per_second": tokens_generated / response_time if response_time > 0 else 0
                            }
                        }
                except json.JSONDecodeError:
                    continue
        
        return {"response": full_response, "model": model.value, "done": True}
    
    async def _get_memory_usage(self) -> int:
        """Get current memory usage (simplified)"""
        # In a real implementation, this would use psutil or similar
        return 0
    
    async def _benchmark_available_models(self):
        """Benchmark all available models"""
        logger.info("Starting model benchmarking...")
        
        test_prompt = "Write a short explanation of machine learning."
        
        for model_name in self._available_models:
            try:
                # Find corresponding enum
                model_enum = None
                for model in OllamaModel:
                    if model.value == model_name:
                        model_enum = model
                        break
                
                if not model_enum:
                    continue
                    
                logger.info(f"Benchmarking {model_name}...")
                result = await self.generate(model_enum, test_prompt)
                
                logger.info(f"Benchmark complete for {model_name}: "
                          f"{result['performance']['tokens_per_second']:.2f} tokens/sec")
                          
            except Exception as e:
                logger.warning(f"Benchmark failed for {model_name}: {e}")
    
    def get_model_recommendations(self, 
                                 use_case: str,
                                 performance_tier: str = "balanced",
                                 memory_limit: Optional[int] = None) -> List[OllamaModel]:
        """
        Get model recommendations based on requirements
        
        Args:
            use_case: Type of task ("chat", "code", "reasoning", etc.)
            performance_tier: "fast", "balanced", or "quality"
            memory_limit: Maximum memory in MB
        
        Returns:
            List of recommended models
        """
        recommendations = []
        
        for model, config in self.MODEL_CONFIGS.items():
            # Check use case match
            if any(use_case.lower() in use.lower() for use in config.recommended_use_cases):
                # Check performance tier
                if config.performance_tier == performance_tier:
                    # Check memory limit
                    if memory_limit is None or config.memory_requirement <= memory_limit:
                        recommendations.append(model)
        
        return recommendations
    
    def get_model_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive model comparison"""
        comparison = {}
        
        for model, config in self.MODEL_CONFIGS.items():
            model_name = model.value
            
            comparison[model_name] = {
                "configuration": {
                    "context_length": config.context_length,
                    "memory_requirement": config.memory_requirement,
                    "disk_requirement": config.disk_requirement,
                    "performance_tier": config.performance_tier,
                    "cost_tier": config.cost_tier,
                    "use_cases": config.recommended_use_cases
                },
                "performance": self.performance_metrics.get(model_name),
                "usage": self.usage_stats.model_usage.get(model_name, 0),
                "available": model_name in self._available_models
            }
        
        return comparison
    
    def get_usage_statistics(self) -> OllamaUsageStats:
        """Get current usage statistics"""
        return self.usage_stats
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Integration with PRSM model routing
class PRSMOllamaIntegration:
    """
    Integration layer for Ollama with PRSM systems
    
    🔗 INTEGRATIONS:
    - Multi-provider routing and cost optimization
    - Performance comparison with cloud models
    - Local model caching and optimization
    - Resource usage monitoring
    """
    
    def __init__(self, ollama_client: EnhancedOllamaClient):
        self.ollama = ollama_client
    
    async def get_cost_comparison(self, prompt: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare costs between local and cloud models
        
        Returns cost analysis for decision making
        """
        # Local models are "free" but have infrastructure costs
        local_cost = {
            "financial_cost": 0.0,
            "infrastructure_cost": "electricity + hardware",
            "response_time": "variable",
            "privacy": "high",
            "reliability": "depends on hardware"
        }
        
        # Could integrate with cloud pricing APIs for comparison
        cloud_cost = {
            "financial_cost": "variable",
            "infrastructure_cost": "none",
            "response_time": "network dependent",
            "privacy": "depends on provider",
            "reliability": "high"
        }
        
        return {
            "local": local_cost,
            "cloud": cloud_cost,
            "recommendation": await self._recommend_deployment(prompt)
        }
    
    async def _recommend_deployment(self, prompt: str) -> str:
        """Recommend local vs cloud deployment based on prompt characteristics"""
        # Simple heuristics for demonstration
        if len(prompt) > 10000:
            return "cloud"  # Long prompts might need more powerful models
        elif "private" in prompt.lower() or "confidential" in prompt.lower():
            return "local"  # Privacy-sensitive content
        else:
            return "balanced"  # Use both based on availability and performance

# Example usage
async def example_usage():
    """Example of enhanced Ollama client usage"""
    
    async with EnhancedOllamaClient(
        auto_download=True,
        performance_monitoring=True,
        benchmark_models=True
    ) as ollama:
        
        # Get model recommendations
        code_models = ollama.get_model_recommendations("code", "balanced", memory_limit=16384)
        print(f"Recommended code models: {[m.value for m in code_models]}")
        
        # Generate with automatic model management
        result = await ollama.generate(
            OllamaModel.CODELLAMA_7B,
            "Write a Python function to calculate factorial"
        )
        
        print(f"Response: {result['response']}")
        print(f"Performance: {result['performance']}")
        
        # Get comprehensive comparison
        comparison = ollama.get_model_comparison()
        print(f"Available models: {len(comparison)}")

if __name__ == "__main__":
    asyncio.run(example_usage())
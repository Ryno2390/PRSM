"""
Real Marketplace Integration for Model Router
Connects to actual model APIs and marketplaces for dynamic model discovery
"""

import asyncio
import aiohttp
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import structlog

from prsm.core.config import get_settings
from prsm.core.redis_client import get_redis_client

logger = structlog.get_logger(__name__)
settings = get_settings()


class MarketplaceProvider(str, Enum):
    """Supported marketplace providers"""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    REPLICATE = "replicate"
    TOGETHER = "together"


@dataclass
class MarketplaceModel:
    """Model information from marketplace"""
    model_id: str
    name: str
    provider: MarketplaceProvider
    specialization: str
    performance_score: float
    cost_per_token: Optional[float]
    estimated_latency: Optional[float]
    provider_reputation: float
    marketplace_url: str
    capabilities: List[str]
    limitations: List[str]
    context_length: Optional[int]
    availability_score: float
    last_updated: datetime


class MarketplaceIntegration:
    """
    Real marketplace integration for dynamic model discovery
    Fetches live model data from various AI model marketplaces
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_ttl = 3600  # 1 hour cache
        self.redis_client = None
        self.rate_limits = {
            MarketplaceProvider.HUGGINGFACE: {"requests_per_hour": 1000, "last_reset": datetime.now()},
            MarketplaceProvider.OPENAI: {"requests_per_hour": 3000, "last_reset": datetime.now()},
            MarketplaceProvider.ANTHROPIC: {"requests_per_hour": 1000, "last_reset": datetime.now()},
        }
        
    async def initialize(self):
        """Initialize HTTP session and Redis connection"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": "PRSM-ModelRouter/1.0"}
            )
        
        if not self.redis_client:
            try:
                self.redis_client = get_redis_client()
            except Exception as e:
                logger.warning("Redis not available, caching disabled", error=str(e))
                
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            
    async def discover_marketplace_models(self, task_description: str, 
                                        limit: int = 10) -> List[MarketplaceModel]:
        """
        Discover models from all configured marketplaces
        
        Args:
            task_description: Description of task to find models for
            limit: Maximum number of models to return
            
        Returns:
            List of marketplace models suitable for the task
        """
        await self.initialize()
        
        # Check cache first
        cache_key = f"marketplace_models:{hashlib.sha256(task_description.encode()).hexdigest()}"
        cached_models = await self._get_cached_models(cache_key)
        if cached_models:
            logger.info("Using cached marketplace models", 
                       task=task_description[:50], count=len(cached_models))
            return cached_models[:limit]
        
        # Discover from all providers
        all_models = []
        providers_to_query = self._get_available_providers()
        
        # Run discovery in parallel
        tasks = []
        for provider in providers_to_query:
            if await self._check_rate_limit(provider):
                tasks.append(self._discover_from_provider(provider, task_description))
        
        if not tasks:
            logger.warning("No providers available due to rate limits")
            return []
        
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        for result in provider_results:
            if isinstance(result, list):
                all_models.extend(result)
            elif isinstance(result, Exception):
                logger.warning("Provider discovery failed", error=str(result))
        
        # Score and rank models
        scored_models = await self._score_and_rank_models(all_models, task_description)
        
        # Cache results
        await self._cache_models(cache_key, scored_models)
        
        logger.info("Discovered marketplace models", 
                   task=task_description[:50], 
                   total_found=len(scored_models),
                   returning=min(limit, len(scored_models)))
        
        return scored_models[:limit]
    
    def _get_available_providers(self) -> List[MarketplaceProvider]:
        """Get list of available providers based on configuration"""
        available = []
        
        if settings.openai_api_key:
            available.append(MarketplaceProvider.OPENAI)
        if settings.anthropic_api_key:
            available.append(MarketplaceProvider.ANTHROPIC)
        
        # Always include HuggingFace (public API)
        available.append(MarketplaceProvider.HUGGINGFACE)
        
        # Add other providers if configured
        if hasattr(settings, 'cohere_api_key') and settings.cohere_api_key:
            available.append(MarketplaceProvider.COHERE)
        if hasattr(settings, 'replicate_api_key') and settings.replicate_api_key:
            available.append(MarketplaceProvider.REPLICATE)
            
        return available
    
    async def _check_rate_limit(self, provider: MarketplaceProvider) -> bool:
        """Check if provider is within rate limits"""
        if provider not in self.rate_limits:
            return True
            
        limit_info = self.rate_limits[provider]
        now = datetime.now()
        
        # Reset if hour has passed
        if now - limit_info["last_reset"] > timedelta(hours=1):
            limit_info["requests_made"] = 0
            limit_info["last_reset"] = now
        
        requests_made = limit_info.get("requests_made", 0)
        requests_per_hour = limit_info["requests_per_hour"]
        
        if requests_made >= requests_per_hour:
            logger.warning("Rate limit exceeded for provider", provider=provider.value)
            return False
            
        limit_info["requests_made"] = requests_made + 1
        return True
    
    async def _discover_from_provider(self, provider: MarketplaceProvider, 
                                    task_description: str) -> List[MarketplaceModel]:
        """Discover models from specific provider"""
        try:
            if provider == MarketplaceProvider.HUGGINGFACE:
                return await self._discover_huggingface_models(task_description)
            elif provider == MarketplaceProvider.OPENAI:
                return await self._discover_openai_models(task_description)
            elif provider == MarketplaceProvider.ANTHROPIC:
                return await self._discover_anthropic_models(task_description)
            elif provider == MarketplaceProvider.COHERE:
                return await self._discover_cohere_models(task_description)
            else:
                logger.warning("Provider not implemented", provider=provider.value)
                return []
                
        except Exception as e:
            logger.error("Error discovering from provider", 
                        provider=provider.value, error=str(e))
            return []
    
    async def _discover_huggingface_models(self, task_description: str) -> List[MarketplaceModel]:
        """Discover models from HuggingFace Hub"""
        models = []
        
        # Determine search query based on task
        search_query = self._extract_search_terms(task_description)
        
        try:
            # Search HuggingFace models API
            url = "https://huggingface.co/api/models"
            params = {
                "search": search_query,
                "limit": 20,
                "filter": "text-generation,text-classification,question-answering",
                "sort": "downloads"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for model_data in data:
                        # Extract model information
                        model_id = model_data.get("modelId", "")
                        downloads = model_data.get("downloads", 0)
                        
                        # Calculate scores based on popularity and task relevance
                        performance_score = min(downloads / 100000.0, 1.0)  # Normalize downloads
                        availability_score = 0.9  # HF models generally available
                        
                        # Estimate cost (HF Inference API pricing)
                        cost_per_token = 0.0002  # Approximate HF pricing
                        
                        model = MarketplaceModel(
                            model_id=model_id,
                            name=model_data.get("name", model_id),
                            provider=MarketplaceProvider.HUGGINGFACE,
                            specialization=self._infer_specialization(model_id, model_data),
                            performance_score=performance_score,
                            cost_per_token=cost_per_token,
                            estimated_latency=2.0,  # Typical HF API latency
                            provider_reputation=0.85,
                            marketplace_url=f"https://huggingface.co/{model_id}",
                            capabilities=self._extract_capabilities(model_data),
                            limitations=["Rate limited", "May require authentication"],
                            context_length=self._extract_context_length(model_data),
                            availability_score=availability_score,
                            last_updated=datetime.now(timezone.utc)
                        )
                        models.append(model)
                        
        except Exception as e:
            logger.error("Error fetching HuggingFace models", error=str(e))
            
        return models
    
    async def _discover_openai_models(self, task_description: str) -> List[MarketplaceModel]:
        """Discover models from OpenAI API"""
        models = []
        
        if not settings.openai_api_key:
            return models
            
        try:
            headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
            url = "https://api.openai.com/v1/models"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for model_data in data.get("data", []):
                        model_id = model_data.get("id", "")
                        
                        # Only include chat/completion models
                        if any(prefix in model_id for prefix in ["gpt-", "text-davinci", "claude"]):
                            # Determine pricing based on model
                            cost_per_token = self._get_openai_pricing(model_id)
                            performance_score = self._get_openai_performance(model_id)
                            
                            model = MarketplaceModel(
                                model_id=model_id,
                                name=model_data.get("name", model_id),
                                provider=MarketplaceProvider.OPENAI,
                                specialization=self._infer_openai_specialization(model_id),
                                performance_score=performance_score,
                                cost_per_token=cost_per_token,
                                estimated_latency=1.5,
                                provider_reputation=0.95,
                                marketplace_url="https://api.openai.com/v1/chat/completions",
                                capabilities=["Chat", "Completion", "Function calling"],
                                limitations=["Rate limited", "Paid API"],
                                context_length=self._get_openai_context_length(model_id),
                                availability_score=0.98,
                                last_updated=datetime.now(timezone.utc)
                            )
                            models.append(model)
                            
        except Exception as e:
            logger.error("Error fetching OpenAI models", error=str(e))
            
        return models
    
    async def _discover_anthropic_models(self, task_description: str) -> List[MarketplaceModel]:
        """Discover models from Anthropic"""
        models = []
        
        if not settings.anthropic_api_key:
            return models
        
        # Anthropic doesn't have a models API, so we'll use known models
        claude_models = [
            {
                "model_id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3.5 Sonnet",
                "performance_score": 0.95,
                "cost_per_token": 0.015,
                "context_length": 200000,
                "specialization": "reasoning"
            },
            {
                "model_id": "claude-3-haiku-20240307", 
                "name": "Claude 3 Haiku",
                "performance_score": 0.85,
                "cost_per_token": 0.001,
                "context_length": 200000,
                "specialization": "general"
            }
        ]
        
        for model_data in claude_models:
            model = MarketplaceModel(
                model_id=model_data["model_id"],
                name=model_data["name"],
                provider=MarketplaceProvider.ANTHROPIC,
                specialization=model_data["specialization"],
                performance_score=model_data["performance_score"],
                cost_per_token=model_data["cost_per_token"],
                estimated_latency=1.8,
                provider_reputation=0.92,
                marketplace_url="https://api.anthropic.com/v1/messages",
                capabilities=["Chat", "Analysis", "Code generation"],
                limitations=["Rate limited", "Paid API"],
                context_length=model_data["context_length"],
                availability_score=0.96,
                last_updated=datetime.now(timezone.utc)
            )
            models.append(model)
        
        return models
    
    async def _discover_cohere_models(self, task_description: str) -> List[MarketplaceModel]:
        """Discover models from Cohere"""
        models = []
        
        # Placeholder for Cohere integration
        # Would implement similar to OpenAI with their models API
        
        return models
    
    def _extract_search_terms(self, task_description: str) -> str:
        """Extract relevant search terms from task description"""
        task_lower = task_description.lower()
        
        if "code" in task_lower or "programming" in task_lower:
            return "code generation"
        elif "chat" in task_lower or "conversation" in task_lower:
            return "conversational"
        elif "analyze" in task_lower or "analysis" in task_lower:
            return "text analysis"
        elif "question" in task_lower or "answer" in task_lower:
            return "question answering"
        elif "translate" in task_lower:
            return "translation"
        else:
            return "text generation"
    
    def _infer_specialization(self, model_id: str, model_data: Dict) -> str:
        """Infer model specialization from metadata"""
        model_id_lower = model_id.lower()
        
        if "code" in model_id_lower:
            return "code_generation"
        elif "chat" in model_id_lower:
            return "conversation"
        elif "instruct" in model_id_lower:
            return "instruction_following"
        elif "math" in model_id_lower:
            return "mathematics"
        else:
            return "general"
    
    def _extract_capabilities(self, model_data: Dict) -> List[str]:
        """Extract model capabilities from metadata"""
        capabilities = ["text_generation"]
        
        # Extract from tags or description
        tags = model_data.get("tags", [])
        for tag in tags:
            if tag in ["conversational", "code", "math", "multilingual"]:
                capabilities.append(tag)
        
        return capabilities
    
    def _extract_context_length(self, model_data: Dict) -> Optional[int]:
        """Extract context length from model metadata"""
        # Try to extract from config or model card
        config = model_data.get("config", {})
        
        # Common context length fields
        for field in ["max_position_embeddings", "n_positions", "seq_length"]:
            if field in config:
                return config[field]
        
        # Default based on model type
        return 4096
    
    def _get_openai_pricing(self, model_id: str) -> float:
        """Get OpenAI model pricing per token"""
        pricing_map = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
            "text-davinci-003": 0.02,
        }
        
        for model_prefix, price in pricing_map.items():
            if model_id.startswith(model_prefix):
                return price
        
        return 0.002  # Default
    
    def _get_openai_performance(self, model_id: str) -> float:
        """Get estimated performance score for OpenAI models"""
        if "gpt-4" in model_id:
            return 0.95
        elif "gpt-3.5" in model_id:
            return 0.85
        elif "davinci" in model_id:
            return 0.90
        else:
            return 0.75
    
    def _infer_openai_specialization(self, model_id: str) -> str:
        """Infer OpenAI model specialization"""
        if "code" in model_id:
            return "code_generation"
        elif "instruct" in model_id:
            return "instruction_following"
        else:
            return "general"
    
    def _get_openai_context_length(self, model_id: str) -> int:
        """Get OpenAI model context length"""
        if "gpt-4-turbo" in model_id:
            return 128000
        elif "gpt-4" in model_id:
            return 8192
        elif "gpt-3.5-turbo" in model_id:
            return 4096
        else:
            return 4096
    
    async def _score_and_rank_models(self, models: List[MarketplaceModel], 
                                   task_description: str) -> List[MarketplaceModel]:
        """Score and rank models based on task relevance"""
        # Calculate task relevance score for each model
        for model in models:
            relevance_score = await self._calculate_task_relevance(model, task_description)
            
            # Combine multiple factors for final score
            model.performance_score = (
                model.performance_score * 0.4 +
                relevance_score * 0.3 +
                model.availability_score * 0.2 +
                model.provider_reputation * 0.1
            )
        
        # Sort by performance score
        models.sort(key=lambda m: m.performance_score, reverse=True)
        
        return models
    
    async def _calculate_task_relevance(self, model: MarketplaceModel, 
                                      task_description: str) -> float:
        """Calculate how relevant a model is to the task"""
        relevance = 0.5  # Base relevance
        
        task_lower = task_description.lower()
        specialization = model.specialization.lower()
        
        # Direct specialization match
        if specialization in task_lower or any(word in specialization for word in task_lower.split()):
            relevance += 0.3
        
        # Capability matching
        for capability in model.capabilities:
            if capability.lower() in task_lower:
                relevance += 0.1
        
        # Model name/ID matching
        if any(word in model.model_id.lower() for word in task_lower.split()):
            relevance += 0.1
        
        return min(relevance, 1.0)
    
    async def _get_cached_models(self, cache_key: str) -> Optional[List[MarketplaceModel]]:
        """Get cached models from Redis"""
        if not self.redis_client:
            return None
            
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                # Would deserialize from JSON/pickle
                # For now, return None to always fetch fresh
                pass
        except Exception as e:
            logger.warning("Cache read error", error=str(e))
            
        return None
    
    async def _cache_models(self, cache_key: str, models: List[MarketplaceModel]):
        """Cache models to Redis"""
        if not self.redis_client:
            return
            
        try:
            # Would serialize models to JSON/pickle
            # For now, skip caching due to serialization complexity
            pass
        except Exception as e:
            logger.warning("Cache write error", error=str(e))


# Global instance
marketplace_integration = MarketplaceIntegration()
#!/usr/bin/env python3
"""
DEPRECATED: This module is deprecated.

Use prsm.economy.marketplace.expanded_models for marketplace data models.
Use prsm.node.compute_provider for external model integration.

This module is kept for backward compatibility only.

---
Real Marketplace Service

Provides integration with external AI model marketplaces and federated
model discovery services for the PRSM ecosystem.

Core Functions:
- External marketplace integration (Hugging Face, OpenAI, etc.)
- Model discovery and capability assessment
- Dynamic pricing and availability checking
- Marketplace transaction management
"""

import structlog
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4
import asyncio

logger = structlog.get_logger(__name__)


class MarketplaceProvider(Enum):
    """External marketplace providers"""
    HUGGING_FACE = "hugging_face"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    REPLICATE = "replicate"
    TOGETHER_AI = "together_ai"
    RUNPOD = "runpod"
    MODAL = "modal"


class ModelCapability(Enum):
    """AI model capabilities"""
    TEXT_GENERATION = "text_generation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    FINE_TUNING = "fine_tuning"
    EMBEDDINGS = "embeddings"


@dataclass
class MarketplaceModel:
    """Model information from marketplace"""
    model_id: str
    model_name: str
    provider: MarketplaceProvider
    capabilities: List[ModelCapability]
    pricing: Dict[str, Decimal]  # e.g., {'per_token': 0.001}
    performance_metrics: Dict[str, float]
    availability: bool
    api_endpoint: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketplaceQuery:
    """Query for marketplace model discovery"""
    required_capabilities: List[ModelCapability]
    max_price_per_token: Optional[Decimal] = None
    min_performance_score: Optional[float] = None
    preferred_providers: Optional[List[MarketplaceProvider]] = None
    context_length_min: Optional[int] = None
    multimodal_support: bool = False


@dataclass
class ResourceListing:
    """Generic resource listing for marketplace resources"""
    id: UUID
    resource_type: str
    name: str
    description: str
    owner_user_id: UUID
    status: str
    quality_grade: str
    pricing_model: str
    base_price: Decimal
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealMarketplaceService:
    """
    Real Marketplace Service for external AI model integration
    
    Integrates with external marketplaces to discover, evaluate,
    and utilize AI models for PRSM operations.
    """
    
    def __init__(self):
        """Initialize marketplace service"""
        from decimal import Decimal
        
        # Platform configuration
        self.platform_fee_percentage = Decimal('0.025')  # 2.5% platform fee
        self.db_service = None  # Will be set when database is available
        
        # Quality and pricing configuration
        self.quality_boost_multipliers = {
            'verified': Decimal('1.2'),
            'popular': Decimal('1.1'),
            'premium': Decimal('1.5')
        }
        self.pricing_models = {
            'token_based': 'per_1k_tokens',
            'request_based': 'per_request',
            'subscription': 'monthly'
        }
        
        self.provider_configs = {
            MarketplaceProvider.HUGGING_FACE: {
                'base_url': 'https://api-inference.huggingface.co',
                'auth_header': 'Authorization',
                'rate_limit': 100  # requests per minute
            },
            MarketplaceProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1',
                'auth_header': 'Authorization',
                'rate_limit': 3500  # tokens per minute
            },
            MarketplaceProvider.ANTHROPIC: {
                'base_url': 'https://api.anthropic.com/v1',
                'auth_header': 'x-api-key',
                'rate_limit': 4000  # tokens per minute
            }
        }
        
        # Cache for model discovery results
        self.model_cache: Dict[str, MarketplaceModel] = {}
        self.last_cache_update = datetime.now(timezone.utc)
        
        # Resource listings storage
        self._resource_listings: Dict[UUID, Any] = {}
        
        # Mock model registry for testing
        self._populate_mock_models()
        
        logger.info("RealMarketplaceService initialized", providers=len(self.provider_configs))
    
    def _populate_mock_models(self):
        """Populate mock models for testing and development"""
        mock_models = [
            MarketplaceModel(
                model_id="gpt-3.5-turbo",
                model_name="GPT-3.5 Turbo",
                provider=MarketplaceProvider.OPENAI,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.CODE_GENERATION],
                pricing={'per_token': Decimal('0.002')},
                performance_metrics={'quality_score': 0.85, 'speed_score': 0.9},
                availability=True,
                api_endpoint="https://api.openai.com/v1/chat/completions",
                metadata={'context_length': 4096, 'max_tokens': 2048}
            ),
            MarketplaceModel(
                model_id="claude-3-sonnet",
                model_name="Claude 3 Sonnet",
                provider=MarketplaceProvider.ANTHROPIC,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.MULTIMODAL],
                pricing={'per_token': Decimal('0.003')},
                performance_metrics={'quality_score': 0.92, 'speed_score': 0.8},
                availability=True,
                api_endpoint="https://api.anthropic.com/v1/messages",
                metadata={'context_length': 200000, 'max_tokens': 4096}
            ),
            MarketplaceModel(
                model_id="llama-2-70b-chat",
                model_name="Llama 2 70B Chat",
                provider=MarketplaceProvider.HUGGING_FACE,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.QUESTION_ANSWERING],
                pricing={'per_token': Decimal('0.001')},
                performance_metrics={'quality_score': 0.78, 'speed_score': 0.7},
                availability=True,
                api_endpoint="https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf",
                metadata={'context_length': 4096, 'max_tokens': 2048}
            ),
            MarketplaceModel(
                model_id="code-llama-34b",
                model_name="Code Llama 34B",
                provider=MarketplaceProvider.HUGGING_FACE,
                capabilities=[ModelCapability.CODE_GENERATION, ModelCapability.TEXT_GENERATION],
                pricing={'per_token': Decimal('0.0015')},
                performance_metrics={'quality_score': 0.82, 'speed_score': 0.75},
                availability=True,
                api_endpoint="https://api-inference.huggingface.co/models/codellama/CodeLlama-34b-Instruct-hf",
                metadata={'context_length': 16384, 'max_tokens': 4096}
            )
        ]
        
        for model in mock_models:
            self.model_cache[model.model_id] = model
    
    async def discover_models(self, query: MarketplaceQuery) -> List[MarketplaceModel]:
        """Discover models matching query criteria"""
        try:
            logger.info("Discovering marketplace models",
                       required_capabilities=[c.value for c in query.required_capabilities],
                       max_price=float(query.max_price_per_token) if query.max_price_per_token else None)
            
            # Filter models based on query criteria
            matching_models = []
            
            for model in self.model_cache.values():
                # Check capabilities
                if not all(cap in model.capabilities for cap in query.required_capabilities):
                    continue
                
                # Check pricing
                if query.max_price_per_token:
                    model_price = model.pricing.get('per_token', Decimal('999'))
                    if model_price > query.max_price_per_token:
                        continue
                
                # Check performance
                if query.min_performance_score:
                    quality_score = model.performance_metrics.get('quality_score', 0.0)
                    if quality_score < query.min_performance_score:
                        continue
                
                # Check provider preference
                if query.preferred_providers and model.provider not in query.preferred_providers:
                    continue
                
                # Check context length
                if query.context_length_min:
                    context_length = model.metadata.get('context_length', 0)
                    if context_length < query.context_length_min:
                        continue
                
                # Check multimodal support
                if query.multimodal_support and ModelCapability.MULTIMODAL not in model.capabilities:
                    continue
                
                # Check availability
                if not model.availability:
                    continue
                
                matching_models.append(model)
            
            # Sort by quality score descending
            matching_models.sort(
                key=lambda m: m.performance_metrics.get('quality_score', 0.0),
                reverse=True
            )
            
            logger.info("Model discovery completed",
                       total_models=len(self.model_cache),
                       matching_models=len(matching_models))
            
            return matching_models
            
        except Exception as e:
            logger.error(f"Failed to discover models: {e}")
            return []
    
    async def get_model_info(self, model_id: str) -> Optional[MarketplaceModel]:
        """Get detailed information about a specific model"""
        return self.model_cache.get(model_id)
    
    async def check_model_availability(self, model_id: str) -> Tuple[bool, str]:
        """Check if model is currently available"""
        model = self.model_cache.get(model_id)
        if not model:
            return False, "Model not found"
        
        # Mock availability check - in real implementation, this would ping the API
        if model.availability:
            return True, "Model available"
        else:
            return False, "Model temporarily unavailable"
    
    async def estimate_request_cost(self,
                                  model_id: str,
                                  input_tokens: int,
                                  max_output_tokens: int) -> Optional[Decimal]:
        """Estimate cost for a model request"""
        model = self.model_cache.get(model_id)
        if not model:
            return None
        
        per_token_cost = model.pricing.get('per_token', Decimal('0.001'))
        total_tokens = input_tokens + max_output_tokens
        
        return per_token_cost * Decimal(str(total_tokens))
    
    async def query_model(self,
                         model_id: str,
                         prompt: str,
                         max_tokens: int = 1024,
                         temperature: float = 0.7) -> Dict[str, Any]:
        """Query a marketplace model (mock implementation)"""
        try:
            model = self.model_cache.get(model_id)
            if not model:
                return {'error': 'Model not found'}
            
            if not model.availability:
                return {'error': 'Model not available'}
            
            # Mock response - in real implementation, this would make HTTP requests
            logger.info("Querying marketplace model",
                       model_id=model_id,
                       prompt_length=len(prompt),
                       max_tokens=max_tokens)
            
            # Simulate API delay
            await asyncio.sleep(0.1)
            
            mock_response = f"""Based on the query "{prompt[:50]}...", this is a mock response from {model.model_name}. 
            
This model has the following capabilities: {', '.join([c.value for c in model.capabilities])}.
            
Key insights:
            - The model demonstrates {model.performance_metrics.get('quality_score', 0.8):.1%} quality performance
            - Response time is optimized at {model.performance_metrics.get('speed_score', 0.8):.1%} efficiency
            - Context length supports up to {model.metadata.get('context_length', 4096)} tokens
            
This response showcases the model's reasoning and generation capabilities within the PRSM ecosystem."""
            
            return {
                'response': mock_response,
                'model_id': model_id,
                'tokens_used': len(mock_response.split()),
                'cost_estimate': float(model.pricing.get('per_token', Decimal('0.001')) * len(mock_response.split())),
                'provider': model.provider.value,
                'quality_score': model.performance_metrics.get('quality_score', 0.8)
            }
            
        except Exception as e:
            logger.error(f"Failed to query model {model_id}: {e}")
            return {'error': str(e)}
    
    def get_supported_providers(self) -> List[MarketplaceProvider]:
        """Get list of supported marketplace providers"""
        return list(self.provider_configs.keys())
    
    def get_provider_models(self, provider: MarketplaceProvider) -> List[MarketplaceModel]:
        """Get all models from a specific provider"""
        return [
            model for model in self.model_cache.values()
            if model.provider == provider
        ]
    
    async def refresh_model_cache(self) -> int:
        """Refresh model cache from marketplace APIs"""
        try:
            logger.info("Refreshing marketplace model cache")
            
            # In real implementation, this would query each provider's API
            # For now, we'll just update the timestamp
            self.last_cache_update = datetime.now(timezone.utc)
            
            # Mock: Add a new model to demonstrate cache refresh
            new_model = MarketplaceModel(
                model_id=f"refreshed-model-{int(datetime.now().timestamp())}",
                model_name="Refreshed Test Model",
                provider=MarketplaceProvider.HUGGING_FACE,
                capabilities=[ModelCapability.TEXT_GENERATION],
                pricing={'per_token': Decimal('0.0005')},
                performance_metrics={'quality_score': 0.75, 'speed_score': 0.85},
                availability=True,
                api_endpoint="https://api-inference.huggingface.co/models/test",
                metadata={'cache_refresh': True}
            )
            
            self.model_cache[new_model.model_id] = new_model
            
            logger.info("Model cache refreshed", total_models=len(self.model_cache))
            return len(self.model_cache)
            
        except Exception as e:
            logger.error(f"Failed to refresh model cache: {e}")
            return 0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get marketplace cache statistics"""
        provider_counts = {}
        capability_counts = {}
        total_models = len(self.model_cache)
        
        for model in self.model_cache.values():
            # Count by provider
            provider = model.provider.value
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            
            # Count by capability
            for capability in model.capabilities:
                cap_name = capability.value
                capability_counts[cap_name] = capability_counts.get(cap_name, 0) + 1
        
        available_models = sum(1 for model in self.model_cache.values() if model.availability)
        
        return {
            'total_models': total_models,
            'available_models': available_models,
            'availability_percentage': (available_models / max(total_models, 1)) * 100,
            'providers': provider_counts,
            'capabilities': capability_counts,
            'last_cache_update': self.last_cache_update.isoformat(),
            'supported_providers': len(self.provider_configs)
        }
    
    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics
        
        Returns:
            Dictionary containing marketplace statistics
        """
        cache_stats = self.get_cache_statistics()
        
        return {
            'platform': {
                'fee_percentage': str(self.platform_fee_percentage),
                'supported_providers': len(self.provider_configs),
                'quality_tiers': len(self.quality_boost_multipliers),
                'pricing_models': len(self.pricing_models)
            },
            'models': {
                'total': cache_stats['total_models'],
                'available': cache_stats['available_models'],
                'by_provider': cache_stats['providers'],
                'by_capability': cache_stats['capabilities']
            },
            'cache': {
                'last_update': cache_stats['last_cache_update'],
                'availability_percentage': cache_stats['availability_percentage']
            }
        }
    
    async def create_resource_listing(
        self,
        resource_type: str,
        name: str,
        description: str,
        owner_user_id: Union[str, UUID],
        specific_data: Optional[Dict[str, Any]] = None,
        pricing_model: str = "free",
        base_price: float = 0.0,
        tags: Optional[List[str]] = None,
        quality_grade: str = "community",
        **kwargs
    ) -> UUID:
        """Create a universal resource listing in the marketplace"""
        try:
            resource_id = uuid4()
            owner_id = UUID(str(owner_user_id)) if isinstance(owner_user_id, str) else owner_user_id
            
            resource = ResourceListing(
                id=resource_id,
                resource_type=resource_type,
                name=name,
                description=description,
                owner_user_id=owner_id,
                status="active",
                quality_grade=quality_grade,
                pricing_model=pricing_model,
                base_price=Decimal(str(base_price)),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=tags or [],
                metadata=specific_data or {}
            )
            
            self._resource_listings[resource_id] = resource
            
            logger.info("Created resource listing",
                       resource_id=str(resource_id),
                       resource_type=resource_type,
                       name=name)
            
            return resource_id
            
        except Exception as e:
            logger.error(f"Failed to create resource listing: {e}")
            raise
    
    async def create_ai_model_listing(
        self,
        request: "CreateModelListingRequest",
        owner_user_id: Union[str, UUID]
    ) -> ResourceListing:
        """Create an AI model listing in the marketplace"""
        from .models import CreateModelListingRequest
        
        try:
            resource_id = uuid4()
            owner_id = UUID(str(owner_user_id)) if isinstance(owner_user_id, str) else owner_user_id
            
            resource = ResourceListing(
                id=resource_id,
                resource_type="ai_model",
                name=request.name,
                description=request.description,
                owner_user_id=owner_id,
                status="active",
                quality_grade="community",
                pricing_model=request.pricing_tier.value if hasattr(request.pricing_tier, 'value') else str(request.pricing_tier),
                base_price=request.base_price or Decimal('0'),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=request.tags or [],
                metadata={
                    "model_id": request.model_id,
                    "provider": request.provider.value if hasattr(request.provider, 'value') else str(request.provider),
                    "category": request.category.value if hasattr(request.category, 'value') else str(request.category),
                    "provider_name": request.provider_name,
                    "model_version": request.model_version,
                    "context_length": request.context_length,
                    "max_tokens": request.max_tokens,
                    "input_modalities": request.input_modalities,
                    "output_modalities": request.output_modalities,
                    "languages_supported": request.languages_supported,
                    "api_endpoint": request.api_endpoint,
                    "documentation_url": request.documentation_url,
                    "license_type": request.license_type
                }
            )
            
            self._resource_listings[resource_id] = resource
            
            logger.info("Created AI model listing",
                       resource_id=str(resource_id),
                       model_id=request.model_id,
                       name=request.name)
            
            return resource
            
        except Exception as e:
            logger.error(f"Failed to create AI model listing: {e}")
            raise
    
    async def create_dataset_listing(
        self,
        name: str,
        description: str,
        category: str,
        size_bytes: int,
        record_count: int,
        data_format: str,
        owner_user_id: Union[str, UUID],
        quality_grade: str = "community",
        pricing_model: str = "free",
        base_price: float = 0.0,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> UUID:
        """Create a dataset listing in the marketplace"""
        try:
            resource_id = uuid4()
            owner_id = UUID(str(owner_user_id)) if isinstance(owner_user_id, str) else owner_user_id
            
            resource = ResourceListing(
                id=resource_id,
                resource_type="dataset",
                name=name,
                description=description,
                owner_user_id=owner_id,
                status="active",
                quality_grade=quality_grade,
                pricing_model=pricing_model,
                base_price=Decimal(str(base_price)),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=tags or [],
                metadata={
                    "category": category,
                    "size_bytes": size_bytes,
                    "record_count": record_count,
                    "data_format": data_format,
                    **kwargs
                }
            )
            
            self._resource_listings[resource_id] = resource
            
            logger.info("Created dataset listing",
                       resource_id=str(resource_id),
                       name=name,
                       category=category)
            
            return resource_id
            
        except Exception as e:
            logger.error(f"Failed to create dataset listing: {e}")
            raise
    
    async def create_agent_listing(
        self,
        name: str,
        description: str,
        agent_type: str,
        capabilities: List[str],
        required_models: List[str],
        owner_user_id: Union[str, UUID],
        quality_grade: str = "community",
        pricing_model: str = "pay_per_use",
        base_price: float = 0.0,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> UUID:
        """Create an AI agent listing in the marketplace"""
        try:
            resource_id = uuid4()
            owner_id = UUID(str(owner_user_id)) if isinstance(owner_user_id, str) else owner_user_id
            
            resource = ResourceListing(
                id=resource_id,
                resource_type="agent_workflow",
                name=name,
                description=description,
                owner_user_id=owner_id,
                status="active",
                quality_grade=quality_grade,
                pricing_model=pricing_model,
                base_price=Decimal(str(base_price)),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=tags or [],
                metadata={
                    "agent_type": agent_type,
                    "capabilities": capabilities,
                    "required_models": required_models,
                    **kwargs
                }
            )
            
            self._resource_listings[resource_id] = resource
            
            logger.info("Created agent listing",
                       resource_id=str(resource_id),
                       name=name,
                       agent_type=agent_type)
            
            return resource_id
            
        except Exception as e:
            logger.error(f"Failed to create agent listing: {e}")
            raise
    
    async def create_tool_listing(
        self,
        name: str,
        description: str,
        tool_category: str,
        functions_provided: List[Dict[str, str]],
        owner_user_id: Union[str, UUID],
        quality_grade: str = "community",
        pricing_model: str = "free",
        base_price: float = 0.0,
        installation_method: Optional[str] = None,
        package_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> UUID:
        """Create an MCP tool listing in the marketplace"""
        try:
            resource_id = uuid4()
            owner_id = UUID(str(owner_user_id)) if isinstance(owner_user_id, str) else owner_user_id
            
            resource = ResourceListing(
                id=resource_id,
                resource_type="mcp_tool",
                name=name,
                description=description,
                owner_user_id=owner_id,
                status="active",
                quality_grade=quality_grade,
                pricing_model=pricing_model,
                base_price=Decimal(str(base_price)),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=tags or [],
                metadata={
                    "tool_category": tool_category,
                    "functions_provided": functions_provided,
                    "installation_method": installation_method,
                    "package_name": package_name,
                    **kwargs
                }
            )
            
            self._resource_listings[resource_id] = resource
            
            logger.info("Created tool listing",
                       resource_id=str(resource_id),
                       name=name,
                       tool_category=tool_category)
            
            return resource_id
            
        except Exception as e:
            logger.error(f"Failed to create tool listing: {e}")
            raise
    
    async def search_resources(
        self,
        search_query: Optional[str] = None,
        resource_types: Optional[List[str]] = None,
        pricing_models: Optional[List[str]] = None,
        quality_grades: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        verified_only: bool = False,
        featured_only: bool = False,
        tags: Optional[List[str]] = None,
        sort_by: str = "popularity",
        sort_order: str = "desc",
        limit: int = 20,
        offset: int = 0,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Search across marketplace resources with filters"""
        try:
            results = []
            
            for resource_id, resource in self._resource_listings.items():
                if resource_types and resource.resource_type not in resource_types:
                    continue
                
                if search_query:
                    query_lower = search_query.lower()
                    if (query_lower not in resource.name.lower() and 
                        query_lower not in resource.description.lower() and
                        not any(query_lower in tag.lower() for tag in resource.tags)):
                        continue
                
                if pricing_models and resource.pricing_model not in pricing_models:
                    continue
                
                if quality_grades and resource.quality_grade not in quality_grades:
                    continue
                
                if min_price is not None and float(resource.base_price) < min_price:
                    continue
                
                if max_price is not None and float(resource.base_price) > max_price:
                    continue
                
                results.append({
                    "id": str(resource.id),
                    "resource_type": resource.resource_type,
                    "name": resource.name,
                    "description": resource.description,
                    "owner_user_id": str(resource.owner_user_id),
                    "status": resource.status,
                    "quality_grade": resource.quality_grade,
                    "pricing_model": resource.pricing_model,
                    "base_price": float(resource.base_price),
                    "tags": resource.tags,
                    "created_at": resource.created_at.isoformat(),
                    "updated_at": resource.updated_at.isoformat(),
                    "metadata": resource.metadata
                })
            
            total_count = len(results)
            
            if sort_by == "popularity":
                results.sort(key=lambda x: x.get("metadata", {}).get("download_count", 0), reverse=(sort_order == "desc"))
            elif sort_by == "price":
                results.sort(key=lambda x: x["base_price"], reverse=(sort_order == "desc"))
            elif sort_by == "created_at":
                results.sort(key=lambda x: x["created_at"], reverse=(sort_order == "desc"))
            elif sort_by == "name":
                results.sort(key=lambda x: x["name"].lower(), reverse=(sort_order == "desc"))
            elif sort_by == "rating":
                results.sort(key=lambda x: x.get("metadata", {}).get("rating_average", 0), reverse=(sort_order == "desc"))
            
            paginated = results[offset:offset + limit]
            
            logger.info("Searched resources",
                       query=search_query,
                       total_results=total_count,
                       returned=len(paginated))
            
            return paginated, total_count
            
        except Exception as e:
            logger.error(f"Failed to search resources: {e}")
            return [], 0
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive marketplace statistics"""
        try:
            resource_counts = {
                "total": len(self._resource_listings),
                "ai_models": 0,
                "datasets": 0,
                "agents": 0,
                "tools": 0,
                "compute_resources": 0,
                "knowledge_resources": 0,
                "evaluation_services": 0,
                "training_services": 0,
                "safety_tools": 0
            }
            
            total_revenue = Decimal('0')
            quality_distribution = {}
            pricing_distribution = {}
            
            for resource in self._resource_listings.values():
                resource_counts[resource.resource_type] = resource_counts.get(resource.resource_type, 0) + 1
                total_revenue += resource.base_price
                
                quality_grade = resource.quality_grade
                quality_distribution[quality_grade] = quality_distribution.get(quality_grade, 0) + 1
                
                pricing_model = resource.pricing_model
                pricing_distribution[pricing_model] = pricing_distribution.get(pricing_model, 0) + 1
            
            top_downloads = sorted(
                [
                    {
                        "id": str(r.id),
                        "name": r.name,
                        "resource_type": r.resource_type,
                        "download_count": r.metadata.get("download_count", 0)
                    }
                    for r in self._resource_listings.values()
                ],
                key=lambda x: x["download_count"],
                reverse=True
            )[:10]
            
            stats = {
                "resource_counts": resource_counts,
                "revenue_stats": {
                    "total_revenue": float(total_revenue),
                    "monthly_revenue": float(total_revenue * Decimal('0.1')),
                    "avg_transaction_value": float(total_revenue / max(len(self._resource_listings), 1))
                },
                "quality_distribution": quality_distribution,
                "pricing_distribution": pricing_distribution,
                "top_downloads": top_downloads,
                "growth_trend": {
                    "resources_this_week": len(self._resource_listings),
                    "growth_rate": 0.0
                },
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info("Retrieved comprehensive stats", total_resources=resource_counts["total"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive stats: {e}")
            return {
                "resource_counts": {"total": 0},
                "revenue_stats": {"total_revenue": 0},
                "quality_distribution": {},
                "top_downloads": [],
                "growth_trend": {},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_resource_details(self, resource_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """Get details of a specific resource"""
        try:
            rid = UUID(str(resource_id)) if isinstance(resource_id, str) else resource_id
            resource = self._resource_listings.get(rid)
            
            if not resource:
                raise ValueError(f"Resource not found: {resource_id}")
            
            return {
                "id": str(resource.id),
                "resource_type": resource.resource_type,
                "name": resource.name,
                "description": resource.description,
                "owner_user_id": str(resource.owner_user_id),
                "status": resource.status,
                "quality_grade": resource.quality_grade,
                "pricing_model": resource.pricing_model,
                "base_price": float(resource.base_price),
                "tags": resource.tags,
                "created_at": resource.created_at.isoformat(),
                "updated_at": resource.updated_at.isoformat(),
                "metadata": resource.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource details: {e}")
            raise
    
    async def create_order(
        self,
        resource_id: Union[str, UUID],
        buyer_user_id: Union[str, UUID],
        order_type: str = "purchase",
        quantity: int = 1,
        **kwargs
    ) -> UUID:
        """Create a purchase order for a resource"""
        try:
            order_id = uuid4()
            
            rid = UUID(str(resource_id)) if isinstance(resource_id, str) else resource_id
            resource = self._resource_listings.get(rid)
            
            if not resource:
                raise ValueError(f"Resource not found: {resource_id}")
            
            logger.info("Created order",
                       order_id=str(order_id),
                       resource_id=str(resource_id),
                       buyer_user_id=str(buyer_user_id))
            
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise
    
    async def create_purchase_order(
        self,
        resource_id: Union[str, UUID],
        buyer_user_id: Union[str, UUID],
        order_type: str = "purchase",
        quantity: int = 1,
        **kwargs
    ) -> UUID:
        """Create a purchase order (alias for create_order)"""
        return await self.create_order(resource_id, buyer_user_id, order_type, quantity, **kwargs)


# Global marketplace service instance
_global_marketplace_service: Optional[RealMarketplaceService] = None


def get_real_marketplace_service() -> RealMarketplaceService:
    """Get global marketplace service instance"""
    global _global_marketplace_service
    if _global_marketplace_service is None:
        _global_marketplace_service = RealMarketplaceService()
    return _global_marketplace_service


# Convenience functions
async def find_best_model_for_task(capabilities: List[ModelCapability],
                                  max_price: Optional[Decimal] = None,
                                  min_quality: Optional[float] = 0.7) -> Optional[MarketplaceModel]:
    """Find best marketplace model for specific task"""
    service = get_real_marketplace_service()
    
    query = MarketplaceQuery(
        required_capabilities=capabilities,
        max_price_per_token=max_price,
        min_performance_score=min_quality
    )
    
    models = await service.discover_models(query)
    return models[0] if models else None


async def get_reasoning_model() -> Optional[MarketplaceModel]:
    """Get best available model for reasoning tasks"""
    return await find_best_model_for_task(
        capabilities=[ModelCapability.REASONING, ModelCapability.TEXT_GENERATION],
        min_quality=0.8
    )

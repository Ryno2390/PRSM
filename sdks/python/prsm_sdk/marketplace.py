"""
PRSM SDK Model Marketplace
Access to PRSM's model ecosystem for discovering and using AI models
"""

import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from .models import ModelProvider
from .exceptions import ModelNotFoundError, PRSMError

logger = structlog.get_logger(__name__)


class ModelCategory(str, Enum):
    """Model categories in the marketplace"""
    LANGUAGE = "language"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    SCIENTIFIC = "scientific"
    REASONING = "reasoning"
    CODE = "code"


class ModelInfo(BaseModel):
    """Information about an AI model in the marketplace"""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable name")
    provider: ModelProvider = Field(..., description="Model provider")
    description: str = Field(..., description="Model description")
    category: ModelCategory = Field(..., description="Model category")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    cost_per_token: float = Field(..., description="FTNS cost per token")
    max_tokens: int = Field(..., description="Maximum token limit")
    context_window: int = Field(..., description="Context window size")
    is_available: bool = Field(True, description="Currently available")
    performance_rating: float = Field(..., ge=0, le=1, description="Performance score")
    safety_rating: float = Field(..., ge=0, le=1, description="Safety score")
    popularity: int = Field(0, description="Usage count")
    created_at: datetime = Field(..., description="Model creation date")
    updated_at: datetime = Field(..., description="Last update date")


class ModelSearchRequest(BaseModel):
    """Search request for models"""
    query: Optional[str] = Field(None, description="Search query")
    provider: Optional[ModelProvider] = Field(None, description="Filter by provider")
    category: Optional[ModelCategory] = Field(None, description="Filter by category")
    max_cost: Optional[float] = Field(None, description="Maximum cost per token")
    min_performance: Optional[float] = Field(None, ge=0, le=1, description="Minimum performance")
    min_safety: Optional[float] = Field(None, ge=0, le=1, description="Minimum safety rating")
    capabilities: Optional[List[str]] = Field(None, description="Required capabilities")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class ModelSearchResult(BaseModel):
    """Search result containing models"""
    models: List[ModelInfo] = Field(default_factory=list, description="Found models")
    total: int = Field(..., description="Total matching models")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Current limit")


class ModelRental(BaseModel):
    """Model rental information"""
    model_id: str = Field(..., description="Rented model ID")
    rental_id: str = Field(..., description="Rental transaction ID")
    start_time: datetime = Field(..., description="Rental start time")
    end_time: datetime = Field(..., description="Rental end time")
    cost: float = Field(..., description="Total rental cost in FTNS")
    requests_used: int = Field(0, description="Number of requests made")
    request_limit: Optional[int] = Field(None, description="Maximum requests allowed")


class ModelStats(BaseModel):
    """Model usage statistics"""
    model_id: str = Field(..., description="Model ID")
    total_requests: int = Field(..., description="Total requests")
    total_tokens: int = Field(..., description="Total tokens processed")
    total_cost: float = Field(..., description="Total FTNS cost")
    avg_latency: float = Field(..., description="Average latency in seconds")
    success_rate: float = Field(..., description="Success rate (0-1)")
    last_used: datetime = Field(..., description="Last usage time")


class ModelMarketplace:
    """
    Access to PRSM's model marketplace
    
    Provides methods for:
    - Searching and discovering models
    - Getting model information
    - Renting models for use
    - Tracking model usage
    """
    
    def __init__(self, client):
        """
        Initialize marketplace manager
        
        Args:
            client: PRSMClient instance for making API requests
        """
        self._client = client
    
    async def search_models(
        self,
        query: Optional[str] = None,
        provider: Optional[ModelProvider] = None,
        category: Optional[ModelCategory] = None,
        max_cost: Optional[float] = None,
        min_performance: Optional[float] = None,
        min_safety: Optional[float] = None,
        capabilities: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0
    ) -> ModelSearchResult:
        """
        Search for models in the marketplace
        
        Args:
            query: Search query string
            provider: Filter by provider
            category: Filter by category
            max_cost: Maximum cost per token
            min_performance: Minimum performance rating
            min_safety: Minimum safety rating
            capabilities: Required capabilities
            limit: Maximum results to return
            offset: Offset for pagination
            
        Returns:
            ModelSearchResult with matching models
            
        Example:
            results = await client.marketplace.search_models(
                query="gpt",
                min_performance=0.8,
                limit=10
            )
            for model in results.models:
                print(f"{model.name}: {model.cost_per_token} FTNS/token")
        """
        request = ModelSearchRequest(
            query=query,
            provider=provider,
            category=category,
            max_cost=max_cost,
            min_performance=min_performance,
            min_safety=min_safety,
            capabilities=capabilities,
            limit=limit,
            offset=offset
        )
        
        response = await self._client._request(
            "POST",
            "/marketplace/search",
            json_data=request.model_dump(exclude_none=True)
        )
        
        return ModelSearchResult(**response)
    
    async def get_model(self, model_id: str) -> ModelInfo:
        """
        Get detailed information about a specific model
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelInfo with model details
            
        Raises:
            ModelNotFoundError: If model doesn't exist
            
        Example:
            model = await client.marketplace.get_model("gpt-4")
            print(f"Model: {model.name}, Context: {model.context_window}")
        """
        try:
            response = await self._client._request(
                "GET",
                f"/marketplace/models/{model_id}"
            )
            return ModelInfo(**response)
        except PRSMError as e:
            if "not found" in str(e).lower():
                raise ModelNotFoundError(model_id)
            raise
    
    async def list_models(
        self,
        category: Optional[ModelCategory] = None,
        provider: Optional[ModelProvider] = None,
        limit: int = 20
    ) -> List[ModelInfo]:
        """
        List available models
        
        Args:
            category: Filter by category
            provider: Filter by provider
            limit: Maximum results
            
        Returns:
            List of ModelInfo objects
        """
        params = {"limit": limit}
        if category:
            params["category"] = category.value
        if provider:
            params["provider"] = provider.value
        
        response = await self._client._request(
            "GET",
            "/marketplace/models",
            params=params
        )
        
        return [ModelInfo(**m) for m in response.get("models", [])]
    
    async def rent_model(
        self,
        model_id: str,
        duration_hours: int = 1,
        max_requests: Optional[int] = None
    ) -> ModelRental:
        """
        Rent a model for use
        
        Args:
            model_id: Model to rent
            duration_hours: Rental duration in hours
            max_requests: Maximum requests allowed (None for unlimited)
            
        Returns:
            ModelRental with rental details
            
        Example:
            rental = await client.marketplace.rent_model(
                "gpt-4",
                duration_hours=24,
                max_requests=1000
            )
            print(f"Rental cost: {rental.cost} FTNS")
        """
        response = await self._client._request(
            "POST",
            f"/marketplace/models/{model_id}/rent",
            json_data={
                "duration_hours": duration_hours,
                "max_requests": max_requests
            }
        )
        
        return ModelRental(**response)
    
    async def get_rental(self, rental_id: str) -> ModelRental:
        """
        Get rental information
        
        Args:
            rental_id: Rental identifier
            
        Returns:
            ModelRental with rental details
        """
        response = await self._client._request(
            "GET",
            f"/marketplace/rentals/{rental_id}"
        )
        return ModelRental(**response)
    
    async def list_rentals(self, active_only: bool = True) -> List[ModelRental]:
        """
        List user's model rentals
        
        Args:
            active_only: Only return active rentals
            
        Returns:
            List of ModelRental objects
        """
        params = {"active_only": str(active_only).lower()}
        response = await self._client._request(
            "GET",
            "/marketplace/rentals",
            params=params
        )
        return [ModelRental(**r) for r in response.get("rentals", [])]
    
    async def get_model_stats(self, model_id: str) -> ModelStats:
        """
        Get usage statistics for a model
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelStats with usage statistics
        """
        response = await self._client._request(
            "GET",
            f"/marketplace/models/{model_id}/stats"
        )
        return ModelStats(**response)
    
    async def get_featured_models(self, limit: int = 10) -> List[ModelInfo]:
        """
        Get featured/popular models
        
        Args:
            limit: Maximum results
            
        Returns:
            List of featured ModelInfo objects
        """
        response = await self._client._request(
            "GET",
            "/marketplace/featured",
            params={"limit": limit}
        )
        return [ModelInfo(**m) for m in response.get("models", [])]
    
    async def get_recommended_models(
        self,
        task: str,
        limit: int = 5
    ) -> List[ModelInfo]:
        """
        Get recommended models for a specific task
        
        Args:
            task: Task description (e.g., "code generation", "summarization")
            limit: Maximum results
            
        Returns:
            List of recommended ModelInfo objects
        """
        response = await self._client._request(
            "POST",
            "/marketplace/recommend",
            json_data={"task": task, "limit": limit}
        )
        return [ModelInfo(**m) for m in response.get("models", [])]
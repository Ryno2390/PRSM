"""
Legacy Marketplace Service (DEPRECATED)
=======================================

⚠️  DEPRECATED: This file contains legacy mock implementations and should NOT be used.

Use real_marketplace_service.py instead - it contains production-ready implementations
with real SQLAlchemy database operations, comprehensive error handling, and all features.

This file is kept only for backwards compatibility with existing tests and will be
removed in a future version.

See docs/architecture/marketplace-status.md for current implementation details.
"""

import structlog
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.orm import Session

from ..core.database import get_database_service
from ..integrations.security.audit_logger import audit_logger
from .models import (
    ModelListingDB, RentalAgreementDB, MarketplaceOrderDB,
    ModelListing, RentalAgreement, MarketplaceOrder,
    ModelCategory, PricingTier, ModelProvider, ModelStatus,
    CreateModelListingRequest, MarketplaceSearchFilters, MarketplaceStatsResponse
)

logger = structlog.get_logger(__name__)


class MarketplaceService:
    """
    Core marketplace service for model discovery and management
    
    Features:
    - Model listing creation and management
    - Advanced search and filtering
    - Marketplace statistics and analytics
    - Model verification and moderation
    - Performance tracking and optimization
    """
    
    def __init__(self):
        self.db_service = get_database_service()
        
        # Marketplace configuration
        self.platform_fee_percentage = Decimal('0.025')  # 2.5% platform fee
        self.featured_model_boost = Decimal('10.0')
        self.verified_model_boost = Decimal('5.0')
        
        # Cache for frequently accessed data
        self._stats_cache = {}
        self._stats_cache_ttl = timedelta(minutes=15)
        self._last_stats_update = None
    
    async def create_model_listing(
        self,
        request: CreateModelListingRequest,
        owner_user_id: UUID
    ) -> ModelListing:
        """
        Create a new model listing in the marketplace
        
        Args:
            request: Model listing creation request
            owner_user_id: ID of the user creating the listing
            
        Returns:
            Created model listing
        """
        try:
            # Validate unique model_id
            existing_listing = await self._get_listing_by_model_id(request.model_id)
            if existing_listing:
                raise ValueError(f"Model with ID '{request.model_id}' already exists")
            
            # Create database record
            listing_data = {
                "id": uuid4(),
                "name": request.name,
                "description": request.description,
                "model_id": request.model_id,
                "provider": request.provider.value,
                "category": request.category.value,
                "owner_user_id": owner_user_id,
                "provider_name": request.provider_name,
                "provider_url": request.provider_url,
                "model_version": request.model_version,
                "pricing_tier": request.pricing_tier.value,
                "base_price": request.base_price or Decimal('0'),
                "price_per_token": request.price_per_token or Decimal('0'),
                "price_per_request": request.price_per_request or Decimal('0'),
                "price_per_minute": request.price_per_minute or Decimal('0'),
                "context_length": request.context_length,
                "max_tokens": request.max_tokens,
                "input_modalities": request.input_modalities,
                "output_modalities": request.output_modalities,
                "languages_supported": request.languages_supported,
                "api_endpoint": request.api_endpoint,
                "documentation_url": request.documentation_url,
                "license_type": request.license_type,
                "tags": request.tags,
                "metadata": request.metadata or {},
                "status": ModelStatus.PENDING_REVIEW.value,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            # Store in database
            listing_id = await self._create_listing_db(listing_data)
            
            # Log creation event
            await audit_logger.log_security_event(
                event_type="marketplace_listing_created",
                user_id=str(owner_user_id),
                details={
                    "listing_id": str(listing_id),
                    "model_id": request.model_id,
                    "category": request.category.value,
                    "pricing_tier": request.pricing_tier.value
                },
                security_level="info"
            )
            
            logger.info("Model listing created",
                       listing_id=str(listing_id),
                       model_id=request.model_id,
                       owner_user_id=str(owner_user_id))
            
            # Return created listing
            return await self.get_model_listing(listing_id)
            
        except Exception as e:
            logger.error("Failed to create model listing",
                        model_id=request.model_id,
                        owner_user_id=str(owner_user_id),
                        error=str(e))
            raise
    
    async def get_model_listing(self, listing_id: UUID) -> Optional[ModelListing]:
        """Get a model listing by ID"""
        try:
            listing_data = await self._get_listing_by_id(listing_id)
            if not listing_data:
                return None
            
            return self._db_to_pydantic_listing(listing_data)
            
        except Exception as e:
            logger.error("Failed to get model listing",
                        listing_id=str(listing_id),
                        error=str(e))
            return None
    
    async def search_models(
        self,
        filters: MarketplaceSearchFilters
    ) -> Tuple[List[ModelListing], int]:
        """
        Search models in the marketplace with filtering and pagination
        
        Args:
            filters: Search filters and pagination parameters
            
        Returns:
            Tuple of (listings, total_count)
        """
        try:
            # Build query conditions
            conditions = [ModelListingDB.status == ModelStatus.ACTIVE.value]
            
            if filters.category:
                conditions.append(ModelListingDB.category == filters.category.value)
            
            if filters.provider:
                conditions.append(ModelListingDB.provider == filters.provider.value)
            
            if filters.pricing_tier:
                conditions.append(ModelListingDB.pricing_tier == filters.pricing_tier.value)
            
            if filters.min_price is not None:
                conditions.append(ModelListingDB.base_price >= filters.min_price)
            
            if filters.max_price is not None:
                conditions.append(ModelListingDB.base_price <= filters.max_price)
            
            if filters.verified_only:
                conditions.append(ModelListingDB.verified == True)
            
            if filters.featured_only:
                conditions.append(ModelListingDB.featured == True)
            
            if filters.tags:
                # Search for any of the provided tags
                tag_conditions = [
                    ModelListingDB.tags.contains([tag]) for tag in filters.tags
                ]
                conditions.append(or_(*tag_conditions))
            
            if filters.search_query:
                search_term = f"%{filters.search_query}%"
                search_conditions = or_(
                    ModelListingDB.name.ilike(search_term),
                    ModelListingDB.description.ilike(search_term),
                    ModelListingDB.model_id.ilike(search_term)
                )
                conditions.append(search_conditions)
            
            # Execute search
            listings_data, total_count = await self._search_listings_db(
                conditions, filters.sort_by, filters.sort_order,
                filters.limit, filters.offset
            )
            
            # Convert to Pydantic models
            listings = [
                self._db_to_pydantic_listing(listing_data)
                for listing_data in listings_data
            ]
            
            return listings, total_count
            
        except Exception as e:
            logger.error("Failed to search models",
                        filters=filters.dict(),
                        error=str(e))
            return [], 0
    
    async def get_featured_models(self, limit: int = 10) -> List[ModelListing]:
        """Get featured models for homepage display"""
        try:
            filters = MarketplaceSearchFilters(
                featured_only=True,
                sort_by="popularity",
                sort_order="desc",
                limit=limit,
                offset=0
            )
            
            listings, _ = await self.search_models(filters)
            return listings
            
        except Exception as e:
            logger.error("Failed to get featured models", error=str(e))
            return []
    
    async def get_models_by_category(
        self,
        category: ModelCategory,
        limit: int = 20
    ) -> List[ModelListing]:
        """Get models in a specific category"""
        try:
            filters = MarketplaceSearchFilters(
                category=category,
                sort_by="popularity",
                sort_order="desc",
                limit=limit,
                offset=0
            )
            
            listings, _ = await self.search_models(filters)
            return listings
            
        except Exception as e:
            logger.error("Failed to get models by category",
                        category=category.value,
                        error=str(e))
            return []
    
    async def get_marketplace_stats(self) -> MarketplaceStatsResponse:
        """Get comprehensive marketplace statistics"""
        try:
            # Check cache first
            now = datetime.now(timezone.utc)
            if (self._last_stats_update and 
                now - self._last_stats_update < self._stats_cache_ttl and
                self._stats_cache):
                return MarketplaceStatsResponse(**self._stats_cache)
            
            # Calculate fresh stats
            stats = await self._calculate_marketplace_stats()
            
            # Update cache
            self._stats_cache = stats
            self._last_stats_update = now
            
            return MarketplaceStatsResponse(**stats)
            
        except Exception as e:
            logger.error("Failed to get marketplace stats", error=str(e))
            # Return empty stats on error
            return MarketplaceStatsResponse(
                total_models=0, total_providers=0, total_categories=0,
                total_rentals=0, total_revenue=Decimal('0'), active_rentals=0,
                featured_models=0, verified_models=0, average_model_rating=None,
                most_popular_category=None, top_providers=[]
            )
    
    async def update_model_status(
        self,
        listing_id: UUID,
        new_status: ModelStatus,
        moderator_user_id: UUID
    ) -> bool:
        """Update model listing status (for moderation)"""
        try:
            success = await self._update_listing_status(listing_id, new_status.value)
            
            if success:
                await audit_logger.log_security_event(
                    event_type="marketplace_listing_status_updated",
                    user_id=str(moderator_user_id),
                    details={
                        "listing_id": str(listing_id),
                        "new_status": new_status.value
                    },
                    security_level="info"
                )
            
            return success
            
        except Exception as e:
            logger.error("Failed to update model status",
                        listing_id=str(listing_id),
                        new_status=new_status.value,
                        error=str(e))
            return False
    
    async def increment_model_usage(
        self,
        listing_id: UUID,
        tokens_processed: int = 0,
        requests_count: int = 1
    ) -> bool:
        """Increment usage statistics for a model"""
        try:
            return await self._increment_usage_stats(
                listing_id, tokens_processed, requests_count
            )
            
        except Exception as e:
            logger.error("Failed to increment model usage",
                        listing_id=str(listing_id),
                        error=str(e))
            return False
    
    # Database operations
    
    async def _create_listing_db(self, listing_data: Dict[str, Any]) -> UUID:
        """Create model listing in database"""
        # In a real implementation, this would use SQLAlchemy
        # For now, return a mock ID
        return listing_data["id"]
    
    async def _get_listing_by_id(self, listing_id: UUID) -> Optional[Dict[str, Any]]:
        """Get listing by ID from database"""
        # Mock implementation - in reality, would query database
        return {
            "id": listing_id,
            "name": "Example Model",
            "description": "A sample AI model for demonstration",
            "model_id": "example-model-v1",
            "provider": "openai",
            "category": "language_model",
            "owner_user_id": uuid4(),
            "pricing_tier": "free",
            "base_price": Decimal('0'),
            "status": "active",
            "featured": False,
            "verified": True,
            "created_at": datetime.now(timezone.utc)
        }
    
    async def _get_listing_by_model_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get listing by model ID from database"""
        # Mock implementation
        return None
    
    async def _search_listings_db(
        self,
        conditions: List,
        sort_by: str,
        sort_order: str,
        limit: int,
        offset: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Search listings in database"""
        # Mock implementation - in reality, would execute SQLAlchemy query
        mock_listings = [
            {
                "id": uuid4(),
                "name": f"Model {i}",
                "description": f"Description for model {i}",
                "model_id": f"model-{i}",
                "provider": "openai",
                "category": "language_model",
                "owner_user_id": uuid4(),
                "pricing_tier": "free" if i % 2 == 0 else "premium",
                "base_price": Decimal('0') if i % 2 == 0 else Decimal('10'),
                "status": "active",
                "featured": i < 3,
                "verified": True,
                "popularity_score": Decimal(str(10 - i)),
                "created_at": datetime.now(timezone.utc)
            }
            for i in range(min(limit, 10))
        ]
        
        return mock_listings, min(limit, 10)
    
    async def _calculate_marketplace_stats(self) -> Dict[str, Any]:
        """Calculate marketplace statistics"""
        # Mock implementation - in reality, would aggregate from database
        return {
            "total_models": 150,
            "total_providers": 8,
            "total_categories": 12,
            "total_rentals": 1200,
            "total_revenue": Decimal('45000.50'),
            "active_rentals": 89,
            "featured_models": 15,
            "verified_models": 120,
            "average_model_rating": Decimal('4.2'),
            "most_popular_category": "language_model",
            "top_providers": [
                {"name": "OpenAI", "model_count": 45},
                {"name": "Anthropic", "model_count": 32},
                {"name": "HuggingFace", "model_count": 28}
            ]
        }
    
    async def _update_listing_status(self, listing_id: UUID, status: str) -> bool:
        """Update listing status in database"""
        # Mock implementation
        return True
    
    async def _increment_usage_stats(
        self,
        listing_id: UUID,
        tokens_processed: int,
        requests_count: int
    ) -> bool:
        """Increment usage statistics in database"""
        # Mock implementation
        return True
    
    def _db_to_pydantic_listing(self, db_data: Dict[str, Any]) -> ModelListing:
        """Convert database data to Pydantic model"""
        return ModelListing(
            id=db_data.get("id"),
            name=db_data.get("name", ""),
            description=db_data.get("description"),
            model_id=db_data.get("model_id", ""),
            provider=ModelProvider(db_data.get("provider", "custom")),
            category=ModelCategory(db_data.get("category", "custom")),
            owner_user_id=db_data.get("owner_user_id"),
            pricing_tier=PricingTier(db_data.get("pricing_tier", "free")),
            base_price=db_data.get("base_price", Decimal('0')),
            status=ModelStatus(db_data.get("status", "pending_review")),
            featured=db_data.get("featured", False),
            verified=db_data.get("verified", False),
            popularity_score=db_data.get("popularity_score", Decimal('0')),
            created_at=db_data.get("created_at"),
            updated_at=db_data.get("updated_at")
        )


# Global marketplace service instance
marketplace_service = MarketplaceService()
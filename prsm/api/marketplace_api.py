"""
Marketplace API
===============

REST API endpoints for the PRSM AI model marketplace.
Provides model discovery, listing management, and rental operations.
"""

import structlog
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel

from prsm.auth import get_current_user
from prsm.auth.models import UserRole
from prsm.auth.auth_manager import auth_manager
from prsm.security import create_secure_user_input
from prsm.marketplace import (
    marketplace_service, model_listing_service, rental_engine,
    ModelListing, RentalAgreement, MarketplaceOrder,
    ModelCategory, PricingTier, ModelProvider, ModelStatus,
    CreateModelListingRequest, RentModelRequest, MarketplaceSearchFilters
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/marketplace", tags=["marketplace"])


class ModelListingResponse(BaseModel):
    """Response model for model listing"""
    success: bool
    listing: ModelListing
    message: str


class SearchResponse(BaseModel):
    """Response model for marketplace search"""
    success: bool
    listings: List[ModelListing]
    total_count: int
    page: int
    page_size: int
    has_next: bool


class RentalResponse(BaseModel):
    """Response model for model rental"""
    success: bool
    rental_agreement: RentalAgreement
    order: MarketplaceOrder
    message: str


# Model Listing Endpoints

@router.post("/models", response_model=ModelListingResponse)
async def create_model_listing(
    request: CreateModelListingRequest,
    current_user: str = Depends(get_current_user)
) -> ModelListingResponse:
    """
    Create a new model listing in the marketplace
    
    🚀 MARKETPLACE LISTING:
    - Creates a new AI model listing for marketplace discovery
    - Automatically sets status to pending review for moderation
    - Supports comprehensive model metadata and pricing
    - Integrates with FTNS token economy for payments
    """
    try:
        # Create the model listing
        listing = await marketplace_service.create_model_listing(
            request=request,
            owner_user_id=UUID(current_user)
        )
        
        return ModelListingResponse(
            success=True,
            listing=listing,
            message="Model listing created successfully. It will be reviewed before going live."
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to create model listing",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create model listing"
        )


@router.get("/models/{listing_id}", response_model=ModelListingResponse)
async def get_model_listing(
    listing_id: UUID,
    current_user: str = Depends(get_current_user)
) -> ModelListingResponse:
    """
    Get detailed information about a specific model listing
    
    🔍 MODEL DETAILS:
    - Returns comprehensive model information and metadata
    - Includes pricing, performance metrics, and availability
    - Shows rental options and current usage statistics
    """
    try:
        listing = await marketplace_service.get_model_listing(listing_id)
        
        if not listing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model listing not found"
            )
        
        return ModelListingResponse(
            success=True,
            listing=listing,
            message="Model listing retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model listing",
                    listing_id=str(listing_id),
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model listing"
        )


@router.get("/models", response_model=SearchResponse)
async def search_models(
    # Search parameters
    category: Optional[ModelCategory] = Query(None, description="Filter by model category"),
    provider: Optional[ModelProvider] = Query(None, description="Filter by model provider"),
    pricing_tier: Optional[PricingTier] = Query(None, description="Filter by pricing tier"),
    min_price: Optional[Decimal] = Query(None, ge=0, description="Minimum price in FTNS"),
    max_price: Optional[Decimal] = Query(None, ge=0, description="Maximum price in FTNS"),
    verified_only: bool = Query(False, description="Show only verified models"),
    featured_only: bool = Query(False, description="Show only featured models"),
    search_query: Optional[str] = Query(None, max_length=255, description="Search query"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    
    # Sorting and pagination
    sort_by: str = Query("popularity", regex=r'^(popularity|price|created_at|name)$', description="Sort field"),
    sort_order: str = Query("desc", regex=r'^(asc|desc)$', description="Sort order"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    
    current_user: str = Depends(get_current_user)
) -> SearchResponse:
    """
    Search and discover AI models in the marketplace
    
    🔍 MODEL DISCOVERY:
    - Advanced filtering by category, provider, pricing, and more
    - Full-text search across model names and descriptions
    - Tag-based discovery for specific use cases
    - Sorting by popularity, price, and recency
    - Pagination for large result sets
    """
    try:
        # Parse tags if provided
        tag_list = []
        if tags:
            tag_list = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
        
        # Create search filters
        filters = MarketplaceSearchFilters(
            category=category,
            provider=provider,
            pricing_tier=pricing_tier,
            min_price=min_price,
            max_price=max_price,
            verified_only=verified_only,
            featured_only=featured_only,
            tags=tag_list if tag_list else None,
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=page_size,
            offset=(page - 1) * page_size
        )
        
        # Execute search
        listings, total_count = await marketplace_service.search_models(filters)
        
        # Calculate pagination info
        has_next = (page * page_size) < total_count
        
        return SearchResponse(
            success=True,
            listings=listings,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=has_next
        )
        
    except Exception as e:
        logger.error("Failed to search models",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search models"
        )


@router.get("/featured", response_model=List[ModelListing])
async def get_featured_models(
    limit: int = Query(10, ge=1, le=50, description="Number of featured models to return"),
    current_user: str = Depends(get_current_user)
) -> List[ModelListing]:
    """
    Get featured models for homepage display
    
    ⭐ FEATURED MODELS:
    - Curated selection of high-quality models
    - Optimized for discovery and user engagement
    - Regularly updated based on performance and popularity
    """
    try:
        featured_models = await marketplace_service.get_featured_models(limit)
        return featured_models
        
    except Exception as e:
        logger.error("Failed to get featured models",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve featured models"
        )


@router.get("/categories/{category}", response_model=List[ModelListing])
async def get_models_by_category(
    category: ModelCategory,
    limit: int = Query(20, ge=1, le=100, description="Number of models to return"),
    current_user: str = Depends(get_current_user)
) -> List[ModelListing]:
    """
    Get models in a specific category
    
    📂 CATEGORY BROWSE:
    - Browse models by AI category (language, image, code, etc.)
    - Sorted by popularity within category
    - Optimized for category-specific discovery
    """
    try:
        models = await marketplace_service.get_models_by_category(category, limit)
        return models
        
    except Exception as e:
        logger.error("Failed to get models by category",
                    category=category.value,
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve models by category"
        )


# Model Rental Endpoints

@router.post("/models/{listing_id}/rent", response_model=RentalResponse)
async def rent_model(
    listing_id: UUID,
    request: RentModelRequest,
    current_user: str = Depends(get_current_user)
) -> RentalResponse:
    """
    Rent an AI model for usage
    
    💰 MODEL RENTAL:
    - Creates a rental agreement with specified terms
    - Processes payment using FTNS tokens
    - Provides secure access to the rented model
    - Tracks usage and enforces limits
    """
    try:
        # Verify model listing exists
        listing = await marketplace_service.get_model_listing(listing_id)
        if not listing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model listing not found"
            )
        
        if listing.status != ModelStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model is not available for rental"
            )
        
        # Create rental agreement and process payment
        rental_result = await rental_engine.create_rental(
            renter_user_id=UUID(current_user),
            model_listing_id=listing_id,
            rental_request=request
        )
        
        return RentalResponse(
            success=True,
            rental_agreement=rental_result["rental_agreement"],
            order=rental_result["order"],
            message="Model rented successfully. You can now access the model."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to rent model",
                    listing_id=str(listing_id),
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rent model"
        )


@router.get("/rentals", response_model=List[RentalAgreement])
async def get_my_rentals(
    status: Optional[str] = Query(None, regex=r'^(active|expired|cancelled|suspended)$'),
    current_user: str = Depends(get_current_user)
) -> List[RentalAgreement]:
    """
    Get current user's model rentals
    
    📋 MY RENTALS:
    - View all active and past model rentals
    - Filter by rental status
    - Track usage and remaining time/tokens
    """
    try:
        rentals = await rental_engine.get_user_rentals(
            user_id=UUID(current_user),
            status_filter=status
        )
        return rentals
        
    except Exception as e:
        logger.error("Failed to get user rentals",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve rentals"
        )


@router.get("/rentals/{rental_id}", response_model=RentalAgreement)
async def get_rental_details(
    rental_id: UUID,
    current_user: str = Depends(get_current_user)
) -> RentalAgreement:
    """
    Get detailed information about a specific rental
    
    📄 RENTAL DETAILS:
    - View complete rental agreement terms
    - Check usage statistics and remaining allowances
    - Access model-specific information and endpoints
    """
    try:
        rental = await rental_engine.get_rental(rental_id)
        
        if not rental:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rental agreement not found"
            )
        
        # Verify user owns this rental
        if str(rental.renter_user_id) != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this rental"
            )
        
        return rental
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get rental details",
                    rental_id=str(rental_id),
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve rental details"
        )


# Marketplace Statistics

@router.get("/stats")
async def get_marketplace_stats(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get marketplace statistics and analytics
    
    📊 MARKETPLACE STATS:
    - Total models, providers, and categories
    - Revenue and rental statistics
    - Popular categories and trending models
    - Public marketplace metrics
    """
    try:
        stats = await marketplace_service.get_marketplace_stats()
        return {
            "success": True,
            "stats": stats.dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get marketplace stats",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve marketplace statistics"
        )


# Admin/Moderation Endpoints

@router.patch("/models/{listing_id}/status")
async def update_model_status(
    listing_id: UUID,
    new_status: ModelStatus,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update model listing status (admin only)
    
    🔨 MODERATION:
    - Approve or reject pending model listings
    - Suspend or activate existing models
    - Manage marketplace content quality
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required"
            )
        
        success = await marketplace_service.update_model_status(
            listing_id=listing_id,
            new_status=new_status,
            moderator_user_id=UUID(current_user)
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model listing not found"
            )
        
        return {
            "success": True,
            "message": f"Model status updated to {new_status.value}",
            "listing_id": str(listing_id),
            "new_status": new_status.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update model status",
                    listing_id=str(listing_id),
                    new_status=new_status.value,
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update model status"
        )


# Health and Discovery Endpoints

@router.get("/categories")
async def list_categories(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all available model categories
    
    📂 CATEGORIES:
    - Browse all AI model categories
    - Optimized for navigation and discovery
    """
    try:
        categories = [
            {
                "value": category.value,
                "name": category.value.replace("_", " ").title(),
                "description": f"Models in the {category.value.replace('_', ' ')} category"
            }
            for category in ModelCategory
        ]
        
        return {
            "success": True,
            "categories": categories,
            "total_count": len(categories)
        }
        
    except Exception as e:
        logger.error("Failed to list categories",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve categories"
        )


@router.get("/providers")
async def list_providers(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all available model providers
    
    🏢 PROVIDERS:
    - Browse all AI model providers
    - Information about supported platforms
    """
    try:
        providers = [
            {
                "value": provider.value,
                "name": provider.value.replace("_", " ").title(),
                "description": f"Models from {provider.value.replace('_', ' ')}"
            }
            for provider in ModelProvider
        ]
        
        return {
            "success": True,
            "providers": providers,
            "total_count": len(providers)
        }
        
    except Exception as e:
        logger.error("Failed to list providers",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve providers"
        )
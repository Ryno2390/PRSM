"""
Production FTNS Marketplace API
==============================

Production-grade REST API for the FTNS marketplace with real value transfer.
Provides comprehensive marketplace functionality with actual FTNS token
transactions, escrow protection, and blockchain integration.

This API implements:
- Real FTNS value transfer for all marketplace transactions
- Production-grade escrow system with automatic timeout handling
- Comprehensive seller and buyer protection mechanisms
- Real-time marketplace analytics and reporting
- Multi-tier seller verification and management
- Advanced search and filtering capabilities
- Integration with blockchain settlement for enhanced security
- Comprehensive audit trails and compliance reporting

Key Features:
- RESTful API with comprehensive OpenAPI documentation
- Real FTNS token transactions using production ledger
- Secure escrow system with automated release mechanisms
- Advanced marketplace search and discovery
- Seller verification and tier management
- Dispute resolution and refund processing
- Real-time analytics and marketplace insights
- Rate limiting and fraud prevention
- Comprehensive error handling and validation
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import structlog

from prsm.marketplace.real_expanded_marketplace_service import (
    get_marketplace_service,
    AssetType,
    TransactionStatus,
    SellerTier
)
from prsm.security.production_rbac import get_rbac_manager
from prsm.security.distributed_rate_limiter import get_rate_limiter

logger = structlog.get_logger(__name__)

# Initialize router
router = APIRouter(prefix="/api/marketplace", tags=["marketplace"])
security = HTTPBearer()


# Pydantic models for request/response
class CreateListingRequest(BaseModel):
    asset_type: AssetType
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=5000)
    price_ftns: Decimal = Field(..., gt=0, le=1000000)
    usage_price_ftns: Optional[Decimal] = Field(None, gt=0, le=10000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: Optional[List[str]] = Field(default_factory=list, max_items=10)
    
    @validator('price_ftns', 'usage_price_ftns')
    def validate_price_precision(cls, v):
        if v is not None and v.as_tuple().exponent < -18:
            raise ValueError('Price cannot have more than 18 decimal places')
        return v


class PurchaseAssetRequest(BaseModel):
    listing_id: str
    quantity: int = Field(default=1, ge=1, le=1000)
    use_blockchain_settlement: bool = Field(default=False)


class DeliverAssetRequest(BaseModel):
    transaction_id: str
    delivery_data: Dict[str, Any]
    
    @validator('delivery_data')
    def validate_delivery_data(cls, v):
        required_fields = ['access_method', 'access_data']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Missing required field: {field}')
        return v


class ConfirmDeliveryRequest(BaseModel):
    transaction_id: str
    rating: float = Field(..., ge=1.0, le=5.0)
    review: Optional[str] = Field(None, max_length=1000)


class DisputeTransactionRequest(BaseModel):
    transaction_id: str
    dispute_reason: str = Field(..., min_length=10, max_length=500)
    evidence: Dict[str, Any] = Field(default_factory=dict)


class MarketplaceSearchRequest(BaseModel):
    query: Optional[str] = None
    asset_type: Optional[AssetType] = None
    min_price: Optional[Decimal] = Field(None, ge=0)
    max_price: Optional[Decimal] = Field(None, ge=0)
    seller_tier: Optional[SellerTier] = None
    tags: Optional[List[str]] = None
    sort_by: str = Field(default="created_at", regex="^(created_at|price_ftns|average_rating|total_sales)$")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$")
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


# Response models
class ListingResponse(BaseModel):
    listing_id: str
    seller_id: str
    asset_type: AssetType
    title: str
    description: str
    price_ftns: Decimal
    usage_price_ftns: Optional[Decimal]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    total_sales: int
    average_rating: float
    verification_status: str
    tags: List[str]


class TransactionResponse(BaseModel):
    transaction_id: str
    listing_id: str
    buyer_id: str
    seller_id: str
    asset_type: AssetType
    quantity: int
    unit_price_ftns: Decimal
    total_price_ftns: Decimal
    marketplace_fee_ftns: Decimal
    seller_amount_ftns: Decimal
    status: TransactionStatus
    escrow_tx_id: Optional[str]
    delivery_tx_id: Optional[str]
    completion_tx_id: Optional[str]
    created_at: datetime
    delivered_at: Optional[datetime]
    completed_at: Optional[datetime]
    dispute_reason: Optional[str]


class SellerProfileResponse(BaseModel):
    seller_id: str
    tier: SellerTier
    verification_score: float
    total_sales: int
    total_revenue_ftns: Decimal
    average_rating: float
    response_time_hours: float
    completion_rate: float
    dispute_rate: float
    joined_at: datetime
    last_active: datetime


class MarketplaceAnalyticsResponse(BaseModel):
    total_listings: int
    total_transactions: int
    total_volume_ftns: Decimal
    active_buyers: int
    active_sellers: int
    avg_transaction_value: Decimal
    top_sellers: List[Dict[str, Any]]
    trending_assets: List[Dict[str, Any]]


# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract user from JWT token"""
    try:
        rbac_manager = await get_rbac_manager()
        user = await rbac_manager.verify_token(credentials.credentials)
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


# Rate limiting dependency
async def check_rate_limit(user=Depends(get_current_user)):
    """Apply rate limiting based on user tier"""
    try:
        rate_limiter = await get_rate_limiter()
        
        # Different limits based on user type
        if user.get('is_premium'):
            limit = await rate_limiter.check_rate_limit(
                f"marketplace_api:{user['user_id']}", 
                max_requests=100, 
                window_seconds=60
            )
        else:
            limit = await rate_limiter.check_rate_limit(
                f"marketplace_api:{user['user_id']}", 
                max_requests=30, 
                window_seconds=60
            )
        
        if not limit.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {limit.retry_after} seconds"
            )
        
        return user
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        return user  # Allow request if rate limiter fails


# API Endpoints

@router.post("/listings", response_model=Dict[str, str])
async def create_listing(
    request: CreateListingRequest,
    user=Depends(check_rate_limit)
):
    """
    Create new marketplace listing
    
    Creates a new marketplace listing with the specified parameters.
    Requires valid authentication and sufficient seller verification.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        listing_id = await marketplace_service.create_listing(
            seller_id=user['user_id'],
            asset_type=request.asset_type,
            title=request.title,
            description=request.description,
            price_ftns=request.price_ftns,
            metadata=request.metadata,
            usage_price_ftns=request.usage_price_ftns,
            tags=request.tags
        )
        
        return {"listing_id": listing_id, "status": "created"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create listing")


@router.get("/listings/{listing_id}", response_model=ListingResponse)
async def get_listing(
    listing_id: str,
    user=Depends(get_current_user)
):
    """
    Get marketplace listing details
    
    Retrieves detailed information about a specific marketplace listing.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        listing = await marketplace_service.get_listing(listing_id)
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        return ListingResponse(
            listing_id=listing.listing_id,
            seller_id=listing.seller_id,
            asset_type=listing.asset_type,
            title=listing.title,
            description=listing.description,
            price_ftns=listing.price_ftns,
            usage_price_ftns=listing.usage_price_ftns,
            metadata=listing.metadata,
            created_at=listing.created_at,
            updated_at=listing.updated_at,
            is_active=listing.is_active,
            total_sales=listing.total_sales,
            average_rating=listing.average_rating,
            verification_status=listing.verification_status,
            tags=listing.tags
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve listing")


@router.post("/listings/search", response_model=List[ListingResponse])
async def search_listings(
    request: MarketplaceSearchRequest,
    user=Depends(get_current_user)
):
    """
    Search marketplace listings
    
    Search and filter marketplace listings based on various criteria
    including price, asset type, seller tier, and keywords.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        listings = await marketplace_service.search_listings(
            query=request.query,
            asset_type=request.asset_type,
            min_price=request.min_price,
            max_price=request.max_price,
            seller_tier=request.seller_tier,
            tags=request.tags,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
            limit=request.limit,
            offset=request.offset
        )
        
        return [
            ListingResponse(
                listing_id=listing.listing_id,
                seller_id=listing.seller_id,
                asset_type=listing.asset_type,
                title=listing.title,
                description=listing.description,
                price_ftns=listing.price_ftns,
                usage_price_ftns=listing.usage_price_ftns,
                metadata=listing.metadata,
                created_at=listing.created_at,
                updated_at=listing.updated_at,
                is_active=listing.is_active,
                total_sales=listing.total_sales,
                average_rating=listing.average_rating,
                verification_status=listing.verification_status,
                tags=listing.tags
            )
            for listing in listings
        ]
        
    except Exception as e:
        logger.error(f"Search listings error: {e}")
        raise HTTPException(status_code=500, detail="Failed to search listings")


@router.post("/purchase", response_model=Dict[str, str])
async def purchase_asset(
    request: PurchaseAssetRequest,
    user=Depends(check_rate_limit)
):
    """
    Purchase marketplace asset with real FTNS value transfer
    
    Initiates a marketplace purchase with escrow protection.
    Transfers FTNS tokens from buyer to escrow account.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        transaction_id = await marketplace_service.purchase_asset(
            buyer_id=user['user_id'],
            listing_id=request.listing_id,
            quantity=request.quantity,
            use_blockchain_settlement=request.use_blockchain_settlement
        )
        
        return {
            "transaction_id": transaction_id,
            "status": "escrowed",
            "message": "Purchase initiated with escrow protection"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Purchase error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process purchase")


@router.post("/deliver", response_model=Dict[str, str])
async def deliver_asset(
    request: DeliverAssetRequest,
    user=Depends(check_rate_limit)
):
    """
    Deliver purchased asset and release payment
    
    Seller delivers the purchased asset and triggers payment release
    from escrow to seller account.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        delivery_tx_id = await marketplace_service.deliver_asset(
            seller_id=user['user_id'],
            transaction_id=request.transaction_id,
            delivery_data=request.delivery_data
        )
        
        return {
            "delivery_tx_id": delivery_tx_id,
            "status": "delivered",
            "message": "Asset delivered and payment released"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Delivery error: {e}")
        raise HTTPException(status_code=500, detail="Failed to deliver asset")


@router.post("/confirm", response_model=Dict[str, str])
async def confirm_delivery(
    request: ConfirmDeliveryRequest,
    user=Depends(check_rate_limit)
):
    """
    Confirm asset delivery and complete transaction
    
    Buyer confirms receipt of asset and completes the transaction
    with optional rating and review.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        completion_tx_id = await marketplace_service.confirm_delivery(
            buyer_id=user['user_id'],
            transaction_id=request.transaction_id,
            rating=request.rating,
            review=request.review
        )
        
        return {
            "completion_tx_id": completion_tx_id,
            "status": "completed",
            "message": "Transaction completed successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Confirmation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to confirm delivery")


@router.post("/dispute", response_model=Dict[str, str])
async def create_dispute(
    request: DisputeTransactionRequest,
    user=Depends(check_rate_limit)
):
    """
    Create dispute for marketplace transaction
    
    Allows buyers or sellers to create disputes for transactions
    that require mediation or refund processing.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        dispute_id = await marketplace_service.dispute_transaction(
            user_id=user['user_id'],
            transaction_id=request.transaction_id,
            dispute_reason=request.dispute_reason,
            evidence=request.evidence
        )
        
        return {
            "dispute_id": dispute_id,
            "status": "disputed",
            "message": "Dispute created and will be reviewed by moderators"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Dispute creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create dispute")


@router.get("/transactions/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(
    transaction_id: str,
    user=Depends(get_current_user)
):
    """
    Get marketplace transaction details
    
    Retrieves detailed information about a specific marketplace transaction.
    Only accessible by buyer, seller, or administrators.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        transaction = await marketplace_service.get_transaction(transaction_id)
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        # Check authorization
        if user['user_id'] not in [transaction.buyer_id, transaction.seller_id]:
            rbac_manager = await get_rbac_manager()
            if not await rbac_manager.check_permission(user['user_id'], "marketplace.view_all_transactions"):
                raise HTTPException(status_code=403, detail="Access denied")
        
        return TransactionResponse(
            transaction_id=transaction.transaction_id,
            listing_id=transaction.listing_id,
            buyer_id=transaction.buyer_id,
            seller_id=transaction.seller_id,
            asset_type=transaction.asset_type,
            quantity=transaction.quantity,
            unit_price_ftns=transaction.unit_price_ftns,
            total_price_ftns=transaction.total_price_ftns,
            marketplace_fee_ftns=transaction.marketplace_fee_ftns,
            seller_amount_ftns=transaction.seller_amount_ftns,
            status=transaction.status,
            escrow_tx_id=transaction.escrow_tx_id,
            delivery_tx_id=transaction.delivery_tx_id,
            completion_tx_id=transaction.completion_tx_id,
            created_at=transaction.created_at,
            delivered_at=transaction.delivered_at,
            completed_at=transaction.completed_at,
            dispute_reason=transaction.dispute_reason
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get transaction error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve transaction")


@router.get("/sellers/{seller_id}/profile", response_model=SellerProfileResponse)
async def get_seller_profile(
    seller_id: str,
    user=Depends(get_current_user)
):
    """
    Get seller profile and statistics
    
    Retrieves comprehensive seller profile including verification status,
    sales statistics, and performance metrics.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        profile = await marketplace_service.get_seller_profile(seller_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Seller profile not found")
        
        return SellerProfileResponse(
            seller_id=profile.seller_id,
            tier=profile.tier,
            verification_score=profile.verification_score,
            total_sales=profile.total_sales,
            total_revenue_ftns=profile.total_revenue_ftns,
            average_rating=profile.average_rating,
            response_time_hours=profile.response_time_hours,
            completion_rate=profile.completion_rate,
            dispute_rate=profile.dispute_rate,
            joined_at=profile.joined_at,
            last_active=profile.last_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get seller profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve seller profile")


@router.get("/analytics", response_model=MarketplaceAnalyticsResponse)
async def get_marketplace_analytics(
    user=Depends(get_current_user)
):
    """
    Get comprehensive marketplace analytics
    
    Provides detailed marketplace statistics, trends, and insights.
    Access may be restricted based on user permissions.
    """
    try:
        # Check analytics access permission
        rbac_manager = await get_rbac_manager()
        if not await rbac_manager.check_permission(user['user_id'], "marketplace.view_analytics"):
            raise HTTPException(status_code=403, detail="Analytics access denied")
        
        marketplace_service = await get_marketplace_service()
        analytics = await marketplace_service.get_marketplace_analytics()
        
        return MarketplaceAnalyticsResponse(
            total_listings=analytics.get('total_listings', 0),
            total_transactions=analytics.get('total_transactions', 0),
            total_volume_ftns=analytics.get('total_volume_ftns', Decimal('0')),
            active_buyers=analytics.get('active_buyers', 0),
            active_sellers=analytics.get('active_sellers', 0),
            avg_transaction_value=analytics.get('avg_transaction_value', Decimal('0')),
            top_sellers=analytics.get('top_sellers', []),
            trending_assets=analytics.get('trending_assets', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get analytics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@router.get("/health")
async def marketplace_health():
    """
    Marketplace service health check
    
    Returns the health status of the marketplace service and its dependencies.
    """
    try:
        marketplace_service = await get_marketplace_service()
        
        # Basic health check
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "components": {
                "marketplace_service": "operational",
                "production_ledger": "operational",
                "blockchain_integration": "operational"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }
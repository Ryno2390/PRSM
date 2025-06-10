"""
Marketplace Launch API
======================

Special API endpoints for launching the marketplace with initial listings
and managing the marketplace launch process.
"""

import structlog
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from prsm.auth import get_current_user
from prsm.auth.models import UserRole
from prsm.auth.auth_manager import auth_manager
from prsm.marketplace.initial_listings import launch_marketplace_with_initial_listings

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/marketplace", tags=["marketplace-launch"])


class LaunchResponse(BaseModel):
    """Response model for marketplace launch"""
    success: bool
    message: str
    launch_summary: Dict[str, Any]
    timestamp: str


@router.post("/launch", response_model=LaunchResponse)
async def launch_marketplace(
    current_user: str = Depends(get_current_user)
) -> LaunchResponse:
    """
    Launch the PRSM marketplace with initial model listings
    
    🚀 MARKETPLACE LAUNCH:
    - Creates initial high-quality model listings across categories
    - Sets up featured models for discovery
    - Initializes marketplace with diverse AI models
    - Validates marketplace readiness for public use
    
    🔐 SECURITY:
    - Requires admin permissions to launch marketplace
    - Creates system-owned initial listings
    - Logs launch activities for audit trail
    """
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(current_user)
        if not user or user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required to launch marketplace"
            )
        
        logger.info("Marketplace launch initiated by admin",
                   admin_user_id=current_user)
        
        # Launch marketplace with initial listings
        launch_result = await launch_marketplace_with_initial_listings()
        
        if not launch_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=launch_result.get("message", "Marketplace launch failed")
            )
        
        return LaunchResponse(
            success=True,
            message="🎉 PRSM Marketplace successfully launched with initial model listings!",
            launch_summary=launch_result["launch_summary"],
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to launch marketplace",
                    admin_user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to launch marketplace"
        )


@router.get("/launch/status")
async def get_launch_status(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get marketplace launch readiness status
    
    📊 LAUNCH STATUS:
    - Check if marketplace has sufficient initial listings
    - Validate category and provider coverage
    - Assess marketplace readiness for public launch
    """
    try:
        from prsm.marketplace.initial_listings import initial_listings_creator
        
        # Get launch summary without creating listings
        launch_summary = await initial_listings_creator.get_launch_summary()
        
        return {
            "success": True,
            "launch_ready": launch_summary.get("launch_ready", False),
            "summary": launch_summary,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get launch status",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get launch status"
        )


@router.get("/launch/preview")
async def preview_initial_listings(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Preview the initial listings that would be created during launch
    
    👀 PREVIEW:
    - Shows what models will be added during marketplace launch
    - Displays categories, providers, and pricing information
    - Helps validate launch content before execution
    """
    try:
        from prsm.marketplace.initial_listings import initial_listings_creator
        
        # Get initial listings data without creating them
        preview_data = []
        for listing_data in initial_listings_creator.initial_listings:
            preview_item = {
                "name": listing_data["name"],
                "provider": listing_data["provider"].value,
                "category": listing_data["category"].value,
                "pricing_tier": listing_data["pricing_tier"].value,
                "description": listing_data["description"][:200] + "..." if len(listing_data["description"]) > 200 else listing_data["description"]
            }
            preview_data.append(preview_item)
        
        # Group by category for better overview
        categories = {}
        providers = set()
        pricing_tiers = set()
        
        for item in preview_data:
            category = item["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
            providers.add(item["provider"])
            pricing_tiers.add(item["pricing_tier"])
        
        return {
            "success": True,
            "preview": {
                "listings": preview_data,
                "by_category": categories,
                "total_listings": len(preview_data),
                "categories_count": len(categories),
                "providers": sorted(list(providers)),
                "pricing_tiers": sorted(list(pricing_tiers))
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to preview initial listings",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to preview initial listings"
        )
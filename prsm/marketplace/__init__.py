"""
PRSM Marketplace Module
======================

Comprehensive marketplace for AI model discovery, rental, and collaboration.
Provides secure, decentralized access to AI models with FTNS token integration.

✅ PRODUCTION-READY: Uses real database operations and complete implementations.
All marketplace functionality has been consolidated into real_marketplace_service.py
"""

# Import production-ready service
from .real_marketplace_service import RealMarketplaceService

# Import models
from .models import (
    ModelListing, ModelCategory, PricingTier, ModelMetadata,
    MarketplaceOrder, RentalAgreement, ModelProvider
)

# Create service instance for backwards compatibility
marketplace_service = RealMarketplaceService()

# Alias for backwards compatibility
MarketplaceService = RealMarketplaceService

__all__ = [
    # Services
    "marketplace_service",
    "MarketplaceService",
    # "model_listing_service", 
    # "ModelListingService",
    # "rental_engine",
    # "RentalEngine",
    # "discovery_engine",
    # "DiscoveryEngine",
    
    # Models
    "ModelListing",
    "ModelCategory",
    "PricingTier", 
    "ModelMetadata",
    "MarketplaceOrder",
    "RentalAgreement",
    "ModelProvider"
]
"""
PRSM Model Marketplace
Facilitates model rentals, transactions, and marketplace operations

Migration Notice:
- Migrated from deprecated ftns_service to AtomicFTNSService
- All balance operations now use atomic transactions with idempotency keys
- Race condition vulnerabilities have been addressed
"""

import asyncio
import math
import random
import structlog
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

# Set precision for financial calculations
getcontext().prec = 18

from prsm.core.config import settings
from prsm.core.models import (
    MarketplaceListing, MarketplaceTransaction, PricingModel,
    FTNSTransaction, FTNSBalance
)
from prsm.core.safety.monitor import SafetyMonitor
from .atomic_ftns_service import get_atomic_ftns_service, AtomicFTNSService

logger = structlog.get_logger(__name__)

# === Marketplace Configuration ===

# Platform fees
PLATFORM_FEE_PERCENTAGE = float(getattr(settings, "MARKETPLACE_PLATFORM_FEE", 0.05))  # 5%
TRANSACTION_FEE = float(getattr(settings, "MARKETPLACE_TRANSACTION_FEE", 0.1))  # 0.1 FTNS
LISTING_FEE = float(getattr(settings, "MARKETPLACE_LISTING_FEE", 1.0))  # 1 FTNS

# Escrow settings
ESCROW_PERCENTAGE = float(getattr(settings, "MARKETPLACE_ESCROW_PERCENTAGE", 0.1))  # 10%
ESCROW_RELEASE_DELAY_HOURS = float(getattr(settings, "MARKETPLACE_ESCROW_DELAY", 24.0))

# Quality and reputation settings
MINIMUM_QUALITY_SCORE = float(getattr(settings, "MARKETPLACE_MIN_QUALITY", 0.6))
REPUTATION_WEIGHT = float(getattr(settings, "MARKETPLACE_REPUTATION_WEIGHT", 0.3))

# Revenue sharing
CREATOR_REVENUE_SHARE = float(getattr(settings, "MARKETPLACE_CREATOR_SHARE", 0.85))  # 85%
PLATFORM_REVENUE_SHARE = float(getattr(settings, "MARKETPLACE_PLATFORM_SHARE", 0.15))  # 15%


class ModelMarketplace:
    """
    Model marketplace for renting and trading AI models
    Handles listings, transactions, payments, and platform operations
    
    Migration Note:
        Uses AtomicFTNSService for all FTNS operations to prevent race conditions.
        All balance operations use atomic transactions with idempotency keys.
    """
    
    def __init__(self):
        # Marketplace state
        self.listings: Dict[UUID, MarketplaceListing] = {}
        self.transactions: Dict[UUID, MarketplaceTransaction] = {}
        self.user_ratings: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.model_performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Revenue tracking
        self.revenue_streams: Dict[str, float] = defaultdict(float)
        self.transaction_volume: Dict[str, int] = defaultdict(int)
        
        # Safety integration
        self.safety_monitor = SafetyMonitor()
        
        # Atomic FTNS service (initialized lazily)
        self._ftns_service: Optional[AtomicFTNSService] = None
        
        # Performance statistics
        self.marketplace_stats = {
            "total_listings": 0,
            "active_listings": 0,
            "total_transactions": 0,
            "successful_transactions": 0,
            "total_revenue": 0.0,
            "platform_fees_collected": 0.0,
            "average_transaction_value": 0.0,
            "user_satisfaction_score": 0.0
        }
        
        # Synchronization
        self._listings_lock = asyncio.Lock()
        self._transactions_lock = asyncio.Lock()
        
        logger.info("ModelMarketplace initialized")
    
    async def _get_ftns_service(self) -> AtomicFTNSService:
        """Get or initialize the atomic FTNS service."""
        if self._ftns_service is None:
            self._ftns_service = await get_atomic_ftns_service()
        return self._ftns_service
    
    
    async def list_model_for_rent(self, model_id: str, pricing: PricingModel, 
                                 owner_id: str, listing_details: Dict[str, Any]) -> MarketplaceListing:
        """
        List a model for rent in the marketplace
        
        Args:
            model_id: Unique identifier for the model
            pricing: Pricing model for the rental
            owner_id: Owner of the model
            listing_details: Additional listing information
            
        Returns:
            Created marketplace listing
        """
        try:
            async with self._listings_lock:
                # Validate model safety
                safety_check = await self.safety_monitor.validate_model_output(
                    {"model_id": model_id, "listing_details": listing_details},
                    ["no_malicious_content", "content_appropriateness"]
                )
                
                if not safety_check:
                    raise ValueError("Model failed safety validation")
                
                # Charge listing fee using atomic FTNS service
                ftns = await self._get_ftns_service()
                listing_fee = Decimal(str(LISTING_FEE))
                idempotency_key = f"listing_fee:{owner_id}:{model_id}:{uuid4().hex[:8]}"
                
                fee_result = await ftns.deduct_tokens_atomic(
                    user_id=owner_id,
                    amount=listing_fee,
                    idempotency_key=idempotency_key,
                    description=f"Marketplace listing fee for model {model_id}",
                    metadata={"model_id": model_id, "fee_type": "listing"}
                )
                
                if not fee_result.success:
                    logger.warning("Insufficient balance for listing fee",
                                  owner_id=owner_id, model_id=model_id,
                                  error=fee_result.error_message)
                    raise ValueError(f"Insufficient FTNS balance for listing fee: {fee_result.error_message}")
                
                # Create marketplace listing
                listing = MarketplaceListing(
                    model_id=model_id,
                    owner_id=owner_id,
                    title=listing_details.get("title", f"Model {model_id}"),
                    description=listing_details.get("description", "AI model for rent"),
                    pricing_model=pricing,
                    performance_metrics=listing_details.get("performance_metrics", {}),
                    resource_requirements=listing_details.get("resource_requirements", {}),
                    supported_features=listing_details.get("supported_features", []),
                    terms_of_service=listing_details.get("terms_of_service", "Standard rental terms"),
                    maximum_concurrent_users=listing_details.get("max_concurrent_users", 1),
                    geographical_restrictions=listing_details.get("geo_restrictions", [])
                )
                
                # Store listing
                self.listings[listing.listing_id] = listing
                
                # Update statistics
                self.marketplace_stats["total_listings"] += 1
                self.marketplace_stats["active_listings"] += 1
                self.marketplace_stats["platform_fees_collected"] += LISTING_FEE
                
                logger.info("Model listed for rent",
                           model_id=model_id,
                           listing_id=str(listing.listing_id),
                           owner_id=owner_id)
                
                return listing
                
        except Exception as e:
            logger.error("Error listing model for rent",
                        model_id=model_id,
                        owner_id=owner_id,
                        error=str(e))
            raise
    
    
    async def facilitate_model_transactions(self, buyer_id: str, seller_id: str, 
                                          listing_id: UUID, transaction_details: Dict[str, Any]) -> MarketplaceTransaction:
        """
        Facilitate a model rental/purchase transaction
        
        Args:
            buyer_id: ID of the buyer/renter
            seller_id: ID of the seller/owner
            listing_id: Listing being transacted
            transaction_details: Transaction specifics
            
        Returns:
            Created marketplace transaction
        """
        try:
            async with self._transactions_lock:
                # Get listing
                if listing_id not in self.listings:
                    raise ValueError(f"Listing {listing_id} not found")
                
                listing = self.listings[listing_id]
                
                # Verify ownership
                if listing.owner_id != seller_id:
                    raise ValueError("Seller is not the owner of the listing")
                
                # Verify availability
                if listing.availability_status != "available":
                    raise ValueError(f"Model is not available (status: {listing.availability_status})")
                
                # Calculate transaction amount
                transaction_amount = await self._calculate_transaction_amount(
                    listing, transaction_details
                )
                
                # Calculate platform fee
                platform_fee = await self.calculate_platform_fees(transaction_amount)
                
                # Calculate escrow amount
                escrow_amount = transaction_amount * ESCROW_PERCENTAGE
                
                # Check buyer balance using atomic service
                ftns = await self._get_ftns_service()
                buyer_balance = await ftns.get_balance(buyer_id)
                total_required = Decimal(str(transaction_amount + platform_fee + escrow_amount))
                
                if buyer_balance.available_balance < total_required:
                    logger.warning("Insufficient balance for transaction",
                                  buyer_id=buyer_id,
                                  required=float(total_required),
                                  available=float(buyer_balance.available_balance))
                    raise ValueError(f"Insufficient balance. Required: {total_required}, Available: {buyer_balance.available_balance}")
                
                # Create transaction record
                transaction = MarketplaceTransaction(
                    listing_id=listing_id,
                    buyer_id=buyer_id,
                    seller_id=seller_id,
                    transaction_type=transaction_details.get("type", "rental"),
                    amount=transaction_amount,
                    duration=transaction_details.get("duration"),
                    platform_fee=platform_fee,
                    escrow_amount=escrow_amount,
                    status="pending"
                )
                
                # Execute payment using atomic operations
                payment_success = await self._execute_transaction_payment(transaction)
                
                if payment_success:
                    transaction.status = "completed"
                    transaction.started_at = datetime.now(timezone.utc)
                    
                    # Update listing availability
                    if transaction.transaction_type == "rental":
                        listing.availability_status = "rented"
                    elif transaction.transaction_type == "purchase":
                        listing.availability_status = "sold"
                    
                    # Store transaction
                    self.transactions[transaction.transaction_id] = transaction
                    
                    # Update statistics
                    self.marketplace_stats["total_transactions"] += 1
                    self.marketplace_stats["successful_transactions"] += 1
                    self.marketplace_stats["total_revenue"] += transaction_amount
                    self.marketplace_stats["platform_fees_collected"] += platform_fee
                    
                    # Update average transaction value
                    total_transactions = self.marketplace_stats["successful_transactions"]
                    self.marketplace_stats["average_transaction_value"] = (
                        self.marketplace_stats["total_revenue"] / total_transactions
                    )
                    
                    logger.info("Transaction completed",
                               buyer_id=buyer_id,
                               seller_id=seller_id,
                               amount=transaction_amount,
                               transaction_id=str(transaction.transaction_id))
                    
                else:
                    transaction.status = "failed"
                    self.transactions[transaction.transaction_id] = transaction
                
                return transaction
                
        except Exception as e:
            logger.error("Error facilitating transaction",
                        buyer_id=buyer_id,
                        seller_id=seller_id,
                        listing_id=str(listing_id),
                        error=str(e))
            raise
    
    
    async def calculate_platform_fees(self, transaction_value: float) -> float:
        """
        Calculate platform fees for a transaction
        
        Args:
            transaction_value: Value of the transaction
            
        Returns:
            Platform fee amount
        """
        # Base percentage fee
        percentage_fee = transaction_value * PLATFORM_FEE_PERCENTAGE
        
        # Fixed transaction fee
        fixed_fee = TRANSACTION_FEE
        
        # Total platform fee
        total_fee = percentage_fee + fixed_fee
        
        # Minimum fee protection
        min_fee = 0.01  # 0.01 FTNS minimum
        
        return max(min_fee, total_fee)
    
    
    async def search_listings(self, search_criteria: Dict[str, Any]) -> List[MarketplaceListing]:
        """Search marketplace listings based on criteria"""
        matching_listings = []
        
        for listing in self.listings.values():
            if await self._matches_search_criteria(listing, search_criteria):
                matching_listings.append(listing)
        
        # Sort by relevance/rating
        matching_listings.sort(key=lambda x: self._calculate_listing_score(x), reverse=True)
        
        return matching_listings
    
    
    async def get_user_transaction_history(self, user_id: str) -> List[MarketplaceTransaction]:
        """Get transaction history for a user"""
        user_transactions = []
        
        for transaction in self.transactions.values():
            if transaction.buyer_id == user_id or transaction.seller_id == user_id:
                user_transactions.append(transaction)
        
        # Sort by creation date (newest first)
        user_transactions.sort(key=lambda x: x.created_at, reverse=True)
        
        return user_transactions
    
    
    async def update_model_performance(self, model_id: str, performance_data: Dict[str, Any]) -> bool:
        """Update model performance metrics"""
        try:
            # Record performance data
            performance_record = {
                "timestamp": datetime.now(timezone.utc),
                "metrics": performance_data,
                "source": "marketplace_usage"
            }
            
            self.model_performance_history[model_id].append(performance_record)
            
            # Update listing performance metrics
            for listing in self.listings.values():
                if listing.model_id == model_id:
                    listing.performance_metrics.update(performance_data)
                    listing.updated_at = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            logger.error("Error updating model performance",
                        model_id=model_id,
                        error=str(e))
            return False
    
    
    async def rate_transaction(self, transaction_id: UUID, rater_id: str,
                             rating: float, review: str = "") -> bool:
        """Rate a completed transaction"""
        try:
            if transaction_id not in self.transactions:
                return False
            
            transaction = self.transactions[transaction_id]
            
            # Verify rater is part of transaction
            if rater_id not in [transaction.buyer_id, transaction.seller_id]:
                return False
            
            # Determine who is being rated
            rated_user = transaction.seller_id if rater_id == transaction.buyer_id else transaction.buyer_id
            
            # Store rating
            self.user_ratings[rated_user][rater_id] = rating
            
            # Update user satisfaction score
            await self._update_user_satisfaction_score()
            
            logger.info("Transaction rated",
                       transaction_id=str(transaction_id),
                       rating=rating,
                       rater_id=rater_id,
                       rated_user=rated_user)
            
            return True
            
        except Exception as e:
            logger.error("Error rating transaction",
                        transaction_id=str(transaction_id),
                        error=str(e))
            return False
    
    
    async def get_marketplace_statistics(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        async with self._listings_lock, self._transactions_lock:
            # Update active listings count
            active_count = sum(
                1 for listing in self.listings.values() 
                if listing.availability_status == "available"
            )
            self.marketplace_stats["active_listings"] = active_count
            
            return {
                **self.marketplace_stats,
                "listings_by_status": self._get_listings_by_status(),
                "transactions_by_type": self._get_transactions_by_type(),
                "revenue_breakdown": dict(self.revenue_streams),
                "top_performing_models": await self._get_top_performing_models(),
                "configuration": {
                    "platform_fee_percentage": PLATFORM_FEE_PERCENTAGE,
                    "transaction_fee": TRANSACTION_FEE,
                    "listing_fee": LISTING_FEE,
                    "escrow_percentage": ESCROW_PERCENTAGE
                }
            }
    
    
    # === Private Helper Methods ===
    
    async def _calculate_transaction_amount(self, listing: MarketplaceListing, 
                                          transaction_details: Dict[str, Any]) -> float:
        """Calculate total transaction amount based on pricing model"""
        pricing = listing.pricing_model
        base_price = pricing.base_price
        
        # Apply pricing based on type
        if pricing.pricing_type == "hourly":
            duration = transaction_details.get("duration", 1.0)  # Hours
            amount = base_price * duration
        elif pricing.pricing_type == "usage":
            usage_units = transaction_details.get("usage_units", 1.0)
            amount = base_price * usage_units
        elif pricing.pricing_type == "subscription":
            subscription_period = transaction_details.get("subscription_period", 1)  # Months
            amount = base_price * subscription_period
        else:  # one_time
            amount = base_price
        
        # Apply volume discounts
        if pricing.volume_discounts:
            amount = self._apply_volume_discounts(amount, pricing.volume_discounts)
        
        # Apply dynamic pricing
        if pricing.dynamic_pricing_enabled:
            amount = await self._apply_dynamic_pricing(amount, listing, pricing)
        
        return amount
    
    
    def _apply_volume_discounts(self, amount: float, volume_discounts: Dict[str, float]) -> float:
        """Apply volume discounts to transaction amount"""
        # Sort discount tiers by threshold
        sorted_tiers = sorted(
            [(float(threshold), discount) for threshold, discount in volume_discounts.items()],
            key=lambda x: x[0],
            reverse=True
        )
        
        # Apply highest applicable discount
        for threshold, discount in sorted_tiers:
            if amount >= threshold:
                return amount * (1 - discount)
        
        return amount
    
    
    async def _apply_dynamic_pricing(self, amount: float, listing: MarketplaceListing, 
                                   pricing: PricingModel) -> float:
        """Apply dynamic pricing based on demand and other factors"""
        # Simulate demand calculation
        model_demand = await self._calculate_model_demand(listing.model_id)
        
        # Apply demand multiplier
        amount *= (1 + model_demand * pricing.demand_multiplier)
        
        # Apply peak hour multiplier (simulate peak hours)
        current_hour = datetime.now(timezone.utc).hour
        if 9 <= current_hour <= 17:  # Business hours UTC
            amount *= pricing.peak_hour_multiplier
        
        return amount
    
    
    async def _calculate_model_demand(self, model_id: str) -> float:
        """Calculate current demand for a model (0.0 to 1.0)"""
        # Count recent transactions for this model
        recent_transactions = 0
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for transaction in self.transactions.values():
            listing = self.listings.get(transaction.listing_id)
            if listing and listing.model_id == model_id and transaction.created_at >= cutoff_time:
                recent_transactions += 1
        
        # Normalize to 0-1 scale (assuming max 10 transactions/day = high demand)
        demand = min(1.0, recent_transactions / 10.0)
        
        return demand
    
    
    async def _execute_transaction_payment(self, transaction: MarketplaceTransaction) -> bool:
        """
        Execute the payment for a transaction using atomic FTNS operations.
        
        This method uses atomic transfers to prevent race conditions during
        payment processing. All operations use idempotency keys for safety.
        """
        try:
            ftns = await self._get_ftns_service()
            
            # Calculate amounts
            transaction_amount = Decimal(str(transaction.amount))
            platform_fee = Decimal(str(transaction.platform_fee))
            escrow_amount = Decimal(str(transaction.escrow_amount))
            total_deduction = transaction_amount + platform_fee + escrow_amount
            seller_amount = Decimal(str(transaction.amount * CREATOR_REVENUE_SHARE))
            
            # Generate unique idempotency keys for each operation
            tx_id_short = str(transaction.transaction_id)[:8]
            
            # Step 1: Deduct total amount from buyer atomically
            buyer_deduct_key = f"marketplace_tx:{transaction.transaction_id}:buyer_deduct"
            buyer_result = await ftns.deduct_tokens_atomic(
                user_id=transaction.buyer_id,
                amount=total_deduction,
                idempotency_key=buyer_deduct_key,
                description=f"Marketplace transaction {transaction.transaction_id}",
                metadata={
                    "transaction_id": str(transaction.transaction_id),
                    "transaction_type": transaction.transaction_type,
                    "listing_id": str(transaction.listing_id)
                }
            )
            
            if not buyer_result.success:
                logger.warning("Failed to deduct from buyer",
                              transaction_id=str(transaction.transaction_id),
                              buyer_id=transaction.buyer_id,
                              error=buyer_result.error_message)
                return False
            
            # Step 2: Transfer seller amount atomically
            seller_transfer_key = f"marketplace_tx:{transaction.transaction_id}:seller_transfer"
            seller_result = await ftns.mint_tokens_atomic(
                to_user_id=transaction.seller_id,
                amount=seller_amount,
                idempotency_key=seller_transfer_key,
                description=f"Marketplace sale payment for {transaction.transaction_id}",
                metadata={
                    "transaction_id": str(transaction.transaction_id),
                    "buyer_id": transaction.buyer_id,
                    "sale_type": "marketplace_sale"
                }
            )
            
            if not seller_result.success:
                # Attempt to refund buyer
                refund_key = f"marketplace_tx:{transaction.transaction_id}:refund"
                await ftns.mint_tokens_atomic(
                    to_user_id=transaction.buyer_id,
                    amount=total_deduction,
                    idempotency_key=refund_key,
                    description=f"Refund for failed transaction {transaction.transaction_id}"
                )
                logger.error("Failed to pay seller, refunding buyer",
                            transaction_id=str(transaction.transaction_id),
                            seller_id=transaction.seller_id)
                return False
            
            # Schedule escrow release
            asyncio.create_task(self._schedule_escrow_release(transaction))
            
            logger.info("Transaction payment completed",
                       transaction_id=str(transaction.transaction_id),
                       buyer_id=transaction.buyer_id,
                       seller_id=transaction.seller_id,
                       amount=float(transaction_amount),
                       platform_fee=float(platform_fee),
                       escrow=float(escrow_amount))
            
            return True
            
        except Exception as e:
            logger.error("Transaction payment failed",
                        transaction_id=str(transaction.transaction_id),
                        error=str(e))
            return False
    
    
    async def _schedule_escrow_release(self, transaction: MarketplaceTransaction):
        """Schedule release of escrow amount after delay"""
        try:
            # Wait for escrow release delay
            await asyncio.sleep(ESCROW_RELEASE_DELAY_HOURS * 3600)  # Convert hours to seconds
            
            # Release escrow to seller using atomic mint
            ftns = await self._get_ftns_service()
            escrow_amount = Decimal(str(transaction.escrow_amount))
            escrow_release_key = f"marketplace_escrow:{transaction.transaction_id}:release"
            
            result = await ftns.mint_tokens_atomic(
                to_user_id=transaction.seller_id,
                amount=escrow_amount,
                idempotency_key=escrow_release_key,
                description=f"Escrow release for transaction {transaction.transaction_id}",
                metadata={
                    "transaction_id": str(transaction.transaction_id),
                    "release_type": "escrow"
                }
            )
            
            if result.success:
                logger.info("Escrow released",
                           transaction_id=str(transaction.transaction_id),
                           seller_id=transaction.seller_id,
                           amount=float(escrow_amount))
            else:
                logger.error("Failed to release escrow",
                            transaction_id=str(transaction.transaction_id),
                            error=result.error_message)
            
        except Exception as e:
            logger.error("Error releasing escrow",
                        transaction_id=str(transaction.transaction_id),
                        error=str(e))
    
    
    async def _matches_search_criteria(self, listing: MarketplaceListing, 
                                     criteria: Dict[str, Any]) -> bool:
        """Check if listing matches search criteria"""
        # Price range filter
        if "max_price" in criteria:
            if listing.pricing_model.base_price > criteria["max_price"]:
                return False
        
        if "min_price" in criteria:
            if listing.pricing_model.base_price < criteria["min_price"]:
                return False
        
        # Feature filter
        if "required_features" in criteria:
            required_features = set(criteria["required_features"])
            listing_features = set(listing.supported_features)
            if not required_features.issubset(listing_features):
                return False
        
        # Availability filter
        if "availability_status" in criteria:
            if listing.availability_status != criteria["availability_status"]:
                return False
        
        # Text search in title and description
        if "search_text" in criteria:
            search_text = criteria["search_text"].lower()
            if (search_text not in listing.title.lower() and 
                search_text not in listing.description.lower()):
                return False
        
        return True
    
    
    def _calculate_listing_score(self, listing: MarketplaceListing) -> float:
        """Calculate relevance/quality score for listing"""
        score = 0.0
        
        # Performance metrics contribution
        if listing.performance_metrics:
            avg_performance = sum(listing.performance_metrics.values()) / len(listing.performance_metrics)
            score += avg_performance * 0.4
        
        # Pricing competitiveness (lower price = higher score, normalized)
        price_score = max(0, 1.0 - (listing.pricing_model.base_price / 100.0))
        score += price_score * 0.3
        
        # Owner reputation (simulated)
        owner_reputation = self._get_user_reputation(listing.owner_id)
        score += owner_reputation * 0.3
        
        return score
    
    
    def _get_user_reputation(self, user_id: str) -> float:
        """Get user reputation score (0.0 to 1.0)"""
        if user_id not in self.user_ratings:
            return 0.5  # Neutral for new users
        
        user_ratings_received = self.user_ratings[user_id]
        if not user_ratings_received:
            return 0.5
        
        avg_rating = sum(user_ratings_received.values()) / len(user_ratings_received)
        return avg_rating / 5.0  # Normalize to 0-1
    
    
    async def _update_user_satisfaction_score(self):
        """Update global user satisfaction score"""
        all_ratings = []
        for user_ratings in self.user_ratings.values():
            all_ratings.extend(user_ratings.values())
        
        if all_ratings:
            avg_rating = sum(all_ratings) / len(all_ratings)
            self.marketplace_stats["user_satisfaction_score"] = avg_rating / 5.0
    
    
    def _get_listings_by_status(self) -> Dict[str, int]:
        """Get count of listings by status"""
        status_counts = defaultdict(int)
        for listing in self.listings.values():
            status_counts[listing.availability_status] += 1
        return dict(status_counts)
    
    
    def _get_transactions_by_type(self) -> Dict[str, int]:
        """Get count of transactions by type"""
        type_counts = defaultdict(int)
        for transaction in self.transactions.values():
            type_counts[transaction.transaction_type] += 1
        return dict(type_counts)
    
    
    async def _get_top_performing_models(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing models by revenue"""
        model_revenues = defaultdict(float)
        
        for transaction in self.transactions.values():
            if transaction.status == "completed":
                listing = self.listings.get(transaction.listing_id)
                if listing:
                    model_revenues[listing.model_id] += transaction.amount
        
        # Sort by revenue
        sorted_models = sorted(
            model_revenues.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        return [
            {"model_id": model_id, "revenue": revenue}
            for model_id, revenue in sorted_models
        ]


# === Global Model Marketplace Instance ===

_marketplace_instance: Optional[ModelMarketplace] = None

def get_marketplace() -> ModelMarketplace:
    """Get or create the global model marketplace instance"""
    global _marketplace_instance
    if _marketplace_instance is None:
        _marketplace_instance = ModelMarketplace()
    return _marketplace_instance
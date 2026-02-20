"""
PRSM Advanced FTNS Economy
Enhanced tokenomics features including dynamic pricing, dividend distribution, and research impact tracking

Migration Notice:
- Migrated from deprecated ftns_service to AtomicFTNSService
- All balance operations now use atomic transactions with idempotency keys
- Race condition vulnerabilities have been addressed
"""

import asyncio
import math
import statistics
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
    ImpactMetrics, DividendDistribution, RoyaltyPayment,
    FTNSTransaction, FTNSBalance
)
from .atomic_ftns_service import get_atomic_ftns_service, AtomicFTNSService

logger = structlog.get_logger(__name__)

# === Advanced FTNS Configuration ===

# Dynamic pricing parameters
BASE_CONTEXT_PRICE = float(getattr(settings, "FTNS_BASE_CONTEXT_PRICE", 0.1))
DEMAND_ELASTICITY = float(getattr(settings, "FTNS_DEMAND_ELASTICITY", 0.3))
SUPPLY_ELASTICITY = float(getattr(settings, "FTNS_SUPPLY_ELASTICITY", 0.2))
PRICE_VOLATILITY_DAMPING = float(getattr(settings, "FTNS_PRICE_DAMPING", 0.1))

# Dividend distribution parameters
QUARTERLY_DISTRIBUTION_PERCENTAGE = float(getattr(settings, "FTNS_QUARTERLY_PERCENTAGE", 0.05))  # 5% of total pool
MINIMUM_HOLDING_PERIOD_DAYS = int(getattr(settings, "FTNS_MIN_HOLDING_DAYS", 30))
DIVIDEND_BONUS_MULTIPLIER = float(getattr(settings, "FTNS_DIVIDEND_BONUS", 1.2))

# Research impact parameters
CITATION_VALUE = float(getattr(settings, "FTNS_CITATION_VALUE", 10.0))
DOWNLOAD_VALUE = float(getattr(settings, "FTNS_DOWNLOAD_VALUE", 0.1))
USAGE_HOUR_VALUE = float(getattr(settings, "FTNS_USAGE_HOUR_VALUE", 1.0))
COLLABORATION_BONUS = float(getattr(settings, "FTNS_COLLABORATION_BONUS", 1.5))

# Royalty system parameters
BASE_ROYALTY_RATE = float(getattr(settings, "FTNS_BASE_ROYALTY_RATE", 0.1))  # 10%
IMPACT_ROYALTY_MULTIPLIER = float(getattr(settings, "FTNS_IMPACT_ROYALTY_MULTIPLIER", 2.0))
QUALITY_ROYALTY_MULTIPLIER = float(getattr(settings, "FTNS_QUALITY_ROYALTY_MULTIPLIER", 1.5))


class AdvancedFTNSEconomy:
    """
    Advanced FTNS economy features for sophisticated tokenomics
    Includes dynamic pricing, dividend distribution, and research impact tracking
    
    Migration Note:
        Uses AtomicFTNSService for all FTNS operations to prevent race conditions.
        All balance operations use atomic transactions with idempotency keys.
    """
    
    def __init__(self):
        # Price tracking
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.demand_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.supply_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Impact tracking
        self.impact_metrics: Dict[str, ImpactMetrics] = {}
        self.research_citations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Dividend management
        self.dividend_distributions: Dict[str, DividendDistribution] = {}
        self.eligible_holders: Dict[str, List[str]] = {}
        
        # Royalty tracking
        self.royalty_payments: Dict[UUID, RoyaltyPayment] = {}
        self.content_usage_log: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Atomic FTNS service (initialized lazily)
        self._ftns_service: Optional[AtomicFTNSService] = None
        
        # Performance statistics
        self.economy_stats = {
            "total_context_purchased": 0,
            "total_dividends_distributed": 0.0,
            "total_royalties_paid": 0.0,
            "total_research_impact_value": 0.0,
            "price_calculations": 0,
            "dividend_distributions": 0,
            "royalty_payments": 0
        }
        
        # Synchronization
        self._pricing_lock = asyncio.Lock()
        self._dividend_lock = asyncio.Lock()
        self._royalty_lock = asyncio.Lock()
        
        logger.info("AdvancedFTNSEconomy initialized")
    
    async def _get_ftns_service(self) -> AtomicFTNSService:
        """Get or initialize the atomic FTNS service."""
        if self._ftns_service is None:
            self._ftns_service = await get_atomic_ftns_service()
        return self._ftns_service
    
    
    async def calculate_context_pricing(self, demand: float, supply: float) -> float:
        """
        Calculate dynamic context pricing based on supply and demand
        
        Args:
            demand: Current demand level (0.0 to 1.0)
            supply: Current supply level (0.0 to 1.0)
            
        Returns:
            Dynamic price for context units
        """
        try:
            async with self._pricing_lock:
                current_time = datetime.now(timezone.utc)
                
                # Calculate demand/supply ratio
                if supply == 0:
                    supply = 0.001  # Prevent division by zero
                
                demand_supply_ratio = demand / supply
                
                # Apply elasticity adjustments
                demand_adjustment = 1 + (demand - 0.5) * DEMAND_ELASTICITY
                supply_adjustment = 1 - (supply - 0.5) * SUPPLY_ELASTICITY
                
                # Calculate raw price
                raw_price = BASE_CONTEXT_PRICE * demand_adjustment * supply_adjustment
                
                # Apply volatility damping based on recent price history
                price_smoothing = await self._calculate_price_smoothing("context")
                
                # Calculate final price
                final_price = raw_price * (1 + price_smoothing * PRICE_VOLATILITY_DAMPING)
                
                # Ensure price bounds (10% to 500% of base price)
                min_price = BASE_CONTEXT_PRICE * 0.1
                max_price = BASE_CONTEXT_PRICE * 5.0
                final_price = max(min_price, min(max_price, final_price))
                
                # Record price history
                self.price_history["context"].append((current_time, final_price))
                self.demand_history["context"].append((current_time, demand))
                self.supply_history["context"].append((current_time, supply))
                
                # Limit history size
                if len(self.price_history["context"]) > 1000:
                    self.price_history["context"] = self.price_history["context"][-500:]
                    self.demand_history["context"] = self.demand_history["context"][-500:]
                    self.supply_history["context"] = self.supply_history["context"][-500:]
                
                # Update statistics
                self.economy_stats["price_calculations"] += 1
                
                logger.info("Context pricing calculated",
                           demand=demand,
                           supply=supply,
                           price=final_price)
                
                return final_price
                
        except Exception as e:
            logger.error("Error calculating context pricing", error=str(e))
            return BASE_CONTEXT_PRICE
    
    
    async def distribute_quarterly_dividends(self, token_holders: List[str]) -> DividendDistribution:
        """
        Distribute quarterly dividends to eligible token holders
        
        Args:
            token_holders: List of token holder IDs
            
        Returns:
            Dividend distribution record
        """
        try:
            async with self._dividend_lock:
                current_quarter = self._get_current_quarter()
                distribution_id = uuid4()
                
                # Check if dividends already distributed this quarter
                if current_quarter in self.dividend_distributions:
                    existing = self.dividend_distributions[current_quarter]
                    logger.info("Dividends already distributed for quarter",
                               quarter=current_quarter)
                    return existing
                
                # Calculate total dividend pool
                total_pool = await self._calculate_dividend_pool()
                
                # Filter eligible holders
                eligible_holders = await self._filter_eligible_holders(token_holders)
                
                # Calculate individual distribution amounts
                distribution_amounts = await self._calculate_distribution_amounts(
                    eligible_holders, total_pool
                )
                
                # Apply bonus multipliers
                bonus_multipliers = await self._calculate_bonus_multipliers(eligible_holders)
                
                # Create dividend distribution record
                distribution = DividendDistribution(
                    distribution_id=distribution_id,
                    quarter=current_quarter,
                    total_pool=total_pool,
                    distribution_date=datetime.now(timezone.utc),
                    eligible_holders=eligible_holders,
                    distribution_amounts=distribution_amounts,
                    bonus_multipliers=bonus_multipliers,
                    status="processing"
                )
                
                # Execute distributions using atomic operations
                successful_distributions = 0
                for holder_id, amount in distribution_amounts.items():
                    # Apply bonus multiplier
                    bonus = bonus_multipliers.get(holder_id, 1.0)
                    final_amount = amount * bonus
                    
                    # Distribute to holder using atomic operation
                    success = await self._distribute_to_holder(holder_id, final_amount, current_quarter)
                    if success:
                        successful_distributions += 1
                
                # Update distribution status
                if successful_distributions == len(distribution_amounts):
                    distribution.status = "completed"
                else:
                    distribution.status = "partial"
                
                # Store distribution record
                self.dividend_distributions[current_quarter] = distribution
                
                # Update statistics
                self.economy_stats["dividend_distributions"] += 1
                self.economy_stats["total_dividends_distributed"] += total_pool
                
                logger.info("Dividends distributed",
                           quarter=current_quarter,
                           total_pool=total_pool,
                           successful=successful_distributions,
                           total_holders=len(distribution_amounts))
                
                return distribution
                
        except Exception as e:
            logger.error("Error distributing dividends", error=str(e))
            return DividendDistribution(
                quarter=self._get_current_quarter(),
                total_pool=0.0,
                distribution_date=datetime.now(timezone.utc),
                eligible_holders=[],
                status="failed"
            )
    
    
    async def track_research_impact(self, research_cid: str) -> ImpactMetrics:
        """
        Track and calculate research impact metrics
        
        Args:
            research_cid: Content ID of the research
            
        Returns:
            Updated impact metrics
        """
        try:
            # Get or create impact metrics
            if research_cid in self.impact_metrics:
                metrics = self.impact_metrics[research_cid]
            else:
                metrics = ImpactMetrics(content_id=research_cid)
            
            # Update usage statistics (simulated real-world data)
            await self._update_usage_statistics(metrics)
            
            # Calculate impact score
            impact_score = await self._calculate_impact_score(metrics)
            metrics.impact_score = impact_score
            
            # Update timestamps
            metrics.calculated_at = datetime.now(timezone.utc)
            
            # Store updated metrics
            self.impact_metrics[research_cid] = metrics
            
            # Update global statistics
            self.economy_stats["total_research_impact_value"] += impact_score
            
            logger.info("Research impact tracked",
                       research_cid=research_cid[:8],
                       impact_score=impact_score)
            
            return metrics
            
        except Exception as e:
            logger.error("Error tracking research impact",
                        research_cid=research_cid,
                        error=str(e))
            return ImpactMetrics(content_id=research_cid)
    
    
    async def implement_royalty_system(self, content_usage: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Implement royalty payment system for content usage
        
        Args:
            content_usage: List of content usage records
            
        Returns:
            Royalty processing results
        """
        try:
            async with self._royalty_lock:
                royalty_results = {
                    "payments_processed": 0,
                    "total_amount_paid": 0.0,
                    "payment_details": [],
                    "failed_payments": []
                }
                
                # Process each usage record
                for usage_record in content_usage:
                    payment_result = await self._process_royalty_payment(usage_record)
                    
                    if payment_result["success"]:
                        royalty_results["payments_processed"] += 1
                        royalty_results["total_amount_paid"] += payment_result["amount"]
                        royalty_results["payment_details"].append(payment_result)
                    else:
                        royalty_results["failed_payments"].append(payment_result)
                
                # Update statistics
                self.economy_stats["royalty_payments"] += royalty_results["payments_processed"]
                self.economy_stats["total_royalties_paid"] += royalty_results["total_amount_paid"]
                
                logger.info("Royalty payments processed",
                           payments_count=royalty_results["payments_processed"],
                           total_amount=royalty_results["total_amount_paid"])
                
                return royalty_results
                
        except Exception as e:
            logger.error("Error implementing royalty system", error=str(e))
            return {
                "payments_processed": 0,
                "total_amount_paid": 0.0,
                "payment_details": [],
                "failed_payments": [],
                "error": str(e)
            }
    
    
    async def get_price_history(self, resource_type: str = "context", hours: int = 24) -> List[Tuple[datetime, float]]:
        """Get price history for a resource type"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        if resource_type not in self.price_history:
            return []
        
        return [
            (timestamp, price) for timestamp, price in self.price_history[resource_type]
            if timestamp >= cutoff_time
        ]
    
    
    async def get_economy_statistics(self) -> Dict[str, Any]:
        """Get advanced economy statistics"""
        return {
            **self.economy_stats,
            "price_history_size": {
                resource: len(history) for resource, history in self.price_history.items()
            },
            "tracked_research_items": len(self.impact_metrics),
            "dividend_distributions_count": len(self.dividend_distributions),
            "royalty_payments_count": len(self.royalty_payments),
            "configuration": {
                "base_context_price": BASE_CONTEXT_PRICE,
                "demand_elasticity": DEMAND_ELASTICITY,
                "quarterly_distribution": QUARTERLY_DISTRIBUTION_PERCENTAGE,
                "base_royalty_rate": BASE_ROYALTY_RATE
            }
        }
    
    
    # === Private Helper Methods ===
    
    async def _calculate_price_smoothing(self, resource_type: str) -> float:
        """Calculate price smoothing factor based on recent volatility"""
        if resource_type not in self.price_history or len(self.price_history[resource_type]) < 5:
            return 0.0
        
        recent_prices = [price for _, price in self.price_history[resource_type][-10:]]
        
        if len(recent_prices) < 2:
            return 0.0
        
        # Calculate price volatility (coefficient of variation)
        mean_price = statistics.mean(recent_prices)
        if mean_price == 0:
            return 0.0
        
        std_price = statistics.stdev(recent_prices)
        volatility = std_price / mean_price
        
        # Return smoothing factor (higher volatility = more smoothing)
        return min(0.5, volatility)
    
    
    def _get_current_quarter(self) -> str:
        """Get current quarter string"""
        now = datetime.now(timezone.utc)
        quarter = (now.month - 1) // 3 + 1
        return f"{now.year}-Q{quarter}"
    
    
    async def _calculate_dividend_pool(self) -> float:
        """Calculate total dividend pool for the quarter"""
        # Simulate total FTNS in circulation
        total_circulation = 1000000.0  # 1M FTNS
        
        # Calculate quarterly pool
        quarterly_pool = total_circulation * QUARTERLY_DISTRIBUTION_PERCENTAGE
        
        return quarterly_pool
    
    
    async def _filter_eligible_holders(self, token_holders: List[str]) -> List[str]:
        """Filter token holders eligible for dividends"""
        eligible = []
        ftns = await self._get_ftns_service()
        
        for holder_id in token_holders:
            # Check minimum holding period (simulated)
            holding_period_days = 45  # Simulated
            
            # Check minimum balance using atomic service
            balance = await ftns.get_balance(holder_id)
            
            if holding_period_days >= MINIMUM_HOLDING_PERIOD_DAYS and balance.balance >= Decimal("1.0"):
                eligible.append(holder_id)
        
        return eligible
    
    
    async def _calculate_distribution_amounts(self, eligible_holders: List[str], total_pool: float) -> Dict[str, float]:
        """Calculate individual distribution amounts"""
        distribution_amounts = {}
        
        if not eligible_holders:
            return distribution_amounts
        
        ftns = await self._get_ftns_service()
        
        # Get balances for proportional distribution
        total_eligible_balance = Decimal("0.0")
        holder_balances = {}
        
        for holder_id in eligible_holders:
            balance = await ftns.get_balance(holder_id)
            holder_balances[holder_id] = balance.balance
            total_eligible_balance += balance.balance
        
        # Calculate proportional amounts
        for holder_id in eligible_holders:
            if total_eligible_balance > 0:
                proportion = float(holder_balances[holder_id] / total_eligible_balance)
                amount = total_pool * proportion
                distribution_amounts[holder_id] = amount
        
        return distribution_amounts
    
    
    async def _calculate_bonus_multipliers(self, eligible_holders: List[str]) -> Dict[str, float]:
        """Calculate bonus multipliers for holders"""
        bonus_multipliers = {}
        ftns = await self._get_ftns_service()
        
        for holder_id in eligible_holders:
            # Base multiplier
            multiplier = 1.0
            
            # Long-term holding bonus (simulated)
            holding_period_days = 60  # Simulated
            if holding_period_days >= 90:
                multiplier *= DIVIDEND_BONUS_MULTIPLIER
            
            # High balance bonus (simulated)
            balance = await ftns.get_balance(holder_id)
            if balance.balance >= 10000:  # 10K FTNS
                multiplier *= 1.1
            
            bonus_multipliers[holder_id] = multiplier
        
        return bonus_multipliers
    
    
    async def _distribute_to_holder(self, holder_id: str, amount: float, quarter: str) -> bool:
        """
        Distribute dividend to individual holder using atomic operations.
        
        Uses idempotency keys to prevent duplicate distributions.
        """
        try:
            ftns = await self._get_ftns_service()
            
            # Generate idempotency key for this distribution
            idempotency_key = f"dividend:{quarter}:{holder_id}"
            
            # Use atomic mint for distribution
            result = await ftns.mint_tokens_atomic(
                to_user_id=holder_id,
                amount=Decimal(str(amount)),
                idempotency_key=idempotency_key,
                description=f"Dividend distribution for {quarter}",
                metadata={
                    "distribution_type": "dividend",
                    "quarter": quarter
                }
            )
            
            if not result.success:
                logger.warning("Failed to distribute dividend",
                              holder_id=holder_id,
                              quarter=quarter,
                              error=result.error_message)
            
            return result.success
            
        except Exception as e:
            logger.error("Failed to distribute to holder",
                        holder_id=holder_id,
                        quarter=quarter,
                        error=str(e))
            return False
    
    
    async def _update_usage_statistics(self, metrics: ImpactMetrics) -> None:
        """Update usage statistics for research content"""
        # Simulate real-world usage data updates
        import random
        
        # Simulate new citations
        new_citations = random.randint(0, 5)
        metrics.total_citations += new_citations
        metrics.academic_citations += random.randint(0, new_citations)
        
        # Simulate new downloads
        new_downloads = random.randint(0, 50)
        metrics.total_downloads += new_downloads
        
        # Simulate usage hours
        new_usage_hours = random.uniform(0, 100)
        metrics.total_usage_hours += new_usage_hours
        
        # Simulate unique users
        new_unique_users = random.randint(0, 20)
        metrics.unique_users += new_unique_users
        
        # Update geographical reach
        metrics.geographical_reach += random.randint(0, 2)
        
        # Update collaboration coefficient
        if metrics.total_citations > 0:
            metrics.collaboration_coefficient = min(2.0, metrics.academic_citations / metrics.total_citations)
    
    
    async def _calculate_impact_score(self, metrics: ImpactMetrics) -> float:
        """Calculate overall impact score"""
        # Base score from different metrics
        citation_score = metrics.total_citations * CITATION_VALUE
        download_score = metrics.total_downloads * DOWNLOAD_VALUE
        usage_score = metrics.total_usage_hours * USAGE_HOUR_VALUE
        
        # Collaboration bonus
        collaboration_bonus = metrics.collaboration_coefficient * COLLABORATION_BONUS
        
        # Geographical reach bonus
        reach_bonus = min(10.0, metrics.geographical_reach * 0.5)
        
        # Calculate total impact score
        total_score = (citation_score + download_score + usage_score) * (1 + collaboration_bonus + reach_bonus)
        
        return total_score
    
    
    async def _process_royalty_payment(self, usage_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process individual royalty payment using atomic operations.
        
        Uses idempotency keys to prevent duplicate payments.
        """
        try:
            content_id = usage_record.get("content_id")
            creator_id = usage_record.get("creator_id")
            usage_amount = usage_record.get("usage_amount", 0.0)
            usage_type = usage_record.get("usage_type", "download")
            
            if not content_id or not creator_id:
                return {"success": False, "error": "Missing content_id or creator_id"}
            
            # Calculate base royalty
            base_amount = usage_amount * BASE_ROYALTY_RATE
            
            # Get impact multiplier
            impact_multiplier = 1.0
            if content_id in self.impact_metrics:
                metrics = self.impact_metrics[content_id]
                # Higher impact = higher royalty
                impact_multiplier = min(IMPACT_ROYALTY_MULTIPLIER, 1.0 + metrics.impact_score / 1000.0)
            
            # Quality multiplier (simulated)
            quality_score = usage_record.get("quality_score", 1.0)
            quality_multiplier = min(QUALITY_ROYALTY_MULTIPLIER, quality_score)
            
            # Calculate total amount
            total_amount = base_amount * impact_multiplier * quality_multiplier
            
            # Create royalty payment record
            royalty_payment = RoyaltyPayment(
                content_id=content_id,
                creator_id=creator_id,
                usage_period_start=usage_record.get("period_start", datetime.now(timezone.utc)),
                usage_period_end=usage_record.get("period_end", datetime.now(timezone.utc)),
                total_usage=usage_amount,
                usage_type=usage_type,
                royalty_rate=BASE_ROYALTY_RATE,
                base_amount=base_amount,
                total_amount=total_amount,
                impact_multiplier=impact_multiplier,
                quality_score=quality_score,
                status="pending"
            )
            
            # Execute payment using atomic mint
            ftns = await self._get_ftns_service()
            idempotency_key = f"royalty:{royalty_payment.payment_id}"
            
            result = await ftns.mint_tokens_atomic(
                to_user_id=creator_id,
                amount=Decimal(str(total_amount)),
                idempotency_key=idempotency_key,
                description=f"Royalty payment for {usage_type} - {content_id[:8]}",
                metadata={
                    "payment_type": "royalty",
                    "content_id": content_id,
                    "usage_type": usage_type,
                    "impact_multiplier": impact_multiplier,
                    "quality_score": quality_score
                }
            )
            
            if result.success:
                royalty_payment.status = "paid"
                royalty_payment.payment_date = datetime.now(timezone.utc)
                
                # Store payment record
                self.royalty_payments[royalty_payment.payment_id] = royalty_payment
                
                logger.info("Royalty payment processed",
                           payment_id=royalty_payment.payment_id,
                           creator_id=creator_id,
                           amount=total_amount,
                           usage_type=usage_type)
                
                return {
                    "success": True,
                    "payment_id": royalty_payment.payment_id,
                    "creator_id": creator_id,
                    "amount": total_amount,
                    "usage_type": usage_type
                }
            else:
                royalty_payment.status = "failed"
                logger.warning("Royalty payment failed",
                              payment_id=royalty_payment.payment_id,
                              creator_id=creator_id,
                              error=result.error_message)
                return {"success": False, "error": result.error_message or "Payment execution failed"}
                
        except Exception as e:
            logger.error("Error processing royalty payment", error=str(e))
            return {"success": False, "error": str(e)}


# === Global Advanced FTNS Economy Instance ===

_advanced_ftns_instance: Optional[AdvancedFTNSEconomy] = None

def get_advanced_ftns() -> AdvancedFTNSEconomy:
    """Get or create the global advanced FTNS economy instance"""
    global _advanced_ftns_instance
    if _advanced_ftns_instance is None:
        _advanced_ftns_instance = AdvancedFTNSEconomy()
    return _advanced_ftns_instance
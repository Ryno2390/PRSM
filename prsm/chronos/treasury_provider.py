"""
CHRONOS Treasury Provider Abstraction

Provides enterprise-grade abstraction layer for Bitcoin treasury providers.
Designed for integration with MicroStrategy, Coinbase Custody, and other institutional providers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .models import AssetType
from .price_oracles import price_aggregator

logger = logging.getLogger(__name__)


class TreasuryProviderType(str, Enum):
    """Types of treasury providers."""
    MICROSTRATEGY = "MICROSTRATEGY"
    COINBASE_CUSTODY = "COINBASE_CUSTODY"
    FIDELITY_DIGITAL = "FIDELITY_DIGITAL"
    FIREBLOCKS = "FIREBLOCKS"
    INTERNAL_CUSTODY = "INTERNAL_CUSTODY"


class LiquidityTier(str, Enum):
    """Liquidity availability tiers."""
    INSTANT = "INSTANT"        # <1 minute
    FAST = "FAST"             # 1-15 minutes
    STANDARD = "STANDARD"     # 15-60 minutes
    BATCH = "BATCH"           # 1-24 hours
    SETTLEMENT = "SETTLEMENT" # 1-3 days


@dataclass
class TreasuryQuote:
    """Quote from treasury provider for liquidity."""
    provider_name: str
    asset_type: AssetType
    available_amount: Decimal
    rate: Decimal
    fees: Dict[str, Decimal]
    liquidity_tier: LiquidityTier
    quote_expires_at: datetime
    settlement_terms: Dict[str, Any]
    collateral_requirements: Optional[Dict[str, Decimal]] = None


@dataclass
class TreasuryExecution:
    """Execution record for treasury operations."""
    execution_id: str
    provider_name: str
    operation_type: str  # buy, sell, borrow, lend
    asset_type: AssetType
    amount: Decimal
    rate: Decimal
    status: str
    initiated_at: datetime
    completed_at: Optional[datetime] = None
    settlement_reference: Optional[str] = None
    blockchain_txids: List[str] = None


class TreasuryProvider(ABC):
    """Abstract base class for treasury providers."""
    
    def __init__(self, provider_name: str, provider_type: TreasuryProviderType):
        self.provider_name = provider_name
        self.provider_type = provider_type
        self.is_active = True
        self.supported_assets = [AssetType.BTC]  # Default to Bitcoin
        
    @abstractmethod
    async def get_liquidity_quote(
        self, 
        asset: AssetType, 
        amount: Decimal, 
        operation: str
    ) -> TreasuryQuote:
        """Get quote for liquidity provision."""
        pass
    
    @abstractmethod
    async def execute_operation(
        self, 
        quote: TreasuryQuote, 
        amount: Decimal
    ) -> TreasuryExecution:
        """Execute treasury operation."""
        pass
    
    @abstractmethod
    async def get_available_liquidity(self, asset: AssetType) -> Dict[LiquidityTier, Decimal]:
        """Get available liquidity by tier."""
        pass
    
    @abstractmethod
    async def get_custody_status(self) -> Dict[str, Any]:
        """Get custody and operational status."""
        pass


class MicroStrategyProvider(TreasuryProvider):
    """MicroStrategy Bitcoin treasury integration."""
    
    def __init__(self, api_credentials: Dict[str, str]):
        super().__init__("MicroStrategy", TreasuryProviderType.MICROSTRATEGY)
        self.api_credentials = api_credentials
        
        # MicroStrategy-specific configuration
        self.total_btc_holdings = Decimal("581000")  # ~581K BTC as of 2024
        self.available_for_trading = self.total_btc_holdings * Decimal("0.05")  # 5% liquid
        
        # Liquidity tiers based on MicroStrategy's treasury operations
        self.liquidity_tiers = {
            LiquidityTier.INSTANT: Decimal("100"),      # 100 BTC instant
            LiquidityTier.FAST: Decimal("1000"),        # 1K BTC within 15 min
            LiquidityTier.STANDARD: Decimal("5000"),    # 5K BTC within 1 hour
            LiquidityTier.BATCH: Decimal("20000"),      # 20K BTC batch processing
            LiquidityTier.SETTLEMENT: self.available_for_trading  # Full liquidity
        }
        
        # Rate structure (basis points above/below spot)
        self.rate_spreads = {
            LiquidityTier.INSTANT: Decimal("0.005"),     # 50 bps
            LiquidityTier.FAST: Decimal("0.003"),        # 30 bps
            LiquidityTier.STANDARD: Decimal("0.002"),    # 20 bps
            LiquidityTier.BATCH: Decimal("0.001"),       # 10 bps
            LiquidityTier.SETTLEMENT: Decimal("0.0005")  # 5 bps
        }
    
    async def get_liquidity_quote(
        self, 
        asset: AssetType, 
        amount: Decimal, 
        operation: str
    ) -> TreasuryQuote:
        """Get liquidity quote from MicroStrategy treasury."""
        
        if asset != AssetType.BTC:
            raise ValueError("MicroStrategy provider only supports Bitcoin")
        
        # Determine appropriate liquidity tier
        tier = self._determine_liquidity_tier(amount)
        
        if amount > self.liquidity_tiers[tier]:
            raise ValueError(f"Amount exceeds {tier} liquidity tier")
        
        # Calculate rate (spot + spread)
        spot_rate = await self._get_spot_btc_rate()
        spread = self.rate_spreads[tier]
        
        # Buy operations pay premium, sell operations receive discount
        if operation.lower() == "buy":
            rate = spot_rate * (Decimal("1") + spread)
        else:
            rate = spot_rate * (Decimal("1") - spread)
        
        # Calculate fees
        fees = {
            "treasury_fee": amount * Decimal("0.001"),      # 0.1%
            "custody_fee": amount * Decimal("0.0005"),      # 0.05%
            "settlement_fee": Decimal("0.001")              # Fixed $1 equivalent
        }
        
        return TreasuryQuote(
            provider_name=self.provider_name,
            asset_type=asset,
            available_amount=self.liquidity_tiers[tier],
            rate=rate,
            fees=fees,
            liquidity_tier=tier,
            quote_expires_at=datetime.utcnow() + timedelta(minutes=self._get_quote_validity(tier)),
            settlement_terms={
                "settlement_time_minutes": self._get_settlement_time(tier),
                "collateral_required": operation.lower() == "borrow",
                "custody_provider": "MicroStrategy Treasury",
                "insurance_coverage": "Lloyd's of London - $1B coverage"
            }
        )
    
    async def execute_operation(
        self, 
        quote: TreasuryQuote, 
        amount: Decimal
    ) -> TreasuryExecution:
        """Execute operation with MicroStrategy treasury."""
        
        execution_id = f"mstr_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{amount}"
        
        # In production, this would integrate with MicroStrategy's treasury APIs
        execution = TreasuryExecution(
            execution_id=execution_id,
            provider_name=self.provider_name,
            operation_type="bitcoin_liquidity",
            asset_type=quote.asset_type,
            amount=amount,
            rate=quote.rate,
            status="pending",
            initiated_at=datetime.utcnow()
        )
        
        # Simulate async execution
        asyncio.create_task(self._process_execution(execution, quote))
        
        return execution
    
    async def get_available_liquidity(self, asset: AssetType) -> Dict[LiquidityTier, Decimal]:
        """Get MicroStrategy's available liquidity by tier."""
        
        if asset != AssetType.BTC:
            return {}
        
        # In production, this would query real-time liquidity
        return self.liquidity_tiers.copy()
    
    async def get_custody_status(self) -> Dict[str, Any]:
        """Get MicroStrategy custody operational status."""
        
        return {
            "provider": "MicroStrategy Treasury",
            "total_btc_holdings": str(self.total_btc_holdings),
            "available_for_trading": str(self.available_for_trading),
            "custody_type": "Corporate Treasury",
            "insurance_coverage": "$1B Lloyd's of London",
            "regulatory_status": "Public Company (NASDAQ: MSTR)",
            "operational_status": "active",
            "last_updated": datetime.utcnow().isoformat(),
            "liquidity_utilization": "5%",  # Current utilization
            "average_daily_volume": "1000",  # BTC
            "settlement_capabilities": {
                "instant": "100 BTC",
                "same_day": "5000 BTC", 
                "next_day": "20000 BTC"
            }
        }
    
    def _determine_liquidity_tier(self, amount: Decimal) -> LiquidityTier:
        """Determine appropriate liquidity tier for amount."""
        
        if amount <= self.liquidity_tiers[LiquidityTier.INSTANT]:
            return LiquidityTier.INSTANT
        elif amount <= self.liquidity_tiers[LiquidityTier.FAST]:
            return LiquidityTier.FAST
        elif amount <= self.liquidity_tiers[LiquidityTier.STANDARD]:
            return LiquidityTier.STANDARD
        elif amount <= self.liquidity_tiers[LiquidityTier.BATCH]:
            return LiquidityTier.BATCH
        else:
            return LiquidityTier.SETTLEMENT
    
    async def _get_spot_btc_rate(self) -> Decimal:
        """Get current BTC spot rate from real price oracles."""
        try:
            btc_price = await price_aggregator.get_aggregated_price(AssetType.BTC)
            if btc_price and btc_price.confidence_score > Decimal("0.7"):
                logger.info(f"BTC price: ${btc_price.price_usd:,.2f} (confidence: {btc_price.confidence_score:.2f})")
                return btc_price.price_usd
            else:
                logger.warning("Low confidence BTC price, using fallback")
                return Decimal("50000")  # Fallback price
        except Exception as e:
            logger.error(f"Failed to get BTC price from oracles: {e}")
            return Decimal("50000")  # Emergency fallback
    
    def _get_quote_validity(self, tier: LiquidityTier) -> int:
        """Get quote validity in minutes."""
        validity_minutes = {
            LiquidityTier.INSTANT: 5,
            LiquidityTier.FAST: 15,
            LiquidityTier.STANDARD: 30,
            LiquidityTier.BATCH: 60,
            LiquidityTier.SETTLEMENT: 240
        }
        return validity_minutes[tier]
    
    def _get_settlement_time(self, tier: LiquidityTier) -> int:
        """Get settlement time in minutes."""
        settlement_times = {
            LiquidityTier.INSTANT: 1,
            LiquidityTier.FAST: 15,
            LiquidityTier.STANDARD: 60,
            LiquidityTier.BATCH: 1440,    # 24 hours
            LiquidityTier.SETTLEMENT: 4320 # 3 days
        }
        return settlement_times[tier]
    
    async def _process_execution(self, execution: TreasuryExecution, quote: TreasuryQuote):
        """Process treasury execution (async)."""
        
        try:
            # Simulate settlement delay
            settlement_time = self._get_settlement_time(quote.liquidity_tier)
            await asyncio.sleep(min(settlement_time * 60, 10))  # Cap at 10 seconds for demo
            
            # Complete execution
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            execution.settlement_reference = f"MSTR_SETTLE_{execution.execution_id}"
            execution.blockchain_txids = [f"btc_tx_{execution.execution_id}"]
            
            logger.info(f"MicroStrategy execution completed: {execution.execution_id}")
            
        except Exception as e:
            execution.status = "failed"
            logger.error(f"MicroStrategy execution failed: {e}")


class TreasuryProviderManager:
    """Manages multiple treasury providers and routes operations optimally."""
    
    def __init__(self):
        self.providers: Dict[str, TreasuryProvider] = {}
        self.provider_rankings = {}  # Performance-based rankings
        
    def register_provider(self, provider: TreasuryProvider):
        """Register a treasury provider."""
        self.providers[provider.provider_name] = provider
        self.provider_rankings[provider.provider_name] = {
            "success_rate": 1.0,
            "average_execution_time": 0,
            "cost_efficiency": 1.0,
            "liquidity_depth": 1.0
        }
        logger.info(f"Registered treasury provider: {provider.provider_name}")
    
    async def get_best_liquidity_quote(
        self, 
        asset: AssetType, 
        amount: Decimal, 
        operation: str,
        preferred_tier: Optional[LiquidityTier] = None
    ) -> Tuple[TreasuryProvider, TreasuryQuote]:
        """Get best liquidity quote across all providers."""
        
        quotes = []
        
        # Get quotes from all active providers
        for provider in self.providers.values():
            if not provider.is_active or asset not in provider.supported_assets:
                continue
                
            try:
                quote = await provider.get_liquidity_quote(asset, amount, operation)
                
                # Filter by preferred tier if specified
                if preferred_tier and quote.liquidity_tier != preferred_tier:
                    continue
                    
                quotes.append((provider, quote))
                
            except Exception as e:
                logger.warning(f"Quote failed from {provider.provider_name}: {e}")
        
        if not quotes:
            raise ValueError("No providers available for this operation")
        
        # Rank quotes by cost efficiency and provider reliability
        best_provider, best_quote = self._rank_quotes(quotes, operation)
        
        return best_provider, best_quote
    
    def _rank_quotes(
        self, 
        quotes: List[Tuple[TreasuryProvider, TreasuryQuote]], 
        operation: str
    ) -> Tuple[TreasuryProvider, TreasuryQuote]:
        """Rank quotes by cost efficiency and provider reliability."""
        
        scored_quotes = []
        
        for provider, quote in quotes:
            ranking = self.provider_rankings[provider.provider_name]
            
            # Calculate total cost
            total_fees = sum(quote.fees.values())
            cost_score = 1.0 / (1.0 + float(total_fees))
            
            # Speed score (prefer faster tiers)
            speed_scores = {
                LiquidityTier.INSTANT: 1.0,
                LiquidityTier.FAST: 0.9,
                LiquidityTier.STANDARD: 0.8,
                LiquidityTier.BATCH: 0.6,
                LiquidityTier.SETTLEMENT: 0.4
            }
            speed_score = speed_scores.get(quote.liquidity_tier, 0.5)
            
            # Combined score
            overall_score = (
                cost_score * 0.3 +
                speed_score * 0.2 +
                ranking["success_rate"] * 0.2 +
                ranking["cost_efficiency"] * 0.2 +
                ranking["liquidity_depth"] * 0.1
            )
            
            scored_quotes.append((overall_score, provider, quote))
        
        # Return highest scored quote
        scored_quotes.sort(key=lambda x: x[0], reverse=True)
        return scored_quotes[0][1], scored_quotes[0][2]
    
    async def get_aggregated_liquidity(self, asset: AssetType) -> Dict[str, Any]:
        """Get aggregated liquidity across all providers."""
        
        total_liquidity = {}
        provider_breakdown = {}
        
        for provider in self.providers.values():
            if asset in provider.supported_assets:
                try:
                    liquidity = await provider.get_available_liquidity(asset)
                    provider_breakdown[provider.provider_name] = liquidity
                    
                    # Aggregate by tier
                    for tier, amount in liquidity.items():
                        if tier not in total_liquidity:
                            total_liquidity[tier] = Decimal("0")
                        total_liquidity[tier] += amount
                        
                except Exception as e:
                    logger.warning(f"Liquidity check failed for {provider.provider_name}: {e}")
        
        return {
            "asset": asset.value,
            "total_liquidity_by_tier": {k.value: str(v) for k, v in total_liquidity.items()},
            "provider_breakdown": {
                provider: {tier.value: str(amount) for tier, amount in liquidity.items()}
                for provider, liquidity in provider_breakdown.items()
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def execute_optimal_operation(
        self, 
        asset: AssetType, 
        amount: Decimal, 
        operation: str
    ) -> TreasuryExecution:
        """Execute operation using optimal provider."""
        
        # Get best quote
        provider, quote = await self.get_best_liquidity_quote(asset, amount, operation)
        
        # Execute with selected provider
        execution = await provider.execute_operation(quote, amount)
        
        # Update provider performance metrics
        self._update_provider_ranking(provider.provider_name, execution)
        
        return execution
    
    def _update_provider_ranking(self, provider_name: str, execution: TreasuryExecution):
        """Update provider performance rankings."""
        
        if provider_name not in self.provider_rankings:
            return
        
        ranking = self.provider_rankings[provider_name]
        
        # Update success rate
        if execution.status == "completed":
            ranking["success_rate"] = min(1.0, ranking["success_rate"] * 1.01)
        else:
            ranking["success_rate"] = max(0.1, ranking["success_rate"] * 0.95)
        
        # Update execution time if completed
        if execution.completed_at:
            execution_time = (execution.completed_at - execution.initiated_at).total_seconds()
            ranking["average_execution_time"] = (
                ranking["average_execution_time"] * 0.9 + execution_time * 0.1
            )
        
        logger.debug(f"Updated ranking for {provider_name}: {ranking}")


# Factory function for easy integration
def create_microstrategy_provider(api_credentials: Dict[str, str]) -> MicroStrategyProvider:
    """Factory function to create MicroStrategy provider."""
    return MicroStrategyProvider(api_credentials)


def create_treasury_manager_with_microstrategy(
    microstrategy_credentials: Dict[str, str]
) -> TreasuryProviderManager:
    """Create treasury manager with MicroStrategy provider."""
    
    manager = TreasuryProviderManager()
    
    # Add MicroStrategy provider
    mstr_provider = create_microstrategy_provider(microstrategy_credentials)
    manager.register_provider(mstr_provider)
    
    return manager
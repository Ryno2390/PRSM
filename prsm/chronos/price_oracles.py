"""
CHRONOS Price Oracle Integration

Real-time price feeds from multiple sources with fallback mechanisms.
Provides enterprise-grade price data for treasury operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import aiohttp
from dataclasses import dataclass

from .models import AssetType

logger = logging.getLogger(__name__)


@dataclass
class PriceQuote:
    """Price quote from oracle source."""
    source: str
    asset: AssetType
    price_usd: Decimal
    timestamp: datetime
    volume_24h: Optional[Decimal] = None
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    spread: Optional[Decimal] = None


@dataclass
class AggregatedPrice:
    """Aggregated price from multiple sources."""
    asset: AssetType
    price_usd: Decimal
    confidence_score: Decimal  # 0-1 based on source agreement
    sources_used: List[str]
    price_range: Tuple[Decimal, Decimal]  # min, max from sources
    last_updated: datetime


class PriceOracle:
    """Base class for price oracle implementations."""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.is_active = True
        self.error_count = 0
        self.last_success = None
        
    async def get_price(self, asset: AssetType) -> Optional[PriceQuote]:
        """Get price for asset. Returns None if failed."""
        raise NotImplementedError
    
    async def get_multiple_prices(self, assets: List[AssetType]) -> Dict[AssetType, PriceQuote]:
        """Get prices for multiple assets."""
        results = {}
        for asset in assets:
            price = await self.get_price(asset)
            if price:
                results[asset] = price
        return results


class CoinGeckoOracle(PriceOracle):
    """CoinGecko price oracle - free tier with good reliability."""
    
    def __init__(self):
        super().__init__("CoinGecko")
        self.base_url = "https://api.coingecko.com/api/v3"
        self.asset_mapping = {
            AssetType.BTC: "bitcoin",
            AssetType.ETH: "ethereum", 
            AssetType.USDC: "usd-coin",
            AssetType.USDT: "tether",
            AssetType.ADA: "cardano",
            AssetType.SOL: "solana",
            AssetType.DOT: "polkadot"
        }
        
    async def get_price(self, asset: AssetType) -> Optional[PriceQuote]:
        """Get price from CoinGecko."""
        
        if asset == AssetType.USD:
            return PriceQuote(
                source=self.source_name,
                asset=asset,
                price_usd=Decimal("1.0"),
                timestamp=datetime.utcnow()
            )
        
        if asset not in self.asset_mapping:
            return None
            
        try:
            coin_id = self.asset_mapping[asset]
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_vol": "true"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if coin_id in data:
                            coin_data = data[coin_id]
                            
                            quote = PriceQuote(
                                source=self.source_name,
                                asset=asset,
                                price_usd=Decimal(str(coin_data["usd"])),
                                timestamp=datetime.utcnow(),
                                volume_24h=Decimal(str(coin_data.get("usd_24h_vol", 0)))
                            )
                            
                            self.error_count = 0
                            self.last_success = datetime.utcnow()
                            return quote
                        
        except Exception as e:
            logger.warning(f"CoinGecko price fetch failed for {asset}: {e}")
            self.error_count += 1
            
        return None


class CoinCapOracle(PriceOracle):
    """CoinCap price oracle - good for real-time data."""
    
    def __init__(self):
        super().__init__("CoinCap")
        self.base_url = "https://api.coincap.io/v2"
        self.asset_mapping = {
            AssetType.BTC: "bitcoin",
            AssetType.ETH: "ethereum",
            AssetType.USDC: "usd-coin", 
            AssetType.USDT: "tether",
            AssetType.ADA: "cardano",
            AssetType.SOL: "solana",
            AssetType.DOT: "polkadot"
        }
    
    async def get_price(self, asset: AssetType) -> Optional[PriceQuote]:
        """Get price from CoinCap."""
        
        if asset == AssetType.USD:
            return PriceQuote(
                source=self.source_name,
                asset=asset,
                price_usd=Decimal("1.0"),
                timestamp=datetime.utcnow()
            )
            
        if asset not in self.asset_mapping:
            return None
            
        try:
            asset_id = self.asset_mapping[asset]
            url = f"{self.base_url}/assets/{asset_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "data" in data:
                            asset_data = data["data"]
                            
                            quote = PriceQuote(
                                source=self.source_name,
                                asset=asset,
                                price_usd=Decimal(asset_data["priceUsd"]),
                                timestamp=datetime.utcnow(),
                                volume_24h=Decimal(asset_data.get("volumeUsd24Hr", "0"))
                            )
                            
                            self.error_count = 0
                            self.last_success = datetime.utcnow()
                            return quote
                            
        except Exception as e:
            logger.warning(f"CoinCap price fetch failed for {asset}: {e}")
            self.error_count += 1
            
        return None


class BitstampOracle(PriceOracle):
    """Bitstamp exchange prices - good for BTC/USD specifically."""
    
    def __init__(self):
        super().__init__("Bitstamp")
        self.base_url = "https://www.bitstamp.net/api/v2"
        self.supported_pairs = {
            AssetType.BTC: "btcusd",
            AssetType.ETH: "ethusd"
        }
    
    async def get_price(self, asset: AssetType) -> Optional[PriceQuote]:
        """Get price from Bitstamp."""
        
        if asset == AssetType.USD:
            return PriceQuote(
                source=self.source_name,
                asset=asset, 
                price_usd=Decimal("1.0"),
                timestamp=datetime.utcnow()
            )
            
        if asset not in self.supported_pairs:
            return None
            
        try:
            pair = self.supported_pairs[asset]
            url = f"{self.base_url}/ticker/{pair}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        quote = PriceQuote(
                            source=self.source_name,
                            asset=asset,
                            price_usd=Decimal(data["last"]),
                            timestamp=datetime.utcnow(),
                            volume_24h=Decimal(data.get("volume", "0")),
                            bid=Decimal(data.get("bid", "0")),
                            ask=Decimal(data.get("ask", "0"))
                        )
                        
                        if quote.bid and quote.ask:
                            quote.spread = quote.ask - quote.bid
                            
                        self.error_count = 0
                        self.last_success = datetime.utcnow()
                        return quote
                        
        except Exception as e:
            logger.warning(f"Bitstamp price fetch failed for {asset}: {e}")
            self.error_count += 1
            
        return None


class PriceAggregator:
    """Aggregates prices from multiple oracles with confidence scoring."""
    
    def __init__(self):
        self.oracles = [
            CoinGeckoOracle(),
            CoinCapOracle(), 
            BitstampOracle()
        ]
        
        # Price cache with TTL
        self.price_cache = {}
        self.cache_ttl = timedelta(seconds=30)
        
        # Oracle reliability tracking
        self.oracle_weights = {
            "CoinGecko": 0.4,   # Most comprehensive
            "CoinCap": 0.3,     # Good real-time data
            "Bitstamp": 0.3     # Exchange-based pricing
        }
    
    async def get_aggregated_price(self, asset: AssetType) -> Optional[AggregatedPrice]:
        """Get aggregated price from multiple oracles."""
        
        # Check cache first
        cache_key = asset.value
        if cache_key in self.price_cache:
            cached_price, timestamp = self.price_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                return cached_price
        
        # Fetch from all oracles in parallel
        tasks = [oracle.get_price(asset) for oracle in self.oracles if oracle.is_active]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_quotes = []
        for result in results:
            if isinstance(result, PriceQuote):
                valid_quotes.append(result)
        
        if not valid_quotes:
            logger.error(f"No valid price quotes for {asset}")
            return None
        
        # Calculate weighted average
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")
        prices = []
        sources_used = []
        
        for quote in valid_quotes:
            weight = Decimal(str(self.oracle_weights.get(quote.source, 0.1)))
            total_weight += weight
            weighted_sum += quote.price_usd * weight
            prices.append(quote.price_usd)
            sources_used.append(quote.source)
        
        if total_weight == 0:
            return None
            
        aggregated_price = weighted_sum / total_weight
        
        # Calculate confidence score based on price agreement
        if len(prices) > 1:
            price_std = self._calculate_std_dev(prices)
            price_mean = sum(prices) / len(prices)
            coefficient_of_variation = price_std / price_mean
            
            # Higher agreement = higher confidence
            confidence_score = max(Decimal("0.1"), Decimal("1.0") - coefficient_of_variation * 10)
        else:
            confidence_score = Decimal("0.7")  # Single source
        
        # Create aggregated result
        aggregated = AggregatedPrice(
            asset=asset,
            price_usd=aggregated_price,
            confidence_score=min(confidence_score, Decimal("1.0")),
            sources_used=sources_used,
            price_range=(min(prices), max(prices)),
            last_updated=datetime.utcnow()
        )
        
        # Cache result
        self.price_cache[cache_key] = (aggregated, datetime.utcnow())
        
        return aggregated
    
    async def get_multiple_aggregated_prices(
        self, 
        assets: List[AssetType]
    ) -> Dict[AssetType, AggregatedPrice]:
        """Get aggregated prices for multiple assets."""
        
        tasks = [self.get_aggregated_price(asset) for asset in assets]
        results = await asyncio.gather(*tasks)
        
        return {
            asset: price for asset, price in zip(assets, results) 
            if price is not None
        }
    
    def _calculate_std_dev(self, prices: List[Decimal]) -> Decimal:
        """Calculate standard deviation of prices."""
        if len(prices) < 2:
            return Decimal("0")
            
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        return variance ** Decimal("0.5")
    
    async def get_oracle_health(self) -> Dict[str, Dict]:
        """Get health status of all oracles."""
        
        health_status = {}
        
        for oracle in self.oracles:
            status = {
                "name": oracle.source_name,
                "is_active": oracle.is_active,
                "error_count": oracle.error_count,
                "last_success": oracle.last_success.isoformat() if oracle.last_success else None,
                "weight": self.oracle_weights.get(oracle.source_name, 0.1)
            }
            
            # Health score based on recent performance
            if oracle.error_count == 0 and oracle.last_success:
                time_since_success = datetime.utcnow() - oracle.last_success
                if time_since_success < timedelta(minutes=5):
                    status["health_score"] = "excellent"
                elif time_since_success < timedelta(minutes=30):
                    status["health_score"] = "good"
                else:
                    status["health_score"] = "degraded"
            else:
                status["health_score"] = "poor"
            
            health_status[oracle.source_name] = status
        
        return health_status


# Global price aggregator instance
price_aggregator = PriceAggregator()


# Convenience functions for easy integration
async def get_btc_price() -> Optional[AggregatedPrice]:
    """Get current BTC price."""
    return await price_aggregator.get_aggregated_price(AssetType.BTC)


async def get_asset_price(asset: AssetType) -> Optional[AggregatedPrice]:
    """Get current price for any supported asset."""
    return await price_aggregator.get_aggregated_price(asset)


async def get_exchange_rate(from_asset: AssetType, to_asset: AssetType) -> Optional[Decimal]:
    """Get exchange rate between two assets."""
    
    if from_asset == to_asset:
        return Decimal("1.0")
    
    # Get both prices in USD
    from_price = await price_aggregator.get_aggregated_price(from_asset)
    to_price = await price_aggregator.get_aggregated_price(to_asset)
    
    if from_price and to_price:
        return from_price.price_usd / to_price.price_usd
    
    return None


async def validate_price_feeds() -> Dict[str, str]:
    """Validate all price feeds are working."""
    
    test_assets = [AssetType.BTC, AssetType.ETH, AssetType.USDC]
    results = {}
    
    for asset in test_assets:
        price = await price_aggregator.get_aggregated_price(asset)
        if price:
            results[asset.value] = f"${price.price_usd:,.2f} (confidence: {price.confidence_score:.2f})"
        else:
            results[asset.value] = "FAILED"
    
    return results
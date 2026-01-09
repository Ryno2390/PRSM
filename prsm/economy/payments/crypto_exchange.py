"""
PRSM Crypto Exchange Integration
===============================

Production-grade cryptocurrency exchange integration for real-time pricing,
liquidity sourcing, and automated token conversion with multiple DEX support.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union

import aiohttp
import structlog

from prsm.core.config import get_settings
from .payment_models import (
    FiatCurrency, CryptoCurrency, ExchangeRate, PaymentStatus
)

logger = structlog.get_logger(__name__)


class ExchangeProvider:
    """Base exchange provider interface"""
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        self.provider_name = provider_name
        self.config = config
        self.session = None
        self.rate_cache = {}
        self.cache_expiry = timedelta(minutes=1)  # Cache rates for 1 minute
        
    async def initialize(self):
        """Initialize the exchange provider"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "PRSM-Crypto-Exchange/1.0"}
        )
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """Get exchange rate (to be implemented by subclasses)"""
        raise NotImplementedError
    
    async def get_multiple_rates(self, pairs: List[tuple]) -> Dict[str, ExchangeRate]:
        """Get multiple exchange rates"""
        rates = {}
        for from_currency, to_currency in pairs:
            rate = await self.get_exchange_rate(from_currency, to_currency)
            if rate:
                rates[f"{from_currency}_{to_currency}"] = rate
        return rates
    
    def _is_rate_cached(self, cache_key: str) -> bool:
        """Check if rate is cached and still valid"""
        if cache_key not in self.rate_cache:
            return False
        
        cached_data = self.rate_cache[cache_key]
        return datetime.now(timezone.utc) < cached_data["expires_at"]
    
    def _cache_rate(self, cache_key: str, rate: ExchangeRate):
        """Cache exchange rate"""
        self.rate_cache[cache_key] = {
            "rate": rate,
            "expires_at": datetime.now(timezone.utc) + self.cache_expiry
        }
    
    def _get_cached_rate(self, cache_key: str) -> Optional[ExchangeRate]:
        """Get cached exchange rate"""
        if self._is_rate_cached(cache_key):
            return self.rate_cache[cache_key]["rate"]
        return None


class CoinGeckoProvider(ExchangeProvider):
    """CoinGecko price API provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("coingecko", config)
        self.api_key = config.get("api_key")  # Optional for free tier
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Mapping of our currencies to CoinGecko IDs
        self.currency_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "MATIC": "matic-network",
            "USDC": "usd-coin",
            "USDT": "tether",
            "FTNS": "ftns-token"  # This would be the actual CoinGecko ID
        }
        
        # Fiat currency mapping
        self.fiat_mapping = {
            "USD": "usd",
            "EUR": "eur", 
            "GBP": "gbp",
            "JPY": "jpy",
            "CAD": "cad",
            "AUD": "aud"
        }
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """Get exchange rate from CoinGecko"""
        cache_key = f"{from_currency}_{to_currency}"
        
        # Check cache first
        cached_rate = self._get_cached_rate(cache_key)
        if cached_rate:
            return cached_rate
        
        try:
            # Handle fiat to crypto conversion
            if from_currency in self.fiat_mapping and to_currency in self.currency_mapping:
                rate = await self._get_fiat_to_crypto_rate(from_currency, to_currency)
            # Handle crypto to crypto conversion
            elif from_currency in self.currency_mapping and to_currency in self.currency_mapping:
                rate = await self._get_crypto_to_crypto_rate(from_currency, to_currency)
            # Handle crypto to fiat conversion
            elif from_currency in self.currency_mapping and to_currency in self.fiat_mapping:
                rate = await self._get_crypto_to_fiat_rate(from_currency, to_currency)
            else:
                logger.warning(f"Unsupported currency pair: {from_currency}/{to_currency}")
                return None
            
            if rate:
                self._cache_rate(cache_key, rate)
            
            return rate
            
        except Exception as e:
            logger.error("Failed to get CoinGecko exchange rate", 
                        from_currency=from_currency, to_currency=to_currency, error=str(e))
            return None
    
    async def _get_fiat_to_crypto_rate(self, fiat: str, crypto: str) -> Optional[ExchangeRate]:
        """Get fiat to cryptocurrency rate"""
        crypto_id = self.currency_mapping.get(crypto)
        fiat_code = self.fiat_mapping.get(fiat)
        
        if not crypto_id or not fiat_code:
            return None
        
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        
        url = f"{self.base_url}/simple/price"
        params = {
            "ids": crypto_id,
            "vs_currencies": fiat_code,
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_market_cap": "true"
        }
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                if crypto_id in data and fiat_code in data[crypto_id]:
                    price = Decimal(str(data[crypto_id][fiat_code]))
                    
                    return ExchangeRate(
                        from_currency=fiat,
                        to_currency=crypto,
                        rate=Decimal("1") / price,  # Convert to fiat-per-crypto rate
                        source="coingecko",
                        volume_24h=data[crypto_id].get(f"{fiat_code}_24h_vol"),
                        price_change_24h=data[crypto_id].get(f"{fiat_code}_24h_change"),
                        market_cap=data[crypto_id].get(f"{fiat_code}_market_cap")
                    )
            
            return None
    
    async def _get_crypto_to_crypto_rate(self, from_crypto: str, to_crypto: str) -> Optional[ExchangeRate]:
        """Get cryptocurrency to cryptocurrency rate"""
        from_id = self.currency_mapping.get(from_crypto)
        to_id = self.currency_mapping.get(to_crypto)
        
        if not from_id or not to_id:
            return None
        
        # Get both prices in USD and calculate cross rate
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        
        url = f"{self.base_url}/simple/price"
        params = {
            "ids": f"{from_id},{to_id}",
            "vs_currencies": "usd",
            "include_24hr_vol": "true"
        }
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                if (from_id in data and "usd" in data[from_id] and 
                    to_id in data and "usd" in data[to_id]):
                    
                    from_price_usd = Decimal(str(data[from_id]["usd"]))
                    to_price_usd = Decimal(str(data[to_id]["usd"]))
                    
                    if to_price_usd > 0:
                        cross_rate = from_price_usd / to_price_usd
                        
                        return ExchangeRate(
                            from_currency=from_crypto,
                            to_currency=to_crypto,
                            rate=cross_rate,
                            source="coingecko",
                            volume_24h=data[from_id].get("usd_24h_vol")
                        )
            
            return None
    
    async def _get_crypto_to_fiat_rate(self, crypto: str, fiat: str) -> Optional[ExchangeRate]:
        """Get cryptocurrency to fiat rate"""
        crypto_id = self.currency_mapping.get(crypto)
        fiat_code = self.fiat_mapping.get(fiat)
        
        if not crypto_id or not fiat_code:
            return None
        
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        
        url = f"{self.base_url}/simple/price"
        params = {
            "ids": crypto_id,
            "vs_currencies": fiat_code,
            "include_24hr_vol": "true",
            "include_24hr_change": "true"
        }
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                if crypto_id in data and fiat_code in data[crypto_id]:
                    price = Decimal(str(data[crypto_id][fiat_code]))
                    
                    return ExchangeRate(
                        from_currency=crypto,
                        to_currency=fiat,
                        rate=price,
                        source="coingecko",
                        volume_24h=data[crypto_id].get(f"{fiat_code}_24h_vol"),
                        price_change_24h=data[crypto_id].get(f"{fiat_code}_24h_change")
                    )
            
            return None


class OneinchProvider(ExchangeProvider):
    """1inch DEX aggregator provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("1inch", config)
        self.api_key = config.get("api_key")
        self.base_url = "https://api.1inch.dev"
        self.chain_id = config.get("chain_id", 137)  # Polygon mainnet
        
        # Token addresses on Polygon
        self.token_addresses = {
            "MATIC": "0x0000000000000000000000000000000000001010",  # Native MATIC
            "USDC": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
            "USDT": "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",
            "ETH": "0x7ceb23fd6509b91748f5c0dd96b81b0f0b3e8b10",
            "FTNS": "0x1234567890123456789012345678901234567890"  # Placeholder - actual FTNS address
        }
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """Get exchange rate from 1inch"""
        cache_key = f"{from_currency}_{to_currency}_1inch"
        
        # Check cache first
        cached_rate = self._get_cached_rate(cache_key)
        if cached_rate:
            return cached_rate
        
        try:
            from_token = self.token_addresses.get(from_currency)
            to_token = self.token_addresses.get(to_currency)
            
            if not from_token or not to_token:
                return None
            
            # Use 1 unit for quote (in smallest denomination)
            amount = "1000000000000000000"  # 1 token with 18 decimals
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            url = f"{self.base_url}/v5.0/{self.chain_id}/quote"
            params = {
                "fromTokenAddress": from_token,
                "toTokenAddress": to_token,
                "amount": amount
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    from_amount = Decimal(data["fromTokenAmount"])
                    to_amount = Decimal(data["toTokenAmount"])
                    
                    if from_amount > 0:
                        rate = to_amount / from_amount
                        
                        exchange_rate = ExchangeRate(
                            from_currency=from_currency,
                            to_currency=to_currency,
                            rate=rate,
                            source="1inch"
                        )
                        
                        self._cache_rate(cache_key, exchange_rate)
                        return exchange_rate
            
            return None
            
        except Exception as e:
            logger.error("Failed to get 1inch exchange rate", 
                        from_currency=from_currency, to_currency=to_currency, error=str(e))
            return None


class MockExchangeProvider(ExchangeProvider):
    """Mock exchange provider for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("mock", config)
        
        # Mock exchange rates (simplified for testing)
        self.mock_rates = {
            "USD_MATIC": Decimal("0.8000"),   # 1 USD = 0.8 MATIC
            "USD_ETH": Decimal("0.0003"),     # 1 USD = 0.0003 ETH  
            "USD_USDC": Decimal("1.0000"),    # 1 USD = 1 USDC
            "USD_FTNS": Decimal("10.0000"),   # 1 USD = 10 FTNS
            "EUR_MATIC": Decimal("0.7500"),   # 1 EUR = 0.75 MATIC
            "MATIC_USD": Decimal("1.2500"),   # 1 MATIC = 1.25 USD
            "ETH_USD": Decimal("3000.0000"),  # 1 ETH = 3000 USD
            "FTNS_USD": Decimal("0.1000"),    # 1 FTNS = 0.10 USD
        }
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """Get mock exchange rate"""
        rate_key = f"{from_currency}_{to_currency}"
        
        if rate_key in self.mock_rates:
            base_rate = self.mock_rates[rate_key]
            
            # Add some randomness (+/- 2%) to simulate real market conditions
            import random
            variation = Decimal(str(random.uniform(0.98, 1.02)))
            final_rate = base_rate * variation
            
            return ExchangeRate(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=final_rate,
                source="mock",
                volume_24h=Decimal("1000000"),  # Mock volume
                price_change_24h=Decimal(str(random.uniform(-5, 5)))  # Mock 24h change
            )
        
        # Try reverse rate
        reverse_key = f"{to_currency}_{from_currency}"
        if reverse_key in self.mock_rates:
            base_rate = Decimal("1") / self.mock_rates[reverse_key]
            
            import random
            variation = Decimal(str(random.uniform(0.98, 1.02)))
            final_rate = base_rate * variation
            
            return ExchangeRate(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=final_rate,
                source="mock",
                volume_24h=Decimal("1000000")
            )
        
        return None


class CryptoExchange:
    """
    Production-grade cryptocurrency exchange integration
    
    ⚡ CRYPTO EXCHANGE FEATURES:
    - Multi-provider exchange rate aggregation
    - Real-time price feeds with caching
    - DEX integration for best pricing
    - Slippage protection and price impact analysis
    - Historical rate tracking and analytics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.providers: Dict[str, ExchangeProvider] = {}
        self.primary_provider = self.config.get("primary_provider", "mock")
        
        # Rate aggregation settings
        self.use_aggregation = self.config.get("use_aggregation", True)
        self.max_price_deviation = Decimal(self.config.get("max_price_deviation", "0.05"))  # 5%
        
        print("⚡ Crypto Exchange initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default exchange configuration"""
        return {
            "primary_provider": "mock",
            "use_aggregation": True,
            "max_price_deviation": "0.05",
            "providers": {
                "coingecko": {
                    "enabled": True,
                    "api_key": "",
                    "priority": 1
                },
                "1inch": {
                    "enabled": False,
                    "api_key": "",
                    "chain_id": 137,
                    "priority": 2
                },
                "mock": {
                    "enabled": True,
                    "priority": 3
                }
            }
        }
    
    async def initialize(self):
        """Initialize exchange providers"""
        provider_configs = self.config.get("providers", {})
        
        for provider_name, provider_config in provider_configs.items():
            if not provider_config.get("enabled", False):
                continue
                
            try:
                if provider_name == "coingecko":
                    provider = CoinGeckoProvider(provider_config)
                elif provider_name == "1inch":
                    provider = OneinchProvider(provider_config)
                elif provider_name == "mock":
                    provider = MockExchangeProvider(provider_config)
                else:
                    logger.warning(f"Unknown exchange provider: {provider_name}")
                    continue
                
                await provider.initialize()
                self.providers[provider_name] = provider
                logger.info(f"✅ Exchange provider initialized: {provider_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}", error=str(e))
        
        if not self.providers:
            logger.warning("No exchange providers initialized")
    
    async def cleanup(self):
        """Cleanup exchange providers"""
        for provider in self.providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup provider {provider.provider_name}", error=str(e))
    
    async def get_exchange_rate(
        self, 
        from_currency: str, 
        to_currency: str,
        amount: Optional[Decimal] = None
    ) -> Optional[ExchangeRate]:
        """Get best exchange rate across providers"""
        try:
            if self.use_aggregation and len(self.providers) > 1:
                return await self._get_aggregated_rate(from_currency, to_currency, amount)
            else:
                return await self._get_single_provider_rate(from_currency, to_currency, amount)
                
        except Exception as e:
            logger.error("Failed to get exchange rate", 
                        from_currency=from_currency, to_currency=to_currency, error=str(e))
            return None
    
    async def _get_single_provider_rate(
        self, 
        from_currency: str, 
        to_currency: str,
        amount: Optional[Decimal] = None
    ) -> Optional[ExchangeRate]:
        """Get rate from primary provider"""
        provider = self.providers.get(self.primary_provider)
        if provider:
            return await provider.get_exchange_rate(from_currency, to_currency)
        return None
    
    async def _get_aggregated_rate(
        self, 
        from_currency: str, 
        to_currency: str,
        amount: Optional[Decimal] = None
    ) -> Optional[ExchangeRate]:
        """Get aggregated rate across multiple providers"""
        tasks = []
        
        # Get rates from all providers concurrently
        for provider in self.providers.values():
            task = provider.get_exchange_rate(from_currency, to_currency)
            tasks.append(task)
        
        rates = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid rates
        valid_rates = []
        for rate in rates:
            if isinstance(rate, ExchangeRate):
                valid_rates.append(rate)
        
        if not valid_rates:
            return None
        
        # Simple aggregation strategy: use median rate
        if len(valid_rates) == 1:
            return valid_rates[0]
        
        sorted_rates = sorted(valid_rates, key=lambda r: r.rate)
        median_idx = len(sorted_rates) // 2
        
        if len(sorted_rates) % 2 == 0:
            # Even number of rates - average the two middle values
            median_rate = (sorted_rates[median_idx - 1].rate + sorted_rates[median_idx].rate) / 2
        else:
            # Odd number of rates - use middle value
            median_rate = sorted_rates[median_idx].rate
        
        # Check for excessive deviation
        for rate in valid_rates:
            deviation = abs(rate.rate - median_rate) / median_rate
            if deviation > self.max_price_deviation:
                logger.warning(f"High price deviation detected: {deviation:.2%} from {rate.source}")
        
        # Return rate with median price but aggregate volume
        total_volume = sum(r.volume_24h or Decimal("0") for r in valid_rates)
        
        return ExchangeRate(
            from_currency=from_currency,
            to_currency=to_currency,
            rate=median_rate,
            source="aggregated",
            volume_24h=total_volume if total_volume > 0 else None,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def calculate_conversion(
        self, 
        amount: Decimal,
        from_currency: str, 
        to_currency: str,
        include_slippage: bool = True
    ) -> Dict[str, Any]:
        """Calculate currency conversion with slippage"""
        rate = await self.get_exchange_rate(from_currency, to_currency, amount)
        
        if not rate:
            return {
                "success": False,
                "error": "Exchange rate not available"
            }
        
        base_output = amount * rate.rate
        
        # Apply slippage for large amounts (simplified model)
        slippage = Decimal("0")
        if include_slippage and amount > Decimal("1000"):
            # Simple slippage model: 0.1% for every $1000
            slippage_factor = (amount / Decimal("1000")) * Decimal("0.001")
            slippage = base_output * slippage_factor
        
        final_output = base_output - slippage
        
        return {
            "success": True,
            "input_amount": amount,
            "input_currency": from_currency,
            "output_amount": final_output,
            "output_currency": to_currency,
            "exchange_rate": rate.rate,
            "base_output": base_output,
            "slippage": slippage,
            "slippage_percent": (slippage / base_output * 100) if base_output > 0 else Decimal("0"),
            "rate_source": rate.source,
            "timestamp": rate.timestamp
        }
    
    async def get_multiple_rates(self, pairs: List[tuple]) -> Dict[str, ExchangeRate]:
        """Get multiple exchange rates efficiently"""
        tasks = []
        
        for from_currency, to_currency in pairs:
            task = self.get_exchange_rate(from_currency, to_currency)
            tasks.append((f"{from_currency}_{to_currency}", task))
        
        rates = {}
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (pair_key, _), result in zip(tasks, results):
            if isinstance(result, ExchangeRate):
                rates[pair_key] = result
        
        return rates
    
    def get_supported_currencies(self) -> Dict[str, List[str]]:
        """Get supported currencies by provider"""
        return {
            "coingecko": ["USD", "EUR", "GBP", "BTC", "ETH", "MATIC", "USDC", "USDT"],
            "1inch": ["MATIC", "ETH", "USDC", "USDT", "FTNS"],
            "mock": ["USD", "EUR", "MATIC", "ETH", "USDC", "FTNS"]
        }
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                "enabled": True,
                "cache_size": len(provider.rate_cache),
                "last_updated": max(
                    (data["expires_at"] - provider.cache_expiry 
                     for data in provider.rate_cache.values()),
                    default=None
                )
            }
        return status


# Global crypto exchange instance  
_crypto_exchange: Optional[CryptoExchange] = None

async def get_crypto_exchange() -> CryptoExchange:
    """Get or create the global crypto exchange instance"""
    global _crypto_exchange
    if _crypto_exchange is None:
        _crypto_exchange = CryptoExchange()
        await _crypto_exchange.initialize()
    return _crypto_exchange
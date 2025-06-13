"""
CHRONOS Exchange Router

Routes trades through optimal exchange combinations for best execution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import aiohttp

from .models import AssetType, ExchangeConfig


logger = logging.getLogger(__name__)


class ExchangeRouter:
    """Routes trades through multiple exchanges for optimal execution."""
    
    def __init__(self):
        # Mock exchange configurations
        self.exchanges = self._initialize_mock_exchanges()
        
        # Rate limiting
        self.rate_limits = {}
        
        # Price cache
        self.price_cache = {}
        self.cache_ttl = timedelta(seconds=30)
    
    def _initialize_mock_exchanges(self) -> Dict[str, ExchangeConfig]:
        """Initialize mock exchange configurations."""
        exchanges = {}
        
        # Coinbase Pro
        exchanges["coinbase"] = ExchangeConfig(
            name="Coinbase Pro",
            api_key="mock_coinbase_key",
            api_secret="mock_coinbase_secret",
            sandbox_mode=True,
            supported_pairs=[
                (AssetType.BTC, AssetType.USD),
                (AssetType.ETH, AssetType.USD),
                (AssetType.BTC, AssetType.ETH)
            ],
            fee_rates={
                "maker": Decimal("0.005"),  # 0.5%
                "taker": Decimal("0.005")   # 0.5%
            },
            rate_limits={
                "requests_per_second": 10,
                "orders_per_second": 5
            }
        )
        
        # Binance
        exchanges["binance"] = ExchangeConfig(
            name="Binance",
            api_key="mock_binance_key", 
            api_secret="mock_binance_secret",
            sandbox_mode=True,
            supported_pairs=[
                (AssetType.BTC, AssetType.USD),
                (AssetType.ETH, AssetType.USD),
                (AssetType.BTC, AssetType.ETH)
            ],
            fee_rates={
                "maker": Decimal("0.001"),  # 0.1%
                "taker": Decimal("0.001")   # 0.1%
            },
            rate_limits={
                "requests_per_second": 20,
                "orders_per_second": 10
            }
        )
        
        # Kraken
        exchanges["kraken"] = ExchangeConfig(
            name="Kraken",
            api_key="mock_kraken_key",
            api_secret="mock_kraken_secret", 
            sandbox_mode=True,
            supported_pairs=[
                (AssetType.BTC, AssetType.USD),
                (AssetType.ETH, AssetType.USD)
            ],
            fee_rates={
                "maker": Decimal("0.0016"), # 0.16%
                "taker": Decimal("0.0026")  # 0.26%
            },
            rate_limits={
                "requests_per_second": 1,
                "orders_per_second": 1
            }
        )
        
        return exchanges
    
    async def get_best_price(
        self, 
        from_asset: AssetType, 
        to_asset: AssetType, 
        amount: Decimal
    ) -> Dict:
        """Get best available price across all exchanges."""
        
        # Check cache first
        cache_key = f"{from_asset}_{to_asset}_{amount}"
        if cache_key in self.price_cache:
            cached_data, timestamp = self.price_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                return cached_data
        
        # Query all exchanges in parallel
        tasks = []
        for exchange_name, config in self.exchanges.items():
            if self._supports_pair(config, from_asset, to_asset):
                tasks.append(self._get_exchange_price(exchange_name, config, from_asset, to_asset, amount))
        
        if not tasks:
            return {"error": f"No exchanges support {from_asset}->{to_asset} pair"}
        
        # Wait for all price quotes
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_quotes = []
        for result in results:
            if isinstance(result, dict) and "error" not in result:
                valid_quotes.append(result)
        
        if not valid_quotes:
            return {"error": "No valid quotes received"}
        
        # Find best quote (highest output amount)
        best_quote = max(valid_quotes, key=lambda x: Decimal(x["output_amount"]))
        
        # Cache result
        self.price_cache[cache_key] = (best_quote, datetime.utcnow())
        
        return best_quote
    
    async def _get_exchange_price(
        self,
        exchange_name: str,
        config: ExchangeConfig,
        from_asset: AssetType,
        to_asset: AssetType,
        amount: Decimal
    ) -> Dict:
        """Get price quote from specific exchange."""
        
        # Check rate limits
        if not await self._check_rate_limit(exchange_name):
            return {"error": f"Rate limit exceeded for {exchange_name}"}
        
        try:
            # In real implementation, this would call actual exchange APIs
            # For now, return mock data with realistic variation
            
            base_rates = {
                ("BTC", "USD"): Decimal("50000"),
                ("USD", "BTC"): Decimal("0.00002"),
                ("ETH", "USD"): Decimal("3000"),
                ("USD", "ETH"): Decimal("0.000333"),
                ("BTC", "ETH"): Decimal("16.67"),
                ("ETH", "BTC"): Decimal("0.06")
            }
            
            pair_key = (from_asset.value, to_asset.value)
            if pair_key not in base_rates:
                return {"error": f"Pair {pair_key} not supported"}
            
            # Add exchange-specific spread and fees
            base_rate = base_rates[pair_key]
            
            # Simulate different exchange rates
            exchange_multipliers = {
                "coinbase": Decimal("0.999"),  # Slightly lower rate
                "binance": Decimal("1.001"),   # Slightly higher rate  
                "kraken": Decimal("0.998")     # Lower rate, higher fees
            }
            
            rate = base_rate * exchange_multipliers.get(exchange_name, Decimal("1.0"))
            
            # Calculate output amount
            output_amount = amount * rate
            
            # Apply fees
            fee_rate = config.fee_rates["taker"]
            fee_amount = output_amount * fee_rate
            net_output = output_amount - fee_amount
            
            return {
                "exchange": exchange_name,
                "from_asset": from_asset.value,
                "to_asset": to_asset.value,
                "input_amount": str(amount),
                "output_amount": str(net_output),
                "gross_output": str(output_amount),
                "fee_amount": str(fee_amount),
                "fee_rate": str(fee_rate),
                "exchange_rate": str(rate),
                "timestamp": datetime.utcnow().isoformat(),
                "estimated_fill_time": "1-3 minutes"
            }
            
        except Exception as e:
            logger.error(f"Error getting price from {exchange_name}: {e}")
            return {"error": f"Failed to get quote from {exchange_name}"}
    
    async def execute_trade(
        self,
        exchange_name: str,
        from_asset: AssetType,
        to_asset: AssetType,
        amount: Decimal
    ) -> Dict:
        """Execute trade on specified exchange."""
        
        if exchange_name not in self.exchanges:
            return {"error": f"Exchange {exchange_name} not configured"}
        
        config = self.exchanges[exchange_name]
        
        if not self._supports_pair(config, from_asset, to_asset):
            return {"error": f"Exchange {exchange_name} doesn't support {from_asset}->{to_asset}"}
        
        # Check rate limits
        if not await self._check_rate_limit(exchange_name, action="trade"):
            return {"error": f"Trade rate limit exceeded for {exchange_name}"}
        
        try:
            # In real implementation, this would place actual orders
            
            # Simulate trade execution
            trade_id = f"{exchange_name}_{from_asset.value}_{to_asset.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Get current price
            quote = await self._get_exchange_price(exchange_name, config, from_asset, to_asset, amount)
            
            if "error" in quote:
                return quote
            
            # Simulate execution delay
            await asyncio.sleep(1)
            
            return {
                "trade_id": trade_id,
                "exchange": exchange_name,
                "status": "filled",
                "from_asset": from_asset.value,
                "to_asset": to_asset.value,
                "input_amount": str(amount),
                "output_amount": quote["output_amount"],
                "fee_amount": quote["fee_amount"],
                "exchange_rate": quote["exchange_rate"],
                "executed_at": datetime.utcnow().isoformat(),
                "settlement_time": "T+0"  # Immediate settlement for simulation
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed on {exchange_name}: {e}")
            return {"error": f"Trade execution failed: {str(e)}"}
    
    async def get_optimal_route(
        self,
        from_asset: AssetType,
        to_asset: AssetType,
        amount: Decimal
    ) -> List[Dict]:
        """Calculate optimal trading route across exchanges."""
        
        # For direct pairs, compare all exchanges
        if self._has_direct_pair(from_asset, to_asset):
            best_quote = await self.get_best_price(from_asset, to_asset, amount)
            if "error" not in best_quote:
                return [best_quote]
        
        # For indirect pairs, find multi-hop routes
        # Example: FTNS -> USD via FTNS -> BTC -> USD
        indirect_routes = await self._find_indirect_routes(from_asset, to_asset, amount)
        
        if indirect_routes:
            return indirect_routes
        
        return [{"error": f"No route found for {from_asset} -> {to_asset}"}]
    
    async def _find_indirect_routes(
        self,
        from_asset: AssetType,
        to_asset: AssetType,
        amount: Decimal
    ) -> List[Dict]:
        """Find indirect trading routes through intermediate assets."""
        
        # Common intermediate assets
        intermediates = [AssetType.BTC, AssetType.ETH, AssetType.USD]
        
        best_routes = []
        
        for intermediate in intermediates:
            if intermediate == from_asset or intermediate == to_asset:
                continue
            
            # Check if both legs exist
            if (self._has_direct_pair(from_asset, intermediate) and 
                self._has_direct_pair(intermediate, to_asset)):
                
                # Get quotes for both legs
                leg1_quote = await self.get_best_price(from_asset, intermediate, amount)
                
                if "error" not in leg1_quote:
                    intermediate_amount = Decimal(leg1_quote["output_amount"])
                    leg2_quote = await self.get_best_price(intermediate, to_asset, intermediate_amount)
                    
                    if "error" not in leg2_quote:
                        # Calculate total output
                        final_amount = Decimal(leg2_quote["output_amount"])
                        
                        route = {
                            "route_type": "indirect",
                            "intermediate_asset": intermediate.value,
                            "total_output": str(final_amount),
                            "leg1": leg1_quote,
                            "leg2": leg2_quote,
                            "estimated_time": "2-5 minutes"
                        }
                        
                        best_routes.append(route)
        
        # Sort by total output (descending)
        best_routes.sort(key=lambda x: Decimal(x["total_output"]), reverse=True)
        
        return best_routes[:3]  # Return top 3 routes
    
    def _supports_pair(self, config: ExchangeConfig, from_asset: AssetType, to_asset: AssetType) -> bool:
        """Check if exchange supports trading pair."""
        pair = (from_asset, to_asset)
        reverse_pair = (to_asset, from_asset)
        return pair in config.supported_pairs or reverse_pair in config.supported_pairs
    
    def _has_direct_pair(self, from_asset: AssetType, to_asset: AssetType) -> bool:
        """Check if any exchange supports direct trading pair."""
        for config in self.exchanges.values():
            if config.is_active and self._supports_pair(config, from_asset, to_asset):
                return True
        return False
    
    async def _check_rate_limit(self, exchange_name: str, action: str = "request") -> bool:
        """Check if exchange rate limit allows the action."""
        # Simplified rate limiting - in production this would be more sophisticated
        current_time = datetime.utcnow()
        rate_key = f"{exchange_name}_{action}"
        
        if rate_key not in self.rate_limits:
            self.rate_limits[rate_key] = {"count": 0, "window_start": current_time}
            return True
        
        rate_info = self.rate_limits[rate_key]
        
        # Reset window if needed (1 second windows)
        if current_time - rate_info["window_start"] > timedelta(seconds=1):
            rate_info["count"] = 0
            rate_info["window_start"] = current_time
        
        # Check limits
        config = self.exchanges[exchange_name]
        limit_key = f"{action}s_per_second"
        
        if limit_key in config.rate_limits:
            if rate_info["count"] >= config.rate_limits[limit_key]:
                return False
        
        # Increment counter
        rate_info["count"] += 1
        return True
    
    async def get_exchange_status(self) -> Dict:
        """Get status of all configured exchanges."""
        status = {}
        
        for name, config in self.exchanges.items():
            status[name] = {
                "name": config.name,
                "is_active": config.is_active,
                "sandbox_mode": config.sandbox_mode,
                "supported_pairs": [f"{pair[0].value}-{pair[1].value}" for pair in config.supported_pairs],
                "fee_rates": {k: str(v) for k, v in config.fee_rates.items()},
                "rate_limits": config.rate_limits
            }
        
        return status
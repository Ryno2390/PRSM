"""
CHRONOS Hub-and-Spoke Router

Implements Bitcoin-centric hub-and-spoke architecture for cross-crypto conversions.
Optimizes routing through BTC as primary hub with USDC as USD bridge.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass

from .models import AssetType, SwapType
from .exchange_router import ExchangeRouter
from .price_oracles import price_aggregator, get_exchange_rate

logger = logging.getLogger(__name__)


@dataclass
class RoutingNode:
    """Represents a node in the routing graph."""
    asset: AssetType
    is_hub: bool = False
    is_fiat_bridge: bool = False
    liquidity_score: Decimal = Decimal("0")
    average_fees: Decimal = Decimal("0")


@dataclass
class RoutingPath:
    """Represents a complete routing path between assets."""
    path: List[AssetType]
    total_fees: Decimal
    estimated_slippage: Decimal
    estimated_time_minutes: int
    confidence_score: Decimal  # 0-1 based on liquidity and reliability


class HubSpokeRouter:
    """Bitcoin-centric hub-and-spoke routing for optimal crypto conversions."""
    
    def __init__(self, exchange_router: ExchangeRouter):
        self.exchange_router = exchange_router
        
        # Define network topology
        self.nodes = self._initialize_network_topology()
        
        # Hub hierarchy (primary -> secondary -> tertiary)
        self.hub_hierarchy = [
            AssetType.BTC,    # Primary hub - highest liquidity
            AssetType.ETH,    # Secondary hub - good for DeFi
            AssetType.USDC    # Tertiary hub - fiat bridge
        ]
        
        # Fiat bridges (stablecoins with real USD backing)
        self.fiat_bridges = [
            AssetType.USDC,   # Primary - Circle/regulated
            AssetType.USDT    # Secondary - Tether (higher risk)
        ]
        
        # Cache for routing calculations
        self.routing_cache = {}
        
    def _initialize_network_topology(self) -> Dict[AssetType, RoutingNode]:
        """Initialize the network topology with liquidity scores."""
        nodes = {}
        
        # Bitcoin - Primary hub
        nodes[AssetType.BTC] = RoutingNode(
            asset=AssetType.BTC,
            is_hub=True,
            liquidity_score=Decimal("100"),  # Highest liquidity
            average_fees=Decimal("0.001")    # 0.1% average
        )
        
        # USDC - Fiat bridge
        nodes[AssetType.USDC] = RoutingNode(
            asset=AssetType.USDC,
            is_fiat_bridge=True,
            liquidity_score=Decimal("90"),
            average_fees=Decimal("0.0005")   # 0.05% average
        )
        
        # FTNS - Native token
        nodes[AssetType.FTNS] = RoutingNode(
            asset=AssetType.FTNS,
            liquidity_score=Decimal("75"),   # Good liquidity in PRSM ecosystem
            average_fees=Decimal("0.003")    # 0.3% average
        )
        
        # ETH - Secondary hub
        nodes[AssetType.ETH] = RoutingNode(
            asset=AssetType.ETH,
            is_hub=True,
            liquidity_score=Decimal("85"),
            average_fees=Decimal("0.002")    # 0.2% average
        )
        
        # USD - Fiat
        nodes[AssetType.USD] = RoutingNode(
            asset=AssetType.USD,
            is_fiat_bridge=True,
            liquidity_score=Decimal("100"),  # Always liquid
            average_fees=Decimal("0.005")    # 0.5% for bank transfers
        )
        
        # Other cryptos as spokes
        for asset in [AssetType.USDT, AssetType.ADA, AssetType.SOL, AssetType.DOT]:
            nodes[asset] = RoutingNode(
                asset=asset,
                liquidity_score=Decimal("60"),  # Moderate liquidity
                average_fees=Decimal("0.002")   # 0.2% average
            )
        
        return nodes
    
    async def find_optimal_route(
        self, 
        from_asset: AssetType, 
        to_asset: AssetType,
        amount: Decimal,
        max_hops: int = 3
    ) -> List[RoutingPath]:
        """Find optimal routing paths using hub-and-spoke topology."""
        
        # Check cache first
        cache_key = f"{from_asset}_{to_asset}_{amount}_{max_hops}"
        if cache_key in self.routing_cache:
            cached_data, timestamp = self.routing_cache[cache_key]
            if (datetime.utcnow() - timestamp).seconds < 300:  # 5 minute cache
                return cached_data
        
        if from_asset == to_asset:
            return []  # No conversion needed
        
        # Find all possible paths
        all_paths = []
        
        # 1. Direct path (if available)
        direct_path = await self._calculate_direct_path(from_asset, to_asset, amount)
        if direct_path:
            all_paths.append(direct_path)
        
        # 2. Hub-routed paths
        for hub in self.hub_hierarchy:
            if hub != from_asset and hub != to_asset:
                hub_path = await self._calculate_hub_path(from_asset, to_asset, hub, amount)
                if hub_path:
                    all_paths.append(hub_path)
        
        # 3. Special USD routing (for fiat conversions)
        if from_asset == AssetType.USD or to_asset == AssetType.USD:
            usd_paths = await self._calculate_usd_paths(from_asset, to_asset, amount)
            all_paths.extend(usd_paths)
        
        # Sort by confidence score (fees, slippage, time)
        all_paths.sort(key=lambda p: p.confidence_score, reverse=True)
        
        # Cache results
        self.routing_cache[cache_key] = (all_paths, datetime.utcnow())
        
        return all_paths[:5]  # Return top 5 paths
    
    async def _calculate_direct_path(
        self, 
        from_asset: AssetType, 
        to_asset: AssetType, 
        amount: Decimal
    ) -> Optional[RoutingPath]:
        """Calculate direct path if pair exists."""
        
        # Check if direct pair is available through exchange router
        quote = await self.exchange_router.get_best_price(from_asset, to_asset, amount)
        
        if "error" in quote:
            return None
        
        # Calculate path metrics
        total_fees = Decimal(quote.get("fee_amount", "0"))
        estimated_slippage = self._estimate_slippage(from_asset, to_asset, amount)
        
        return RoutingPath(
            path=[from_asset, to_asset],
            total_fees=total_fees,
            estimated_slippage=estimated_slippage,
            estimated_time_minutes=2,
            confidence_score=self._calculate_confidence_score([from_asset, to_asset], total_fees)
        )
    
    async def _calculate_hub_path(
        self,
        from_asset: AssetType,
        to_asset: AssetType,
        hub: AssetType,
        amount: Decimal
    ) -> Optional[RoutingPath]:
        """Calculate path through specified hub (e.g., BTC)."""
        
        # Step 1: from_asset -> hub
        quote1 = await self.exchange_router.get_best_price(from_asset, hub, amount)
        if "error" in quote1:
            return None
        
        # Step 2: hub -> to_asset
        hub_amount = Decimal(quote1["output_amount"])
        quote2 = await self.exchange_router.get_best_price(hub, to_asset, hub_amount)
        if "error" in quote2:
            return None
        
        # Calculate combined metrics
        total_fees = Decimal(quote1.get("fee_amount", "0")) + Decimal(quote2.get("fee_amount", "0"))
        estimated_slippage = self._estimate_slippage(from_asset, hub, amount) + \
                           self._estimate_slippage(hub, to_asset, hub_amount)
        
        path = [from_asset, hub, to_asset]
        
        return RoutingPath(
            path=path,
            total_fees=total_fees,
            estimated_slippage=estimated_slippage,
            estimated_time_minutes=5,  # Longer for 2-hop
            confidence_score=self._calculate_confidence_score(path, total_fees)
        )
    
    async def _calculate_usd_paths(
        self,
        from_asset: AssetType,
        to_asset: AssetType,
        amount: Decimal
    ) -> List[RoutingPath]:
        """Calculate optimal USD conversion paths."""
        paths = []
        
        if from_asset == AssetType.USD:
            # USD -> crypto: USD -> USDC -> BTC -> target
            for stablecoin in self.fiat_bridges:
                if stablecoin == to_asset:
                    # Direct USD -> stablecoin
                    path = await self._calculate_usd_stablecoin_path(from_asset, to_asset, amount)
                    if path:
                        paths.append(path)
                else:
                    # USD -> stablecoin -> BTC -> target
                    path = await self._calculate_multi_hop_path(
                        [AssetType.USD, stablecoin, AssetType.BTC, to_asset], 
                        amount
                    )
                    if path:
                        paths.append(path)
        
        elif to_asset == AssetType.USD:
            # crypto -> USD: source -> BTC -> USDC -> USD
            for stablecoin in self.fiat_bridges:
                if from_asset == stablecoin:
                    # Direct stablecoin -> USD
                    path = await self._calculate_usd_stablecoin_path(from_asset, to_asset, amount)
                    if path:
                        paths.append(path)
                else:
                    # source -> BTC -> stablecoin -> USD
                    path = await self._calculate_multi_hop_path(
                        [from_asset, AssetType.BTC, stablecoin, AssetType.USD],
                        amount
                    )
                    if path:
                        paths.append(path)
        
        return paths
    
    async def _calculate_usd_stablecoin_path(
        self,
        from_asset: AssetType,
        to_asset: AssetType,
        amount: Decimal
    ) -> Optional[RoutingPath]:
        """Calculate direct USD <-> stablecoin conversion."""
        
        # For USD <-> USDC, this would use real banking APIs
        # For now, simulate with minimal fees
        
        if (from_asset == AssetType.USD and to_asset == AssetType.USDC) or \
           (from_asset == AssetType.USDC and to_asset == AssetType.USD):
            
            # Direct 1:1 conversion with minimal banking fees
            return RoutingPath(
                path=[from_asset, to_asset],
                total_fees=amount * Decimal("0.005"),  # 0.5% banking fee
                estimated_slippage=Decimal("0.001"),   # 0.1% slippage
                estimated_time_minutes=60,             # Banking delay
                confidence_score=Decimal("0.95")       # High confidence
            )
        
        return None
    
    async def _calculate_multi_hop_path(
        self,
        asset_path: List[AssetType],
        initial_amount: Decimal
    ) -> Optional[RoutingPath]:
        """Calculate metrics for multi-hop path."""
        
        current_amount = initial_amount
        total_fees = Decimal("0")
        total_slippage = Decimal("0")
        
        # Calculate each hop
        for i in range(len(asset_path) - 1):
            from_hop = asset_path[i]
            to_hop = asset_path[i + 1]
            
            quote = await self.exchange_router.get_best_price(from_hop, to_hop, current_amount)
            if "error" in quote:
                return None
            
            # Update for next hop
            current_amount = Decimal(quote["output_amount"])
            total_fees += Decimal(quote.get("fee_amount", "0"))
            total_slippage += self._estimate_slippage(from_hop, to_hop, current_amount)
        
        return RoutingPath(
            path=asset_path,
            total_fees=total_fees,
            estimated_slippage=total_slippage,
            estimated_time_minutes=len(asset_path) * 2,  # 2 mins per hop
            confidence_score=self._calculate_confidence_score(asset_path, total_fees)
        )
    
    def _estimate_slippage(self, from_asset: AssetType, to_asset: AssetType, amount: Decimal) -> Decimal:
        """Estimate slippage based on asset pair and amount."""
        
        # Get liquidity scores
        from_liquidity = self.nodes.get(from_asset, RoutingNode(from_asset)).liquidity_score
        to_liquidity = self.nodes.get(to_asset, RoutingNode(to_asset)).liquidity_score
        
        # Lower liquidity = higher slippage
        avg_liquidity = (from_liquidity + to_liquidity) / 2
        
        # Base slippage inversely proportional to liquidity
        base_slippage = Decimal("0.01") * (Decimal("100") - avg_liquidity) / Decimal("100")
        
        # Amount impact (larger amounts = more slippage)
        amount_impact = (amount / Decimal("1000000")) * Decimal("0.001")  # 0.1% per 1M units
        
        return base_slippage + amount_impact
    
    def _calculate_confidence_score(self, path: List[AssetType], total_fees: Decimal) -> Decimal:
        """Calculate confidence score for routing path."""
        
        # Base score starts high for shorter paths
        base_score = Decimal("1.0") - (Decimal(str(len(path) - 1)) * Decimal("0.1"))
        
        # Reduce score for high fees
        fee_penalty = total_fees * Decimal("10")  # 10x fee becomes penalty
        
        # Boost score for hub usage (more reliable)
        hub_bonus = Decimal("0")
        for asset in path:
            if asset in self.hub_hierarchy:
                hub_bonus += Decimal("0.1")
        
        # Boost score for high liquidity assets
        liquidity_bonus = Decimal("0")
        for asset in path:
            node = self.nodes.get(asset)
            if node:
                liquidity_bonus += node.liquidity_score / Decimal("1000")  # Small bonus
        
        final_score = base_score - fee_penalty + hub_bonus + liquidity_bonus
        
        # Clamp to 0-1 range
        return max(Decimal("0"), min(Decimal("1"), final_score))
    
    async def execute_hub_spoke_route(
        self,
        route: RoutingPath,
        amount: Decimal,
        user_id: str
    ) -> Dict:
        """Execute a hub-and-spoke routing path."""
        
        execution_results = []
        current_amount = amount
        
        try:
            # Execute each hop in the path
            for i in range(len(route.path) - 1):
                from_asset = route.path[i]
                to_asset = route.path[i + 1]
                
                logger.info(f"Executing hop {i+1}: {current_amount} {from_asset} -> {to_asset}")
                
                # Execute trade through exchange router
                trade_result = await self.exchange_router.execute_trade(
                    exchange_name="auto_select",  # Let router choose best exchange
                    from_asset=from_asset,
                    to_asset=to_asset,
                    amount=current_amount
                )
                
                if "error" in trade_result:
                    raise Exception(f"Hop {i+1} failed: {trade_result['error']}")
                
                execution_results.append(trade_result)
                current_amount = Decimal(trade_result["output_amount"])
                
                # Brief delay between hops for settlement
                await asyncio.sleep(1)
            
            return {
                "status": "completed",
                "route": [asset.value for asset in route.path],
                "initial_amount": str(amount),
                "final_amount": str(current_amount),
                "total_fees": str(route.total_fees),
                "execution_details": execution_results,
                "execution_time_seconds": len(route.path) * 2
            }
            
        except Exception as e:
            logger.error(f"Route execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "completed_hops": execution_results
            }
    
    async def get_ftns_to_usd_quote(self, ftns_amount: Decimal) -> Dict:
        """Get comprehensive quote for FTNS -> USD conversion."""
        
        # Find optimal routes
        routes = await self.find_optimal_route(AssetType.FTNS, AssetType.USD, ftns_amount)
        
        if not routes:
            return {"error": "No routing path available for FTNS -> USD"}
        
        best_route = routes[0]  # Highest confidence score
        
        return {
            "from_asset": "FTNS",
            "to_asset": "USD",
            "input_amount": str(ftns_amount),
            "routes_available": len(routes),
            "recommended_route": {
                "path": [asset.value for asset in best_route.path],
                "estimated_output": str(ftns_amount - best_route.total_fees),  # Simplified
                "total_fees": str(best_route.total_fees),
                "estimated_slippage": str(best_route.estimated_slippage),
                "estimated_time_minutes": best_route.estimated_time_minutes,
                "confidence_score": str(best_route.confidence_score)
            },
            "alternative_routes": [
                {
                    "path": [asset.value for asset in route.path],
                    "total_fees": str(route.total_fees),
                    "confidence_score": str(route.confidence_score)
                }
                for route in routes[1:3]  # Show top 3 alternatives
            ]
        }
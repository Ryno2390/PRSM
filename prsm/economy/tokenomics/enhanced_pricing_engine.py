"""
Enhanced FTNS Token-Based Pricing Engine
==========================================

Advanced tokenomics pricing system that correlates FTNS costs with computational 
complexity using token-equivalent units, reasoning multipliers, and market dynamics.

This system implements LLM-style token pricing with:
- Computational token mapping
- Reasoning complexity multipliers  
- Market-responsive floating rates
- Quality performance bonuses
- Verbosity scaling factors
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import statistics

from prsm.compute.nwtn.meta_reasoning_engine import ThinkingMode
from enum import Enum

class VerbosityLevel(Enum):
    """Output verbosity levels"""
    BRIEF = "brief"
    STANDARD = "standard" 
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    ACADEMIC = "academic"

logger = logging.getLogger(__name__)

@dataclass
class PricingCalculation:
    """Detailed breakdown of FTNS pricing calculation"""
    query_id: str
    base_computational_tokens: int
    reasoning_multiplier: float
    verbosity_factor: float
    market_rate: float
    quality_bonus: float
    
    # Calculated values
    total_ftns_cost: Decimal
    cost_breakdown: Dict[str, Any]
    pricing_tier: str
    estimated_response_tokens: int
    calculation_timestamp: datetime

@dataclass
class MarketConditions:
    """Current market conditions for pricing"""
    active_queries: int
    available_capacity: int
    network_load_percentage: float
    average_response_time: float
    quality_score: float
    current_market_rate: float
    
    supply_demand_factor: float
    congestion_factor: float
    rate_category: str  # "low", "normal", "high", "peak"
    
    # FTNS token price dynamics
    ftns_token_price_usd: Optional[Decimal] = None
    ftns_price_velocity: Optional[float] = None  # % change per hour
    token_price_factor: float = 1.0  # Multiplier based on token price

@dataclass 
class ResourceRateAdjustments:
    """Dynamic rate adjustments for different resource types"""
    compute_rate_multiplier: float = 1.0
    storage_rate_multiplier: float = 1.0  
    content_rate_multiplier: float = 1.0
    
    # Based on token price velocity
    token_appreciation_factor: float = 1.0
    
    # Timestamp and metadata
    effective_timestamp: datetime = None
    adjustment_reason: str = ""

class PricingTier(Enum):
    """Quality-based pricing tiers"""
    STANDARD = ("standard", 1.0, "Baseline performance")
    HIGH_QUALITY = ("high_quality", 1.15, ">90% accuracy, <5s response")
    PREMIUM = ("premium", 1.3, ">95% accuracy, <3s response") 
    EXCELLENCE = ("excellence", 1.5, ">98% accuracy, <2s response")
    
    def __init__(self, tier_name: str, multiplier: float, description: str):
        self.tier_name = tier_name
        self.multiplier = multiplier
        self.description = description

class EnhancedPricingEngine:
    """
    Advanced FTNS pricing engine implementing token-based computational pricing
    with market dynamics and quality-based adjustments.
    """
    
    def __init__(self, base_rate: float = 1.0):
        self.base_rate = base_rate
        self.market_history: List[MarketConditions] = []
        
        # Reasoning complexity multipliers
        self.reasoning_multipliers = {
            ThinkingMode.QUICK: 1.0,
            ThinkingMode.INTERMEDIATE: 2.5,
            ThinkingMode.DEEP: 5.0
        }
        
        # Verbosity scaling factors
        self.verbosity_factors = {
            VerbosityLevel.BRIEF: 0.5,
            VerbosityLevel.STANDARD: 1.0,
            VerbosityLevel.DETAILED: 1.5,
            VerbosityLevel.COMPREHENSIVE: 2.0,
            VerbosityLevel.ACADEMIC: 3.0
        }
        
        # Token estimation models for different query types
        self.base_token_estimators = {
            "factual": lambda query: min(100, max(20, len(query.split()) * 2)),
            "analytical": lambda query: min(300, max(50, len(query.split()) * 4)),
            "research": lambda query: min(500, max(100, len(query.split()) * 6)),
            "complex": lambda query: min(800, max(200, len(query.split()) * 8)),
            "synthesis": lambda query: min(1200, max(300, len(query.split()) * 10))
        }
        
        logger.info("Enhanced FTNS Pricing Engine initialized")
        
        # Resource rate adjustment configuration
        self.resource_adjustments = ResourceRateAdjustments()
        self.price_history: List[Tuple[datetime, Decimal]] = []  # Track FTNS price history
    
    async def calculate_ftns_cost(
        self,
        query: str,
        thinking_mode: ThinkingMode,
        verbosity_level: VerbosityLevel,
        query_id: str,
        user_tier: Optional[str] = "standard",
        query_type: str = "analytical"
    ) -> PricingCalculation:
        """
        Calculate FTNS cost using token-based pricing model
        
        FTNS_Cost = Base_Tokens × Reasoning_Multiplier × Market_Rate × Quality_Bonus × Verbosity_Factor
        """
        try:
            # 1. Estimate base computational tokens
            base_tokens = await self._estimate_base_tokens(query, query_type)
            
            # 2. Get complexity multipliers
            reasoning_multiplier = self.reasoning_multipliers[thinking_mode]
            verbosity_factor = self.verbosity_factors[verbosity_level]
            
            # 3. Calculate current market conditions
            market_conditions = await self._get_market_conditions()
            market_rate = market_conditions.current_market_rate
            
            # 4. Determine quality bonus based on user tier
            quality_tier = await self._determine_quality_tier(user_tier, market_conditions)
            quality_bonus = quality_tier.multiplier
            
            # 5. Calculate total cost
            raw_cost = (
                base_tokens * 
                reasoning_multiplier * 
                market_rate * 
                quality_bonus * 
                verbosity_factor
            )
            
            total_cost = Decimal(str(raw_cost)).quantize(
                Decimal('0.001'), rounding=ROUND_HALF_UP
            )
            
            # 6. Estimate response tokens for transparency
            estimated_response_tokens = await self._estimate_response_tokens(
                base_tokens, reasoning_multiplier, verbosity_factor
            )
            
            # 7. Create detailed breakdown
            cost_breakdown = {
                "base_tokens": base_tokens,
                "reasoning_multiplier": reasoning_multiplier,
                "verbosity_factor": verbosity_factor,
                "market_rate": market_rate,
                "quality_bonus": quality_bonus,
                "raw_calculation": f"{base_tokens} × {reasoning_multiplier} × {market_rate} × {quality_bonus} × {verbosity_factor}",
                "market_conditions": {
                    "rate_category": market_conditions.rate_category,
                    "network_load": market_conditions.network_load_percentage,
                    "active_queries": market_conditions.active_queries
                },
                "cost_per_estimated_token": float(total_cost) / max(1, estimated_response_tokens)
            }
            
            calculation = PricingCalculation(
                query_id=query_id,
                base_computational_tokens=base_tokens,
                reasoning_multiplier=reasoning_multiplier,
                verbosity_factor=verbosity_factor,
                market_rate=market_rate,
                quality_bonus=quality_bonus,
                total_ftns_cost=total_cost,
                cost_breakdown=cost_breakdown,
                pricing_tier=quality_tier.tier_name,
                estimated_response_tokens=estimated_response_tokens,
                calculation_timestamp=datetime.now(timezone.utc)
            )
            
            logger.info(f"Calculated FTNS cost: {total_cost} for query {query_id[:8]}")
            return calculation
            
        except Exception as e:
            logger.error(f"Failed to calculate FTNS cost: {e}")
            raise
    
    async def _estimate_base_tokens(self, query: str, query_type: str) -> int:
        """Estimate base computational tokens required for query processing"""
        try:
            # Use appropriate estimator based on query type
            estimator = self.base_token_estimators.get(query_type, self.base_token_estimators["analytical"])
            base_tokens = estimator(query)
            
            # Apply complexity heuristics
            complexity_indicators = {
                "comparison": ["compare", "versus", "difference", "better"],
                "analysis": ["analyze", "implications", "effects", "impact"],
                "research": ["research", "literature", "studies", "evidence"], 
                "synthesis": ["synthesize", "combine", "integrate", "merge"],
                "prediction": ["predict", "forecast", "future", "trends"]
            }
            
            # Increase base tokens for detected complexity
            for category, keywords in complexity_indicators.items():
                if any(keyword in query.lower() for keyword in keywords):
                    base_tokens = int(base_tokens * 1.2)
                    break
            
            # Multi-paper queries require more processing
            if "papers" in query.lower() or "studies" in query.lower():
                base_tokens = int(base_tokens * 1.5)
            
            return max(20, min(1000, base_tokens))  # Reasonable bounds
            
        except Exception as e:
            logger.error(f"Failed to estimate base tokens: {e}")
            return 100  # Safe fallback
    
    async def _get_market_conditions(self) -> MarketConditions:
        """Analyze current market conditions for pricing adjustments"""
        try:
            # Simulated market conditions - in production, this would query actual system metrics
            import random
            
            # Mock current system load
            active_queries = random.randint(5, 50)
            available_capacity = random.randint(40, 100)
            network_load = random.uniform(0.2, 0.9)
            avg_response_time = random.uniform(1.5, 8.0)
            quality_score = random.uniform(0.85, 0.98)
            
            # Calculate market factors
            supply_demand_factor = max(0, (active_queries - available_capacity) / available_capacity)
            congestion_factor = network_load
            
            # Get FTNS token price dynamics
            ftns_price = await self._get_ftns_token_price()
            price_velocity = await self._calculate_price_velocity() if ftns_price else 0.0
            token_price_factor = self._calculate_token_price_factor(ftns_price, price_velocity)
            
            # Determine base market rate with token price adjustment
            market_rate = self.base_rate * (1 + supply_demand_factor * 0.5 + congestion_factor * 0.3 + token_price_factor * 0.4)
            
            # Categorize market conditions
            if market_rate <= 1.0:
                rate_category = "low"
            elif market_rate <= 1.2:
                rate_category = "normal"
            elif market_rate <= 2.0:
                rate_category = "high"
            else:
                rate_category = "peak"
            
            conditions = MarketConditions(
                active_queries=active_queries,
                available_capacity=available_capacity,
                network_load_percentage=network_load * 100,
                average_response_time=avg_response_time,
                quality_score=quality_score,
                current_market_rate=market_rate,
                supply_demand_factor=supply_demand_factor,
                congestion_factor=congestion_factor,
                rate_category=rate_category,
                ftns_token_price_usd=ftns_price,
                ftns_price_velocity=price_velocity,
                token_price_factor=token_price_factor
            )
            
            # Store for history
            self.market_history.append(conditions)
            if len(self.market_history) > 1000:  # Keep last 1000 entries
                self.market_history.pop(0)
            
            return conditions
            
        except Exception as e:
            logger.error(f"Failed to get market conditions: {e}")
            # Return safe defaults
            return MarketConditions(
                active_queries=20,
                available_capacity=50,
                network_load_percentage=50.0,
                average_response_time=3.0,
                quality_score=0.90,
                current_market_rate=1.0,
                supply_demand_factor=0.0,
                congestion_factor=0.5,
                rate_category="normal",
                ftns_token_price_usd=None,
                ftns_price_velocity=None,
                token_price_factor=1.0
            )
    
    async def _determine_quality_tier(self, user_tier: str, conditions: MarketConditions) -> PricingTier:
        """Determine quality tier based on user status and system performance"""
        try:
            # Map user tiers to quality tiers
            tier_mapping = {
                "basic": PricingTier.STANDARD,
                "standard": PricingTier.STANDARD,
                "premium": PricingTier.HIGH_QUALITY,
                "enterprise": PricingTier.PREMIUM,
                "research": PricingTier.EXCELLENCE
            }
            
            base_tier = tier_mapping.get(user_tier, PricingTier.STANDARD)
            
            # Upgrade tier based on system performance
            if conditions.quality_score > 0.98 and conditions.average_response_time < 2.0:
                if base_tier == PricingTier.STANDARD:
                    return PricingTier.HIGH_QUALITY
                elif base_tier == PricingTier.HIGH_QUALITY:
                    return PricingTier.PREMIUM
                elif base_tier == PricingTier.PREMIUM:
                    return PricingTier.EXCELLENCE
            
            return base_tier
            
        except Exception as e:
            logger.error(f"Failed to determine quality tier: {e}")
            return PricingTier.STANDARD
    
    async def _estimate_response_tokens(
        self, 
        base_tokens: int, 
        reasoning_multiplier: float, 
        verbosity_factor: float
    ) -> int:
        """Estimate expected response length in tokens for transparency"""
        try:
            # Base response estimation
            base_response = int(base_tokens * 0.8)  # Response typically 80% of processing
            
            # Apply reasoning complexity
            reasoning_adjusted = int(base_response * (1 + (reasoning_multiplier - 1) * 0.3))
            
            # Apply verbosity scaling
            final_tokens = int(reasoning_adjusted * verbosity_factor)
            
            return max(50, min(5000, final_tokens))  # Reasonable bounds
            
        except Exception as e:
            logger.error(f"Failed to estimate response tokens: {e}")
            return 200  # Safe fallback
    
    async def get_pricing_preview(
        self, 
        query: str,
        thinking_mode: ThinkingMode,
        verbosity_level: VerbosityLevel
    ) -> Dict[str, Any]:
        """Get pricing preview before executing query"""
        try:
            temp_calculation = await self.calculate_ftns_cost(
                query=query,
                thinking_mode=thinking_mode,
                verbosity_level=verbosity_level,
                query_id="preview",
                query_type=await self._classify_query_type(query)
            )
            
            return {
                "estimated_cost": float(temp_calculation.total_ftns_cost),
                "cost_breakdown": temp_calculation.cost_breakdown,
                "estimated_response_tokens": temp_calculation.estimated_response_tokens,
                "pricing_tier": temp_calculation.pricing_tier,
                "market_rate_category": temp_calculation.cost_breakdown["market_conditions"]["rate_category"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get pricing preview: {e}")
            return {"estimated_cost": 10.0, "error": str(e)}
    
    async def _classify_query_type(self, query: str) -> str:
        """Classify query type for appropriate token estimation"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what is", "define", "explain"]):
            return "factual"
        elif any(word in query_lower for word in ["analyze", "compare", "evaluate"]):
            return "analytical"  
        elif any(word in query_lower for word in ["research", "studies", "literature"]):
            return "research"
        elif any(word in query_lower for word in ["synthesize", "combine", "integrate"]):
            return "synthesis"
        elif len(query.split()) > 20:
            return "complex"
        else:
            return "analytical"
    
    async def get_market_metrics(self) -> Dict[str, Any]:
        """Get current market metrics for monitoring"""
        try:
            if not self.market_history:
                current = await self._get_market_conditions()
            else:
                current = self.market_history[-1]
            
            # Calculate recent averages
            recent_history = self.market_history[-10:] if len(self.market_history) >= 10 else self.market_history
            
            avg_market_rate = statistics.mean([m.current_market_rate for m in recent_history]) if recent_history else 1.0
            avg_load = statistics.mean([m.network_load_percentage for m in recent_history]) if recent_history else 50.0
            
            return {
                "current_market_rate": current.current_market_rate,
                "average_market_rate": avg_market_rate,
                "rate_category": current.rate_category,
                "network_load_percent": current.network_load_percentage,
                "average_load_percent": avg_load,
                "active_queries": current.active_queries,
                "available_capacity": current.available_capacity,
                "system_quality_score": current.quality_score,
                "average_response_time": current.average_response_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get market metrics: {e}")
            return {"error": str(e)}
    
    async def _get_ftns_token_price(self) -> Optional[Decimal]:
        """Get current FTNS token price from oracle"""
        try:
            # Import oracle here to avoid circular imports
            from prsm.economy.blockchain.ftns_oracle import get_ftns_oracle
            
            oracle = await get_ftns_oracle()
            if oracle:
                price_data = await oracle.get_oracle_price()
                current_price = Decimal(str(price_data.price_usd))
                
                # Store price history
                now = datetime.now(timezone.utc)
                self.price_history.append((now, current_price))
                
                # Keep only last 100 price points
                if len(self.price_history) > 100:
                    self.price_history.pop(0)
                
                return current_price
            else:
                logger.warning("FTNS oracle not available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get FTNS token price: {e}")
            return None
    
    async def _calculate_price_velocity(self) -> float:
        """Calculate FTNS token price velocity (% change per hour)"""
        try:
            if len(self.price_history) < 2:
                return 0.0
            
            # Get price from 1 hour ago (or closest available)
            now = datetime.now(timezone.utc)
            one_hour_ago = now.timestamp() - 3600  # 1 hour in seconds
            
            # Find closest price point to 1 hour ago
            closest_point = None
            min_time_diff = float('inf')
            
            for timestamp, price in self.price_history:
                time_diff = abs(timestamp.timestamp() - one_hour_ago)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_point = (timestamp, price)
            
            if closest_point and len(self.price_history) > 0:
                old_price = closest_point[1]
                current_price = self.price_history[-1][1]
                
                if old_price > 0:
                    velocity = float((current_price - old_price) / old_price * 100)  # % change per hour
                    return velocity
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate price velocity: {e}")
            return 0.0
    
    def _calculate_token_price_factor(self, price: Optional[Decimal], velocity: float) -> float:
        """
        Calculate adjustment factor based on FTNS token price dynamics
        
        Logic: 
        - Higher token price → Higher service costs (but dampened to maintain accessibility)
        - Rapid appreciation → Temporary rate increases to attract more resources
        - Price depreciation → Rate decreases to maintain competitiveness
        """
        try:
            if not price:
                return 0.0  # No adjustment if price unavailable
            
            # Base factor from absolute price level (very mild adjustment)
            # Assuming $0.10 as baseline FTNS price
            baseline_price = Decimal("0.10")
            price_level_factor = float((price / baseline_price - 1) * 0.1)  # 10% adjustment per 100% price change
            
            # Velocity-based factor (more responsive to price changes)
            velocity_factor = 0.0
            if abs(velocity) > 1.0:  # Only adjust for significant velocity (>1% per hour)
                # Cap velocity impact to prevent extreme adjustments
                capped_velocity = max(-20, min(20, velocity))  # Cap at ±20% per hour
                velocity_factor = capped_velocity * 0.02  # 2% adjustment per 1% velocity
            
            # Combined factor (price level + velocity)
            total_factor = price_level_factor + velocity_factor
            
            # Cap total adjustment to reasonable range (-50% to +100%)
            total_factor = max(-0.5, min(1.0, total_factor))
            
            logger.info(f"Token price factor: price={price}, velocity={velocity}%, adjustment={total_factor:.3f}")
            
            return total_factor
            
        except Exception as e:
            logger.error(f"Failed to calculate token price factor: {e}")
            return 0.0
    
    def update_resource_rate_adjustments(self, token_appreciation_rate: float) -> ResourceRateAdjustments:
        """
        Update resource rate adjustments based on token appreciation
        
        Higher token prices → Higher resource rates → More providers → Lower operational costs
        """
        try:
            now = datetime.now(timezone.utc)
            
            # Calculate resource-specific multipliers based on token appreciation
            if token_appreciation_rate > 0:
                # Token appreciating - increase resource rates to attract providers
                compute_multiplier = 1.0 + (token_appreciation_rate * 1.5)  # Most aggressive
                storage_multiplier = 1.0 + (token_appreciation_rate * 1.2)  # Moderate
                content_multiplier = 1.0 + (token_appreciation_rate * 0.8)  # Conservative
                reason = f"Token appreciation at {token_appreciation_rate:.2%}/hour"
            elif token_appreciation_rate < -0.05:  # Only adjust for significant depreciation
                # Token depreciating - reduce rates to maintain provider participation
                compute_multiplier = 1.0 + (token_appreciation_rate * 0.8)  # Gentler reduction
                storage_multiplier = 1.0 + (token_appreciation_rate * 0.6)
                content_multiplier = 1.0 + (token_appreciation_rate * 0.4)
                reason = f"Token depreciation at {token_appreciation_rate:.2%}/hour"
            else:
                # Stable token price - no adjustment
                compute_multiplier = 1.0
                storage_multiplier = 1.0
                content_multiplier = 1.0
                reason = "Token price stable"
            
            # Update resource adjustments
            self.resource_adjustments = ResourceRateAdjustments(
                compute_rate_multiplier=compute_multiplier,
                storage_rate_multiplier=storage_multiplier,
                content_rate_multiplier=content_multiplier,
                token_appreciation_factor=1.0 + token_appreciation_rate,
                effective_timestamp=now,
                adjustment_reason=reason
            )
            
            logger.info(f"Resource rate adjustments updated: {reason}")
            logger.info(f"  Compute: {compute_multiplier:.3f}x, Storage: {storage_multiplier:.3f}x, Content: {content_multiplier:.3f}x")
            
            return self.resource_adjustments
            
        except Exception as e:
            logger.error(f"Failed to update resource rate adjustments: {e}")
            return self.resource_adjustments
    
    def get_resource_adjusted_cost(self, base_cost: Decimal, resource_type: str) -> Decimal:
        """Apply resource rate adjustments to base costs"""
        try:
            if resource_type.lower() in ['compute', 'gpu', 'cpu', 'inference']:
                multiplier = self.resource_adjustments.compute_rate_multiplier
            elif resource_type.lower() in ['storage', 'data', 'ipfs', 'cache']:
                multiplier = self.resource_adjustments.storage_rate_multiplier
            elif resource_type.lower() in ['content', 'corpus', 'dataset']:
                multiplier = self.resource_adjustments.content_rate_multiplier
            else:
                multiplier = 1.0  # Default: no adjustment
            
            adjusted_cost = base_cost * Decimal(str(multiplier))
            return adjusted_cost.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
            
        except Exception as e:
            logger.error(f"Failed to apply resource adjustment: {e}")
            return base_cost

# Initialize global pricing engine
pricing_engine = EnhancedPricingEngine()

async def calculate_query_cost(
    query: str,
    thinking_mode: ThinkingMode,
    verbosity_level: VerbosityLevel,
    query_id: str,
    user_tier: str = "standard"
) -> PricingCalculation:
    """Convenience function for calculating FTNS query costs"""
    return await pricing_engine.calculate_ftns_cost(
        query=query,
        thinking_mode=thinking_mode,
        verbosity_level=verbosity_level,
        query_id=query_id,
        user_tier=user_tier
    )

async def get_pricing_preview(
    query: str,
    thinking_mode: ThinkingMode,
    verbosity_level: VerbosityLevel
) -> Dict[str, Any]:
    """Get pricing preview for query"""
    return await pricing_engine.get_pricing_preview(query, thinking_mode, verbosity_level)
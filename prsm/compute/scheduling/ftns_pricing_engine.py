"""
Dynamic FTNS Pricing Engine

💰 FTNS MARKETPLACE PRICING SYSTEM:
- Real-time demand monitoring and dynamic price calculation
- Historical pricing data storage and trend analysis
- Cost optimization recommendations for users
- Load balancing through economic incentives
- Arbitrage opportunity detection for FTNS trading

This module implements sophisticated pricing algorithms that:
1. Balance system load through dynamic pricing
2. Provide cost optimization opportunities for users
3. Enable FTNS trading marketplace functionality
4. Predict optimal execution times based on pricing trends
"""

import asyncio
import json
import hashlib
import time
import math
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4
from collections import defaultdict, deque
import statistics
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import PRSMBaseModel, TimestampMixin
from prsm.compute.scheduling.workflow_scheduler import ResourceType

logger = structlog.get_logger(__name__)


class PricingModel(str, Enum):
    """Pricing model types"""
    FIXED = "fixed"                    # Fixed pricing regardless of demand
    LINEAR_DEMAND = "linear_demand"    # Linear relationship with demand
    EXPONENTIAL_DEMAND = "exponential_demand"  # Exponential scaling with demand
    SURGE_PRICING = "surge_pricing"    # Uber-style surge pricing
    AUCTION_BASED = "auction_based"    # Auction-style dynamic pricing
    ML_PREDICTED = "ml_predicted"      # Machine learning predicted pricing


class MarketCondition(str, Enum):
    """Market condition indicators"""
    OVERSUPPLY = "oversupply"          # Low demand, high availability
    BALANCED = "balanced"              # Normal supply/demand balance
    HIGH_DEMAND = "high_demand"        # High demand, normal supply
    CONSTRAINED = "constrained"        # High demand, low supply
    CRITICAL = "critical"              # Very high demand, very low supply


class ArbitrageOpportunity(PRSMBaseModel):
    """FTNS arbitrage opportunity"""
    opportunity_id: UUID = Field(default_factory=uuid4)
    resource_type: ResourceType
    
    # Timing
    buy_time: datetime
    sell_time: datetime
    opportunity_window: timedelta
    
    # Pricing
    current_price: float
    predicted_peak_price: float
    profit_potential: float
    profit_percentage: float
    
    # Risk assessment
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(default="medium")  # "low", "medium", "high"
    
    # Market conditions
    current_demand: float
    predicted_demand: float
    supply_forecast: float
    
    # Recommendation
    recommended_action: str = Field(default="monitor")  # "buy", "sell", "hold", "monitor"
    recommendation_reason: str = Field(default="")


class PricingDataPoint(PRSMBaseModel):
    """Historical pricing data point"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resource_type: ResourceType
    
    # Price data
    base_price: float
    actual_price: float
    demand_multiplier: float
    
    # Market conditions
    demand_level: float
    supply_level: float
    utilization_percentage: float
    
    # Context
    active_workflows: int
    queued_workflows: int
    peak_hour: bool
    weekend: bool
    
    # External factors
    special_events: List[str] = Field(default_factory=list)
    system_load: float = Field(default=0.0)


class PricingForecast(PRSMBaseModel):
    """Price forecast for future time periods"""
    forecast_id: UUID = Field(default_factory=uuid4)
    resource_type: ResourceType
    
    # Forecast period
    start_time: datetime
    end_time: datetime
    forecast_horizon: timedelta
    
    # Price predictions
    predicted_prices: List[Tuple[datetime, float]] = Field(default_factory=list)
    confidence_intervals: List[Tuple[datetime, float, float]] = Field(default_factory=list)  # (time, low, high)
    
    # Market predictions
    demand_forecast: List[Tuple[datetime, float]] = Field(default_factory=list)
    supply_forecast: List[Tuple[datetime, float]] = Field(default_factory=list)
    
    # Recommendations
    optimal_execution_windows: List[Tuple[datetime, datetime, float]] = Field(default_factory=list)  # (start, end, price)
    cost_saving_opportunities: List[ArbitrageOpportunity] = Field(default_factory=list)
    
    # Forecast metadata
    model_used: str = Field(default="hybrid")
    forecast_accuracy: Optional[float] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FTNSPricingEngine(TimestampMixin):
    """
    Dynamic FTNS Pricing Engine
    
    Sophisticated pricing system that balances system load through economic
    incentives while providing cost optimization opportunities for users.
    """
    
    def __init__(self):
        super().__init__()
        
        # Pricing configuration
        self.pricing_models: Dict[ResourceType, PricingModel] = {}
        self.base_prices: Dict[ResourceType, float] = {}
        self.price_history: Dict[ResourceType, List[PricingDataPoint]] = defaultdict(list)
        
        # Market monitoring
        self.current_demand: Dict[ResourceType, float] = defaultdict(float)
        self.current_supply: Dict[ResourceType, float] = defaultdict(float)
        self.market_conditions: Dict[ResourceType, MarketCondition] = {}
        
        # Forecasting
        self.active_forecasts: Dict[ResourceType, PricingForecast] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        
        # Performance tracking
        self.pricing_statistics: Dict[str, Any] = defaultdict(float)
        self.revenue_optimization: float = 0.0
        self.user_cost_savings: float = 0.0
        
        # Configuration
        self.price_update_interval = timedelta(minutes=5)
        self.max_price_multiplier = 5.0  # Maximum 5x surge pricing
        self.min_price_multiplier = 0.3  # Minimum 30% of base price
        self.volatility_threshold = 0.2  # 20% price change triggers analysis
        self.forecast_horizon = timedelta(hours=24)
        
        self._initialize_pricing_models()
        self._start_pricing_monitor()
        
        logger.info("FTNSPricingEngine initialized")
    
    def _initialize_pricing_models(self):
        """Initialize pricing models and base prices"""
        # Set default pricing models
        default_models = {
            ResourceType.CPU_CORES: PricingModel.EXPONENTIAL_DEMAND,
            ResourceType.MEMORY_GB: PricingModel.LINEAR_DEMAND,
            ResourceType.GPU_UNITS: PricingModel.SURGE_PRICING,
            ResourceType.FTNS_CREDITS: PricingModel.AUCTION_BASED
        }
        
        # Set base prices (per hour)
        default_base_prices = {
            ResourceType.CPU_CORES: 0.10,  # $0.10 per core-hour
            ResourceType.MEMORY_GB: 0.05,  # $0.05 per GB-hour
            ResourceType.GPU_UNITS: 2.50,  # $2.50 per GPU-hour
            ResourceType.STORAGE_GB: 0.01, # $0.01 per GB-hour
            ResourceType.NETWORK_MBPS: 0.02, # $0.02 per Mbps-hour
            ResourceType.MODEL_INFERENCE_TOKENS: 0.001, # $0.001 per 1K tokens
            ResourceType.FTNS_CREDITS: 1.00  # $1.00 per FTNS credit
        }
        
        for resource_type in ResourceType:
            self.pricing_models[resource_type] = default_models.get(
                resource_type, PricingModel.LINEAR_DEMAND
            )
            self.base_prices[resource_type] = default_base_prices.get(
                resource_type, 1.0
            )
            self.market_conditions[resource_type] = MarketCondition.BALANCED
    
    def _start_pricing_monitor(self):
        """Start the pricing monitoring loop"""
        # In production, this would run as a background task
        pass
    
    async def update_market_conditions(
        self,
        resource_type: ResourceType,
        demand_level: float,
        supply_level: float,
        active_workflows: int,
        queued_workflows: int
    ):
        """
        Update market conditions for dynamic pricing
        
        Args:
            resource_type: Type of resource
            demand_level: Current demand level (0.0-1.0)
            supply_level: Current supply level (0.0-1.0) 
            active_workflows: Number of active workflows
            queued_workflows: Number of queued workflows
        """
        try:
            # Update current market state
            self.current_demand[resource_type] = demand_level
            self.current_supply[resource_type] = supply_level
            
            # Determine market condition
            if supply_level > 0.8 and demand_level < 0.3:
                condition = MarketCondition.OVERSUPPLY
            elif supply_level < 0.2 and demand_level > 0.8:
                condition = MarketCondition.CRITICAL
            elif demand_level > 0.7:
                condition = MarketCondition.HIGH_DEMAND
            elif supply_level < 0.4 and demand_level > 0.5:
                condition = MarketCondition.CONSTRAINED
            else:
                condition = MarketCondition.BALANCED
            
            self.market_conditions[resource_type] = condition
            
            # Calculate new price
            new_price = await self._calculate_dynamic_price(
                resource_type, demand_level, supply_level
            )
            
            # Record pricing data point
            data_point = PricingDataPoint(
                resource_type=resource_type,
                base_price=self.base_prices[resource_type],
                actual_price=new_price,
                demand_multiplier=new_price / self.base_prices[resource_type],
                demand_level=demand_level,
                supply_level=supply_level,
                utilization_percentage=demand_level * 100,
                active_workflows=active_workflows,
                queued_workflows=queued_workflows,
                peak_hour=self._is_peak_hour(),
                weekend=self._is_weekend()
            )
            
            self.price_history[resource_type].append(data_point)
            
            # Trim history to last 30 days
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=30)
            self.price_history[resource_type] = [
                dp for dp in self.price_history[resource_type]
                if dp.timestamp > cutoff_time
            ]
            
            # Update statistics
            self.pricing_statistics["price_updates"] += 1
            self.pricing_statistics[f"updates_{resource_type.value}"] += 1
            
            # Detect arbitrage opportunities
            await self._detect_arbitrage_opportunities(resource_type)
            
            logger.info(
                "Market conditions updated",
                resource_type=resource_type.value,
                demand=demand_level,
                supply=supply_level,
                condition=condition.value,
                new_price=new_price
            )
            
        except Exception as e:
            logger.error("Error updating market conditions", error=str(e))
    
    async def _calculate_dynamic_price(
        self,
        resource_type: ResourceType,
        demand_level: float,
        supply_level: float
    ) -> float:
        """
        Calculate dynamic price based on demand and supply
        
        Args:
            resource_type: Type of resource
            demand_level: Current demand level (0.0-1.0)
            supply_level: Current supply level (0.0-1.0)
            
        Returns:
            Calculated price
        """
        try:
            base_price = self.base_prices[resource_type]
            pricing_model = self.pricing_models[resource_type]
            
            # Calculate demand/supply ratio
            if supply_level > 0:
                demand_supply_ratio = demand_level / supply_level
            else:
                demand_supply_ratio = 10.0  # High ratio when no supply
            
            # Apply pricing model
            if pricing_model == PricingModel.FIXED:
                multiplier = 1.0
                
            elif pricing_model == PricingModel.LINEAR_DEMAND:
                # Linear scaling: 0.5x to 2.0x based on demand
                multiplier = 0.5 + (1.5 * demand_level)
                
            elif pricing_model == PricingModel.EXPONENTIAL_DEMAND:
                # Exponential scaling for high demand sensitivity
                multiplier = math.pow(2.0, demand_level * 2 - 1)
                
            elif pricing_model == PricingModel.SURGE_PRICING:
                # Uber-style surge pricing
                if demand_supply_ratio > 2.0:
                    multiplier = min(5.0, 1.5 + (demand_supply_ratio - 2.0) * 0.5)
                elif demand_supply_ratio < 0.5:
                    multiplier = max(0.3, 1.0 - (0.5 - demand_supply_ratio) * 0.4)
                else:
                    multiplier = 1.0
                    
            elif pricing_model == PricingModel.AUCTION_BASED:
                # Auction-style pricing based on competition
                competition_factor = min(3.0, demand_supply_ratio)
                multiplier = 0.5 + (competition_factor * 0.5)
                
            elif pricing_model == PricingModel.ML_PREDICTED:
                # Placeholder for ML-based pricing
                multiplier = await self._ml_predict_price_multiplier(
                    resource_type, demand_level, supply_level
                )
                
            else:
                multiplier = 1.0
            
            # Apply constraints
            multiplier = max(self.min_price_multiplier, min(self.max_price_multiplier, multiplier))
            
            # Add time-based adjustments
            time_multiplier = self._get_time_based_multiplier()
            final_multiplier = multiplier * time_multiplier
            
            return base_price * final_multiplier
            
        except Exception as e:
            logger.error("Error calculating dynamic price", error=str(e))
            return self.base_prices[resource_type]
    
    async def _ml_predict_price_multiplier(
        self,
        resource_type: ResourceType,
        demand_level: float,
        supply_level: float
    ) -> float:
        """ML-based price prediction (placeholder implementation)"""
        # In production, this would use trained ML models
        # For now, use a hybrid approach
        
        # Analyze historical patterns
        history = self.price_history[resource_type]
        if len(history) < 10:
            return 1.0  # Not enough data
        
        # Find similar conditions in history
        similar_conditions = []
        for dp in history[-100:]:  # Last 100 data points
            demand_diff = abs(dp.demand_level - demand_level)
            supply_diff = abs(dp.supply_level - supply_level)
            
            if demand_diff < 0.1 and supply_diff < 0.1:
                similar_conditions.append(dp.demand_multiplier)
        
        if similar_conditions:
            # Use median of similar conditions
            return statistics.median(similar_conditions)
        else:
            # Fallback to exponential model
            return math.pow(2.0, demand_level * 2 - 1)
    
    def _get_time_based_multiplier(self) -> float:
        """Get time-based pricing multiplier"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        day_of_week = now.weekday()
        
        # Peak business hours (9 AM - 5 PM weekdays)
        if day_of_week < 5 and 9 <= hour <= 17:
            return 1.3  # 30% premium
        # Evening hours (6 PM - 10 PM)
        elif 18 <= hour <= 22:
            return 1.1  # 10% premium
        # Late night/early morning (11 PM - 6 AM)
        elif hour >= 23 or hour <= 6:
            return 0.7  # 30% discount
        # Weekend
        elif day_of_week >= 5:
            return 0.9  # 10% discount
        else:
            return 1.0  # Standard pricing
    
    def _is_peak_hour(self) -> bool:
        """Check if current time is peak hour"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        day_of_week = now.weekday()
        return day_of_week < 5 and 9 <= hour <= 17
    
    def _is_weekend(self) -> bool:
        """Check if current time is weekend"""
        return datetime.now(timezone.utc).weekday() >= 5
    
    async def get_current_price(self, resource_type: ResourceType) -> float:
        """Get current price for resource type"""
        if not self.price_history[resource_type]:
            return self.base_prices[resource_type]
        
        latest_data = self.price_history[resource_type][-1]
        return latest_data.actual_price
    
    async def generate_price_forecast(
        self,
        resource_type: ResourceType,
        forecast_horizon: Optional[timedelta] = None
    ) -> PricingForecast:
        """
        Generate price forecast for specified horizon
        
        Args:
            resource_type: Type of resource to forecast
            forecast_horizon: Time horizon for forecast
            
        Returns:
            Price forecast with predictions and recommendations
        """
        try:
            horizon = forecast_horizon or self.forecast_horizon
            start_time = datetime.now(timezone.utc)
            end_time = start_time + horizon
            
            forecast = PricingForecast(
                resource_type=resource_type,
                start_time=start_time,
                end_time=end_time,
                forecast_horizon=horizon
            )
            
            # Generate hourly predictions
            current_time = start_time
            predicted_prices = []
            demand_forecasts = []
            supply_forecasts = []
            
            while current_time <= end_time:
                # Predict demand and supply
                predicted_demand = await self._predict_demand_at_time(resource_type, current_time)
                predicted_supply = await self._predict_supply_at_time(resource_type, current_time)
                
                # Calculate predicted price
                predicted_price = await self._calculate_dynamic_price(
                    resource_type, predicted_demand, predicted_supply
                )
                
                predicted_prices.append((current_time, predicted_price))
                demand_forecasts.append((current_time, predicted_demand))
                supply_forecasts.append((current_time, predicted_supply))
                
                current_time += timedelta(hours=1)
            
            forecast.predicted_prices = predicted_prices
            forecast.demand_forecast = demand_forecasts
            forecast.supply_forecast = supply_forecasts
            
            # Calculate confidence intervals (simple approach)
            confidence_intervals = []
            for time_point, price in predicted_prices:
                volatility = 0.1  # 10% volatility
                low_price = price * (1 - volatility)
                high_price = price * (1 + volatility)
                confidence_intervals.append((time_point, low_price, high_price))
            forecast.confidence_intervals = confidence_intervals
            
            # Find optimal execution windows
            forecast.optimal_execution_windows = self._find_optimal_windows(predicted_prices)
            
            # Detect arbitrage opportunities
            forecast.cost_saving_opportunities = await self._find_forecast_arbitrage(
                resource_type, predicted_prices
            )
            
            # Store forecast
            self.active_forecasts[resource_type] = forecast
            
            logger.info(
                "Price forecast generated",
                resource_type=resource_type.value,
                horizon_hours=horizon.total_seconds() / 3600,
                price_points=len(predicted_prices),
                optimal_windows=len(forecast.optimal_execution_windows)
            )
            
            return forecast
            
        except Exception as e:
            logger.error("Error generating price forecast", error=str(e))
            # Return empty forecast
            return PricingForecast(
                resource_type=resource_type,
                start_time=start_time,
                end_time=end_time,
                forecast_horizon=horizon
            )
    
    async def _predict_demand_at_time(
        self,
        resource_type: ResourceType,
        target_time: datetime
    ) -> float:
        """Predict demand level at specific time"""
        # Simple pattern-based prediction
        hour = target_time.hour
        day_of_week = target_time.weekday()
        
        # Base demand patterns
        if day_of_week < 5:  # Weekday
            if 9 <= hour <= 17:  # Business hours
                base_demand = 0.8
            elif 18 <= hour <= 22:  # Evening
                base_demand = 0.6
            else:  # Night/early morning
                base_demand = 0.3
        else:  # Weekend
            if 10 <= hour <= 18:  # Casual hours
                base_demand = 0.5
            else:
                base_demand = 0.2
        
        # Add randomness and trends
        import random
        random_factor = random.uniform(0.8, 1.2)
        
        return min(1.0, base_demand * random_factor)
    
    async def _predict_supply_at_time(
        self,
        resource_type: ResourceType,
        target_time: datetime
    ) -> float:
        """Predict supply level at specific time"""
        # Assume relatively stable supply with slight variations
        import random
        base_supply = 0.7  # 70% baseline availability
        
        # Lower supply during peak maintenance windows
        hour = target_time.hour
        if 2 <= hour <= 5:  # Maintenance window
            base_supply *= 0.8
        
        random_factor = random.uniform(0.9, 1.1)
        return min(1.0, base_supply * random_factor)
    
    def _find_optimal_windows(
        self,
        predicted_prices: List[Tuple[datetime, float]]
    ) -> List[Tuple[datetime, datetime, float]]:
        """Find optimal execution windows with lowest prices"""
        if len(predicted_prices) < 2:
            return []
        
        # Sort by price to find lowest cost periods
        sorted_prices = sorted(predicted_prices, key=lambda x: x[1])
        
        # Find continuous low-price windows
        optimal_windows = []
        low_price_threshold = sorted_prices[len(sorted_prices) // 4][1]  # 25th percentile
        
        window_start = None
        for time_point, price in predicted_prices:
            if price <= low_price_threshold:
                if window_start is None:
                    window_start = time_point
            else:
                if window_start is not None:
                    # End of low-price window
                    window_end = time_point - timedelta(hours=1)
                    avg_price = statistics.mean([p for t, p in predicted_prices 
                                               if window_start <= t <= window_end])
                    optimal_windows.append((window_start, window_end, avg_price))
                    window_start = None
        
        # Handle case where window extends to end
        if window_start is not None:
            window_end = predicted_prices[-1][0]
            avg_price = statistics.mean([p for t, p in predicted_prices 
                                       if window_start <= t <= window_end])
            optimal_windows.append((window_start, window_end, avg_price))
        
        return optimal_windows[:5]  # Return top 5 windows
    
    async def _find_forecast_arbitrage(
        self,
        resource_type: ResourceType,
        predicted_prices: List[Tuple[datetime, float]]
    ) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities in price forecast"""
        opportunities = []
        
        if len(predicted_prices) < 2:
            return opportunities
        
        # Find buy-low, sell-high opportunities
        min_price = min(predicted_prices, key=lambda x: x[1])
        max_price = max(predicted_prices, key=lambda x: x[1])
        
        # Only consider significant price differences (>20%)
        if max_price[1] / min_price[1] > 1.2:
            profit_potential = max_price[1] - min_price[1]
            profit_percentage = (profit_potential / min_price[1]) * 100
            
            opportunity = ArbitrageOpportunity(
                resource_type=resource_type,
                buy_time=min_price[0],
                sell_time=max_price[0],
                opportunity_window=max_price[0] - min_price[0],
                current_price=min_price[1],
                predicted_peak_price=max_price[1],
                profit_potential=profit_potential,
                profit_percentage=profit_percentage,
                confidence_score=0.7,  # Medium confidence for forecasts
                risk_level="medium",
                current_demand=self.current_demand.get(resource_type, 0.5),
                predicted_demand=0.8,  # Assume high demand at peak price
                supply_forecast=0.6,
                recommended_action="buy" if profit_percentage > 25 else "monitor",
                recommendation_reason=f"Potential {profit_percentage:.1f}% profit opportunity"
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _detect_arbitrage_opportunities(self, resource_type: ResourceType):
        """Detect immediate arbitrage opportunities"""
        try:
            history = self.price_history[resource_type]
            if len(history) < 10:
                return
            
            # Analyze recent price volatility
            recent_prices = [dp.actual_price for dp in history[-10:]]
            recent_times = [dp.timestamp for dp in history[-10:]]
            
            if len(recent_prices) < 2:
                return
            
            # Look for significant price drops that might indicate good buy opportunities
            current_price = recent_prices[-1]
            max_recent_price = max(recent_prices[:-1])
            
            if current_price < max_recent_price * 0.8:  # 20% price drop
                # Generate forecast to check if price might recover
                forecast = await self.generate_price_forecast(resource_type, timedelta(hours=6))
                
                if forecast.predicted_prices:
                    future_prices = [price for _, price in forecast.predicted_prices]
                    max_future_price = max(future_prices)
                    
                    if max_future_price > current_price * 1.15:  # 15% potential gain
                        opportunity = ArbitrageOpportunity(
                            resource_type=resource_type,
                            buy_time=datetime.now(timezone.utc),
                            sell_time=forecast.predicted_prices[future_prices.index(max_future_price)][0],
                            opportunity_window=timedelta(hours=6),
                            current_price=current_price,
                            predicted_peak_price=max_future_price,
                            profit_potential=max_future_price - current_price,
                            profit_percentage=((max_future_price - current_price) / current_price) * 100,
                            confidence_score=0.6,
                            risk_level="medium",
                            current_demand=self.current_demand.get(resource_type, 0.5),
                            predicted_demand=0.8,
                            supply_forecast=0.6,
                            recommended_action="buy",
                            recommendation_reason="Price dropped significantly, recovery expected"
                        )
                        
                        self.arbitrage_opportunities.append(opportunity)
                        
                        # Trim old opportunities
                        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                        self.arbitrage_opportunities = [
                            opp for opp in self.arbitrage_opportunities
                            if opp.buy_time > cutoff_time
                        ]
                        
                        logger.info(
                            "Arbitrage opportunity detected",
                            resource_type=resource_type.value,
                            profit_percentage=opportunity.profit_percentage,
                            confidence=opportunity.confidence_score
                        )
            
        except Exception as e:
            logger.error("Error detecting arbitrage opportunities", error=str(e))
    
    def get_cost_optimization_recommendations(
        self,
        resource_type: ResourceType,
        execution_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """
        Get cost optimization recommendations for users
        
        Args:
            resource_type: Type of resource
            execution_window: Time window for execution flexibility
            
        Returns:
            Cost optimization recommendations
        """
        try:
            current_price = self.price_history[resource_type][-1].actual_price if self.price_history[resource_type] else self.base_prices[resource_type]
            
            # Get forecast if available
            forecast = self.active_forecasts.get(resource_type)
            if not forecast:
                return {
                    "current_price": current_price,
                    "recommendation": "execute_now",
                    "reason": "No forecast data available",
                    "potential_savings": 0.0
                }
            
            # Find best execution time within window
            end_time = datetime.now(timezone.utc) + execution_window
            relevant_prices = [
                (time, price) for time, price in forecast.predicted_prices
                if time <= end_time
            ]
            
            if not relevant_prices:
                return {
                    "current_price": current_price,
                    "recommendation": "execute_now", 
                    "reason": "No forecast data within execution window",
                    "potential_savings": 0.0
                }
            
            # Find minimum price in window
            min_price_point = min(relevant_prices, key=lambda x: x[1])
            min_price = min_price_point[1]
            min_price_time = min_price_point[0]
            
            savings = current_price - min_price
            savings_percentage = (savings / current_price) * 100 if current_price > 0 else 0
            
            if savings_percentage > 5:  # Significant savings available
                recommendation = "schedule_later"
                reason = f"Wait for lower prices at {min_price_time.strftime('%H:%M UTC')}"
            elif savings_percentage < -5:  # Prices will rise significantly
                recommendation = "execute_immediately"
                reason = "Prices expected to rise soon"
            else:
                recommendation = "execute_now"
                reason = "Minimal price variation expected"
            
            return {
                "current_price": current_price,
                "recommended_price": min_price,
                "recommended_time": min_price_time.isoformat(),
                "potential_savings": max(0, savings),
                "savings_percentage": max(0, savings_percentage),
                "recommendation": recommendation,
                "reason": reason,
                "market_condition": self.market_conditions.get(resource_type, MarketCondition.BALANCED).value,
                "optimal_windows": forecast.optimal_execution_windows
            }
            
        except Exception as e:
            logger.error("Error generating cost optimization recommendations", error=str(e))
            return {
                "current_price": self.base_prices.get(resource_type, 1.0),
                "recommendation": "execute_now",
                "reason": f"Error in analysis: {str(e)}",
                "potential_savings": 0.0
            }
    
    def get_arbitrage_opportunities(
        self,
        resource_type: Optional[ResourceType] = None,
        min_profit_percentage: float = 10.0
    ) -> List[ArbitrageOpportunity]:
        """Get current arbitrage opportunities"""
        opportunities = self.arbitrage_opportunities
        
        if resource_type:
            opportunities = [opp for opp in opportunities if opp.resource_type == resource_type]
        
        if min_profit_percentage > 0:
            opportunities = [opp for opp in opportunities if opp.profit_percentage >= min_profit_percentage]
        
        # Sort by profit percentage
        return sorted(opportunities, key=lambda x: x.profit_percentage, reverse=True)
    
    def get_pricing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pricing statistics"""
        total_data_points = sum(len(history) for history in self.price_history.values())
        
        price_volatility = {}
        for resource_type, history in self.price_history.items():
            if len(history) > 1:
                prices = [dp.actual_price for dp in history[-24:]]  # Last 24 hours
                if len(prices) > 1:
                    volatility = statistics.stdev(prices) / statistics.mean(prices)
                    price_volatility[resource_type.value] = volatility
        
        return {
            "pricing_statistics": dict(self.pricing_statistics),
            "total_price_data_points": total_data_points,
            "active_forecasts": len(self.active_forecasts),
            "arbitrage_opportunities": len(self.arbitrage_opportunities),
            "price_volatility": price_volatility,
            "market_conditions": {
                rt.value: condition.value 
                for rt, condition in self.market_conditions.items()
            },
            "revenue_optimization": self.revenue_optimization,
            "user_cost_savings": self.user_cost_savings
        }


# Global instance for easy access
_ftns_pricing_engine = None

def get_ftns_pricing_engine() -> FTNSPricingEngine:
    """Get global FTNS pricing engine instance"""
    global _ftns_pricing_engine
    if _ftns_pricing_engine is None:
        _ftns_pricing_engine = FTNSPricingEngine()
    return _ftns_pricing_engine
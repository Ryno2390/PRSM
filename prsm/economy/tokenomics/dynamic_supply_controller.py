"""
FTNS Dynamic Supply Controller
Implements asymptotic appreciation and dynamic supply adjustment

This module manages the FTNS token supply dynamics to achieve a stable,
predictable appreciation rate that starts high (50% annually) and asymptotically
approaches the target steady-state rate (2% annually). The controller actively
adjusts token distribution rates based on market price velocity and volatility.

Core Features:
- Asymptotic appreciation algorithm (50% → 2% over time)
- Price velocity tracking and analysis
- Volatility-dampened supply adjustments
- Governance-configurable economic parameters
- Comprehensive audit trails and transparency

Integration Points:
- Price Oracle: For real-time price data and historical analysis
- FTNS Service: For reward rate adjustments and distribution
- Database: For persistent storage of adjustments and metrics
- Governance System: For parameter configuration and oversight
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from math import exp, sqrt
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from prsm.economy.tokenomics.models import (
    FTNSSupplyAdjustment, FTNSPriceMetrics, FTNSRewardRates,
    SupplyAdjustmentStatus, AdjustmentTrigger
)

logger = structlog.get_logger(__name__)


@dataclass
class PriceMetrics:
    """Price metrics for supply adjustment calculations"""
    current_price: Decimal
    price_velocity: float  # Annualized appreciation rate
    volatility: float      # Annualized volatility
    volume_24h: Decimal
    market_cap: Decimal
    timestamp: datetime


@dataclass
class AdjustmentResult:
    """Result of supply adjustment calculation"""
    adjustment_factor: float
    target_rate: float
    actual_rate: float
    rate_ratio: float
    volatility: float
    volatility_damping: float
    adjustment_applied: bool
    reason: str
    metadata: Dict[str, Any]


class DynamicSupplyController:
    """
    Implements asymptotic appreciation and dynamic supply adjustment
    
    The DynamicSupplyController is responsible for:
    1. Calculating the current target appreciation rate (asymptotic decay)
    2. Measuring actual price velocity and volatility
    3. Computing optimal supply adjustment factors
    4. Applying adjustments to network reward rates
    5. Maintaining comprehensive audit trails
    """
    
    def __init__(self, db_session: AsyncSession, price_oracle=None, ftns_service=None):
        self.db = db_session
        self.oracle = price_oracle
        self.ftns = ftns_service
        self.adjustment_cache = {}
        
        # Economic parameters (configurable via governance)
        self.launch_date = datetime(2025, 1, 1, tzinfo=timezone.utc)  # Set actual launch date
        self.target_final_rate = 0.02            # 2% annual appreciation target
        self.initial_rate = 0.50                 # 50% initial annual rate
        self.decay_constant = 0.003              # Controls transition speed (configurable)
        
        # Adjustment parameters (governance configurable)
        self.max_daily_adjustment = 0.10         # Max 10% daily reward change
        self.price_velocity_window = 30          # Days for velocity calculation
        self.adjustment_cooldown_hours = 24      # Hours between adjustments
        self.volatility_threshold = 0.5          # High volatility threshold
        self.min_data_points = 7                 # Minimum price history required
        
        # Rate thresholds for adjustment triggers
        self.fast_appreciation_threshold = 1.5   # 50% above target
        self.moderate_appreciation_threshold = 1.2  # 20% above target
        self.slow_appreciation_threshold = 0.5   # 50% below target
        self.moderate_slow_threshold = 0.8       # 20% below target
    
    # === TARGET RATE CALCULATION ===
    
    async def calculate_target_appreciation_rate(self, as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate current target appreciation rate using asymptotic decay formula
        
        Formula: target_rate + (initial_rate - target_rate) * e^(-decay * days_since_launch)
        
        Args:
            as_of_date: Calculate rate as of specific date (defaults to now)
            
        Returns:
            float: Current target appreciation rate (annualized)
        """
        
        if as_of_date is None:
            as_of_date = datetime.now(timezone.utc)
        
        # Ensure timezone compatibility
        if as_of_date.tzinfo is None:
            as_of_date = as_of_date.replace(tzinfo=timezone.utc)
        
        days_since_launch = max(0, (as_of_date - self.launch_date).days)
        
        # Asymptotic formula: target + (initial - target) * e^(-decay * days)
        rate_difference = self.initial_rate - self.target_final_rate
        decay_factor = exp(-self.decay_constant * days_since_launch)
        current_rate = self.target_final_rate + (rate_difference * decay_factor)
        
        # Ensure rate doesn't go below target
        current_rate = max(self.target_final_rate, current_rate)
        
        await logger.adebug(
            "Target appreciation rate calculated",
            days_since_launch=days_since_launch,
            target_rate=current_rate,
            decay_factor=decay_factor,
            as_of_date=as_of_date.isoformat()
        )
        
        return current_rate
    
    async def get_days_to_target_rate(self, target_percentage: float = 0.95) -> int:
        """
        Calculate days until target rate is reached within specified percentage
        
        Args:
            target_percentage: How close to final rate (0.95 = 95% of the way there)
            
        Returns:
            int: Days until target is reached
        """
        
        # Solve: target_final + (initial - target_final) * e^(-decay * days) = target_final * (1 + tolerance)
        # Where tolerance allows for target_percentage approach
        
        rate_difference = self.initial_rate - self.target_final_rate
        target_excess = rate_difference * (1 - target_percentage)
        
        if target_excess <= 0:
            return 0  # Already at target
        
        # Solve for days: ln(target_excess / rate_difference) = -decay * days
        # days = -ln(target_excess / rate_difference) / decay
        
        import math
        days = -math.log(target_excess / rate_difference) / self.decay_constant
        
        return max(0, int(days))
    
    # === PRICE VELOCITY ANALYSIS ===
    
    async def calculate_price_velocity(self, days: Optional[int] = None, 
                                     end_date: Optional[datetime] = None) -> float:
        """
        Calculate recent price appreciation rate (annualized)
        
        Args:
            days: Number of days to analyze (defaults to price_velocity_window)
            end_date: End date for analysis (defaults to now)
            
        Returns:
            float: Annualized price appreciation rate
        """
        
        if days is None:
            days = self.price_velocity_window
        
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        start_date = end_date - timedelta(days=days)
        
        # Get price history from oracle
        if not self.oracle:
            await logger.awarning("Price oracle not available, using fallback velocity")
            return 0.0
        
        try:
            price_history = await self.oracle.get_price_history(
                start_date=start_date,
                end_date=end_date,
                interval="daily"
            )
            
            if len(price_history) < 2:
                await logger.awarning("Insufficient price history for velocity calculation")
                return 0.0
            
            # Calculate price change
            start_price = float(price_history[0].price)
            end_price = float(price_history[-1].price)
            
            if start_price <= 0:
                await logger.awarning("Invalid start price for velocity calculation")
                return 0.0
            
            # Annualized rate calculation
            price_change_ratio = (end_price - start_price) / start_price
            annualized_rate = price_change_ratio * (365.0 / days)
            
            await logger.adebug(
                "Price velocity calculated",
                days=days,
                start_price=start_price,
                end_price=end_price,
                price_change_ratio=price_change_ratio,
                annualized_rate=annualized_rate
            )
            
            return annualized_rate
            
        except Exception as e:
            await logger.aerror("Failed to calculate price velocity", error=str(e))
            return 0.0
    
    async def calculate_price_volatility(self, days: int = 7, 
                                       end_date: Optional[datetime] = None) -> float:
        """
        Calculate recent price volatility (annualized standard deviation)
        
        Args:
            days: Number of days to analyze
            end_date: End date for analysis (defaults to now)
            
        Returns:
            float: Annualized volatility (standard deviation of returns)
        """
        
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        start_date = end_date - timedelta(days=days)
        
        if not self.oracle:
            await logger.awarning("Price oracle not available, using fallback volatility")
            return 0.0
        
        try:
            price_history = await self.oracle.get_price_history(
                start_date=start_date,
                end_date=end_date,
                interval="daily"
            )
            
            if len(price_history) < 2:
                return 0.0
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(price_history)):
                prev_price = float(price_history[i-1].price)
                curr_price = float(price_history[i].price)
                
                if prev_price > 0:
                    daily_return = (curr_price - prev_price) / prev_price
                    returns.append(daily_return)
            
            if len(returns) < 2:
                return 0.0
            
            # Calculate standard deviation
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            daily_volatility = sqrt(variance)
            
            # Annualize volatility (sqrt of time scaling)
            annualized_volatility = daily_volatility * sqrt(365)
            
            await logger.adebug(
                "Price volatility calculated",
                days=days,
                num_returns=len(returns),
                mean_return=mean_return,
                daily_volatility=daily_volatility,
                annualized_volatility=annualized_volatility
            )
            
            return annualized_volatility
            
        except Exception as e:
            await logger.aerror("Failed to calculate price volatility", error=str(e))
            return 0.0
    
    async def get_current_price_metrics(self) -> Optional[PriceMetrics]:
        """Get current comprehensive price metrics"""
        
        if not self.oracle:
            return None
        
        try:
            current_data = await self.oracle.get_current_price_data()
            
            # Calculate derived metrics
            velocity = await self.calculate_price_velocity()
            volatility = await self.calculate_price_volatility()
            
            return PriceMetrics(
                current_price=current_data.price,
                price_velocity=velocity,
                volatility=volatility,
                volume_24h=current_data.volume_24h,
                market_cap=current_data.market_cap,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            await logger.aerror("Failed to get current price metrics", error=str(e))
            return None
    
    # === SUPPLY ADJUSTMENT LOGIC ===
    
    async def calculate_adjustment_factor(self) -> AdjustmentResult:
        """
        Calculate optimal adjustment factor for all reward rates
        
        Returns:
            AdjustmentResult: Comprehensive adjustment calculation result
        """
        
        target_rate = await self.calculate_target_appreciation_rate()
        actual_rate = await self.calculate_price_velocity()
        volatility = await self.calculate_price_volatility()
        
        # Rate ratio (actual vs target)
        rate_ratio = actual_rate / target_rate if target_rate > 0 else 1.0
        
        # Volatility damping (reduce adjustments during high volatility)
        volatility_damping = max(0.3, 1.0 - min(volatility * 2, 0.7))  # 30-100% damping
        
        # Base adjustment calculation
        base_adjustment = 1.0
        reason = "in_target_range"
        
        if rate_ratio > self.fast_appreciation_threshold:  # Price appreciating too fast
            base_adjustment = 1.4  # Increase supply significantly
            reason = "fast_appreciation"
        elif rate_ratio > self.moderate_appreciation_threshold:  # Moderately fast
            base_adjustment = 1.2  # Moderate supply increase
            reason = "moderate_fast_appreciation"
        elif rate_ratio < self.slow_appreciation_threshold:  # Price appreciating too slowly
            base_adjustment = 0.6  # Decrease supply significantly
            reason = "slow_appreciation"
        elif rate_ratio < self.moderate_slow_threshold:  # Moderately slow
            base_adjustment = 0.8  # Moderate supply decrease
            reason = "moderate_slow_appreciation"
        
        # Apply volatility damping
        if base_adjustment != 1.0:
            adjustment_magnitude = abs(base_adjustment - 1.0)
            damped_magnitude = adjustment_magnitude * volatility_damping
            adjustment_factor = 1.0 + damped_magnitude if base_adjustment > 1.0 else 1.0 - damped_magnitude
        else:
            adjustment_factor = 1.0
        
        # Ensure adjustment is within daily limits
        max_adjustment = 1.0 + self.max_daily_adjustment
        min_adjustment = 1.0 - self.max_daily_adjustment
        
        final_adjustment = max(min_adjustment, min(max_adjustment, adjustment_factor))
        
        # Determine if adjustment should be applied
        adjustment_applied = abs(final_adjustment - 1.0) > 0.01  # Only apply if >1% change
        
        if not adjustment_applied:
            reason = "adjustment_too_small"
        
        await logger.ainfo(
            "Supply adjustment calculated",
            target_rate=target_rate,
            actual_rate=actual_rate,
            rate_ratio=rate_ratio,
            volatility=volatility,
            volatility_damping=volatility_damping,
            base_adjustment=base_adjustment,
            final_adjustment=final_adjustment,
            adjustment_applied=adjustment_applied,
            reason=reason
        )
        
        return AdjustmentResult(
            adjustment_factor=final_adjustment,
            target_rate=target_rate,
            actual_rate=actual_rate,
            rate_ratio=rate_ratio,
            volatility=volatility,
            volatility_damping=volatility_damping,
            adjustment_applied=adjustment_applied,
            reason=reason,
            metadata={
                "base_adjustment": base_adjustment,
                "adjustment_magnitude": abs(final_adjustment - 1.0),
                "velocity_window_days": self.price_velocity_window,
                "volatility_window_days": 7
            }
        )
    
    async def apply_network_reward_adjustment(self, adjustment_factor: float, 
                                            force: bool = False) -> Dict[str, Any]:
        """
        Apply adjustment factor to all network reward rates
        
        Args:
            adjustment_factor: Multiplier for reward rates (1.0 = no change)
            force: Override cooldown period if True
            
        Returns:
            Dict containing application results and metadata
        """
        
        # Check cooldown period
        if not force:
            last_adjustment = await self._get_last_adjustment_time()
            if last_adjustment:
                hours_since_adjustment = (datetime.now(timezone.utc) - last_adjustment).total_seconds() / 3600
                if hours_since_adjustment < self.adjustment_cooldown_hours:
                    return {
                        "status": "skipped",
                        "reason": "cooldown_period",
                        "hours_until_next": self.adjustment_cooldown_hours - hours_since_adjustment
                    }
        
        # Get current reward rates
        current_rates = await self._get_current_reward_rates()
        
        # Apply adjustment to all rates
        new_rates = {}
        for rate_type, current_value in current_rates.items():
            new_value = current_value * adjustment_factor
            new_rates[rate_type] = new_value
        
        # Update rates in database
        await self._update_reward_rates(new_rates)
        
        # Record adjustment for audit trail
        adjustment_record = await self._record_adjustment(
            adjustment_factor, current_rates, new_rates
        )
        
        await logger.ainfo(
            "Network reward adjustment applied",
            adjustment_factor=adjustment_factor,
            adjustment_id=adjustment_record.adjustment_id,
            rates_changed=len(new_rates)
        )
        
        return {
            "status": "applied",
            "adjustment_id": str(adjustment_record.adjustment_id),
            "adjustment_factor": adjustment_factor,
            "previous_rates": current_rates,
            "new_rates": new_rates,
            "effective_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def execute_daily_adjustment(self) -> Dict[str, Any]:
        """
        Execute daily supply adjustment routine
        
        This is the main entry point for automated daily adjustments.
        Should be called by scheduled task or governance trigger.
        
        Returns:
            Dict containing execution results and metrics
        """
        
        try:
            # Calculate adjustment
            adjustment_result = await self.calculate_adjustment_factor()
            
            if not adjustment_result.adjustment_applied:
                return {
                    "status": "no_adjustment",
                    "reason": adjustment_result.reason,
                    "metrics": {
                        "target_rate": adjustment_result.target_rate,
                        "actual_rate": adjustment_result.actual_rate,
                        "rate_ratio": adjustment_result.rate_ratio,
                        "volatility": adjustment_result.volatility
                    }
                }
            
            # Apply adjustment
            application_result = await self.apply_network_reward_adjustment(
                adjustment_result.adjustment_factor
            )
            
            if application_result["status"] == "skipped":
                return {
                    "status": "skipped",
                    "reason": application_result["reason"],
                    "adjustment_calculated": adjustment_result.adjustment_factor
                }
            
            # Store metrics
            await self._store_price_metrics(adjustment_result)
            
            return {
                "status": "success",
                "adjustment_applied": adjustment_result.adjustment_factor,
                "adjustment_id": application_result["adjustment_id"],
                "metrics": {
                    "target_rate": adjustment_result.target_rate,
                    "actual_rate": adjustment_result.actual_rate,
                    "rate_ratio": adjustment_result.rate_ratio,
                    "volatility": adjustment_result.volatility,
                    "volatility_damping": adjustment_result.volatility_damping
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            await logger.aerror("Daily adjustment execution failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    # === RATE MANAGEMENT ===
    
    async def _get_current_reward_rates(self) -> Dict[str, float]:
        """Get all current network reward rates from database"""
        
        try:
            # Query latest reward rates from database
            result = await self.db.execute(
                select(FTNSRewardRates)
                .where(FTNSRewardRates.active == True)
                .order_by(desc(FTNSRewardRates.effective_date))
                .limit(1)
            )
            
            rate_record = result.scalar_one_or_none()
            
            if rate_record:
                return {
                    "context_cost_multiplier": float(rate_record.context_cost_multiplier),
                    "storage_reward_per_gb_hour": float(rate_record.storage_reward_per_gb_hour),
                    "compute_reward_per_unit": float(rate_record.compute_reward_per_unit),
                    "data_contribution_base": float(rate_record.data_contribution_base),
                    "governance_participation": float(rate_record.governance_participation),
                    "documentation_reward": float(rate_record.documentation_reward),
                    "staking_apy": float(rate_record.staking_apy),
                    "burn_rate_multiplier": float(rate_record.burn_rate_multiplier)
                }
            else:
                # Return default rates if no database record
                return self._get_default_reward_rates()
                
        except Exception as e:
            await logger.aerror("Failed to get current reward rates", error=str(e))
            return self._get_default_reward_rates()
    
    def _get_default_reward_rates(self) -> Dict[str, float]:
        """Get default reward rates (fallback)"""
        
        return {
            "context_cost_multiplier": 1.0,
            "storage_reward_per_gb_hour": 0.01,
            "compute_reward_per_unit": 0.05,
            "data_contribution_base": 10.0,
            "governance_participation": 2.0,
            "documentation_reward": 5.0,
            "staking_apy": 0.08,  # 8% annual
            "burn_rate_multiplier": 1.0
        }
    
    async def _update_reward_rates(self, new_rates: Dict[str, float]):
        """Update reward rates in database"""
        
        try:
            # Deactivate current rates
            await self.db.execute(
                update(FTNSRewardRates)
                .where(FTNSRewardRates.active == True)
                .values(active=False, deactivated_at=datetime.now(timezone.utc))
            )
            
            # Create new rate record
            new_rate_record = FTNSRewardRates(
                context_cost_multiplier=Decimal(str(new_rates["context_cost_multiplier"])),
                storage_reward_per_gb_hour=Decimal(str(new_rates["storage_reward_per_gb_hour"])),
                compute_reward_per_unit=Decimal(str(new_rates["compute_reward_per_unit"])),
                data_contribution_base=Decimal(str(new_rates["data_contribution_base"])),
                governance_participation=Decimal(str(new_rates["governance_participation"])),
                documentation_reward=Decimal(str(new_rates["documentation_reward"])),
                staking_apy=Decimal(str(new_rates["staking_apy"])),
                burn_rate_multiplier=Decimal(str(new_rates["burn_rate_multiplier"])),
                effective_date=datetime.now(timezone.utc),
                active=True
            )
            
            self.db.add(new_rate_record)
            await self.db.commit()
            
        except Exception as e:
            await logger.aerror("Failed to update reward rates", error=str(e))
            await self.db.rollback()
            raise
    
    async def _get_last_adjustment_time(self) -> Optional[datetime]:
        """Get timestamp of last supply adjustment"""
        
        try:
            result = await self.db.execute(
                select(FTNSSupplyAdjustment.applied_at)
                .where(FTNSSupplyAdjustment.status == SupplyAdjustmentStatus.APPLIED.value)
                .order_by(desc(FTNSSupplyAdjustment.applied_at))
                .limit(1)
            )
            
            return result.scalar_one_or_none()
            
        except Exception as e:
            await logger.aerror("Failed to get last adjustment time", error=str(e))
            return None
    
    async def _record_adjustment(self, adjustment_factor: float, 
                               previous_rates: Dict[str, float], 
                               new_rates: Dict[str, float]) -> FTNSSupplyAdjustment:
        """Record supply adjustment in database for audit trail"""
        
        try:
            adjustment_record = FTNSSupplyAdjustment(
                adjustment_factor=Decimal(str(adjustment_factor)),
                trigger=AdjustmentTrigger.AUTOMATED.value,
                previous_rates=previous_rates,
                new_rates=new_rates,
                calculated_at=datetime.now(timezone.utc),
                applied_at=datetime.now(timezone.utc),
                status=SupplyAdjustmentStatus.APPLIED.value,
                metadata={
                    "controller_version": "1.0",
                    "calculation_method": "asymptotic_velocity_adjustment"
                }
            )
            
            self.db.add(adjustment_record)
            await self.db.commit()
            await self.db.refresh(adjustment_record)
            
            return adjustment_record
            
        except Exception as e:
            await logger.aerror("Failed to record adjustment", error=str(e))
            await self.db.rollback()
            raise
    
    async def _store_price_metrics(self, adjustment_result: AdjustmentResult):
        """Store price metrics for historical analysis"""
        
        try:
            metrics_record = FTNSPriceMetrics(
                target_appreciation_rate=Decimal(str(adjustment_result.target_rate)),
                actual_appreciation_rate=Decimal(str(adjustment_result.actual_rate)),
                price_volatility=Decimal(str(adjustment_result.volatility)),
                rate_ratio=Decimal(str(adjustment_result.rate_ratio)),
                volatility_damping=Decimal(str(adjustment_result.volatility_damping)),
                recorded_at=datetime.now(timezone.utc),
                metadata=adjustment_result.metadata
            )
            
            self.db.add(metrics_record)
            await self.db.commit()
            
        except Exception as e:
            await logger.aerror("Failed to store price metrics", error=str(e))
            # Don't rollback here as this is supplementary data
    
    # === UTILITY METHODS ===
    
    async def get_adjustment_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent adjustment history for analysis"""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        try:
            result = await self.db.execute(
                select(FTNSSupplyAdjustment)
                .where(FTNSSupplyAdjustment.applied_at >= cutoff_date)
                .order_by(desc(FTNSSupplyAdjustment.applied_at))
            )
            
            adjustments = result.scalars().all()
            
            return [
                {
                    "adjustment_id": str(adj.adjustment_id),
                    "adjustment_factor": float(adj.adjustment_factor),
                    "trigger": adj.trigger,
                    "applied_at": adj.applied_at.isoformat(),
                    "status": adj.status,
                    "previous_rates": adj.previous_rates,
                    "new_rates": adj.new_rates
                }
                for adj in adjustments
            ]
            
        except Exception as e:
            await logger.aerror("Failed to get adjustment history", error=str(e))
            return []
    
    async def get_controller_status(self) -> Dict[str, Any]:
        """Get current controller status and configuration"""
        
        target_rate = await self.calculate_target_appreciation_rate()
        days_to_target = await self.get_days_to_target_rate()
        last_adjustment = await self._get_last_adjustment_time()
        
        return {
            "controller_active": True,
            "current_target_rate": target_rate,
            "days_since_launch": (datetime.now(timezone.utc) - self.launch_date).days,
            "days_to_target_rate": days_to_target,
            "last_adjustment": last_adjustment.isoformat() if last_adjustment else None,
            "configuration": {
                "initial_rate": self.initial_rate,
                "target_final_rate": self.target_final_rate,
                "decay_constant": self.decay_constant,
                "max_daily_adjustment": self.max_daily_adjustment,
                "adjustment_cooldown_hours": self.adjustment_cooldown_hours,
                "price_velocity_window": self.price_velocity_window
            }
        }
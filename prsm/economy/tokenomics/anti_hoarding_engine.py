"""
FTNS Anti-Hoarding Engine
Implements velocity incentives and demurrage to encourage circulation

This module implements sophisticated anti-hoarding mechanisms to ensure FTNS
tokens circulate actively rather than being held indefinitely. The system uses
demurrage fees (gradual value decay for inactive tokens) and velocity tracking
to incentivize productive token usage while discouraging speculation.

Core Features:
- Token velocity tracking at user and network levels
- Dynamic demurrage rate calculation based on circulation patterns
- Contributor status integration for fee adjustments
- Grace periods for new users and active contributors
- Automated daily demurrage collection with comprehensive logging

Economic Design:
- Target monthly velocity: 1.2x (tokens should circulate 1.2x per month)
- Base demurrage rate: 0.2% monthly (for users meeting velocity targets)
- Maximum demurrage rate: 1.0% monthly (for inactive holders)
- Contributor discounts: Up to 50% reduction for power users
- Grace period: 90 days for new users to establish circulation patterns
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from prsm.economy.tokenomics.models import (
    FTNSVelocityMetrics, FTNSDemurrageRecord, FTNSAntiHoardingConfig,
    VelocityCategory, DemurrageStatus, ContributorTier
)

logger = structlog.get_logger(__name__)


@dataclass
class VelocityMetrics:
    """User velocity analysis result"""
    user_id: str
    velocity: float  # Monthly velocity ratio
    transaction_volume: Decimal
    current_balance: Decimal
    velocity_category: str
    calculation_period_days: int
    calculated_at: datetime


@dataclass
class DemurrageCalculation:
    """Demurrage fee calculation result"""
    user_id: str
    monthly_rate: float
    daily_rate: float
    fee_amount: Decimal
    current_balance: Decimal
    velocity: float
    contributor_status: str
    grace_period_active: bool
    applicable: bool
    reason: str


class AntiHoardingEngine:
    """
    Implements velocity incentives and demurrage to encourage circulation
    
    The AntiHoardingEngine is responsible for:
    1. Tracking token velocity at user and network levels
    2. Calculating appropriate demurrage rates based on circulation patterns
    3. Applying daily demurrage fees to discourage hoarding
    4. Integrating with contributor status for fee adjustments
    5. Maintaining comprehensive audit trails for transparency
    """
    
    def __init__(self, db_session: AsyncSession, ftns_service=None, contributor_manager=None):
        self.db = db_session
        self.ftns = ftns_service
        self.contributor_manager = contributor_manager
        
        # Demurrage parameters (governance configurable)
        self.target_velocity = 1.2          # Monthly velocity target
        self.base_demurrage_rate = 0.002    # 0.2% monthly base rate
        self.max_demurrage_rate = 0.01      # 1.0% monthly maximum
        self.velocity_calculation_days = 30  # Period for velocity calculation
        self.grace_period_days = 90         # Grace period for new users
        self.min_fee_threshold = Decimal("0.001")  # Minimum fee to charge
        
        # Velocity thresholds for demurrage calculation
        self.high_velocity_threshold = 1.0    # >= target velocity
        self.moderate_velocity_threshold = 0.7  # >= 70% of target
        self.low_velocity_threshold = 0.3      # >= 30% of target
        
        # Contributor status modifiers
        self.status_modifiers = {
            ContributorTier.NONE.value: 1.5,         # Higher demurrage for non-contributors
            ContributorTier.BASIC.value: 1.0,        # Standard rate
            ContributorTier.ACTIVE.value: 0.7,       # 30% reduction
            ContributorTier.POWER_USER.value: 0.5    # 50% reduction
        }
    
    # === VELOCITY CALCULATIONS ===
    
    async def calculate_user_velocity(self, user_id: str, days: Optional[int] = None) -> VelocityMetrics:
        """
        Calculate token velocity for a specific user
        
        Velocity is calculated as the ratio of transaction volume to average balance
        over the specified time period, normalized to monthly terms.
        
        Args:
            user_id: User identifier
            days: Period for calculation (defaults to velocity_calculation_days)
            
        Returns:
            VelocityMetrics: Comprehensive velocity analysis
        """
        
        if days is None:
            days = self.velocity_calculation_days
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Get user's transaction history
        transactions = await self._get_user_transactions(user_id, start_date, end_date)
        current_balance = await self._get_user_balance(user_id)
        
        if current_balance <= 0:
            return VelocityMetrics(
                user_id=user_id,
                velocity=0.0,
                transaction_volume=Decimal('0'),
                current_balance=current_balance,
                velocity_category=VelocityCategory.INACTIVE.value,
                calculation_period_days=days,
                calculated_at=end_date
            )
        
        # Calculate total transaction volume (excluding demurrage and system transactions)
        outgoing_volume = Decimal('0')
        incoming_volume = Decimal('0')
        
        for tx in transactions:
            if tx.transaction_type in ['demurrage', 'system_adjustment']:
                continue  # Exclude system transactions from velocity calculation
            
            if tx.from_user_id == user_id:
                outgoing_volume += tx.amount
            elif tx.to_user_id == user_id:
                incoming_volume += tx.amount
        
        total_volume = outgoing_volume + incoming_volume
        
        # Calculate velocity (normalized to monthly rate)
        # Velocity = (transaction_volume / average_balance) * (30 / days)
        monthly_velocity = float(total_volume / current_balance) * (30.0 / days)
        
        # Categorize velocity
        velocity_category = self._categorize_velocity(monthly_velocity)
        
        await logger.adebug(
            "User velocity calculated",
            user_id=user_id,
            velocity=monthly_velocity,
            transaction_volume=float(total_volume),
            current_balance=float(current_balance),
            velocity_category=velocity_category,
            days=days
        )
        
        return VelocityMetrics(
            user_id=user_id,
            velocity=monthly_velocity,
            transaction_volume=total_volume,
            current_balance=current_balance,
            velocity_category=velocity_category,
            calculation_period_days=days,
            calculated_at=end_date
        )
    
    async def calculate_network_velocity(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate overall network token velocity
        
        Returns weighted average velocity across all active users,
        plus distribution statistics and health metrics.
        
        Args:
            days: Period for calculation
            
        Returns:
            Dict containing network velocity metrics
        """
        
        if days is None:
            days = self.velocity_calculation_days
        
        # Get all users with balances
        active_users = await self._get_users_with_balances(min_balance=Decimal('0.001'))
        
        if not active_users:
            return {
                "network_velocity": 0.0,
                "total_users": 0,
                "total_balance": 0.0,
                "velocity_distribution": {},
                "health_score": 0.0
            }
        
        total_weighted_velocity = 0.0
        total_balance = Decimal('0')
        velocity_distribution = {
            VelocityCategory.HIGH.value: 0,
            VelocityCategory.MODERATE.value: 0,
            VelocityCategory.LOW.value: 0,
            VelocityCategory.INACTIVE.value: 0
        }
        
        for user_id in active_users:
            user_metrics = await self.calculate_user_velocity(user_id, days)
            user_balance = user_metrics.current_balance
            
            # Weight velocity by balance size
            total_weighted_velocity += user_metrics.velocity * float(user_balance)
            total_balance += user_balance
            
            # Track velocity distribution
            velocity_distribution[user_metrics.velocity_category] += 1
        
        # Calculate network-wide metrics
        network_velocity = total_weighted_velocity / float(total_balance) if total_balance > 0 else 0.0
        
        # Calculate health score (higher is better)
        # Based on percentage of users in high/moderate velocity categories
        total_users = len(active_users)
        healthy_users = (velocity_distribution[VelocityCategory.HIGH.value] + 
                        velocity_distribution[VelocityCategory.MODERATE.value])
        health_score = healthy_users / total_users if total_users > 0 else 0.0
        
        await logger.ainfo(
            "Network velocity calculated",
            network_velocity=network_velocity,
            total_users=total_users,
            total_balance=float(total_balance),
            health_score=health_score
        )
        
        return {
            "network_velocity": network_velocity,
            "total_users": total_users,
            "total_balance": float(total_balance),
            "velocity_distribution": velocity_distribution,
            "health_score": health_score,
            "target_velocity": self.target_velocity,
            "calculation_period_days": days,
            "calculated_at": datetime.now(timezone.utc).isoformat()
        }
    
    # === DEMURRAGE CALCULATION ===
    
    async def calculate_demurrage_rate(self, user_id: str) -> DemurrageCalculation:
        """
        Calculate appropriate demurrage rate for a user
        
        Demurrage rates are based on:
        1. Token velocity (higher velocity = lower demurrage)
        2. Contributor status (active contributors get discounts)
        3. Grace periods (new users exempt for initial period)
        4. Minimum thresholds (avoid charging tiny amounts)
        
        Args:
            user_id: User identifier
            
        Returns:
            DemurrageCalculation: Complete demurrage analysis
        """
        
        # Check grace period for new users
        user_creation_date = await self._get_user_creation_date(user_id)
        grace_period_active = False
        
        if user_creation_date:
            days_since_creation = (datetime.now(timezone.utc) - user_creation_date).days
            grace_period_active = days_since_creation < self.grace_period_days
        
        # Get current user state
        user_metrics = await self.calculate_user_velocity(user_id)
        current_balance = user_metrics.current_balance
        
        if current_balance <= 0:
            return DemurrageCalculation(
                user_id=user_id,
                monthly_rate=0.0,
                daily_rate=0.0,
                fee_amount=Decimal('0'),
                current_balance=current_balance,
                velocity=user_metrics.velocity,
                contributor_status="none",
                grace_period_active=grace_period_active,
                applicable=False,
                reason="zero_balance"
            )
        
        if grace_period_active:
            return DemurrageCalculation(
                user_id=user_id,
                monthly_rate=0.0,
                daily_rate=0.0,
                fee_amount=Decimal('0'),
                current_balance=current_balance,
                velocity=user_metrics.velocity,
                contributor_status="grace_period",
                grace_period_active=True,
                applicable=False,
                reason="grace_period"
            )
        
        # Get contributor status
        contributor_status = "none"
        status_modifier = self.status_modifiers[ContributorTier.NONE.value]
        
        if self.contributor_manager:
            try:
                status_record = await self.contributor_manager.get_contributor_status(user_id)
                if status_record:
                    contributor_status = status_record.status
                    status_modifier = self.status_modifiers.get(contributor_status, 1.0)
            except Exception as e:
                await logger.awarning("Failed to get contributor status", user_id=user_id, error=str(e))
        
        # Base demurrage calculation based on velocity
        user_velocity = user_metrics.velocity
        velocity_ratio = user_velocity / self.target_velocity
        
        if velocity_ratio >= self.high_velocity_threshold:
            # Good velocity = minimal demurrage
            velocity_based_rate = self.base_demurrage_rate * 0.5
        elif velocity_ratio >= self.moderate_velocity_threshold:
            # Moderate velocity = base rate
            velocity_based_rate = self.base_demurrage_rate
        elif velocity_ratio >= self.low_velocity_threshold:
            # Low velocity = increased rate
            velocity_based_rate = self.base_demurrage_rate * 2.0
        else:
            # Very low velocity = maximum rate
            velocity_based_rate = self.max_demurrage_rate
        
        # Apply contributor status modifier
        final_monthly_rate = velocity_based_rate * status_modifier
        
        # Cap at maximum rate
        final_monthly_rate = min(final_monthly_rate, self.max_demurrage_rate)
        
        # Calculate daily rate and fee amount
        daily_rate = final_monthly_rate / 30.0
        fee_amount = current_balance * Decimal(str(daily_rate))
        
        # Check minimum fee threshold
        applicable = fee_amount >= self.min_fee_threshold
        reason = "applicable" if applicable else "below_minimum_threshold"
        
        return DemurrageCalculation(
            user_id=user_id,
            monthly_rate=final_monthly_rate,
            daily_rate=daily_rate,
            fee_amount=fee_amount,
            current_balance=current_balance,
            velocity=user_velocity,
            contributor_status=contributor_status,
            grace_period_active=False,
            applicable=applicable,
            reason=reason
        )
    
    async def apply_demurrage_fees(self, user_id: Optional[str] = None, 
                                  dry_run: bool = False) -> Dict[str, Any]:
        """
        Apply demurrage fees to user(s)
        
        Args:
            user_id: Specific user to process (None for all users)
            dry_run: Calculate fees without applying them
            
        Returns:
            Dict containing processing results and statistics
        """
        
        if user_id:
            users_to_process = [user_id]
        else:
            # Process all users with positive balances
            users_to_process = await self._get_users_with_balances(min_balance=self.min_fee_threshold)
        
        results = []
        total_fees_collected = Decimal('0')
        
        for uid in users_to_process:
            try:
                result = await self._apply_user_demurrage(uid, dry_run)
                results.append(result)
                
                if result.get("status") == "applied":
                    total_fees_collected += result.get("fee_amount", Decimal('0'))
                    
            except Exception as e:
                await logger.aerror("Demurrage application failed", user_id=uid, error=str(e))
                results.append({
                    "user_id": uid,
                    "status": "error",
                    "error": str(e),
                    "fee_amount": Decimal('0')
                })
        
        # Store network-level demurrage metrics
        if not dry_run:
            await self._store_demurrage_metrics(results, total_fees_collected)
        
        success_count = sum(1 for r in results if r.get("status") == "applied")
        skip_count = sum(1 for r in results if r.get("status") == "skipped")
        error_count = sum(1 for r in results if r.get("status") == "error")
        
        await logger.ainfo(
            "Demurrage fees processed",
            dry_run=dry_run,
            total_users=len(results),
            fees_applied=success_count,
            skipped=skip_count,
            errors=error_count,
            total_fees_collected=float(total_fees_collected)
        )
        
        return {
            "status": "completed",
            "dry_run": dry_run,
            "processed_users": len(results),
            "fees_applied": success_count,
            "skipped": skip_count,
            "errors": error_count,
            "total_fees_collected": float(total_fees_collected),
            "results": results,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def _apply_user_demurrage(self, user_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """Apply demurrage fee to a specific user"""
        
        # Calculate demurrage
        calculation = await self.calculate_demurrage_rate(user_id)
        
        if not calculation.applicable:
            return {
                "user_id": user_id,
                "status": "skipped",
                "reason": calculation.reason,
                "fee_amount": Decimal('0'),
                "balance_before": calculation.current_balance,
                "balance_after": calculation.current_balance
            }
        
        if dry_run:
            return {
                "user_id": user_id,
                "status": "calculated",
                "reason": "dry_run",
                "fee_amount": calculation.fee_amount,
                "balance_before": calculation.current_balance,
                "balance_after": calculation.current_balance - calculation.fee_amount,
                "monthly_rate": calculation.monthly_rate,
                "daily_rate": calculation.daily_rate
            }
        
        # Apply the fee
        if not self.ftns:
            raise ValueError("FTNS service not available for demurrage application")
        
        try:
            # Deduct fee from balance
            success = await self.ftns.debit_balance(
                user_id=user_id,
                amount=calculation.fee_amount,
                description=f"Daily demurrage fee ({calculation.daily_rate:.4%})",
                transaction_type="demurrage"
            )
            
            if not success:
                return {
                    "user_id": user_id,
                    "status": "error",
                    "reason": "debit_failed",
                    "fee_amount": Decimal('0')
                }
            
            # Record demurrage transaction
            await self._record_demurrage_transaction(user_id, calculation)
            
            new_balance = calculation.current_balance - calculation.fee_amount
            
            return {
                "user_id": user_id,
                "status": "applied",
                "reason": "demurrage_collected",
                "fee_amount": calculation.fee_amount,
                "balance_before": calculation.current_balance,
                "balance_after": new_balance,
                "monthly_rate": calculation.monthly_rate,
                "daily_rate": calculation.daily_rate,
                "velocity": calculation.velocity,
                "contributor_status": calculation.contributor_status
            }
            
        except Exception as e:
            await logger.aerror("Failed to apply demurrage", user_id=user_id, error=str(e))
            return {
                "user_id": user_id,
                "status": "error",
                "reason": "application_failed",
                "error": str(e),
                "fee_amount": Decimal('0')
            }
    
    # === REPORTING AND ANALYTICS ===
    
    async def get_velocity_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive velocity analysis report"""
        
        network_metrics = await self.calculate_network_velocity(days)
        
        # Get velocity distribution details
        velocity_ranges = {
            "high_velocity": [],      # >= target
            "moderate_velocity": [],  # 70% - 100% of target
            "low_velocity": [],       # 30% - 70% of target
            "inactive": []            # < 30% of target
        }
        
        users_with_balances = await self._get_users_with_balances(min_balance=Decimal('0.001'))
        
        for user_id in users_with_balances[:100]:  # Sample first 100 for performance
            metrics = await self.calculate_user_velocity(user_id, days)
            
            velocity_ratio = metrics.velocity / self.target_velocity
            
            if velocity_ratio >= 1.0:
                velocity_ranges["high_velocity"].append({
                    "user_id": user_id,
                    "velocity": metrics.velocity,
                    "balance": float(metrics.current_balance)
                })
            elif velocity_ratio >= 0.7:
                velocity_ranges["moderate_velocity"].append({
                    "user_id": user_id,
                    "velocity": metrics.velocity,
                    "balance": float(metrics.current_balance)
                })
            elif velocity_ratio >= 0.3:
                velocity_ranges["low_velocity"].append({
                    "user_id": user_id,
                    "velocity": metrics.velocity,
                    "balance": float(metrics.current_balance)
                })
            else:
                velocity_ranges["inactive"].append({
                    "user_id": user_id,
                    "velocity": metrics.velocity,
                    "balance": float(metrics.current_balance)
                })
        
        return {
            "network_metrics": network_metrics,
            "velocity_ranges": velocity_ranges,
            "recommendations": self._generate_velocity_recommendations(network_metrics),
            "report_generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_demurrage_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate demurrage analysis and projections"""
        
        # Get recent demurrage history
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        demurrage_records = await self._get_demurrage_history(start_date, end_date)
        
        # Calculate totals and trends
        total_fees_collected = sum(float(record.fee_amount) for record in demurrage_records)
        unique_users_charged = len(set(record.user_id for record in demurrage_records))
        
        # Calculate current potential fees (dry run)
        dry_run_result = await self.apply_demurrage_fees(dry_run=True)
        
        return {
            "historical_data": {
                "period_days": days,
                "total_fees_collected": total_fees_collected,
                "unique_users_charged": unique_users_charged,
                "average_daily_fees": total_fees_collected / days,
                "records_count": len(demurrage_records)
            },
            "current_projections": {
                "potential_daily_fees": dry_run_result.get("total_fees_collected", 0),
                "users_subject_to_demurrage": dry_run_result.get("fees_applied", 0),
                "users_in_grace_period": dry_run_result.get("skipped", 0)
            },
            "recommendations": self._generate_demurrage_recommendations(dry_run_result),
            "report_generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    # === UTILITY METHODS ===
    
    def _categorize_velocity(self, velocity: float) -> str:
        """Categorize velocity into standard ranges"""
        
        velocity_ratio = velocity / self.target_velocity
        
        if velocity_ratio >= 1.0:
            return VelocityCategory.HIGH.value
        elif velocity_ratio >= 0.7:
            return VelocityCategory.MODERATE.value
        elif velocity_ratio >= 0.3:
            return VelocityCategory.LOW.value
        else:
            return VelocityCategory.INACTIVE.value
    
    def _generate_velocity_recommendations(self, network_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on velocity metrics"""
        
        recommendations = []
        network_velocity = network_metrics.get("network_velocity", 0)
        health_score = network_metrics.get("health_score", 0)
        
        if network_velocity < self.target_velocity * 0.8:
            recommendations.append("Network velocity below target - consider incentives for circulation")
        
        if health_score < 0.6:
            recommendations.append("High proportion of inactive users - review demurrage parameters")
        
        distribution = network_metrics.get("velocity_distribution", {})
        inactive_count = distribution.get(VelocityCategory.INACTIVE.value, 0)
        total_users = network_metrics.get("total_users", 1)
        
        if inactive_count / total_users > 0.3:
            recommendations.append("Consider educational outreach to encourage token usage")
        
        return recommendations
    
    def _generate_demurrage_recommendations(self, dry_run_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on demurrage analysis"""
        
        recommendations = []
        
        fees_applied = dry_run_result.get("fees_applied", 0)
        total_users = dry_run_result.get("processed_users", 1)
        
        if fees_applied / total_users > 0.5:
            recommendations.append("High demurrage applicability - consider reducing rates")
        elif fees_applied / total_users < 0.1:
            recommendations.append("Low demurrage applicability - rates may be too conservative")
        
        return recommendations
    
    # === DATABASE OPERATIONS ===
    
    async def _get_user_transactions(self, user_id: str, start_date: datetime, end_date: datetime):
        """Get user transactions within date range"""
        
        try:
            # This would integrate with the actual FTNS transaction system
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            await logger.aerror("Failed to get user transactions", user_id=user_id, error=str(e))
            return []
    
    async def _get_user_balance(self, user_id: str) -> Decimal:
        """Get current user balance"""
        
        if self.ftns:
            try:
                return await self.ftns.get_balance(user_id)
            except Exception as e:
                await logger.aerror("Failed to get user balance", user_id=user_id, error=str(e))
        
        return Decimal('0')
    
    async def _get_users_with_balances(self, min_balance: Decimal = Decimal('0.001')) -> List[str]:
        """Get list of users with balances above threshold"""
        
        try:
            # This would integrate with the actual FTNS balance system
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            await logger.aerror("Failed to get users with balances", error=str(e))
            return []
    
    async def _get_user_creation_date(self, user_id: str) -> Optional[datetime]:
        """Get user account creation date"""
        
        try:
            # This would integrate with the user management system
            # For now, return None as placeholder
            return None
        except Exception as e:
            await logger.aerror("Failed to get user creation date", user_id=user_id, error=str(e))
            return None
    
    async def _record_demurrage_transaction(self, user_id: str, calculation: DemurrageCalculation):
        """Record demurrage transaction in database"""
        
        try:
            demurrage_record = FTNSDemurrageRecord(
                user_id=user_id,
                fee_amount=calculation.fee_amount,
                monthly_rate=Decimal(str(calculation.monthly_rate)),
                daily_rate=Decimal(str(calculation.daily_rate)),
                balance_before=calculation.current_balance,
                balance_after=calculation.current_balance - calculation.fee_amount,
                velocity=Decimal(str(calculation.velocity)),
                contributor_status=calculation.contributor_status,
                status=DemurrageStatus.APPLIED.value,
                applied_at=datetime.now(timezone.utc),
                metadata={
                    "velocity_category": self._categorize_velocity(calculation.velocity),
                    "target_velocity": self.target_velocity,
                    "calculation_method": "velocity_based_demurrage"
                }
            )
            
            self.db.add(demurrage_record)
            await self.db.commit()
            
        except Exception as e:
            await logger.aerror("Failed to record demurrage transaction", user_id=user_id, error=str(e))
            await self.db.rollback()
    
    async def _store_demurrage_metrics(self, results: List[Dict], total_fees: Decimal):
        """Store aggregate demurrage metrics"""
        
        try:
            # Store network-level metrics for analytics
            pass  # Implementation would store aggregate metrics
        except Exception as e:
            await logger.aerror("Failed to store demurrage metrics", error=str(e))
    
    async def _get_demurrage_history(self, start_date: datetime, end_date: datetime):
        """Get demurrage records within date range"""
        
        try:
            result = await self.db.execute(
                select(FTNSDemurrageRecord)
                .where(
                    and_(
                        FTNSDemurrageRecord.applied_at >= start_date,
                        FTNSDemurrageRecord.applied_at <= end_date,
                        FTNSDemurrageRecord.status == DemurrageStatus.APPLIED.value
                    )
                )
                .order_by(desc(FTNSDemurrageRecord.applied_at))
            )
            
            return result.scalars().all()
            
        except Exception as e:
            await logger.aerror("Failed to get demurrage history", error=str(e))
            return []
    
    # === CONFIGURATION AND STATUS ===
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get current anti-hoarding engine status and configuration"""
        
        network_metrics = await self.calculate_network_velocity()
        
        return {
            "engine_active": True,
            "configuration": {
                "target_velocity": self.target_velocity,
                "base_demurrage_rate": self.base_demurrage_rate,
                "max_demurrage_rate": self.max_demurrage_rate,
                "velocity_calculation_days": self.velocity_calculation_days,
                "grace_period_days": self.grace_period_days,
                "min_fee_threshold": float(self.min_fee_threshold)
            },
            "current_metrics": network_metrics,
            "status_timestamp": datetime.now(timezone.utc).isoformat()
        }
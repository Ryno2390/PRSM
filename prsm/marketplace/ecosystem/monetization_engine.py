#!/usr/bin/env python3
"""
Monetization Engine
===================

Comprehensive monetization and billing system for marketplace integrations
with multiple pricing models, subscription management, and revenue analytics.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import uuid
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

from prsm.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class BillingCycle(Enum):
    """Billing cycle types"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    ONE_TIME = "one_time"
    USAGE_BASED = "usage_based"
    PAY_PER_USE = "pay_per_use"


class PaymentStatus(Enum):
    """Payment status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    DISPUTED = "disputed"
    PARTIAL_REFUND = "partial_refund"


class SubscriptionStatus(Enum):
    """Subscription status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    TRIAL = "trial"
    PAST_DUE = "past_due"


class PricingTier(Enum):
    """Pricing tiers"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class PricingPlan:
    """Pricing plan definition"""
    plan_id: str
    name: str
    description: str = ""
    
    # Pricing details
    base_price: Decimal = Decimal('0.00')
    currency: str = "USD"
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    
    # Usage-based pricing
    usage_based: bool = False
    usage_unit: str = ""  # requests, tokens, MB, etc.
    price_per_unit: Decimal = Decimal('0.00')
    included_units: int = 0
    
    # Limits and quotas
    request_limit_per_month: Optional[int] = None
    data_limit_mb: Optional[int] = None
    concurrent_users: Optional[int] = None
    
    # Features and capabilities
    features: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    
    # Trial configuration
    trial_available: bool = False
    trial_duration_days: int = 14
    
    # Discounts
    annual_discount_percent: Decimal = Decimal('0.00')
    volume_discounts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Enterprise features
    custom_pricing: bool = False
    contract_required: bool = False
    sla_included: bool = False
    
    # Metadata
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_price(self, units_used: int = 0, billing_cycle: Optional[BillingCycle] = None) -> Decimal:
        """Calculate price for given usage"""
        
        cycle = billing_cycle or self.billing_cycle
        
        if cycle == BillingCycle.ONE_TIME:
            return self.base_price
        
        # Base subscription price
        if cycle == BillingCycle.MONTHLY:
            base = self.base_price
        elif cycle == BillingCycle.QUARTERLY:
            base = self.base_price * Decimal('3')
        elif cycle == BillingCycle.ANNUALLY:
            base = self.base_price * Decimal('12')
            # Apply annual discount
            if self.annual_discount_percent > 0:
                discount = base * (self.annual_discount_percent / Decimal('100'))
                base -= discount
        else:
            base = self.base_price
        
        # Add usage-based charges
        usage_charge = Decimal('0.00')
        if self.usage_based and units_used > self.included_units:
            overage_units = units_used - self.included_units
            usage_charge = Decimal(str(overage_units)) * self.price_per_unit
        
        total = base + usage_charge
        
        # Apply volume discounts
        for discount in self.volume_discounts:
            min_amount = Decimal(str(discount.get('min_amount', 0)))
            discount_percent = Decimal(str(discount.get('discount_percent', 0)))
            
            if total >= min_amount:
                discount_amount = total * (discount_percent / Decimal('100'))
                total -= discount_amount
                break  # Apply highest applicable discount only
        
        return total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "base_price": float(self.base_price),
            "currency": self.currency,
            "billing_cycle": self.billing_cycle.value,
            "usage_based": self.usage_based,
            "usage_unit": self.usage_unit,
            "price_per_unit": float(self.price_per_unit),
            "included_units": self.included_units,
            "request_limit_per_month": self.request_limit_per_month,
            "data_limit_mb": self.data_limit_mb,
            "concurrent_users": self.concurrent_users,
            "features": self.features,
            "restrictions": self.restrictions,
            "trial_available": self.trial_available,
            "trial_duration_days": self.trial_duration_days,
            "annual_discount_percent": float(self.annual_discount_percent),
            "volume_discounts": self.volume_discounts,
            "custom_pricing": self.custom_pricing,
            "contract_required": self.contract_required,
            "sla_included": self.sla_included,
            "active": self.active,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Subscription:
    """User subscription to an integration"""
    subscription_id: str
    user_id: str
    integration_id: str
    plan_id: str
    
    # Subscription details
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    current_period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30))
    
    # Billing information
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    amount: Decimal = Decimal('0.00')
    currency: str = "USD"
    
    # Usage tracking
    usage_current_period: int = 0
    usage_limit: Optional[int] = None
    
    # Trial information
    trial_end: Optional[datetime] = None
    is_trial: bool = False
    
    # Payment information
    payment_method_id: Optional[str] = None
    last_payment_date: Optional[datetime] = None
    next_payment_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30))
    
    # Cancellation
    cancelled_at: Optional[datetime] = None
    cancel_at_period_end: bool = False
    cancellation_reason: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_active(self) -> bool:
        """Check if subscription is active"""
        return self.status == SubscriptionStatus.ACTIVE and datetime.now(timezone.utc) < self.current_period_end
    
    def is_in_trial(self) -> bool:
        """Check if subscription is in trial period"""
        return self.is_trial and self.trial_end and datetime.now(timezone.utc) < self.trial_end
    
    def days_until_renewal(self) -> int:
        """Get days until next renewal"""
        delta = self.next_payment_date - datetime.now(timezone.utc)
        return max(0, delta.days)
    
    def calculate_usage_percentage(self) -> float:
        """Calculate usage percentage for current period"""
        if not self.usage_limit:
            return 0.0
        
        return min(100.0, (self.usage_current_period / self.usage_limit) * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "subscription_id": self.subscription_id,
            "user_id": self.user_id,
            "integration_id": self.integration_id,
            "plan_id": self.plan_id,
            "status": self.status.value,
            "current_period_start": self.current_period_start.isoformat(),
            "current_period_end": self.current_period_end.isoformat(),
            "billing_cycle": self.billing_cycle.value,
            "amount": float(self.amount),
            "currency": self.currency,
            "usage_current_period": self.usage_current_period,
            "usage_limit": self.usage_limit,
            "usage_percentage": self.calculate_usage_percentage(),
            "trial_end": self.trial_end.isoformat() if self.trial_end else None,
            "is_trial": self.is_trial,
            "is_in_trial": self.is_in_trial(),
            "payment_method_id": self.payment_method_id,
            "last_payment_date": self.last_payment_date.isoformat() if self.last_payment_date else None,
            "next_payment_date": self.next_payment_date.isoformat(),
            "days_until_renewal": self.days_until_renewal(),
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "cancel_at_period_end": self.cancel_at_period_end,
            "cancellation_reason": self.cancellation_reason,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class Transaction:
    """Financial transaction record"""
    transaction_id: str
    subscription_id: Optional[str] = None
    user_id: str = ""
    integration_id: str = ""
    
    # Transaction details
    transaction_type: str = "subscription"  # subscription, usage, one_time, refund
    amount: Decimal = Decimal('0.00')
    currency: str = "USD"
    description: str = ""
    
    # Payment information
    payment_method: str = ""
    payment_processor: str = ""
    processor_transaction_id: Optional[str] = None
    
    # Status and timing
    status: PaymentStatus = PaymentStatus.PENDING
    processed_at: Optional[datetime] = None
    
    # Revenue sharing
    platform_fee: Decimal = Decimal('0.00')
    developer_payout: Decimal = Decimal('0.00')
    revenue_share_percent: Decimal = Decimal('70.00')
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_revenue_split(self, platform_fee_percent: Decimal = Decimal('30.00')):
        """Calculate revenue split between platform and developer"""
        
        self.platform_fee = (self.amount * platform_fee_percent / Decimal('100')).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )
        
        self.developer_payout = (self.amount - self.platform_fee).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )
        
        self.revenue_share_percent = Decimal('100.00') - platform_fee_percent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "transaction_id": self.transaction_id,
            "subscription_id": self.subscription_id,
            "user_id": self.user_id,
            "integration_id": self.integration_id,
            "transaction_type": self.transaction_type,
            "amount": float(self.amount),
            "currency": self.currency,
            "description": self.description,
            "payment_method": self.payment_method,
            "payment_processor": self.payment_processor,
            "processor_transaction_id": self.processor_transaction_id,
            "status": self.status.value,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "platform_fee": float(self.platform_fee),
            "developer_payout": float(self.developer_payout),
            "revenue_share_percent": float(self.revenue_share_percent),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class UsageTracker:
    """Usage tracking for billing purposes"""
    
    def __init__(self):
        self.usage_records: Dict[str, Dict[str, Any]] = {}
        self.usage_cache: Dict[str, int] = {}
        self.cache_ttl_seconds = 3600  # 1 hour
    
    def record_usage(self, subscription_id: str, usage_type: str, units: int = 1,
                    metadata: Optional[Dict[str, Any]] = None):
        """Record usage for a subscription"""
        
        now = datetime.now(timezone.utc)
        
        if subscription_id not in self.usage_records:
            self.usage_records[subscription_id] = {}
        
        if usage_type not in self.usage_records[subscription_id]:
            self.usage_records[subscription_id][usage_type] = []
        
        usage_record = {
            "timestamp": now.isoformat(),
            "units": units,
            "metadata": metadata or {}
        }
        
        self.usage_records[subscription_id][usage_type].append(usage_record)
        
        # Update cache
        cache_key = f"{subscription_id}:{usage_type}"
        self.usage_cache[cache_key] = self.usage_cache.get(cache_key, 0) + units
    
    def get_usage(self, subscription_id: str, usage_type: str,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> int:
        """Get usage count for subscription"""
        
        cache_key = f"{subscription_id}:{usage_type}"
        
        # Return cached value if no date range specified
        if not start_date and not end_date:
            return self.usage_cache.get(cache_key, 0)
        
        # Calculate usage for date range
        if subscription_id not in self.usage_records:
            return 0
        
        if usage_type not in self.usage_records[subscription_id]:
            return 0
        
        total_usage = 0
        
        for record in self.usage_records[subscription_id][usage_type]:
            record_time = datetime.fromisoformat(record["timestamp"].replace('Z', '+00:00'))
            
            if start_date and record_time < start_date:
                continue
            
            if end_date and record_time > end_date:
                continue
            
            total_usage += record["units"]
        
        return total_usage
    
    def get_usage_summary(self, subscription_id: str) -> Dict[str, Any]:
        """Get usage summary for subscription"""
        
        if subscription_id not in self.usage_records:
            return {}
        
        summary = {}
        
        for usage_type, records in self.usage_records[subscription_id].items():
            total_units = sum(record["units"] for record in records)
            
            summary[usage_type] = {
                "total_units": total_units,
                "record_count": len(records),
                "first_usage": records[0]["timestamp"] if records else None,
                "last_usage": records[-1]["timestamp"] if records else None
            }
        
        return summary
    
    def reset_usage(self, subscription_id: str, usage_type: Optional[str] = None):
        """Reset usage for subscription"""
        
        if subscription_id not in self.usage_records:
            return
        
        if usage_type:
            # Reset specific usage type
            if usage_type in self.usage_records[subscription_id]:
                self.usage_records[subscription_id][usage_type] = []
                cache_key = f"{subscription_id}:{usage_type}"
                self.usage_cache[cache_key] = 0
        else:
            # Reset all usage for subscription
            for utype in self.usage_records[subscription_id]:
                self.usage_records[subscription_id][utype] = []
                cache_key = f"{subscription_id}:{utype}"
                self.usage_cache[cache_key] = 0


class BillingProcessor:
    """Billing and payment processing"""
    
    def __init__(self):
        self.payment_processors = {
            "stripe": self._process_stripe_payment,
            "paypal": self._process_paypal_payment,
            "square": self._process_square_payment
        }
        
        self.webhook_handlers = {}
    
    async def process_subscription_payment(self, subscription: Subscription,
                                         amount: Decimal) -> Transaction:
        """Process subscription payment"""
        
        transaction = Transaction(
            transaction_id=f"txn_{uuid.uuid4().hex[:8]}",
            subscription_id=subscription.subscription_id,
            user_id=subscription.user_id,
            integration_id=subscription.integration_id,
            transaction_type="subscription",
            amount=amount,
            currency=subscription.currency,
            description=f"Subscription payment for {subscription.integration_id}"
        )
        
        try:
            # Process payment through configured processor
            processor = "stripe"  # Would be configurable
            
            if processor in self.payment_processors:
                success = await self.payment_processors[processor](transaction)
                
                if success:
                    transaction.status = PaymentStatus.COMPLETED
                    transaction.processed_at = datetime.now(timezone.utc)
                    
                    # Update subscription
                    subscription.last_payment_date = transaction.processed_at
                    subscription.status = SubscriptionStatus.ACTIVE
                    
                    # Calculate next payment date
                    if subscription.billing_cycle == BillingCycle.MONTHLY:
                        subscription.next_payment_date = subscription.current_period_end + timedelta(days=30)
                    elif subscription.billing_cycle == BillingCycle.ANNUALLY:
                        subscription.next_payment_date = subscription.current_period_end + timedelta(days=365)
                    
                else:
                    transaction.status = PaymentStatus.FAILED
                    subscription.status = SubscriptionStatus.PAST_DUE
            
            else:
                transaction.status = PaymentStatus.FAILED
                logger.error(f"Unknown payment processor: {processor}")
        
        except Exception as e:
            transaction.status = PaymentStatus.FAILED
            logger.error(f"Payment processing failed: {e}")
        
        return transaction
    
    async def process_usage_payment(self, subscription: Subscription,
                                  usage_amount: Decimal) -> Transaction:
        """Process usage-based payment"""
        
        transaction = Transaction(
            transaction_id=f"txn_{uuid.uuid4().hex[:8]}",
            subscription_id=subscription.subscription_id,
            user_id=subscription.user_id,
            integration_id=subscription.integration_id,
            transaction_type="usage",
            amount=usage_amount,
            currency=subscription.currency,
            description=f"Usage charges for {subscription.integration_id}"
        )
        
        # Process similar to subscription payment
        processor = "stripe"
        
        try:
            if processor in self.payment_processors:
                success = await self.payment_processors[processor](transaction)
                
                if success:
                    transaction.status = PaymentStatus.COMPLETED
                    transaction.processed_at = datetime.now(timezone.utc)
                else:
                    transaction.status = PaymentStatus.FAILED
            else:
                transaction.status = PaymentStatus.FAILED
        
        except Exception as e:
            transaction.status = PaymentStatus.FAILED
            logger.error(f"Usage payment processing failed: {e}")
        
        return transaction
    
    async def process_refund(self, original_transaction: Transaction,
                           refund_amount: Decimal, reason: str = "") -> Transaction:
        """Process refund"""
        
        refund_transaction = Transaction(
            transaction_id=f"rfn_{uuid.uuid4().hex[:8]}",
            subscription_id=original_transaction.subscription_id,
            user_id=original_transaction.user_id,
            integration_id=original_transaction.integration_id,
            transaction_type="refund",
            amount=refund_amount,
            currency=original_transaction.currency,
            description=f"Refund for {original_transaction.transaction_id}: {reason}"
        )
        
        try:
            # Process refund through payment processor
            processor = original_transaction.payment_processor or "stripe"
            
            if processor in self.payment_processors:
                success = await self._process_refund_with_processor(
                    processor, original_transaction, refund_amount
                )
                
                if success:
                    refund_transaction.status = PaymentStatus.COMPLETED
                    refund_transaction.processed_at = datetime.now(timezone.utc)
                else:
                    refund_transaction.status = PaymentStatus.FAILED
            else:
                refund_transaction.status = PaymentStatus.FAILED
        
        except Exception as e:
            refund_transaction.status = PaymentStatus.FAILED
            logger.error(f"Refund processing failed: {e}")
        
        return refund_transaction
    
    async def _process_stripe_payment(self, transaction: Transaction) -> bool:
        """Process payment through Stripe"""
        
        # Mock Stripe integration
        # In production, this would use the actual Stripe API
        
        try:
            # Simulate payment processing
            await asyncio.sleep(0.1)
            
            # Calculate revenue split
            transaction.calculate_revenue_split()
            
            # Simulate success/failure
            import random
            success_rate = 0.95  # 95% success rate
            
            return random.random() < success_rate
        
        except Exception as e:
            logger.error(f"Stripe payment processing error: {e}")
            return False
    
    async def _process_paypal_payment(self, transaction: Transaction) -> bool:
        """Process payment through PayPal"""
        
        # Mock PayPal integration
        try:
            await asyncio.sleep(0.15)
            transaction.calculate_revenue_split()
            
            import random
            return random.random() < 0.92  # 92% success rate
        
        except Exception as e:
            logger.error(f"PayPal payment processing error: {e}")
            return False
    
    async def _process_square_payment(self, transaction: Transaction) -> bool:
        """Process payment through Square"""
        
        # Mock Square integration
        try:
            await asyncio.sleep(0.12)
            transaction.calculate_revenue_split()
            
            import random
            return random.random() < 0.94  # 94% success rate
        
        except Exception as e:
            logger.error(f"Square payment processing error: {e}")
            return False
    
    async def _process_refund_with_processor(self, processor: str,
                                           original_transaction: Transaction,
                                           refund_amount: Decimal) -> bool:
        """Process refund with specific processor"""
        
        # Mock refund processing
        try:
            await asyncio.sleep(0.2)
            
            # Simulate refund success
            import random
            return random.random() < 0.98  # 98% refund success rate
        
        except Exception as e:
            logger.error(f"Refund processing error: {e}")
            return False


class RevenueAnalytics:
    """Revenue analytics and reporting"""
    
    def __init__(self):
        self.analytics_cache: Dict[str, Any] = {}
        self.cache_ttl_seconds = 1800  # 30 minutes
    
    async def get_revenue_summary(self, integration_id: Optional[str] = None,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get revenue summary"""
        
        # Mock analytics data
        # In production, this would query actual transaction data
        
        if not start_date:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        if not end_date:
            end_date = datetime.now(timezone.utc)
        
        summary = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "revenue": {
                "total_revenue": 15750.50,
                "subscription_revenue": 12500.00,
                "usage_revenue": 2850.50,
                "one_time_revenue": 400.00,
                "platform_fees": 4725.15,
                "developer_payouts": 11025.35
            },
            "subscriptions": {
                "new_subscriptions": 45,
                "cancelled_subscriptions": 8,
                "active_subscriptions": 320,
                "churn_rate": 0.025,
                "mrr_growth": 0.12
            },
            "usage": {
                "total_usage_charges": 2850.50,
                "average_usage_per_user": 1250,
                "top_usage_integration": integration_id or "ai_model_premium"
            },
            "trends": {
                "revenue_growth_rate": 0.15,
                "subscriber_growth_rate": 0.18,
                "average_revenue_per_user": 49.22
            }
        }
        
        return summary
    
    async def get_integration_revenue(self, integration_id: str,
                                    period_days: int = 30) -> Dict[str, Any]:
        """Get revenue analytics for specific integration"""
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=period_days)
        
        # Mock integration-specific data
        analytics = {
            "integration_id": integration_id,
            "period_days": period_days,
            "revenue_breakdown": {
                "total_revenue": 2450.75,
                "subscription_revenue": 1800.00,
                "usage_revenue": 550.75,
                "one_time_revenue": 100.00,
                "developer_payout": 1715.53,
                "platform_fee": 735.22
            },
            "subscriber_metrics": {
                "total_subscribers": 85,
                "new_subscribers": 12,
                "churned_subscribers": 3,
                "subscriber_growth_rate": 0.11,
                "churn_rate": 0.035
            },
            "usage_metrics": {
                "total_usage_units": 125000,
                "average_usage_per_user": 1470,
                "usage_revenue_per_unit": 0.0044,
                "high_usage_users": 15
            },
            "performance_indicators": {
                "revenue_per_subscriber": 28.83,
                "customer_lifetime_value": 245.50,
                "conversion_rate": 0.08,
                "revenue_growth_30d": 0.22
            }
        }
        
        return analytics
    
    async def get_developer_earnings(self, developer_id: str,
                                   period_days: int = 30) -> Dict[str, Any]:
        """Get earnings analytics for developer"""
        
        # Mock developer earnings data
        earnings = {
            "developer_id": developer_id,
            "period_days": period_days,
            "earnings_summary": {
                "total_earnings": 8750.25,
                "subscription_earnings": 7200.00,
                "usage_earnings": 1350.25,
                "one_time_earnings": 200.00,
                "pending_payouts": 850.50,
                "paid_out": 7899.75
            },
            "integration_breakdown": [
                {
                    "integration_id": "ai_model_premium",
                    "earnings": 3500.50,
                    "subscribers": 140,
                    "revenue_share": 0.75
                },
                {
                    "integration_id": "data_connector_pro",
                    "earnings": 2800.25,
                    "subscribers": 95,
                    "revenue_share": 0.70
                }
            ],
            "payout_schedule": {
                "next_payout_date": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                "payout_frequency": "weekly",
                "minimum_payout": 100.00
            },
            "performance_metrics": {
                "earnings_growth_rate": 0.28,
                "average_earnings_per_integration": 1458.38,
                "highest_earning_integration": "ai_model_premium",
                "subscriber_retention_rate": 0.92
            }
        }
        
        return earnings
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        
        if cache_key not in self.analytics_cache:
            return False
        
        cached_entry = self.analytics_cache[cache_key]
        age = (datetime.now(timezone.utc) - cached_entry["cached_at"]).total_seconds()
        
        return age < self.cache_ttl_seconds


class MonetizationEngine:
    """Main monetization and billing engine"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./monetization_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Core components
        self.pricing_plans: Dict[str, PricingPlan] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.transactions: Dict[str, Transaction] = {}
        
        self.usage_tracker = UsageTracker()
        self.billing_processor = BillingProcessor()
        self.revenue_analytics = RevenueAnalytics()
        
        # Default platform settings
        self.platform_fee_percent = Decimal('30.00')
        self.trial_duration_days = 14
        
        # Statistics
        self.stats = {
            "total_revenue": Decimal('0.00'),
            "monthly_recurring_revenue": Decimal('0.00'),
            "active_subscriptions": 0,
            "total_transactions": 0,
            "churn_rate": 0.0,
            "average_revenue_per_user": Decimal('0.00')
        }
        
        logger.info("Monetization Engine initialized")
    
    def create_pricing_plan(self, plan: PricingPlan) -> bool:
        """Create a new pricing plan"""
        
        try:
            self.pricing_plans[plan.plan_id] = plan
            
            logger.info(f"Created pricing plan: {plan.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create pricing plan: {e}")
            return False
    
    def get_pricing_plan(self, plan_id: str) -> Optional[PricingPlan]:
        """Get pricing plan by ID"""
        return self.pricing_plans.get(plan_id)
    
    def list_pricing_plans(self, integration_id: Optional[str] = None,
                          active_only: bool = True) -> List[Dict[str, Any]]:
        """List pricing plans"""
        
        plans = list(self.pricing_plans.values())
        
        if active_only:
            plans = [plan for plan in plans if plan.active]
        
        return [plan.to_dict() for plan in plans]
    
    async def create_subscription(self, user_id: str, integration_id: str,
                                plan_id: str, payment_method_id: str,
                                start_trial: bool = False) -> Optional[str]:
        """Create new subscription"""
        
        plan = self.get_pricing_plan(plan_id)
        if not plan:
            logger.error(f"Pricing plan not found: {plan_id}")
            return None
        
        try:
            subscription_id = f"sub_{uuid.uuid4().hex[:8]}"
            
            # Calculate subscription period
            now = datetime.now(timezone.utc)
            
            if start_trial and plan.trial_available:
                # Start with trial
                trial_end = now + timedelta(days=plan.trial_duration_days)
                period_end = trial_end
                is_trial = True
                status = SubscriptionStatus.TRIAL
            else:
                # Regular subscription
                if plan.billing_cycle == BillingCycle.MONTHLY:
                    period_end = now + timedelta(days=30)
                elif plan.billing_cycle == BillingCycle.QUARTERLY:
                    period_end = now + timedelta(days=90)
                elif plan.billing_cycle == BillingCycle.ANNUALLY:
                    period_end = now + timedelta(days=365)
                else:
                    period_end = now + timedelta(days=30)
                
                trial_end = None
                is_trial = False
                status = SubscriptionStatus.ACTIVE
            
            subscription = Subscription(
                subscription_id=subscription_id,
                user_id=user_id,
                integration_id=integration_id,
                plan_id=plan_id,
                status=status,
                current_period_start=now,
                current_period_end=period_end,
                billing_cycle=plan.billing_cycle,
                amount=plan.base_price,
                currency=plan.currency,
                usage_limit=plan.request_limit_per_month,
                trial_end=trial_end,
                is_trial=is_trial,
                payment_method_id=payment_method_id,
                next_payment_date=period_end
            )
            
            self.subscriptions[subscription_id] = subscription
            
            # Update statistics
            self.stats["active_subscriptions"] += 1
            self._update_mrr()
            
            logger.info(f"Created subscription: {subscription_id} for user {user_id}")
            
            return subscription_id
        
        except Exception as e:
            logger.error(f"Failed to create subscription: {e}")
            return None
    
    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID"""
        return self.subscriptions.get(subscription_id)
    
    def get_user_subscriptions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all subscriptions for user"""
        
        user_subscriptions = [
            sub for sub in self.subscriptions.values()
            if sub.user_id == user_id
        ]
        
        return [sub.to_dict() for sub in user_subscriptions]
    
    async def cancel_subscription(self, subscription_id: str,
                                cancel_immediately: bool = False,
                                reason: str = "") -> bool:
        """Cancel subscription"""
        
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            logger.error(f"Subscription not found: {subscription_id}")
            return False
        
        try:
            if cancel_immediately:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.cancelled_at = datetime.now(timezone.utc)
                subscription.current_period_end = datetime.now(timezone.utc)
                
                # Update statistics
                self.stats["active_subscriptions"] -= 1
            else:
                subscription.cancel_at_period_end = True
                subscription.cancelled_at = datetime.now(timezone.utc)
            
            subscription.cancellation_reason = reason
            subscription.updated_at = datetime.now(timezone.utc)
            
            self._update_mrr()
            
            logger.info(f"Cancelled subscription: {subscription_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to cancel subscription: {e}")
            return False
    
    def record_usage(self, subscription_id: str, usage_type: str = "requests",
                    units: int = 1, metadata: Optional[Dict[str, Any]] = None):
        """Record usage for subscription"""
        
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            logger.warning(f"Subscription not found for usage recording: {subscription_id}")
            return
        
        # Record usage
        self.usage_tracker.record_usage(subscription_id, usage_type, units, metadata)
        
        # Update subscription usage counter
        subscription.usage_current_period += units
        subscription.updated_at = datetime.now(timezone.utc)
    
    def get_usage(self, subscription_id: str, usage_type: str = "requests") -> int:
        """Get usage for subscription"""
        return self.usage_tracker.get_usage(subscription_id, usage_type)
    
    async def process_billing_cycle(self) -> Dict[str, Any]:
        """Process billing for all subscriptions"""
        
        results = {
            "processed_subscriptions": 0,
            "successful_payments": 0,
            "failed_payments": 0,
            "total_amount": Decimal('0.00'),
            "errors": []
        }
        
        now = datetime.now(timezone.utc)
        
        # Find subscriptions due for billing
        due_subscriptions = [
            sub for sub in self.subscriptions.values()
            if (sub.next_payment_date <= now and 
                sub.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.PAST_DUE])
        ]
        
        for subscription in due_subscriptions:
            try:
                results["processed_subscriptions"] += 1
                
                # Get pricing plan
                plan = self.get_pricing_plan(subscription.plan_id)
                if not plan:
                    results["errors"].append(f"Plan not found for subscription {subscription.subscription_id}")
                    continue
                
                # Calculate billing amount
                usage_units = subscription.usage_current_period
                billing_amount = plan.calculate_price(usage_units, subscription.billing_cycle)
                
                # Process payment
                transaction = await self.billing_processor.process_subscription_payment(
                    subscription, billing_amount
                )
                
                self.transactions[transaction.transaction_id] = transaction
                
                if transaction.status == PaymentStatus.COMPLETED:
                    results["successful_payments"] += 1
                    results["total_amount"] += billing_amount
                    
                    # Update subscription period
                    if subscription.billing_cycle == BillingCycle.MONTHLY:
                        subscription.current_period_start = subscription.current_period_end
                        subscription.current_period_end += timedelta(days=30)
                        subscription.next_payment_date = subscription.current_period_end
                    elif subscription.billing_cycle == BillingCycle.ANNUALLY:
                        subscription.current_period_start = subscription.current_period_end
                        subscription.current_period_end += timedelta(days=365)
                        subscription.next_payment_date = subscription.current_period_end
                    
                    # Reset usage counter
                    subscription.usage_current_period = 0
                    self.usage_tracker.reset_usage(subscription.subscription_id)
                    
                else:
                    results["failed_payments"] += 1
                    results["errors"].append(f"Payment failed for subscription {subscription.subscription_id}")
            
            except Exception as e:
                results["errors"].append(f"Billing error for subscription {subscription.subscription_id}: {e}")
        
        # Update statistics
        self.stats["total_transactions"] += results["processed_subscriptions"]
        self.stats["total_revenue"] += results["total_amount"]
        self._update_mrr()
        
        logger.info(f"Processed billing cycle: {results}")
        
        return results
    
    async def get_revenue_analytics(self, integration_id: Optional[str] = None,
                                  period_days: int = 30) -> Dict[str, Any]:
        """Get revenue analytics"""
        
        if integration_id:
            return await self.revenue_analytics.get_integration_revenue(integration_id, period_days)
        else:
            return await self.revenue_analytics.get_revenue_summary()
    
    async def get_developer_earnings(self, developer_id: str,
                                   period_days: int = 30) -> Dict[str, Any]:
        """Get developer earnings"""
        
        return await self.revenue_analytics.get_developer_earnings(developer_id, period_days)
    
    def _update_mrr(self):
        """Update monthly recurring revenue"""
        
        active_subs = [
            sub for sub in self.subscriptions.values()
            if sub.status == SubscriptionStatus.ACTIVE
        ]
        
        mrr = Decimal('0.00')
        
        for subscription in active_subs:
            plan = self.get_pricing_plan(subscription.plan_id)
            if plan:
                if plan.billing_cycle == BillingCycle.MONTHLY:
                    mrr += plan.base_price
                elif plan.billing_cycle == BillingCycle.ANNUALLY:
                    mrr += plan.base_price / Decimal('12')
                elif plan.billing_cycle == BillingCycle.QUARTERLY:
                    mrr += plan.base_price / Decimal('3')
        
        self.stats["monthly_recurring_revenue"] = mrr
        
        # Calculate ARPU
        if len(active_subs) > 0:
            self.stats["average_revenue_per_user"] = mrr / Decimal(str(len(active_subs)))
        else:
            self.stats["average_revenue_per_user"] = Decimal('0.00')
    
    def get_monetization_stats(self) -> Dict[str, Any]:
        """Get comprehensive monetization statistics"""
        
        # Update dynamic stats
        active_subscriptions = len([
            sub for sub in self.subscriptions.values()
            if sub.status == SubscriptionStatus.ACTIVE
        ])
        
        self.stats["active_subscriptions"] = active_subscriptions
        
        return {
            "monetization_statistics": {
                key: float(value) if isinstance(value, Decimal) else value
                for key, value in self.stats.items()
            },
            "pricing_plans_count": len(self.pricing_plans),
            "subscription_breakdown": {
                status.value: len([s for s in self.subscriptions.values() if s.status == status])
                for status in SubscriptionStatus
            },
            "billing_cycles": {
                cycle.value: len([s for s in self.subscriptions.values() 
                                if s.billing_cycle == cycle and s.status == SubscriptionStatus.ACTIVE])
                for cycle in BillingCycle
            },
            "recent_transactions": len([
                t for t in self.transactions.values()
                if (datetime.now(timezone.utc) - t.created_at).days <= 7
            ]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Export main classes
__all__ = [
    'BillingCycle',
    'PaymentStatus',
    'SubscriptionStatus',
    'PricingTier',
    'PricingPlan',
    'Subscription',
    'Transaction',
    'UsageTracker',
    'BillingProcessor',
    'RevenueAnalytics',
    'MonetizationEngine'
]
"""
FTNS (Fungible Tokens for Node Support) Database Models
Production-ready SQLAlchemy models for PRSM token economy

This module defines the complete database schema for FTNS tokens,
replacing the in-memory simulation with persistent storage.

Core Features:
- User account balances with locked/unlocked funds
- Complete transaction history with blockchain integration
- Provenance tracking for royalty payments
- Marketplace listings and rental transactions
- Impact metrics for research and content
- Dividend distribution management
- Governance voting and participation tracking
- Security audit trails and compliance data

Blockchain Integration:
- Smart contract transaction hashes
- Blockchain confirmation status
- Multi-signature wallet support
- Cross-chain transaction bridging
- Cryptographic signature verification

Economic Analytics:
- Price history and volatility tracking
- Supply/demand metrics
- User behavior analytics
- Economic performance indicators
- Market manipulation detection
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, CheckConstraint, UniqueConstraint, DECIMAL
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

# Create a separate Base for FTNS models to avoid conflicts with core models
Base = declarative_base()


# === Enums ===

class TransactionType(str, Enum):
    """Types of FTNS transactions"""
    REWARD = "reward"               # System rewards for contributions
    CHARGE = "charge"               # Charges for resource usage
    TRANSFER = "transfer"           # User-to-user transfers
    DIVIDEND = "dividend"           # Quarterly dividend payments
    ROYALTY = "royalty"            # Content usage royalties
    MARKETPLACE = "marketplace"     # Marketplace transactions
    GOVERNANCE = "governance"       # Governance participation rewards
    BURN = "burn"                  # Token burning for deflation
    MINT = "mint"                  # New token minting
    STAKE = "stake"                # Staking for governance
    UNSTAKE = "unstake"            # Unstaking tokens
    PENALTY = "penalty"            # Economic penalties
    REFUND = "refund"              # Transaction refunds


class TransactionStatus(str, Enum):
    """Status of FTNS transactions"""
    PENDING = "pending"            # Transaction created but not confirmed
    CONFIRMED = "confirmed"        # Transaction confirmed on blockchain
    COMPLETED = "completed"        # Transaction fully processed
    FAILED = "failed"              # Transaction failed
    CANCELLED = "cancelled"        # Transaction cancelled
    DISPUTED = "disputed"          # Transaction under dispute


class WalletType(str, Enum):
    """Types of FTNS wallets"""
    STANDARD = "standard"          # Regular user wallet
    MULTISIG = "multisig"          # Multi-signature wallet
    INSTITUTIONAL = "institutional" # Enterprise wallet
    ESCROW = "escrow"              # Escrow wallet for marketplace
    TREASURY = "treasury"          # System treasury wallet
    STAKING = "staking"            # Staking rewards wallet


class DividendStatus(str, Enum):
    """Status of dividend distributions"""
    PENDING = "pending"            # Distribution not yet started
    CALCULATING = "calculating"    # Calculating distribution amounts
    DISTRIBUTING = "distributing"  # Currently distributing
    COMPLETED = "completed"        # Distribution completed
    FAILED = "failed"              # Distribution failed


class RoyaltyStatus(str, Enum):
    """Status of royalty payments"""
    PENDING = "pending"            # Royalty calculated but not paid
    PAID = "paid"                  # Royalty payment completed
    DISPUTED = "disputed"          # Royalty payment disputed
    CANCELLED = "cancelled"        # Royalty payment cancelled


class ContributorTier(str, Enum):
    """Contributor status levels for FTNS earning eligibility"""
    NONE = "none"           # No recent contributions - cannot earn FTNS
    BASIC = "basic"         # Minimal contribution threshold met
    ACTIVE = "active"       # Strong contribution history
    POWER_USER = "power"    # Exceptional contributions


class ContributionType(str, Enum):
    """Types of contributions that can be verified"""
    STORAGE = "storage"             # IPFS storage provision
    COMPUTE = "compute"             # Computational work provision  
    DATA = "data"                   # Dataset contribution
    GOVERNANCE = "governance"       # Voting and proposal participation
    DOCUMENTATION = "documentation" # Documentation and guides
    MODEL = "model"                 # AI model contribution
    RESEARCH = "research"           # Research publication
    TEACHING = "teaching"           # Educational content


class SupplyAdjustmentStatus(str, Enum):
    """Status of supply adjustments"""
    CALCULATED = "calculated"
    APPLIED = "applied"
    FAILED = "failed"
    REVERTED = "reverted"


class AdjustmentTrigger(str, Enum):
    """Trigger types for supply adjustments"""
    AUTOMATED = "automated"
    GOVERNANCE = "governance"
    EMERGENCY = "emergency"
    MANUAL = "manual"


class VelocityCategory(str, Enum):
    """Token velocity categories"""
    HIGH = "high"           # >= target velocity
    MODERATE = "moderate"   # 70-100% of target
    LOW = "low"            # 30-70% of target  
    INACTIVE = "inactive"   # < 30% of target


class DemurrageStatus(str, Enum):
    """Status of demurrage fee application"""
    CALCULATED = "calculated"
    APPLIED = "applied"
    FAILED = "failed"
    REVERSED = "reversed"


class ProofStatus(str, Enum):
    """Status of contribution proof verification"""
    PENDING = "pending"             # Proof submitted, awaiting verification
    VERIFIED = "verified"           # Proof successfully verified
    REJECTED = "rejected"           # Proof failed verification
    DISPUTED = "disputed"           # Proof under dispute
    EXPIRED = "expired"             # Proof validity expired


class EmergencyTriggerType(str, Enum):
    """Types of emergency triggers for circuit breakers"""
    PRICE_CRASH = "price_crash"         # Sudden price drop detection
    VOLUME_SPIKE = "volume_spike"       # Unusual trading volume
    SYSTEM_ERROR = "system_error"       # Technical system failures
    GOVERNANCE_HALT = "governance_halt" # Governance-requested halt
    SECURITY_BREACH = "security_breach" # Security incident detection
    ORACLE_FAILURE = "oracle_failure"   # Price oracle malfunction


class EmergencyStatus(str, Enum):
    """Status of emergency triggers and responses"""
    DETECTED = "detected"           # Emergency condition detected
    EVALUATING = "evaluating"       # Automated evaluation in progress
    CONFIRMED = "confirmed"         # Emergency confirmed, actions triggered
    RESOLVING = "resolving"         # Emergency response actions in progress
    RESOLVED = "resolved"           # Emergency resolved, normal operations
    FALSE_ALARM = "false_alarm"     # False positive, no action taken


class EmergencyActionType(str, Enum):
    """Types of emergency response actions"""
    HALT_TRANSACTIONS = "halt_transactions"     # Temporarily halt all FTNS transactions
    REDUCE_LIMITS = "reduce_limits"             # Reduce transaction limits
    NOTIFY_GOVERNANCE = "notify_governance"     # Alert governance for manual review
    ADJUST_RATES = "adjust_rates"               # Emergency rate adjustments
    FREEZE_ACCOUNTS = "freeze_accounts"         # Freeze suspicious accounts
    ACTIVATE_CIRCUIT_BREAKER = "activate_circuit_breaker"  # Full system halt


# === Core FTNS Models ===

class FTNSWallet(Base):
    """User FTNS wallet with balance management"""
    __tablename__ = "ftns_wallets"
    
    # Primary identification
    wallet_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    wallet_type = Column(String(50), nullable=False, default=WalletType.STANDARD.value)
    
    # Balance management
    balance = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    locked_balance = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    staked_balance = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    
    # Blockchain integration
    blockchain_address = Column(String(255), nullable=True, unique=True, index=True)
    public_key = Column(Text, nullable=True)
    wallet_version = Column(String(20), nullable=False, default="1.0")
    
    # Security features
    multisig_threshold = Column(Integer, nullable=True)  # For multisig wallets
    multisig_participants = Column(JSONB, nullable=True)
    security_level = Column(String(20), nullable=False, default="standard")
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True, onupdate=func.now())
    last_transaction = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('balance >= 0', name='positive_balance'),
        CheckConstraint('locked_balance >= 0', name='positive_locked_balance'),
        CheckConstraint('staked_balance >= 0', name='positive_staked_balance'),
        Index('idx_ftns_wallets_user_type', 'user_id', 'wallet_type'),
    )
    
    @validates('balance', 'locked_balance', 'staked_balance')
    def validate_positive_amounts(self, key, value):
        if value < 0:
            raise ValueError(f"{key} cannot be negative")
        return value
    
    @property
    def available_balance(self) -> Decimal:
        """Calculate available (unlocked, unstaked) balance"""
        return self.balance - self.locked_balance - self.staked_balance
    
    @property
    def total_balance(self) -> Decimal:
        """Calculate total balance including locked and staked"""
        return self.balance + self.locked_balance + self.staked_balance


class FTNSTransaction(Base):
    """Complete FTNS transaction record with blockchain integration"""
    __tablename__ = "ftns_transactions"
    
    # Primary identification
    transaction_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    transaction_hash = Column(String(255), nullable=True, unique=True, index=True)
    
    # Transaction details
    from_wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=True)
    to_wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=False)
    amount = Column(DECIMAL(20, 8), nullable=False)
    transaction_type = Column(String(50), nullable=False, index=True)
    status = Column(String(50), nullable=False, default=TransactionStatus.PENDING.value, index=True)
    
    # Blockchain data
    block_number = Column(Integer, nullable=True, index=True)
    block_hash = Column(String(255), nullable=True, index=True)
    gas_fee = Column(DECIMAL(20, 8), nullable=True)
    confirmation_count = Column(Integer, nullable=False, default=0)
    
    # Transaction metadata
    description = Column(Text, nullable=True)
    context_units = Column(Integer, nullable=True)  # For context-based charges
    reference_id = Column(String(255), nullable=True, index=True)  # Reference to related records
    transaction_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Security and audit
    signature = Column(Text, nullable=True)
    nonce = Column(Integer, nullable=True)
    fee_paid = Column(DECIMAL(20, 8), nullable=True)
    
    # Relationships
    from_wallet = relationship("FTNSWallet", foreign_keys=[from_wallet_id], backref="outgoing_transactions")
    to_wallet = relationship("FTNSWallet", foreign_keys=[to_wallet_id], backref="incoming_transactions")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('amount > 0', name='positive_amount'),
        CheckConstraint('confirmation_count >= 0', name='positive_confirmations'),
        Index('idx_ftns_transactions_type_status', 'transaction_type', 'status'),
        Index('idx_ftns_transactions_created', 'created_at'),
        Index('idx_ftns_transactions_amount', 'amount'),
    )
    
    @validates('amount')
    def validate_positive_amount(self, key, value):
        if value <= 0:
            raise ValueError("Transaction amount must be positive")
        return value


class FTNSProvenanceRecord(Base):
    """Content provenance tracking for royalty calculations"""
    __tablename__ = "ftns_provenance_records"
    
    # Primary identification
    record_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    content_cid = Column(String(255), nullable=False, index=True)
    content_type = Column(String(50), nullable=False)  # model, dataset, research, code
    
    # Creator information
    creator_wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=False)
    original_creator_id = Column(String(255), nullable=False, index=True)
    
    # Usage tracking
    access_count = Column(Integer, nullable=False, default=0)
    download_count = Column(Integer, nullable=False, default=0)
    citation_count = Column(Integer, nullable=False, default=0)
    computational_usage_hours = Column(DECIMAL(10, 2), nullable=False, default=Decimal('0.0'))
    
    # Financial tracking
    total_royalties_earned = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    total_royalties_paid = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    royalty_rate = Column(DECIMAL(5, 4), nullable=False, default=Decimal('0.05'))  # 5% default
    
    # Impact metrics
    impact_score = Column(DECIMAL(10, 4), nullable=False, default=Decimal('0.0'))
    quality_score = Column(DECIMAL(3, 2), nullable=False, default=Decimal('1.0'))
    academic_citations = Column(Integer, nullable=False, default=0)
    industry_applications = Column(Integer, nullable=False, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_accessed = Column(DateTime(timezone=True), nullable=True)
    last_royalty_calculation = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    content_metadata = Column(JSONB, nullable=True)
    licensing_terms = Column(Text, nullable=True)
    geographical_restrictions = Column(JSONB, nullable=True)
    
    # Relationships
    creator_wallet = relationship("FTNSWallet", backref="created_content")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('access_count >= 0', name='positive_access_count'),
        CheckConstraint('royalty_rate >= 0 AND royalty_rate <= 1', name='valid_royalty_rate'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 5', name='valid_quality_score'),
        Index('idx_provenance_content_creator', 'content_cid', 'creator_wallet_id'),
        Index('idx_provenance_impact', 'impact_score'),
    )


class FTNSDividendDistribution(Base):
    """Quarterly dividend distribution management"""
    __tablename__ = "ftns_dividend_distributions"
    
    # Primary identification
    distribution_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    quarter = Column(String(20), nullable=False, unique=True, index=True)  # e.g., "2025-Q2"
    
    # Distribution parameters
    total_pool = Column(DECIMAL(20, 8), nullable=False)
    distribution_method = Column(String(50), nullable=False, default="proportional")
    minimum_holding_period_days = Column(Integer, nullable=False, default=30)
    minimum_balance = Column(DECIMAL(20, 8), nullable=False, default=Decimal('1.0'))
    
    # Status tracking
    status = Column(String(50), nullable=False, default=DividendStatus.PENDING.value, index=True)
    eligible_wallets_count = Column(Integer, nullable=False, default=0)
    total_distributed = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    
    # Timestamps
    calculation_started = Column(DateTime(timezone=True), nullable=True)
    distribution_started = Column(DateTime(timezone=True), nullable=True)
    distribution_completed = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Bonus multipliers and metadata
    bonus_multipliers = Column(JSONB, nullable=True)
    distribution_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('total_pool > 0', name='positive_pool'),
        CheckConstraint('minimum_holding_period_days >= 0', name='positive_holding_period'),
        CheckConstraint('total_distributed >= 0', name='positive_distributed'),
    )


class FTNSDividendPayment(Base):
    """Individual dividend payments to wallet holders"""
    __tablename__ = "ftns_dividend_payments"
    
    # Primary identification
    payment_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    distribution_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_dividend_distributions.distribution_id'), nullable=False)
    wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=False)
    
    # Payment details
    base_amount = Column(DECIMAL(20, 8), nullable=False)
    bonus_amount = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    total_amount = Column(DECIMAL(20, 8), nullable=False)
    
    # Calculation factors
    wallet_balance = Column(DECIMAL(20, 8), nullable=False)
    holding_period_days = Column(Integer, nullable=False)
    bonus_multiplier = Column(DECIMAL(5, 4), nullable=False, default=Decimal('1.0'))
    
    # Payment tracking
    transaction_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_transactions.transaction_id'), nullable=True)
    paid_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    distribution = relationship("FTNSDividendDistribution", backref="payments")
    wallet = relationship("FTNSWallet", backref="dividend_payments")
    transaction = relationship("FTNSTransaction", backref="dividend_payment")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('base_amount >= 0', name='positive_base_amount'),
        CheckConstraint('total_amount >= 0', name='positive_total_amount'),
        CheckConstraint('holding_period_days >= 0', name='positive_holding_period'),
        UniqueConstraint('distribution_id', 'wallet_id', name='unique_distribution_wallet'),
        Index('idx_dividend_payments_distribution', 'distribution_id'),
    )


class FTNSRoyaltyPayment(Base):
    """Royalty payments for content usage"""
    __tablename__ = "ftns_royalty_payments"
    
    # Primary identification
    payment_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    provenance_record_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_provenance_records.record_id'), nullable=False)
    
    # Payment period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Usage metrics
    total_usage = Column(DECIMAL(15, 6), nullable=False)
    usage_type = Column(String(50), nullable=False)  # download, citation, computation, derived_work
    unique_users = Column(Integer, nullable=False, default=0)
    
    # Payment calculation
    royalty_rate = Column(DECIMAL(5, 4), nullable=False)
    base_amount = Column(DECIMAL(20, 8), nullable=False)
    impact_multiplier = Column(DECIMAL(5, 4), nullable=False, default=Decimal('1.0'))
    quality_multiplier = Column(DECIMAL(5, 4), nullable=False, default=Decimal('1.0'))
    bonus_amount = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    total_amount = Column(DECIMAL(20, 8), nullable=False)
    
    # Payment tracking
    status = Column(String(50), nullable=False, default=RoyaltyStatus.PENDING.value, index=True)
    transaction_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_transactions.transaction_id'), nullable=True)
    paid_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Metadata
    calculation_metadata = Column(JSONB, nullable=True)
    
    # Relationships
    provenance_record = relationship("FTNSProvenanceRecord", backref="royalty_payments")
    transaction = relationship("FTNSTransaction", backref="royalty_payment")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('total_usage >= 0', name='positive_usage'),
        CheckConstraint('royalty_rate >= 0 AND royalty_rate <= 1', name='valid_royalty_rate'),
        CheckConstraint('total_amount >= 0', name='positive_total_amount'),
        Index('idx_royalty_payments_period', 'period_start', 'period_end'),
        Index('idx_royalty_payments_status', 'status'),
    )


class FTNSMarketplaceListing(Base):
    """Marketplace listings for model rentals and sales"""
    __tablename__ = "ftns_marketplace_listings"
    
    # Primary identification
    listing_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id = Column(String(255), nullable=False, index=True)
    owner_wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=False)
    
    # Listing details
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    pricing_model = Column(String(50), nullable=False)  # hourly, usage, subscription, one_time
    base_price = Column(DECIMAL(20, 8), nullable=False)
    currency = Column(String(10), nullable=False, default="FTNS")
    
    # Availability and restrictions
    availability_status = Column(String(50), nullable=False, default="available", index=True)
    maximum_concurrent_users = Column(Integer, nullable=False, default=1)
    minimum_rental_duration = Column(Integer, nullable=True)  # Hours
    maximum_rental_duration = Column(Integer, nullable=True)  # Hours
    
    # Performance and requirements
    performance_metrics = Column(JSONB, nullable=True)
    resource_requirements = Column(JSONB, nullable=True)
    supported_features = Column(JSONB, nullable=True)
    geographical_restrictions = Column(JSONB, nullable=True)
    
    # Financial tracking
    total_revenue = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    total_rentals = Column(Integer, nullable=False, default=0)
    average_rating = Column(DECIMAL(3, 2), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True, onupdate=func.now())
    last_rented = Column(DateTime(timezone=True), nullable=True)
    
    # Terms and metadata
    terms_of_service = Column(Text, nullable=True)
    listing_metadata = Column(JSONB, nullable=True)
    
    # Relationships
    owner_wallet = relationship("FTNSWallet", backref="marketplace_listings")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('base_price > 0', name='positive_base_price'),
        CheckConstraint('maximum_concurrent_users > 0', name='positive_concurrent_users'),
        CheckConstraint('average_rating IS NULL OR (average_rating >= 0 AND average_rating <= 5)', name='valid_rating'),
        Index('idx_marketplace_listings_price', 'base_price'),
        Index('idx_marketplace_listings_status', 'availability_status'),
    )


class FTNSMarketplaceTransaction(Base):
    """Marketplace transaction records"""
    __tablename__ = "ftns_marketplace_transactions"
    
    # Primary identification
    marketplace_transaction_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    listing_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_marketplace_listings.listing_id'), nullable=False)
    
    # Transaction parties
    buyer_wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=False)
    seller_wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=False)
    
    # Transaction details
    transaction_type = Column(String(50), nullable=False)  # rental, purchase, subscription
    amount = Column(DECIMAL(20, 8), nullable=False)
    platform_fee = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    escrow_amount = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    
    # Rental specific
    rental_duration_hours = Column(Integer, nullable=True)
    rental_started_at = Column(DateTime(timezone=True), nullable=True)
    rental_ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status tracking
    status = Column(String(50), nullable=False, default="pending", index=True)
    payment_transaction_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_transactions.transaction_id'), nullable=True)
    
    # Usage tracking
    usage_metrics = Column(JSONB, nullable=True)
    completion_rating = Column(DECIMAL(3, 2), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    listing = relationship("FTNSMarketplaceListing", backref="transactions")
    buyer_wallet = relationship("FTNSWallet", foreign_keys=[buyer_wallet_id], backref="purchases")
    seller_wallet = relationship("FTNSWallet", foreign_keys=[seller_wallet_id], backref="sales")
    payment_transaction = relationship("FTNSTransaction", backref="marketplace_transaction")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('amount > 0', name='positive_amount'),
        CheckConstraint('platform_fee >= 0', name='positive_platform_fee'),
        CheckConstraint('completion_rating IS NULL OR (completion_rating >= 0 AND completion_rating <= 5)', name='valid_completion_rating'),
        Index('idx_marketplace_transactions_listing', 'listing_id'),
        Index('idx_marketplace_transactions_status', 'status'),
    )


class FTNSPriceHistory(Base):
    """FTNS token price history for analytics and market data"""
    __tablename__ = "ftns_price_history"
    
    # Primary identification
    price_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Price data
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    price_usd = Column(DECIMAL(20, 8), nullable=False)
    price_btc = Column(DECIMAL(20, 8), nullable=True)
    price_eth = Column(DECIMAL(20, 8), nullable=True)
    
    # Volume and market data
    volume_24h = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    market_cap = Column(DECIMAL(20, 8), nullable=True)
    circulating_supply = Column(DECIMAL(20, 8), nullable=False)
    
    # Exchange data
    exchange_name = Column(String(100), nullable=True)
    trading_pair = Column(String(20), nullable=False, default="FTNS/USD")
    
    # Market metrics
    volatility_24h = Column(DECIMAL(10, 6), nullable=True)
    price_change_24h = Column(DECIMAL(10, 6), nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('price_usd > 0', name='positive_price_usd'),
        CheckConstraint('volume_24h >= 0', name='positive_volume'),
        CheckConstraint('circulating_supply > 0', name='positive_supply'),
        Index('idx_price_history_timestamp', 'timestamp'),
        Index('idx_price_history_exchange_pair', 'exchange_name', 'trading_pair'),
    )


class FTNSGovernanceVote(Base):
    """Governance voting records with FTNS stake weighting"""
    __tablename__ = "ftns_governance_votes"
    
    # Primary identification
    vote_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    proposal_id = Column(String(255), nullable=False, index=True)
    voter_wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=False)
    
    # Vote details
    vote_choice = Column(Boolean, nullable=False)  # True for yes, False for no
    voting_power = Column(DECIMAL(20, 8), nullable=False)
    staked_amount = Column(DECIMAL(20, 8), nullable=False)
    
    # Metadata
    rationale = Column(Text, nullable=True)
    vote_metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    voted_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    stake_locked_until = Column(DateTime(timezone=True), nullable=False)
    
    # Relationships
    voter_wallet = relationship("FTNSWallet", backref="governance_votes")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('voting_power > 0', name='positive_voting_power'),
        CheckConstraint('staked_amount > 0', name='positive_staked_amount'),
        UniqueConstraint('proposal_id', 'voter_wallet_id', name='unique_proposal_vote'),
        Index('idx_governance_votes_proposal', 'proposal_id'),
    )


class FTNSAuditLog(Base):
    """Comprehensive audit log for all FTNS operations"""
    __tablename__ = "ftns_audit_logs"
    
    # Primary identification
    log_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Event details
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)  # transaction, governance, security, admin
    severity = Column(String(20), nullable=False, default="info")  # debug, info, warning, error, critical
    
    # Actor information
    actor_wallet_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_wallets.wallet_id'), nullable=True)
    actor_ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    
    # Event data
    description = Column(Text, nullable=False)
    event_data = Column(JSONB, nullable=True)
    before_state = Column(JSONB, nullable=True)
    after_state = Column(JSONB, nullable=True)
    
    # References
    related_transaction_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_transactions.transaction_id'), nullable=True)
    related_entity_type = Column(String(100), nullable=True)
    related_entity_id = Column(String(255), nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    
    # Security
    signature = Column(Text, nullable=True)
    checksum = Column(String(255), nullable=True)
    
    # Relationships
    actor_wallet = relationship("FTNSWallet", backref="audit_logs")
    related_transaction = relationship("FTNSTransaction", backref="audit_logs")
    
    # Constraints
    __table_args__ = (
        Index('idx_audit_logs_event_type', 'event_type'),
        Index('idx_audit_logs_category_severity', 'event_category', 'severity'),
        Index('idx_audit_logs_timestamp', 'timestamp'),
        Index('idx_audit_logs_actor', 'actor_wallet_id'),
    )


# === Contributor Status & Proof-of-Contribution Models ===

class FTNSContributorStatus(Base):
    """Track contribution status and eligibility for FTNS earning"""
    __tablename__ = "ftns_contributor_status"
    
    # Primary identification  
    status_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Contributor status
    status = Column(String(50), nullable=False, default=ContributorTier.NONE.value, index=True)
    last_contribution_date = Column(DateTime(timezone=True), nullable=True, index=True)
    contribution_score = Column(DECIMAL(10, 4), nullable=False, default=Decimal('0.0'))
    
    # Active contribution tracking
    storage_provided_gb = Column(DECIMAL(15, 6), nullable=False, default=Decimal('0.0'))
    compute_hours_provided = Column(DECIMAL(15, 6), nullable=False, default=Decimal('0.0'))
    data_contributions_verified = Column(Integer, nullable=False, default=0)
    governance_votes_cast = Column(Integer, nullable=False, default=0)
    documentation_contributions = Column(Integer, nullable=False, default=0)
    model_contributions = Column(Integer, nullable=False, default=0)
    research_publications = Column(Integer, nullable=False, default=0)
    
    # Grace period management
    grace_period_expires = Column(DateTime(timezone=True), nullable=True)
    last_status_update = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Metadata and tracking
    contribution_history = Column(JSONB, nullable=True, default=dict)
    peer_validations = Column(JSONB, nullable=True, default=list)
    quality_scores = Column(JSONB, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True, onupdate=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('contribution_score >= 0', name='positive_contribution_score'),
        CheckConstraint('storage_provided_gb >= 0', name='positive_storage_provided'),
        CheckConstraint('compute_hours_provided >= 0', name='positive_compute_hours'),
        Index('idx_contributor_status_user', 'user_id'),
        Index('idx_contributor_status_tier', 'status'),
        Index('idx_contributor_last_contribution', 'last_contribution_date'),
        Index('idx_contributor_score', 'contribution_score'),
    )


class FTNSContributionProof(Base):
    """Cryptographic proofs of contribution for verification"""
    __tablename__ = "ftns_contribution_proofs"
    
    # Primary identification
    proof_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    contribution_type = Column(String(50), nullable=False, index=True)
    
    # Proof verification
    proof_hash = Column(String(255), nullable=False, unique=True)
    verification_timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    verification_status = Column(String(50), nullable=False, default=ProofStatus.PENDING.value, index=True)
    verified_by_peers = Column(JSONB, nullable=True, default=list)
    verification_confidence = Column(Float, nullable=False, default=0.0)
    
    # Contribution quantification
    contribution_value = Column(DECIMAL(20, 8), nullable=False)
    quality_score = Column(Float, nullable=False, default=0.0)
    impact_multiplier = Column(Float, nullable=False, default=1.0)
    
    # Proof data and metadata
    proof_data = Column(JSONB, nullable=False, default=dict)
    blockchain_hash = Column(String(255), nullable=True)  # For on-chain verification
    ipfs_hash = Column(String(255), nullable=True)        # For IPFS storage proofs
    
    # Validation details
    validation_criteria = Column(JSONB, nullable=True, default=dict)
    validation_results = Column(JSONB, nullable=True, default=dict)
    rejection_reason = Column(Text, nullable=True)
    
    # Expiration and lifecycle
    expires_at = Column(DateTime(timezone=True), nullable=True)
    submitted_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    contributor_status = relationship(
        "FTNSContributorStatus", 
        foreign_keys=[user_id],
        primaryjoin="FTNSContributionProof.user_id == FTNSContributorStatus.user_id",
        backref="contribution_proofs"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint('contribution_value >= 0', name='positive_contribution_value'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='valid_quality_score'),
        CheckConstraint('impact_multiplier >= 0.1 AND impact_multiplier <= 10', name='valid_impact_multiplier'),
        CheckConstraint('verification_confidence >= 0 AND verification_confidence <= 1', name='valid_verification_confidence'),
        Index('idx_proof_user_type', 'user_id', 'contribution_type'),
        Index('idx_proof_status', 'verification_status'),
        Index('idx_proof_timestamp', 'verification_timestamp'),
        Index('idx_proof_hash', 'proof_hash'),
        Index('idx_proof_expires', 'expires_at'),
    )


class FTNSContributionMetrics(Base):
    """Aggregated contribution metrics for analytics and rewards"""
    __tablename__ = "ftns_contribution_metrics"
    
    # Primary identification
    metric_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # daily, weekly, monthly, quarterly
    metric_period = Column(String(20), nullable=False, index=True)  # YYYY-MM-DD or YYYY-MM format
    
    # Contribution counts by type
    storage_contributions = Column(Integer, nullable=False, default=0)
    compute_contributions = Column(Integer, nullable=False, default=0)
    data_contributions = Column(Integer, nullable=False, default=0)
    governance_contributions = Column(Integer, nullable=False, default=0)
    documentation_contributions = Column(Integer, nullable=False, default=0)
    model_contributions = Column(Integer, nullable=False, default=0)
    research_contributions = Column(Integer, nullable=False, default=0)
    teaching_contributions = Column(Integer, nullable=False, default=0)
    
    # Quality and impact metrics
    average_quality_score = Column(Float, nullable=False, default=0.0)
    total_impact_score = Column(DECIMAL(15, 6), nullable=False, default=Decimal('0.0'))
    peer_validation_count = Column(Integer, nullable=False, default=0)
    unique_contribution_types = Column(Integer, nullable=False, default=0)
    
    # Rewards and recognition
    ftns_earned_period = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    ftns_earned_cumulative = Column(DECIMAL(20, 8), nullable=False, default=Decimal('0.0'))
    tier_advancement_count = Column(Integer, nullable=False, default=0)
    
    # Period metadata
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    calculated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('average_quality_score >= 0 AND average_quality_score <= 1', name='valid_avg_quality'),
        CheckConstraint('total_impact_score >= 0', name='positive_impact_score'),
        CheckConstraint('ftns_earned_period >= 0', name='positive_period_earnings'),
        CheckConstraint('ftns_earned_cumulative >= 0', name='positive_cumulative_earnings'),
        CheckConstraint('unique_contribution_types >= 0 AND unique_contribution_types <= 8', name='valid_contribution_types'),
        Index('idx_metrics_user_period', 'user_id', 'metric_period'),
        Index('idx_metrics_type_period', 'metric_type', 'metric_period'),
        Index('idx_metrics_calculated', 'calculated_at'),
        UniqueConstraint('user_id', 'metric_type', 'metric_period', name='unique_user_metric_period'),
    )


# === PHASE 2: DYNAMIC SUPPLY ADJUSTMENT MODELS ===

class FTNSSupplyAdjustment(Base):
    """Track supply adjustment decisions and applications"""
    __tablename__ = "ftns_supply_adjustments"
    
    # Primary identification
    adjustment_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Adjustment details
    adjustment_factor = Column(DECIMAL(8, 6), nullable=False)  # Multiplier for rates (e.g., 1.15 = 15% increase)
    trigger = Column(String(50), nullable=False, index=True)  # automated, governance, emergency, manual
    
    # Rate snapshots
    previous_rates = Column(JSONB, nullable=False)  # Previous reward rates
    new_rates = Column(JSONB, nullable=False)       # New reward rates after adjustment
    
    # Timing
    calculated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    applied_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status tracking
    status = Column(String(50), nullable=False, default=SupplyAdjustmentStatus.CALCULATED.value, index=True)
    
    # Economic context
    target_appreciation_rate = Column(DECIMAL(8, 6), nullable=True)  # Target rate at time of adjustment
    actual_appreciation_rate = Column(DECIMAL(8, 6), nullable=True)  # Actual rate at time of adjustment
    price_volatility = Column(DECIMAL(8, 6), nullable=True)          # Price volatility measure
    
    # Governance and approval
    approved_by = Column(String(255), nullable=True)  # Governance proposal ID or admin user
    approval_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata and audit
    adjustment_metadata = Column(JSONB, nullable=True)  # Additional context, calculations, etc.
    
    # Constraints
    __table_args__ = (
        CheckConstraint('adjustment_factor > 0', name='positive_adjustment_factor'),
        CheckConstraint('target_appreciation_rate >= 0', name='non_negative_target_rate'),
        Index('idx_adjustments_status', 'status'),
        Index('idx_adjustments_trigger', 'trigger'),
        Index('idx_adjustments_calculated', 'calculated_at'),
        Index('idx_adjustments_applied', 'applied_at'),
    )


class FTNSPriceMetrics(Base):
    """Historical price metrics for supply adjustment analysis"""
    __tablename__ = "ftns_price_metrics"
    
    # Primary identification
    metric_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Price data
    current_price = Column(DECIMAL(20, 8), nullable=True)
    target_appreciation_rate = Column(DECIMAL(8, 6), nullable=False)
    actual_appreciation_rate = Column(DECIMAL(8, 6), nullable=False)
    price_volatility = Column(DECIMAL(8, 6), nullable=False)
    
    # Analysis metrics
    rate_ratio = Column(DECIMAL(8, 6), nullable=False)  # actual / target rate ratio
    volatility_damping = Column(DECIMAL(5, 4), nullable=False)  # Volatility damping factor applied
    
    # Market data (if available)
    volume_24h = Column(DECIMAL(20, 8), nullable=True)
    market_cap = Column(DECIMAL(20, 8), nullable=True)
    
    # Timing and context
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    data_source = Column(String(100), nullable=False, default="price_oracle")
    
    # Metadata
    price_metadata = Column(JSONB, nullable=True)  # Calculation details, data quality indicators, etc.
    
    # Constraints
    __table_args__ = (
        CheckConstraint('target_appreciation_rate >= 0', name='non_negative_target_rate_metrics'),
        CheckConstraint('price_volatility >= 0', name='non_negative_volatility'),
        CheckConstraint('rate_ratio > 0', name='positive_rate_ratio'),
        CheckConstraint('volatility_damping >= 0 AND volatility_damping <= 1', name='valid_volatility_damping'),
        Index('idx_price_metrics_recorded', 'recorded_at'),
        Index('idx_price_metrics_source', 'data_source'),
    )


class FTNSRewardRates(Base):
    """Current and historical reward rates for all network activities"""
    __tablename__ = "ftns_reward_rates"
    
    # Primary identification
    rate_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Core reward rates
    context_cost_multiplier = Column(DECIMAL(8, 6), nullable=False, default=Decimal('1.0'))
    storage_reward_per_gb_hour = Column(DECIMAL(10, 8), nullable=False, default=Decimal('0.01'))
    compute_reward_per_unit = Column(DECIMAL(10, 8), nullable=False, default=Decimal('0.05'))
    data_contribution_base = Column(DECIMAL(10, 6), nullable=False, default=Decimal('10.0'))
    governance_participation = Column(DECIMAL(10, 6), nullable=False, default=Decimal('2.0'))
    documentation_reward = Column(DECIMAL(10, 6), nullable=False, default=Decimal('5.0'))
    
    # Advanced economic parameters
    staking_apy = Column(DECIMAL(6, 4), nullable=False, default=Decimal('0.08'))  # 8% default
    burn_rate_multiplier = Column(DECIMAL(6, 4), nullable=False, default=Decimal('1.0'))
    
    # Validity and versioning
    effective_date = Column(DateTime(timezone=True), nullable=False, default=func.now())
    deactivated_at = Column(DateTime(timezone=True), nullable=True)
    active = Column(Boolean, nullable=False, default=True, index=True)
    version = Column(String(50), nullable=False, default="1.0")
    
    # Governance and approval
    approved_by = Column(String(255), nullable=True)  # Governance proposal or admin
    adjustment_reference = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_supply_adjustments.adjustment_id'), nullable=True)
    
    # Metadata
    rate_metadata = Column(JSONB, nullable=True)  # Rate calculation context, external factors, etc.
    
    # Relationships
    adjustment = relationship("FTNSSupplyAdjustment", backref="rate_updates")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('context_cost_multiplier > 0', name='positive_context_multiplier'),
        CheckConstraint('storage_reward_per_gb_hour >= 0', name='non_negative_storage_reward'),
        CheckConstraint('compute_reward_per_unit >= 0', name='non_negative_compute_reward'),
        CheckConstraint('data_contribution_base >= 0', name='non_negative_data_reward'),
        CheckConstraint('governance_participation >= 0', name='non_negative_governance_reward'),
        CheckConstraint('documentation_reward >= 0', name='non_negative_doc_reward'),
        CheckConstraint('staking_apy >= 0 AND staking_apy <= 1', name='valid_staking_apy'),
        CheckConstraint('burn_rate_multiplier >= 0', name='non_negative_burn_rate'),
        Index('idx_reward_rates_active', 'active'),
        Index('idx_reward_rates_effective', 'effective_date'),
        Index('idx_reward_rates_version', 'version'),
    )


# === PHASE 3: ANTI-HOARDING MECHANISMS MODELS ===

class FTNSVelocityMetrics(Base):
    """Track token velocity metrics for users and network analysis"""
    __tablename__ = "ftns_velocity_metrics"
    
    # Primary identification
    metric_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Velocity measurements
    velocity = Column(DECIMAL(8, 4), nullable=False)  # Monthly velocity ratio
    transaction_volume = Column(DECIMAL(20, 8), nullable=False)  # Total transaction volume
    current_balance = Column(DECIMAL(20, 8), nullable=False)  # Balance at time of calculation
    velocity_category = Column(String(20), nullable=False, index=True)  # high, moderate, low, inactive
    
    # Calculation context
    calculation_period_days = Column(Integer, nullable=False)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    calculated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Additional metrics
    outgoing_transactions = Column(Integer, nullable=False, default=0)
    incoming_transactions = Column(Integer, nullable=False, default=0)
    unique_counterparties = Column(Integer, nullable=False, default=0)
    
    # Metadata
    calculation_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('velocity >= 0', name='non_negative_velocity'),
        CheckConstraint('transaction_volume >= 0', name='non_negative_volume'),
        CheckConstraint('current_balance >= 0', name='non_negative_balance'),
        CheckConstraint('calculation_period_days > 0', name='positive_calculation_period'),
        CheckConstraint('outgoing_transactions >= 0', name='non_negative_outgoing'),
        CheckConstraint('incoming_transactions >= 0', name='non_negative_incoming'),
        Index('idx_velocity_user_calculated', 'user_id', 'calculated_at'),
        Index('idx_velocity_category', 'velocity_category'),
        Index('idx_velocity_period', 'period_start', 'period_end'),
    )


class FTNSDemurrageRecord(Base):
    """Track demurrage fee applications and calculations"""
    __tablename__ = "ftns_demurrage_records"
    
    # Primary identification
    record_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Fee details
    fee_amount = Column(DECIMAL(20, 8), nullable=False)
    monthly_rate = Column(DECIMAL(6, 4), nullable=False)  # Monthly demurrage rate
    daily_rate = Column(DECIMAL(8, 6), nullable=False)    # Daily demurrage rate applied
    
    # Balance information
    balance_before = Column(DECIMAL(20, 8), nullable=False)
    balance_after = Column(DECIMAL(20, 8), nullable=False)
    
    # Context information
    velocity = Column(DECIMAL(8, 4), nullable=False)
    contributor_status = Column(String(50), nullable=False, index=True)
    velocity_category = Column(String(20), nullable=False, index=True)
    
    # Status and timing
    status = Column(String(20), nullable=False, default=DemurrageStatus.CALCULATED.value, index=True)
    calculated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    applied_at = Column(DateTime(timezone=True), nullable=True)
    
    # Grace period and exemptions
    grace_period_active = Column(Boolean, nullable=False, default=False)
    exemption_reason = Column(String(100), nullable=True)
    
    # Related transaction
    transaction_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_transactions.transaction_id'), nullable=True)
    
    # Metadata
    calculation_metadata = Column(JSONB, nullable=True)
    
    # Relationships
    transaction = relationship("FTNSTransaction", backref="demurrage_record")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('fee_amount >= 0', name='non_negative_fee'),
        CheckConstraint('monthly_rate >= 0 AND monthly_rate <= 1', name='valid_monthly_rate'),
        CheckConstraint('daily_rate >= 0 AND daily_rate <= 1', name='valid_daily_rate'),
        CheckConstraint('balance_before >= 0', name='non_negative_balance_before'),
        CheckConstraint('balance_after >= 0', name='non_negative_balance_after'),
        CheckConstraint('velocity >= 0', name='non_negative_velocity_demurrage'),
        Index('idx_demurrage_user_applied', 'user_id', 'applied_at'),
        Index('idx_demurrage_status', 'status'),
        Index('idx_demurrage_calculated', 'calculated_at'),
        Index('idx_demurrage_contributor', 'contributor_status'),
    )


class FTNSAntiHoardingConfig(Base):
    """Configuration parameters for anti-hoarding mechanisms"""
    __tablename__ = "ftns_anti_hoarding_config"
    
    # Primary identification
    config_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    config_name = Column(String(100), nullable=False, unique=True, index=True)
    
    # Velocity parameters
    target_velocity = Column(DECIMAL(6, 4), nullable=False, default=Decimal('1.2'))
    velocity_calculation_days = Column(Integer, nullable=False, default=30)
    
    # Demurrage parameters
    base_demurrage_rate = Column(DECIMAL(6, 4), nullable=False, default=Decimal('0.002'))  # 0.2% monthly
    max_demurrage_rate = Column(DECIMAL(6, 4), nullable=False, default=Decimal('0.01'))    # 1.0% monthly
    min_fee_threshold = Column(DECIMAL(10, 8), nullable=False, default=Decimal('0.001'))   # Minimum fee
    
    # Grace periods and exemptions
    grace_period_days = Column(Integer, nullable=False, default=90)
    contributor_exemption_enabled = Column(Boolean, nullable=False, default=True)
    
    # Velocity thresholds
    high_velocity_threshold = Column(DECIMAL(5, 4), nullable=False, default=Decimal('1.0'))    # 100% of target
    moderate_velocity_threshold = Column(DECIMAL(5, 4), nullable=False, default=Decimal('0.7'))  # 70% of target
    low_velocity_threshold = Column(DECIMAL(5, 4), nullable=False, default=Decimal('0.3'))      # 30% of target
    
    # Contributor status modifiers
    status_modifiers = Column(JSONB, nullable=False, default={
        "none": 1.5,        # 50% higher demurrage for non-contributors
        "basic": 1.0,       # Standard rate
        "active": 0.7,      # 30% reduction
        "power": 0.5        # 50% reduction
    })
    
    # Version and activation
    version = Column(String(20), nullable=False, default="1.0")
    active = Column(Boolean, nullable=False, default=True, index=True)
    effective_date = Column(DateTime(timezone=True), nullable=False, default=func.now())
    deactivated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Governance
    approved_by = Column(String(255), nullable=True)  # Governance proposal or admin
    approval_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    config_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('target_velocity > 0', name='positive_target_velocity'),
        CheckConstraint('velocity_calculation_days > 0', name='positive_velocity_calc_days'),
        CheckConstraint('base_demurrage_rate >= 0 AND base_demurrage_rate <= 1', name='valid_base_rate'),
        CheckConstraint('max_demurrage_rate >= 0 AND max_demurrage_rate <= 1', name='valid_max_rate'),
        CheckConstraint('max_demurrage_rate >= base_demurrage_rate', name='max_rate_above_base'),
        CheckConstraint('min_fee_threshold >= 0', name='non_negative_min_fee'),
        CheckConstraint('grace_period_days >= 0', name='non_negative_grace_period'),
        CheckConstraint('high_velocity_threshold > 0', name='positive_high_threshold'),
        CheckConstraint('moderate_velocity_threshold > 0', name='positive_moderate_threshold'),
        CheckConstraint('low_velocity_threshold >= 0', name='non_negative_low_threshold'),
        Index('idx_config_active', 'active'),
        Index('idx_config_effective', 'effective_date'),
        Index('idx_config_version', 'version'),
    )


class FTNSNetworkVelocitySnapshot(Base):
    """Periodic snapshots of network-wide velocity metrics"""
    __tablename__ = "ftns_network_velocity_snapshots"
    
    # Primary identification
    snapshot_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Network metrics
    network_velocity = Column(DECIMAL(8, 4), nullable=False)
    total_users = Column(Integer, nullable=False, default=0)
    total_balance = Column(DECIMAL(25, 8), nullable=False, default=Decimal('0'))
    total_transaction_volume = Column(DECIMAL(25, 8), nullable=False, default=Decimal('0'))
    
    # Velocity distribution
    high_velocity_users = Column(Integer, nullable=False, default=0)
    moderate_velocity_users = Column(Integer, nullable=False, default=0)
    low_velocity_users = Column(Integer, nullable=False, default=0)
    inactive_users = Column(Integer, nullable=False, default=0)
    
    # Health metrics
    health_score = Column(DECIMAL(5, 4), nullable=False)  # 0.0 to 1.0
    target_velocity = Column(DECIMAL(6, 4), nullable=False)
    velocity_target_ratio = Column(DECIMAL(6, 4), nullable=False)  # actual / target
    
    # Demurrage impact
    daily_demurrage_collected = Column(DECIMAL(20, 8), nullable=True)
    users_subject_to_demurrage = Column(Integer, nullable=False, default=0)
    average_demurrage_rate = Column(DECIMAL(6, 4), nullable=True)
    
    # Snapshot metadata
    calculation_period_days = Column(Integer, nullable=False)
    snapshot_date = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Additional metrics
    snapshot_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('network_velocity >= 0', name='non_negative_network_velocity'),
        CheckConstraint('total_users >= 0', name='non_negative_total_users'),
        CheckConstraint('total_balance >= 0', name='non_negative_total_balance'),
        CheckConstraint('health_score >= 0 AND health_score <= 1', name='valid_health_score'),
        CheckConstraint('target_velocity > 0', name='positive_target_velocity_snapshot'),
        CheckConstraint('velocity_target_ratio >= 0', name='non_negative_velocity_ratio'),
        CheckConstraint('calculation_period_days > 0', name='positive_calc_period_snapshot'),
        Index('idx_snapshot_date', 'snapshot_date'),
        Index('idx_snapshot_health', 'health_score'),
        Index('idx_snapshot_velocity', 'network_velocity'),
    )


# === PHASE 4: EMERGENCY CIRCUIT BREAKERS MODELS ===

class FTNSEmergencyTrigger(Base):
    """Track emergency triggers and conditions for circuit breakers"""
    __tablename__ = "ftns_emergency_triggers"
    
    # Primary identification
    trigger_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    trigger_type = Column(String(50), nullable=False, index=True)  # price_crash, volume_spike, etc.
    
    # Trigger conditions
    threshold_value = Column(DECIMAL(10, 6), nullable=False)  # Threshold that triggered emergency
    actual_value = Column(DECIMAL(10, 6), nullable=False)     # Actual value that breached threshold
    confidence_score = Column(DECIMAL(5, 4), nullable=False)  # Confidence in trigger (0.0-1.0)
    severity_level = Column(String(20), nullable=False)       # low, medium, high, critical
    
    # Detection context
    data_source = Column(String(100), nullable=False)         # Source that detected the trigger
    detection_algorithm = Column(String(100), nullable=False)  # Algorithm used for detection
    time_window_seconds = Column(Integer, nullable=False)      # Time window for trigger evaluation
    
    # Status and timing
    status = Column(String(20), nullable=False, default=EmergencyStatus.DETECTED.value, index=True)
    detected_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Emergency response
    auto_response_enabled = Column(Boolean, nullable=False, default=True)
    governance_required = Column(Boolean, nullable=False, default=False)
    response_delay_seconds = Column(Integer, nullable=False, default=0)
    
    # Additional data
    trigger_metadata = Column(JSONB, nullable=True)  # Raw trigger data, market conditions, etc.
    
    # Constraints
    __table_args__ = (
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='valid_confidence_score'),
        CheckConstraint('time_window_seconds > 0', name='positive_time_window'),
        CheckConstraint('response_delay_seconds >= 0', name='non_negative_response_delay'),
        Index('idx_emergency_triggers_type_status', 'trigger_type', 'status'),
        Index('idx_emergency_triggers_detected', 'detected_at'),
        Index('idx_emergency_triggers_severity', 'severity_level'),
    )


class FTNSEmergencyAction(Base):
    """Track emergency response actions taken by the system"""
    __tablename__ = "ftns_emergency_actions"
    
    # Primary identification
    action_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    trigger_id = Column(PG_UUID(as_uuid=True), ForeignKey('ftns_emergency_triggers.trigger_id'), nullable=False)
    action_type = Column(String(50), nullable=False, index=True)  # halt_transactions, reduce_limits, etc.
    
    # Action parameters
    action_severity = Column(String(20), nullable=False)       # mild, moderate, severe, extreme
    target_scope = Column(String(50), nullable=False)          # all_users, specific_users, system_wide
    affected_users = Column(JSONB, nullable=True)              # List of affected user IDs
    duration_seconds = Column(Integer, nullable=True)          # Action duration (null = indefinite)
    
    # Action specifics
    transaction_halt_types = Column(JSONB, nullable=True)      # Which transaction types to halt
    rate_adjustments = Column(JSONB, nullable=True)           # Rate adjustment parameters
    limit_reductions = Column(JSONB, nullable=True)           # Limit reduction parameters
    
    # Execution tracking
    status = Column(String(20), nullable=False, default="pending", index=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    executed_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    reverted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Authorization
    authorized_by = Column(String(255), nullable=True)        # User/system that authorized action
    governance_proposal_id = Column(String(255), nullable=True)  # Associated governance proposal
    override_reason = Column(Text, nullable=True)             # Reason for manual override
    
    # Results and impact
    users_affected = Column(Integer, nullable=False, default=0)
    transactions_halted = Column(Integer, nullable=False, default=0)
    volume_impact = Column(DECIMAL(20, 8), nullable=True)     # Volume of transactions affected
    
    # Metadata
    action_metadata = Column(JSONB, nullable=True)            # Action-specific data and results
    
    # Relationships
    trigger = relationship("FTNSEmergencyTrigger", backref="emergency_actions")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('duration_seconds IS NULL OR duration_seconds > 0', name='positive_duration'),
        CheckConstraint('users_affected >= 0', name='non_negative_users_affected'),
        CheckConstraint('transactions_halted >= 0', name='non_negative_transactions_halted'),
        Index('idx_emergency_actions_trigger', 'trigger_id'),
        Index('idx_emergency_actions_type', 'action_type'),
        Index('idx_emergency_actions_status', 'status'),
        Index('idx_emergency_actions_executed', 'executed_at'),
    )


class FTNSEmergencyConfig(Base):
    """Configuration parameters for emergency detection and response"""
    __tablename__ = "ftns_emergency_config"
    
    # Primary identification
    config_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    config_name = Column(String(100), nullable=False, unique=True, index=True)
    config_version = Column(String(20), nullable=False, default="1.0")
    
    # Trigger thresholds
    price_crash_threshold = Column(DECIMAL(5, 4), nullable=False, default=Decimal('0.4'))  # 40% drop
    volume_spike_threshold = Column(DECIMAL(6, 2), nullable=False, default=Decimal('500.0'))  # 5x normal
    oracle_deviation_threshold = Column(DECIMAL(5, 4), nullable=False, default=Decimal('0.1'))  # 10% deviation
    system_error_threshold = Column(Integer, nullable=False, default=10)  # Error count threshold
    
    # Detection parameters
    price_monitoring_window_minutes = Column(Integer, nullable=False, default=60)
    volume_monitoring_window_minutes = Column(Integer, nullable=False, default=30)
    confidence_threshold = Column(DECIMAL(5, 4), nullable=False, default=Decimal('0.8'))  # 80% confidence
    
    # Response parameters
    auto_response_enabled = Column(Boolean, nullable=False, default=True)
    max_auto_halt_duration_hours = Column(Integer, nullable=False, default=24)
    governance_escalation_threshold = Column(DECIMAL(5, 4), nullable=False, default=Decimal('0.9'))
    emergency_contact_list = Column(JSONB, nullable=True)
    
    # Rate limits and restrictions
    max_transaction_amount_emergency = Column(DECIMAL(20, 8), nullable=True)
    max_daily_volume_emergency = Column(DECIMAL(25, 8), nullable=True)
    restricted_transaction_types = Column(JSONB, nullable=True)
    
    # Governance integration
    governance_voting_window_hours = Column(Integer, nullable=False, default=6)  # Fast-track voting
    governance_activation_threshold = Column(DECIMAL(5, 4), nullable=False, default=Decimal('0.15'))  # 15%
    governance_override_enabled = Column(Boolean, nullable=False, default=True)
    
    # Configuration lifecycle
    active = Column(Boolean, nullable=False, default=True, index=True)
    effective_date = Column(DateTime(timezone=True), nullable=False, default=func.now())
    deactivated_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String(255), nullable=False)
    approved_by = Column(String(255), nullable=True)
    
    # Metadata
    config_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('price_crash_threshold > 0 AND price_crash_threshold <= 1', name='valid_price_crash_threshold'),
        CheckConstraint('volume_spike_threshold > 1', name='valid_volume_spike_threshold'),
        CheckConstraint('confidence_threshold > 0 AND confidence_threshold <= 1', name='valid_confidence_threshold'),
        CheckConstraint('governance_activation_threshold > 0 AND governance_activation_threshold <= 1', name='valid_governance_threshold'),
        CheckConstraint('price_monitoring_window_minutes > 0', name='positive_price_window'),
        CheckConstraint('volume_monitoring_window_minutes > 0', name='positive_volume_window'),
        CheckConstraint('max_auto_halt_duration_hours > 0', name='positive_halt_duration'),
        CheckConstraint('governance_voting_window_hours > 0', name='positive_voting_window'),
        Index('idx_emergency_config_active', 'active'),
        Index('idx_emergency_config_effective', 'effective_date'),
        Index('idx_emergency_config_name', 'config_name'),
    )
"""
PRSM Payment Processing Models
=============================

Comprehensive data models for payment processing, fiat-to-crypto conversion,
and financial transaction management with enterprise compliance features.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime, Text, Integer, Numeric, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..core.database import Base


class PaymentMethod(str, Enum):
    """Supported payment methods"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    ACH = "ach"
    SEPA = "sepa"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    CRYPTO_WALLET = "crypto_wallet"


class PaymentStatus(str, Enum):
    """Payment transaction status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    EXPIRED = "expired"


class FiatCurrency(str, Enum):
    """Supported fiat currencies"""
    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    JPY = "JPY"  # Japanese Yen
    CAD = "CAD"  # Canadian Dollar
    AUD = "AUD"  # Australian Dollar
    CHF = "CHF"  # Swiss Franc
    CNY = "CNY"  # Chinese Yuan
    KRW = "KRW"  # South Korean Won
    INR = "INR"  # Indian Rupee


class CryptoCurrency(str, Enum):
    """Supported cryptocurrencies"""
    MATIC = "MATIC"    # Polygon
    ETH = "ETH"        # Ethereum
    USDC = "USDC"      # USD Coin
    USDT = "USDT"      # Tether
    BTC = "BTC"        # Bitcoin
    FTNS = "FTNS"      # PRSM FTNS Token


class KYCStatus(str, Enum):
    """KYC verification status"""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


# === Database Models ===

class PaymentTransaction(Base):
    """Payment transaction database model"""
    __tablename__ = "payment_transactions"
    
    # Primary identifiers
    transaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String(255), unique=True, nullable=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Transaction details
    payment_method = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default=PaymentStatus.PENDING.value)
    
    # Financial amounts
    fiat_amount = Column(Numeric(20, 8), nullable=False)
    fiat_currency = Column(String(10), nullable=False)
    crypto_amount = Column(Numeric(30, 18), nullable=True)
    crypto_currency = Column(String(20), nullable=False)
    
    # Exchange and fees
    exchange_rate = Column(Numeric(30, 18), nullable=True)
    processing_fee = Column(Numeric(20, 8), nullable=False, default=0)
    network_fee = Column(Numeric(30, 18), nullable=True)
    
    # Transaction metadata
    provider = Column(String(100), nullable=False)
    provider_reference = Column(String(255), nullable=True)
    blockchain_tx_hash = Column(String(255), nullable=True)
    
    # Compliance and verification
    kyc_status = Column(String(50), default=KYCStatus.NOT_REQUIRED.value)
    aml_checked = Column(Boolean, default=False)
    risk_score = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional data
    additional_data = Column(JSONB, default=dict)
    failure_reason = Column(Text, nullable=True)
    refund_transaction_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Relationships
    payment_methods = relationship("UserPaymentMethod", back_populates="transactions")


class UserPaymentMethod(Base):
    """User payment method database model"""
    __tablename__ = "user_payment_methods"
    
    # Primary identifiers
    method_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Payment method details
    payment_type = Column(String(50), nullable=False)
    provider = Column(String(100), nullable=False)
    provider_method_id = Column(String(255), nullable=False)
    
    # Method information (encrypted/tokenized)
    display_name = Column(String(255), nullable=False)
    last_four = Column(String(10), nullable=True)
    expiry_month = Column(Integer, nullable=True)
    expiry_year = Column(Integer, nullable=True)
    
    # Status and verification
    is_verified = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional data
    additional_data = Column(JSONB, default=dict)
    
    # Relationships
    transactions = relationship("PaymentTransaction", back_populates="payment_methods")


class ExchangeRateHistory(Base):
    """Exchange rate history database model"""
    __tablename__ = "exchange_rate_history"
    
    # Primary identifiers
    rate_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Currency pair
    from_currency = Column(String(20), nullable=False)
    to_currency = Column(String(20), nullable=False)
    
    # Rate information
    rate = Column(Numeric(30, 18), nullable=False)
    source = Column(String(100), nullable=False)
    volume_24h = Column(Numeric(30, 18), nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Additional data
    additional_data = Column(JSONB, default=dict)


# === Pydantic Models ===

class PaymentRequest(BaseModel):
    """Payment request model"""
    user_id: str = Field(description="User ID making the payment")
    fiat_amount: Decimal = Field(description="Amount in fiat currency", gt=0)
    fiat_currency: FiatCurrency = Field(description="Fiat currency")
    crypto_currency: CryptoCurrency = Field(description="Target cryptocurrency")
    payment_method: PaymentMethod = Field(description="Payment method")
    payment_method_id: Optional[str] = Field(default=None, description="Saved payment method ID")
    
    # Optional parameters
    return_url: Optional[str] = Field(default=None, description="Return URL after payment")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for status updates")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('fiat_amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class PaymentResponse(BaseModel):
    """Payment response model"""
    success: bool
    transaction_id: str
    status: PaymentStatus
    
    # Financial details
    fiat_amount: Decimal
    fiat_currency: FiatCurrency
    crypto_amount: Optional[Decimal] = None
    crypto_currency: CryptoCurrency
    exchange_rate: Optional[Decimal] = None
    
    # Processing details
    processing_fee: Decimal = Field(default=Decimal("0"))
    network_fee: Optional[Decimal] = None
    estimated_completion: Optional[datetime] = None
    
    # URLs and references
    payment_url: Optional[str] = None
    provider_reference: Optional[str] = None
    
    # Status information
    message: str
    next_action: Optional[str] = None
    requires_action: bool = False
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


class ExchangeRate(BaseModel):
    """Exchange rate model"""
    from_currency: str
    to_currency: str
    rate: Decimal
    source: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional market data
    volume_24h: Optional[Decimal] = None
    price_change_24h: Optional[Decimal] = None
    market_cap: Optional[Decimal] = None


class PaymentMethodRequest(BaseModel):
    """Payment method registration request"""
    payment_type: PaymentMethod
    
    # For card payments
    card_number: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    cvv: Optional[str] = None
    cardholder_name: Optional[str] = None
    
    # For bank transfers
    account_number: Optional[str] = None
    routing_number: Optional[str] = None
    account_holder_name: Optional[str] = None
    
    # General
    billing_address: Optional[Dict[str, str]] = None
    is_default: bool = False
    additional_data: Dict[str, Any] = Field(default_factory=dict)


class PaymentMethodResponse(BaseModel):
    """Payment method response"""
    success: bool
    method_id: Optional[str] = None
    display_name: Optional[str] = None
    last_four: Optional[str] = None
    is_verified: bool = False
    requires_verification: bool = False
    verification_url: Optional[str] = None
    message: str


class TransactionQuery(BaseModel):
    """Transaction query parameters"""
    user_id: Optional[str] = None
    status: Optional[PaymentStatus] = None
    payment_method: Optional[PaymentMethod] = None
    fiat_currency: Optional[FiatCurrency] = None
    crypto_currency: Optional[CryptoCurrency] = None
    
    # Date range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Pagination
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=50, ge=1, le=100)
    
    # Sorting
    sort_by: str = Field(default="created_at")
    sort_order: str = Field(default="desc")


class TransactionList(BaseModel):
    """Transaction list response"""
    transactions: List[PaymentResponse]
    total_count: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_previous: bool


class PaymentStats(BaseModel):
    """Payment statistics"""
    total_transactions: int
    total_volume_usd: Decimal
    successful_transactions: int
    failed_transactions: int
    average_transaction_size: Decimal
    
    # By currency
    volume_by_fiat_currency: Dict[str, Decimal]
    volume_by_crypto_currency: Dict[str, Decimal]
    
    # By payment method
    transactions_by_method: Dict[str, int]
    
    # Time period
    period_start: datetime
    period_end: datetime
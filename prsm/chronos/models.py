"""
CHRONOS Data Models

Core data structures for clearing protocol operations.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid


class AssetType(str, Enum):
    """Supported asset types for clearing operations."""
    FTNS = "FTNS"
    BTC = "BTC"
    USD = "USD"
    ETH = "ETH"


class SwapType(str, Enum):
    """Types of swap operations."""
    FTNS_TO_BTC = "FTNS_TO_BTC"
    BTC_TO_FTNS = "BTC_TO_FTNS"
    FTNS_TO_USD = "FTNS_TO_USD"
    USD_TO_FTNS = "USD_TO_FTNS"
    BTC_TO_USD = "BTC_TO_USD"
    USD_TO_BTC = "USD_TO_BTC"


class TransactionStatus(str, Enum):
    """Transaction processing states."""
    PENDING = "PENDING"
    VERIFYING = "VERIFYING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SwapRequest(BaseModel):
    """Request for asset swap through CHRONOS."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    from_asset: AssetType
    to_asset: AssetType
    from_amount: Decimal
    to_amount: Optional[Decimal] = None  # Calculated by pricing engine
    swap_type: SwapType
    max_slippage: Decimal = Field(default=Decimal("0.005"))  # 0.5% default
    expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Settlement(BaseModel):
    """Settlement record for completed transactions."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    swap_request_id: str
    from_asset: AssetType
    to_asset: AssetType
    from_amount: Decimal
    to_amount: Decimal
    exchange_rate: Decimal
    fees: Dict[str, Decimal]  # Fee breakdown by type
    total_fees: Decimal
    net_amount: Decimal  # Final amount after fees
    settlement_hash: str  # IPFS provenance hash
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class ClearingTransaction(BaseModel):
    """Complete transaction record through CHRONOS."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    swap_request: SwapRequest
    status: TransactionStatus = TransactionStatus.PENDING
    settlement: Optional[Settlement] = None
    exchange_route: Optional[List[str]] = None  # Exchanges used
    error_message: Optional[str] = None
    blockchain_txids: Dict[str, str] = Field(default_factory=dict)  # Chain -> TxID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


@dataclass
class LiquidityPool:
    """Liquidity pool for asset pairs."""
    
    asset_a: AssetType
    asset_b: AssetType
    reserve_a: Decimal
    reserve_b: Decimal
    total_liquidity: Decimal
    fee_rate: Decimal
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def get_exchange_rate(self, from_asset: AssetType) -> Decimal:
        """Calculate current exchange rate."""
        if from_asset == self.asset_a:
            return self.reserve_b / self.reserve_a
        elif from_asset == self.asset_b:
            return self.reserve_a / self.reserve_b
        else:
            raise ValueError(f"Asset {from_asset} not in pool")
    
    def calculate_output(self, from_asset: AssetType, amount: Decimal) -> Decimal:
        """Calculate output amount with slippage."""
        if from_asset == self.asset_a:
            # Using constant product formula with fees
            amount_with_fee = amount * (Decimal("1") - self.fee_rate)
            numerator = amount_with_fee * self.reserve_b
            denominator = self.reserve_a + amount_with_fee
            return numerator / denominator
        elif from_asset == self.asset_b:
            amount_with_fee = amount * (Decimal("1") - self.fee_rate)
            numerator = amount_with_fee * self.reserve_a
            denominator = self.reserve_b + amount_with_fee
            return numerator / denominator
        else:
            raise ValueError(f"Asset {from_asset} not in pool")


@dataclass
class ExchangeConfig:
    """Configuration for external exchange integration."""
    
    name: str
    api_key: str
    api_secret: str
    sandbox_mode: bool
    supported_pairs: List[tuple[AssetType, AssetType]]
    fee_rates: Dict[str, Decimal]
    rate_limits: Dict[str, int]
    is_active: bool = True


class StakingProgramStatus(str, Enum):
    """Staking program lifecycle states."""
    PENDING_APPROVAL = "PENDING_APPROVAL"
    ACTIVE = "ACTIVE"
    AUCTION_PHASE = "AUCTION_PHASE"
    FUNDED = "FUNDED"
    MATURED = "MATURED"
    DEFAULTED = "DEFAULTED"
    CANCELLED = "CANCELLED"


class StakingProgram(BaseModel):
    """Staking program issued by companies on PRSM network."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    issuer_id: str  # Company/organization identifier
    issuer_name: str
    program_name: str
    description: str
    
    # Financial terms
    target_raise: Decimal  # Total FTNS to raise
    min_stake: Decimal  # Minimum individual stake
    max_stake: Optional[Decimal] = None  # Maximum individual stake
    duration_months: int  # Lock-up period
    base_apy: Decimal  # Base APY offered
    risk_profile: str  # conservative, growth, moonshot
    
    # Auction parameters
    auction_start: datetime
    auction_end: datetime
    auction_reserve_apy: Decimal  # Minimum APY issuer will accept
    
    # Collateral and guarantees
    collateral_amount: Decimal
    collateral_asset: AssetType
    insurance_coverage: Optional[Decimal] = None
    
    # Program status
    status: StakingProgramStatus = StakingProgramStatus.PENDING_APPROVAL
    total_staked: Decimal = Field(default=Decimal("0"))
    current_apy: Optional[Decimal] = None  # Final auction-determined APY
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    funded_at: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class StakePosition(BaseModel):
    """Individual stake position in a program."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    program_id: str
    staker_address: str  # Wallet address of staker
    
    # Position details
    principal_amount: Decimal  # Original FTNS staked
    guaranteed_apy: Decimal  # Locked-in APY rate
    currency_preference: AssetType = AssetType.FTNS  # Preferred payout currency
    
    # Position lifecycle
    staked_at: datetime = Field(default_factory=datetime.utcnow)
    maturity_timestamp: datetime
    is_transferable: bool = True
    current_owner: str  # Current position holder (for secondary market)
    
    # Returns tracking
    accrued_interest: Decimal = Field(default=Decimal("0"))
    last_interest_update: datetime = Field(default_factory=datetime.utcnow)
    payments_received: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class StakingAuction(BaseModel):
    """Auction for staking program funding."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    program_id: str
    
    # Auction parameters
    start_time: datetime
    end_time: datetime
    min_bid_apy: Decimal  # Minimum APY bidders can offer
    max_bid_apy: Decimal  # Maximum APY allowed
    
    # Current state
    total_bids: Decimal = Field(default=Decimal("0"))
    highest_apy: Decimal = Field(default=Decimal("0"))
    lowest_apy: Decimal = Field(default=Decimal("100"))  # Will decrease
    
    # Results
    winning_apy: Optional[Decimal] = None
    total_funded: Optional[Decimal] = None
    is_successful: Optional[bool] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class StakingBid(BaseModel):
    """Individual bid in staking auction."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    auction_id: str
    bidder_address: str
    
    # Bid details
    stake_amount: Decimal  # FTNS amount willing to stake
    bid_apy: Decimal  # APY rate bidder is willing to accept
    currency_preference: AssetType = AssetType.FTNS
    
    # Bid status
    is_winning: bool = False
    is_filled: bool = False
    fill_amount: Decimal = Field(default=Decimal("0"))
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class CHRONOSStakingRequest(BaseModel):
    """Request to stake in preferred currency (converted via CHRONOS)."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    program_id: str
    staker_address: str
    
    # Multi-currency staking
    stake_amount: Decimal
    stake_currency: AssetType  # Currency user wants to stake in
    target_currency: AssetType = AssetType.FTNS  # Always FTNS for programs
    
    # Conversion details
    max_slippage: Decimal = Field(default=Decimal("0.01"))  # 1%
    conversion_quote: Optional[Dict[str, Any]] = None
    swap_transaction_id: Optional[str] = None
    
    # Resulting stake position
    final_ftns_amount: Optional[Decimal] = None
    stake_position_id: Optional[str] = None
    
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }
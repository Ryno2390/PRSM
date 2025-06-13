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
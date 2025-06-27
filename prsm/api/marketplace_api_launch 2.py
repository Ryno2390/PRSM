#!/usr/bin/env python3
"""
PRSM FTNS Marketplace Launch API
===============================

RESTful API for the real-money FTNS token marketplace, enabling:
- Token purchases with fiat and cryptocurrency
- Staking and yield generation
- Liquidity provision and rewards
- Portfolio management
- Market analytics and reporting

🎯 MARKETPLACE API FEATURES:
✅ Fiat and crypto token purchases
✅ Staking position management
✅ Liquidity pool operations
✅ Real-time market data
✅ Portfolio analytics
✅ Transaction history
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime

from ..marketplace.ftns_marketplace import (
    FTNSMarketplace, PaymentMethod, TransactionType,
    get_ftns_marketplace
)
from ..auth import get_current_user

router = APIRouter(prefix="/marketplace", tags=["FTNS Marketplace"])
security = HTTPBearer()


# Request/Response Models
class FiatPurchaseRequest(BaseModel):
    """Fiat currency purchase request"""
    amount_usd: Decimal
    payment_method: str
    payment_details: Dict[str, Any]
    
    @validator('amount_usd')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v < Decimal('10'):
            raise ValueError("Minimum purchase amount is $10")
        if v > Decimal('100000'):
            raise ValueError("Maximum purchase amount is $100,000")
        return v
    
    @validator('payment_method')
    def validate_payment_method(cls, v):
        valid_methods = [pm.value for pm in PaymentMethod if pm in [
            PaymentMethod.CREDIT_CARD, PaymentMethod.PAYPAL, PaymentMethod.BANK_TRANSFER
        ]]
        if v not in valid_methods:
            raise ValueError(f"Payment method must be one of: {valid_methods}")
        return v


class CryptoPurchaseRequest(BaseModel):
    """Cryptocurrency purchase request"""
    crypto_amount: Decimal
    crypto_symbol: str
    wallet_address: str
    
    @validator('crypto_amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v
    
    @validator('crypto_symbol')
    def validate_crypto(cls, v):
        valid_cryptos = ['BTC', 'ETH', 'USDC', 'USDT']
        if v.upper() not in valid_cryptos:
            raise ValueError(f"Supported cryptocurrencies: {valid_cryptos}")
        return v.upper()


class StakingRequest(BaseModel):
    """Staking request"""
    amount: Decimal
    duration_days: int
    auto_compound: bool = True
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Staking amount must be positive")
        if v < Decimal('100'):
            raise ValueError("Minimum staking amount is 100 FTNS")
        return v
    
    @validator('duration_days')
    def validate_duration(cls, v):
        valid_durations = [30, 90, 180, 365]
        if v not in valid_durations:
            raise ValueError(f"Valid staking durations: {valid_durations} days")
        return v


class LiquidityRequest(BaseModel):
    """Liquidity provision request"""
    ftns_amount: Decimal
    paired_token_amount: Decimal
    paired_token_symbol: str
    
    @validator('ftns_amount')
    def validate_ftns_amount(cls, v):
        if v <= 0:
            raise ValueError("FTNS amount must be positive")
        if v < Decimal('50'):
            raise ValueError("Minimum liquidity provision is 50 FTNS")
        return v
    
    @validator('paired_token_symbol')
    def validate_paired_token(cls, v):
        valid_tokens = ['USDC', 'USDT', 'ETH', 'BTC']
        if v.upper() not in valid_tokens:
            raise ValueError(f"Supported paired tokens: {valid_tokens}")
        return v.upper()


class PurchaseResponse(BaseModel):
    """Purchase response"""
    success: bool
    transaction_id: str
    amount_ftns: Decimal
    exchange_rate: Decimal
    fees: Dict[str, Decimal]
    estimated_completion: str
    message: str


class MarketDataResponse(BaseModel):
    """Market data response"""
    ftns_price_usd: Decimal
    volume_24h: Decimal
    market_cap: Decimal
    circulating_supply: Decimal
    staking_apy: Decimal
    liquidity_pool_size: Decimal
    price_change_24h: Decimal
    updated_at: datetime


# Market Data Endpoints
@router.get("/market-data", response_model=MarketDataResponse)
async def get_market_data(
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Get current FTNS market data"""
    
    market_data = await marketplace.get_current_market_data()
    
    return MarketDataResponse(
        ftns_price_usd=market_data.ftns_price_usd,
        volume_24h=market_data.volume_24h,
        market_cap=market_data.market_cap,
        circulating_supply=market_data.circulating_supply,
        staking_apy=market_data.staking_apy,
        liquidity_pool_size=market_data.liquidity_pool_size,
        price_change_24h=market_data.price_change_24h,
        updated_at=market_data.updated_at
    )


@router.get("/analytics")
async def get_marketplace_analytics(
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Get comprehensive marketplace analytics"""
    
    analytics = await marketplace.get_marketplace_analytics()
    return analytics


# Purchase Endpoints
@router.post("/purchase/fiat", response_model=PurchaseResponse)
async def purchase_ftns_with_fiat(
    request: FiatPurchaseRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Purchase FTNS tokens with fiat currency"""
    
    try:
        transaction_id = await marketplace.purchase_ftns_fiat(
            user_id=current_user["user_id"],
            amount_usd=request.amount_usd,
            payment_method=PaymentMethod(request.payment_method),
            payment_details=request.payment_details
        )
        
        # Get transaction details
        transaction = marketplace.transactions[transaction_id]
        
        return PurchaseResponse(
            success=True,
            transaction_id=transaction_id,
            amount_ftns=transaction.amount_ftns,
            exchange_rate=transaction.exchange_rate,
            fees=transaction.fees,
            estimated_completion="2-5 minutes",
            message="Purchase initiated successfully. FTNS tokens will be added to your account upon payment confirmation."
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/purchase/crypto", response_model=PurchaseResponse)
async def purchase_ftns_with_crypto(
    request: CryptoPurchaseRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Purchase FTNS tokens with cryptocurrency"""
    
    try:
        transaction_id = await marketplace.purchase_ftns_crypto(
            user_id=current_user["user_id"],
            crypto_amount=request.crypto_amount,
            crypto_symbol=request.crypto_symbol,
            wallet_address=request.wallet_address
        )
        
        transaction = marketplace.transactions[transaction_id]
        
        return PurchaseResponse(
            success=True,
            transaction_id=transaction_id,
            amount_ftns=transaction.amount_ftns,
            exchange_rate=transaction.exchange_rate,
            fees=transaction.fees,
            estimated_completion="10-30 minutes",
            message=f"Crypto purchase initiated. Please send {request.crypto_amount} {request.crypto_symbol} to the provided address."
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Staking Endpoints
@router.post("/stake")
async def stake_ftns_tokens(
    request: StakingRequest,
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Stake FTNS tokens for rewards"""
    
    try:
        position_id = await marketplace.stake_ftns(
            user_id=current_user["user_id"],
            amount=request.amount,
            duration_days=request.duration_days,
            auto_compound=request.auto_compound
        )
        
        position = marketplace.staking_positions[position_id]
        
        return {
            "success": True,
            "position_id": position_id,
            "staked_amount": float(position.staked_amount),
            "duration_days": position.stake_duration_days,
            "annual_yield_rate": float(position.annual_yield_rate),
            "end_date": position.end_date.isoformat(),
            "estimated_rewards": float(position.staked_amount * position.annual_yield_rate * Decimal(str(position.stake_duration_days)) / Decimal('365')),
            "auto_compound": position.auto_compound,
            "message": f"Successfully staked {request.amount} FTNS for {request.duration_days} days"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/unstake/{position_id}")
async def unstake_ftns_tokens(
    position_id: str,
    force: bool = False,
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Unstake FTNS tokens and claim rewards"""
    
    try:
        principal, rewards = await marketplace.unstake_ftns(
            user_id=current_user["user_id"],
            position_id=position_id,
            force=force
        )
        
        return {
            "success": True,
            "principal_returned": float(principal),
            "rewards_earned": float(rewards),
            "total_returned": float(principal + rewards),
            "message": f"Successfully unstaked {principal} FTNS and earned {rewards} FTNS in rewards"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/staking/positions")
async def get_staking_positions(
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Get user's staking positions"""
    
    user_positions = [
        {
            "position_id": pos.position_id,
            "staked_amount": float(pos.staked_amount),
            "duration_days": pos.stake_duration_days,
            "annual_yield_rate": float(pos.annual_yield_rate),
            "start_date": pos.start_date.isoformat(),
            "end_date": pos.end_date.isoformat(),
            "days_remaining": (pos.end_date - datetime.now()).days,
            "auto_compound": pos.auto_compound,
            "current_rewards": float(
                pos.staked_amount * pos.annual_yield_rate * 
                Decimal(str((datetime.now() - pos.start_date).days)) / Decimal('365')
            )
        }
        for pos in marketplace.staking_positions.values()
        if pos.user_id == current_user["user_id"]
    ]
    
    return {
        "positions": user_positions,
        "total_staked": sum(pos["staked_amount"] for pos in user_positions),
        "total_pending_rewards": sum(pos["current_rewards"] for pos in user_positions)
    }


# Liquidity Endpoints
@router.post("/liquidity/add")
async def add_liquidity(
    request: LiquidityRequest,
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Add liquidity to FTNS trading pool"""
    
    try:
        position_id = await marketplace.add_liquidity(
            user_id=current_user["user_id"],
            ftns_amount=request.ftns_amount,
            paired_token_amount=request.paired_token_amount,
            paired_token_symbol=request.paired_token_symbol
        )
        
        position = marketplace.liquidity_positions[position_id]
        
        return {
            "success": True,
            "position_id": position_id,
            "ftns_amount": float(position.ftns_amount),
            "paired_amount": float(position.paired_token_amount),
            "paired_symbol": position.paired_token_symbol,
            "share_percentage": float(position.share_percentage),
            "message": f"Successfully added liquidity: {request.ftns_amount} FTNS + {request.paired_token_amount} {request.paired_token_symbol}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/liquidity/positions")
async def get_liquidity_positions(
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Get user's liquidity positions"""
    
    user_positions = [
        {
            "position_id": pos.position_id,
            "ftns_amount": float(pos.ftns_amount),
            "paired_amount": float(pos.paired_token_amount),
            "paired_symbol": pos.paired_token_symbol,
            "share_percentage": float(pos.share_percentage),
            "fees_earned": float(pos.fees_earned),
            "created_at": pos.created_at.isoformat()
        }
        for pos in marketplace.liquidity_positions.values()
        if pos.user_id == current_user["user_id"]
    ]
    
    return {
        "positions": user_positions,
        "total_liquidity_ftns": sum(pos["ftns_amount"] for pos in user_positions),
        "total_fees_earned": sum(pos["fees_earned"] for pos in user_positions)
    }


# Portfolio Management
@router.get("/portfolio")
async def get_user_portfolio(
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Get comprehensive user portfolio"""
    
    portfolio = await marketplace.get_user_portfolio(current_user["user_id"])
    return portfolio


# Transaction History
@router.get("/transactions")
async def get_transaction_history(
    limit: int = 50,
    transaction_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Get user transaction history"""
    
    user_transactions = [
        {
            "transaction_id": tx.transaction_id,
            "type": tx.transaction_type.value,
            "payment_method": tx.payment_method.value,
            "amount_usd": float(tx.amount_usd),
            "amount_ftns": float(tx.amount_ftns),
            "exchange_rate": float(tx.exchange_rate),
            "status": tx.status.value,
            "created_at": tx.created_at.isoformat(),
            "completed_at": tx.completed_at.isoformat() if tx.completed_at else None,
            "fees": {k: float(v) for k, v in tx.fees.items()},
            "metadata": tx.metadata
        }
        for tx in marketplace.transactions.values()
        if tx.user_id == current_user["user_id"]
    ]
    
    # Filter by transaction type if specified
    if transaction_type:
        user_transactions = [
            tx for tx in user_transactions 
            if tx["type"] == transaction_type
        ]
    
    # Sort by creation date (newest first) and limit
    user_transactions.sort(key=lambda x: x["created_at"], reverse=True)
    user_transactions = user_transactions[:limit]
    
    return {
        "transactions": user_transactions,
        "total_count": len(user_transactions),
        "filters_applied": {"transaction_type": transaction_type} if transaction_type else {}
    }


@router.get("/transaction/{transaction_id}")
async def get_transaction_details(
    transaction_id: str,
    current_user: dict = Depends(get_current_user),
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Get detailed transaction information"""
    
    if transaction_id not in marketplace.transactions:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    transaction = marketplace.transactions[transaction_id]
    
    if transaction.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "transaction_id": transaction.transaction_id,
        "type": transaction.transaction_type.value,
        "payment_method": transaction.payment_method.value,
        "amount_usd": float(transaction.amount_usd),
        "amount_ftns": float(transaction.amount_ftns),
        "exchange_rate": float(transaction.exchange_rate),
        "status": transaction.status.value,
        "created_at": transaction.created_at.isoformat(),
        "completed_at": transaction.completed_at.isoformat() if transaction.completed_at else None,
        "fees": {k: float(v) for k, v in transaction.fees.items()},
        "metadata": transaction.metadata,
        "external_transaction_id": transaction.external_transaction_id
    }


# Price Calculator
@router.get("/price-calculator")
async def calculate_purchase_price(
    amount_usd: Optional[Decimal] = None,
    amount_ftns: Optional[Decimal] = None,
    payment_method: str = "credit_card",
    marketplace: FTNSMarketplace = Depends(get_ftns_marketplace)
):
    """Calculate purchase price and fees"""
    
    if not amount_usd and not amount_ftns:
        raise HTTPException(status_code=400, detail="Must specify either amount_usd or amount_ftns")
    
    market_data = await marketplace.get_current_market_data()
    
    # Determine fee rate based on payment method
    fee_rates = {
        "credit_card": Decimal('0.029'),  # 2.9%
        "paypal": Decimal('0.025'),       # 2.5%
        "bank_transfer": Decimal('0.01'), # 1%
        "bitcoin": Decimal('0.01'),       # 1%
        "ethereum": Decimal('0.01'),      # 1%
        "usdc": Decimal('0.005'),         # 0.5%
        "usdt": Decimal('0.005')          # 0.5%
    }
    
    fee_rate = fee_rates.get(payment_method, Decimal('0.025'))
    
    if amount_usd:
        # Calculate FTNS amount from USD
        gross_ftns = amount_usd / market_data.ftns_price_usd
        fees_usd = amount_usd * fee_rate
        net_ftns = gross_ftns * (Decimal('1') - fee_rate)
        
        return {
            "amount_usd": float(amount_usd),
            "amount_ftns": float(net_ftns),
            "gross_ftns": float(gross_ftns),
            "exchange_rate": float(market_data.ftns_price_usd),
            "fees_usd": float(fees_usd),
            "fee_rate": float(fee_rate),
            "payment_method": payment_method
        }
    
    else:
        # Calculate USD amount from FTNS
        gross_usd = amount_ftns * market_data.ftns_price_usd
        net_usd = gross_usd / (Decimal('1') - fee_rate)
        fees_usd = net_usd - gross_usd
        
        return {
            "amount_usd": float(net_usd),
            "amount_ftns": float(amount_ftns),
            "gross_usd": float(gross_usd),
            "exchange_rate": float(market_data.ftns_price_usd),
            "fees_usd": float(fees_usd),
            "fee_rate": float(fee_rate),
            "payment_method": payment_method
        }


# Health Check
@router.get("/health")
async def marketplace_health_check():
    """Marketplace health check"""
    return {
        "status": "healthy",
        "service": "FTNS Marketplace",
        "version": "v1.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "fiat_purchases",
            "crypto_purchases", 
            "staking",
            "liquidity_provision",
            "portfolio_management",
            "real_time_analytics"
        ]
    }
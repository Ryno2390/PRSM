"""
FTNS Marketplace Trading Infrastructure

💱 FTNS TRADING MARKETPLACE:
- User-to-user FTNS credit trading and exchange
- Real-time order book with bid/ask spread management
- Secure escrow system for transaction safety
- Futures contracts for compute time reservations
- Automated market making and liquidity provision
- Trading analytics and profit/loss tracking

This module implements a comprehensive trading marketplace that enables:
1. Users to buy/sell FTNS credits at market prices
2. Arbitrage opportunities during demand fluctuations  
3. Futures contracts for guaranteed compute time pricing
4. Escrow-protected transactions with dispute resolution
5. Market making to ensure liquidity and price discovery
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
from prsm.compute.scheduling.ftns_pricing_engine import FTNSPricingEngine, ArbitrageOpportunity

logger = structlog.get_logger(__name__)


class OrderType(str, Enum):
    """Order types in the marketplace"""
    MARKET_BUY = "market_buy"          # Buy at current market price
    MARKET_SELL = "market_sell"        # Sell at current market price
    LIMIT_BUY = "limit_buy"           # Buy at specific price or better
    LIMIT_SELL = "limit_sell"         # Sell at specific price or better
    STOP_LOSS = "stop_loss"           # Sell when price drops to level
    TAKE_PROFIT = "take_profit"       # Sell when price rises to level
    FUTURES_BUY = "futures_buy"       # Buy futures contract
    FUTURES_SELL = "futures_sell"     # Sell futures contract


class OrderStatus(str, Enum):
    """Order execution status"""
    PENDING = "pending"               # Order placed, waiting for match
    PARTIALLY_FILLED = "partially_filled"  # Partially executed
    FILLED = "filled"                 # Completely executed
    CANCELLED = "cancelled"           # Cancelled by user
    EXPIRED = "expired"               # Expired due to time limit
    REJECTED = "rejected"             # Rejected due to insufficient funds/credits


class TransactionType(str, Enum):
    """Transaction types"""
    SPOT_TRADE = "spot_trade"         # Immediate FTNS transfer
    FUTURES_TRADE = "futures_trade"   # Futures contract execution
    ESCROW_DEPOSIT = "escrow_deposit" # Funds deposited to escrow
    ESCROW_RELEASE = "escrow_release" # Funds released from escrow
    ARBITRAGE_TRADE = "arbitrage_trade"  # Automated arbitrage execution


class ContractType(str, Enum):
    """Futures contract types"""
    COMPUTE_TIME = "compute_time"     # Reserve compute time at fixed price
    RESOURCE_ALLOCATION = "resource_allocation"  # Reserve specific resources
    WORKFLOW_EXECUTION = "workflow_execution"   # Reserve workflow execution slots
    PRIORITY_ACCESS = "priority_access"         # Reserve priority queue access


class MarketOrder(PRSMBaseModel):
    """Market order for FTNS trading"""
    order_id: UUID = Field(default_factory=uuid4)
    user_id: str
    
    # Order details
    order_type: OrderType
    resource_type: ResourceType = Field(default=ResourceType.FTNS_CREDITS)
    quantity: float = Field(gt=0)
    price: Optional[float] = None  # None for market orders
    
    # Execution parameters
    time_in_force: str = Field(default="GTC")  # GTC, IOC, FOK, DAY
    expires_at: Optional[datetime] = None
    min_fill_quantity: Optional[float] = None
    
    # Order state
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    filled_quantity: float = Field(default=0.0)
    remaining_quantity: float = Field(default=0.0)
    average_fill_price: Optional[float] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Fees and costs
    trading_fee: float = Field(default=0.0)
    escrow_fee: float = Field(default=0.0)
    total_cost: Optional[float] = None
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity


class FuturesContract(PRSMBaseModel):
    """Futures contract for compute time reservation"""
    contract_id: UUID = Field(default_factory=uuid4)
    contract_type: ContractType
    
    # Contract parties
    buyer_id: str
    seller_id: Optional[str] = None  # None for system-issued contracts
    
    # Contract terms
    resource_type: ResourceType
    quantity: float
    strike_price: float  # Fixed price for future execution
    premium: float = Field(default=0.0)  # Upfront premium paid
    
    # Timing
    contract_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expiration_date: datetime
    execution_window_start: datetime
    execution_window_end: datetime
    
    # Contract state
    is_exercised: bool = Field(default=False)
    is_expired: bool = Field(default=False)
    settlement_value: Optional[float] = None
    
    # Terms and conditions
    auto_exercise: bool = Field(default=True)
    transferable: bool = Field(default=True)
    minimum_notice_hours: int = Field(default=1)
    
    # Collateral and margin
    buyer_collateral: float = Field(default=0.0)
    seller_collateral: float = Field(default=0.0)
    margin_requirement: float = Field(default=0.1)  # 10% margin
    
    def calculate_intrinsic_value(self, current_market_price: float) -> float:
        """Calculate intrinsic value of the contract"""
        if self.contract_type in [ContractType.COMPUTE_TIME, ContractType.RESOURCE_ALLOCATION]:
            # For call-like contracts (right to buy at strike price)
            return max(0, current_market_price - self.strike_price) * self.quantity
        return 0.0
    
    def is_in_the_money(self, current_market_price: float) -> bool:
        """Check if contract is in the money"""
        return self.calculate_intrinsic_value(current_market_price) > 0


class EscrowAccount(PRSMBaseModel):
    """Escrow account for secure transactions"""
    escrow_id: UUID = Field(default_factory=uuid4)
    
    # Parties
    buyer_id: str
    seller_id: str
    arbiter_id: Optional[str] = None
    
    # Escrow details
    resource_type: ResourceType
    quantity: float
    agreed_price: float
    total_value: float
    
    # Funds management
    deposited_amount: float = Field(default=0.0)
    released_amount: float = Field(default=0.0)
    disputed_amount: float = Field(default=0.0)
    
    # State tracking
    is_funded: bool = Field(default=False)
    is_completed: bool = Field(default=False)
    is_disputed: bool = Field(default=False)
    dispute_reason: Optional[str] = None
    
    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    funded_at: Optional[datetime] = None
    release_deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Fees
    escrow_fee_percentage: float = Field(default=0.01)  # 1% escrow fee
    dispute_fee: float = Field(default=10.0)  # Fixed dispute fee
    
    def calculate_escrow_fee(self) -> float:
        """Calculate total escrow fee"""
        return self.total_value * self.escrow_fee_percentage


class MarketTransaction(PRSMBaseModel):
    """Completed marketplace transaction"""
    transaction_id: UUID = Field(default_factory=uuid4)
    transaction_type: TransactionType
    
    # Transaction parties
    buyer_id: str
    seller_id: str
    
    # Transaction details
    resource_type: ResourceType
    quantity: float
    price_per_unit: float
    total_value: float
    
    # Order references
    buyer_order_id: UUID
    seller_order_id: UUID
    
    # Execution details
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    settlement_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=5))
    
    # Fees and charges
    buyer_fee: float = Field(default=0.0)
    seller_fee: float = Field(default=0.0)
    platform_fee: float = Field(default=0.0)
    
    # Settlement status
    is_settled: bool = Field(default=False)
    settlement_method: str = Field(default="instant")
    
    # Associated contracts
    escrow_id: Optional[UUID] = None
    futures_contract_id: Optional[UUID] = None


class OrderBook(PRSMBaseModel):
    """Order book for a specific resource type"""
    resource_type: ResourceType
    
    # Order lists (sorted by price)
    buy_orders: List[MarketOrder] = Field(default_factory=list)  # Highest price first
    sell_orders: List[MarketOrder] = Field(default_factory=list)  # Lowest price first
    
    # Market data
    last_trade_price: Optional[float] = None
    bid_price: Optional[float] = None  # Highest buy order price
    ask_price: Optional[float] = None  # Lowest sell order price
    spread: Optional[float] = None     # Ask - Bid
    
    # Volume data
    total_buy_volume: float = Field(default=0.0)
    total_sell_volume: float = Field(default=0.0)
    daily_volume: float = Field(default=0.0)
    
    # Market statistics
    daily_high: Optional[float] = None
    daily_low: Optional[float] = None
    price_change_24h: float = Field(default=0.0)
    price_change_percentage: float = Field(default=0.0)
    
    def update_market_data(self):
        """Update bid/ask prices and spread"""
        # Update bid price (highest buy order)
        active_buy_orders = [order for order in self.buy_orders if order.status == OrderStatus.PENDING]
        self.bid_price = max([order.price for order in active_buy_orders if order.price], default=None)
        
        # Update ask price (lowest sell order)
        active_sell_orders = [order for order in self.sell_orders if order.status == OrderStatus.PENDING]
        self.ask_price = min([order.price for order in active_sell_orders if order.price], default=None)
        
        # Calculate spread
        if self.bid_price and self.ask_price:
            self.spread = self.ask_price - self.bid_price
        else:
            self.spread = None
        
        # Update volumes
        self.total_buy_volume = sum(order.remaining_quantity for order in active_buy_orders)
        self.total_sell_volume = sum(order.remaining_quantity for order in active_sell_orders)


class FTNSMarketplace(TimestampMixin):
    """
    FTNS Marketplace Trading Infrastructure
    
    Comprehensive trading platform enabling user-to-user FTNS transactions,
    futures contracts, and arbitrage opportunities.
    """
    
    def __init__(self):
        super().__init__()
        
        # Trading infrastructure
        self.order_books: Dict[ResourceType, OrderBook] = {}
        self.active_orders: Dict[UUID, MarketOrder] = {}
        self.completed_transactions: List[MarketTransaction] = []
        
        # Futures and derivatives
        self.futures_contracts: Dict[UUID, FuturesContract] = {}
        self.active_escrows: Dict[UUID, EscrowAccount] = {}
        
        # User accounts and balances
        self.user_balances: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))
        self.user_portfolios: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Market making and liquidity
        self.market_makers: Dict[ResourceType, Dict[str, Any]] = {}
        self.liquidity_providers: List[str] = []
        
        # Trading statistics
        self.trading_statistics: Dict[str, Any] = defaultdict(float)
        self.daily_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Integration with pricing engine
        self.pricing_engine: Optional[FTNSPricingEngine] = None
        
        # Configuration
        self.trading_fee_percentage = 0.005  # 0.5% trading fee
        self.market_maker_fee_discount = 0.002  # 0.2% discount for market makers
        self.minimum_order_size = 1.0
        self.maximum_order_size = 10000.0
        self.max_orders_per_user = 100
        
        self._initialize_order_books()
        self._start_market_operations()
        
        logger.info("FTNSMarketplace initialized")
    
    def _initialize_order_books(self):
        """Initialize order books for all resource types"""
        for resource_type in ResourceType:
            self.order_books[resource_type] = OrderBook(resource_type=resource_type)
    
    def set_pricing_engine(self, engine: FTNSPricingEngine):
        """Set pricing engine for market integration"""
        self.pricing_engine = engine
        logger.info("Pricing engine integrated with marketplace")
    
    def _start_market_operations(self):
        """Start market operations and monitoring"""
        # In production, this would run background tasks for:
        # - Order matching
        # - Market making
        # - Contract settlement
        # - Statistics calculation
        pass
    
    async def place_order(
        self,
        user_id: str,
        order_type: OrderType,
        resource_type: ResourceType,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        expires_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Place a trading order in the marketplace
        
        Args:
            user_id: ID of user placing order
            order_type: Type of order (market, limit, etc.)
            resource_type: Type of resource to trade
            quantity: Quantity to trade
            price: Price for limit orders (None for market orders)
            time_in_force: Order duration (GTC, IOC, FOK, DAY)
            expires_at: Expiration time for order
            
        Returns:
            Order placement result
        """
        try:
            # Validate order parameters
            validation_result = await self._validate_order(
                user_id, order_type, resource_type, quantity, price
            )
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["error"]}
            
            # Create order
            order = MarketOrder(
                user_id=user_id,
                order_type=order_type,
                resource_type=resource_type,
                quantity=quantity,
                price=price,
                time_in_force=time_in_force,
                expires_at=expires_at,
                remaining_quantity=quantity
            )
            
            # Calculate fees
            order.trading_fee = await self._calculate_trading_fee(user_id, quantity, price)
            
            # For market orders, get current market price
            if order_type in [OrderType.MARKET_BUY, OrderType.MARKET_SELL]:
                market_price = await self._get_market_price(resource_type, order_type)
                if market_price is None:
                    return {"success": False, "error": "No market price available"}
                order.price = market_price
            
            # Check user has sufficient balance/credits
            balance_check = await self._check_user_balance(user_id, order)
            if not balance_check["sufficient"]:
                return {"success": False, "error": "Insufficient balance"}
            
            # Add order to order book
            order_book = self.order_books[resource_type]
            
            if order_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY]:
                order_book.buy_orders.append(order)
                order_book.buy_orders.sort(key=lambda x: x.price or 0, reverse=True)
            else:
                order_book.sell_orders.append(order)
                order_book.sell_orders.sort(key=lambda x: x.price or float('inf'))
            
            # Store active order
            self.active_orders[order.order_id] = order
            
            # Update market data
            order_book.update_market_data()
            
            # Attempt immediate matching
            match_results = await self._match_orders(resource_type)
            
            # Update statistics
            self.trading_statistics["orders_placed"] += 1
            self.trading_statistics[f"orders_{order_type.value}"] += 1
            
            logger.info(
                "Order placed successfully",
                order_id=str(order.order_id),
                user_id=user_id,
                order_type=order_type.value,
                quantity=quantity,
                price=price,
                matches=len(match_results)
            )
            
            return {
                "success": True,
                "order_id": str(order.order_id),
                "status": order.status.value,
                "price": order.price,
                "trading_fee": order.trading_fee,
                "immediate_matches": len(match_results),
                "order_book_position": self._get_order_book_position(order)
            }
            
        except Exception as e:
            logger.error("Error placing order", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _validate_order(
        self,
        user_id: str,
        order_type: OrderType,
        resource_type: ResourceType,
        quantity: float,
        price: Optional[float]
    ) -> Dict[str, Any]:
        """Validate order parameters"""
        # Check quantity limits
        if quantity < self.minimum_order_size:
            return {"valid": False, "error": f"Minimum order size is {self.minimum_order_size}"}
        
        if quantity > self.maximum_order_size:
            return {"valid": False, "error": f"Maximum order size is {self.maximum_order_size}"}
        
        # Check price for limit orders
        if order_type in [OrderType.LIMIT_BUY, OrderType.LIMIT_SELL] and price is None:
            return {"valid": False, "error": "Price required for limit orders"}
        
        if price is not None and price <= 0:
            return {"valid": False, "error": "Price must be positive"}
        
        # Check user order limit
        user_active_orders = [order for order in self.active_orders.values() if order.user_id == user_id]
        if len(user_active_orders) >= self.max_orders_per_user:
            return {"valid": False, "error": f"Maximum {self.max_orders_per_user} active orders per user"}
        
        return {"valid": True}
    
    async def _calculate_trading_fee(
        self,
        user_id: str,
        quantity: float,
        price: Optional[float]
    ) -> float:
        """Calculate trading fee for order"""
        base_fee_rate = self.trading_fee_percentage
        
        # Apply market maker discount
        if user_id in self.liquidity_providers:
            base_fee_rate -= self.market_maker_fee_discount
        
        # Calculate fee based on order value
        if price is not None:
            order_value = quantity * price
            return order_value * base_fee_rate
        else:
            # For market orders, estimate based on current market price
            return quantity * 1.0 * base_fee_rate  # Assume $1 per unit as estimate
    
    async def _get_market_price(
        self,
        resource_type: ResourceType,
        order_type: OrderType
    ) -> Optional[float]:
        """Get current market price for resource type"""
        order_book = self.order_books[resource_type]
        
        if order_type == OrderType.MARKET_BUY:
            # For market buy, use ask price (lowest sell order)
            return order_book.ask_price
        elif order_type == OrderType.MARKET_SELL:
            # For market sell, use bid price (highest buy order)
            return order_book.bid_price
        
        # Fallback to last trade price or pricing engine
        if order_book.last_trade_price:
            return order_book.last_trade_price
        
        if self.pricing_engine:
            return await self.pricing_engine.get_current_price(resource_type)
        
        return None
    
    async def _check_user_balance(
        self,
        user_id: str,
        order: MarketOrder
    ) -> Dict[str, Any]:
        """Check if user has sufficient balance for order"""
        required_amount = 0.0
        
        if order.order_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY]:
            # For buy orders, need cash/credits
            if order.price:
                required_amount = order.quantity * order.price + order.trading_fee
            
            available_cash = self.user_balances[user_id].get("CASH", 0.0)
            sufficient = available_cash >= required_amount
            
        else:
            # For sell orders, need the resource being sold
            available_resources = self.user_balances[user_id].get(order.resource_type, 0.0)
            sufficient = available_resources >= order.quantity
        
        return {
            "sufficient": sufficient,
            "required": required_amount,
            "available": available_cash if order.order_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY] 
                        else self.user_balances[user_id].get(order.resource_type, 0.0)
        }
    
    def _get_order_book_position(self, order: MarketOrder) -> Dict[str, Any]:
        """Get order's position in the order book"""
        order_book = self.order_books[order.resource_type]
        
        if order.order_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY]:
            orders_list = order_book.buy_orders
            position = next((i for i, o in enumerate(orders_list) if o.order_id == order.order_id), -1)
            total_ahead_volume = sum(o.remaining_quantity for o in orders_list[:position])
        else:
            orders_list = order_book.sell_orders
            position = next((i for i, o in enumerate(orders_list) if o.order_id == order.order_id), -1)
            total_ahead_volume = sum(o.remaining_quantity for o in orders_list[:position])
        
        return {
            "position": position + 1,
            "total_orders_ahead": position,
            "total_volume_ahead": total_ahead_volume,
            "total_orders_in_book": len(orders_list)
        }
    
    async def _match_orders(self, resource_type: ResourceType) -> List[MarketTransaction]:
        """Match buy and sell orders for a resource type"""
        try:
            matches = []
            order_book = self.order_books[resource_type]
            
            # Get active orders
            buy_orders = [o for o in order_book.buy_orders if o.status == OrderStatus.PENDING]
            sell_orders = [o for o in order_book.sell_orders if o.status == OrderStatus.PENDING]
            
            # Sort orders (buy orders by price descending, sell orders by price ascending)
            buy_orders.sort(key=lambda x: x.price or 0, reverse=True)
            sell_orders.sort(key=lambda x: x.price or float('inf'))
            
            i, j = 0, 0
            while i < len(buy_orders) and j < len(sell_orders):
                buy_order = buy_orders[i]
                sell_order = sell_orders[j]
                
                # Check if orders can match
                if buy_order.price is None or sell_order.price is None:
                    # Market orders match at any price
                    can_match = True
                    match_price = sell_order.price or buy_order.price or order_book.last_trade_price or 1.0
                elif buy_order.price >= sell_order.price:
                    can_match = True
                    match_price = sell_order.price  # Seller's price takes precedence
                else:
                    can_match = False
                    match_price = None
                
                if can_match:
                    # Calculate match quantity
                    match_quantity = min(buy_order.remaining_quantity, sell_order.remaining_quantity)
                    
                    # Create transaction
                    transaction = await self._create_transaction(
                        buy_order, sell_order, match_quantity, match_price
                    )
                    
                    if transaction:
                        matches.append(transaction)
                        
                        # Update order statuses
                        buy_order.filled_quantity += match_quantity
                        buy_order.remaining_quantity -= match_quantity
                        sell_order.filled_quantity += match_quantity
                        sell_order.remaining_quantity -= match_quantity
                        
                        # Update average fill prices
                        buy_order.average_fill_price = (
                            (buy_order.average_fill_price or 0) * (buy_order.filled_quantity - match_quantity) +
                            match_price * match_quantity
                        ) / buy_order.filled_quantity if buy_order.filled_quantity > 0 else match_price
                        
                        sell_order.average_fill_price = (
                            (sell_order.average_fill_price or 0) * (sell_order.filled_quantity - match_quantity) +
                            match_price * match_quantity
                        ) / sell_order.filled_quantity if sell_order.filled_quantity > 0 else match_price
                        
                        # Update order statuses
                        if buy_order.remaining_quantity == 0:
                            buy_order.status = OrderStatus.FILLED
                        else:
                            buy_order.status = OrderStatus.PARTIALLY_FILLED
                            
                        if sell_order.remaining_quantity == 0:
                            sell_order.status = OrderStatus.FILLED
                        else:
                            sell_order.status = OrderStatus.PARTIALLY_FILLED
                        
                        # Update market data
                        order_book.last_trade_price = match_price
                        order_book.daily_volume += match_quantity
                        
                        # Update daily stats
                        today = datetime.now(timezone.utc).date().isoformat()
                        if order_book.daily_high is None or match_price > order_book.daily_high:
                            order_book.daily_high = match_price
                        if order_book.daily_low is None or match_price < order_book.daily_low:
                            order_book.daily_low = match_price
                        
                        # Move to next orders if filled
                        if buy_order.remaining_quantity == 0:
                            i += 1
                        if sell_order.remaining_quantity == 0:
                            j += 1
                else:
                    # No more matches possible
                    break
            
            # Update order book market data
            order_book.update_market_data()
            
            # Update statistics
            self.trading_statistics["transactions_executed"] += len(matches)
            total_volume = sum(t.quantity for t in matches)
            self.trading_statistics["total_volume"] += total_volume
            
            logger.info(
                "Order matching completed",
                resource_type=resource_type.value,
                matches=len(matches),
                total_volume=total_volume
            )
            
            return matches
            
        except Exception as e:
            logger.error("Error matching orders", error=str(e))
            return []
    
    async def _create_transaction(
        self,
        buy_order: MarketOrder,
        sell_order: MarketOrder,
        quantity: float,
        price: float
    ) -> Optional[MarketTransaction]:
        """Create a transaction from matched orders"""
        try:
            transaction = MarketTransaction(
                transaction_type=TransactionType.SPOT_TRADE,
                buyer_id=buy_order.user_id,
                seller_id=sell_order.user_id,
                resource_type=buy_order.resource_type,
                quantity=quantity,
                price_per_unit=price,
                total_value=quantity * price,
                buyer_order_id=buy_order.order_id,
                seller_order_id=sell_order.order_id
            )
            
            # Calculate fees
            transaction.buyer_fee = quantity * price * self.trading_fee_percentage
            transaction.seller_fee = quantity * price * self.trading_fee_percentage
            transaction.platform_fee = transaction.buyer_fee + transaction.seller_fee
            
            # Apply fee discounts for market makers
            if buy_order.user_id in self.liquidity_providers:
                transaction.buyer_fee *= 0.6  # 40% discount
            if sell_order.user_id in self.liquidity_providers:
                transaction.seller_fee *= 0.6  # 40% discount
            
            # Execute settlement
            await self._settle_transaction(transaction)
            
            # Store transaction
            self.completed_transactions.append(transaction)
            
            logger.info(
                "Transaction created",
                transaction_id=str(transaction.transaction_id),
                buyer_id=buy_order.user_id,
                seller_id=sell_order.user_id,
                quantity=quantity,
                price=price
            )
            
            return transaction
            
        except Exception as e:
            logger.error("Error creating transaction", error=str(e))
            return None
    
    async def _settle_transaction(self, transaction: MarketTransaction):
        """Settle a completed transaction"""
        try:
            # Transfer resources from seller to buyer
            seller_balance = self.user_balances[transaction.seller_id]
            buyer_balance = self.user_balances[transaction.buyer_id]
            
            # Remove resources from seller
            seller_balance[transaction.resource_type] -= transaction.quantity
            
            # Add resources to buyer
            buyer_balance[transaction.resource_type] += transaction.quantity
            
            # Handle cash transfers
            total_cost = transaction.total_value + transaction.buyer_fee
            seller_credit = transaction.total_value - transaction.seller_fee
            
            # Deduct cash from buyer
            buyer_balance["CASH"] = buyer_balance.get("CASH", 0.0) - total_cost
            
            # Credit cash to seller
            seller_balance["CASH"] = seller_balance.get("CASH", 0.0) + seller_credit
            
            # Mark as settled
            transaction.is_settled = True
            transaction.settlement_time = datetime.now(timezone.utc)
            
            logger.info(
                "Transaction settled",
                transaction_id=str(transaction.transaction_id),
                buyer_cost=total_cost,
                seller_credit=seller_credit
            )
            
        except Exception as e:
            logger.error("Error settling transaction", error=str(e))
            raise
    
    async def create_futures_contract(
        self,
        buyer_id: str,
        contract_type: ContractType,
        resource_type: ResourceType,
        quantity: float,
        strike_price: float,
        expiration_date: datetime,
        execution_window_start: datetime,
        execution_window_end: datetime,
        premium: float = 0.0
    ) -> Dict[str, Any]:
        """
        Create a futures contract for compute time reservation
        
        Args:
            buyer_id: ID of contract buyer
            contract_type: Type of futures contract
            resource_type: Type of resource
            quantity: Quantity to reserve
            strike_price: Fixed execution price
            expiration_date: Contract expiration
            execution_window_start: Start of execution window
            execution_window_end: End of execution window
            premium: Upfront premium payment
            
        Returns:
            Contract creation result
        """
        try:
            # Calculate margin requirements
            margin_requirement = strike_price * quantity * 0.1  # 10% margin
            
            # Check buyer balance
            buyer_balance = self.user_balances[buyer_id].get("CASH", 0.0)
            required_funds = premium + margin_requirement
            
            if buyer_balance < required_funds:
                return {
                    "success": False,
                    "error": "Insufficient funds for premium and margin",
                    "required": required_funds,
                    "available": buyer_balance
                }
            
            # Create contract
            contract = FuturesContract(
                contract_type=contract_type,
                buyer_id=buyer_id,
                resource_type=resource_type,
                quantity=quantity,
                strike_price=strike_price,
                premium=premium,
                expiration_date=expiration_date,
                execution_window_start=execution_window_start,
                execution_window_end=execution_window_end,
                buyer_collateral=margin_requirement
            )
            
            # Deduct premium and margin from buyer
            self.user_balances[buyer_id]["CASH"] -= required_funds
            
            # Store contract
            self.futures_contracts[contract.contract_id] = contract
            
            # Update statistics
            self.trading_statistics["futures_contracts_created"] += 1
            
            logger.info(
                "Futures contract created",
                contract_id=str(contract.contract_id),
                buyer_id=buyer_id,
                contract_type=contract_type.value,
                strike_price=strike_price,
                quantity=quantity
            )
            
            return {
                "success": True,
                "contract_id": str(contract.contract_id),
                "strike_price": strike_price,
                "premium_paid": premium,
                "margin_requirement": margin_requirement,
                "execution_window": {
                    "start": execution_window_start.isoformat(),
                    "end": execution_window_end.isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Error creating futures contract", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def create_escrow_account(
        self,
        buyer_id: str,
        seller_id: str,
        resource_type: ResourceType,
        quantity: float,
        agreed_price: float,
        release_deadline: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Create an escrow account for secure transactions
        
        Args:
            buyer_id: ID of buyer
            seller_id: ID of seller
            resource_type: Type of resource
            quantity: Quantity being traded
            agreed_price: Agreed price per unit
            release_deadline: Deadline for automatic release
            
        Returns:
            Escrow creation result
        """
        try:
            total_value = quantity * agreed_price
            
            # Create escrow account
            escrow = EscrowAccount(
                buyer_id=buyer_id,
                seller_id=seller_id,
                resource_type=resource_type,
                quantity=quantity,
                agreed_price=agreed_price,
                total_value=total_value,
                release_deadline=release_deadline or (datetime.now(timezone.utc) + timedelta(days=7))
            )
            
            # Calculate escrow fee
            escrow_fee = escrow.calculate_escrow_fee()
            total_required = total_value + escrow_fee
            
            # Check buyer balance
            buyer_balance = self.user_balances[buyer_id].get("CASH", 0.0)
            if buyer_balance < total_required:
                return {
                    "success": False,
                    "error": "Insufficient funds for escrow deposit",
                    "required": total_required,
                    "available": buyer_balance
                }
            
            # Deduct funds from buyer
            self.user_balances[buyer_id]["CASH"] -= total_required
            escrow.deposited_amount = total_value
            escrow.is_funded = True
            escrow.funded_at = datetime.now(timezone.utc)
            
            # Store escrow account
            self.active_escrows[escrow.escrow_id] = escrow
            
            # Update statistics
            self.trading_statistics["escrow_accounts_created"] += 1
            
            logger.info(
                "Escrow account created",
                escrow_id=str(escrow.escrow_id),
                buyer_id=buyer_id,
                seller_id=seller_id,
                total_value=total_value
            )
            
            return {
                "success": True,
                "escrow_id": str(escrow.escrow_id),
                "total_value": total_value,
                "escrow_fee": escrow_fee,
                "release_deadline": escrow.release_deadline.isoformat()
            }
            
        except Exception as e:
            logger.error("Error creating escrow account", error=str(e))
            return {"success": False, "error": str(e)}
    
    def get_market_data(self, resource_type: ResourceType) -> Dict[str, Any]:
        """Get current market data for a resource type"""
        order_book = self.order_books[resource_type]
        
        return {
            "resource_type": resource_type.value,
            "bid_price": order_book.bid_price,
            "ask_price": order_book.ask_price,
            "spread": order_book.spread,
            "last_trade_price": order_book.last_trade_price,
            "daily_volume": order_book.daily_volume,
            "daily_high": order_book.daily_high,
            "daily_low": order_book.daily_low,
            "price_change_24h": order_book.price_change_24h,
            "total_buy_volume": order_book.total_buy_volume,
            "total_sell_volume": order_book.total_sell_volume,
            "buy_orders_count": len([o for o in order_book.buy_orders if o.status == OrderStatus.PENDING]),
            "sell_orders_count": len([o for o in order_book.sell_orders if o.status == OrderStatus.PENDING])
        }
    
    def get_user_portfolio(self, user_id: str) -> Dict[str, Any]:
        """Get user's trading portfolio and statistics"""
        user_balances = dict(self.user_balances[user_id])
        user_orders = [order for order in self.active_orders.values() if order.user_id == user_id]
        user_transactions = [tx for tx in self.completed_transactions if tx.buyer_id == user_id or tx.seller_id == user_id]
        user_contracts = [contract for contract in self.futures_contracts.values() if contract.buyer_id == user_id]
        user_escrows = [escrow for escrow in self.active_escrows.values() 
                       if escrow.buyer_id == user_id or escrow.seller_id == user_id]
        
        # Calculate portfolio value
        total_portfolio_value = user_balances.get("CASH", 0.0)
        for resource_type, quantity in user_balances.items():
            if resource_type != "CASH" and quantity > 0:
                # Get current market price
                market_data = self.get_market_data(ResourceType(resource_type))
                market_price = market_data.get("last_trade_price") or 1.0
                total_portfolio_value += quantity * market_price
        
        # Calculate trading statistics
        total_trades = len(user_transactions)
        total_volume = sum(tx.quantity for tx in user_transactions)
        total_fees_paid = sum(
            (tx.buyer_fee if tx.buyer_id == user_id else 0) + 
            (tx.seller_fee if tx.seller_id == user_id else 0)
            for tx in user_transactions
        )
        
        return {
            "user_id": user_id,
            "balances": user_balances,
            "portfolio_value": total_portfolio_value,
            "active_orders": len(user_orders),
            "total_trades": total_trades,
            "total_volume": total_volume,
            "total_fees_paid": total_fees_paid,
            "futures_contracts": len(user_contracts),
            "active_escrows": len(user_escrows),
            "recent_transactions": user_transactions[-10:] if user_transactions else []
        }
    
    def get_marketplace_statistics(self) -> Dict[str, Any]:
        """Get comprehensive marketplace statistics"""
        total_users = len(self.user_balances)
        total_active_orders = len(self.active_orders)
        total_transactions = len(self.completed_transactions)
        
        # Calculate total volume across all resource types
        daily_volumes = {}
        for resource_type, order_book in self.order_books.items():
            daily_volumes[resource_type.value] = order_book.daily_volume
        
        return {
            "trading_statistics": dict(self.trading_statistics),
            "total_users": total_users,
            "total_active_orders": total_active_orders,
            "total_transactions": total_transactions,
            "total_futures_contracts": len(self.futures_contracts),
            "total_escrow_accounts": len(self.active_escrows),
            "daily_volumes": daily_volumes,
            "market_data": {
                resource_type.value: self.get_market_data(resource_type)
                for resource_type in ResourceType
            }
        }


# Global instance for easy access
_ftns_marketplace = None

def get_ftns_marketplace() -> FTNSMarketplace:
    """Get global FTNS marketplace instance"""
    global _ftns_marketplace
    if _ftns_marketplace is None:
        _ftns_marketplace = FTNSMarketplace()
    return _ftns_marketplace
"""
CHRONOS Clearing Engine

Core clearing and settlement logic for cross-asset swaps.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import hashlib
import json

from .models import (
    SwapRequest, Settlement, ClearingTransaction, LiquidityPool,
    AssetType, SwapType, TransactionStatus
)
from .wallet_manager import MultiSigWalletManager
from .exchange_router import ExchangeRouter
from .hub_spoke_router import HubSpokeRouter
from prsm.core.ipfs_client import IPFSClient


logger = logging.getLogger(__name__)


class ChronosEngine:
    """Main clearing engine for CHRONOS protocol with hub-and-spoke routing."""
    
    def __init__(
        self,
        wallet_manager: MultiSigWalletManager,
        exchange_router: ExchangeRouter,
        ipfs_client: IPFSClient
    ):
        self.wallet_manager = wallet_manager
        self.exchange_router = exchange_router
        self.ipfs_client = ipfs_client
        
        # Initialize hub-and-spoke router
        self.hub_spoke_router = HubSpokeRouter(exchange_router)
        
        # Mock liquidity pools for proof-of-concept
        self.liquidity_pools = self._initialize_mock_pools()
        
        # Active transactions
        self.active_transactions: Dict[str, ClearingTransaction] = {}
        
        # Enhanced fee structure for multi-hop routing
        self.base_fees = {
            "clearing_fee": Decimal("0.001"),    # 0.1%
            "network_fee": Decimal("0.0005"),    # 0.05%
            "liquidity_fee": Decimal("0.003"),   # 0.3%
            "hub_routing_fee": Decimal("0.0005"), # 0.05% per hub hop
            "compliance_fee": Decimal("0.002")   # 0.2% for regulated conversions
        }
    
    def _initialize_mock_pools(self) -> Dict[str, LiquidityPool]:
        """Initialize mock liquidity pools for testing."""
        pools = {}
        
        # FTNS/BTC pool
        pools["FTNS_BTC"] = LiquidityPool(
            asset_a=AssetType.FTNS,
            asset_b=AssetType.BTC,
            reserve_a=Decimal("1000000"),  # 1M FTNS
            reserve_b=Decimal("10"),       # 10 BTC
            total_liquidity=Decimal("3162.27"),  # sqrt(1M * 10)
            fee_rate=Decimal("0.003")
        )
        
        # FTNS/USD pool
        pools["FTNS_USD"] = LiquidityPool(
            asset_a=AssetType.FTNS,
            asset_b=AssetType.USD,
            reserve_a=Decimal("1000000"),  # 1M FTNS
            reserve_b=Decimal("500000"),   # 500K USD
            total_liquidity=Decimal("707106.78"),  # sqrt(1M * 500K)
            fee_rate=Decimal("0.003")
        )
        
        # BTC/USD pool
        pools["BTC_USD"] = LiquidityPool(
            asset_a=AssetType.BTC,
            asset_b=AssetType.USD,
            reserve_a=Decimal("100"),      # 100 BTC
            reserve_b=Decimal("5000000"),  # 5M USD
            total_liquidity=Decimal("22360.67"),  # sqrt(100 * 5M)
            fee_rate=Decimal("0.001")
        )
        
        return pools
    
    async def submit_swap_request(self, request: SwapRequest) -> ClearingTransaction:
        """Submit a new swap request for processing."""
        logger.info(f"Processing swap request: {request.id}")
        
        # Create transaction record
        transaction = ClearingTransaction(
            swap_request=request,
            status=TransactionStatus.PENDING
        )
        
        self.active_transactions[transaction.id] = transaction
        
        # Start async processing
        asyncio.create_task(self._process_transaction(transaction))
        
        return transaction
    
    async def _process_transaction(self, transaction: ClearingTransaction):
        """Process a clearing transaction through all stages."""
        try:
            transaction.status = TransactionStatus.VERIFYING
            transaction.updated_at = datetime.utcnow()
            
            # Stage 1: Verify liquidity and pricing
            if not await self._verify_liquidity(transaction):
                raise Exception("Insufficient liquidity")
            
            # Stage 2: Calculate optimal route
            route = await self._calculate_optimal_route(transaction)
            transaction.exchange_route = route
            
            # Stage 3: Execute atomic swap
            transaction.status = TransactionStatus.EXECUTING
            transaction.updated_at = datetime.utcnow()
            
            settlement = await self._execute_atomic_swap(transaction)
            transaction.settlement = settlement
            
            # Stage 4: Record settlement on IPFS
            settlement_hash = await self._record_settlement(settlement)
            settlement.settlement_hash = settlement_hash
            
            # Stage 5: Complete transaction
            transaction.status = TransactionStatus.COMPLETED
            transaction.completed_at = datetime.utcnow()
            transaction.updated_at = datetime.utcnow()
            
            logger.info(f"Transaction completed: {transaction.id}")
            
        except Exception as e:
            logger.error(f"Transaction failed {transaction.id}: {str(e)}")
            transaction.status = TransactionStatus.FAILED
            transaction.error_message = str(e)
            transaction.updated_at = datetime.utcnow()
    
    async def _verify_liquidity(self, transaction: ClearingTransaction) -> bool:
        """Verify sufficient liquidity exists for the swap."""
        request = transaction.swap_request
        
        # Get appropriate liquidity pool
        pool_key = self._get_pool_key(request.from_asset, request.to_asset)
        
        if pool_key not in self.liquidity_pools:
            # Try indirect route (e.g., FTNS->USD via FTNS->BTC->USD)
            return await self._verify_indirect_liquidity(request)
        
        pool = self.liquidity_pools[pool_key]
        
        try:
            # Check if pool can handle the requested amount
            output_amount = pool.calculate_output(request.from_asset, request.from_amount)
            
            # Apply slippage check
            if request.to_amount:
                min_acceptable = request.to_amount * (Decimal("1") - request.max_slippage)
                if output_amount < min_acceptable:
                    return False
            
            # Update calculated amount
            request.to_amount = output_amount
            return True
            
        except Exception as e:
            logger.error(f"Liquidity verification failed: {e}")
            return False
    
    async def _verify_indirect_liquidity(self, request: SwapRequest) -> bool:
        """Verify liquidity for indirect swaps (e.g., FTNS->USD via BTC)."""
        # For proof-of-concept, implement basic indirect routing
        if request.from_asset == AssetType.FTNS and request.to_asset == AssetType.USD:
            # Route: FTNS -> BTC -> USD
            btc_pool = self.liquidity_pools.get("FTNS_BTC")
            usd_pool = self.liquidity_pools.get("BTC_USD")
            
            if btc_pool and usd_pool:
                try:
                    # Step 1: FTNS -> BTC
                    btc_amount = btc_pool.calculate_output(AssetType.FTNS, request.from_amount)
                    
                    # Step 2: BTC -> USD
                    usd_amount = usd_pool.calculate_output(AssetType.BTC, btc_amount)
                    
                    request.to_amount = usd_amount
                    return True
                except Exception:
                    return False
        
        return False
    
    async def _calculate_optimal_route(self, transaction: ClearingTransaction) -> List[str]:
        """Calculate optimal routing through exchanges."""
        request = transaction.swap_request
        
        # For proof-of-concept, use simple routing logic
        if self._is_direct_swap(request.from_asset, request.to_asset):
            return ["internal_pool"]
        else:
            # Use indirect routing
            if request.from_asset == AssetType.FTNS and request.to_asset == AssetType.USD:
                return ["internal_pool", "internal_pool"]  # FTNS->BTC->USD
            else:
                return ["external_exchange"]
    
    async def _execute_atomic_swap(self, transaction: ClearingTransaction) -> Settlement:
        """Execute the atomic swap operation."""
        request = transaction.swap_request
        
        # Calculate fees
        fees = self._calculate_fees(request.from_amount, request.to_amount)
        total_fees = sum(fees.values())
        net_amount = request.to_amount - total_fees
        
        # For proof-of-concept, simulate blockchain transactions
        blockchain_txids = await self._simulate_blockchain_transactions(request)
        transaction.blockchain_txids = blockchain_txids
        
        # Update liquidity pools (mock operation)
        await self._update_liquidity_pools(request)
        
        # Create settlement record
        settlement = Settlement(
            swap_request_id=request.id,
            from_asset=request.from_asset,
            to_asset=request.to_asset,
            from_amount=request.from_amount,
            to_amount=request.to_amount,
            exchange_rate=request.to_amount / request.from_amount,
            fees=fees,
            total_fees=total_fees,
            net_amount=net_amount,
            settlement_hash=""  # Will be set after IPFS recording
        )
        
        return settlement
    
    def _calculate_fees(self, from_amount: Decimal, to_amount: Decimal) -> Dict[str, Decimal]:
        """Calculate fee breakdown for the transaction."""
        fees = {}
        
        # Clearing fee (percentage of from_amount)
        fees["clearing_fee"] = from_amount * self.base_fees["clearing_fee"]
        
        # Network fee (fixed + percentage)
        fees["network_fee"] = to_amount * self.base_fees["network_fee"]
        
        # Liquidity provider fee
        fees["liquidity_fee"] = to_amount * self.base_fees["liquidity_fee"]
        
        return fees
    
    async def _simulate_blockchain_transactions(self, request: SwapRequest) -> Dict[str, str]:
        """Simulate blockchain transaction IDs."""
        # In real implementation, this would interact with actual blockchains
        txids = {}
        
        if request.from_asset == AssetType.BTC or request.to_asset == AssetType.BTC:
            txids["bitcoin"] = f"btc_tx_{request.id[:8]}"
        
        if request.from_asset == AssetType.FTNS or request.to_asset == AssetType.FTNS:
            txids["iota"] = f"iota_tx_{request.id[:8]}"
        
        # Simulate transaction confirmation delay
        await asyncio.sleep(1)
        
        return txids
    
    async def _update_liquidity_pools(self, request: SwapRequest):
        """Update liquidity pool reserves after swap."""
        pool_key = self._get_pool_key(request.from_asset, request.to_asset)
        
        if pool_key in self.liquidity_pools:
            pool = self.liquidity_pools[pool_key]
            
            if request.from_asset == pool.asset_a:
                pool.reserve_a += request.from_amount
                pool.reserve_b -= request.to_amount
            else:
                pool.reserve_b += request.from_amount
                pool.reserve_a -= request.to_amount
            
            pool.last_updated = datetime.utcnow()
    
    async def _record_settlement(self, settlement: Settlement) -> str:
        """Record settlement on IPFS for provenance."""
        settlement_data = {
            "settlement_id": settlement.id,
            "swap_request_id": settlement.swap_request_id,
            "from_asset": settlement.from_asset.value,
            "to_asset": settlement.to_asset.value,
            "from_amount": str(settlement.from_amount),
            "to_amount": str(settlement.to_amount),
            "exchange_rate": str(settlement.exchange_rate),
            "fees": {k: str(v) for k, v in settlement.fees.items()},
            "total_fees": str(settlement.total_fees),
            "net_amount": str(settlement.net_amount),
            "timestamp": settlement.created_at.isoformat()
        }
        
        # Create hash for integrity
        data_json = json.dumps(settlement_data, sort_keys=True)
        settlement_hash = hashlib.sha256(data_json.encode()).hexdigest()
        
        # In real implementation, this would use actual IPFS
        logger.info(f"Settlement recorded with hash: {settlement_hash}")
        
        return settlement_hash
    
    def _get_pool_key(self, asset_a: AssetType, asset_b: AssetType) -> str:
        """Get standardized pool key for asset pair."""
        assets = sorted([asset_a.value, asset_b.value])
        return f"{assets[0]}_{assets[1]}"
    
    def _is_direct_swap(self, from_asset: AssetType, to_asset: AssetType) -> bool:
        """Check if direct swap is available."""
        pool_key = self._get_pool_key(from_asset, to_asset)
        return pool_key in self.liquidity_pools
    
    async def get_transaction_status(self, transaction_id: str) -> Optional[ClearingTransaction]:
        """Get current status of a transaction."""
        return self.active_transactions.get(transaction_id)
    
    async def get_quote(self, from_asset: AssetType, to_asset: AssetType, amount: Decimal) -> Dict:
        """Get quote for potential swap."""
        pool_key = self._get_pool_key(from_asset, to_asset)
        
        if pool_key in self.liquidity_pools:
            pool = self.liquidity_pools[pool_key]
            try:
                output_amount = pool.calculate_output(from_asset, amount)
                exchange_rate = output_amount / amount
                fees = self._calculate_fees(amount, output_amount)
                total_fees = sum(fees.values())
                net_amount = output_amount - total_fees
                
                return {
                    "from_asset": from_asset.value,
                    "to_asset": to_asset.value,
                    "from_amount": str(amount),
                    "to_amount": str(output_amount),
                    "net_amount": str(net_amount),
                    "exchange_rate": str(exchange_rate),
                    "fees": {k: str(v) for k, v in fees.items()},
                    "total_fees": str(total_fees),
                    "estimated_time": "30-60 seconds",
                    "route": ["internal_pool"]
                }
            except Exception as e:
                logger.error(f"Quote calculation failed: {e}")
                return {"error": "Unable to calculate quote"}
        
        return {"error": "No liquidity available for this pair"}
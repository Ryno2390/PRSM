"""
CHRONOS API Endpoints

FastAPI endpoints for the CHRONOS clearing protocol.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
import logging

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from .clearing_engine import ChronosEngine
from .wallet_manager import MultiSigWalletManager
from .exchange_router import ExchangeRouter
from .models import AssetType, SwapRequest, SwapType, TransactionStatus
from ..core.ipfs_client import IPFSClient
from ..auth import get_current_user


logger = logging.getLogger(__name__)

# Initialize CHRONOS components
wallet_manager = MultiSigWalletManager()
exchange_router = ExchangeRouter()
ipfs_client = IPFSClient()  # This would be properly configured
chronos_engine = ChronosEngine(wallet_manager, exchange_router, ipfs_client)

router = APIRouter(prefix="/chronos", tags=["CHRONOS Clearing Protocol"])


# Request/Response Models
class SwapRequestInput(BaseModel):
    """Input model for swap requests."""
    from_asset: AssetType
    to_asset: AssetType
    from_amount: Decimal
    max_slippage: Optional[Decimal] = Field(default=Decimal("0.005"))
    expires_in_minutes: Optional[int] = Field(default=60)


class QuoteRequest(BaseModel):
    """Request model for price quotes."""
    from_asset: AssetType
    to_asset: AssetType
    amount: Decimal


class SignTransactionRequest(BaseModel):
    """Request model for signing multi-sig transactions."""
    transaction_id: str
    signer_id: str


# API Endpoints
@router.get("/health")
async def health_check():
    """Health check for CHRONOS service."""
    return {
        "status": "healthy",
        "service": "CHRONOS Clearing Protocol",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0-poc"
    }


@router.post("/quote")
async def get_quote(request: QuoteRequest):
    """Get price quote for asset swap."""
    try:
        quote = await chronos_engine.get_quote(
            request.from_asset,
            request.to_asset,
            request.amount
        )
        
        return {
            "success": True,
            "quote": quote,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Quote request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/swap")
async def submit_swap(
    request: SwapRequestInput,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Submit a new swap request."""
    try:
        # Create swap request
        swap_request = SwapRequest(
            user_id=current_user.id,
            from_asset=request.from_asset,
            to_asset=request.to_asset,
            from_amount=request.from_amount,
            swap_type=SwapType(f"{request.from_asset.value}_TO_{request.to_asset.value}"),
            max_slippage=request.max_slippage,
            expires_at=datetime.utcnow() + timedelta(minutes=request.expires_in_minutes)
        )
        
        # Submit to clearing engine
        transaction = await chronos_engine.submit_swap_request(swap_request)
        
        return {
            "success": True,
            "transaction_id": transaction.id,
            "status": transaction.status.value,
            "swap_request": swap_request.dict(),
            "message": "Swap request submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Swap submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/swap/{transaction_id}")
async def get_swap_status(transaction_id: str):
    """Get status of a swap transaction."""
    try:
        transaction = await chronos_engine.get_transaction_status(transaction_id)
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        response = {
            "transaction_id": transaction.id,
            "status": transaction.status.value,
            "created_at": transaction.created_at.isoformat(),
            "updated_at": transaction.updated_at.isoformat(),
            "swap_request": transaction.swap_request.dict()
        }
        
        if transaction.settlement:
            response["settlement"] = transaction.settlement.dict()
        
        if transaction.exchange_route:
            response["exchange_route"] = transaction.exchange_route
        
        if transaction.blockchain_txids:
            response["blockchain_txids"] = transaction.blockchain_txids
        
        if transaction.error_message:
            response["error_message"] = transaction.error_message
        
        if transaction.completed_at:
            response["completed_at"] = transaction.completed_at.isoformat()
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets")
async def get_wallet_info():
    """Get information about CHRONOS reserve wallets."""
    try:
        wallets = {}
        
        for asset_type in AssetType:
            wallet_info = await wallet_manager.get_wallet_info(asset_type)
            wallets[asset_type.value] = wallet_info
        
        return {
            "success": True,
            "wallets": wallets,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Wallet info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{asset_type}")
async def get_asset_wallet_info(asset_type: AssetType):
    """Get detailed information for specific asset wallet."""
    try:
        wallet_info = await wallet_manager.get_wallet_info(asset_type)
        
        if "error" in wallet_info:
            raise HTTPException(status_code=404, detail=wallet_info["error"])
        
        return {
            "success": True,
            "wallet": wallet_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Asset wallet info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{asset_type}/transactions")
async def get_pending_transactions(asset_type: Optional[AssetType] = None):
    """Get pending multi-sig transactions."""
    try:
        pending = await wallet_manager.get_pending_transactions(asset_type)
        
        return {
            "success": True,
            "pending_transactions": pending,
            "count": len(pending),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pending transactions request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/wallets/sign")
async def sign_transaction(
    request: SignTransactionRequest,
    current_user = Depends(get_current_user)
):
    """Sign a pending multi-sig transaction."""
    try:
        # In production, verify signer authorization
        success = await wallet_manager.sign_transaction(
            request.transaction_id,
            request.signer_id
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to sign transaction")
        
        return {
            "success": True,
            "transaction_id": request.transaction_id,
            "signer_id": request.signer_id,
            "message": "Transaction signed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transaction signing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exchanges")
async def get_exchange_status():
    """Get status of connected exchanges."""
    try:
        status = await exchange_router.get_exchange_status()
        
        return {
            "success": True,
            "exchanges": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Exchange status request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exchanges/prices/{from_asset}/{to_asset}")
async def get_exchange_prices(
    from_asset: AssetType,
    to_asset: AssetType,
    amount: Decimal
):
    """Get prices from all exchanges for comparison."""
    try:
        best_price = await exchange_router.get_best_price(from_asset, to_asset, amount)
        
        return {
            "success": True,
            "best_price": best_price,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Exchange prices request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exchanges/routes/{from_asset}/{to_asset}")
async def get_optimal_routes(
    from_asset: AssetType,
    to_asset: AssetType,
    amount: Decimal
):
    """Get optimal trading routes for asset pair."""
    try:
        routes = await exchange_router.get_optimal_route(from_asset, to_asset, amount)
        
        return {
            "success": True,
            "routes": routes,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Optimal routes request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/volume")
async def get_volume_analytics():
    """Get CHRONOS trading volume analytics."""
    try:
        # Mock analytics data for proof-of-concept
        analytics = {
            "24h_volume": {
                "FTNS": "2500000",
                "BTC": "15.5",
                "USD": "750000"
            },
            "7d_volume": {
                "FTNS": "18000000",
                "BTC": "110.2",
                "USD": "5200000"
            },
            "total_swaps_24h": 1247,
            "total_swaps_7d": 8934,
            "average_swap_time": "45 seconds",
            "total_fees_collected": {
                "24h": "2250.50",
                "7d": "15680.25"
            },
            "most_popular_pairs": [
                {"pair": "FTNS-USD", "volume": "1800000", "count": 567},
                {"pair": "FTNS-BTC", "volume": "700000", "count": 234},
                {"pair": "BTC-USD", "volume": "450000", "count": 123}
            ]
        }
        
        return {
            "success": True,
            "analytics": analytics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_chronos_status():
    """Get comprehensive CHRONOS system status."""
    try:
        # Get system status
        wallet_status = {}
        for asset_type in AssetType:
            wallet_info = await wallet_manager.get_wallet_info(asset_type)
            wallet_status[asset_type.value] = {
                "available_balance": wallet_info.get("available_balance", "0"),
                "reserved_balance": wallet_info.get("reserved_balance", "0"),
                "total_balance": wallet_info.get("total_balance", "0")
            }
        
        exchange_status = await exchange_router.get_exchange_status()
        active_exchanges = [name for name, info in exchange_status.items() if info["is_active"]]
        
        return {
            "success": True,
            "status": {
                "service": "CHRONOS Clearing Protocol",
                "version": "1.0.0-poc",
                "uptime": "99.9%",  # Mock data
                "active_exchanges": active_exchanges,
                "wallet_reserves": wallet_status,
                "total_active_transactions": len(chronos_engine.active_transactions),
                "system_health": "healthy"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
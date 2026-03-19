"""
FTNS Token API Routes

Provides REST API endpoints for FTNS token management including:
- Balance queries
- Token transfers
- Transaction history
"""

import structlog
from typing import Optional
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from prsm.interface.auth import get_current_user, User
from prsm.core.database import FTNSQueries

logger = structlog.get_logger(__name__)

# Create router with /api/v1/ftns prefix
router = APIRouter(prefix="/api/v1/ftns", tags=["FTNS"])


# ============================================================================
# Request/Response Models
# ============================================================================

class FTNSTransferRequest(BaseModel):
    """Request model for FTNS token transfer"""
    recipient: str
    amount: float
    description: str = ""


class FTNSBalanceResponse(BaseModel):
    """Response model for FTNS balance"""
    user_id: str
    balance: float
    locked_balance: float
    available_balance: float


class FTNSTransferResponse(BaseModel):
    """Response model for FTNS transfer"""
    status: str
    amount: float
    recipient: str


class FTNSTransactionsResponse(BaseModel):
    """Response model for transaction history"""
    transactions: list
    user_id: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/balance", response_model=FTNSBalanceResponse)
async def get_balance(current_user: User = Depends(get_current_user)):
    """Get the calling user's FTNS balance."""
    try:
        balance = await FTNSQueries.get_user_balance(str(current_user.id))
        return FTNSBalanceResponse(
            user_id=str(current_user.id),
            balance=balance["balance"],
            locked_balance=balance["locked_balance"],
            available_balance=balance["available_balance"],
        )
    except Exception as e:
        logger.error("Balance query failed", user_id=str(current_user.id), error=str(e))
        raise HTTPException(status_code=503, detail="FTNS service temporarily unavailable")


@router.post("/transfer", response_model=FTNSTransferResponse)
async def transfer_tokens(
    request: FTNSTransferRequest,
    current_user: User = Depends(get_current_user)
):
    """Transfer FTNS tokens to another user."""
    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")

    # Idempotency key includes a uuid so retried requests don't double-transfer
    idempotency_key = f"transfer:{current_user.id}:{request.recipient}:{uuid4().hex}"

    try:
        result = await FTNSQueries.execute_atomic_transfer(
            from_user_id=str(current_user.id),
            to_user_id=request.recipient,
            amount=request.amount,
            idempotency_key=idempotency_key,
            description=request.description or f"Transfer to {request.recipient}",
        )
    except Exception as e:
        logger.error("Transfer failed", user_id=str(current_user.id), error=str(e))
        raise HTTPException(status_code=503, detail="FTNS service temporarily unavailable")

    if not result["success"]:
        error = result.get("error_message", "Transfer failed")
        status_code = 402 if "insufficient" in error.lower() else 400
        raise HTTPException(status_code=status_code, detail=error)

    return FTNSTransferResponse(
        status="success",
        amount=request.amount,
        recipient=request.recipient,
    )


@router.get("/transactions", response_model=FTNSTransactionsResponse)
async def get_transactions(
    current_user: User = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None),
):
    """Get transaction history for the current user."""
    try:
        transactions = await FTNSQueries.get_user_transactions(
            user_id=str(current_user.id),
            limit=limit,
            search=search,
        )
    except Exception as e:
        logger.error("Transaction history query failed",
                     user_id=str(current_user.id), error=str(e))
        raise HTTPException(status_code=503, detail="FTNS service temporarily unavailable")

    return FTNSTransactionsResponse(
        transactions=transactions,
        user_id=str(current_user.id),
    )

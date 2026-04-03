"""
Settler Registry API Routes — Phase 6 Governance & Staking
===========================================================

REST endpoints for settler management, multi-sig settlement,
and ledger export (challenge system).

Simple L2-style security for batch settlement.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

logger = __import__("structlog").get_logger(__name__)

router = APIRouter(prefix="/settler", tags=["settler", "governance", "phase6"])


# === Request/Response Models ===

class RegisterSettlerRequest(BaseModel):
    """Request to register as a batch settler."""
    settler_id: str = Field(..., description="Unique identifier for the settler")
    address: str = Field(..., description="Ethereum address (0x...) for on-chain ops")
    bond_amount: float = Field(..., gt=0, description="FTNS to stake as bond")


class UnbondRequest(BaseModel):
    """Request to unbond a settler."""
    settler_id: str


class SignBatchRequest(BaseModel):
    """Request to sign a pending batch."""
    batch_id: str
    settler_id: str
    signature: str = Field(..., description="Cryptographic signature on batch hash")


class ProposeSlashRequest(BaseModel):
    """Request to propose slashing a settler."""
    settler_id: str
    slash_amount: float = Field(..., gt=0)
    reason: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    proposer_id: str


class SettlerResponse(BaseModel):
    """Response with settler details."""
    settler_id: str
    address: str
    bond_amount: float
    status: str
    staked_at: str
    unbonding_at: Optional[str] = None
    total_settled: int
    total_volume: float
    slashed_amount: float
    can_settle: bool


class BatchResponse(BaseModel):
    """Response with batch details."""
    batch_id: str
    batch_hash: str
    transfer_count: int
    total_amount: float
    signature_count: int
    threshold: int
    approved: bool
    created_at: str
    signatures: List[Dict[str, Any]]


class LedgerExportResponse(BaseModel):
    """Response with ledger export."""
    exported_at: str
    integrity_hash: str
    settlers: List[Dict[str, Any]]
    pending_batches: List[Dict[str, Any]]


# === Settler Routes ===

@router.post("/register", response_model=SettlerResponse)
async def register_settler(request: RegisterSettlerRequest):
    """
    Register as a batch settler with staked bond.
    
    Requires staking the minimum bond amount to participate in
    multi-sig batch settlement approval.
    
    - **settler_id**: Unique identifier (e.g., node ID)
    - **address**: Ethereum address for on-chain operations
    - **bond_amount**: FTNS to stake (must meet minimum)
    """
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    
    try:
        settler = await registry.register_settler(
            settler_id=request.settler_id,
            address=request.address,
            bond_amount=request.bond_amount,
        )
        
        return SettlerResponse(
            settler_id=settler.settler_id,
            address=settler.address,
            bond_amount=settler.bond_amount,
            status=settler.status.value,
            staked_at=settler.staked_at.isoformat(),
            unbonding_at=settler.unbonding_at.isoformat() if settler.unbonding_at else None,
            total_settled=settler.total_settled,
            total_volume=settler.total_volume,
            slashed_amount=settler.slashed_amount,
            can_settle=settler.can_settle,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/unbond", response_model=Dict[str, Any])
async def unbond_settler(request: UnbondRequest):
    """
    Initiate unbonding for a settler.
    
    The settler will be inactive during the lock period (30 days default),
    after which the bond can be withdrawn.
    """
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    
    try:
        unbond_at = await registry.unbond_settler(request.settler_id)
        
        return {
            "settler_id": request.settler_id,
            "status": "unbonding",
            "unbond_at": unbond_at.isoformat() if unbond_at else None,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/withdraw", response_model=Dict[str, Any])
async def withdraw_bond(settler_id: str):
    """
    Withdraw bond after unbonding period completes.
    """
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    
    try:
        amount = await registry.withdraw_bond(settler_id)
        
        return {
            "settler_id": settler_id,
            "withdrawn_amount": amount,
            "status": "inactive",
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{settler_id}", response_model=SettlerResponse)
async def get_settler(settler_id: str):
    """Get details of a specific settler."""
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    settler = registry.get_settler(settler_id)
    
    if not settler:
        raise HTTPException(404, f"Settler {settler_id} not found")
    
    return SettlerResponse(
        settler_id=settler.settler_id,
        address=settler.address,
        bond_amount=settler.bond_amount,
        status=settler.status.value,
        staked_at=settler.staked_at.isoformat(),
        unbonding_at=settler.unbonding_at.isoformat() if settler.unbonding_at else None,
        total_settled=settler.total_settled,
        total_volume=settler.total_volume,
        slashed_amount=settler.slashed_amount,
        can_settle=settler.can_settle,
    )


@router.get("/list/active", response_model=List[SettlerResponse])
async def list_active_settlers():
    """List all active settlers."""
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    settlers = registry.list_active_settlers()
    
    return [
        SettlerResponse(
            settler_id=s.settler_id,
            address=s.address,
            bond_amount=s.bond_amount,
            status=s.status.value,
            staked_at=s.staked_at.isoformat(),
            unbonding_at=s.unbonding_at.isoformat() if s.unbonding_at else None,
            total_settled=s.total_settled,
            total_volume=s.total_volume,
            slashed_amount=s.slashed_amount,
            can_settle=s.can_settle,
        )
        for s in settlers
    ]


# === Batch Routes ===

@router.post("/batch/sign", response_model=Dict[str, Any])
async def sign_batch(request: SignBatchRequest):
    """
    Sign a pending batch for multi-sig approval.
    
    Once the signature threshold is reached (default: 3), the batch
    is approved for on-chain settlement.
    """
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    
    try:
        signature = await registry.sign_batch(
            batch_id=request.batch_id,
            settler_id=request.settler_id,
            signature=request.signature,
        )
        
        batch = registry.get_pending_batch(request.batch_id)
        
        return {
            "batch_id": request.batch_id,
            "settler_id": request.settler_id,
            "signed_at": signature.signed_at.isoformat(),
            "signature_count": batch.signature_count if batch else 1,
            "threshold": registry.settlement_threshold,
            "approved": registry.is_batch_approved(request.batch_id),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/batch/{batch_id}", response_model=BatchResponse)
async def get_batch(batch_id: str):
    """Get details of a pending batch."""
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    batch = registry.get_pending_batch(batch_id)
    
    if not batch:
        raise HTTPException(404, f"Batch {batch_id} not found")
    
    return BatchResponse(
        batch_id=batch.batch_id,
        batch_hash=batch.batch_hash,
        transfer_count=len(batch.transfers),
        total_amount=batch.total_amount,
        signature_count=batch.signature_count,
        threshold=registry.settlement_threshold,
        approved=batch.signature_count >= registry.settlement_threshold,
        created_at=batch.created_at.isoformat(),
        signatures=[s.to_dict() for s in batch.signatures],
    )


@router.get("/batch/list/pending", response_model=List[BatchResponse])
async def list_pending_batches():
    """List all batches awaiting approval."""
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    batches = registry.list_pending_batches()
    
    return [
        BatchResponse(
            batch_id=b.batch_id,
            batch_hash=b.batch_hash,
            transfer_count=len(b.transfers),
            total_amount=b.total_amount,
            signature_count=b.signature_count,
            threshold=registry.settlement_threshold,
            approved=b.signature_count >= registry.settlement_threshold,
            created_at=b.created_at.isoformat(),
            signatures=[s.to_dict() for s in b.signatures],
        )
        for b in batches
        if not b.settled
    ]


# === Ledger Export ===

@router.get("/ledger/export", response_model=Dict[str, Any])
async def export_ledger():
    """
    Export local ledger state for public audit.
    
    This enables the "Challenge" mechanism where anyone can
    compare local state against on-chain settlement.
    """
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    
    # Gather ledger data from node
    ledger_data = {}
    if hasattr(node, "ledger"):
        ledger_data = {
            "balances": dict(node.ledger._balances) if hasattr(node.ledger, "_balances") else {},
            "total_supply": node.ledger.total_supply if hasattr(node.ledger, "total_supply") else 0,
            "transaction_count": len(node.ledger._transactions) if hasattr(node.ledger, "_transactions") else 0,
        }
    
    export = await registry.export_ledger(ledger_data)
    return export


# === Slashing ===

@router.post("/slash/propose", response_model=Dict[str, Any])
async def propose_slash(request: ProposeSlashRequest):
    """
    Propose to slash a settler's bond.
    
    This creates a governance proposal that must be voted on
    before the slash can be executed.
    """
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    
    try:
        proposal = await registry.propose_slash(
            settler_id=request.settler_id,
            slash_amount=request.slash_amount,
            reason=request.reason,
            evidence=request.evidence,
            proposer_id=request.proposer_id,
        )
        
        return {
            "proposal_id": proposal.proposal_id,
            "settler_id": proposal.settler_id,
            "slash_amount": proposal.slash_amount,
            "reason": proposal.reason,
            "created_at": proposal.created_at.isoformat(),
            "status": "pending_vote",
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/slash/{proposal_id}/execute", response_model=Dict[str, Any])
async def execute_slash(proposal_id: str):
    """
    Execute an approved slash proposal.
    
    In production, this would be gated by governance vote confirmation.
    """
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    
    try:
        amount = await registry.execute_slash(proposal_id)
        
        return {
            "proposal_id": proposal_id,
            "executed": True,
            "slashed_amount": amount,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


# === Stats ===

@router.get("/stats", response_model=Dict[str, Any])
async def get_settler_stats():
    """Get settler registry statistics."""
    from prsm.node.node import PRSMNode
    
    node = PRSMNode.get_instance()
    if not node or not hasattr(node, "_settler_registry"):
        raise HTTPException(503, "Settler registry not initialized")
    
    registry = node._settler_registry
    return registry.get_stats()

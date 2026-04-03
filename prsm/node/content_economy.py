"""
Content Economy - FTNS Payment Integration for Content Access
==============================================================

Wires together content uploads, retrieval, and FTNS payments.
Implements the Phase 4 content economy requirements:

1. FTNS payment on content access
2. Royalty distribution (8% original, 1% derivative)
3. Replication guarantees with minimum replica tracking
4. Content retrieval marketplace with provider bidding
5. Vector DB integration for semantic search

Integration Points:
- ContentUploader: Triggers payments on access
- ContentProvider: Handles retrieval with payment
- LocalLedger: FTNS accounting
- OnChainFTNSLedger: Base mainnet FTNS transfers
- ContentIndex: Provider discovery
- VectorStore: Semantic indexing
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.gossip import GossipProtocol

if TYPE_CHECKING:
    from prsm.node.identity import NodeIdentity
    from prsm.node.content_index import ContentIndex
    from prsm.economy.ftns_onchain import OnChainFTNSLedger

logger = logging.getLogger(__name__)


# ── Royalty Configuration ─────────────────────────────────────────────────

# Phase 4 spec: 8% original creator, 1% derivative creators
ORIGINAL_CREATOR_ROYALTY_RATE = 0.08  # 8% to original creator
DERIVATIVE_CREATOR_ROYALTY_RATE = 0.01  # 1% to each derivative creator
NETWORK_FEE_RATE = 0.02  # 2% network fee
MAX_ROYALTY_CHAIN_DEPTH = 5  # Max derivative chain depth to track

# Alternative pricing model (existing 70/25/5 split for backward compatibility)
LEGACY_DERIVATIVE_SHARE = 0.70
LEGACY_SOURCE_SHARE = 0.25
LEGACY_NETWORK_SHARE = 0.05


class RoyaltyModel(str, Enum):
    """Royalty distribution models."""
    PHASE4 = "phase4"  # 8% original, 1% derivative, 2% network
    LEGACY = "legacy"  # 70/25/5 split


class PaymentStatus(str, Enum):
    """Status of a content access payment."""
    PENDING = "pending"
    ESCROWED = "escrowed"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class ContentAccessPayment:
    """Tracks a single content access payment."""
    payment_id: str
    cid: str
    accessor_id: str
    creator_id: str
    amount: Decimal
    royalty_model: RoyaltyModel
    status: PaymentStatus = PaymentStatus.PENDING
    escrow_tx_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    royalty_distributions: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ReplicationStatus:
    """Tracks replication status for a CID."""
    cid: str
    min_replicas: int
    current_replicas: int
    providers: Set[str] = field(default_factory=set)
    last_verified: float = field(default_factory=time.time)
    pending_requests: int = 0
    verification_failures: int = 0


@dataclass
class ProviderBid:
    """A bid from a storage provider for content retrieval."""
    provider_id: str
    cid: str
    price_ftns: Decimal
    estimated_latency_ms: float
    available_bandwidth_mbps: float
    reputation_score: float
    bid_timestamp: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 60.0)


@dataclass
class RetrievalRequest:
    """A content retrieval request with bidding."""
    request_id: str
    cid: str
    requester_id: str
    max_price_ftns: Decimal
    min_replicas_required: int = 1
    status: str = "open"  # open, bidding, fulfilled, failed
    bids: List[ProviderBid] = field(default_factory=list)
    selected_provider: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    bid_deadline: float = field(default_factory=lambda: time.time() + 10.0)


# ── Content Economy Manager ───────────────────────────────────────────────

class ContentEconomy:
    """Manages FTNS payments, royalties, and content marketplace.
    
    Responsibilities:
    1. Process FTNS payments on content access
    2. Distribute royalties to creators (original + derivative chain)
    3. Track and enforce replication guarantees
    4. Manage content retrieval marketplace with provider bidding
    5. Index content into vector store for semantic search
    """
    
    def __init__(
        self,
        identity: "NodeIdentity",
        ledger: LocalLedger,
        gossip: GossipProtocol,
        content_index: "ContentIndex",
        ftns_ledger: Optional["OnChainFTNSLedger"] = None,
        royalty_model: RoyaltyModel = RoyaltyModel.PHASE4,
        min_replicas: int = 3,
        replication_check_interval: float = 300.0,
        vector_store: Optional[Any] = None,
        embedding_fn: Optional[Callable] = None,
    ):
        self.identity = identity
        self.ledger = ledger
        self.gossip = gossip
        self.content_index = content_index
        self.ftns_ledger = ftns_ledger
        self.royalty_model = royalty_model
        self.min_replicas = min_replicas
        self.replication_check_interval = replication_check_interval
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn
        
        # In-memory tracking
        self._pending_payments: Dict[str, ContentAccessPayment] = {}
        self._replication_status: Dict[str, ReplicationStatus] = {}
        self._retrieval_requests: Dict[str, RetrievalRequest] = {}
        self._provider_reputation: Dict[str, float] = {}
        
        # Multi-party escrow for batch settlements
        self._escrow: Optional[Any] = None  # Set via set_escrow()
        
        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
    async def start(self) -> None:
        """Start background tasks for replication monitoring."""
        self._running = True
        
        # Start replication check loop
        self._tasks.append(asyncio.create_task(self._replication_monitor_loop()))
        
        # Start bid cleanup loop
        self._tasks.append(asyncio.create_task(self._bid_cleanup_loop()))
        
        # Register gossip handlers
        self.gossip.subscribe("retrieval_bid", self._on_retrieval_bid)
        self.gossip.subscribe("retrieval_fulfill", self._on_retrieval_fulfill)
        
        logger.info(
            f"Content economy started (royalty_model={self.royalty_model.value}, "
            f"min_replicas={self.min_replicas})"
        )
    
    def set_escrow(self, escrow: Any) -> None:
        """Set the multi-party escrow for batch settlements."""
        self._escrow = escrow
    
    async def stop(self) -> None:
        """Stop background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        
    # ── Content Access Payment ─────────────────────────────────────────────
    
    async def process_content_access(
        self,
        cid: str,
        accessor_id: str,
        content_metadata: Dict[str, Any],
    ) -> ContentAccessPayment:
        """Process FTNS payment for content access.
        
        Flow:
        1. Calculate total payment based on royalty_rate
        2. Lock FTNS in escrow (on-chain if available)
        3. Resolve provenance chain (original + derivatives)
        4. Distribute royalties according to model
        5. Record payment completion
        
        Args:
            cid: Content identifier
            accessor_id: Node/user accessing the content
            content_metadata: Metadata including royalty_rate, creator_id, parent_cids
            
        Returns:
            ContentAccessPayment with status and distribution details
        """
        payment_id = f"pay-{uuid.uuid4().hex[:12]}"
        
        # Get royalty rate from metadata or default
        royalty_rate = content_metadata.get("royalty_rate", 0.01)
        creator_id = content_metadata.get("creator_id", "")
        parent_cids = content_metadata.get("parent_cids", [])
        
        # Calculate total payment amount
        # For Phase4: base access fee is the royalty_rate
        # For Legacy: royalty_rate is already the per-access fee
        total_amount = Decimal(str(royalty_rate))
        
        payment = ContentAccessPayment(
            payment_id=payment_id,
            cid=cid,
            accessor_id=accessor_id,
            creator_id=creator_id,
            amount=total_amount,
            royalty_model=self.royalty_model,
            status=PaymentStatus.PENDING,
        )
        
        self._pending_payments[payment_id] = payment
        
        try:
            # Step 1: Lock FTNS in escrow
            if self.ftns_ledger and accessor_id == self.identity.node_id:
                # On-chain escrow for our own accesses
                escrow_result = await self.ftns_ledger.lock_escrow(
                    amount=float(total_amount),
                    purpose=f"content_access:{cid}",
                )
                if escrow_result.get("success"):
                    payment.escrow_tx_hash = escrow_result.get("tx_hash")
                    payment.status = PaymentStatus.ESCROWED
                else:
                    # Fall back to local ledger debit
                    await self.ledger.debit(
                        wallet_id=accessor_id,
                        amount=float(total_amount),
                        tx_type=TransactionType.COMPUTE_PAYMENT,
                        description=f"Content access: {cid[:12]}...",
                    )
            else:
                # Local ledger for cross-node or non-on-chain payments
                if accessor_id == self.identity.node_id:
                    await self.ledger.debit(
                        wallet_id=accessor_id,
                        amount=float(total_amount),
                        tx_type=TransactionType.COMPUTE_PAYMENT,
                        description=f"Content access: {cid[:12]}...",
                    )
                # Remote accessors pay via their own node - we just track
            
            # Step 2: Distribute royalties
            distributions = await self._distribute_royalties(
                payment=payment,
                creator_id=creator_id,
                parent_cids=parent_cids,
            )
            payment.royalty_distributions = distributions
            
            # Step 3: Release escrow (if on-chain)
            if payment.escrow_tx_hash and self.ftns_ledger:
                # Sum of distributed amounts goes to creators
                total_distributed = sum(d["amount"] for d in distributions)
                await self.ftns_ledger.release_escrow(
                    escrow_id=payment.escrow_tx_hash,
                    recipient=creator_id,  # Simplified - real impl would multi-sig
                    amount=float(total_distributed),
                )
            
            payment.status = PaymentStatus.COMPLETED
            payment.completed_at = time.time()
            
            logger.info(
                f"Content access payment completed: {payment_id} "
                f"({total_amount} FTNS for {cid[:12]}... by {accessor_id[:8]})"
            )
            
        except Exception as e:
            payment.status = PaymentStatus.FAILED
            payment.error = str(e)
            logger.error(f"Content access payment failed: {payment_id} - {e}")
            
            # Refund if escrowed
            if payment.escrow_tx_hash and self.ftns_ledger:
                try:
                    await self.ftns_ledger.release_escrow(
                        escrow_id=payment.escrow_tx_hash,
                        recipient=accessor_id,
                        amount=float(total_amount),
                    )
                    payment.status = PaymentStatus.REFUNDED
                except Exception:
                    pass
        
        return payment
    
    async def _distribute_royalties(
        self,
        payment: ContentAccessPayment,
        creator_id: str,
        parent_cids: List[str],
    ) -> List[Dict[str, Any]]:
        """Distribute royalties according to the configured model.
        
        Phase4 Model:
        - 8% to original creator (first in chain)
        - 1% to each derivative creator (up to MAX_ROYALTY_CHAIN_DEPTH)
        - 2% network fee
        - Remaining to direct creator
        
        Legacy Model:
        - 70% to derivative creator
        - 25% split among source creators
        - 5% network fee
        """
        distributions = []
        total_amount = payment.amount
        
        if self.royalty_model == RoyaltyModel.PHASE4:
            # Resolve provenance chain
            provenance_chain = await self._resolve_provenance_chain(
                cid=payment.cid,
                parent_cids=parent_cids,
            )
            
            # Original creator gets 8%
            if provenance_chain.original_creator:
                original_share = total_amount * Decimal(str(ORIGINAL_CREATOR_ROYALTY_RATE))
                distributions.append({
                    "recipient_id": provenance_chain.original_creator,
                    "amount": float(original_share),
                    "type": "original_creator",
                    "cid": provenance_chain.original_cid,
                })
                await self._credit_royalty(
                    recipient_id=provenance_chain.original_creator,
                    amount=float(original_share),
                    cid=payment.cid,
                    description=f"Original creator royalty (8%): {payment.cid[:12]}...",
                )
            
            # Derivative creators get 1% each
            for i, derivative in enumerate(provenance_chain.derivative_creators[:MAX_ROYALTY_CHAIN_DEPTH]):
                derivative_share = total_amount * Decimal(str(DERIVATIVE_CREATOR_ROYALTY_RATE))
                distributions.append({
                    "recipient_id": derivative["creator_id"],
                    "amount": float(derivative_share),
                    "type": "derivative_creator",
                    "depth": derivative["depth"],
                    "cid": derivative["cid"],
                })
                await self._credit_royalty(
                    recipient_id=derivative["creator_id"],
                    amount=float(derivative_share),
                    cid=payment.cid,
                    description=f"Derivative royalty (1%, depth={derivative['depth']}): {payment.cid[:12]}...",
                )
            
            # Network fee
            network_fee = total_amount * Decimal(str(NETWORK_FEE_RATE))
            distributions.append({
                "recipient_id": "system",
                "amount": float(network_fee),
                "type": "network_fee",
            })
            await self.ledger.credit(
                wallet_id="system",
                amount=float(network_fee),
                tx_type=TransactionType.CONTENT_ROYALTY,
                description=f"Network fee: {payment.cid[:12]}...",
            )
            
            # Direct creator gets remainder
            distributed = sum(d["amount"] for d in distributions)
            remainder = float(total_amount) - distributed
            if remainder > 0:
                distributions.append({
                    "recipient_id": creator_id,
                    "amount": remainder,
                    "type": "direct_creator",
                })
                await self._credit_royalty(
                    recipient_id=creator_id,
                    amount=remainder,
                    cid=payment.cid,
                    description=f"Direct creator royalty: {payment.cid[:12]}...",
                )
                
        else:  # Legacy model
            if parent_cids:
                # Derivative work - split royalties
                derivative_share = total_amount * Decimal(str(LEGACY_DERIVATIVE_SHARE))
                source_pool = total_amount * Decimal(str(LEGACY_SOURCE_SHARE))
                network_fee = total_amount * Decimal(str(LEGACY_NETWORK_SHARE))
                
                # Credit derivative creator (this node)
                distributions.append({
                    "recipient_id": creator_id,
                    "amount": float(derivative_share),
                    "type": "derivative_creator",
                })
                await self._credit_royalty(
                    recipient_id=creator_id,
                    amount=float(derivative_share),
                    cid=payment.cid,
                    description=f"Derivative royalty (70%): {payment.cid[:12]}...",
                )
                
                # Split source pool among parent creators
                parent_creators = await self._resolve_parent_creators(parent_cids)
                if parent_creators:
                    per_parent = float(source_pool) / len(parent_creators)
                    for parent_creator_id in parent_creators:
                        distributions.append({
                            "recipient_id": parent_creator_id,
                            "amount": per_parent,
                            "type": "source_creator",
                        })
                        await self._credit_royalty(
                            recipient_id=parent_creator_id,
                            amount=per_parent,
                            cid=payment.cid,
                            description=f"Source royalty: {payment.cid[:12]}...",
                        )
                else:
                    # No resolvable parents - derivative creator gets source pool too
                    distributions.append({
                        "recipient_id": creator_id,
                        "amount": float(source_pool),
                        "type": "unclaimed_source",
                    })
                    await self._credit_royalty(
                        recipient_id=creator_id,
                        amount=float(source_pool),
                        cid=payment.cid,
                        description=f"Unclaimed source pool: {payment.cid[:12]}...",
                    )
                
                # Network fee
                distributions.append({
                    "recipient_id": "system",
                    "amount": float(network_fee),
                    "type": "network_fee",
                })
                await self.ledger.credit(
                    wallet_id="system",
                    amount=float(network_fee),
                    tx_type=TransactionType.CONTENT_ROYALTY,
                    description=f"Network fee: {payment.cid[:12]}...",
                )
            else:
                # Original work - full royalty to creator
                distributions.append({
                    "recipient_id": creator_id,
                    "amount": float(total_amount),
                    "type": "original_creator",
                })
                await self._credit_royalty(
                    recipient_id=creator_id,
                    amount=float(total_amount),
                    cid=payment.cid,
                    description=f"Content royalty: {payment.cid[:12]}...",
                )
        
        # Accumulate to escrow for batch on-chain settlement (Phase 4)
        if self._escrow:
            try:
                from prsm.node.multi_party_escrow import ContentEconomyEscrowBridge
                for dist in distributions:
                    if dist["recipient_id"] != "system":
                        await self._escrow.accumulate(
                            creator_id=dist["recipient_id"],
                            amount=dist["amount"],
                            source_cid=payment.cid,
                            accessor_id=payment.accessor_id,
                        )
            except Exception as e:
                logger.debug(f"Escrow accumulation failed: {e}")
        
        return distributions
    
    async def _credit_royalty(
        self,
        recipient_id: str,
        amount: float,
        cid: str,
        description: str,
    ) -> None:
        """Credit royalty to a recipient via local ledger and optionally on-chain."""
        # Local ledger credit
        await self.ledger.credit(
            wallet_id=recipient_id,
            amount=amount,
            tx_type=TransactionType.CONTENT_ROYALTY,
            description=description,
        )
        
        # On-chain transfer if available and we know recipient's address
        # (simplified - real impl would need address resolution)
        if self.ftns_ledger and recipient_id == self.identity.node_id:
            # Self-royalty already credited locally
            pass
        # For remote recipients, they'll receive via their own ledger sync
    
    async def _resolve_provenance_chain(
        self,
        cid: str,
        parent_cids: List[str],
    ) -> "ProvenanceChain":
        """Resolve the full provenance chain for royalty distribution."""
        chain = ProvenanceChain()
        
        # Track original creator (deepest ancestor)
        visited: Set[str] = set()
        
        async def trace_ancestors(current_cid: str, depth: int) -> None:
            if current_cid in visited or depth > MAX_ROYALTY_CHAIN_DEPTH:
                return
            visited.add(current_cid)
            
            record = self.content_index.lookup(current_cid)
            if not record:
                return
            
            if depth == 0 and not parent_cids:
                # This is the content itself - the direct creator
                chain.direct_creator = record.creator_id
            
            if record.parent_cids:
                for parent_cid in record.parent_cids:
                    parent_record = self.content_index.lookup(parent_cid)
                    if parent_record:
                        if depth == 0:
                            # First level parent - derivative relationship
                            chain.derivative_creators.append({
                                "creator_id": record.creator_id,
                                "cid": current_cid,
                                "depth": depth + 1,
                            })
                        await trace_ancestors(parent_cid, depth + 1)
            else:
                # No parents = original creator
                if chain.original_creator is None:
                    chain.original_creator = record.creator_id
                    chain.original_cid = current_cid
        
        # Start tracing from parents
        for parent_cid in parent_cids:
            await trace_ancestors(parent_cid, 0)
        
        # If no parents found, this is original content
        if not chain.original_creator:
            record = self.content_index.lookup(cid)
            if record:
                chain.original_creator = record.creator_id
                chain.original_cid = cid
        
        return chain
    
    async def _resolve_parent_creators(self, parent_cids: List[str]) -> List[str]:
        """Resolve creator IDs for parent CIDs."""
        creators = []
        for parent_cid in parent_cids:
            record = self.content_index.lookup(parent_cid)
            if record and record.creator_id and record.creator_id not in creators:
                creators.append(record.creator_id)
        return creators
    
    # ── Replication Management ─────────────────────────────────────────────
    
    async def track_content_upload(
        self,
        cid: str,
        size_bytes: int,
        replicas_requested: int,
    ) -> ReplicationStatus:
        """Start tracking replication status for newly uploaded content."""
        status = ReplicationStatus(
            cid=cid,
            min_replicas=max(self.min_replicas, replicas_requested),
            current_replicas=1,  # We have it locally
            providers={self.identity.node_id},
        )
        self._replication_status[cid] = status
        
        logger.info(
            f"Tracking replication for {cid[:12]}... "
            f"(min={status.min_replicas}, current={status.current_replicas})"
        )
        
        return status
    
    async def update_replication_status(
        self,
        cid: str,
        provider_id: str,
        has_content: bool,
    ) -> None:
        """Update replication status when a provider announces or removes content."""
        status = self._replication_status.get(cid)
        if not status:
            # Auto-create tracking for known content
            record = self.content_index.lookup(cid)
            if record:
                status = ReplicationStatus(
                    cid=cid,
                    min_replicas=self.min_replicas,
                    current_replicas=len(record.providers),
                    providers=record.providers.copy(),
                )
                self._replication_status[cid] = status
            else:
                return
        
        if has_content:
            if provider_id not in status.providers:
                status.providers.add(provider_id)
                status.current_replicas = len(status.providers)
                status.last_verified = time.time()
        else:
            status.providers.discard(provider_id)
            status.current_replicas = len(status.providers)
        
        # Check if we need more replicas
        await self._check_replication_needs(cid, status)
    
    async def _check_replication_needs(
        self,
        cid: str,
        status: ReplicationStatus,
    ) -> None:
        """Request additional replicas if below minimum."""
        if status.current_replicas >= status.min_replicas:
            return
        
        needed = status.min_replicas - status.current_replicas
        if needed <= 0 or status.pending_requests >= needed:
            return
        
        # Request additional storage via gossip
        status.pending_requests += needed
        
        await self.gossip.publish("storage_request", {
            "cid": cid,
            "size_bytes": 0,  # Would need to look up
            "requester_id": self.identity.node_id,
            "replicas_needed": needed,
            "priority": "high" if status.current_replicas == 0 else "normal",
        })
        
        logger.info(
            f"Requested {needed} additional replicas for {cid[:12]}... "
            f"(current={status.current_replicas}, min={status.min_replicas})"
        )
    
    async def _replication_monitor_loop(self) -> None:
        """Periodically check replication status and request more replicas if needed."""
        while self._running:
            await asyncio.sleep(self.replication_check_interval)
            
            try:
                for cid, status in list(self._replication_status.items()):
                    # Decay pending requests over time
                    if status.pending_requests > 0:
                        status.pending_requests = max(0, status.pending_requests - 1)
                    
                    await self._check_replication_needs(cid, status)
                    
            except Exception as e:
                logger.error(f"Replication monitor error: {e}")
    
    # ── Content Retrieval Marketplace ───────────────────────────────────────
    
    async def request_content_retrieval(
        self,
        cid: str,
        max_price_ftns: Decimal,
        timeout: float = 30.0,
    ) -> Optional[bytes]:
        """Request content from the network with marketplace bidding.
        
        Flow:
        1. Broadcast retrieval request with max price
        2. Collect bids from providers
        3. Select best provider (price + reputation + latency)
        4. Pay and retrieve content
        
        Args:
            cid: Content identifier to retrieve
            max_price_ftns: Maximum willing to pay
            timeout: Seconds to wait for bids
            
        Returns:
            Content bytes, or None if not available
        """
        request_id = f"ret-{uuid.uuid4().hex[:12]}"
        
        request = RetrievalRequest(
            request_id=request_id,
            cid=cid,
            requester_id=self.identity.node_id,
            max_price_ftns=max_price_ftns,
            bid_deadline=time.time() + min(timeout / 3, 10.0),
        )
        self._retrieval_requests[request_id] = request
        
        # Broadcast retrieval request
        await self.gossip.publish("retrieval_request", {
            "request_id": request_id,
            "cid": cid,
            "requester_id": self.identity.node_id,
            "max_price_ftns": float(max_price_ftns),
            "deadline": request.bid_deadline,
        })
        
        # Wait for bids
        await asyncio.sleep(min(timeout / 3, 10.0))
        
        if not request.bids:
            logger.debug(f"No bids received for retrieval request {request_id}")
            request.status = "failed"
            return None
        
        # Select best provider
        selected_bid = self._select_best_bid(request.bids, max_price_ftns)
        if not selected_bid:
            logger.debug(f"No suitable bids for retrieval request {request_id}")
            request.status = "failed"
            return None
        
        request.selected_provider = selected_bid.provider_id
        request.status = "fulfilled"
        
        # Process payment
        payment = await self.process_content_access(
            cid=cid,
            accessor_id=self.identity.node_id,
            content_metadata={
                "royalty_rate": float(selected_bid.price_ftns),
                "creator_id": selected_bid.provider_id,
                "parent_cids": [],
            },
        )
        
        if payment.status != PaymentStatus.COMPLETED:
            logger.error(f"Payment failed for retrieval {request_id}: {payment.error}")
            return None
        
        # Request content from selected provider
        # (ContentProvider will handle the actual retrieval)
        return None  # Actual retrieval handled by ContentProvider
    
    def _select_best_bid(
        self,
        bids: List[ProviderBid],
        max_price: Decimal,
    ) -> Optional[ProviderBid]:
        """Select the best bid based on price, reputation, and latency."""
        valid_bids = [
            b for b in bids
            if Decimal(str(b.price_ftns)) <= max_price
            and b.expires_at > time.time()
        ]
        
        if not valid_bids:
            return None
        
        # Score: lower price, higher reputation, lower latency
        def score(bid: ProviderBid) -> float:
            price_score = float(max_price - Decimal(str(bid.price_ftns))) / float(max_price) if max_price > 0 else 0
            rep_score = bid.reputation_score
            latency_score = max(0, 1 - (bid.estimated_latency_ms / 1000.0))  # Normalize to 0-1
            return price_score * 0.5 + rep_score * 0.3 + latency_score * 0.2
        
        return max(valid_bids, key=score)
    
    async def _on_retrieval_bid(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str,
    ) -> None:
        """Handle a bid for a retrieval request."""
        request_id = data.get("request_id")
        if not request_id:
            return
        
        request = self._retrieval_requests.get(request_id)
        if not request or request.status != "open":
            return
        
        bid = ProviderBid(
            provider_id=origin,
            cid=request.cid,
            price_ftns=Decimal(str(data.get("price_ftns", 0))),
            estimated_latency_ms=data.get("estimated_latency_ms", 100.0),
            available_bandwidth_mbps=data.get("available_bandwidth_mbps", 100.0),
            reputation_score=self._provider_reputation.get(origin, 0.5),
            expires_at=data.get("expires_at", time.time() + 60.0),
        )
        
        request.bids.append(bid)
        logger.debug(f"Received bid from {origin[:8]} for request {request_id}")
    
    async def _on_retrieval_fulfill(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str,
    ) -> None:
        """Handle content fulfillment from a provider."""
        request_id = data.get("request_id")
        # Actual content delivery handled by ContentProvider
        logger.debug(f"Retrieval fulfillment from {origin[:8]} for {request_id}")
    
    async def _bid_cleanup_loop(self) -> None:
        """Clean up expired retrieval requests."""
        while self._running:
            await asyncio.sleep(60.0)
            
            now = time.time()
            expired = [
                rid for rid, req in self._retrieval_requests.items()
                if req.bid_deadline < now and req.status == "open"
            ]
            
            for rid in expired:
                self._retrieval_requests[rid].status = "expired"
                del self._retrieval_requests[rid]
    
    # ── Vector DB Integration ───────────────────────────────────────────────
    
    async def index_content_embedding(
        self,
        cid: str,
        content: bytes,
        metadata: Dict[str, Any],
    ) -> bool:
        """Index content into vector store for semantic search.
        
        Args:
            cid: Content identifier
            content: Raw content bytes
            metadata: Content metadata (creator_id, royalty_rate, etc.)
            
        Returns:
            True if indexed successfully
        """
        if not self.vector_store or not self.embedding_fn:
            return False
        
        try:
            # Generate embedding
            text = content.decode("utf-8", errors="ignore").strip()
            if len(text) < 50:
                return False  # Too short for meaningful embedding
            
            embedding = await self.embedding_fn(text[:32_000])
            if embedding is None:
                return False
            
            # Store in vector DB
            await self.vector_store.upsert(
                content_cid=cid,
                embedding=embedding,
                metadata={
                    "creator_id": metadata.get("creator_id", ""),
                    "royalty_rate": metadata.get("royalty_rate", 0.01),
                    "content_type": metadata.get("content_type", "text"),
                    "filename": metadata.get("filename", ""),
                    **metadata,
                },
            )
            
            logger.debug(f"Indexed embedding for {cid[:12]}...")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to index embedding for {cid[:12]}...: {e}")
            return False
    
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.7,
        max_royalty_rate: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search content by semantic similarity.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            min_similarity: Minimum similarity score
            max_royalty_rate: Optional maximum royalty rate filter
            
        Returns:
            List of content matches with metadata
        """
        if not self.vector_store or not self.embedding_fn:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_fn(query)
            if query_embedding is None:
                return []
            
            # Search vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                limit=limit,
                filters={
                    "min_similarity": min_similarity,
                    "max_royalty_rate": max_royalty_rate,
                } if max_royalty_rate else None,
            )
            
            return [
                {
                    "cid": r.content_cid,
                    "similarity": r.similarity_score,
                    "creator_id": r.creator_id,
                    "royalty_rate": r.royalty_rate,
                    "metadata": r.metadata,
                }
                for r in results
                if r.similarity_score >= min_similarity
            ]
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []
    
    # ── Statistics ──────────────────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get content economy statistics."""
        return {
            "pending_payments": len(self._pending_payments),
            "tracked_content": len(self._replication_status),
            "active_retrieval_requests": len([
                r for r in self._retrieval_requests.values()
                if r.status == "open"
            ]),
            "royalty_model": self.royalty_model.value,
            "min_replicas": self.min_replicas,
            "vector_store_enabled": self.vector_store is not None,
        }


@dataclass
class ProvenanceChain:
    """Resolved provenance chain for royalty distribution."""
    original_creator: Optional[str] = None
    original_cid: Optional[str] = None
    derivative_creators: List[Dict[str, Any]] = field(default_factory=list)
    direct_creator: Optional[str] = None

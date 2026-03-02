"""
Agent Collaboration Protocols
=============================

Structured protocols for multi-agent collaboration over the PRSM P2P network.

Three protocol types:
1. **Task Delegation** — Agent A posts a task, agents bid, winner executes
2. **Peer Review** — Agent submits work, reviewers evaluate, consensus decides
3. **Knowledge Exchange** — Agent queries for information, responders are paid

All protocols build on the existing gossip/direct-message infrastructure,
the agent registry for discovery, and the ledger for payments.

Sprint 4 enhancements:
- FTNS escrow and payment on task completion, review submission, and query response
- Network-wide broadcasts for all state changes (assignment, completion, reviews, responses)
- Expiry enforcement with automatic escrow refund for timed-out collaborations
- Bounded memory with LRU eviction of completed/archived records
"""

import asyncio
import collections
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from prsm.node.gossip import (
    GOSSIP_TASK_ASSIGN,
    GOSSIP_TASK_CANCEL,
    GOSSIP_TASK_COMPLETE,
    GOSSIP_REVIEW_SUBMIT,
    GOSSIP_KNOWLEDGE_RESPONSE,
    GossipProtocol,
)

logger = logging.getLogger(__name__)

# Gossip subtypes for collaboration protocols
GOSSIP_TASK_OFFER = "agent_task_offer"
GOSSIP_TASK_BID = "agent_task_bid"
GOSSIP_REVIEW_REQUEST = "agent_review_request"
GOSSIP_KNOWLEDGE_QUERY = "agent_knowledge_query"

# Bounds for in-memory record retention
MAX_COMPLETED_RECORDS = 500
CLEANUP_INTERVAL_SECONDS = 60.0
DEFAULT_TASK_TIMEOUT = 3600.0       # 1 hour
DEFAULT_REVIEW_TIMEOUT = 3600.0     # 1 hour
DEFAULT_QUERY_TIMEOUT = 1800.0      # 30 minutes


class TaskStatus(str, Enum):
    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DISPUTED = "disputed"
    CANCELLED = "cancelled"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    ACCEPTED = "accepted"
    REVISION_REQUESTED = "revision_requested"
    REJECTED = "rejected"


@dataclass
class TaskOffer:
    """A task posted by an agent for other agents to bid on."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester_agent_id: str = ""
    requester_node_id: str = ""
    title: str = ""
    description: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    ftns_budget: float = 0.0
    deadline_seconds: float = 3600.0  # Time allowed for completion
    status: TaskStatus = TaskStatus.OPEN
    assigned_agent_id: Optional[str] = None
    bids: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    escrow_tx_id: Optional[str] = None  # Transaction ID for escrowed budget


@dataclass
class ReviewRequest:
    """A work product submitted for peer review."""
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitter_agent_id: str = ""
    submitter_node_id: str = ""
    content_cid: str = ""             # CID of the work product
    description: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    ftns_per_review: float = 0.1
    max_reviewers: int = 3
    status: ReviewStatus = ReviewStatus.PENDING
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    escrow_tx_id: Optional[str] = None  # Transaction ID for escrowed review budget
    paid_reviewers: List[str] = field(default_factory=list)  # Reviewers already paid


@dataclass
class KnowledgeQuery:
    """A query posted by an agent seeking information."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester_agent_id: str = ""
    requester_node_id: str = ""
    topic: str = ""
    question: str = ""
    ftns_per_response: float = 0.05
    max_responses: int = 5
    responses: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    escrow_tx_id: Optional[str] = None  # Transaction ID for escrowed query budget
    paid_responders: List[str] = field(default_factory=list)  # Responders already paid


class AgentCollaboration:
    """Manages collaboration protocol state machines.

    Coordinates task delegation, peer review, and knowledge exchange
    between agents across the P2P network. Integrates with the FTNS
    ledger for escrow-based payments and broadcasts all state changes
    via gossip for network-wide visibility.
    """

    def __init__(
        self,
        gossip: GossipProtocol,
        node_id: str,
        ledger=None,
        ledger_sync=None,
    ):
        self.gossip = gossip
        self.node_id = node_id
        self.ledger = ledger          # LocalLedger for balance operations
        self.ledger_sync = ledger_sync  # LedgerSync for cross-node payments

        self.tasks: Dict[str, TaskOffer] = {}
        self.reviews: Dict[str, ReviewRequest] = {}
        self.queries: Dict[str, KnowledgeQuery] = {}

        # LRU-bounded archives for completed records
        self._completed_tasks: collections.OrderedDict = collections.OrderedDict()
        self._completed_reviews: collections.OrderedDict = collections.OrderedDict()
        self._completed_queries: collections.OrderedDict = collections.OrderedDict()

        # Callbacks for local agents to handle incoming protocol events
        self._task_offer_handlers: List[Callable] = []
        self._review_request_handlers: List[Callable] = []
        self._knowledge_query_handlers: List[Callable] = []

        self._running = False
        self._tasks: List[asyncio.Task] = []

    def start(self) -> None:
        """Subscribe to collaboration gossip subtypes and start cleanup loop."""
        # Original subtypes
        self.gossip.subscribe(GOSSIP_TASK_OFFER, self._on_task_offer)
        self.gossip.subscribe(GOSSIP_TASK_BID, self._on_task_bid)
        self.gossip.subscribe(GOSSIP_REVIEW_REQUEST, self._on_review_request)
        self.gossip.subscribe(GOSSIP_KNOWLEDGE_QUERY, self._on_knowledge_query)

        # Sprint 4: state change subtypes
        self.gossip.subscribe(GOSSIP_TASK_ASSIGN, self._on_task_assign)
        self.gossip.subscribe(GOSSIP_TASK_COMPLETE, self._on_task_complete)
        self.gossip.subscribe(GOSSIP_TASK_CANCEL, self._on_task_cancel)
        self.gossip.subscribe(GOSSIP_REVIEW_SUBMIT, self._on_review_submit)
        self.gossip.subscribe(GOSSIP_KNOWLEDGE_RESPONSE, self._on_knowledge_response)

        # Start expiry cleanup loop
        self._running = True
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))

        logger.info("Agent collaboration protocols started (with FTNS payments)")

    async def stop(self) -> None:
        """Stop background tasks and refund any active escrows."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        # Refund escrows for tasks we own that are still open/assigned
        for task in list(self.tasks.values()):
            if (task.requester_node_id == self.node_id
                    and task.status in (TaskStatus.OPEN, TaskStatus.ASSIGNED)):
                await self._refund_task_escrow(task)
                task.status = TaskStatus.CANCELLED

    # ── Task Delegation Protocol ─────────────────────────────────

    async def post_task(
        self,
        requester_agent_id: str,
        title: str,
        description: str,
        ftns_budget: float,
        required_capabilities: Optional[List[str]] = None,
        deadline_seconds: float = 3600.0,
    ) -> TaskOffer:
        """Post a task offer to the network for agents to bid on.

        Escrows the FTNS budget from the requester's node wallet.
        The budget is released to the assigned agent on completion,
        or refunded to the requester on cancellation/expiry.
        """
        # Escrow the budget
        escrow_tx_id = None
        if self.ledger and ftns_budget > 0:
            try:
                from prsm.node.local_ledger import TransactionType
                tx = await self.ledger.debit(
                    wallet_id=self.node_id,
                    amount=ftns_budget,
                    tx_type=TransactionType.TRANSFER,
                    description=f"Escrow for task: {title[:50]}",
                )
                escrow_tx_id = tx.tx_id
            except ValueError as e:
                raise ValueError(f"Cannot post task: {e}") from e

        task = TaskOffer(
            requester_agent_id=requester_agent_id,
            requester_node_id=self.node_id,
            title=title,
            description=description,
            required_capabilities=required_capabilities or [],
            ftns_budget=ftns_budget,
            deadline_seconds=deadline_seconds,
            escrow_tx_id=escrow_tx_id,
        )
        self.tasks[task.task_id] = task

        await self.gossip.publish(GOSSIP_TASK_OFFER, {
            "task_id": task.task_id,
            "requester_agent_id": requester_agent_id,
            "requester_node_id": self.node_id,
            "title": title,
            "description": description,
            "required_capabilities": task.required_capabilities,
            "ftns_budget": ftns_budget,
            "deadline_seconds": deadline_seconds,
            "created_at": task.created_at,
        })

        logger.info(f"Posted task: {title} ({ftns_budget} FTNS, escrowed)")
        return task

    async def submit_bid(
        self,
        task_id: str,
        bidder_agent_id: str,
        estimated_cost: float,
        estimated_seconds: float,
        message: str = "",
    ) -> bool:
        """Submit a bid on a task offer."""
        bid = {
            "task_id": task_id,
            "bidder_agent_id": bidder_agent_id,
            "bidder_node_id": self.node_id,
            "estimated_cost": estimated_cost,
            "estimated_seconds": estimated_seconds,
            "message": message,
            "timestamp": time.time(),
        }

        # Store locally if we have the task
        task = self.tasks.get(task_id)
        if task:
            task.bids.append(bid)

        await self.gossip.publish(GOSSIP_TASK_BID, bid)
        return True

    async def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to the winning bidder and broadcast the assignment."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.OPEN:
            return False
        task.assigned_agent_id = agent_id
        task.status = TaskStatus.ASSIGNED

        # Broadcast assignment so all nodes (especially the assigned agent) know
        await self.gossip.publish(GOSSIP_TASK_ASSIGN, {
            "task_id": task_id,
            "assigned_agent_id": agent_id,
            "requester_node_id": task.requester_node_id,
        })

        logger.info(f"Task {task_id[:8]} assigned to agent {agent_id[:12]}")
        return True

    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed, pay the assigned agent, and broadcast."""
        task = self.tasks.get(task_id)
        if not task or task.status not in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            return False

        task.result = result
        task.status = TaskStatus.COMPLETED

        # Pay the assigned agent from escrow
        if (self.ledger and task.ftns_budget > 0
                and task.assigned_agent_id
                and task.requester_node_id == self.node_id):
            await self._pay_task_completion(task)

        # Broadcast completion
        await self.gossip.publish(GOSSIP_TASK_COMPLETE, {
            "task_id": task_id,
            "assigned_agent_id": task.assigned_agent_id,
            "requester_node_id": task.requester_node_id,
            "result_summary": str(result)[:200],
        })

        # Archive the completed task
        self._archive_task(task)

        logger.info(
            f"Task {task_id[:8]} completed by {task.assigned_agent_id[:12] if task.assigned_agent_id else '?'}, "
            f"paid {task.ftns_budget} FTNS"
        )
        return True

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task and refund the escrowed budget."""
        task = self.tasks.get(task_id)
        if not task or task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            return False

        task.status = TaskStatus.CANCELLED

        # Refund escrow if we own this task
        if task.requester_node_id == self.node_id:
            await self._refund_task_escrow(task)

        # Broadcast cancellation
        await self.gossip.publish(GOSSIP_TASK_CANCEL, {
            "task_id": task_id,
            "requester_node_id": task.requester_node_id,
        })

        self._archive_task(task)
        logger.info(f"Task {task_id[:8]} cancelled, escrow refunded")
        return True

    # ── Peer Review Protocol ─────────────────────────────────────

    async def request_review(
        self,
        submitter_agent_id: str,
        content_cid: str,
        description: str,
        ftns_per_review: float = 0.1,
        required_capabilities: Optional[List[str]] = None,
        max_reviewers: int = 3,
    ) -> ReviewRequest:
        """Submit a work product for peer review.

        Escrows (ftns_per_review * max_reviewers) FTNS to pay reviewers.
        """
        total_escrow = ftns_per_review * max_reviewers
        escrow_tx_id = None
        if self.ledger and total_escrow > 0:
            try:
                from prsm.node.local_ledger import TransactionType
                tx = await self.ledger.debit(
                    wallet_id=self.node_id,
                    amount=total_escrow,
                    tx_type=TransactionType.TRANSFER,
                    description=f"Escrow for review: {content_cid[:12]}",
                )
                escrow_tx_id = tx.tx_id
            except ValueError as e:
                raise ValueError(f"Cannot request review: {e}") from e

        review = ReviewRequest(
            submitter_agent_id=submitter_agent_id,
            submitter_node_id=self.node_id,
            content_cid=content_cid,
            description=description,
            required_capabilities=required_capabilities or [],
            ftns_per_review=ftns_per_review,
            max_reviewers=max_reviewers,
            escrow_tx_id=escrow_tx_id,
        )
        self.reviews[review.review_id] = review

        await self.gossip.publish(GOSSIP_REVIEW_REQUEST, {
            "review_id": review.review_id,
            "submitter_agent_id": submitter_agent_id,
            "submitter_node_id": self.node_id,
            "content_cid": content_cid,
            "description": description,
            "required_capabilities": review.required_capabilities,
            "ftns_per_review": ftns_per_review,
            "max_reviewers": max_reviewers,
            "created_at": review.created_at,
        })

        logger.info(f"Review requested for CID {content_cid[:12]}... ({total_escrow} FTNS escrowed)")
        return review

    async def submit_review(
        self,
        review_id: str,
        reviewer_agent_id: str,
        reviewer_node_id: str,
        verdict: str,
        comments: str = "",
    ) -> bool:
        """Submit a review for a work product and pay the reviewer."""
        review = self.reviews.get(review_id)
        if not review:
            return False
        if len(review.reviews) >= review.max_reviewers:
            return False

        review.reviews.append({
            "reviewer_agent_id": reviewer_agent_id,
            "reviewer_node_id": reviewer_node_id,
            "verdict": verdict,  # "accept", "revise", "reject"
            "comments": comments,
            "timestamp": time.time(),
        })

        # Pay the reviewer from escrow if we own this review
        if (self.ledger and review.submitter_node_id == self.node_id
                and review.ftns_per_review > 0
                and reviewer_node_id != self.node_id
                and reviewer_agent_id not in review.paid_reviewers):
            await self._pay_reviewer(review, reviewer_node_id, reviewer_agent_id)

        # Check for consensus
        if len(review.reviews) >= review.max_reviewers:
            verdicts = [r["verdict"] for r in review.reviews]
            accept_count = verdicts.count("accept")
            if accept_count > len(verdicts) / 2:
                review.status = ReviewStatus.ACCEPTED
            elif verdicts.count("reject") > len(verdicts) / 2:
                review.status = ReviewStatus.REJECTED
            else:
                review.status = ReviewStatus.REVISION_REQUESTED

            # Refund unused escrow (if fewer reviewers than max)
            if review.submitter_node_id == self.node_id:
                unused_slots = review.max_reviewers - len(review.paid_reviewers)
                if unused_slots > 0 and self.ledger:
                    await self._refund_review_escrow(review, unused_slots)

            self._archive_review(review)

        # Broadcast the review submission
        await self.gossip.publish(GOSSIP_REVIEW_SUBMIT, {
            "review_id": review_id,
            "reviewer_agent_id": reviewer_agent_id,
            "reviewer_node_id": reviewer_node_id,
            "verdict": verdict,
            "comments": comments[:200],
            "review_status": review.status.value,
        })

        return True

    # ── Knowledge Exchange Protocol ──────────────────────────────

    async def post_query(
        self,
        requester_agent_id: str,
        topic: str,
        question: str,
        ftns_per_response: float = 0.05,
        max_responses: int = 5,
    ) -> KnowledgeQuery:
        """Post a knowledge query to the network.

        Escrows (ftns_per_response * max_responses) FTNS to pay responders.
        """
        total_escrow = ftns_per_response * max_responses
        escrow_tx_id = None
        if self.ledger and total_escrow > 0:
            try:
                from prsm.node.local_ledger import TransactionType
                tx = await self.ledger.debit(
                    wallet_id=self.node_id,
                    amount=total_escrow,
                    tx_type=TransactionType.TRANSFER,
                    description=f"Escrow for query: {topic[:50]}",
                )
                escrow_tx_id = tx.tx_id
            except ValueError as e:
                raise ValueError(f"Cannot post query: {e}") from e

        query = KnowledgeQuery(
            requester_agent_id=requester_agent_id,
            requester_node_id=self.node_id,
            topic=topic,
            question=question,
            ftns_per_response=ftns_per_response,
            max_responses=max_responses,
            escrow_tx_id=escrow_tx_id,
        )
        self.queries[query.query_id] = query

        await self.gossip.publish(GOSSIP_KNOWLEDGE_QUERY, {
            "query_id": query.query_id,
            "requester_agent_id": requester_agent_id,
            "requester_node_id": self.node_id,
            "topic": topic,
            "question": question,
            "ftns_per_response": ftns_per_response,
            "max_responses": max_responses,
            "created_at": query.created_at,
        })

        logger.info(f"Knowledge query posted: {topic} ({total_escrow} FTNS escrowed)")
        return query

    async def submit_response(
        self,
        query_id: str,
        responder_agent_id: str,
        responder_node_id: str,
        answer: str,
        content_cids: Optional[List[str]] = None,
    ) -> bool:
        """Submit a response to a knowledge query and pay the responder."""
        query = self.queries.get(query_id)
        if not query:
            return False
        if len(query.responses) >= query.max_responses:
            return False

        query.responses.append({
            "responder_agent_id": responder_agent_id,
            "responder_node_id": responder_node_id,
            "answer": answer,
            "content_cids": content_cids or [],
            "timestamp": time.time(),
        })

        # Pay the responder from escrow if we own this query
        if (self.ledger and query.requester_node_id == self.node_id
                and query.ftns_per_response > 0
                and responder_node_id != self.node_id
                and responder_agent_id not in query.paid_responders):
            await self._pay_responder(query, responder_node_id, responder_agent_id)

        # If we've reached max responses, refund unused escrow and archive
        if len(query.responses) >= query.max_responses:
            if query.requester_node_id == self.node_id:
                unused_slots = query.max_responses - len(query.paid_responders)
                if unused_slots > 0 and self.ledger:
                    await self._refund_query_escrow(query, unused_slots)
            self._archive_query(query)

        # Broadcast the response
        await self.gossip.publish(GOSSIP_KNOWLEDGE_RESPONSE, {
            "query_id": query_id,
            "responder_agent_id": responder_agent_id,
            "responder_node_id": responder_node_id,
            "answer_preview": answer[:200],
            "content_cids": content_cids or [],
        })

        return True

    # ── FTNS Payment Helpers ─────────────────────────────────────

    async def _pay_task_completion(self, task: TaskOffer) -> None:
        """Release escrowed FTNS to the assigned agent's node."""
        if not self.ledger or not task.assigned_agent_id:
            return
        try:
            # Find the assigned agent's node from the bid
            target_node = None
            for bid in task.bids:
                if bid.get("bidder_agent_id") == task.assigned_agent_id:
                    target_node = bid.get("bidder_node_id")
                    break

            if not target_node or target_node == self.node_id:
                # Local agent — credit directly from escrow
                from prsm.node.local_ledger import TransactionType
                await self.ledger.credit(
                    wallet_id=self.node_id,
                    amount=task.ftns_budget,
                    tx_type=TransactionType.COMPUTE_EARNING,
                    description=f"Task payment: {task.title[:30]}",
                )
            elif self.ledger_sync:
                # Remote agent — use ledger_sync for signed cross-node transfer
                await self.ledger_sync.signed_transfer(
                    to_wallet=target_node,
                    amount=task.ftns_budget,
                    description=f"Task payment: {task.title[:30]}",
                )
            else:
                # No ledger_sync — credit locally (will reconcile later)
                from prsm.node.local_ledger import TransactionType
                await self.ledger.credit(
                    wallet_id=self.node_id,
                    amount=task.ftns_budget,
                    tx_type=TransactionType.COMPUTE_EARNING,
                    description=f"Task payment (pending sync): {task.title[:30]}",
                )
        except Exception as e:
            logger.error(f"Task payment failed for {task.task_id[:8]}: {e}")

    async def _refund_task_escrow(self, task: TaskOffer) -> None:
        """Refund escrowed FTNS back to the requester's wallet."""
        if not self.ledger or task.ftns_budget <= 0:
            return
        try:
            from prsm.node.local_ledger import TransactionType
            await self.ledger.credit(
                wallet_id=self.node_id,
                amount=task.ftns_budget,
                tx_type=TransactionType.TRANSFER,
                description=f"Escrow refund: {task.title[:30]}",
            )
            logger.info(f"Refunded {task.ftns_budget} FTNS for task {task.task_id[:8]}")
        except Exception as e:
            logger.error(f"Escrow refund failed for task {task.task_id[:8]}: {e}")

    async def _pay_reviewer(self, review: ReviewRequest, reviewer_node_id: str, reviewer_agent_id: str) -> None:
        """Pay a reviewer from the escrowed budget."""
        try:
            if reviewer_node_id == self.node_id:
                from prsm.node.local_ledger import TransactionType
                await self.ledger.credit(
                    wallet_id=self.node_id,
                    amount=review.ftns_per_review,
                    tx_type=TransactionType.COMPUTE_EARNING,
                    description=f"Review payment: {review.review_id[:8]}",
                )
            elif self.ledger_sync:
                await self.ledger_sync.signed_transfer(
                    to_wallet=reviewer_node_id,
                    amount=review.ftns_per_review,
                    description=f"Review payment: {review.review_id[:8]}",
                )
            review.paid_reviewers.append(reviewer_agent_id)
        except Exception as e:
            logger.error(f"Reviewer payment failed for {review.review_id[:8]}: {e}")

    async def _refund_review_escrow(self, review: ReviewRequest, unused_slots: int) -> None:
        """Refund unused review escrow slots."""
        refund = review.ftns_per_review * unused_slots
        if refund <= 0:
            return
        try:
            from prsm.node.local_ledger import TransactionType
            await self.ledger.credit(
                wallet_id=self.node_id,
                amount=refund,
                tx_type=TransactionType.TRANSFER,
                description=f"Review escrow refund ({unused_slots} unused): {review.review_id[:8]}",
            )
        except Exception as e:
            logger.error(f"Review escrow refund failed: {e}")

    async def _pay_responder(self, query: KnowledgeQuery, responder_node_id: str, responder_agent_id: str) -> None:
        """Pay a knowledge query responder from the escrowed budget."""
        try:
            if responder_node_id == self.node_id:
                from prsm.node.local_ledger import TransactionType
                await self.ledger.credit(
                    wallet_id=self.node_id,
                    amount=query.ftns_per_response,
                    tx_type=TransactionType.COMPUTE_EARNING,
                    description=f"Query response payment: {query.query_id[:8]}",
                )
            elif self.ledger_sync:
                await self.ledger_sync.signed_transfer(
                    to_wallet=responder_node_id,
                    amount=query.ftns_per_response,
                    description=f"Query response payment: {query.query_id[:8]}",
                )
            query.paid_responders.append(responder_agent_id)
        except Exception as e:
            logger.error(f"Responder payment failed for {query.query_id[:8]}: {e}")

    async def _refund_query_escrow(self, query: KnowledgeQuery, unused_slots: int) -> None:
        """Refund unused query response escrow slots."""
        refund = query.ftns_per_response * unused_slots
        if refund <= 0:
            return
        try:
            from prsm.node.local_ledger import TransactionType
            await self.ledger.credit(
                wallet_id=self.node_id,
                amount=refund,
                tx_type=TransactionType.TRANSFER,
                description=f"Query escrow refund ({unused_slots} unused): {query.query_id[:8]}",
            )
        except Exception as e:
            logger.error(f"Query escrow refund failed: {e}")

    # ── Gossip handlers ──────────────────────────────────────────

    async def _on_task_offer(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Process incoming task offers."""
        if origin == self.node_id:
            return
        task_id = data.get("task_id", "")
        if task_id and task_id not in self.tasks:
            self.tasks[task_id] = TaskOffer(
                task_id=task_id,
                requester_agent_id=data.get("requester_agent_id", ""),
                requester_node_id=data.get("requester_node_id", origin),
                title=data.get("title", ""),
                description=data.get("description", ""),
                required_capabilities=data.get("required_capabilities", []),
                ftns_budget=data.get("ftns_budget", 0),
                deadline_seconds=data.get("deadline_seconds", 3600),
                created_at=data.get("created_at", time.time()),
            )

    async def _on_task_bid(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Process incoming task bids."""
        task_id = data.get("task_id", "")
        task = self.tasks.get(task_id)
        if task and task.requester_node_id == self.node_id:
            task.bids.append(data)

    async def _on_task_assign(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Process incoming task assignment notifications."""
        if origin == self.node_id:
            return
        task_id = data.get("task_id", "")
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.OPEN:
            task.assigned_agent_id = data.get("assigned_agent_id")
            task.status = TaskStatus.ASSIGNED
            logger.info(f"Task {task_id[:8]} assigned to {task.assigned_agent_id[:12] if task.assigned_agent_id else '?'}")

    async def _on_task_complete(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Process incoming task completion notifications."""
        if origin == self.node_id:
            return
        task_id = data.get("task_id", "")
        task = self.tasks.get(task_id)
        if task and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.OPEN):
            task.status = TaskStatus.COMPLETED
            self._archive_task(task)
            logger.info(f"Task {task_id[:8]} completed (remote notification)")

    async def _on_task_cancel(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Process incoming task cancellation notifications."""
        if origin == self.node_id:
            return
        task_id = data.get("task_id", "")
        task = self.tasks.get(task_id)
        if task and task.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            task.status = TaskStatus.CANCELLED
            self._archive_task(task)

    async def _on_review_request(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Process incoming review requests."""
        if origin == self.node_id:
            return
        review_id = data.get("review_id", "")
        if review_id and review_id not in self.reviews:
            self.reviews[review_id] = ReviewRequest(
                review_id=review_id,
                submitter_agent_id=data.get("submitter_agent_id", ""),
                submitter_node_id=data.get("submitter_node_id", origin),
                content_cid=data.get("content_cid", ""),
                description=data.get("description", ""),
                required_capabilities=data.get("required_capabilities", []),
                ftns_per_review=data.get("ftns_per_review", 0.1),
                max_reviewers=data.get("max_reviewers", 3),
                created_at=data.get("created_at", time.time()),
            )

    async def _on_review_submit(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Process incoming review submission notifications."""
        if origin == self.node_id:
            return
        review_id = data.get("review_id", "")
        review = self.reviews.get(review_id)
        if review:
            # Add the review record if we haven't seen it
            reviewer_agent_id = data.get("reviewer_agent_id", "")
            already_reviewed = any(
                r.get("reviewer_agent_id") == reviewer_agent_id
                for r in review.reviews
            )
            if not already_reviewed:
                review.reviews.append({
                    "reviewer_agent_id": reviewer_agent_id,
                    "reviewer_node_id": data.get("reviewer_node_id", origin),
                    "verdict": data.get("verdict", ""),
                    "comments": data.get("comments", ""),
                    "timestamp": time.time(),
                })

            # Update status from broadcast
            new_status = data.get("review_status")
            if new_status:
                try:
                    review.status = ReviewStatus(new_status)
                    if review.status in (ReviewStatus.ACCEPTED, ReviewStatus.REJECTED, ReviewStatus.REVISION_REQUESTED):
                        self._archive_review(review)
                except ValueError:
                    pass

    async def _on_knowledge_query(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Process incoming knowledge queries."""
        if origin == self.node_id:
            return
        query_id = data.get("query_id", "")
        if query_id and query_id not in self.queries:
            self.queries[query_id] = KnowledgeQuery(
                query_id=query_id,
                requester_agent_id=data.get("requester_agent_id", ""),
                requester_node_id=data.get("requester_node_id", origin),
                topic=data.get("topic", ""),
                question=data.get("question", ""),
                ftns_per_response=data.get("ftns_per_response", 0.05),
                max_responses=data.get("max_responses", 5),
                created_at=data.get("created_at", time.time()),
            )

    async def _on_knowledge_response(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Process incoming knowledge response notifications."""
        if origin == self.node_id:
            return
        query_id = data.get("query_id", "")
        query = self.queries.get(query_id)
        if query:
            responder_agent_id = data.get("responder_agent_id", "")
            already_responded = any(
                r.get("responder_agent_id") == responder_agent_id
                for r in query.responses
            )
            if not already_responded and len(query.responses) < query.max_responses:
                query.responses.append({
                    "responder_agent_id": responder_agent_id,
                    "responder_node_id": data.get("responder_node_id", origin),
                    "answer": data.get("answer_preview", ""),
                    "content_cids": data.get("content_cids", []),
                    "timestamp": time.time(),
                })

    # ── Expiry Enforcement & Cleanup ─────────────────────────────

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired collaborations and enforce memory bounds."""
        while self._running:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            try:
                await self._expire_stale_records()
                self._enforce_archive_bounds()
            except Exception as e:
                logger.error(f"Collaboration cleanup error: {e}")

    async def _expire_stale_records(self) -> None:
        """Cancel expired tasks, reviews, and queries. Refund escrows."""
        now = time.time()

        # Expire open/assigned tasks past their deadline
        for task in list(self.tasks.values()):
            if task.status in (TaskStatus.OPEN, TaskStatus.ASSIGNED):
                age = now - task.created_at
                if age > task.deadline_seconds:
                    task.status = TaskStatus.CANCELLED
                    if task.requester_node_id == self.node_id:
                        await self._refund_task_escrow(task)
                        await self.gossip.publish(GOSSIP_TASK_CANCEL, {
                            "task_id": task.task_id,
                            "requester_node_id": task.requester_node_id,
                            "reason": "expired",
                        })
                    self._archive_task(task)
                    logger.info(f"Task {task.task_id[:8]} expired after {age:.0f}s")

        # Expire pending reviews past timeout
        for review in list(self.reviews.values()):
            if review.status == ReviewStatus.PENDING:
                age = now - review.created_at
                if age > DEFAULT_REVIEW_TIMEOUT:
                    review.status = ReviewStatus.REJECTED
                    if review.submitter_node_id == self.node_id:
                        unused = review.max_reviewers - len(review.paid_reviewers)
                        if unused > 0:
                            await self._refund_review_escrow(review, unused)
                    self._archive_review(review)
                    logger.info(f"Review {review.review_id[:8]} expired")

        # Expire open queries past timeout
        for query in list(self.queries.values()):
            age = now - query.created_at
            if age > DEFAULT_QUERY_TIMEOUT and len(query.responses) < query.max_responses:
                if query.requester_node_id == self.node_id:
                    unused = query.max_responses - len(query.paid_responders)
                    if unused > 0:
                        await self._refund_query_escrow(query, unused)
                self._archive_query(query)
                logger.info(f"Query {query.query_id[:8]} expired")

    # ── Archive & Memory Bounds ──────────────────────────────────

    def _archive_task(self, task: TaskOffer) -> None:
        """Move a completed/cancelled task from active to archive."""
        self.tasks.pop(task.task_id, None)
        self._completed_tasks[task.task_id] = task
        self._completed_tasks.move_to_end(task.task_id)

    def _archive_review(self, review: ReviewRequest) -> None:
        """Move a finalized review from active to archive."""
        self.reviews.pop(review.review_id, None)
        self._completed_reviews[review.review_id] = review
        self._completed_reviews.move_to_end(review.review_id)

    def _archive_query(self, query: KnowledgeQuery) -> None:
        """Move a completed/expired query from active to archive."""
        self.queries.pop(query.query_id, None)
        self._completed_queries[query.query_id] = query
        self._completed_queries.move_to_end(query.query_id)

    def _enforce_archive_bounds(self) -> None:
        """Evict oldest records from archives when they exceed bounds."""
        while len(self._completed_tasks) > MAX_COMPLETED_RECORDS:
            self._completed_tasks.popitem(last=False)
        while len(self._completed_reviews) > MAX_COMPLETED_RECORDS:
            self._completed_reviews.popitem(last=False)
        while len(self._completed_queries) > MAX_COMPLETED_RECORDS:
            self._completed_queries.popitem(last=False)

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Collaboration protocol statistics."""
        open_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.OPEN)
        assigned_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.ASSIGNED)
        pending_reviews = sum(1 for r in self.reviews.values() if r.status == ReviewStatus.PENDING)
        return {
            "active_tasks": len(self.tasks),
            "open_tasks": open_tasks,
            "assigned_tasks": assigned_tasks,
            "archived_tasks": len(self._completed_tasks),
            "active_reviews": len(self.reviews),
            "pending_reviews": pending_reviews,
            "archived_reviews": len(self._completed_reviews),
            "active_queries": len(self.queries),
            "archived_queries": len(self._completed_queries),
        }

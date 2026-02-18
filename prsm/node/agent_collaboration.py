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
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from prsm.node.gossip import GossipProtocol

logger = logging.getLogger(__name__)

# Gossip subtypes for collaboration protocols
GOSSIP_TASK_OFFER = "agent_task_offer"
GOSSIP_TASK_BID = "agent_task_bid"
GOSSIP_REVIEW_REQUEST = "agent_review_request"
GOSSIP_KNOWLEDGE_QUERY = "agent_knowledge_query"


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


class AgentCollaboration:
    """Manages collaboration protocol state machines.

    Coordinates task delegation, peer review, and knowledge exchange
    between agents across the P2P network.
    """

    def __init__(self, gossip: GossipProtocol, node_id: str):
        self.gossip = gossip
        self.node_id = node_id

        self.tasks: Dict[str, TaskOffer] = {}
        self.reviews: Dict[str, ReviewRequest] = {}
        self.queries: Dict[str, KnowledgeQuery] = {}

        # Callbacks for local agents to handle incoming protocol events
        self._task_offer_handlers: List[Callable] = []
        self._review_request_handlers: List[Callable] = []
        self._knowledge_query_handlers: List[Callable] = []

    def start(self) -> None:
        """Subscribe to collaboration gossip subtypes."""
        self.gossip.subscribe(GOSSIP_TASK_OFFER, self._on_task_offer)
        self.gossip.subscribe(GOSSIP_TASK_BID, self._on_task_bid)
        self.gossip.subscribe(GOSSIP_REVIEW_REQUEST, self._on_review_request)
        self.gossip.subscribe(GOSSIP_KNOWLEDGE_QUERY, self._on_knowledge_query)
        logger.info("Agent collaboration protocols started")

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
        """Post a task offer to the network for agents to bid on."""
        task = TaskOffer(
            requester_agent_id=requester_agent_id,
            requester_node_id=self.node_id,
            title=title,
            description=description,
            required_capabilities=required_capabilities or [],
            ftns_budget=ftns_budget,
            deadline_seconds=deadline_seconds,
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

        logger.info(f"Posted task: {title} ({ftns_budget} FTNS)")
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

    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to the winning bidder (called by requester)."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.OPEN:
            return False
        task.assigned_agent_id = agent_id
        task.status = TaskStatus.ASSIGNED
        return True

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed with a result."""
        task = self.tasks.get(task_id)
        if not task or task.status not in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            return False
        task.result = result
        task.status = TaskStatus.COMPLETED
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
        """Submit a work product for peer review."""
        review = ReviewRequest(
            submitter_agent_id=submitter_agent_id,
            submitter_node_id=self.node_id,
            content_cid=content_cid,
            description=description,
            required_capabilities=required_capabilities or [],
            ftns_per_review=ftns_per_review,
            max_reviewers=max_reviewers,
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

        logger.info(f"Review requested for CID {content_cid[:12]}...")
        return review

    def submit_review(
        self,
        review_id: str,
        reviewer_agent_id: str,
        verdict: str,
        comments: str = "",
    ) -> bool:
        """Submit a review for a work product."""
        review = self.reviews.get(review_id)
        if not review:
            return False
        if len(review.reviews) >= review.max_reviewers:
            return False

        review.reviews.append({
            "reviewer_agent_id": reviewer_agent_id,
            "reviewer_node_id": self.node_id,
            "verdict": verdict,  # "accept", "revise", "reject"
            "comments": comments,
            "timestamp": time.time(),
        })

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
        """Post a knowledge query to the network."""
        query = KnowledgeQuery(
            requester_agent_id=requester_agent_id,
            requester_node_id=self.node_id,
            topic=topic,
            question=question,
            ftns_per_response=ftns_per_response,
            max_responses=max_responses,
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

        logger.info(f"Knowledge query posted: {topic}")
        return query

    def submit_response(
        self,
        query_id: str,
        responder_agent_id: str,
        answer: str,
        content_cids: Optional[List[str]] = None,
    ) -> bool:
        """Submit a response to a knowledge query."""
        query = self.queries.get(query_id)
        if not query:
            return False
        if len(query.responses) >= query.max_responses:
            return False

        query.responses.append({
            "responder_agent_id": responder_agent_id,
            "responder_node_id": self.node_id,
            "answer": answer,
            "content_cids": content_cids or [],
            "timestamp": time.time(),
        })
        return True

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

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Collaboration protocol statistics."""
        open_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.OPEN)
        pending_reviews = sum(1 for r in self.reviews.values() if r.status == ReviewStatus.PENDING)
        return {
            "total_tasks": len(self.tasks),
            "open_tasks": open_tasks,
            "total_reviews": len(self.reviews),
            "pending_reviews": pending_reviews,
            "total_queries": len(self.queries),
        }

"""
PRSM Collaboration Module
==========================

Provides collaboration infrastructure for multi-agent interaction over the PRSM P2P network.
This module interfaces with the node-level agent collaboration protocols.

Components:
- Collaboration protocols for task delegation, peer review, and knowledge exchange
- Agent-to-agent messaging and coordination
- Collaboration session management
- Real-time multi-user session support

Sprint 4 Phase 5: CollaborationManager now bridges to the P2P AgentCollaboration
layer. Sessions of type TASK_DELEGATION, PEER_REVIEW, and KNOWLEDGE_EXCHANGE
can be dispatched to the network, and P2P protocol completions automatically
update the corresponding session state.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# Import session management
from prsm.collaboration.session_manager import (
    SessionManager,
    CollaborationSession as RealtimeCollaborationSession,
    SessionState,
    Participant,
    ParticipantRole,
    get_session_manager,
)


class CollaborationType(str, Enum):
    """Types of collaboration between agents"""
    TASK_DELEGATION = "task_delegation"
    PEER_REVIEW = "peer_review"
    KNOWLEDGE_EXCHANGE = "knowledge_exchange"
    JOINT_REASONING = "joint_reasoning"
    CONSENSUS_BUILDING = "consensus_building"


class CollaborationStatus(str, Enum):
    """Status of a collaboration session"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CollaborationSession:
    """Represents a collaboration session between agents"""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    collaboration_type: CollaborationType = CollaborationType.JOINT_REASONING
    initiator_agent_id: str = ""
    participant_agent_ids: List[str] = field(default_factory=list)
    status: CollaborationStatus = CollaborationStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    
    def start(self) -> None:
        self.status = CollaborationStatus.ACTIVE
        self.started_at = datetime.now(timezone.utc)
    
    def complete(self, results: Optional[Dict[str, Any]] = None) -> None:
        self.status = CollaborationStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if results:
            self.results = results
    
    def fail(self, error: str) -> None:
        self.status = CollaborationStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.metadata["error"] = error
    
    def cancel(self) -> None:
        self.status = CollaborationStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)


@dataclass
class CollaborationProposal:
    """Proposal for a collaboration session"""
    proposal_id: str = field(default_factory=lambda: str(uuid4()))
    collaboration_type: CollaborationType = CollaborationType.JOINT_REASONING
    proposer_agent_id: str = ""
    target_agent_ids: List[str] = field(default_factory=list)
    description: str = ""
    estimated_duration_seconds: int = 60
    required_capabilities: List[str] = field(default_factory=list)
    ftns_budget: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    accepted_by: List[str] = field(default_factory=list)
    rejected_by: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def accept(self, agent_id: str) -> None:
        if agent_id not in self.accepted_by:
            self.accepted_by.append(agent_id)
    
    def reject(self, agent_id: str) -> None:
        if agent_id not in self.rejected_by:
            self.rejected_by.append(agent_id)
    
    def is_unanimous(self) -> bool:
        return len(self.accepted_by) == len(self.target_agent_ids)
    
    def is_rejected(self) -> bool:
        return len(self.rejected_by) > 0


@dataclass
class CollaborationResult:
    """Result from a collaboration session"""
    session_id: str
    collaboration_type: CollaborationType
    success: bool
    participants: List[str]
    duration_seconds: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    consensus_score: Optional[float] = None
    ftns_spent: float = 0.0
    quality_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollaborationManager:
    """
    Manager for collaboration sessions and proposals.

    Coordinates collaboration between agents on the PRSM network.
    When an AgentCollaboration instance is wired in via set_agent_collaboration(),
    sessions can be dispatched to the P2P network and protocol completions
    automatically update session state.
    """

    def __init__(self):
        self._sessions: Dict[str, CollaborationSession] = {}
        self._proposals: Dict[str, CollaborationProposal] = {}
        self._results: Dict[str, CollaborationResult] = {}

        # P2P bridge: AgentCollaboration reference
        self._agent_collab = None  # Set via set_agent_collaboration()

        # Bidirectional mapping: session_id ↔ protocol_id
        self._session_to_protocol: Dict[str, str] = {}  # session_id → protocol_id
        self._protocol_to_session: Dict[str, str] = {}  # protocol_id → session_id

    def set_agent_collaboration(self, agent_collab) -> None:
        """Wire in the P2P agent collaboration layer for network dispatch."""
        self._agent_collab = agent_collab

    # ── Proposals ────────────────────────────────────────────────

    def create_proposal(
        self,
        collaboration_type: CollaborationType,
        proposer_agent_id: str,
        target_agent_ids: List[str],
        description: str,
        **kwargs
    ) -> CollaborationProposal:
        proposal = CollaborationProposal(
            collaboration_type=collaboration_type,
            proposer_agent_id=proposer_agent_id,
            target_agent_ids=target_agent_ids,
            description=description,
            **kwargs
        )
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def get_proposal(self, proposal_id: str) -> Optional[CollaborationProposal]:
        return self._proposals.get(proposal_id)

    # ── Sessions ─────────────────────────────────────────────────

    def create_session(
        self,
        collaboration_type: CollaborationType,
        initiator_agent_id: str,
        participant_agent_ids: List[str],
        **kwargs
    ) -> CollaborationSession:
        session = CollaborationSession(
            collaboration_type=collaboration_type,
            initiator_agent_id=initiator_agent_id,
            participant_agent_ids=participant_agent_ids,
            **kwargs
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        return self._sessions.get(session_id)

    def complete_session(
        self,
        session_id: str,
        success: bool,
        outputs: Optional[Dict[str, Any]] = None,
        consensus_score: Optional[float] = None,
        ftns_spent: float = 0.0
    ) -> Optional[CollaborationResult]:
        session = self._sessions.get(session_id)
        if not session:
            return None

        if success:
            session.complete(outputs)
        else:
            session.fail(outputs.get("error", "Unknown error") if outputs else "Unknown error")

        duration = 0.0
        if session.started_at and session.completed_at:
            duration = (session.completed_at - session.started_at).total_seconds()

        result = CollaborationResult(
            session_id=session_id,
            collaboration_type=session.collaboration_type,
            success=success,
            participants=[session.initiator_agent_id] + session.participant_agent_ids,
            duration_seconds=duration,
            outputs=outputs or {},
            consensus_score=consensus_score,
            ftns_spent=ftns_spent,
        )

        self._results[session_id] = result
        return result

    def get_result(self, session_id: str) -> Optional[CollaborationResult]:
        return self._results.get(session_id)

    def get_active_sessions(self) -> List[CollaborationSession]:
        return [s for s in self._sessions.values() if s.status == CollaborationStatus.ACTIVE]

    def get_pending_proposals(self) -> List[CollaborationProposal]:
        now = datetime.now(timezone.utc)
        return [
            p for p in self._proposals.values()
            if p.accepted_by or (p.expires_at is None or p.expires_at > now)
        ]

    # ── P2P Network Dispatch ─────────────────────────────────────

    async def dispatch_session(self, session_id: str) -> Optional[str]:
        """Dispatch a session to the P2P network based on its collaboration type.

        Creates the corresponding protocol object (task, review, or query)
        on the AgentCollaboration layer, starts the session, and links them
        so that protocol completions update the session automatically.

        Args:
            session_id: ID of the session to dispatch

        Returns:
            The protocol ID (task_id, review_id, or query_id), or None on failure
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Cannot dispatch: session {session_id} not found")
            return None

        if not self._agent_collab:
            logger.warning("Cannot dispatch: no AgentCollaboration wired in")
            return None

        if session.status != CollaborationStatus.PENDING:
            logger.warning(f"Cannot dispatch: session {session_id} is {session.status.value}")
            return None

        collab_type = session.collaboration_type
        description = session.metadata.get("description", "")
        ftns_budget = session.metadata.get("ftns_budget", 0.0)
        capabilities = session.metadata.get("required_capabilities", [])

        try:
            if collab_type == CollaborationType.TASK_DELEGATION:
                protocol_id = await self._dispatch_task(session, description, ftns_budget, capabilities)
            elif collab_type == CollaborationType.PEER_REVIEW:
                protocol_id = await self._dispatch_review(session, description, capabilities)
            elif collab_type == CollaborationType.KNOWLEDGE_EXCHANGE:
                protocol_id = await self._dispatch_query(session, description)
            else:
                logger.info(
                    f"Session {session_id} type {collab_type.value} does not map to a P2P protocol; "
                    f"starting locally"
                )
                session.start()
                return None

        except ValueError as e:
            session.fail(str(e))
            logger.error(f"Dispatch failed for session {session_id}: {e}")
            return None

        # Link session ↔ protocol
        self._session_to_protocol[session_id] = protocol_id
        self._protocol_to_session[protocol_id] = session_id
        session.start()

        logger.info(
            f"Session {session_id[:8]} dispatched as {collab_type.value} → protocol {protocol_id[:8]}"
        )
        return protocol_id

    async def _dispatch_task(
        self,
        session: CollaborationSession,
        description: str,
        ftns_budget: float,
        capabilities: List[str],
    ) -> str:
        """Dispatch a TASK_DELEGATION session to AgentCollaboration.post_task()."""
        title = session.metadata.get("title", description[:50] or "Untitled task")
        deadline = session.metadata.get("deadline_seconds", 3600.0)

        task = await self._agent_collab.post_task(
            requester_agent_id=session.initiator_agent_id,
            title=title,
            description=description,
            ftns_budget=ftns_budget,
            required_capabilities=capabilities,
            deadline_seconds=deadline,
        )
        return task.task_id

    async def _dispatch_review(
        self,
        session: CollaborationSession,
        description: str,
        capabilities: List[str],
    ) -> str:
        """Dispatch a PEER_REVIEW session to AgentCollaboration.request_review()."""
        content_cid = session.metadata.get("content_cid", "")
        ftns_per_review = session.metadata.get("ftns_per_review", 0.1)
        max_reviewers = session.metadata.get("max_reviewers", 3)

        review = await self._agent_collab.request_review(
            submitter_agent_id=session.initiator_agent_id,
            content_cid=content_cid,
            description=description,
            ftns_per_review=ftns_per_review,
            required_capabilities=capabilities,
            max_reviewers=max_reviewers,
        )
        return review.review_id

    async def _dispatch_query(
        self,
        session: CollaborationSession,
        description: str,
    ) -> str:
        """Dispatch a KNOWLEDGE_EXCHANGE session to AgentCollaboration.post_query()."""
        topic = session.metadata.get("topic", "general")
        ftns_per_response = session.metadata.get("ftns_per_response", 0.05)
        max_responses = session.metadata.get("max_responses", 5)

        query = await self._agent_collab.post_query(
            requester_agent_id=session.initiator_agent_id,
            topic=topic,
            question=description,
            ftns_per_response=ftns_per_response,
            max_responses=max_responses,
        )
        return query.query_id

    # ── P2P Protocol Completion Callback ──────────────────────────

    def on_protocol_complete(
        self,
        protocol_id: str,
        success: bool,
        outputs: Optional[Dict[str, Any]] = None,
        ftns_spent: float = 0.0,
    ) -> Optional[CollaborationResult]:
        """Called when a P2P protocol (task/review/query) completes.

        Looks up the linked session and completes it with the result.
        This method can be called from AgentCollaboration gossip handlers
        or from application code that monitors protocol state.

        Args:
            protocol_id: The task_id, review_id, or query_id that completed
            success: Whether the protocol completed successfully
            outputs: Result data from the protocol
            ftns_spent: Total FTNS spent during the protocol

        Returns:
            CollaborationResult if a linked session was found, None otherwise
        """
        session_id = self._protocol_to_session.get(protocol_id)
        if not session_id:
            return None

        result = self.complete_session(
            session_id=session_id,
            success=success,
            outputs=outputs,
            ftns_spent=ftns_spent,
        )

        # Clean up mappings
        self._protocol_to_session.pop(protocol_id, None)
        self._session_to_protocol.pop(session_id, None)

        if result:
            logger.info(
                f"Protocol {protocol_id[:8]} completed → session {session_id[:8]} "
                f"({'success' if success else 'failed'})"
            )
        return result

    def get_protocol_id(self, session_id: str) -> Optional[str]:
        """Get the P2P protocol ID linked to a session."""
        return self._session_to_protocol.get(session_id)

    def get_session_for_protocol(self, protocol_id: str) -> Optional[CollaborationSession]:
        """Get the session linked to a P2P protocol ID."""
        session_id = self._protocol_to_session.get(protocol_id)
        if session_id:
            return self._sessions.get(session_id)
        return None


_collaboration_manager: Optional[CollaborationManager] = None


def get_collaboration_manager() -> CollaborationManager:
    """Get the global collaboration manager instance"""
    global _collaboration_manager
    if _collaboration_manager is None:
        _collaboration_manager = CollaborationManager()
    return _collaboration_manager


__all__ = [
    'CollaborationType',
    'CollaborationStatus',
    'CollaborationSession',
    'CollaborationProposal',
    'CollaborationResult',
    'CollaborationManager',
    'get_collaboration_manager',
    # Session management
    'SessionManager',
    'SessionState',
    'Participant',
    'ParticipantRole',
    'get_session_manager',
]

"""
PRSM Collaboration Module

Provides collaboration infrastructure for multi-agent interaction over the PRSM P2P network.
This module interfaces with the node-level agent collaboration protocols.

Components:
- Collaboration protocols for task delegation, peer review, and knowledge exchange
- Agent-to-agent messaging and coordination
- Collaboration session management
- Real-time multi-user session support

Note: The core collaboration protocols are implemented in prsm/node/agent_collaboration.py
This package provides additional high-level interfaces and utilities.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

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
    Manager for collaboration sessions and proposals
    
    Coordinates collaboration between agents on the PRSM network.
    """
    
    def __init__(self):
        self._sessions: Dict[str, CollaborationSession] = {}
        self._proposals: Dict[str, CollaborationProposal] = {}
        self._results: Dict[str, CollaborationResult] = {}
    
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

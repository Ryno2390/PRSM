"""
PRSM Collaboration Session Manager

Manages real-time collaboration sessions for multi-user interactions
in the PRSM system.

Usage:
    from prsm.collaboration.session_manager import SessionManager, get_session_manager
    
    manager = get_session_manager()
    session = await manager.create_session(user_id, session_config)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4
import structlog

logger = structlog.get_logger(__name__)


class SessionState(str, Enum):
    """State of a collaboration session"""
    CREATED = "created"
    WAITING = "waiting"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ParticipantRole(str, Enum):
    """Role of a participant in a session"""
    HOST = "host"
    MODERATOR = "moderator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


@dataclass
class Participant:
    """Participant in a collaboration session"""
    user_id: str
    username: str
    role: ParticipantRole = ParticipantRole.PARTICIPANT
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """Update last active timestamp"""
        self.last_active = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role.value,
            "joined_at": self.joined_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


@dataclass
class CollaborationSession:
    """A collaboration session for multi-user interaction"""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    session_name: str = ""
    session_type: str = "collaborative"
    creator_id: str = ""
    state: SessionState = SessionState.CREATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    max_participants: int = 10
    participants: Dict[str, Participant] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    websocket_url: Optional[str] = None
    
    @property
    def participant_count(self) -> int:
        return len([p for p in self.participants.values() if p.is_active])
    
    @property
    def is_full(self) -> bool:
        return self.participant_count >= self.max_participants
    
    def add_participant(self, participant: Participant) -> bool:
        """Add a participant to the session"""
        if self.is_full:
            return False
        
        self.participants[participant.user_id] = participant
        logger.info(
            "Participant added to session",
            session_id=self.session_id,
            user_id=participant.user_id,
            participant_count=self.participant_count,
        )
        return True
    
    def remove_participant(self, user_id: str) -> bool:
        """Remove a participant from the session"""
        if user_id in self.participants:
            self.participants[user_id].is_active = False
            logger.info(
                "Participant removed from session",
                session_id=self.session_id,
                user_id=user_id,
            )
            return True
        return False
    
    def get_participant(self, user_id: str) -> Optional[Participant]:
        """Get a participant by user ID"""
        return self.participants.get(user_id)
    
    def start(self) -> None:
        """Start the session"""
        self.state = SessionState.ACTIVE
        self.started_at = datetime.now(timezone.utc)
        logger.info("Session started", session_id=self.session_id)
    
    def pause(self) -> None:
        """Pause the session"""
        self.state = SessionState.PAUSED
        logger.info("Session paused", session_id=self.session_id)
    
    def resume(self) -> None:
        """Resume a paused session"""
        self.state = SessionState.ACTIVE
        logger.info("Session resumed", session_id=self.session_id)
    
    def end(self) -> None:
        """End the session"""
        self.state = SessionState.COMPLETED
        self.ended_at = datetime.now(timezone.utc)
        logger.info("Session ended", session_id=self.session_id)
    
    def cancel(self) -> None:
        """Cancel the session"""
        self.state = SessionState.CANCELLED
        self.ended_at = datetime.now(timezone.utc)
        logger.info("Session cancelled", session_id=self.session_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "session_type": self.session_type,
            "creator_id": self.creator_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "max_participants": self.max_participants,
            "participant_count": self.participant_count,
            "participants": [p.to_dict() for p in self.participants.values() if p.is_active],
            "config": self.config,
            "websocket_url": self.websocket_url,
        }


class SessionManager:
    """
    Manager for collaboration sessions
    
    Handles session creation, lifecycle, and participant management.
    """
    
    def __init__(self, max_sessions: int = 1000):
        self.max_sessions = max_sessions
        self._sessions: Dict[str, CollaborationSession] = {}
        self._user_sessions: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(
        self,
        user_id: str,
        session_name: str = "",
        session_type: str = "collaborative",
        max_participants: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ) -> CollaborationSession:
        """
        Create a new collaboration session
        
        Args:
            user_id: ID of the user creating the session
            session_name: Name for the session
            session_type: Type of session (collaborative, review, etc.)
            max_participants: Maximum number of participants
            config: Session configuration
        
        Returns:
            Created session
        """
        async with self._lock:
            # Check session limit
            if len(self._sessions) >= self.max_sessions:
                # Remove oldest inactive sessions
                await self._cleanup_inactive_sessions()
            
            session = CollaborationSession(
                session_name=session_name,
                session_type=session_type,
                creator_id=user_id,
                max_participants=max_participants,
                config=config or {},
            )
            
            # Add creator as host
            host = Participant(
                user_id=user_id,
                username=f"user_{user_id[:8]}",
                role=ParticipantRole.HOST,
            )
            session.add_participant(host)
            
            # Generate websocket URL
            session.websocket_url = f"ws://localhost:8000/ws/collab/{session.session_id}"
            
            self._sessions[session.session_id] = session
            
            # Track user sessions
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session.session_id)
            
            logger.info(
                "Session created",
                session_id=session.session_id,
                user_id=user_id,
                session_name=session_name,
            )
            
            return session
    
    async def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get a session by ID"""
        return self._sessions.get(session_id)
    
    async def join_session(
        self,
        session_id: str,
        user_id: str,
        username: Optional[str] = None,
    ) -> Optional[CollaborationSession]:
        """
        Join a collaboration session
        
        Args:
            session_id: ID of session to join
            user_id: ID of user joining
            username: Display name for user
        
        Returns:
            Session if join successful, None otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning("Session not found", session_id=session_id)
                return None
            
            if session.state not in [SessionState.CREATED, SessionState.WAITING, SessionState.ACTIVE]:
                logger.warning("Cannot join session in current state", session_id=session_id, state=session.state)
                return None
            
            if session.is_full:
                logger.warning("Session is full", session_id=session_id)
                return None
            
            participant = Participant(
                user_id=user_id,
                username=username or f"user_{user_id[:8]}",
                role=ParticipantRole.PARTICIPANT,
            )
            
            if not session.add_participant(participant):
                return None
            
            # Track user sessions
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session_id)
            
            logger.info(
                "User joined session",
                session_id=session_id,
                user_id=user_id,
            )
            
            return session
    
    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a collaboration session"""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            # If host leaves, end the session
            participant = session.get_participant(user_id)
            if participant and participant.role == ParticipantRole.HOST:
                session.end()
                logger.info("Host left, ending session", session_id=session_id, user_id=user_id)
            else:
                session.remove_participant(user_id)
            
            # Update user sessions tracking
            if user_id in self._user_sessions:
                self._user_sessions[user_id].discard(session_id)
            
            return True
    
    async def end_session(self, session_id: str, user_id: str) -> bool:
        """End a collaboration session (host only)"""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            participant = session.get_participant(user_id)
            if not participant or participant.role != ParticipantRole.HOST:
                logger.warning("Only host can end session", session_id=session_id, user_id=user_id)
                return False
            
            session.end()
            
            # Clean up user session tracking
            for pid in session.participants:
                if pid in self._user_sessions:
                    self._user_sessions[pid].discard(session_id)
            
            return True
    
    async def get_user_sessions(self, user_id: str) -> List[CollaborationSession]:
        """Get all sessions a user is participating in"""
        session_ids = self._user_sessions.get(user_id, set())
        sessions = []
        for sid in session_ids:
            session = self._sessions.get(sid)
            if session and session.state in [SessionState.CREATED, SessionState.ACTIVE, SessionState.PAUSED]:
                sessions.append(session)
        return sessions
    
    async def get_active_sessions(self) -> List[CollaborationSession]:
        """Get all active sessions"""
        return [
            s for s in self._sessions.values()
            if s.state in [SessionState.ACTIVE, SessionState.WAITING]
        ]
    
    async def _cleanup_inactive_sessions(self) -> None:
        """Remove inactive or completed sessions"""
        to_remove = []
        for session_id, session in self._sessions.items():
            if session.state in [SessionState.COMPLETED, SessionState.CANCELLED]:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self._sessions[session_id]
        
        if to_remove:
            logger.info("Cleaned up inactive sessions", count=len(to_remove))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        states = {}
        for session in self._sessions.values():
            state = session.state.value
            states[state] = states.get(state, 0) + 1
        
        return {
            "total_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "total_users_with_sessions": len(self._user_sessions),
            "sessions_by_state": states,
        }


_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


__all__ = [
    'SessionManager',
    'CollaborationSession',
    'SessionState',
    'Participant',
    'ParticipantRole',
    'get_session_manager',
]

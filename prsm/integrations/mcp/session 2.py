"""
MCP Session Management
=====================

Session management for MCP tool interactions with context preservation.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from .models import MCPSession, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages MCP sessions for context preservation across tool calls
    
    Features:
    - Session lifecycle management
    - Context preservation and sharing
    - Session-based tool call tracking
    - Automatic cleanup of expired sessions
    """
    
    def __init__(self):
        self.sessions: Dict[UUID, MCPSession] = {}
        self.user_sessions: Dict[str, List[UUID]] = {}  # user_id -> session_ids
        
        # Configuration
        self.default_session_timeout = timedelta(hours=2)
        self.max_sessions_per_user = 10
        self.cleanup_interval = timedelta(minutes=30)
        
        self.last_cleanup = datetime.utcnow()
        
        logger.info("Initialized MCP session manager")
    
    async def create_session(self, server_uri: str, user_id: str, 
                           initial_context: Optional[Dict[str, Any]] = None) -> MCPSession:
        """
        Create a new MCP session
        
        Args:
            server_uri: URI of the MCP server
            user_id: ID of the user creating the session
            initial_context: Optional initial context data
            
        Returns:
            Created MCP session
        """
        # Cleanup expired sessions first
        await self._cleanup_expired_sessions()
        
        # Check user session limits
        user_session_count = len(self.user_sessions.get(user_id, []))
        if user_session_count >= self.max_sessions_per_user:
            # Remove oldest session for this user
            await self._remove_oldest_user_session(user_id)
        
        # Create new session
        session = MCPSession(
            server_uri=server_uri,
            user_id=user_id,
            context=initial_context or {}
        )
        
        # Store session
        self.sessions[session.session_id] = session
        
        # Track user sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session.session_id)
        
        logger.info(f"Created MCP session {session.session_id} for user {user_id}")
        return session
    
    async def get_session(self, session_id: UUID) -> Optional[MCPSession]:
        """
        Get an existing session
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            Session if found and active, None otherwise
        """
        session = self.sessions.get(session_id)
        
        if session is None:
            return None
        
        # Check if session is expired
        if self._is_session_expired(session):
            await self.close_session(session_id)
            return None
        
        # Update activity
        session.update_activity()
        return session
    
    async def update_session_context(self, session_id: UUID, 
                                   context_updates: Dict[str, Any]) -> bool:
        """
        Update session context
        
        Args:
            session_id: ID of the session to update
            context_updates: Context data to merge
            
        Returns:
            True if session was found and updated
        """
        session = await self.get_session(session_id)
        if session is None:
            return False
        
        # Merge context updates
        session.context.update(context_updates)
        session.update_activity()
        
        logger.debug(f"Updated context for session {session_id}")
        return True
    
    async def add_tool_call_to_session(self, session_id: UUID, 
                                     tool_call: ToolCall) -> bool:
        """
        Add a tool call to session tracking
        
        Args:
            session_id: ID of the session
            tool_call: Tool call to track
            
        Returns:
            True if session was found and updated
        """
        session = await self.get_session(session_id)
        if session is None:
            return False
        
        # Add tool call ID to session
        session.tool_calls.append(tool_call.call_id)
        session.update_activity()
        
        # Store tool call context in session
        session.context[f"tool_call_{tool_call.call_id}"] = {
            "tool_name": tool_call.tool_name,
            "parameters": tool_call.parameters,
            "timestamp": tool_call.created_at.isoformat()
        }
        
        logger.debug(f"Added tool call {tool_call.call_id} to session {session_id}")
        return True
    
    async def get_session_tool_calls(self, session_id: UUID) -> List[UUID]:
        """
        Get list of tool call IDs for a session
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of tool call IDs
        """
        session = await self.get_session(session_id)
        if session is None:
            return []
        
        return session.tool_calls.copy()
    
    async def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[MCPSession]:
        """
        Get all sessions for a user
        
        Args:
            user_id: ID of the user
            active_only: Whether to return only active sessions
            
        Returns:
            List of user sessions
        """
        session_ids = self.user_sessions.get(user_id, [])
        sessions = []
        
        for session_id in session_ids.copy():  # Copy to avoid modification during iteration
            session = self.sessions.get(session_id)
            
            if session is None:
                # Clean up orphaned session ID
                session_ids.remove(session_id)
                continue
            
            if active_only and not session.active:
                continue
            
            if active_only and self._is_session_expired(session):
                await self.close_session(session_id)
                continue
            
            sessions.append(session)
        
        return sessions
    
    async def close_session(self, session_id: UUID) -> bool:
        """
        Close a session
        
        Args:
            session_id: ID of the session to close
            
        Returns:
            True if session was found and closed
        """
        session = self.sessions.get(session_id)
        if session is None:
            return False
        
        # Mark as inactive
        session.active = False
        
        # Remove from tracking
        user_id = session.user_id
        if user_id in self.user_sessions:
            self.user_sessions[user_id] = [
                sid for sid in self.user_sessions[user_id] 
                if sid != session_id
            ]
            
            # Clean up empty user session list
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
        
        # Remove session
        del self.sessions[session_id]
        
        logger.info(f"Closed MCP session {session_id}")
        return True
    
    async def close_user_sessions(self, user_id: str) -> int:
        """
        Close all sessions for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            Number of sessions closed
        """
        session_ids = self.user_sessions.get(user_id, []).copy()
        closed_count = 0
        
        for session_id in session_ids:
            if await self.close_session(session_id):
                closed_count += 1
        
        logger.info(f"Closed {closed_count} sessions for user {user_id}")
        return closed_count
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        return await self._cleanup_expired_sessions()
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session management statistics"""
        now = datetime.utcnow()
        
        # Count sessions by state
        active_sessions = sum(1 for s in self.sessions.values() if s.active)
        expired_sessions = sum(1 for s in self.sessions.values() if self._is_session_expired(s))
        
        # Calculate session ages
        session_ages = [
            (now - s.created_at).total_seconds() / 3600  # Hours
            for s in self.sessions.values()
        ]
        
        avg_session_age = sum(session_ages) / max(len(session_ages), 1)
        
        # Tool call statistics
        total_tool_calls = sum(len(s.tool_calls) for s in self.sessions.values())
        
        # User distribution
        users_with_sessions = len(self.user_sessions)
        sessions_per_user = [len(sessions) for sessions in self.user_sessions.values()]
        avg_sessions_per_user = sum(sessions_per_user) / max(len(sessions_per_user), 1)
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "expired_sessions": expired_sessions,
            "users_with_sessions": users_with_sessions,
            "average_sessions_per_user": avg_sessions_per_user,
            "total_tool_calls": total_tool_calls,
            "average_session_age_hours": avg_session_age,
            "last_cleanup": self.last_cleanup.isoformat(),
            "default_timeout_hours": self.default_session_timeout.total_seconds() / 3600
        }
    
    # Private methods
    
    def _is_session_expired(self, session: MCPSession) -> bool:
        """Check if a session has expired"""
        if not session.active:
            return True
        
        timeout = datetime.utcnow() - self.default_session_timeout
        return session.last_activity < timeout
    
    async def _cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        now = datetime.utcnow()
        
        # Only cleanup if enough time has passed
        if now - self.last_cleanup < self.cleanup_interval:
            return 0
        
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if self._is_session_expired(session)
        ]
        
        cleanup_count = 0
        for session_id in expired_sessions:
            if await self.close_session(session_id):
                cleanup_count += 1
        
        self.last_cleanup = now
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} expired sessions")
        
        return cleanup_count
    
    async def _remove_oldest_user_session(self, user_id: str):
        """Remove the oldest session for a user"""
        user_session_ids = self.user_sessions.get(user_id, [])
        if not user_session_ids:
            return
        
        # Find oldest session
        oldest_session_id = None
        oldest_time = None
        
        for session_id in user_session_ids:
            session = self.sessions.get(session_id)
            if session and (oldest_time is None or session.created_at < oldest_time):
                oldest_session_id = session_id
                oldest_time = session.created_at
        
        if oldest_session_id:
            await self.close_session(oldest_session_id)
            logger.info(f"Removed oldest session {oldest_session_id} for user {user_id}")
    
    async def get_session_context(self, session_id: UUID, 
                                context_key: Optional[str] = None) -> Optional[Any]:
        """
        Get session context data
        
        Args:
            session_id: ID of the session
            context_key: Specific context key to retrieve, or None for all context
            
        Returns:
            Context data or None if session not found
        """
        session = await self.get_session(session_id)
        if session is None:
            return None
        
        if context_key is None:
            return session.context.copy()
        else:
            return session.context.get(context_key)
    
    async def set_session_context(self, session_id: UUID, 
                                context_key: str, value: Any) -> bool:
        """
        Set a specific context value for a session
        
        Args:
            session_id: ID of the session
            context_key: Context key to set
            value: Value to set
            
        Returns:
            True if session was found and updated
        """
        session = await self.get_session(session_id)
        if session is None:
            return False
        
        session.context[context_key] = value
        session.update_activity()
        
        return True
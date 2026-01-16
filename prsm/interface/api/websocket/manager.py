"""
WebSocket Connection Manager
============================

Manages WebSocket connections for real-time communication in PRSM.
"""

import asyncio
import json
import structlog
from typing import Dict, Any, Set

from fastapi import WebSocket

logger = structlog.get_logger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time communication

    Features:
    - Connection lifecycle management
    - Message broadcasting to specific users or all clients
    - Conversation-specific subscriptions
    - Automatic cleanup and reconnection handling
    """

    def __init__(self) -> None:
        # Active connections: user_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Conversation subscriptions: conversation_id -> Set[WebSocket]
        self.conversation_subscriptions: Dict[str, Set[WebSocket]] = {}
        # Connection metadata: WebSocket -> Dict[str, Any]
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, user_id: str, connection_type: str = "general") -> None:
        """Accept new WebSocket connection"""
        await websocket.accept()

        # Add to user connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)

        # Store connection metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connection_type": connection_type,
            "connected_at": asyncio.get_event_loop().time(),
            "last_activity": asyncio.get_event_loop().time()
        }

        logger.info("WebSocket connected",
                   user_id=user_id,
                   connection_type=connection_type,
                   total_connections=len(self.connection_metadata))

        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": asyncio.get_event_loop().time(),
            "message": "Real-time connection established"
        }, websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection"""
        if websocket not in self.connection_metadata:
            return

        metadata = self.connection_metadata[websocket]
        user_id = metadata["user_id"]

        # Remove from user connections
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

        # Remove from conversation subscriptions
        for conversation_id, subscribers in self.conversation_subscriptions.items():
            subscribers.discard(websocket)

        # Clean up empty conversation subscriptions
        self.conversation_subscriptions = {
            conv_id: subs for conv_id, subs in self.conversation_subscriptions.items()
            if subs
        }

        # Remove metadata
        del self.connection_metadata[websocket]

        logger.info("WebSocket disconnected",
                   user_id=user_id,
                   total_connections=len(self.connection_metadata))

    async def subscribe_to_conversation(self, websocket: WebSocket, conversation_id: str) -> None:
        """Subscribe WebSocket to conversation updates"""
        if conversation_id not in self.conversation_subscriptions:
            self.conversation_subscriptions[conversation_id] = set()

        self.conversation_subscriptions[conversation_id].add(websocket)

        # Update metadata
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata[websocket]
            if "subscriptions" not in metadata:
                metadata["subscriptions"] = set()
            metadata["subscriptions"].add(conversation_id)

        logger.debug("WebSocket subscribed to conversation",
                    conversation_id=conversation_id,
                    subscribers=len(self.conversation_subscriptions[conversation_id]))

    async def unsubscribe_from_conversation(self, websocket: WebSocket, conversation_id: str) -> None:
        """Unsubscribe WebSocket from conversation updates"""
        if conversation_id in self.conversation_subscriptions:
            self.conversation_subscriptions[conversation_id].discard(websocket)

            if not self.conversation_subscriptions[conversation_id]:
                del self.conversation_subscriptions[conversation_id]

        # Update metadata
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata[websocket]
            if "subscriptions" in metadata:
                metadata["subscriptions"].discard(conversation_id)

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> None:
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))

            # Update last activity
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["last_activity"] = asyncio.get_event_loop().time()

        except Exception as e:
            logger.error("Failed to send personal message", error=str(e))
            await self.disconnect(websocket)

    async def send_to_user(self, message: Dict[str, Any], user_id: str) -> None:
        """Send message to all connections for a specific user"""
        if user_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error("Failed to send message to user", user_id=user_id, error=str(e))
                    disconnected.append(websocket)

            # Clean up disconnected websockets
            for ws in disconnected:
                await self.disconnect(ws)

    async def broadcast_to_conversation(self, message: Dict[str, Any], conversation_id: str) -> None:
        """Send message to all subscribers of a conversation"""
        if conversation_id in self.conversation_subscriptions:
            disconnected = []
            for websocket in self.conversation_subscriptions[conversation_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error("Failed to broadcast to conversation",
                               conversation_id=conversation_id,
                               error=str(e))
                    disconnected.append(websocket)

            # Clean up disconnected websockets
            for ws in disconnected:
                await self.disconnect(ws)

    async def broadcast_to_all(self, message: Dict[str, Any]) -> None:
        """Send message to all connected clients"""
        disconnected = []
        for websocket in self.connection_metadata.keys():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error("Failed to broadcast to all", error=str(e))
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for ws in disconnected:
            await self.disconnect(ws)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        return {
            "total_connections": len(self.connection_metadata),
            "unique_users": len(self.active_connections),
            "conversation_subscriptions": len(self.conversation_subscriptions),
            "connections_by_user": {
                user_id: len(connections)
                for user_id, connections in self.active_connections.items()
            }
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager()

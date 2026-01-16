"""
WebSocket Message Handlers
==========================

Handles WebSocket message routing and processing.
"""

import asyncio
import json
import structlog
from typing import Dict, Any

from fastapi import WebSocket

from .manager import websocket_manager

logger = structlog.get_logger(__name__)


async def handle_websocket_message(
    websocket: WebSocket,
    user_id: str,
    message: Dict[str, Any],
    connection=None
) -> None:
    """Handle incoming WebSocket messages with authentication context"""
    from prsm.interface.api.websocket_auth import require_websocket_permission, WebSocketAuthError

    message_type = message.get("type")

    if message_type == "ping":
        # Respond to ping with pong
        await websocket_manager.send_personal_message({
            "type": "pong",
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)

    elif message_type == "subscribe_conversation":
        # Subscribe to conversation updates (requires conversation permission)
        try:
            await require_websocket_permission(websocket, "conversation.read")
        except WebSocketAuthError:
            await websocket_manager.send_personal_message({
                "type": "error",
                "message": "Permission denied: conversation.read required",
                "timestamp": asyncio.get_event_loop().time()
            }, websocket)
            return

        conversation_id = message.get("conversation_id")
        if conversation_id:
            await websocket_manager.subscribe_to_conversation(websocket, conversation_id)
            await websocket_manager.send_personal_message({
                "type": "subscribed",
                "conversation_id": conversation_id,
                "message": f"Subscribed to conversation {conversation_id}"
            }, websocket)

    elif message_type == "unsubscribe_conversation":
        # Unsubscribe from conversation updates
        conversation_id = message.get("conversation_id")
        if conversation_id:
            await websocket_manager.unsubscribe_from_conversation(websocket, conversation_id)
            await websocket_manager.send_personal_message({
                "type": "unsubscribed",
                "conversation_id": conversation_id,
                "message": f"Unsubscribed from conversation {conversation_id}"
            }, websocket)

    elif message_type == "request_status":
        # Send current status information
        stats = websocket_manager.get_connection_stats()
        await websocket_manager.send_personal_message({
            "type": "status_update",
            "user_id": user_id,
            "connection_stats": stats,
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)

    else:
        # Unknown message type
        await websocket_manager.send_personal_message({
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)


async def handle_conversation_message(
    websocket: WebSocket,
    user_id: str,
    conversation_id: str,
    message: Dict[str, Any],
    connection=None
) -> None:
    """Handle conversation-specific WebSocket messages with authentication context"""
    from prsm.interface.api.websocket_auth import require_websocket_permission, WebSocketAuthError

    message_type = message.get("type")

    if message_type == "send_message":
        # Handle real-time message sending with streaming response
        content = message.get("content")
        if not content:
            await websocket_manager.send_personal_message({
                "type": "error",
                "message": "Message content is required"
            }, websocket)
            return

        # Broadcast user message to conversation subscribers
        await websocket_manager.broadcast_to_conversation({
            "type": "user_message",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "content": content,
            "timestamp": asyncio.get_event_loop().time()
        }, conversation_id)

        # Start streaming AI response
        await stream_ai_response(conversation_id, content, user_id)

    elif message_type == "typing_start":
        # Broadcast typing indicator
        await websocket_manager.broadcast_to_conversation({
            "type": "typing_indicator",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "typing": True,
            "timestamp": asyncio.get_event_loop().time()
        }, conversation_id)

    elif message_type == "typing_stop":
        # Broadcast typing stop
        await websocket_manager.broadcast_to_conversation({
            "type": "typing_indicator",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "typing": False,
            "timestamp": asyncio.get_event_loop().time()
        }, conversation_id)


async def stream_ai_response(conversation_id: str, user_message: str, user_id: str) -> None:
    """
    Stream AI response token by token for real-time experience

    In production, this would integrate with actual NWTN orchestrator.
    Currently simulates streaming AI responses for demonstration.
    """
    # Simulate AI response generation
    ai_response = (
        f"Thank you for your message: '{user_message}'. "
        "This is a streaming response from NWTN. "
        "In production, this would be connected to the actual NWTN orchestrator "
        "for real AI processing."
    )

    # Send typing indicator
    await websocket_manager.broadcast_to_conversation({
        "type": "ai_typing",
        "conversation_id": conversation_id,
        "typing": True,
        "timestamp": asyncio.get_event_loop().time()
    }, conversation_id)

    # Stream response token by token
    words = ai_response.split()
    streamed_content = ""

    for i, word in enumerate(words):
        # Add word to streamed content
        if i > 0:
            streamed_content += " "
        streamed_content += word

        # Send partial response
        await websocket_manager.broadcast_to_conversation({
            "type": "ai_response_chunk",
            "conversation_id": conversation_id,
            "partial_content": streamed_content,
            "is_complete": False,
            "timestamp": asyncio.get_event_loop().time()
        }, conversation_id)

        # Simulate typing delay
        await asyncio.sleep(0.1)

    # Send completion message
    await websocket_manager.broadcast_to_conversation({
        "type": "ai_response_complete",
        "conversation_id": conversation_id,
        "final_content": streamed_content,
        "user_id": user_id,
        "model_used": "nwtn-v1",
        "context_tokens": len(user_message.split()) * 1.3 + len(words),
        "timestamp": asyncio.get_event_loop().time()
    }, conversation_id)

    # Stop typing indicator
    await websocket_manager.broadcast_to_conversation({
        "type": "ai_typing",
        "conversation_id": conversation_id,
        "typing": False,
        "timestamp": asyncio.get_event_loop().time()
    }, conversation_id)

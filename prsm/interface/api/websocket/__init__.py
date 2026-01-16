"""
WebSocket Module
================

Real-time communication management for PRSM API.
"""

from .manager import WebSocketManager, websocket_manager
from .handlers import (
    handle_websocket_message,
    handle_conversation_message,
    stream_ai_response
)

__all__ = [
    "WebSocketManager",
    "websocket_manager",
    "handle_websocket_message",
    "handle_conversation_message",
    "stream_ai_response"
]

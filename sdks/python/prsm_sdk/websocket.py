"""
PRSM SDK WebSocket Client
Real-time streaming and event handling
"""

import asyncio
import json
import structlog
from typing import Optional, Callable, Dict, Any, AsyncIterator
from datetime import datetime
from enum import Enum

import aiohttp

from .models import WebSocketMessage
from .exceptions import NetworkError, PRSMError

logger = structlog.get_logger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class WebSocketClient:
    """
    WebSocket client for real-time PRSM features
    
    Provides:
    - Real-time query streaming
    - Session progress updates
    - Network event notifications
    - Automatic reconnection
    """
    
    def __init__(
        self,
        websocket_url: str,
        auth_manager,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0
    ):
        """
        Initialize WebSocket client
        
        Args:
            websocket_url: WebSocket URL (e.g., wss://ws.prsm.ai/v1)
            auth_manager: AuthManager for authentication
            reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Delay between reconnection attempts
            ping_interval: Interval for ping/pong heartbeat
            ping_timeout: Timeout for pong response
        """
        self.websocket_url = websocket_url
        self.auth_manager = auth_manager
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._state = ConnectionState.DISCONNECTED
        self._message_handlers: Dict[str, Callable] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> ConnectionState:
        """Get current connection state"""
        return self._state
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection
        
        Raises:
            NetworkError: If connection fails
        """
        if self._state == ConnectionState.CONNECTED:
            return
        
        self._state = ConnectionState.CONNECTING
        
        try:
            # Create session if needed
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            # Get auth headers
            headers = await self.auth_manager.get_headers()
            
            # Connect to WebSocket
            self._ws = await self._session.ws_connect(
                self.websocket_url,
                headers=headers,
                heartbeat=self.ping_interval,
                receive_timeout=self.ping_timeout
            )
            
            self._state = ConnectionState.CONNECTED
            
            # Start message receiver
            self._receive_task = asyncio.create_task(self._receive_messages())
            
            logger.info("WebSocket connected", url=self.websocket_url)
            
        except Exception as e:
            self._state = ConnectionState.DISCONNECTED
            raise NetworkError(f"Failed to connect WebSocket: {e}")
    
    async def disconnect(self) -> None:
        """
        Close WebSocket connection
        """
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        self._state = ConnectionState.DISCONNECTED
        logger.info("WebSocket disconnected")
    
    async def close(self) -> None:
        """Alias for disconnect()"""
        await self.disconnect()
        
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _receive_messages(self) -> None:
        """Background task to receive and route messages"""
        while self._state == ConnectionState.CONNECTED and self._ws:
            try:
                msg = await self._ws.receive()
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error", error=self._ws.exception())
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket closed by server")
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error receiving message", error=str(e))
                break
        
        self._state = ConnectionState.DISCONNECTED
    
    async def _handle_message(self, data: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            message_data = json.loads(data)
            message_type = message_data.get("type", "unknown")
            
            # Put in queue for streaming queries
            await self._message_queue.put(message_data)
            
            # Call registered handler if exists
            if message_type in self._message_handlers:
                handler = self._message_handlers[message_type]
                await handler(message_data)
                
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON message", error=str(e))
    
    def on_message(self, message_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific message type
        
        Args:
            message_type: Type of message to handle
            handler: Async function to call when message is received
        """
        self._message_handlers[message_type] = handler
    
    def off_message(self, message_type: str) -> None:
        """
        Remove a message handler
        
        Args:
            message_type: Type of message to stop handling
        """
        self._message_handlers.pop(message_type, None)
    
    async def send(self, message: Dict[str, Any]) -> None:
        """
        Send a message through the WebSocket
        
        Args:
            message: Message to send
            
        Raises:
            NetworkError: If not connected
        """
        if self._state != ConnectionState.CONNECTED or not self._ws:
            raise NetworkError("WebSocket not connected")
        
        await self._ws.send_json(message)
    
    async def stream_query(self, request) -> AsyncIterator[WebSocketMessage]:
        """
        Stream query results in real-time
        
        Args:
            request: Query request to stream
            
        Yields:
            WebSocketMessage objects with partial results
        """
        if self._state != ConnectionState.CONNECTED:
            await self.connect()
        
        # Send query request
        await self.send({
            "type": "query",
            "data": request.model_dump()
        })
        
        # Stream responses
        while True:
            try:
                message_data = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=60.0
                )
                
                message = WebSocketMessage(
                    type=message_data.get("type", "unknown"),
                    data=message_data.get("data", {}),
                    request_id=message_data.get("request_id"),
                    timestamp=datetime.utcnow()
                )
                
                yield message
                
                # Check for completion
                if message.type in ("complete", "error"):
                    break
                    
            except asyncio.TimeoutError:
                raise NetworkError("Stream timeout waiting for response")
    
    async def subscribe_to_session(
        self,
        session_id: str,
        on_progress: Optional[Callable] = None
    ) -> None:
        """
        Subscribe to session progress updates
        
        Args:
            session_id: Session to subscribe to
            on_progress: Optional callback for progress updates
        """
        if self._state != ConnectionState.CONNECTED:
            await self.connect()
        
        # Subscribe to session
        await self.send({
            "type": "subscribe",
            "session_id": session_id
        })
        
        # Register progress handler if provided
        if on_progress:
            self.on_message(f"session_{session_id}_progress", on_progress)
    
    async def unsubscribe_from_session(self, session_id: str) -> None:
        """
        Unsubscribe from session updates
        
        Args:
            session_id: Session to unsubscribe from
        """
        await self.send({
            "type": "unsubscribe",
            "session_id": session_id
        })
        
        # Remove handler
        self.off_message(f"session_{session_id}_progress")
    
    async def wait_for_completion(
        self,
        session_id: str,
        timeout: float = 600.0
    ) -> Dict[str, Any]:
        """
        Wait for session completion
        
        Args:
            session_id: Session to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            Final session result
        """
        completion_event = asyncio.Event()
        result = {}
        
        async def on_complete(message: Dict[str, Any]) -> None:
            nonlocal result
            result = message.get("data", {})
            completion_event.set()
        
        self.on_message(f"session_{session_id}_complete", on_complete)
        
        try:
            await asyncio.wait_for(completion_event.wait(), timeout=timeout)
            return result
        finally:
            self.off_message(f"session_{session_id}_complete")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit"""
        await self.close()
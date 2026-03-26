"""
OpenClaw Gateway Client
=======================

WebSocket client for the OpenClaw Gateway at ``ws://127.0.0.1:18789``.

The Gateway is the control plane of an OpenClaw installation.  NWTN
connects to it as a registered agent to:

  1. Receive user messages (the initial goal prompt and interview answers)
  2. Send replies back to the user through whatever channel they used
     (Telegram, Slack, WhatsApp, etc.)
  3. Receive heartbeat events (used by HeartbeatHook to trigger synthesis)
  4. Receive agent file-update events (supplement to watchfiles monitoring)

Protocol
--------
All messages are JSON frames over the WebSocket.

Inbound (Gateway → NWTN):

    {"type": "message",   "id": "…", "from": "user_id", "channel": "telegram",
     "text": "Build me a BSC system"}

    {"type": "heartbeat", "timestamp": "2026-03-26T18:00:00Z"}

    {"type": "agent_event", "agent_id": "coder-20260326",
     "event": "file_updated", "path": "/…/MEMORY.md"}

    {"type": "system",    "event": "connected", "session_token": "…"}

Outbound (NWTN → Gateway):

    {"type": "register",  "agent_id": "nwtn", "version": "1.0"}

    {"type": "reply",     "reply_to": "msg-id", "to": "user_id", "text": "…"}

    {"type": "ask",       "id": "ask-id",       "to": "user_id", "text": "…"}

Dependency injection
--------------------
The ``connection`` parameter accepts any object implementing the
``AbstractGatewayConnection`` protocol, making the Gateway fully testable
without a real WebSocket server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_URL = "ws://127.0.0.1:18789"
NWTN_AGENT_ID       = "nwtn"
RECONNECT_DELAY_S   = 5.0


# ======================================================================
# Data models
# ======================================================================

@dataclass
class OpenClawMessage:
    """A single message received from the OpenClaw Gateway."""
    type: str
    """'message' | 'heartbeat' | 'agent_event' | 'system'"""
    id: str = ""
    from_user: Optional[str] = None
    channel: Optional[str] = None
    text: Optional[str] = None
    agent_id: Optional[str] = None
    event: Optional[str] = None
    path: Optional[str] = None
    timestamp: Optional[datetime] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpenClawMessage":
        ts = d.get("timestamp")
        return cls(
            type=d.get("type", "unknown"),
            id=d.get("id", ""),
            from_user=d.get("from"),
            channel=d.get("channel"),
            text=d.get("text"),
            agent_id=d.get("agent_id"),
            event=d.get("event"),
            path=d.get("path"),
            timestamp=datetime.fromisoformat(ts) if ts else None,
            raw=d,
        )


# ======================================================================
# Abstract connection protocol (dependency injection)
# ======================================================================

@runtime_checkable
class AbstractGatewayConnection(Protocol):
    """
    Minimal protocol for a live Gateway connection.

    The real implementation wraps an ``aiohttp`` ClientWebSocketResponse.
    Tests inject a ``MockGatewayConnection``.
    """

    async def send_json(self, data: Dict[str, Any]) -> None: ...
    async def receive_json(self) -> Optional[Dict[str, Any]]: ...
    async def close(self) -> None: ...

    @property
    def closed(self) -> bool: ...


class _AiohttpConnection:
    """Wraps an aiohttp WebSocket response into AbstractGatewayConnection."""

    def __init__(self, ws) -> None:
        self._ws = ws

    async def send_json(self, data: Dict[str, Any]) -> None:
        await self._ws.send_json(data)

    async def receive_json(self) -> Optional[Dict[str, Any]]:
        msg = await self._ws.receive()
        import aiohttp
        if msg.type == aiohttp.WSMsgType.TEXT:
            return json.loads(msg.data)
        if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
            return None
        return None

    async def close(self) -> None:
        await self._ws.close()

    @property
    def closed(self) -> bool:
        return self._ws.closed


# ======================================================================
# OpenClawGateway
# ======================================================================

class OpenClawGateway:
    """
    NWTN's connection to the OpenClaw Gateway.

    Parameters
    ----------
    url : str
        WebSocket URL of the OpenClaw Gateway.
    connection : AbstractGatewayConnection, optional
        Injected connection for testing.  If None, a real aiohttp
        WebSocket connection is established on ``connect()``.
    """

    def __init__(
        self,
        url: str = DEFAULT_GATEWAY_URL,
        connection: Optional[AbstractGatewayConnection] = None,
    ) -> None:
        self._url = url
        self._conn: Optional[AbstractGatewayConnection] = connection
        self._session = None   # aiohttp.ClientSession (real path only)
        self._connected = False

        # Pending "ask" replies: ask_id → asyncio.Future[str]
        self._pending_asks: Dict[str, asyncio.Future] = {}

        # Inbound message queue
        self._queue: asyncio.Queue[OpenClawMessage] = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the Gateway and register NWTN as an agent."""
        if self._connected:
            return

        if self._conn is None:
            # Real aiohttp connection
            import aiohttp
            self._session = aiohttp.ClientSession()
            ws = await self._session.ws_connect(self._url)
            self._conn = _AiohttpConnection(ws)
            logger.info("OpenClawGateway: connected to %s", self._url)

        await self._conn.send_json({
            "type": "register",
            "agent_id": NWTN_AGENT_ID,
            "version": "1.0",
            "capabilities": ["coordination", "reasoning", "synthesis"],
        })

        self._connected = True
        self._reader_task = asyncio.create_task(
            self._reader_loop(), name="openclaw-gateway-reader"
        )
        logger.info("OpenClawGateway: registered as '%s'", NWTN_AGENT_ID)

    async def disconnect(self) -> None:
        """Disconnect from the Gateway."""
        self._connected = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._conn:
            await self._conn.close()
        if self._session:
            await self._session.close()
        logger.info("OpenClawGateway: disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._conn is not None and not self._conn.closed

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send_text(self, text: str, reply_to: str = "", to_user: str = "") -> None:
        """Send a text reply to the user via the Gateway."""
        if not self.is_connected:
            logger.warning("OpenClawGateway: not connected — cannot send")
            return
        await self._conn.send_json({
            "type": "reply",
            "reply_to": reply_to,
            "to": to_user,
            "text": text,
        })

    async def ask_user(self, question: str, to_user: str = "") -> str:
        """
        Send a question and wait for the user's reply.

        Implements the ``QuestionCallback`` protocol used by ``InterviewSession``.

        Parameters
        ----------
        question : str
        to_user : str, optional
            User ID to direct the question to.

        Returns
        -------
        str
            The user's answer.
        """
        if not self.is_connected:
            return ""

        ask_id = str(uuid.uuid4())
        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._pending_asks[ask_id] = fut

        await self._conn.send_json({
            "type": "ask",
            "id": ask_id,
            "to": to_user,
            "text": question,
        })

        try:
            answer = await asyncio.wait_for(fut, timeout=300.0)  # 5-minute timeout
            return answer
        except asyncio.TimeoutError:
            self._pending_asks.pop(ask_id, None)
            logger.warning("OpenClawGateway: ask timed out for question: %s", question[:60])
            return ""

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    async def messages(self) -> AsyncIterator[OpenClawMessage]:
        """
        Async generator yielding inbound messages from the Gateway.

        Stops when the Gateway disconnects or ``disconnect()`` is called.
        """
        while self._connected:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield msg
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Internal reader loop
    # ------------------------------------------------------------------

    async def _reader_loop(self) -> None:
        """Background task: reads raw WebSocket frames and enqueues them."""
        try:
            while self._connected and not self._conn.closed:
                data = await self._conn.receive_json()
                if data is None:
                    logger.info("OpenClawGateway: connection closed by peer")
                    break
                msg = OpenClawMessage.from_dict(data)
                await self._route_message(msg)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("OpenClawGateway: reader error: %s", exc)
        finally:
            self._connected = False

    async def _route_message(self, msg: OpenClawMessage) -> None:
        """
        Route an inbound message to the right handler.

        - ``answer`` messages resolve pending ask futures.
        - All other messages go to the public queue.
        """
        if msg.type == "answer" and msg.id in self._pending_asks:
            fut = self._pending_asks.pop(msg.id)
            if not fut.done():
                fut.set_result(msg.text or "")
            return

        await self._queue.put(msg)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "OpenClawGateway":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()

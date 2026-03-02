"""
WebSocket P2P Transport
========================

Real peer-to-peer connectivity over WebSockets.
Handles incoming/outgoing connections, message routing,
handshake protocol, and connection health monitoring.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import websockets
import websockets.server
import websockets.client

from prsm.node.identity import NodeIdentity, verify_signature

logger = logging.getLogger(__name__)

# Message type constants
MSG_HANDSHAKE = "handshake"
MSG_HANDSHAKE_ACK = "handshake_ack"
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_GOSSIP = "gossip"
MSG_DIRECT = "direct"


@dataclass
class P2PMessage:
    """A signed message exchanged between peers."""
    msg_type: str
    sender_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    signature: str = ""
    ttl: int = 5
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_bytes(self) -> bytes:
        """Serialize for signing/sending (excludes signature)."""
        data = {
            "msg_type": self.msg_type,
            "sender_id": self.sender_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "nonce": self.nonce,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

    def to_json(self) -> str:
        """Full serialization including signature."""
        data = {
            "msg_type": self.msg_type,
            "sender_id": self.sender_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "ttl": self.ttl,
            "nonce": self.nonce,
        }
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "P2PMessage":
        data = json.loads(raw)
        return cls(
            msg_type=data["msg_type"],
            sender_id=data["sender_id"],
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            signature=data.get("signature", ""),
            ttl=data.get("ttl", 5),
            nonce=data.get("nonce", uuid.uuid4().hex[:16]),
        )

    def sign(self, identity: NodeIdentity) -> None:
        """Sign this message with the node's private key."""
        self.signature = identity.sign(self.to_bytes())


@dataclass
class PeerConnection:
    """Wraps a WebSocket connection with peer metadata."""
    peer_id: str
    address: str
    websocket: Any  # websockets connection object
    public_key_b64: str = ""
    display_name: str = ""
    roles: List[str] = field(default_factory=list)
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    outbound: bool = False  # True if we initiated the connection


# Type alias for message handler callbacks
MessageHandler = Callable[[P2PMessage, "PeerConnection"], Coroutine[Any, Any, None]]


class WebSocketTransport:
    """WebSocket-based P2P transport layer.

    Manages both a server (for incoming connections) and client connections
    (outgoing to other peers). Provides message routing and health monitoring.
    
    Thread Safety:
        All peer connection operations are protected by asyncio locks to prevent
        race conditions when multiple coroutines access shared state concurrently.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        host: str = "0.0.0.0",
        port: int = 9001,
        nonce_window: float = 300.0,
        ws_ping_interval: float = 20.0,
        ws_ping_timeout: float = 10.0,
        handshake_timeout: float = 10.0,
        nonce_cleanup_interval: float = 60.0,
    ):
        self.identity = identity
        self.host = host
        self.port = port
        self.ws_ping_interval = ws_ping_interval
        self.ws_ping_timeout = ws_ping_timeout
        self.handshake_timeout = handshake_timeout
        self.nonce_cleanup_interval = nonce_cleanup_interval

        self.peers: Dict[str, PeerConnection] = {}  # peer_id -> connection
        self._handlers: Dict[str, List[MessageHandler]] = {}
        self._seen_nonces: Set[str] = set()
        self._nonce_window = nonce_window
        self._nonce_timestamps: Dict[str, float] = {}
        self._server = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Thread safety locks for concurrent access to shared state
        self._peers_lock = asyncio.Lock()  # Protects self.peers dictionary
        self._nonces_lock = asyncio.Lock()  # Protects _seen_nonces and _nonce_timestamps

    @property
    def peer_count(self) -> int:
        """Return the number of connected peers (thread-safe read)."""
        # Note: This property is not async, so we can't use the lock here.
        # For accurate count in async context, use get_peer_count() instead.
        return len(self.peers)

    @property
    def peer_addresses(self) -> List[str]:
        """Return list of peer addresses (thread-safe snapshot)."""
        # Note: For accurate snapshot in async context, use get_peer_addresses() instead.
        return [p.address for p in self.peers.values()]

    async def get_peer_count(self) -> int:
        """Async method to get peer count with proper locking."""
        async with self._peers_lock:
            return len(self.peers)

    async def get_peer_addresses(self) -> List[str]:
        """Async method to get peer addresses with proper locking."""
        async with self._peers_lock:
            return [p.address for p in self.peers.values()]

    # ── Message handler registration ─────────────────────────────

    def on_message(self, msg_type: str, handler: MessageHandler) -> None:
        """Register a handler for a specific message type."""
        self._handlers.setdefault(msg_type, []).append(handler)

    async def _dispatch(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Route a message to all registered handlers for its type."""
        handlers = self._handlers.get(msg.msg_type, [])
        for handler in handlers:
            try:
                await handler(msg, peer)
            except Exception as e:
                logger.error(f"Handler error for {msg.msg_type}: {e}")

    # ── Server (incoming connections) ────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket server and background tasks."""
        self._running = True
        self._server = await websockets.server.serve(
            self._handle_incoming,
            self.host,
            self.port,
            ping_interval=self.ws_ping_interval,
            ping_timeout=self.ws_ping_timeout,
        )
        self._tasks.append(asyncio.create_task(self._nonce_cleanup_loop()))
        logger.info(f"P2P transport listening on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Gracefully shut down server and all peer connections."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        # Close all peer connections with proper locking
        async with self._peers_lock:
            for peer in list(self.peers.values()):
                try:
                    await peer.websocket.close()
                except Exception:
                    pass
            self.peers.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("P2P transport stopped")

    async def _handle_incoming(self, websocket: Any) -> None:
        """Handle a new incoming WebSocket connection."""
        peer = None
        try:
            # Wait for handshake
            raw = await asyncio.wait_for(websocket.recv(), timeout=self.handshake_timeout)
            msg = P2PMessage.from_json(raw)

            if msg.msg_type != MSG_HANDSHAKE:
                await websocket.close(1002, "Expected handshake")
                return

            peer_id = msg.sender_id
            if peer_id == self.identity.node_id:
                await websocket.close(1002, "Self-connection rejected")
                return

            # Thread-safe check and add for peer connection
            async with self._peers_lock:
                if peer_id in self.peers:
                    await websocket.close(1002, "Already connected")
                    return

            # Verify handshake signature
            pub_key_b64 = msg.payload.get("public_key", "")
            if pub_key_b64 and not verify_signature(pub_key_b64, msg.to_bytes(), msg.signature):
                await websocket.close(1002, "Invalid signature")
                return

            peer = PeerConnection(
                peer_id=peer_id,
                address=f"{websocket.remote_address[0]}:{websocket.remote_address[1]}",
                websocket=websocket,
                public_key_b64=pub_key_b64,
                display_name=msg.payload.get("display_name", ""),
                roles=msg.payload.get("roles", []),
                outbound=False,
            )
            
            # Thread-safe peer addition
            async with self._peers_lock:
                # Double-check after acquiring lock (race condition prevention)
                if peer_id in self.peers:
                    await websocket.close(1002, "Already connected")
                    return
                self.peers[peer_id] = peer
                peer_count = len(self.peers)

            # Send handshake acknowledgment
            ack = P2PMessage(
                msg_type=MSG_HANDSHAKE_ACK,
                sender_id=self.identity.node_id,
                payload={
                    "public_key": self.identity.public_key_b64,
                    "display_name": self.identity.display_name if hasattr(self.identity, 'display_name') else "",
                    "peer_count": peer_count,
                },
            )
            ack.sign(self.identity)
            await websocket.send(ack.to_json())

            logger.info(f"Peer connected (inbound): {peer_id[:8]}... from {peer.address}")
            await self._dispatch(
                P2PMessage(msg_type="peer_connected", sender_id=peer_id, payload={"direction": "inbound"}),
                peer,
            )

            # Message loop
            await self._read_loop(peer)

        except asyncio.TimeoutError:
            logger.debug("Incoming connection timed out during handshake")
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Error handling incoming connection: {e}")
        finally:
            if peer:
                # Thread-safe peer removal
                async with self._peers_lock:
                    if peer.peer_id in self.peers:
                        del self.peers[peer.peer_id]
                        logger.info(f"Peer disconnected: {peer.peer_id[:8]}...")
                # Dispatch outside lock to prevent deadlock
                if peer:
                    await self._dispatch(
                        P2PMessage(msg_type="peer_disconnected", sender_id=peer.peer_id, payload={}),
                        peer,
                    )

    # ── Client (outgoing connections) ────────────────────────────

    async def connect_to_peer(self, address: str) -> Optional[PeerConnection]:
        """Initiate an outgoing connection to a peer at address (host:port).

        Addresses that already start with ``ws://`` or ``wss://`` are used
        as-is.  Plain ``host:port`` addresses default to ``ws://``, except
        when the port is 443 which implies TLS (``wss://``).
        """
        if address.startswith(("ws://", "wss://")):
            uri = address
        elif address.endswith(":443"):
            uri = f"wss://{address}"
        else:
            uri = f"ws://{address}"
        try:
            websocket = await websockets.client.connect(uri, open_timeout=self.handshake_timeout)

            # Send handshake
            hs = P2PMessage(
                msg_type=MSG_HANDSHAKE,
                sender_id=self.identity.node_id,
                payload={
                    "public_key": self.identity.public_key_b64,
                    "display_name": self.identity.display_name if hasattr(self.identity, 'display_name') else "",
                    "roles": [],
                },
            )
            hs.sign(self.identity)
            await websocket.send(hs.to_json())

            # Wait for ack
            raw = await asyncio.wait_for(websocket.recv(), timeout=self.handshake_timeout)
            ack = P2PMessage.from_json(raw)

            if ack.msg_type != MSG_HANDSHAKE_ACK:
                await websocket.close()
                return None

            peer = PeerConnection(
                peer_id=ack.sender_id,
                address=address,
                websocket=websocket,
                public_key_b64=ack.payload.get("public_key", ""),
                display_name=ack.payload.get("display_name", ""),
                outbound=True,
            )

            # Thread-safe check and add for peer connection
            async with self._peers_lock:
                if peer.peer_id in self.peers:
                    await websocket.close()
                    return self.peers[peer.peer_id]
                self.peers[peer.peer_id] = peer

            logger.info(f"Peer connected (outbound): {peer.peer_id[:8]}... at {address}")

            # Start reading in background
            task = asyncio.create_task(self._read_loop(peer))
            self._tasks.append(task)

            await self._dispatch(
                P2PMessage(msg_type="peer_connected", sender_id=peer.peer_id, payload={"direction": "outbound"}),
                peer,
            )
            return peer

        except Exception as e:
            logger.debug(f"Failed to connect to {address}: {e}")
            return None

    # ── Messaging ────────────────────────────────────────────────

    async def send_to_peer(self, peer_id: str, msg: P2PMessage) -> bool:
        """Send a message directly to a specific peer (thread-safe)."""
        async with self._peers_lock:
            peer = self.peers.get(peer_id)
            if not peer:
                return False
            # Get reference to websocket while holding lock
            websocket = peer.websocket
        
        try:
            msg.sign(self.identity)
            await websocket.send(msg.to_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send to {peer_id[:8]}: {e}")
            return False

    async def broadcast(self, msg: P2PMessage) -> int:
        """Send a message to ALL connected peers (thread-safe)."""
        msg.sign(self.identity)
        raw = msg.to_json()
        
        # Get snapshot of peers under lock
        async with self._peers_lock:
            peers_snapshot = list(self.peers.values())
        
        sent = 0
        for peer in peers_snapshot:
            try:
                await peer.websocket.send(raw)
                sent += 1
            except Exception:
                pass
        return sent

    async def gossip(self, msg: P2PMessage, fanout: int = 3) -> int:
        """Send a message to a random subset of peers (gossip protocol, thread-safe)."""
        import random
        
        # Get snapshot of peers under lock
        async with self._peers_lock:
            targets = list(self.peers.values())
        
        if len(targets) > fanout:
            targets = random.sample(targets, fanout)

        msg.sign(self.identity)
        raw = msg.to_json()
        sent = 0
        for peer in targets:
            try:
                await peer.websocket.send(raw)
                sent += 1
            except Exception:
                pass
        return sent

    # ── Internal message loop ────────────────────────────────────

    async def _read_loop(self, peer: PeerConnection) -> None:
        """Read messages from a peer until disconnect."""
        try:
            async for raw in peer.websocket:
                try:
                    msg = P2PMessage.from_json(raw)
                    peer.last_seen = time.time()

                    # Thread-safe dedup by nonce
                    async with self._nonces_lock:
                        if msg.nonce in self._seen_nonces:
                            continue
                        self._seen_nonces.add(msg.nonce)
                        self._nonce_timestamps[msg.nonce] = time.time()

                    # Handle pings internally
                    if msg.msg_type == MSG_PING:
                        pong = P2PMessage(
                            msg_type=MSG_PONG,
                            sender_id=self.identity.node_id,
                            payload={"echo": msg.nonce},
                        )
                        pong.sign(self.identity)
                        await peer.websocket.send(pong.to_json())
                        continue

                    await self._dispatch(msg, peer)

                except json.JSONDecodeError:
                    logger.debug(f"Invalid JSON from {peer.peer_id[:8]}")
                except Exception as e:
                    logger.error(f"Error processing message from {peer.peer_id[:8]}: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Thread-safe peer removal
            async with self._peers_lock:
                if peer.peer_id in self.peers:
                    del self.peers[peer.peer_id]
                    logger.info(f"Peer disconnected: {peer.peer_id[:8]}...")
            # Dispatch outside lock to prevent deadlock
            await self._dispatch(
                P2PMessage(msg_type="peer_disconnected", sender_id=peer.peer_id, payload={}),
                peer,
            )

    async def _nonce_cleanup_loop(self) -> None:
        """Periodically clean up old nonces to prevent memory growth (thread-safe)."""
        while self._running:
            await asyncio.sleep(self.nonce_cleanup_interval)
            cutoff = time.time() - self._nonce_window
            
            # Thread-safe nonce cleanup
            async with self._nonces_lock:
                expired = [n for n, t in self._nonce_timestamps.items() if t < cutoff]
                for n in expired:
                    self._seen_nonces.discard(n)
                    del self._nonce_timestamps[n]

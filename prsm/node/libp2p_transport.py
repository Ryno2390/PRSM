"""
libp2p Transport Adapter
========================

Drop-in replacement for WebSocketTransport that delegates to a Go-based
libp2p shared library via ctypes FFI.  The Go daemon communicates back to
Python over a Unix Domain Socket (UDS) with a 4-byte length-prefixed JSON
protocol.

The shared library exposes 13 C functions (see libp2p/build/).
"""

import asyncio
import collections
import ctypes
import json
import logging
import os
import platform
import struct
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from prsm.node.identity import NodeIdentity
from prsm.node.transport import P2PMessage, PeerConnection, MSG_GOSSIP, MSG_DIRECT

logger = logging.getLogger(__name__)

# Type alias matching WebSocketTransport
MessageHandler = Callable[[P2PMessage, PeerConnection], Coroutine[Any, Any, None]]


class Libp2pTransportError(Exception):
    """Raised when the libp2p shared library cannot be loaded or returns an error."""


class Libp2pTransport:
    """libp2p-backed P2P transport layer.

    Provides the same public interface as ``WebSocketTransport`` so higher-level
    code can swap transports without changes.  Under the hood every network
    operation is forwarded to the Go shared library via ctypes.
    """

    # ── Construction ────────────────────────────────────────────

    def __init__(
        self,
        identity: NodeIdentity,
        host: str = "0.0.0.0",
        port: int = 9001,
        library_path: str = "",
        **kwargs: Any,
    ):
        self._identity = identity
        self.host = host
        self.port = port

        # Load shared library
        self._lib = self._load_library(library_path)
        self._setup_ctypes()

        # Runtime state
        self._handle: int = -1
        self._uds_path: str = ""
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._running = False

        # Handler registry  (msg_type -> list[handler])
        self._handlers: Dict[str, List[MessageHandler]] = {}

        # Compatibility shim — keyed peer cache
        self._peers: Dict[str, PeerConnection] = {}

        # Telemetry counters (mirroring WebSocketTransport)
        self._telemetry: Dict[str, Any] = {
            "messages_sent": 0,
            "messages_received": 0,
            "publish_count": 0,
            "connect_count": 0,
            "error_count": 0,
            "dispatch_success_total": 0,
            "dispatch_failure_total": 0,
            "dispatch_failure_reasons": collections.Counter(),
        }

    # ── Properties ──────────────────────────────────────────────

    @property
    def identity(self) -> NodeIdentity:
        return self._identity

    @property
    def peer_count(self) -> int:
        if self._handle < 0:
            return 0
        rc = self._lib.PrsmPeerCount(self._handle)
        return rc if rc >= 0 else 0

    @property
    def peer_addresses(self) -> List[str]:
        if self._handle < 0:
            return []
        ptr = self._lib.PrsmPeerList(self._handle)
        raw = self._read_and_free(ptr)
        if not raw:
            return []
        try:
            peers = json.loads(raw)
            return [p.get("addr", "") for p in peers if isinstance(p, dict)]
        except (json.JSONDecodeError, TypeError):
            return []

    @property
    def peers(self) -> Dict[str, PeerConnection]:
        """Compatibility shim — returns cached peer dict."""
        return self._peers

    # ── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Start the Go libp2p daemon and UDS reader."""
        self._loop = asyncio.get_running_loop()

        # Build UDS path
        self._uds_path = os.path.join(
            tempfile.gettempdir(),
            f"prsm_p2p_{self._identity.node_id[:8]}_{uuid.uuid4().hex[:8]}.sock",
        )

        # Build 64-byte Ed25519 key (seed || public)
        seed = self._identity.private_key_bytes  # 32 bytes
        pub = self._identity.public_key_bytes     # 32 bytes
        ed25519_key_64 = seed + pub

        handle = self._lib.PrsmStart(
            ed25519_key_64,
            self.port,
            b"",  # no bootstrap addrs
            self._uds_path.encode("utf-8"),
        )
        if handle < 0:
            raise Libp2pTransportError("PrsmStart returned negative handle")
        self._handle = handle
        self._running = True

        # Give Go time to bind the UDS
        await asyncio.sleep(0.3)

        # Start UDS reader
        self._reader_task = asyncio.create_task(self._uds_reader())
        logger.info(
            "libp2p transport started  handle=%d  uds=%s  port=%d",
            self._handle, self._uds_path, self.port,
        )

    async def stop(self) -> None:
        """Stop the Go daemon, cancel the reader, clean up."""
        self._running = False

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._handle >= 0:
            self._lib.PrsmStop(self._handle)
            self._handle = -1

        # Clean up UDS socket file
        if self._uds_path:
            try:
                os.unlink(self._uds_path)
            except FileNotFoundError:
                pass
            self._uds_path = ""

        self._peers.clear()
        logger.info("libp2p transport stopped")

    # ── Peer operations ─────────────────────────────────────────

    async def connect_to_peer(self, address: str) -> Optional[PeerConnection]:
        """Connect to a peer by address (host:port, ws(s)://…, or /ip4/…)."""
        if self._handle < 0:
            return None

        maddr = self._to_multiaddr(address)
        ptr = self._lib.PrsmConnect(self._handle, maddr.encode("utf-8"))
        peer_id_str = self._read_and_free(ptr)
        if not peer_id_str:
            self._telemetry["error_count"] += 1
            return None

        peer = PeerConnection(
            peer_id=peer_id_str,
            address=address,
            websocket=None,  # no WS object in libp2p mode
            outbound=True,
        )
        self._peers[peer_id_str] = peer
        self._telemetry["connect_count"] += 1
        return peer

    # ── Messaging ───────────────────────────────────────────────

    async def send_to_peer(self, peer_id: str, msg: P2PMessage) -> bool:
        """Send a direct message to a specific peer."""
        if self._handle < 0:
            return False
        msg.sign(self._identity)
        data = msg.to_json().encode("utf-8")
        rc = self._lib.PrsmSend(
            self._handle,
            peer_id.encode("utf-8"),
            b"/prsm/direct/1.0.0",
            data,
            len(data),
        )
        if rc == 0:
            self._telemetry["messages_sent"] += 1
            return True
        self._telemetry["error_count"] += 1
        return False

    async def broadcast(self, msg: P2PMessage) -> int:
        """Broadcast via gossip (compatibility alias)."""
        return await self.gossip(msg)

    async def gossip(self, msg: P2PMessage, fanout: int = 3) -> int:
        """Publish a message over GossipSub."""
        if self._handle < 0:
            return 0

        msg.sign(self._identity)

        # Derive topic from payload subtype
        subtype = ""
        if isinstance(msg.payload, dict):
            subtype = msg.payload.get("subtype", msg.payload.get("type", "general"))
        if not subtype:
            subtype = "general"
        topic = f"prsm/{subtype}"

        data = msg.to_json().encode("utf-8")
        rc = self._lib.PrsmPublish(
            self._handle,
            topic.encode("utf-8"),
            data,
            len(data),
        )
        if rc == 0:
            self._telemetry["publish_count"] += 1
            self._telemetry["messages_sent"] += 1
            return 1
        self._telemetry["error_count"] += 1
        return 0

    # ── Handler registration ────────────────────────────────────

    def on_message(self, msg_type: str, handler: MessageHandler) -> None:
        """Register a callback for a given message type."""
        self._handlers.setdefault(msg_type, []).append(handler)

    # ── Telemetry ───────────────────────────────────────────────

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return a stable copy of transport telemetry counters."""
        return {
            "messages_sent": int(self._telemetry["messages_sent"]),
            "messages_received": int(self._telemetry["messages_received"]),
            "publish_count": int(self._telemetry["publish_count"]),
            "connect_count": int(self._telemetry["connect_count"]),
            "error_count": int(self._telemetry["error_count"]),
            "dispatch_success_total": int(self._telemetry["dispatch_success_total"]),
            "dispatch_failure_total": int(self._telemetry["dispatch_failure_total"]),
            "dispatch_failure_reasons": dict(self._telemetry["dispatch_failure_reasons"]),
        }

    # ── DHT helpers ─────────────────────────────────────────────

    async def dht_provide(self, key: str) -> bool:
        if self._handle < 0:
            return False
        return self._lib.PrsmDHTProvide(self._handle, key.encode("utf-8")) == 0

    async def dht_find_providers(self, key: str, limit: int = 20) -> List[str]:
        if self._handle < 0:
            return []
        ptr = self._lib.PrsmDHTFindProviders(
            self._handle, key.encode("utf-8"), limit,
        )
        raw = self._read_and_free(ptr)
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

    async def get_nat_status(self) -> str:
        if self._handle < 0:
            return "unknown"
        ptr = self._lib.PrsmGetNATStatus(self._handle)
        return self._read_and_free(ptr) or "unknown"

    # ── UDS reader (data plane from Go) ─────────────────────────

    async def _uds_reader(self) -> None:
        """Connect to the Go daemon's UDS and read length-prefixed JSON frames."""
        try:
            reader, _ = await asyncio.open_unix_connection(self._uds_path)
            while self._running:
                # 4-byte big-endian length prefix
                header = await reader.readexactly(4)
                length = struct.unpack(">I", header)[0]
                payload = await reader.readexactly(length)
                try:
                    raw = json.loads(payload.decode("utf-8"))
                    await self._dispatch(raw)
                except (json.JSONDecodeError, Exception) as exc:
                    logger.debug("UDS frame decode error: %s", exc)
        except asyncio.IncompleteReadError:
            logger.debug("UDS reader: Go side closed the connection")
        except asyncio.CancelledError:
            raise  # let cancellation propagate
        except ConnectionRefusedError:
            logger.warning("UDS reader: could not connect to %s", self._uds_path)
        except Exception as exc:
            logger.error("UDS reader error: %s", exc)

    async def _dispatch(self, raw: dict) -> None:
        """Route an incoming UDS frame to registered handlers."""
        msg_type = raw.get("msg_type", raw.get("type", ""))
        if not msg_type:
            return

        self._telemetry["messages_received"] += 1

        # Reconstruct P2PMessage
        try:
            msg = P2PMessage(
                msg_type=msg_type,
                sender_id=raw.get("sender_id", raw.get("from", "")),
                payload=raw.get("payload", raw.get("data", {})),
                timestamp=raw.get("timestamp", time.time()),
                signature=raw.get("signature", ""),
                ttl=raw.get("ttl", 5),
                nonce=raw.get("nonce", uuid.uuid4().hex[:16]),
            )
        except Exception as exc:
            logger.debug("Failed to reconstruct P2PMessage: %s", exc)
            return

        # Build a stub PeerConnection for the sender
        peer = self._peers.get(msg.sender_id) or PeerConnection(
            peer_id=msg.sender_id,
            address=raw.get("remote_addr", ""),
            websocket=None,
        )

        handlers = self._handlers.get(msg_type, [])
        for handler in handlers:
            try:
                await handler(msg, peer)
                self._telemetry["dispatch_success_total"] += 1
            except Exception as exc:
                logger.error("Handler error for %s: %s", msg_type, exc)
                self._telemetry["dispatch_failure_total"] += 1
                self._telemetry["dispatch_failure_reasons"][
                    type(exc).__name__
                ] += 1

    # ── Internal helpers ────────────────────────────────────────

    @staticmethod
    def _load_library(path: str) -> ctypes.CDLL:
        """Locate and load the Go shared library.

        Search order:
        1. Explicit *path* argument.
        2. ``libp2p/build/`` relative to the repository root (two levels up
           from this file).
        """
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError as exc:
                raise Libp2pTransportError(
                    f"Cannot load shared library at {path}: {exc}"
                ) from exc

        # Auto-detect from repo layout
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "darwin":
            ext = "dylib"
        elif system == "windows":
            ext = "dll"
        else:
            ext = "so"

        arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
        lib_name = f"libprsm_p2p_{system}_{arch}.{ext}"

        repo_root = Path(__file__).resolve().parent.parent.parent
        lib_path = repo_root / "libp2p" / "build" / lib_name

        try:
            return ctypes.CDLL(str(lib_path))
        except OSError as exc:
            raise Libp2pTransportError(
                f"Cannot load shared library at {lib_path}: {exc}"
            ) from exc

    def _setup_ctypes(self) -> None:
        """Declare argtypes / restype for every exported C symbol."""
        L = self._lib  # noqa: N806

        # PrsmStart(ed25519Key *C.char, listenPort C.int,
        #           bootstrapAddrs *C.char, udsPath *C.char) → C.int
        L.PrsmStart.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                ctypes.c_char_p, ctypes.c_char_p]
        L.PrsmStart.restype = ctypes.c_int

        # PrsmStop(handle C.int) → C.int
        L.PrsmStop.argtypes = [ctypes.c_int]
        L.PrsmStop.restype = ctypes.c_int

        # PrsmPeerCount(handle C.int) → C.int
        L.PrsmPeerCount.argtypes = [ctypes.c_int]
        L.PrsmPeerCount.restype = ctypes.c_int

        # PrsmConnect(handle C.int, multiaddr *C.char) → *C.char
        L.PrsmConnect.argtypes = [ctypes.c_int, ctypes.c_char_p]
        L.PrsmConnect.restype = ctypes.c_void_p

        # PrsmFree(ptr *C.char) → void
        L.PrsmFree.argtypes = [ctypes.c_void_p]
        L.PrsmFree.restype = None

        # PrsmPublish(handle, topic, data, dataLen) → C.int
        L.PrsmPublish.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                  ctypes.c_char_p, ctypes.c_int]
        L.PrsmPublish.restype = ctypes.c_int

        # PrsmSubscribe(handle, topic) → C.int
        L.PrsmSubscribe.argtypes = [ctypes.c_int, ctypes.c_char_p]
        L.PrsmSubscribe.restype = ctypes.c_int

        # PrsmUnsubscribe(handle, topic) → C.int
        L.PrsmUnsubscribe.argtypes = [ctypes.c_int, ctypes.c_char_p]
        L.PrsmUnsubscribe.restype = ctypes.c_int

        # PrsmSend(handle, peerID, proto, data, dataLen) → C.int
        L.PrsmSend.argtypes = [ctypes.c_int, ctypes.c_char_p,
                               ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        L.PrsmSend.restype = ctypes.c_int

        # PrsmDHTProvide(handle, key) → C.int
        L.PrsmDHTProvide.argtypes = [ctypes.c_int, ctypes.c_char_p]
        L.PrsmDHTProvide.restype = ctypes.c_int

        # PrsmDHTFindProviders(handle, key, limit) → *C.char
        L.PrsmDHTFindProviders.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                           ctypes.c_int]
        L.PrsmDHTFindProviders.restype = ctypes.c_void_p

        # PrsmGetNATStatus(handle) → *C.char
        L.PrsmGetNATStatus.argtypes = [ctypes.c_int]
        L.PrsmGetNATStatus.restype = ctypes.c_void_p

        # PrsmPeerList(handle) → *C.char
        L.PrsmPeerList.argtypes = [ctypes.c_int]
        L.PrsmPeerList.restype = ctypes.c_void_p

    def _read_and_free(self, ptr: Any) -> str:
        """Decode a C string returned by Go, call PrsmFree, return Python str."""
        if not ptr:
            return ""
        try:
            value = ctypes.string_at(ptr).decode("utf-8")
        except Exception:
            value = ""
        self._lib.PrsmFree(ptr)
        return value

    @staticmethod
    def _to_multiaddr(address: str) -> str:
        """Translate common address formats to libp2p multiaddr strings.

        Supported inputs:
        - ``wss://host:port`` or ``ws://host:port`` → ``/ip4/host/tcp/port/ws``
        - ``host:port``                              → ``/ip4/host/udp/port/quic-v1``
        - ``/ip4/...``                               → passthrough
        """
        if address.startswith("/ip4/") or address.startswith("/ip6/"):
            return address

        if address.startswith("wss://") or address.startswith("ws://"):
            # Strip scheme
            without_scheme = address.split("://", 1)[1]
            host, _, port = without_scheme.rpartition(":")
            if not host:
                host = without_scheme
                port = "443" if address.startswith("wss") else "80"
            return f"/ip4/{host}/tcp/{port}/ws"

        # Plain host:port → QUIC
        if ":" in address:
            host, port = address.rsplit(":", 1)
            return f"/ip4/{host}/udp/{port}/quic-v1"

        # Fallback — treat as hostname only
        return f"/ip4/{address}/udp/9001/quic-v1"

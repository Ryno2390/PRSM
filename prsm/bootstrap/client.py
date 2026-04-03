"""
Bootstrap Client

Connects a PRSM node to the bootstrap server for peer discovery.
The bootstrap server speaks a simple JSON protocol (register/heartbeat/get_peers),
while the node's P2P transport uses a different handshake protocol. This client
bridges the two by:

1. Registering the node with the bootstrap server via WSS
2. Receiving the peer list from the bootstrap server
3. Maintaining heartbeat to stay in the peer registry
4. Feeding discovered peer addresses back to the caller
"""

import asyncio
import json
import logging
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import websockets
    import websockets.client
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


@dataclass
class BootstrapPeer:
    """A peer discovered via the bootstrap server."""
    peer_id: str
    address: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    region: Optional[str] = None
    version: Optional[str] = None


class BootstrapClient:
    """Client that connects to a PRSM bootstrap server for peer discovery.

    Usage::

        client = BootstrapClient(
            bootstrap_url="wss://bootstrap1.prsm-network.com:8765",
            node_id="my-node-id",
            port=8000,
            capabilities=["compute", "storage"],
            version="0.22.0",
        )
        peers = await client.connect()
        # peers is a list of BootstrapPeer objects

        # Start background heartbeat (keeps registration alive)
        await client.start_heartbeat()

        # Later...
        await client.disconnect()
    """

    def __init__(
        self,
        bootstrap_url: str,
        node_id: str,
        port: int = 8000,
        capabilities: Optional[List[str]] = None,
        version: str = "0.3.2",
        region: Optional[str] = None,
        connect_timeout: float = 10.0,
        heartbeat_interval: int = 30,
        on_peers_discovered: Optional[Callable[[List[BootstrapPeer]], Any]] = None,
    ):
        self.bootstrap_url = bootstrap_url
        self.node_id = node_id
        self.port = port
        self.capabilities = capabilities or []
        self.version = version
        self.region = region
        self.connect_timeout = connect_timeout
        self.heartbeat_interval = heartbeat_interval
        self.on_peers_discovered = on_peers_discovered

        self._ws: Optional[Any] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._connected = False
        self._registered = False
        self._peers: List[BootstrapPeer] = []
        self._server_time: Optional[str] = None
        self._connect_time: Optional[float] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def discovered_peers(self) -> List[BootstrapPeer]:
        return list(self._peers)

    async def connect(self) -> List[BootstrapPeer]:
        """Connect to the bootstrap server and register this node.

        Returns:
            List of discovered peers from the bootstrap server.

        Raises:
            ImportError: If websockets is not installed.
            ConnectionError: If connection or registration fails.
        """
        if not _WS_AVAILABLE:
            raise ImportError(
                "websockets is required for bootstrap connectivity. "
                "Install with: pip install websockets"
            )

        # Build SSL context for WSS
        ssl_ctx = None
        if self.bootstrap_url.startswith("wss://"):
            ssl_ctx = ssl.create_default_context()
            # Bootstrap servers may use self-signed certs in dev
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        try:
            logger.info(
                "Connecting to bootstrap server: %s", self.bootstrap_url
            )
            kwargs = {"open_timeout": self.connect_timeout}
            if ssl_ctx:
                kwargs["ssl"] = ssl_ctx

            self._ws = await websockets.client.connect(
                self.bootstrap_url, **kwargs
            )
            self._connected = True
            self._connect_time = time.monotonic()
            logger.info("WebSocket connected to bootstrap server")

        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to bootstrap server at "
                f"{self.bootstrap_url}: {e}"
            ) from e

        # Register
        try:
            register_msg = {
                "type": "register",
                "peer_id": self.node_id,
                "port": self.port,
                "capabilities": self.capabilities,
                "version": self.version,
                "region": self.region,
            }
            await self._ws.send(json.dumps(register_msg))
            logger.debug("Sent register message for node %s", self.node_id)

            # Wait for ack
            raw = await asyncio.wait_for(
                self._ws.recv(), timeout=self.connect_timeout
            )
            data = json.loads(raw)

            if data.get("type") == "register_ack":
                self._registered = True
                self._server_time = data.get("server_time")

                # Update heartbeat interval from server
                server_hb = data.get("heartbeat_interval")
                if server_hb and isinstance(server_hb, (int, float)):
                    self.heartbeat_interval = int(server_hb)

                # Parse peer list
                self._peers = []
                for p in data.get("peers", []):
                    self._peers.append(BootstrapPeer(
                        peer_id=p.get("peer_id", ""),
                        address=p.get("address", ""),
                        port=p.get("port", 8000),
                        capabilities=p.get("capabilities", []),
                        region=p.get("region"),
                        version=p.get("version"),
                    ))

                logger.info(
                    "Registered with bootstrap server — peer_id=%s, "
                    "discovered %d peer(s), heartbeat=%ds",
                    self.node_id,
                    len(self._peers),
                    self.heartbeat_interval,
                )

                if self.on_peers_discovered and self._peers:
                    self.on_peers_discovered(self._peers)

                return self._peers

            elif data.get("type") == "error":
                raise ConnectionError(
                    f"Bootstrap registration rejected: {data.get('message')}"
                )
            else:
                raise ConnectionError(
                    f"Unexpected response from bootstrap: {data.get('type')}"
                )

        except asyncio.TimeoutError:
            raise ConnectionError(
                "Timed out waiting for register_ack from bootstrap server"
            )
        except Exception:
            await self.disconnect()
            raise

    async def start_heartbeat(self) -> None:
        """Start background heartbeat to keep the registration alive."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            return  # Already running

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.debug(
            "Heartbeat started (interval=%ds)", self.heartbeat_interval
        )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the bootstrap server."""
        while self._connected and self._ws:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if not self._connected or not self._ws:
                    break

                heartbeat = {
                    "type": "heartbeat",
                    "peer_id": self.node_id,
                    "uptime": time.monotonic() - (self._connect_time or 0),
                }
                await self._ws.send(json.dumps(heartbeat))
                logger.debug("Heartbeat sent to bootstrap server")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Heartbeat failed: %s", e)
                self._connected = False
                break

    async def get_peers(self) -> List[BootstrapPeer]:
        """Request an updated peer list from the bootstrap server."""
        if not self.is_connected or not self._ws:
            raise ConnectionError("Not connected to bootstrap server")

        await self._ws.send(json.dumps({"type": "get_peers"}))
        raw = await asyncio.wait_for(
            self._ws.recv(), timeout=self.connect_timeout
        )
        data = json.loads(raw)

        peers = []
        for p in data.get("peers", []):
            peers.append(BootstrapPeer(
                peer_id=p.get("peer_id", ""),
                address=p.get("address", ""),
                port=p.get("port", 8000),
                capabilities=p.get("capabilities", []),
                region=p.get("region"),
                version=p.get("version"),
            ))

        self._peers = peers
        if self.on_peers_discovered and peers:
            self.on_peers_discovered(peers)

        return peers

    async def disconnect(self) -> None:
        """Cleanly disconnect from the bootstrap server."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.send(json.dumps({
                    "type": "disconnect",
                    "peer_id": self.node_id,
                }))
            except Exception:
                pass

            try:
                await self._ws.close()
            except Exception:
                pass

        self._ws = None
        self._connected = False
        self._registered = False
        logger.info("Disconnected from bootstrap server")

    def get_status(self) -> Dict[str, Any]:
        """Return current bootstrap client status."""
        return {
            "connected": self._connected,
            "registered": self._registered,
            "bootstrap_url": self.bootstrap_url,
            "node_id": self.node_id,
            "discovered_peers": len(self._peers),
            "heartbeat_interval": self.heartbeat_interval,
            "server_time": self._server_time,
            "uptime": (
                time.monotonic() - self._connect_time
                if self._connect_time else None
            ),
        }

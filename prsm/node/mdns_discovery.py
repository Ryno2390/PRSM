"""
mDNS Local Peer Discovery
==========================

Discovers PRSM nodes on the local network when bootstrap servers
are unreachable. Uses UDP multicast for zero-configuration discovery.
"""

import asyncio
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

MDNS_SERVICE_TYPE = "_prsm._tcp"
MDNS_PORT = 5353
BROADCAST_INTERVAL = 30.0  # seconds between announcements
DISCOVERY_TIMEOUT = 10.0  # seconds to wait for responses


@dataclass
class LocalPeer:
    """A PRSM node discovered on the local network."""
    node_id: str
    address: str  # ws://IP:PORT
    display_name: str = ""
    hardware_tier: str = "t1"
    discovered_at: float = field(default_factory=time.time)


class MDNSDiscovery:
    """Discovers PRSM nodes on the local network via UDP broadcast.

    This is a lightweight fallback when bootstrap servers are unreachable.
    Uses a simple UDP broadcast protocol (not full mDNS/Bonjour) for
    maximum compatibility across platforms.
    """

    def __init__(
        self,
        node_id: str,
        p2p_port: int,
        display_name: str = "",
        broadcast_port: int = 19999,
    ):
        self.node_id = node_id
        self.p2p_port = p2p_port
        self.display_name = display_name
        self.broadcast_port = broadcast_port
        self._peers: Dict[str, LocalPeer] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

    @property
    def discovered_peers(self) -> List[LocalPeer]:
        """Return all discovered local peers."""
        return list(self._peers.values())

    @property
    def peer_addresses(self) -> List[str]:
        """Return WebSocket addresses of discovered peers."""
        return [p.address for p in self._peers.values()]

    def start(self) -> None:
        """Start broadcasting and listening for local peers."""
        if self._running:
            return
        self._running = True
        self._tasks.append(asyncio.create_task(self._broadcast_loop()))
        self._tasks.append(asyncio.create_task(self._listen_loop()))
        logger.info(f"mDNS discovery started on port {self.broadcast_port}")

    def stop(self) -> None:
        """Stop discovery."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    async def discover_once(self, timeout: float = DISCOVERY_TIMEOUT) -> List[LocalPeer]:
        """Send a single broadcast and collect responses."""
        self._broadcast_announcement()
        await asyncio.sleep(timeout)
        return self.discovered_peers

    def _broadcast_announcement(self) -> None:
        """Broadcast our presence on the local network."""
        try:
            announcement = json.dumps({
                "type": "prsm_announce",
                "node_id": self.node_id,
                "p2p_port": self.p2p_port,
                "display_name": self.display_name,
                "timestamp": time.time(),
            }).encode()

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(1.0)
            try:
                sock.sendto(announcement, ("<broadcast>", self.broadcast_port))
            finally:
                sock.close()

        except Exception as e:
            logger.debug(f"mDNS broadcast failed: {e}")

    def _handle_announcement(self, data: bytes, addr: tuple) -> None:
        """Process an incoming peer announcement."""
        try:
            msg = json.loads(data.decode())
            if msg.get("type") != "prsm_announce":
                return

            peer_id = msg.get("node_id", "")
            if peer_id == self.node_id:
                return  # Ignore our own announcement

            peer_ip = addr[0]
            peer_port = msg.get("p2p_port", 9001)
            peer_addr = f"ws://{peer_ip}:{peer_port}"

            peer = LocalPeer(
                node_id=peer_id,
                address=peer_addr,
                display_name=msg.get("display_name", ""),
            )
            self._peers[peer_id] = peer
            logger.info(f"Discovered local peer: {peer.display_name or peer_id[:12]} at {peer_addr}")

        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.debug(f"mDNS parse error: {e}")

    async def _broadcast_loop(self) -> None:
        """Periodically broadcast our presence."""
        while self._running:
            self._broadcast_announcement()
            await asyncio.sleep(BROADCAST_INTERVAL)

    async def _listen_loop(self) -> None:
        """Listen for peer announcements via UDP."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except AttributeError:
                pass  # SO_REUSEPORT not available on all platforms
            sock.bind(("", self.broadcast_port))
            sock.setblocking(False)

            logger.debug(f"Listening for local peers on UDP port {self.broadcast_port}")

            while self._running:
                try:
                    data, addr = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: sock.recvfrom(4096)
                    )
                    self._handle_announcement(data, addr)
                except (BlockingIOError, socket.timeout):
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.debug(f"mDNS listen error: {e}")
                    await asyncio.sleep(1.0)

        except Exception as e:
            logger.warning(f"mDNS listener failed to start: {e}")

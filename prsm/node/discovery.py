"""
Peer Discovery
==============

Bootstrap-based peer discovery with gossip propagation.
Nodes connect to bootstrap peers, request their peer lists,
and periodically share their own presence on the network.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from prsm.node.transport import (
    MSG_GOSSIP,
    P2PMessage,
    PeerConnection,
    WebSocketTransport,
)

logger = logging.getLogger(__name__)

# Discovery-specific message subtypes (carried in payload["subtype"])
DISCOVERY_ANNOUNCE = "discovery_announce"
DISCOVERY_PEER_REQUEST = "discovery_peer_request"
DISCOVERY_PEER_RESPONSE = "discovery_peer_response"


@dataclass
class PeerInfo:
    """Lightweight peer descriptor shared during discovery."""
    node_id: str
    address: str
    display_name: str = ""
    roles: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


class PeerDiscovery:
    """Discovers and maintains connections to network peers.

    Strategy:
    1. Connect to bootstrap nodes on startup.
    2. Request their peer lists.
    3. Periodically announce ourselves via gossip.
    4. Maintain a target number of connections.
    """

    def __init__(
        self,
        transport: WebSocketTransport,
        bootstrap_nodes: Optional[List[str]] = None,
        target_peers: int = 8,
        announce_interval: float = 60.0,
        maintenance_interval: float = 30.0,
    ):
        self.transport = transport
        self.bootstrap_nodes = bootstrap_nodes or []
        self.target_peers = target_peers
        self.announce_interval = announce_interval
        self.maintenance_interval = maintenance_interval

        # Known peers (may not be connected)
        self.known_peers: Dict[str, PeerInfo] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Register message handlers
        self.transport.on_message(MSG_GOSSIP, self._handle_gossip)

    async def start(self) -> None:
        """Start discovery: bootstrap then run maintenance loops."""
        self._running = True
        await self.bootstrap()
        self._tasks.append(asyncio.create_task(self._announce_loop()))
        self._tasks.append(asyncio.create_task(self._maintenance_loop()))
        logger.info(f"Discovery started with {len(self.bootstrap_nodes)} bootstrap node(s)")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    async def bootstrap(self) -> int:
        """Connect to bootstrap nodes and request their peer lists."""
        connected = 0
        for address in self.bootstrap_nodes:
            peer = await self.transport.connect_to_peer(address)
            if peer:
                connected += 1
                # Request their peer list
                req = P2PMessage(
                    msg_type=MSG_GOSSIP,
                    sender_id=self.transport.identity.node_id,
                    payload={
                        "subtype": DISCOVERY_PEER_REQUEST,
                        "max_peers": 20,
                    },
                )
                await self.transport.send_to_peer(peer.peer_id, req)

        if connected:
            logger.info(f"Bootstrap: connected to {connected}/{len(self.bootstrap_nodes)} nodes")
        elif self.bootstrap_nodes:
            logger.warning("Bootstrap: could not connect to any bootstrap nodes")
        else:
            logger.info("No bootstrap nodes configured — this node is the first on the network")

        return connected

    async def announce_self(self) -> int:
        """Broadcast our presence to the network."""
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload={
                "subtype": DISCOVERY_ANNOUNCE,
                "address": f"{self.transport.host}:{self.transport.port}",
                "display_name": getattr(self.transport.identity, "display_name", ""),
                "roles": [],
                "peer_count": self.transport.peer_count,
            },
        )
        return await self.transport.gossip(msg, fanout=3)

    async def maintain_connections(self) -> None:
        """Ensure we have enough peer connections, connecting to known peers if needed."""
        current = self.transport.peer_count
        if current >= self.target_peers:
            return

        # Try connecting to known but unconnected peers
        connected_ids = set(self.transport.peers.keys())
        candidates = [
            p for p in self.known_peers.values()
            if p.node_id not in connected_ids
            and p.node_id != self.transport.identity.node_id
        ]
        random.shuffle(candidates)

        needed = self.target_peers - current
        for info in candidates[:needed]:
            peer = await self.transport.connect_to_peer(info.address)
            if peer:
                logger.debug(f"Reconnected to known peer {info.node_id[:8]}...")

    def get_known_peers(self) -> List[PeerInfo]:
        """Return list of all known peers (connected or not)."""
        return list(self.known_peers.values())

    # ── Message handlers ─────────────────────────────────────────

    async def _handle_gossip(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle discovery-related gossip messages."""
        subtype = msg.payload.get("subtype", "")

        if subtype == DISCOVERY_ANNOUNCE:
            await self._handle_announce(msg, peer)
        elif subtype == DISCOVERY_PEER_REQUEST:
            await self._handle_peer_request(msg, peer)
        elif subtype == DISCOVERY_PEER_RESPONSE:
            await self._handle_peer_response(msg, peer)

    async def _handle_announce(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Record a peer announcement."""
        address = msg.payload.get("address", peer.address)
        self.known_peers[msg.sender_id] = PeerInfo(
            node_id=msg.sender_id,
            address=address,
            display_name=msg.payload.get("display_name", ""),
            roles=msg.payload.get("roles", []),
            last_seen=time.time(),
        )
        # Re-gossip if TTL > 0
        if msg.ttl > 1:
            fwd = P2PMessage(
                msg_type=msg.msg_type,
                sender_id=msg.sender_id,
                payload=msg.payload,
                ttl=msg.ttl - 1,
                nonce=msg.nonce,  # same nonce so others dedup
            )
            await self.transport.gossip(fwd, fanout=2)

    async def _handle_peer_request(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Respond with our known peer list."""
        max_peers = msg.payload.get("max_peers", 20)
        peers_data = []
        for info in list(self.known_peers.values())[:max_peers]:
            peers_data.append({
                "node_id": info.node_id,
                "address": info.address,
                "display_name": info.display_name,
                "roles": info.roles,
            })

        # Also include directly connected peers
        for pid, pc in list(self.transport.peers.items())[:max_peers]:
            if pid not in {p["node_id"] for p in peers_data}:
                peers_data.append({
                    "node_id": pid,
                    "address": pc.address,
                    "display_name": pc.display_name,
                    "roles": pc.roles,
                })

        resp = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload={
                "subtype": DISCOVERY_PEER_RESPONSE,
                "peers": peers_data[:max_peers],
            },
        )
        await self.transport.send_to_peer(peer.peer_id, resp)

    async def _handle_peer_response(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Process a peer list response."""
        peers_data = msg.payload.get("peers", [])
        for p in peers_data:
            nid = p.get("node_id", "")
            if nid and nid != self.transport.identity.node_id:
                self.known_peers[nid] = PeerInfo(
                    node_id=nid,
                    address=p.get("address", ""),
                    display_name=p.get("display_name", ""),
                    roles=p.get("roles", []),
                    last_seen=time.time(),
                )
        logger.debug(f"Received {len(peers_data)} peers from {peer.peer_id[:8]}")

    # ── Background loops ─────────────────────────────────────────

    async def _announce_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.announce_interval)
            try:
                await self.announce_self()
            except Exception as e:
                logger.error(f"Announce error: {e}")

    async def _maintenance_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.maintenance_interval)
            try:
                await self.maintain_connections()
                # Prune stale known peers (not seen in 10 minutes)
                cutoff = time.time() - 600
                stale = [nid for nid, p in self.known_peers.items() if p.last_seen < cutoff]
                for nid in stale:
                    del self.known_peers[nid]
            except Exception as e:
                logger.error(f"Maintenance error: {e}")

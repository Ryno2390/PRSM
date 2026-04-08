"""
Peer Discovery
==============

Bootstrap-based peer discovery with gossip propagation.
Nodes connect to bootstrap peers, request their peer lists,
and periodically share their own presence on the network.
"""

import asyncio
import collections
import logging
import random
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

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
DISCOVERY_CAPABILITY_ANNOUNCE = "capability_announce"


@dataclass
class PeerInfo:
    """Lightweight peer descriptor shared during discovery."""
    node_id: str
    address: str
    display_name: str = ""
    roles: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)  # e.g. ["inference", "embedding", "benchmark"]
    supported_backends: List[str] = field(default_factory=list)  # e.g. ["anthropic", "openai", "local"]
    gpu_available: bool = False
    last_seen: float = field(default_factory=time.time)
    last_capability_update: float = field(default_factory=time.time)
    job_success_count: int = 0
    job_failure_count: int = 0
    last_failure_time: float = 0.0
    startup_timestamp: float = 0.0

    @property
    def reliability_score(self) -> float:
        """Compute reliability as success ratio. New peers get benefit of the doubt (1.0)."""
        total = self.job_success_count + self.job_failure_count
        if total == 0:
            return 1.0
        return self.job_success_count / total


def validate_bootstrap_address(address: str) -> Tuple[bool, str]:
    """Validate that a bootstrap address is well-formed.

    Returns (is_valid, reason).  Accepts formats:
      - wss://host:port  or  ws://host:port
      - host:port  (bare host:port, port must be numeric)
      - hostname only (accepted, uses default port)

    Rejects empty strings, whitespace-only, addresses with no parseable
    host, and URLs with unsupported schemes.
    """
    if not address or not address.strip():
        return False, "empty address"

    address = address.strip()

    # URL form
    if "://" in address:
        try:
            parsed = urlparse(address)
        except Exception as exc:
            return False, f"unparseable URL ({exc})"

        if parsed.scheme not in ("ws", "wss"):
            return False, f"unsupported scheme '{parsed.scheme}'"
        if not parsed.hostname:
            return False, "missing hostname"
        return True, ""

    # Bare host:port form
    host, sep, port_str = address.rpartition(":")
    if sep and host:
        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                return False, f"port {port} out of range"
        except ValueError:
            return False, f"non-numeric port '{port_str}'"
        return True, ""

    # Single token (hostname only, no port) — accept
    return True, ""


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
        bootstrap_connect_timeout: float = 5.0,
        bootstrap_retry_attempts: int = 2,
        bootstrap_fallback_enabled: bool = True,
        bootstrap_fallback_nodes: Optional[List[str]] = None,
        bootstrap_validate_addresses: bool = True,
        bootstrap_backoff_base: float = 1.0,
        bootstrap_backoff_max: float = 8.0,
        target_peers: int = 8,
        announce_interval: float = 60.0,
        maintenance_interval: float = 30.0,
        peer_stale_timeout: float = 600.0,
        local_capabilities: Optional[List[str]] = None,
        local_backends: Optional[List[str]] = None,
        local_gpu_available: bool = False,
    ):
        # Default bootstrap node — the live PRSM bootstrap server
        _DEFAULT_BOOTSTRAP = ["wss://bootstrap1.prsm-network.com:8765"]

        self.transport = transport
        self.bootstrap_nodes = bootstrap_nodes if bootstrap_nodes is not None else _DEFAULT_BOOTSTRAP
        self.bootstrap_connect_timeout = max(1.0, float(bootstrap_connect_timeout))
        self.bootstrap_retry_attempts = max(1, int(bootstrap_retry_attempts))
        self.bootstrap_fallback_enabled = bootstrap_fallback_enabled
        self.bootstrap_fallback_nodes = bootstrap_fallback_nodes or []
        self.bootstrap_validate_addresses = bootstrap_validate_addresses
        self.bootstrap_backoff_base = max(0.1, float(bootstrap_backoff_base))
        self.bootstrap_backoff_max = max(self.bootstrap_backoff_base, float(bootstrap_backoff_max))
        self.target_peers = target_peers
        self.announce_interval = announce_interval
        self.maintenance_interval = maintenance_interval
        self.peer_stale_timeout = peer_stale_timeout
        self._local_capabilities = local_capabilities or []
        self._local_backends = local_backends or []
        self._local_gpu_available = local_gpu_available

        # Startup bootstrap status (for first-run observability)
        self.bootstrap_degraded_mode: bool = False
        self.bootstrap_connected_count: int = 0
        self.bootstrap_attempted_nodes: List[str] = []
        self.bootstrap_success_node: Optional[str] = None
        self.bootstrap_failed_nodes: List[str] = []

        # Bootstrap decision telemetry (additive, never alters behavior)
        self._bootstrap_telemetry: Dict[str, Any] = {
            "addresses_validated": 0,
            "addresses_rejected": 0,
            "rejected_reasons": collections.Counter(),
            "fallback_activated": False,
            "fallback_attempted": 0,
            "fallback_succeeded": False,
            "backoff_total_seconds": 0.0,
            "source_policy": "primary_only",
        }

        # Known peers (may not be connected)
        self.known_peers: Dict[str, PeerInfo] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Register message handlers
        self.transport.on_message(MSG_GOSSIP, self._handle_gossip)

    async def start(self) -> None:
        """Start discovery: bootstrap then run maintenance loops."""
        self._running = True
        self._bootstrap_client = None
        await self.bootstrap()

        # If P2P transport bootstrap failed, try the bootstrap client protocol
        if self.bootstrap_degraded_mode and self.bootstrap_nodes:
            connected = await self._try_bootstrap_client()
            if connected:
                self.bootstrap_degraded_mode = False
                self.bootstrap_connected_count = 1

        self._tasks.append(asyncio.create_task(self._announce_loop()))
        self._tasks.append(asyncio.create_task(self._maintenance_loop()))
        if self.bootstrap_degraded_mode:
            logger.warning(
                "Discovery started in DEGRADED local mode: bootstrap unavailable; "
                "peer discovery may be delayed until inbound peers or local announcements arrive"
            )
        else:
            logger.info(f"Discovery started with {len(self.bootstrap_nodes)} bootstrap node(s)")

    async def _try_bootstrap_client(self) -> bool:
        """Fall back to the bootstrap client protocol when P2P handshake fails.

        The bootstrap server speaks a simpler register/heartbeat protocol
        (not the P2P MSG_HANDSHAKE). This method uses BootstrapClient to
        register with the server and discover peers.
        """
        try:
            from prsm.bootstrap.client import BootstrapClient, BootstrapPeer
        except ImportError:
            logger.debug("Bootstrap client not available")
            return False

        for address in self.bootstrap_nodes:
            try:
                logger.info(
                    "Trying bootstrap client protocol for %s", address
                )
                client = BootstrapClient(
                    bootstrap_url=address,
                    node_id=self.transport.identity.node_id,
                    port=getattr(self.transport, 'port', 8000),
                    capabilities=self._local_capabilities,
                    version="0.24.0",
                    connect_timeout=self.bootstrap_connect_timeout,
                )

                peers = await client.connect()
                await client.start_heartbeat()

                self._bootstrap_client = client
                self.bootstrap_success_node = address

                # Feed discovered peers into known_peers
                for bp in peers:
                    if bp.peer_id and bp.peer_id != self.transport.identity.node_id:
                        self.known_peers[bp.peer_id] = PeerInfo(
                            node_id=bp.peer_id,
                            address=f"{bp.address}:{bp.port}",
                            capabilities=bp.capabilities,
                        )

                logger.info(
                    "Bootstrap client connected to %s — "
                    "registered, heartbeat active, %d peer(s) discovered",
                    address, len(peers),
                )
                return True

            except Exception as e:
                logger.debug(
                    "Bootstrap client failed for %s: %s", address, e
                )
                continue

        return False

    async def stop(self) -> None:
        self._running = False

        # Disconnect bootstrap client if active
        if getattr(self, '_bootstrap_client', None):
            try:
                await self._bootstrap_client.disconnect()
            except Exception:
                pass
            self._bootstrap_client = None

        for task in self._tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def _build_bootstrap_candidate_list(self) -> Tuple[List[str], List[str]]:
        """Build ordered candidate list: primary nodes first, then fallback.

        Returns:
            (valid_candidates, rejected_addresses) where rejected_addresses
            contains any addresses that failed validation.
        """
        candidates: List[str] = []
        rejected: List[str] = []

        # Phase 1: configured primary nodes
        for addr in self.bootstrap_nodes:
            if self.bootstrap_validate_addresses:
                ok, reason = validate_bootstrap_address(addr)
                self._bootstrap_telemetry["addresses_validated"] += 1
                if not ok:
                    self._bootstrap_telemetry["addresses_rejected"] += 1
                    self._bootstrap_telemetry["rejected_reasons"][reason] += 1
                    rejected.append(addr)
                    logger.warning(
                        "Bootstrap address rejected (validation): %s — %s",
                        addr, reason,
                    )
                    continue
            candidates.append(addr)

        # Phase 2: fallback nodes (only if feature flag enabled)
        if self.bootstrap_fallback_enabled and self.bootstrap_fallback_nodes:
            self._bootstrap_telemetry["source_policy"] = "primary_then_fallback"
            for addr in self.bootstrap_fallback_nodes:
                if addr in candidates:
                    continue  # Already in primary list, skip duplicate
                if self.bootstrap_validate_addresses:
                    ok, reason = validate_bootstrap_address(addr)
                    self._bootstrap_telemetry["addresses_validated"] += 1
                    if not ok:
                        self._bootstrap_telemetry["addresses_rejected"] += 1
                        self._bootstrap_telemetry["rejected_reasons"][reason] += 1
                        rejected.append(addr)
                        logger.warning(
                            "Fallback bootstrap address rejected (validation): %s — %s",
                            addr, reason,
                        )
                        continue
                candidates.append(addr)

        return candidates, rejected

    async def bootstrap(self) -> int:
        """Connect to bootstrap nodes and request their peer lists.

        Uses source ordering policy: configured nodes first, then trusted
        fallback nodes (when bootstrap_fallback_enabled is True).
        Applies address validation and exponential backoff between retries.
        """
        connected = 0
        self.bootstrap_connected_count = 0
        self.bootstrap_degraded_mode = False
        self.bootstrap_attempted_nodes = []
        self.bootstrap_success_node = None
        self.bootstrap_failed_nodes = []

        # Reset telemetry for this bootstrap attempt
        self._bootstrap_telemetry = {
            "addresses_validated": 0,
            "addresses_rejected": 0,
            "rejected_reasons": collections.Counter(),
            "fallback_activated": False,
            "fallback_attempted": 0,
            "fallback_succeeded": False,
            "backoff_total_seconds": 0.0,
            "source_policy": "primary_only",
        }

        candidates, rejected = self._build_bootstrap_candidate_list()
        primary_count = len([a for a in self.bootstrap_nodes if a in candidates])

        for idx, address in enumerate(candidates):
            is_fallback = idx >= primary_count
            if is_fallback and not self._bootstrap_telemetry["fallback_activated"]:
                self._bootstrap_telemetry["fallback_activated"] = True
                logger.info(
                    "Primary bootstrap nodes exhausted; activating fallback peers"
                )
            if is_fallback:
                self._bootstrap_telemetry["fallback_attempted"] += 1

            self.bootstrap_attempted_nodes.append(address)

            for attempt in range(1, self.bootstrap_retry_attempts + 1):
                try:
                    peer = await asyncio.wait_for(
                        self.transport.connect_to_peer(address),
                        timeout=self.bootstrap_connect_timeout,
                    )
                except asyncio.TimeoutError:
                    peer = None
                    logger.debug(
                        "Bootstrap timeout for %s (attempt %d/%d)",
                        address,
                        attempt,
                        self.bootstrap_retry_attempts,
                    )

                if peer:
                    connected = 1
                    self.bootstrap_success_node = address
                    if is_fallback:
                        self._bootstrap_telemetry["fallback_succeeded"] = True

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
                    break

                # Exponential backoff between retries (not after last attempt)
                if attempt < self.bootstrap_retry_attempts:
                    backoff = min(
                        self.bootstrap_backoff_base * (2 ** (attempt - 1)),
                        self.bootstrap_backoff_max,
                    )
                    self._bootstrap_telemetry["backoff_total_seconds"] += backoff
                    await asyncio.sleep(backoff)

            if connected:
                break

            self.bootstrap_failed_nodes.append(address)

        self.bootstrap_connected_count = connected

        if connected:
            logger.info(
                "Bootstrap success via %s (attempted %d/%d candidates)%s",
                self.bootstrap_success_node,
                len(self.bootstrap_attempted_nodes),
                len(candidates),
                " [fallback]" if self._bootstrap_telemetry.get("fallback_succeeded") else "",
            )
        elif candidates:
            self.bootstrap_degraded_mode = True
            logger.warning(
                "Bootstrap unavailable after %d candidate(s), %d attempt(s) each; "
                "continuing in DEGRADED local mode",
                len(candidates),
                self.bootstrap_retry_attempts,
            )
        else:
            if rejected:
                self.bootstrap_degraded_mode = True
                logger.warning(
                    "All %d bootstrap address(es) rejected as malformed; "
                    "continuing in DEGRADED local mode",
                    len(rejected),
                )
            else:
                logger.info("No bootstrap nodes configured — this node is the first on the network")

        return connected

    def get_bootstrap_status(self) -> Dict[str, object]:
        """Return startup bootstrap state for node/CLI status reporting."""
        return {
            "configured_nodes": list(self.bootstrap_nodes),
            "attempted_nodes": list(self.bootstrap_attempted_nodes),
            "failed_nodes": list(self.bootstrap_failed_nodes),
            "success_node": self.bootstrap_success_node,
            "connected_count": self.bootstrap_connected_count,
            "degraded_mode": self.bootstrap_degraded_mode,
            "retry_attempts": self.bootstrap_retry_attempts,
            "connect_timeout_seconds": self.bootstrap_connect_timeout,
            "fallback_enabled": self.bootstrap_fallback_enabled,
            "fallback_activated": self._bootstrap_telemetry.get("fallback_activated", False),
            "fallback_succeeded": self._bootstrap_telemetry.get("fallback_succeeded", False),
            "addresses_rejected": self._bootstrap_telemetry.get("addresses_rejected", 0),
            "source_policy": self._bootstrap_telemetry.get("source_policy", "primary_only"),
            "bootstrap_client_active": (
                getattr(self, '_bootstrap_client', None) is not None
                and getattr(self._bootstrap_client, 'is_connected', False)
            ),
        }

    def get_bootstrap_telemetry(self) -> Dict[str, Any]:
        """Return a stable copy of bootstrap decision telemetry for observability.

        This data is purely additive and never alters bootstrap behavior.
        """
        return {
            "addresses_validated": int(self._bootstrap_telemetry.get("addresses_validated", 0)),
            "addresses_rejected": int(self._bootstrap_telemetry.get("addresses_rejected", 0)),
            "rejected_reasons": dict(self._bootstrap_telemetry.get("rejected_reasons", {})),
            "fallback_activated": bool(self._bootstrap_telemetry.get("fallback_activated", False)),
            "fallback_attempted": int(self._bootstrap_telemetry.get("fallback_attempted", 0)),
            "fallback_succeeded": bool(self._bootstrap_telemetry.get("fallback_succeeded", False)),
            "backoff_total_seconds": float(self._bootstrap_telemetry.get("backoff_total_seconds", 0.0)),
            "source_policy": str(self._bootstrap_telemetry.get("source_policy", "primary_only")),
        }

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
                "capabilities": self._local_capabilities,
                "supported_backends": self._local_backends,
                "gpu_available": self._local_gpu_available,
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

    def find_peers_by_capability(
        self,
        required: List[str],
        match_all: bool = True,
    ) -> List[PeerInfo]:
        """Find peers that offer the required capabilities.

        Args:
            required: Capability strings to search for.
            match_all: If True, peer must have *all* required capabilities.
                       If False, peer must have *any* of them.

        Returns:
            Matching peers sorted by most-recently-seen first.
        """
        required_lower = {c.lower() for c in required}
        results: List[PeerInfo] = []
        for peer in self.known_peers.values():
            peer_caps = {c.lower() for c in peer.capabilities}
            if match_all:
                if required_lower <= peer_caps:
                    results.append(peer)
            else:
                if required_lower & peer_caps:
                    results.append(peer)
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_capability(self, capability: str) -> List[PeerInfo]:
        """Find peers that have a specific capability.

        Args:
            capability: The capability to search for (e.g., "inference", "embedding").

        Returns:
            List of peers with the specified capability, sorted by most-recently-seen.
        """
        capability_lower = capability.lower()
        results = [
            peer for peer in self.known_peers.values()
            if capability_lower in {c.lower() for c in peer.capabilities}
        ]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_backend(self, backend: str) -> List[PeerInfo]:
        """Find peers that support a specific backend.

        Args:
            backend: The backend to search for (e.g., "anthropic", "openai", "local").

        Returns:
            List of peers with the specified backend support, sorted by most-recently-seen.
        """
        backend_lower = backend.lower()
        results = [
            peer for peer in self.known_peers.values()
            if backend_lower in {b.lower() for b in peer.supported_backends}
        ]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_gpu(self) -> List[PeerInfo]:
        """Find peers that have GPU available.

        Returns:
            List of peers with GPU available, sorted by most-recently-seen.
        """
        results = [peer for peer in self.known_peers.values() if peer.gpu_available]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

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
        elif subtype == DISCOVERY_CAPABILITY_ANNOUNCE:
            await self._handle_capability_announce(msg, peer)

    async def _handle_announce(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Record a peer announcement."""
        address = msg.payload.get("address", peer.address)
        self.known_peers[msg.sender_id] = PeerInfo(
            node_id=msg.sender_id,
            address=address,
            display_name=msg.payload.get("display_name", ""),
            roles=msg.payload.get("roles", []),
            capabilities=msg.payload.get("capabilities", []),
            supported_backends=msg.payload.get("supported_backends", []),
            gpu_available=msg.payload.get("gpu_available", False),
            last_seen=time.time(),
            last_capability_update=time.time(),
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
                "capabilities": info.capabilities,
                "supported_backends": info.supported_backends,
                "gpu_available": info.gpu_available,
            })

        # Also include directly connected peers
        for pid, pc in list(self.transport.peers.items())[:max_peers]:
            if pid not in {p["node_id"] for p in peers_data}:
                peers_data.append({
                    "node_id": pid,
                    "address": pc.address,
                    "display_name": pc.display_name,
                    "roles": pc.roles,
                    "capabilities": getattr(pc, "capabilities", []),
                    "supported_backends": getattr(pc, "supported_backends", []),
                    "gpu_available": getattr(pc, "gpu_available", False),
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
                    capabilities=p.get("capabilities", []),
                    supported_backends=p.get("supported_backends", []),
                    gpu_available=p.get("gpu_available", False),
                    last_seen=time.time(),
                    last_capability_update=time.time(),
                )
        logger.debug(f"Received {len(peers_data)} peers from {peer.peer_id[:8]}")

    async def _handle_capability_announce(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Handle capability announcement from a peer.

        Updates the peer's capability information in the known_peers dict.
        This allows late-joining nodes to receive capability updates.
        """
        node_id = msg.sender_id
        capabilities = msg.payload.get("capabilities", [])
        supported_backends = msg.payload.get("supported_backends", [])
        gpu_available = msg.payload.get("gpu_available", False)

        # Update existing peer info or create new entry
        if node_id in self.known_peers:
            existing = self.known_peers[node_id]
            existing.capabilities = capabilities
            existing.supported_backends = supported_backends
            existing.gpu_available = gpu_available
            existing.last_seen = time.time()
            existing.last_capability_update = time.time()
            logger.debug(
                f"Updated capabilities for peer {node_id[:8]}: "
                f"caps={capabilities}, backends={supported_backends}, gpu={gpu_available}"
            )
        else:
            # Create new peer entry with capability info
            self.known_peers[node_id] = PeerInfo(
                node_id=node_id,
                address=msg.payload.get("address", peer.address),
                display_name=msg.payload.get("display_name", ""),
                roles=msg.payload.get("roles", []),
                capabilities=capabilities,
                supported_backends=supported_backends,
                gpu_available=gpu_available,
                last_seen=time.time(),
                last_capability_update=time.time(),
            )
            logger.debug(
                f"Created new peer entry from capability announce: {node_id[:8]}"
            )

        # Re-gossip if TTL > 0
        if msg.ttl > 1:
            fwd = P2PMessage(
                msg_type=msg.msg_type,
                sender_id=msg.sender_id,
                payload=msg.payload,
                ttl=msg.ttl - 1,
                nonce=msg.nonce,
            )
            await self.transport.gossip(fwd, fanout=2)

    async def announce_capabilities(self) -> int:
        """Broadcast our capabilities to the network.

        This should be called on node startup and when capabilities change.
        Returns the number of peers the announcement was sent to.
        """
        msg = P2PMessage(
            msg_type=MSG_GOSSIP,
            sender_id=self.transport.identity.node_id,
            payload={
                "subtype": DISCOVERY_CAPABILITY_ANNOUNCE,
                "node_id": self.transport.identity.node_id,
                "capabilities": self._local_capabilities,
                "supported_backends": self._local_backends,
                "gpu_available": self._local_gpu_available,
            },
        )
        logger.info(
            f"Announcing capabilities: caps={self._local_capabilities}, "
            f"backends={self._local_backends}, gpu={self._local_gpu_available}"
        )
        return await self.transport.gossip(msg, fanout=3)

    def set_local_capabilities(
        self,
        capabilities: List[str],
        backends: List[str],
        gpu_available: bool = False,
    ) -> None:
        """Set the local node's capabilities.

        Args:
            capabilities: List of capabilities this node offers (e.g., ["inference", "embedding"]).
            backends: List of supported backends (e.g., ["anthropic", "openai", "local"]).
            gpu_available: Whether this node has GPU resources.
        """
        self._local_capabilities = capabilities
        self._local_backends = backends
        self._local_gpu_available = gpu_available
        logger.info(
            f"Set local capabilities: caps={capabilities}, backends={backends}, gpu={gpu_available}"
        )

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
                # Prune stale known peers
                cutoff = time.time() - self.peer_stale_timeout
                stale = [nid for nid, p in self.known_peers.items() if p.last_seen < cutoff]
                for nid in stale:
                    del self.known_peers[nid]
            except Exception as e:
                logger.error(f"Maintenance error: {e}")

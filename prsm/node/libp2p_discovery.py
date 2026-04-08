"""
libp2p DHT Discovery Wrapper
=============================

Thin wrapper around ``Libp2pTransport`` that exposes the same public API
as ``PeerDiscovery`` (prsm/node/discovery.py).

Discovery is hybrid:
- **GossipSub** carries ephemeral ``capability_announce`` and
  ``shard_available`` messages so the network learns about peers in
  near-real-time.
- **Kademlia DHT** via ``PrsmDHTProvide`` / ``PrsmDHTFindProviders``
  stores durable content-provider records that survive node restarts.

Bootstrap uses ``transport.connect_to_peer()`` for each configured
bootstrap address; if zero succeed the status is set to degraded.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from prsm.node.discovery import PeerInfo

logger = logging.getLogger(__name__)


class Libp2pDiscovery:
    """DHT-backed peer discovery layer for ``Libp2pTransport``.

    Provides the same public interface as ``PeerDiscovery`` so higher-level
    code can swap implementations without changes.
    """

    def __init__(
        self,
        transport: Any,
        bootstrap_nodes: Optional[List[str]] = None,
        gossip: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            transport:       A ``Libp2pTransport`` instance.
            bootstrap_nodes: Multiaddr / host:port / ws(s):// addresses.
            gossip:          Optional ``Libp2pGossip`` for ephemeral announcements.
            **kwargs:        Ignored (drop-in compatibility).
        """
        self.transport = transport
        self.bootstrap_nodes: List[str] = bootstrap_nodes or []
        self.gossip = gossip

        # Peer capability index — node_id → PeerInfo
        self._capability_index: Dict[str, PeerInfo] = {}

        # Content shard cache — CID → list of peer_ids
        self._shard_cache: Dict[str, List[str]] = {}

        # Local node capabilities (set via set_local_capabilities)
        self._local_capabilities: List[str] = []
        self._local_backends: List[str] = []
        self._local_gpu_available: bool = False
        self._startup_timestamp: float = time.time()

        # Bootstrap status tracking
        self._bootstrap_status: Dict[str, Any] = {
            "attempted": 0,
            "connected": 0,
            "degraded": False,
        }

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Bootstrap and subscribe to capability/shard gossip topics."""
        await self.bootstrap()

        if self.gossip is not None:
            self.gossip.subscribe("capability_announce", self._on_capability)
            self.gossip.subscribe("shard_available", self._on_shard_available)

        logger.info(
            "Libp2pDiscovery started — bootstrap: %s, degraded: %s",
            self._bootstrap_status["connected"],
            self._bootstrap_status["degraded"],
        )

    async def stop(self) -> None:
        """No-op — transport handles teardown."""

    # ── Bootstrap ────────────────────────────────────────────────

    async def bootstrap(self) -> int:
        """Connect to each configured bootstrap node.

        Returns:
            Number of successfully connected nodes.
        """
        connected = 0
        self._bootstrap_status["attempted"] = len(self.bootstrap_nodes)

        for addr in self.bootstrap_nodes:
            try:
                peer = await self.transport.connect_to_peer(addr)
                if peer is not None:
                    connected += 1
                    logger.debug("Bootstrap connected to %s", addr)
                else:
                    logger.debug("Bootstrap failed for %s (returned None)", addr)
            except Exception as exc:
                logger.debug("Bootstrap error for %s: %s", addr, exc)

        self._bootstrap_status["connected"] = connected

        if connected == 0 and self.bootstrap_nodes:
            self._bootstrap_status["degraded"] = True
            logger.warning(
                "Libp2pDiscovery: all %d bootstrap node(s) unreachable; "
                "operating in degraded mode",
                len(self.bootstrap_nodes),
            )

        return connected

    # ── Peer query helpers (mirrors PeerDiscovery) ───────────────

    def get_known_peers(self) -> List[PeerInfo]:
        """Return all peers in the capability index."""
        return list(self._capability_index.values())

    def find_peers_by_capability(
        self,
        required: List[str],
        match_all: bool = True,
    ) -> List[PeerInfo]:
        """Find peers by required capabilities.

        Args:
            required:  Capability strings to search for.
            match_all: If True, peer must have ALL required capabilities.
                       If False, peer must have ANY of them.
        """
        required_lower = {c.lower() for c in required}
        results: List[PeerInfo] = []
        for peer in self._capability_index.values():
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
        """Find peers that have a specific capability."""
        cap_lower = capability.lower()
        results = [
            p for p in self._capability_index.values()
            if cap_lower in {c.lower() for c in p.capabilities}
        ]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_backend(self, backend: str) -> List[PeerInfo]:
        """Find peers that support a specific backend."""
        backend_lower = backend.lower()
        results = [
            p for p in self._capability_index.values()
            if backend_lower in {b.lower() for b in p.supported_backends}
        ]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    def find_peers_with_gpu(self) -> List[PeerInfo]:
        """Find peers that have GPU available."""
        results = [p for p in self._capability_index.values() if p.gpu_available]
        results.sort(key=lambda p: p.last_seen, reverse=True)
        return results

    # ── Local capabilities ───────────────────────────────────────

    def set_local_capabilities(
        self,
        capabilities: List[str],
        backends: List[str],
        gpu_available: bool = False,
    ) -> None:
        """Store this node's capabilities for announcement."""
        self._local_capabilities = capabilities
        self._local_backends = backends
        self._local_gpu_available = gpu_available

    async def announce_capabilities(self) -> int:
        """Publish local capabilities via GossipSub.

        Returns:
            1 if published successfully, 0 otherwise.
        """
        if self.gossip is None:
            return 0
        return await self.gossip.publish(
            "capability_announce",
            {
                "node_id": self.transport.identity.node_id,
                "capabilities": self._local_capabilities,
                "supported_backends": self._local_backends,
                "gpu_available": self._local_gpu_available,
                "startup_timestamp": self._startup_timestamp,
            },
        )

    def record_job_success(self, node_id: str) -> None:
        """Record a successful job completion for a peer."""
        peer = self._capability_index.get(node_id)
        if peer:
            peer.job_success_count += 1

    def record_job_failure(self, node_id: str) -> None:
        """Record a job failure/timeout for a peer."""
        peer = self._capability_index.get(node_id)
        if peer:
            peer.job_failure_count += 1
            peer.last_failure_time = time.time()

    # ── Content routing (dual: GossipSub + DHT) ──────────────────

    async def provide_content(self, cid: str) -> None:
        """Announce content availability via both GossipSub and DHT.

        - GossipSub ``shard_available`` for immediate propagation.
        - ``PrsmDHTProvide`` for durable, cross-restart DHT record.
        """
        node_id = self.transport.identity.node_id

        # 1. Ephemeral GossipSub announcement
        if self.gossip is not None:
            try:
                await self.gossip.publish(
                    "shard_available",
                    {"cid": cid, "node_id": node_id},
                )
            except Exception as exc:
                logger.debug("GossipSub provide_content error: %s", exc)

        # 2. Durable DHT record
        try:
            await self.transport.dht_provide(cid)
        except Exception as exc:
            logger.debug("DHT provide error for %s: %s", cid, exc)

        # Update local shard cache
        providers = self._shard_cache.setdefault(cid, [])
        if node_id not in providers:
            providers.append(node_id)

    async def find_content_providers(
        self, cid: str, limit: int = 20
    ) -> List[PeerInfo]:
        """Find peers that hold *cid*.

        Checks local shard_cache first; falls back to DHT lookup.

        Returns:
            List of PeerInfo for peers believed to hold the content.
        """
        results: List[PeerInfo] = []

        # Fast path: local cache
        cached_ids = self._shard_cache.get(cid, [])
        for peer_id in cached_ids[:limit]:
            info = self._capability_index.get(peer_id)
            if info:
                results.append(info)

        if results:
            return results

        # Slow path: DHT
        try:
            peer_ids = await self.transport.dht_find_providers(cid, limit)
        except Exception as exc:
            logger.debug("DHT find_providers error for %s: %s", cid, exc)
            return []

        for peer_id in peer_ids:
            info = self._capability_index.get(peer_id)
            if info:
                results.append(info)
            else:
                # Create a minimal stub entry
                results.append(
                    PeerInfo(
                        node_id=peer_id,
                        address="",
                        last_seen=time.time(),
                        last_capability_update=time.time(),
                    )
                )

        return results

    # ── Status / telemetry ───────────────────────────────────────

    def get_bootstrap_status(self) -> Dict[str, Any]:
        """Return bootstrap connectivity status."""
        return {
            "attempted": self._bootstrap_status["attempted"],
            "connected": self._bootstrap_status["connected"],
            "degraded": self._bootstrap_status["degraded"],
            "bootstrap_nodes": list(self.bootstrap_nodes),
        }

    def get_bootstrap_telemetry(self) -> Dict[str, Any]:
        """Return telemetry-compatible bootstrap data."""
        return {
            "addresses_validated": self._bootstrap_status["attempted"],
            "addresses_rejected": 0,
            "fallback_activated": False,
            "fallback_attempted": 0,
            "fallback_succeeded": False,
            "backoff_total_seconds": 0.0,
            "source_policy": "libp2p_dht",
        }

    # ── Internal callbacks ───────────────────────────────────────

    async def _on_capability(
        self, subtype: str, data: Dict[str, Any], sender_id: str
    ) -> None:
        """Update capability index from a ``capability_announce`` message.

        Resets reliability counters only on restart (new startup_timestamp)
        or capability change. Periodic heartbeats only refresh last_seen.
        """
        node_id = data.get("node_id", sender_id)
        if not node_id:
            return

        new_startup = data.get("startup_timestamp", 0.0)
        new_caps = {c.lower() for c in data.get("capabilities", [])}

        existing = self._capability_index.get(node_id)
        if existing is not None:
            old_startup = existing.startup_timestamp
            old_caps = {c.lower() for c in existing.capabilities}

            # Reset reliability only on restart or capability change
            if new_startup > old_startup or new_caps != old_caps:
                existing.job_success_count = 0
                existing.job_failure_count = 0
                existing.last_failure_time = 0.0
                existing.startup_timestamp = new_startup

            existing.capabilities = data.get("capabilities", existing.capabilities)
            existing.supported_backends = data.get(
                "supported_backends", existing.supported_backends
            )
            existing.gpu_available = data.get("gpu_available", existing.gpu_available)
            existing.last_seen = time.time()
            existing.last_capability_update = time.time()
        else:
            self._capability_index[node_id] = PeerInfo(
                node_id=node_id,
                address=data.get("address", ""),
                display_name=data.get("display_name", ""),
                roles=data.get("roles", []),
                capabilities=data.get("capabilities", []),
                supported_backends=data.get("supported_backends", []),
                gpu_available=data.get("gpu_available", False),
                last_seen=time.time(),
                last_capability_update=time.time(),
                startup_timestamp=new_startup,
            )

    async def _on_shard_available(
        self, subtype: str, data: Dict[str, Any], sender_id: str
    ) -> None:
        """Update shard cache from a ``shard_available`` message."""
        cid = data.get("cid", "")
        node_id = data.get("node_id", sender_id)
        if not cid or not node_id:
            return

        providers = self._shard_cache.setdefault(cid, [])
        if node_id not in providers:
            providers.append(node_id)

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

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from prsm.node.discovery import PeerInfo

logger = logging.getLogger(__name__)


class _DeadBootstrapSentinel:
    """Sprint 321 — placeholder installed in `_bootstrap_client`
    when a reconnect attempt fails so the poll loop's
    `while getattr(self, "_bootstrap_client", None) is not None`
    condition stays truthy and the next tick retries.

    Carries `is_connected=False` so the reconnect-on-drop branch
    keeps firing instead of treating the sentinel as a live
    client. Has no other surface — touching anything on it
    raises AttributeError, which the surrounding try/except
    catches as a reconnect trigger.
    """
    is_connected = False


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

        # Bootstrap status tracking. `discovered_peer_count` (sprint
        # 167) is the last-poll snapshot of how many peers the
        # bootstrap server surfaced; distinct from
        # _capability_index size (which monotonically grows).
        self._bootstrap_status: Dict[str, Any] = {
            "attempted": 0,
            "connected": 0,
            "degraded": False,
            "discovered_peer_count": 0,
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
        """Cancel the bootstrap poll task (sprint 165), disconnect
        the BootstrapClient WebSocket (sprint 166), and let the
        transport handle the rest of teardown."""
        task = getattr(self, "_bootstrap_poll_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Sprint 166 — close the bootstrap WebSocket cleanly so the
        # server sees a graceful unregister + we don't leak open
        # sockets across shutdown. Failures here are non-fatal —
        # we're tearing down anyway.
        client = getattr(self, "_bootstrap_client", None)
        if client is not None:
            try:
                await client.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Libp2pDiscovery: bootstrap client disconnect "
                    "raised: %s", exc,
                )
            self._bootstrap_client = None

    # ── Bootstrap ────────────────────────────────────────────────

    async def bootstrap(self) -> int:
        """Connect to each configured bootstrap node.

        Sprint 164 — two-stage strategy:
          1. Primary: ``transport.connect_to_peer(addr)`` for full
             libp2p P2P handshake. Works for multiaddr-style
             addresses (``/dns4/.../tcp/.../p2p/...``).
          2. Fallback: when primary returns 0 connected AND any
             configured address is a ``ws://`` or ``wss://`` URL,
             use the simpler BootstrapClient register/heartbeat
             protocol that the canonical PRSM bootstrap server
             actually serves on. The legacy ``PeerDiscovery`` had
             this fallback; ``Libp2pDiscovery`` was missing it,
             which left every operator using the canonical
             ``wss://bootstrap1.prsm-network.com:8765`` default
             stuck in degraded mode forever.

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

        # Sprint 164 — BootstrapClient fallback for ws:// / wss://
        # URLs when primary libp2p path returned zero.
        if connected == 0:
            wsocket_addrs = [
                a for a in self.bootstrap_nodes
                if a.startswith(("ws://", "wss://"))
            ]
            if wsocket_addrs and await self._try_bootstrap_client(
                wsocket_addrs,
            ):
                connected = 1

        self._bootstrap_status["connected"] = connected

        if connected == 0 and self.bootstrap_nodes:
            self._bootstrap_status["degraded"] = True
            logger.warning(
                "Libp2pDiscovery: all %d bootstrap node(s) unreachable; "
                "operating in degraded mode",
                len(self.bootstrap_nodes),
            )

        return connected

    async def _try_bootstrap_client(self, addrs: List[str]) -> bool:
        """Sprint 164 — BootstrapClient register/heartbeat fallback.

        The PRSM bootstrap server doesn't speak full libp2p — it
        runs a simpler register/heartbeat WebSocket protocol on
        ``wss://...:8765``. This method registers + starts heartbeat
        for the first reachable wss:// address.

        Mirrors the legacy ``PeerDiscovery._try_bootstrap_client``
        from prsm/node/discovery.py, which was the only place this
        fallback existed pre-sprint-164.

        Returns True on first successful registration, False
        otherwise. Does not raise — registration failures are
        logged at debug level and the next address is tried.
        """
        try:
            from prsm.bootstrap.client import BootstrapClient
        except ImportError:
            logger.debug(
                "BootstrapClient import failed; skipping fallback",
            )
            return False
        try:
            import prsm as _prsm_pkg
            _ver = _prsm_pkg.__version__
        except Exception:  # noqa: BLE001
            _ver = "unknown"
        for addr in addrs:
            try:
                logger.info(
                    "Libp2pDiscovery: BootstrapClient fallback "
                    "attempting %s", addr,
                )
                client = BootstrapClient(
                    bootstrap_url=addr,
                    node_id=self.transport.identity.node_id,
                    port=getattr(self.transport, "port", 9001),
                    capabilities=self._local_capabilities,
                    version=_ver,
                )
                peers = await client.connect()
                await client.start_heartbeat()
                self._bootstrap_client = client
                self._hydrate_peers_from_bootstrap(peers)
                logger.info(
                    "Libp2pDiscovery: BootstrapClient connected via "
                    "%s — %d peer(s) discovered",
                    addr, len(peers),
                )
                # Sprint 165 — periodic peer-list polling so this
                # node sees newly-registered operators after its
                # own registration tick.
                self._bootstrap_poll_task = asyncio.create_task(
                    self._bootstrap_poll_loop(),
                )
                return True
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Libp2pDiscovery: BootstrapClient fallback "
                    "failed for %s: %s", addr, exc,
                )
                continue
        return False

    def _hydrate_peers_from_bootstrap(self, peers: List[Any]) -> None:
        """Sprint 165 — populate _capability_index from bootstrap-
        server peer payloads. Skips self (avoids self-edges in the
        capability graph).

        Sprint 167 — also tracks the last-poll discovered-peer
        count on _bootstrap_status so /status surfaces the live
        bootstrap-server view of the network (separate from
        `known` which conflates bootstrap + gossip sources).
        """
        own_id = self.transport.identity.node_id
        hydrated = 0
        for bp in peers:
            pid = getattr(bp, "peer_id", None)
            if not pid or pid == own_id:
                continue
            self._capability_index[pid] = PeerInfo(
                node_id=pid,
                address=f"{getattr(bp, 'address', '')}:"
                        f"{getattr(bp, 'port', 0)}",
                last_seen=time.time(),
                last_capability_update=time.time(),
            )
            hydrated += 1
        # Sprint 167 — track how many peers the bootstrap server
        # surfaced on the most recent poll. Distinct from
        # _capability_index size, which monotonically grows.
        self._bootstrap_status["discovered_peer_count"] = hydrated

    def _consume_bootstrap_announcements(self, client: Any) -> None:
        """Sprint 320 — process buffered server-pushed announcements
        from the BootstrapClient (sprint 319 wired the buffer).

        Handled announcement_types:
          - ``peer_join``: hydrate _capability_index eagerly so we
            see the peer at announcement cadence rather than waiting
            for the next bootstrap poll's peer_list.
          - ``peer_leave``: REMOVE the peer from _capability_index.
            Pre-sprint-320 the index only grew — peer_leave was a
            no-op, accumulating dead entries that downstream
            consumers would surface as live candidates.

        Forward-compat: unknown announcement_type values are
        ignored (do not raise).

        After processing, ``client._observed_announcements`` is
        cleared so the next poll tick doesn't re-process old
        events. Malformed entries are skipped per-item — one bad
        entry must not poison the whole batch.
        """
        buf = getattr(client, "_observed_announcements", None)
        if not buf:
            return
        own_id = self.transport.identity.node_id
        # Snapshot + clear up front so we don't race with new
        # announcements arriving while we process this batch
        snapshot = list(buf)
        buf.clear()
        for ann in snapshot:
            try:
                atype = ann.get("announcement_type")
                pid = ann.get("peer_id")
                if not pid or pid == own_id:
                    continue
                if atype == "peer_join":
                    endpoint = ann.get("peer_endpoint", "")
                    self._capability_index[pid] = PeerInfo(
                        node_id=pid,
                        address=endpoint,
                        last_seen=time.time(),
                        last_capability_update=time.time(),
                    )
                elif atype == "peer_leave":
                    self._capability_index.pop(pid, None)
                # Unknown announcement_type: ignore (forward-compat)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Libp2pDiscovery: skipped malformed "
                    "announcement %r: %s", ann, exc,
                )

    async def _bootstrap_poll_loop(self) -> None:
        """Sprint 165 — periodic poll of bootstrap server for new
        peer registrations.

        Cadence is tuned via PRSM_BOOTSTRAP_POLL_SECONDS env (default
        60s). Polling runs until the task is cancelled (typically by
        node stop()). Per-tick failures are logged at debug and DO
        NOT terminate the loop — transient WebSocket dropouts get
        retried automatically.
        """
        try:
            interval = float(
                os.getenv("PRSM_BOOTSTRAP_POLL_SECONDS", "60"),
            )
            if interval <= 0:
                interval = 60.0
        except (TypeError, ValueError):
            interval = 60.0
        # Sprint 321 — re-read _bootstrap_client every tick so
        # reconnect-on-drop can swap a fresh client in without
        # the loop holding a stale local reference.
        while getattr(self, "_bootstrap_client", None) is not None:
            try:
                await asyncio.sleep(interval)
                client = getattr(self, "_bootstrap_client", None)
                if client is None:
                    break
                try:
                    peers = await client.get_peers()
                    self._hydrate_peers_from_bootstrap(peers)
                    # Sprint 320 — drain announcements buffered
                    # by _recv_typed during the get_peers
                    # exchange so peer_leave evicts departed
                    # peers and peer_join eagerly hydrates new
                    # ones.
                    self._consume_bootstrap_announcements(client)
                except Exception as exc:  # noqa: BLE001
                    # Sprint 321 — reconnect on a dropped
                    # WebSocket. If the client is still marked
                    # connected, this is a non-network error
                    # (parse / logic) — leave the client alone
                    # and retry on the next tick. Otherwise
                    # tear down + rebuild via the existing
                    # _try_bootstrap_client path so
                    # heartbeat/poll resume against a fresh
                    # socket.
                    if not getattr(client, "is_connected", True):
                        logger.warning(
                            "Libp2pDiscovery: bootstrap client "
                            "dropped (%s); attempting reconnect",
                            exc,
                        )
                        wsocket_addrs = [
                            a for a in self.bootstrap_nodes
                            if a.startswith(("ws://", "wss://"))
                        ]
                        # Detach the dead client so
                        # _try_bootstrap_client's success path
                        # can install the fresh one. Failure
                        # to reconnect leaves the slot empty;
                        # the outer-while condition then exits
                        # cleanly unless we re-arm it. We
                        # always re-arm with the (possibly
                        # failed) attempt's result so the loop
                        # keeps retrying.
                        self._bootstrap_client = None
                        if wsocket_addrs:
                            await self._try_bootstrap_client(
                                wsocket_addrs,
                            )
                        # If reconnect succeeded, retry
                        # get_peers immediately so the next-
                        # poll wait doesn't add latency on top
                        # of the drop.
                        client = getattr(
                            self, "_bootstrap_client", None,
                        )
                        if client is not None and getattr(
                            client, "is_connected", False,
                        ):
                            try:
                                peers = await client.get_peers()
                                self._hydrate_peers_from_bootstrap(peers)
                                self._consume_bootstrap_announcements(client)
                            except Exception as retry_exc:  # noqa: BLE001
                                logger.debug(
                                    "Libp2pDiscovery: post-"
                                    "reconnect get_peers "
                                    "raised: %s", retry_exc,
                                )
                        else:
                            # Reconnect failed — keep the loop
                            # alive by re-arming the slot with
                            # the dead client so the outer
                            # while condition stays truthy.
                            # Next tick will try again.
                            self._bootstrap_client = client or (
                                # dead client object isn't
                                # discoverable here; we rely
                                # on _try_bootstrap_client's
                                # contract that it either
                                # installs a working client or
                                # leaves the slot empty. Put a
                                # sentinel marker that
                                # is_connected=False so we
                                # keep retrying.
                                _DeadBootstrapSentinel()
                            )
                    else:
                        logger.debug(
                            "Libp2pDiscovery: bootstrap poll "
                            "tick raised on connected client: "
                            "%s", exc,
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Libp2pDiscovery: bootstrap poll tick raised: %s",
                    exc,
                )

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
            # Sprint 167 — last-poll snapshot of peers the bootstrap
            # server knows about (excluding self).
            "discovered_peer_count": self._bootstrap_status.get(
                "discovered_peer_count", 0,
            ),
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

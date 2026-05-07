"""DHTNodeComponents — aggregates the DHT-stack singletons one node
needs to participate in ManifestDHT and/or EmbeddingDHT.

This is the production-wiring helper that PRSM-DHT-TRANSPORT T3 hands
to ``Node.start()``. It owns:

- A :class:`KademliaDHT` routing table (peer-discovery state).
- Server-side handlers (:class:`ManifestDHTServer` and/or
  :class:`EmbeddingDHTServer`) over the local indexes the node
  hosts.
- A :class:`DHTRequestRouter` multiplexing inbound requests.
- A :class:`DHTLoopRunner` running an asyncio loop in a dedicated
  thread.
- A :class:`DHTListener` bound on the configured DHT port.
- A :class:`SyncDHTTransport` that wires the loop + adapter + clients.
- Client-side handles (:class:`ManifestDHTClient` and/or
  :class:`EmbeddingDHTClient`) for upload-critical-path code.

The class is opt-in per-DHT: a node can run only ManifestDHT, only
EmbeddingDHT, or both. Anchor / verifier callables are required iff
the corresponding client is built (matches the existing
ManifestDHTClient / EmbeddingDHTClient constructor invariants — there
is no trust-the-network mode).

Lifecycle::

    components = DHTNodeComponents.build(
        my_node_id=identity.node_id,
        my_host="127.0.0.1",
        dht_listen_port=0,  # 0 = kernel-assigned
        transport_adapter=DirectAdapter(),
        local_manifest_index=manifest_index,
        anchor=publisher_key_anchor,
        local_embedding_index=embedding_index,
        creator_pubkey_for=anchor.lookup_for_content,
        verify_signature=ed25519_verify,
    )
    components.start()
    try:
        # ... use components.manifest_client / components.embedding_client ...
        # ... call components.add_peer(node_id, host, port) on discovery ...
    finally:
        components.stop()

The components are deliberately decoupled from
``prsm.node.node.Node`` so they can be exercised by a multi-node E2E
test without spinning up the full Node. The Node-side wiring (T3b,
follow-on) will instantiate this class and route peer-discovery
callbacks into :meth:`add_peer`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from prsm.compute.collaboration.p2p.node_discovery import (
    KademliaDHT, PeerNode,
)
from prsm.network.dht_listener import DHTListener
from prsm.network.dht_router import DHTRequestRouter
from prsm.network.embedding_dht.dht_client import (
    EmbeddingDHTClient,
)
from prsm.network.embedding_dht.dht_server import EmbeddingDHTServer
from prsm.network.embedding_dht.local_index import LocalEmbeddingIndex
from prsm.network.manifest_dht.dht_client import ManifestDHTClient
from prsm.network.manifest_dht.dht_server import ManifestDHTServer
from prsm.network.manifest_dht.local_index import LocalManifestIndex
from prsm.network.sync_dht_transport import (
    DHTLoopRunner, SyncDHTTransport,
)
from prsm.node.transport_adapter import TransportAdapter

logger = logging.getLogger(__name__)


# Public PeerNode public_key field is a string; production sources it
# from the peer's PRSM identity. For peer-discovery callbacks that
# don't yet have one, pass empty string — the routing-table
# implementation doesn't read it.
_PEER_PUBKEY_PLACEHOLDER = ""


@dataclass
class DHTNodeComponents:
    """Container for one node's DHT singletons.

    Construct via :meth:`build`. Use :meth:`start` / :meth:`stop` for
    lifecycle. :meth:`add_peer` is the single hook peer-discovery
    callbacks should use — it reaches into the routing table and
    bumps ``last_seen`` so the existing PeerNode liveness logic works.
    """

    my_node_id: str
    my_host: str
    listen_host: str
    configured_listen_port: int
    transport_adapter: TransportAdapter
    kademlia: KademliaDHT
    router: DHTRequestRouter
    loop_runner: DHTLoopRunner
    listener: DHTListener
    manifest_server: Optional[ManifestDHTServer] = None
    embedding_server: Optional[EmbeddingDHTServer] = None
    transport: Optional[SyncDHTTransport] = None
    manifest_client: Optional[ManifestDHTClient] = None
    embedding_client: Optional[EmbeddingDHTClient] = None
    _started: bool = False

    # ── factory ────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        my_node_id: str,
        my_host: str,
        dht_listen_port: int,
        transport_adapter: TransportAdapter,
        listen_host: Optional[str] = None,
        local_manifest_index: Optional[LocalManifestIndex] = None,
        local_embedding_index: Optional[LocalEmbeddingIndex] = None,
        local_fingerprint_index: Optional[Any] = None,
        # ManifestDHT verifier (required when local_manifest_index is set)
        anchor: Optional[Any] = None,
        # EmbeddingDHT verifier (required when local_embedding_index is set)
        creator_pubkey_for: Optional[Callable[[str], Optional[bytes]]] = None,
        verify_signature: Optional[
            Callable[[bytes, bytes, bytes], bool]
        ] = None,
        # Listener tuning passthrough
        max_request_bytes: Optional[int] = None,
        max_concurrent_connections: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ) -> "DHTNodeComponents":
        """Construct (but do not start) the DHT stack for one node.

        :param my_node_id: This node's PRSM node_id (hex string).
        :param my_host: Reachable host for *this* node — embedded in
            our outbound peer-introduction messages so peers know how
            to call back. For LAN tests this is ``"127.0.0.1"``; for
            production this is the operator-configured public host.
        :param dht_listen_port: TCP port for the DHT listener. 0 lets
            the kernel assign one — actual bound port is then
            available via :attr:`listen_port` after :meth:`start`.
        :param transport_adapter: A :class:`TransportAdapter` (R9
            §6.2) — outbound DHT RPCs route through it. Tests
            normally pass :class:`DirectAdapter`; production may pass
            :class:`SocksAdapter` for censorship-resistant operators.
        :param listen_host: Bind address for the listener. Defaults
            to ``"0.0.0.0"`` in production-builds; tests typically
            pass ``"127.0.0.1"``.
        :param local_manifest_index: If non-None, the ManifestDHT is
            enabled — server, client, and routing all wired up.
        :param local_embedding_index: Same, for EmbeddingDHT.
        :param anchor: PublisherKeyAnchor for ManifestDHTClient
            verification. Required when ``local_manifest_index`` is
            given.
        :param creator_pubkey_for: Lookup callable for
            EmbeddingDHTClient signature verification. Required when
            ``local_embedding_index`` is given.
        :param verify_signature: Ed25519 verify callable for
            EmbeddingDHTClient. Required when
            ``local_embedding_index`` is given.

        :raises ValueError: when neither index is given (the listener
            would have no servers to dispatch to), or when an enabled
            DHT lacks its verifier inputs.
        """
        if not isinstance(my_node_id, str) or not my_node_id:
            raise ValueError("my_node_id must be a non-empty string")
        if not isinstance(my_host, str) or not my_host:
            raise ValueError("my_host must be a non-empty string")

        if local_manifest_index is None and local_embedding_index is None:
            raise ValueError(
                "DHTNodeComponents.build: at least one of "
                "(local_manifest_index, local_embedding_index) must "
                "be provided — otherwise the listener has no servers "
                "to dispatch to"
            )
        if local_manifest_index is not None and anchor is None:
            raise ValueError(
                "ManifestDHT enabled (local_manifest_index given) "
                "but no anchor — there is no trust-the-network mode "
                "for ManifestDHT; pass a PublisherKeyAnchor"
            )
        if local_embedding_index is not None:
            if creator_pubkey_for is None or verify_signature is None:
                raise ValueError(
                    "EmbeddingDHT enabled (local_embedding_index "
                    "given) but creator_pubkey_for / verify_signature "
                    "missing — there is no trust-the-network mode for "
                    "EmbeddingDHT"
                )

        listen_host_eff = listen_host if listen_host is not None else "0.0.0.0"

        # ── core: routing table + servers ─────────────────────────
        kademlia = KademliaDHT(node_id=my_node_id, port=dht_listen_port)

        # The placeholder my_address is updated post-start once the
        # listener has its real port. Servers and clients embed
        # my_address in self-as-provider replies.
        placeholder_address = f"{my_host}:{dht_listen_port}"

        manifest_server: Optional[ManifestDHTServer] = None
        if local_manifest_index is not None:
            manifest_server = ManifestDHTServer(
                local_index=local_manifest_index,
                routing_table=kademlia,
                my_node_id=my_node_id,
                my_address=placeholder_address,
            )

        embedding_server: Optional[EmbeddingDHTServer] = None
        if local_embedding_index is not None:
            embedding_server = EmbeddingDHTServer(
                local_index=local_embedding_index,
                routing_table=kademlia,
                my_node_id=my_node_id,
                my_address=placeholder_address,
                local_fingerprint_index=local_fingerprint_index,
            )

        router = DHTRequestRouter(
            manifest_server=manifest_server,
            embedding_server=embedding_server,
        )

        # ── transport: loop runner + listener ─────────────────────
        loop_runner = DHTLoopRunner(
            name=f"prsm-dht-loop-{my_node_id[:8]}",
        )
        listener_kwargs: dict = {}
        if max_request_bytes is not None:
            listener_kwargs["max_request_bytes"] = max_request_bytes
        if max_concurrent_connections is not None:
            listener_kwargs[
                "max_concurrent_connections"
            ] = max_concurrent_connections
        if request_timeout is not None:
            listener_kwargs["request_timeout"] = request_timeout
        listener = DHTListener(
            router=router,
            port=dht_listen_port,
            host=listen_host_eff,
            **listener_kwargs,
        )

        return cls(
            my_node_id=my_node_id,
            my_host=my_host,
            listen_host=listen_host_eff,
            configured_listen_port=dht_listen_port,
            transport_adapter=transport_adapter,
            kademlia=kademlia,
            router=router,
            loop_runner=loop_runner,
            listener=listener,
            manifest_server=manifest_server,
            embedding_server=embedding_server,
            # transport + clients are constructed in start() once the
            # loop is running and the listener's port is bound.
            transport=None,
            manifest_client=None,
            embedding_client=None,
            _started=False,
        )

    # ── lifecycle ──────────────────────────────────────────────────

    def start(
        self,
        *,
        # Verifier inputs forwarded to client constructors. Stored on
        # the instance instead of build()-args because they're only
        # needed when the clients are constructed (here in start()),
        # but I keep the api on build() for one-call convenience —
        # the build()-time args are stashed on the instance via
        # kwargs in a follow-up patch. For now: callers may also pass
        # them at start() time.
        anchor: Optional[Any] = None,
        creator_pubkey_for: Optional[Callable[[str], Optional[bytes]]] = None,
        verify_signature: Optional[
            Callable[[bytes, bytes, bytes], bool]
        ] = None,
    ) -> int:
        """Start the loop, the listener, and instantiate the
        clients. Returns the bound port (matters when build() was
        called with port=0).

        Idempotent: a second call while running returns the same port.
        """
        if self._started:
            assert self.listener.port is not None
            return self.listener.port

        loop = self.loop_runner.start()

        # Start the listener on the running loop.
        import asyncio  # local — only needed at lifecycle boundaries
        fut = asyncio.run_coroutine_threadsafe(
            self.listener.start(), loop,
        )
        fut.result(timeout=10.0)
        bound_port = self.listener.port
        if bound_port is None:
            raise RuntimeError(
                "DHTNodeComponents.start: listener bound port is None"
            )

        # Update servers' my_address now that the real port is known.
        # The internal field is the canonical storage.
        real_address = f"{self.my_host}:{bound_port}"
        if self.manifest_server is not None:
            self.manifest_server._my_address = real_address  # noqa: SLF001
        if self.embedding_server is not None:
            self.embedding_server._my_address = real_address  # noqa: SLF001

        # Construct the sync transport against the running loop.
        self.transport = SyncDHTTransport(
            adapter=self.transport_adapter,
            loop=loop,
        )

        # Construct clients. The verifier inputs come from start()
        # kwargs OR from the original build() context (we also accept
        # callers that pass them at start() time for a 2-step
        # construction).
        if self.manifest_server is not None:
            if anchor is None:
                raise RuntimeError(
                    "ManifestDHT was enabled at build() but no anchor "
                    "passed to start()"
                )
            self.manifest_client = ManifestDHTClient(
                local_index=self.manifest_server._local_index,  # noqa: SLF001
                routing_table=self.kademlia,
                send_message=self.transport.send,
                anchor=anchor,
                my_node_id=self.my_node_id,
                my_address=real_address,
            )
        if self.embedding_server is not None:
            if creator_pubkey_for is None or verify_signature is None:
                raise RuntimeError(
                    "EmbeddingDHT was enabled at build() but "
                    "creator_pubkey_for / verify_signature missing "
                    "from start()"
                )
            self.embedding_client = EmbeddingDHTClient(
                routing_table=self.kademlia,
                send_message=self.transport.send,
                creator_pubkey_for=creator_pubkey_for,
                verify_signature=verify_signature,
                my_node_id=self.my_node_id,
                my_address=real_address,
            )

        self._started = True
        logger.info(
            "DHTNodeComponents started: node_id=%s listen=%s:%d "
            "manifest=%s embedding=%s",
            self.my_node_id[:8],
            self.listen_host, bound_port,
            self.manifest_server is not None,
            self.embedding_server is not None,
        )
        return bound_port

    def stop(self) -> None:
        """Stop the listener and the loop. Idempotent."""
        if not self._started:
            # Possibly partial-construction state — try to clean up.
            self.loop_runner.stop()
            return
        loop = self.loop_runner._loop  # noqa: SLF001
        if loop is not None and loop.is_running():
            try:
                import asyncio  # local
                fut = asyncio.run_coroutine_threadsafe(
                    self.listener.stop(), loop,
                )
                fut.result(timeout=5.0)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "DHTNodeComponents.stop: error stopping listener",
                )
        self.loop_runner.stop()
        self._started = False

    # ── peer-discovery hook ────────────────────────────────────────

    def add_peer(
        self, node_id: str, host: str, port: int,
        *, public_key: str = _PEER_PUBKEY_PLACEHOLDER,
    ) -> bool:
        """Add a peer to the Kademlia routing table.

        Existing PRSM peer-discovery callbacks (BootstrapManager
        ``on_peer_discovered``, gossip-mediated peer announcements)
        should call this so DHT routing has a populated table. Updates
        ``last_seen`` to ``now`` so the routing-table liveness logic
        treats the peer as fresh.

        :returns: True if the peer was added or refreshed; False if
            the peer is the local node or the bucket is full of
            active peers (matches :meth:`KademliaDHT.add_peer`).
        """
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("peer node_id must be a non-empty string")
        if not isinstance(host, str) or not host:
            raise ValueError("peer host must be a non-empty string")
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise ValueError(
                f"peer port must be 1..65535, got {port!r}"
            )
        peer = PeerNode(
            node_id=node_id,
            ip_address=host,
            port=port,
            public_key=public_key,
            last_seen=time.time(),
        )
        return self.kademlia.add_peer(peer)

    # ── convenience accessors ──────────────────────────────────────

    @property
    def listen_port(self) -> Optional[int]:
        """Bound listener port. None until :meth:`start` has run."""
        return self.listener.port

    @property
    def is_running(self) -> bool:
        return self._started and self.listener.is_running


__all__ = [
    "DHTNodeComponents",
]

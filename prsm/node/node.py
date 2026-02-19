"""
PRSM Node Runtime
=================

Main orchestrator that wires together identity, ledger, transport,
discovery, gossip, compute, storage, and the management API into
a single running node.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from prsm.node.config import NodeConfig, NodeRole
from prsm.node.identity import (
    NodeIdentity,
    generate_node_identity,
    load_node_identity,
    save_node_identity,
)
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.dag_ledger import DAGLedger, DAGLedgerAdapter
from prsm.node.transport import WebSocketTransport
from prsm.node.discovery import PeerDiscovery
from prsm.node.gossip import GossipProtocol
from prsm.node.compute_provider import ComputeProvider
from prsm.node.compute_requester import ComputeRequester
from prsm.node.storage_provider import StorageProvider
from prsm.node.content_uploader import ContentUploader
from prsm.node.content_index import ContentIndex
from prsm.node.ledger_sync import LedgerSync
from prsm.node.agent_registry import AgentRegistry
from prsm.node.agent_collaboration import AgentCollaboration

logger = logging.getLogger(__name__)


class PRSMNode:
    """A fully operational PRSM network node.

    Orchestrates all subsystems:
    - Identity (Ed25519 keypair)
    - Local FTNS ledger (SQLite)
    - WebSocket P2P transport
    - Peer discovery (bootstrap + gossip)
    - Gossip protocol
    - Compute provider (accept jobs)
    - Compute requester (submit jobs)
    - Storage provider (IPFS pins)
    - Content uploader (provenance + royalties)
    - Management API (FastAPI)
    """

    def __init__(self, config: Optional[NodeConfig] = None):
        self.config = config or NodeConfig()
        self.config.ensure_dirs()

        # Subsystems (initialized in self.initialize())
        self.identity: Optional[NodeIdentity] = None
        self.ledger: Optional[LocalLedger] = None
        self.transport: Optional[WebSocketTransport] = None
        self.discovery: Optional[PeerDiscovery] = None
        self.gossip: Optional[GossipProtocol] = None
        self.compute_provider: Optional[ComputeProvider] = None
        self.compute_requester: Optional[ComputeRequester] = None
        self.storage_provider: Optional[StorageProvider] = None
        self.content_uploader: Optional[ContentUploader] = None
        self.content_index: Optional[ContentIndex] = None
        self.ledger_sync: Optional[LedgerSync] = None
        self.agent_registry: Optional[AgentRegistry] = None
        self.agent_collaboration: Optional[AgentCollaboration] = None

        self._started = False
        self._start_time: Optional[float] = None
        self._api_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Load or generate identity, initialize all subsystems."""
        # ── Identity ─────────────────────────────────────────────
        self.identity = load_node_identity(self.config.identity_path)
        if self.identity is None:
            self.identity = generate_node_identity(self.config.display_name)
            save_node_identity(self.identity, self.config.identity_path)
            logger.info(f"Generated new node identity: {self.identity.node_id}")
        else:
            logger.info(f"Loaded node identity: {self.identity.node_id}")

        # ── Local Ledger (DAG-based or legacy) ─────────────────────
        if self.config.ledger_type == "dag":
            self.ledger = DAGLedger(str(self.config.ledger_path))
        else:
            self.ledger = LocalLedger(str(self.config.ledger_path))
        await self.ledger.initialize()
        await self.ledger.create_wallet(self.identity.node_id, self.config.display_name)
        await self.ledger.create_wallet("system", "PRSM Network")

        # Issue welcome grant if this is a new wallet
        try:
            await self.ledger.issue_welcome_grant(
                self.identity.node_id, self.config.welcome_grant
            )
            logger.info(f"Welcome grant: {self.config.welcome_grant} FTNS")
        except ValueError:
            pass  # Already received welcome grant

        # ── Transport ────────────────────────────────────────────
        self.transport = WebSocketTransport(
            identity=self.identity,
            host=self.config.listen_host,
            port=self.config.p2p_port,
        )

        # ── Gossip ───────────────────────────────────────────────
        self.gossip = GossipProtocol(
            transport=self.transport,
            fanout=self.config.gossip_fanout,
            default_ttl=self.config.gossip_ttl,
            heartbeat_interval=self.config.heartbeat_interval,
        )

        # ── Discovery ────────────────────────────────────────────
        self.discovery = PeerDiscovery(
            transport=self.transport,
            bootstrap_nodes=self.config.bootstrap_nodes,
        )

        # ── Compute ──────────────────────────────────────────────
        if NodeRole.FULL in self.config.roles or NodeRole.COMPUTE in self.config.roles:
            self.compute_provider = ComputeProvider(
                identity=self.identity,
                transport=self.transport,
                gossip=self.gossip,
                ledger=self.ledger,
                cpu_allocation_pct=self.config.cpu_allocation_pct,
                memory_allocation_pct=self.config.memory_allocation_pct,
            )

        self.compute_requester = ComputeRequester(
            identity=self.identity,
            transport=self.transport,
            gossip=self.gossip,
            ledger=self.ledger,
        )

        # ── Storage ──────────────────────────────────────────────
        if NodeRole.FULL in self.config.roles or NodeRole.STORAGE in self.config.roles:
            self.storage_provider = StorageProvider(
                identity=self.identity,
                gossip=self.gossip,
                ledger=self.ledger,
                ipfs_api_url=self.config.ipfs_api_url,
                pledged_gb=self.config.storage_gb,
            )

        # ── Content Index ─────────────────────────────────────────
        self.content_index = ContentIndex(gossip=self.gossip)

        self.content_uploader = ContentUploader(
            identity=self.identity,
            gossip=self.gossip,
            ledger=self.ledger,
            ipfs_api_url=self.config.ipfs_api_url,
            transport=self.transport,
            content_index=self.content_index,
        )

        # ── Ledger Sync ──────────────────────────────────────────
        self.ledger_sync = LedgerSync(
            identity=self.identity,
            gossip=self.gossip,
            ledger=self.ledger,
            transport=self.transport,
        )

        # ── Agent Registry & Collaboration ────────────────────────
        self.agent_registry = AgentRegistry(
            gossip=self.gossip,
            transport=self.transport,
            node_id=self.identity.node_id,
        )
        self.agent_collaboration = AgentCollaboration(
            gossip=self.gossip,
            node_id=self.identity.node_id,
        )

        # Wire ledger_sync into subsystems for transaction broadcasting
        self.content_uploader.ledger_sync = self.ledger_sync
        if self.compute_provider:
            self.compute_provider.ledger_sync = self.ledger_sync
        self.compute_requester.ledger_sync = self.ledger_sync
        if self.storage_provider:
            self.storage_provider.ledger_sync = self.ledger_sync

        logger.info("Node initialized — all subsystems ready")

    async def start(self) -> None:
        """Start all subsystems concurrently."""
        if self._started:
            return

        await self.transport.start()
        await self.gossip.start()
        await self.discovery.start()

        if self.compute_provider:
            await self.compute_provider.start()
        await self.compute_requester.start()

        if self.storage_provider:
            await self.storage_provider.start()
            self.storage_provider.register_content_handler(self.transport)

        if self.content_index:
            self.content_index.start()
        if self.content_uploader:
            self.content_uploader.start()
        if self.ledger_sync:
            self.ledger_sync.start()
        if self.agent_registry:
            self.agent_registry.start()
        if self.agent_collaboration:
            self.agent_collaboration.start()

        # Start management API in background
        self._api_task = asyncio.create_task(self._run_api())

        self._started = True
        self._start_time = time.time()
        logger.info(
            f"PRSM node started — "
            f"P2P: ws://{self.config.listen_host}:{self.config.p2p_port}, "
            f"API: http://127.0.0.1:{self.config.api_port}"
        )

    async def stop(self) -> None:
        """Gracefully shut down all subsystems."""
        if not self._started:
            return

        logger.info("Shutting down PRSM node...")

        if self._api_task:
            self._api_task.cancel()
            self._api_task = None

        if self.ledger_sync:
            await self.ledger_sync.stop()
        if self.content_uploader:
            await self.content_uploader.close()
        if self.storage_provider:
            await self.storage_provider.stop()
        if self.compute_provider:
            await self.compute_provider.stop()
        if self.compute_requester:
            await self.compute_requester.stop()
        if self.discovery:
            await self.discovery.stop()
        if self.gossip:
            await self.gossip.stop()
        if self.transport:
            await self.transport.stop()
        if self.ledger:
            await self.ledger.close()

        self._started = False
        logger.info("PRSM node stopped")

    async def get_status(self) -> Dict[str, Any]:
        """Comprehensive node status."""
        balance = 0.0
        if self.ledger and self.identity:
            balance = await self.ledger.get_balance(self.identity.node_id)

        uptime = time.time() - self._start_time if self._start_time else 0.0

        status = {
            "node_id": self.identity.node_id if self.identity else None,
            "display_name": self.config.display_name,
            "roles": [r.value for r in self.config.roles],
            "ledger_type": self.config.ledger_type,
            "started": self._started,
            "uptime_seconds": round(uptime, 1),
            "p2p_address": f"ws://{self.config.listen_host}:{self.config.p2p_port}",
            "api_address": f"http://127.0.0.1:{self.config.api_port}",
            "peers": {
                "connected": self.transport.peer_count if self.transport else 0,
                "known": len(self.discovery.known_peers) if self.discovery else 0,
            },
            "ftns_balance": balance,
            "dag_stats": self.ledger.get_stats() if hasattr(self.ledger, 'get_stats') else None,
            "compute": self.compute_provider.get_stats() if self.compute_provider else None,
            "compute_requester": self.compute_requester.get_stats() if self.compute_requester else None,
            "storage": self.storage_provider.get_stats() if self.storage_provider else None,
            "content": self.content_uploader.get_stats() if self.content_uploader else None,
            "content_index": self.content_index.get_stats() if self.content_index else None,
            "ledger_sync": self.ledger_sync.get_stats() if self.ledger_sync else None,
            "agents": self.agent_registry.get_stats() if self.agent_registry else None,
            "collaboration": self.agent_collaboration.get_stats() if self.agent_collaboration else None,
        }
        return status

    async def _run_api(self) -> None:
        """Run the management API server."""
        try:
            from prsm.node.api import create_api_app
            import uvicorn

            app = create_api_app(self)
            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.config.api_port,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            await server.serve()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"API server error: {e}")

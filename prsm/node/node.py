"""
PRSM Node Runtime
=================

Main orchestrator that wires together identity, ledger, transport,
discovery, gossip, compute, storage, and the management API into
a single running node.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from decimal import Decimal

from pathlib import Path

from prsm.node.config import NodeConfig, NodeRole
from prsm.node.identity import (
    NodeIdentity,
    generate_node_identity,
    load_node_identity,
    save_node_identity,
)

try:
    from prsm.data.embeddings.real_embedding_api import RealEmbeddingAPI
    _HAS_EMBEDDING_API = True
except Exception:
    _HAS_EMBEDDING_API = False
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
from prsm.node.content_provider import ContentProvider
from prsm.node.ledger_sync import LedgerSync
from prsm.node.agent_registry import AgentRegistry
from prsm.node.agent_collaboration import AgentCollaboration, BidStrategy
from prsm.economy.tokenomics.staking_manager import StakingManager, StakingConfig, StakeType

logger = logging.getLogger(__name__)


# ── Lightweight NWTN adapters ────────────────────────────────────────
# These shim classes wrap existing node subsystems to match the interface
# expected by NWTNOrchestrator, avoiding a dependency on test mocks.


class _NodeContextAdapter:
    """Minimal context manager for NWTN: tracks nothing, returns defaults."""

    async def get_session_usage(self, session_id: Any) -> Optional[Dict[str, Any]]:
        return None

    async def optimize_context_allocation(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"avg_efficiency": 0.7, "over_allocation_rate": 0.1, "under_allocation_rate": 0.1}

    def record_usage(self, session_id: str, context_used: int, allocated: int) -> None:
        pass


class _NodeFTNSAdapter:
    """Bridges node ledger to the FTNS interface expected by NWTN."""

    def __init__(self, ledger: Any, node_id: str) -> None:
        self._ledger = ledger
        self._node_id = node_id

    async def get_user_balance(self, user_id: str) -> Any:
        balance = await self._ledger.get_balance(self._node_id)

        @dataclass
        class _Balance:
            balance: float
            user_id: str

        return _Balance(balance=balance, user_id=user_id)

    def get_user_balance_sync(self, user_id: str) -> float:
        return 0.0  # sync path not available with async ledger

    async def charge_user(self, user_id: str, amount: float, description: str = "") -> bool:
        try:
            await self._ledger.debit(
                wallet_id=self._node_id,
                amount=amount,
                tx_type=TransactionType.COMPUTE_PAYMENT,
                description=description or "NWTN inference charge",
            )
            return True
        except ValueError:
            return False

    def award_tokens(self, user_id: str, amount: float, description: str = "") -> bool:
        return True  # no-op at node level

    def deduct_tokens(self, user_id: str, amount: float, description: str = "") -> bool:
        return True  # deferred to async charge_user

    async def reward_contribution(self, user_id: str, contribution_type: str, amount: float) -> bool:
        return True


class _StakingFTNSAdapter:
    """Bridges node ledger to the FTNS interface expected by StakingManager."""

    def __init__(self, ledger: Any, node_id: str) -> None:
        self._ledger = ledger
        self._node_id = node_id
        self._locked_balances: Dict[str, Decimal] = {}  # user_id -> locked amount

    async def get_available_balance(self, user_id: str) -> Decimal:
        """Get available (unlocked) balance for a user."""
        balance = await self._ledger.get_balance(user_id)
        locked = self._locked_balances.get(user_id, Decimal('0'))
        return max(Decimal('0'), Decimal(str(balance)) - locked)

    async def lock_tokens(self, user_id: str, amount: Decimal, reason: str = "") -> bool:
        """Lock tokens for staking."""
        try:
            available = await self.get_available_balance(user_id)
            if available < amount:
                raise ValueError(f"Insufficient available balance: {available} < {amount}")
            self._locked_balances[user_id] = self._locked_balances.get(user_id, Decimal('0')) + amount
            return True
        except Exception:
            return False

    async def unlock_tokens(self, user_id: str, amount: Decimal, reason: str = "") -> bool:
        """Unlock tokens when unstaking."""
        try:
            current_locked = self._locked_balances.get(user_id, Decimal('0'))
            self._locked_balances[user_id] = max(Decimal('0'), current_locked - amount)
            return True
        except Exception:
            return False

    async def burn_tokens(self, user_id: str, amount: Decimal, reason: str = "") -> bool:
        """Burn tokens (for slashing)."""
        try:
            await self._ledger.debit(
                wallet_id=user_id,
                amount=float(amount),
                tx_type=TransactionType.PENALTY,
                description=reason or "Slashing penalty",
            )
            # Also reduce locked balance
            current_locked = self._locked_balances.get(user_id, Decimal('0'))
            self._locked_balances[user_id] = max(Decimal('0'), current_locked - amount)
            return True
        except Exception:
            return False

    async def mint_tokens(self, user_id: str, amount: Decimal, reason: str = "") -> bool:
        """Mint tokens (for rewards or appeal refunds)."""
        try:
            await self._ledger.credit(
                wallet_id=user_id,
                amount=float(amount),
                tx_type=TransactionType.REWARD,
                description=reason or "Staking reward",
            )
            return True
        except Exception:
            return False


class _NodeIPFSAdapter:
    """Stub IPFS client — nodes use StorageProvider for actual IPFS ops."""

    def __init__(self, ipfs_api_url: str) -> None:
        self._url = ipfs_api_url

    async def store_model(self, model_data: bytes, metadata: Dict[str, Any]) -> str:
        cid = f"Qm{hashlib.sha256(model_data).hexdigest()[:44]}"
        return cid

    async def retrieve_model(self, cid: str) -> Optional[bytes]:
        return None  # Not implemented at node level


class _NodeModelRegistryAdapter:
    """Stub model registry returning default specialists."""

    async def register_teacher_model(self, model: Any, cid: str) -> bool:
        return True

    async def discover_specialists(self, domain: str) -> list:
        try:
            from prsm.core.models import TeacherModel
            return [
                TeacherModel(name="General Helper", specialization="general", performance_score=0.85),
            ]
        except Exception:
            return []


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

    def __init__(self, config: Optional[NodeConfig] = None) -> None:
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
        self.content_provider: Optional[ContentProvider] = None
        self.ledger_sync: Optional[LedgerSync] = None
        self.agent_registry: Optional[AgentRegistry] = None
        self.agent_collaboration: Optional[AgentCollaboration] = None
        self.staking_manager: Optional[StakingManager] = None

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
            nonce_window=self.config.nonce_window,
            ws_ping_interval=self.config.ws_ping_interval,
            ws_ping_timeout=self.config.ws_ping_timeout,
            handshake_timeout=self.config.handshake_timeout,
            nonce_cleanup_interval=self.config.nonce_cleanup_interval,
        )

        # ── Gossip ───────────────────────────────────────────────
        self.gossip = GossipProtocol(
            transport=self.transport,
            fanout=self.config.gossip_fanout,
            default_ttl=self.config.gossip_ttl,
            heartbeat_interval=self.config.heartbeat_interval,
        )

        # ── Discovery ────────────────────────────────────────────
        # Derive local capabilities from node roles
        local_capabilities: list[str] = []
        for role in self.config.roles:
            if role in (NodeRole.FULL, NodeRole.COMPUTE):
                if "compute" not in local_capabilities:
                    local_capabilities.append("compute")
            if role in (NodeRole.FULL, NodeRole.STORAGE):
                if "storage" not in local_capabilities:
                    local_capabilities.append("storage")

        self.discovery = PeerDiscovery(
            transport=self.transport,
            bootstrap_nodes=self.config.bootstrap_nodes,
            bootstrap_connect_timeout=self.config.bootstrap_connect_timeout,
            bootstrap_retry_attempts=self.config.bootstrap_retry_attempts,
            bootstrap_fallback_enabled=self.config.bootstrap_fallback_enabled,
            bootstrap_fallback_nodes=self.config.bootstrap_fallback_nodes,
            bootstrap_validate_addresses=self.config.bootstrap_validate_addresses,
            bootstrap_backoff_base=self.config.bootstrap_backoff_base,
            bootstrap_backoff_max=self.config.bootstrap_backoff_max,
            target_peers=self.config.target_peers,
            announce_interval=self.config.announce_interval,
            maintenance_interval=self.config.maintenance_interval,
            peer_stale_timeout=self.config.peer_stale_timeout,
            local_capabilities=local_capabilities,
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
                max_concurrent_jobs=self.config.max_concurrent_jobs,
                gpu_allocation_pct=self.config.gpu_allocation_pct,
                config=self.config,
            )
            self.compute_provider.allow_self_compute = self.config.allow_self_compute

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
                config=self.config,
            )
            # Initialize bandwidth limits from config
            if self.config.upload_mbps_limit > 0 or self.config.download_mbps_limit > 0:
                # Update the bandwidth limiter with config values
                # Note: This is synchronous initialization, the async update happens in start()
                self.storage_provider.upload_mbps_limit = self.config.upload_mbps_limit
                self.storage_provider.download_mbps_limit = self.config.download_mbps_limit

        # ── Content Index ─────────────────────────────────────────
        self.content_index = ContentIndex(
            gossip=self.gossip,
            max_indexed_cids=self.config.max_indexed_cids,
            ledger=self.ledger,
        )

        # Optionally attach semantic embedding for near-duplicate detection
        _embedding_fn = None
        if _HAS_EMBEDDING_API:
            try:
                _embed_api = RealEmbeddingAPI()
                _embedding_fn = _embed_api.generate_embedding
            except Exception as _e:
                logger.debug(f"Embedding API unavailable, semantic dedup disabled: {_e}")

        _semantic_index_path = Path.home() / ".prsm" / "semantic_index.json"

        self.content_uploader = ContentUploader(
            identity=self.identity,
            gossip=self.gossip,
            ledger=self.ledger,
            ipfs_api_url=self.config.ipfs_api_url,
            transport=self.transport,
            content_index=self.content_index,
            embedding_fn=_embedding_fn,
            semantic_index_path=_semantic_index_path,
        )

        # ── Content Provider (Cross-Node Retrieval) ───────────────────────
        # Pass bandwidth limiter from storage_provider if available
        _bandwidth_limiter = None
        if self.storage_provider:
            _bandwidth_limiter = self.storage_provider.bandwidth_limiter
        
        self.content_provider = ContentProvider(
            identity=self.identity,
            transport=self.transport,
            gossip=self.gossip,
            ipfs_api_url=self.config.ipfs_api_url,
            content_index=self.content_index,
            bandwidth_limiter=_bandwidth_limiter,
        )

        # ── Ledger Sync ──────────────────────────────────────────
        self.ledger_sync = LedgerSync(
            identity=self.identity,
            gossip=self.gossip,
            ledger=self.ledger,
            transport=self.transport,
            reconciliation_interval=self.config.reconciliation_interval,
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
            ledger=self.ledger,
            bid_strategy=BidStrategy(self.config.bid_strategy),
            bid_window_seconds=self.config.bid_window_seconds,
            min_bids=self.config.min_bids,
            task_timeout=self.config.task_timeout,
            review_timeout=self.config.review_timeout,
            query_timeout=self.config.query_timeout,
            max_completed_records=self.config.max_completed_records,
            cleanup_interval=self.config.collab_cleanup_interval,
        )

        # ── Staking Manager ─────────────────────────────────────────
        # Create FTNS adapter for staking operations
        staking_ftns_adapter = _StakingFTNSAdapter(self.ledger, self.identity.node_id)
        self.staking_manager = StakingManager(
            db_session=None,  # StakingManager uses in-memory storage for now
            ftns_service=staking_ftns_adapter,
            config=StakingConfig(),
        )
        logger.info("Staking manager initialized")

        # Wire ledger_sync and agent_registry into subsystems
        self.content_uploader.ledger_sync = self.ledger_sync
        if self.compute_provider:
            self.compute_provider.ledger_sync = self.ledger_sync
        self.compute_requester.ledger_sync = self.ledger_sync
        if self.storage_provider:
            self.storage_provider.ledger_sync = self.ledger_sync
        self.agent_collaboration.ledger_sync = self.ledger_sync
        self.agent_collaboration.agent_registry = self.agent_registry

        # Wire ledger into gossip for persistence / catch-up
        self.gossip.ledger = self.ledger

        # Best-effort NWTN orchestrator wiring for compute provider
        if self.compute_provider:
            try:
                from prsm.compute.nwtn.orchestrator import NWTNOrchestrator
                from prsm.compute.nwtn.backends import BackendRegistry
                from prsm.compute.nwtn.backends.config import BackendConfig
                
                # Create backend registry from environment for real LLM inference
                backend_config = BackendConfig.from_environment()
                backend_registry = BackendRegistry(backend_config)
                
                orchestrator = NWTNOrchestrator(
                    context_manager=_NodeContextAdapter(),
                    ftns_service=_NodeFTNSAdapter(self.ledger, self.identity.node_id),
                    ipfs_client=_NodeIPFSAdapter(self.config.ipfs_api_url),
                    model_registry=_NodeModelRegistryAdapter(),
                    backend_registry=backend_registry,
                )
                await orchestrator.initialize()
                self.compute_provider.orchestrator = orchestrator
                logger.info("NWTN orchestrator wired to compute provider with backend registry")
            except Exception as e:
                logger.info(f"NWTN orchestrator not available: {e}")

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
        if self.content_provider:
            self.content_provider.start()
        if self.ledger_sync:
            self.ledger_sync.start()
        if self.agent_registry:
            self.agent_registry.start()
        if self.agent_collaboration:
            self.agent_collaboration.start()
            await self.agent_collaboration.load_state()

        # Start management API in background
        self._api_task = asyncio.create_task(self._run_api())

        self._started = True
        self._start_time = time.time()
        bootstrap_status = self.discovery.get_bootstrap_status() if self.discovery else {}
        if bootstrap_status.get("degraded_mode"):
            logger.warning(
                "Node startup in DEGRADED local mode: no bootstrap peers reachable. "
                "Limited features: remote peer discovery and cross-node collaboration may be unavailable "
                "until peers connect or bootstrap targets recover."
            )
        elif bootstrap_status.get("success_node"):
            logger.info(
                "Node startup bootstrap path: connected via %s",
                bootstrap_status.get("success_node"),
            )

        # Emit bootstrap decision telemetry (additive, best-effort)
        if self.discovery:
            bt = self.discovery.get_bootstrap_telemetry()
            if bt.get("fallback_activated"):
                logger.info(
                    "Bootstrap decision: fallback activated, "
                    "fallback_attempted=%d, fallback_succeeded=%s, "
                    "addresses_rejected=%d, source_policy=%s",
                    bt.get("fallback_attempted", 0),
                    bt.get("fallback_succeeded", False),
                    bt.get("addresses_rejected", 0),
                    bt.get("source_policy", "unknown"),
                )
            elif bt.get("addresses_rejected", 0) > 0:
                logger.warning(
                    "Bootstrap decision: %d address(es) rejected during validation",
                    bt["addresses_rejected"],
                )

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

        if self.agent_collaboration:
            await self.agent_collaboration.stop()
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
                "bootstrap": self.discovery.get_bootstrap_status() if self.discovery else {},
                "bootstrap_telemetry": self.discovery.get_bootstrap_telemetry() if self.discovery else {},
            },
            "ftns_balance": balance,
            "dag_stats": (
                await self.ledger.get_stats_async()
                if hasattr(self.ledger, 'get_stats_async')
                else self.ledger.get_stats() if hasattr(self.ledger, 'get_stats') else None
            ),
            "compute": self.compute_provider.get_stats() if self.compute_provider else None,
            "compute_requester": self.compute_requester.get_stats() if self.compute_requester else None,
            "storage": self.storage_provider.get_stats() if self.storage_provider else None,
            "content": self.content_uploader.get_stats() if self.content_uploader else None,
            "content_index": self.content_index.get_stats() if self.content_index else None,
            "content_provider": self.content_provider.get_stats() if self.content_provider else None,
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

"""
PRSM Node Runtime
=================

Main orchestrator that wires together identity, ledger, transport,
discovery, gossip, compute, storage, and the management API into
a single running node.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid as _uuid
from dataclasses import dataclass, field
from enum import Enum as _Enum
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
from prsm.economy.ftns_onchain import OnChainFTNSLedger

# BitTorrent integration
from prsm.core.bittorrent_client import BitTorrentClient, BitTorrentConfig
from prsm.core.bittorrent_manifest import TorrentManifestStore
from prsm.node.bittorrent_provider import BitTorrentProvider, BitTorrentProviderConfig
from prsm.node.bittorrent_requester import BitTorrentRequester, BitTorrentRequesterConfig

logger = logging.getLogger(__name__)


# ── Training Job Status Tracking ────────────────────────────────────────

class TrainingJobStatus(str, _Enum):
    """Status of an async training job for a teacher model."""
    PENDING   = "pending"    # task created, not yet running
    RUNNING   = "running"    # teacher.train() is executing
    COMPLETED = "completed"  # succeeded, result available
    FAILED    = "failed"     # raised an exception
    CANCELLED = "cancelled"  # cancelled via DELETE endpoint


@dataclass
class TrainingJob:
    """Tracks a single async training run for a teacher model."""
    run_id: str
    teacher_id: str
    status: TrainingJobStatus
    started_at: float
    completed_at: Optional[float] = None
    total_epochs: Optional[int] = None   # from training config, known before start
    result: Optional[Any] = None         # TrainingResult on completion
    error: Optional[str] = None
    _task: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)  # asyncio.Task — not serialized

    def to_dict(self) -> Dict[str, Any]:
        """Serializable snapshot — safe for JSON and API responses."""
        d = {
            "run_id":        self.run_id,
            "teacher_id":    self.teacher_id,
            "status":        self.status.value,
            "started_at":    self.started_at,
            "completed_at":  self.completed_at,
            "total_epochs":  self.total_epochs,
            "error":         self.error,
        }
        if self.result is not None:
            # Handle result with to_dict() method or convert to dict directly
            if hasattr(self.result, 'to_dict'):
                d["result"] = self.result.to_dict()
            else:
                d["result"] = self.result
        return d


# ── Lightweight NWTN adapters ────────────────────────────────────────
# These shim classes wrap existing node subsystems to match the interface
# expected by NWTNOrchestrator, avoiding a dependency on test mocks.


class _NodeContextAdapter:
    """Deprecated: use get_context_manager() instead. Retained for reference only."""

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
    """
    Real IPFS client adapter for the PRSM node.

    Wraps the production IPFSClient to provide the store_model/retrieve_model
    interface expected by NWTNOrchestrator. Connection is lazy — the IPFS daemon
    is not required at node startup and is only contacted on the first actual call.

    Fallback behaviour when IPFS is unreachable:
    - store_model: returns a deterministic placeholder CID (same format as a real
      CIDv0) so callers can identify the data, but logs clearly that no bytes
      were persisted on the network.
    - retrieve_model: returns None (same as before, but now logged explicitly).
    """

    def __init__(self, ipfs_api_url: str) -> None:
        self._url = ipfs_api_url
        self._client: Optional[Any] = None
        # Prevent repeated connection attempts after the first failure
        self._connect_attempted: bool = False

    async def _get_client(self) -> Optional[Any]:
        """
        Return the connected IPFSClient, initializing it on first call.

        Returns None if the IPFS daemon is unreachable. Subsequent calls
        return None immediately without retrying (avoids repeated timeouts
        on every store/retrieve call when IPFS is offline).
        """
        if self._client is not None:
            return self._client
        if self._connect_attempted:
            return None  # Already failed once; don't retry on every call

        self._connect_attempted = True
        try:
            from prsm.core.ipfs_client import IPFSClient, IPFSConfig
            config = IPFSConfig(api_url=self._url)
            client = IPFSClient(config)
            await client.initialize()
            if client.connected:
                self._client = client
                logger.info("NodeIPFSAdapter connected to IPFS at %s", self._url)
            else:
                logger.warning(
                    "NodeIPFSAdapter: IPFS daemon not reachable at %s — "
                    "model storage will use placeholder CIDs until IPFS is available",
                    self._url,
                )
        except Exception as exc:
            logger.warning(
                "NodeIPFSAdapter: IPFS connection failed (%s) — "
                "model storage will use placeholder CIDs",
                exc,
            )
        return self._client

    async def store_model(self, model_data: bytes, metadata: Dict[str, Any]) -> str:
        """
        Upload model bytes to IPFS and return the real content CID.

        On success: pins the model bytes and uploads metadata JSON alongside;
        returns the real CIDv0 assigned by the IPFS daemon.

        On IPFS unavailable or upload failure: returns a deterministic
        SHA-256-based placeholder CID so callers can identify the data,
        but logs a clear WARNING that no bytes were persisted on the network.
        The placeholder has the same format as a real CIDv0 but is guaranteed
        not to exist on the IPFS network.
        """
        client = await self._get_client()

        def _placeholder() -> str:
            return f"Qm{hashlib.sha256(model_data).hexdigest()[:44]}"

        if client is None:
            cid = _placeholder()
            logger.warning(
                "NodeIPFSAdapter: IPFS unavailable — model NOT stored on network. "
                "Returning placeholder CID %s",
                cid[:20],
            )
            return cid

        # Upload model bytes
        result = await client.upload_content(
            model_data, filename="model.bin", pin=True
        )
        if not result.success:
            cid = _placeholder()
            logger.error(
                "NodeIPFSAdapter: upload failed (%s) — returning placeholder CID %s",
                result.error,
                cid[:20],
            )
            return cid

        # Upload metadata JSON alongside the model (best-effort; non-fatal)
        if metadata:
            import json as _json
            try:
                meta_bytes = _json.dumps(metadata, default=str).encode()
                await client.upload_content(
                    meta_bytes,
                    filename=f"{result.cid}_metadata.json",
                    pin=True,
                )
            except Exception as exc:
                logger.debug(
                    "NodeIPFSAdapter: metadata upload failed for %s: %s",
                    result.cid[:20],
                    exc,
                )  # Non-fatal

        logger.info(
            "NodeIPFSAdapter: model stored → CID %s (%d bytes)",
            result.cid,
            len(model_data),
        )
        return result.cid

    async def retrieve_model(self, cid: str) -> Optional[bytes]:
        """
        Fetch model bytes from IPFS by CID.

        Returns the raw bytes on success, or None when IPFS is unavailable
        or the CID is not found on the network.
        """
        client = await self._get_client()

        if client is None:
            logger.warning(
                "NodeIPFSAdapter: IPFS unavailable — cannot retrieve CID %s", cid[:20]
            )
            return None

        result = await client.download_content(cid)

        if result.success and result.metadata and "content" in result.metadata:
            content: bytes = result.metadata["content"]
            logger.info(
                "NodeIPFSAdapter: retrieved %d bytes for CID %s",
                len(content),
                cid[:20],
            )
            return content

        logger.warning(
            "NodeIPFSAdapter: content not found or download failed for CID %s — %s",
            cid[:20],
            result.error if result else "unknown error",
        )
        return None


class _NodeModelRegistryAdapter:
    """DB-backed model registry adapter for PRSMNode.

    Persists teacher models to TeacherModelModel and discovers specialists
    via specialization-matched queries ordered by performance score.
    Falls back to empty list when the database is unavailable.
    """

    async def register_teacher_model(self, model: Any, cid: str) -> bool:
        """Persist a teacher model to TeacherModelModel. Idempotent on name+version collision."""
        from prsm.core.database import get_async_session, TeacherModelModel
        from sqlalchemy.exc import IntegrityError
        from uuid import uuid4

        async with get_async_session() as db:
            try:
                row = TeacherModelModel(
                    teacher_id=uuid4(),
                    name=getattr(model, 'name', str(model)),
                    specialization=getattr(model, 'specialization', 'general'),
                    performance_score=float(getattr(model, 'performance_score', 0.0)),
                    ipfs_cid=cid,
                    version=getattr(model, 'version', '1.0.0'),
                    active=True,
                )
                db.add(row)
                await db.commit()
                logger.info(f"Registered teacher model: {row.name} ({row.specialization})")
                return True
            except IntegrityError:
                # Model already registered (name+version UNIQUE constraint) — treat as success
                await db.rollback()
                return True
            except Exception as e:
                await db.rollback()
                logger.warning(f"Failed to register teacher model '{getattr(model, 'name', model)}': {e}")
                return False

    async def discover_specialists(self, domain: str) -> list:
        """Query TeacherModelModel for active specialists matching domain, ordered by performance."""
        from prsm.core.database import get_async_session, TeacherModelModel
        from prsm.core.models import TeacherModel
        from sqlalchemy import select, or_

        async with get_async_session() as db:
            try:
                stmt = (
                    select(TeacherModelModel)
                    .where(TeacherModelModel.active == True)
                    .where(
                        or_(
                            TeacherModelModel.specialization.ilike(f"%{domain}%"),
                            TeacherModelModel.specialization == "general",
                        )
                    )
                    .order_by(TeacherModelModel.performance_score.desc())
                    .limit(10)
                )
                result = await db.execute(stmt)
                rows = result.scalars().all()

                return [
                    TeacherModel(
                        teacher_id=row.teacher_id,
                        name=row.name,
                        specialization=row.specialization,
                        performance_score=row.performance_score or 0.0,
                        ipfs_cid=row.ipfs_cid,
                        version=row.version or "1.0.0",
                    )
                    for row in rows
                ]
            except Exception as e:
                logger.warning(f"discover_specialists query failed for domain '{domain}': {e}")
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
        # On-chain FTNS ledger (Base mainnet)
        self.ftns_ledger: Optional[OnChainFTNSLedger] = None
        
        # BitTorrent components
        self.bt_client: Optional[BitTorrentClient] = None
        self.bt_manifest_store: Optional[TorrentManifestStore] = None
        self.bt_provider: Optional[BitTorrentProvider] = None
        self.bt_requester: Optional[BitTorrentRequester] = None
        self.teacher_registry: Dict[str, Any] = {}  # teacher_id (str) → DistilledTeacher instance
        self.training_jobs: Dict[str, TrainingJob] = {}  # run_id (str UUID) → TrainingJob

        self._started = False
        self._start_time: Optional[float] = None
        self._api_task: Optional[asyncio.Task] = None
        self._ipfs_daemon_proc: Optional[Any] = None  # Track IPFS daemon process if we started it

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
            db_session=None,  # session unused; StakingManager calls get_async_session() internally
            ftns_service=staking_ftns_adapter,
            config=StakingConfig(),
        )
        logger.info("Staking manager initialized")

        # ── BitTorrent Integration ───────────────────────────────────
        # Initialize BitTorrent client
        bt_config = BitTorrentConfig(
            port_range_start=getattr(self.config, 'bt_port_start', 6881),
            port_range_end=getattr(self.config, 'bt_port_end', 6891),
            download_dir=str(Path(self.config.data_dir) / "torrents"),
            dht_enabled=getattr(self.config, 'bt_dht_enabled', True),
        )
        self.bt_client = BitTorrentClient(config=bt_config)
        bt_available = await self.bt_client.initialize()
        if bt_available:
            logger.info("BitTorrent client initialized")

            # Initialize manifest store
            self.bt_manifest_store = TorrentManifestStore(
                database_url=f"sqlite:///{self.config.data_dir}/torrent_manifests.db"
            )
            await self.bt_manifest_store.initialize()

            # Initialize provider (seeder)
            bt_provider_config = BitTorrentProviderConfig(
                max_torrents=getattr(self.config, 'bt_max_torrents', 50),
                data_dir=str(self.config.data_dir / "torrents"),
                seeder_reward_per_gb=getattr(self.config, 'bt_seeder_reward_per_gb', Decimal("0.10")),
            )
            self.bt_provider = BitTorrentProvider(
                identity=self.identity,
                transport=self.transport,
                gossip=self.gossip,
                ledger=self.ledger,
                bt_client=self.bt_client,
                manifest_store=self.bt_manifest_store,
                config=bt_provider_config,
                node_config=self.config,
            )

            # Initialize requester (downloader)
            bt_requester_config = BitTorrentRequesterConfig(
                max_concurrent_downloads=getattr(self.config, 'bt_max_downloads', 10),
                data_dir=str(self.config.data_dir / "torrents"),
                download_cost_per_gb=getattr(self.config, 'bt_download_cost_per_gb', Decimal("0.05")),
            )
            self.bt_requester = BitTorrentRequester(
                identity=self.identity,
                gossip=self.gossip,
                bt_client=self.bt_client,
                manifest_store=self.bt_manifest_store,
                ledger=self.ledger,
                config=bt_requester_config,
            )
            logger.info("BitTorrent provider and requester initialized")
        else:
            logger.info("BitTorrent not available - libtorrent may not be installed")

        # Create FTNS adapter for teacher rewards/charges
        self._ftns_adapter = _NodeFTNSAdapter(self.ledger, self.identity.node_id)

        # ── Payment Escrow & Result Consensus ─────────────────────
        from prsm.node.payment_escrow import PaymentEscrow
        from prsm.node.result_consensus import ResultConsensus
        self._payment_escrow = PaymentEscrow(
            ledger=self.ledger,
            node_id=self.identity.node_id,
        )
        self._result_consensus = ResultConsensus(
            epsilon=0.01,
            timeout_seconds=300.0,
        )

        # ── On-Chain FTNS Ledger (Base mainnet) ────────────────────
        self.ftns_ledger = OnChainFTNSLedger(
            node_id=self.identity.node_id,
        )

        # Wire ledger_sync and agent_registry into subsystems
        self.content_uploader.ledger_sync = self.ledger_sync
        if self.compute_provider:
            self.compute_provider.ledger_sync = self.ledger_sync
            # Wire escrow and consensus into compute provider
            self.compute_provider.escrow = self._payment_escrow
            self.compute_provider.consensus = self._result_consensus
            # Wire on-chain FTNS ledger for real blockchain transfers
            if self.ftns_ledger is not None:
                self._payment_escrow.broadcast_tx = self._on_chain_ftns_transfer
        self.compute_requester.ledger_sync = self.ledger_sync
        if hasattr(self.compute_requester, 'escrow'):
            self.compute_requester.escrow = self._payment_escrow
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
                from prsm.compute.nwtn.context_manager import get_context_manager
                
                # Create backend registry from environment for real LLM inference
                backend_config = BackendConfig.from_environment()
                backend_registry = BackendRegistry(backend_config)
                
                orchestrator = NWTNOrchestrator(
                    context_manager=get_context_manager(),
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

        # Load persisted training run records from disk
        self._load_training_runs()

        # Hydrate content uploader from DB (restores provenance across restarts)
        if self.content_uploader:
            hydrated = await self.content_uploader._hydrate_from_db()
            if hydrated > 0:
                logger.info(f"Restored {hydrated} provenance record(s) from DB")

        logger.info("Node initialized — all subsystems ready")

    async def start(self) -> None:
        """Start all subsystems concurrently."""
        if self._started:
            return

        await self.transport.start()
        await self.gossip.start()
        await self.discovery.start()

        # Ensure IPFS daemon is available (auto-start if possible)
        await self._ensure_ipfs_available()

        # Initialize on-chain FTNS ledger (best-effort)
        if self.ftns_ledger:
            ft_initialized = await self.ftns_ledger.initialize()
            if ft_initialized:
                logger.info("FTNS on-chain ledger connected to Base mainnet")
            else:
                logger.info("FTNS on-chain ledger unavailable — running local mode only")
                self.ftns_ledger = None

        # Seed welcome grant if the node has no balance
        await self._seed_welcome_grant()

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
        if self._payment_escrow:
            self._escrow_cleanup_task = asyncio.create_task(self._payment_escrow.periodic_cleanup())
        if self.agent_registry:
            self.agent_registry.start()
        if self.agent_collaboration:
            self.agent_collaboration.start()
            await self.agent_collaboration.load_state()

        # Start BitTorrent components
        if self.bt_provider:
            await self.bt_provider.start()
        if self.bt_requester:
            await self.bt_requester.start()

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
            f"API: http://127.0.0.1:{self.config.api_port}, "
            f"Dashboard: http://127.0.0.1:{self.config.api_port}/"
        )
        logger.info(
            "Node onboarding UI available",
            url=f"http://127.0.0.1:{self.config.api_port}/onboarding/"
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
        # Stop BitTorrent components
        if self.bt_requester:
            await self.bt_requester.stop()
        if self.bt_provider:
            await self.bt_provider.stop()
        if self.bt_client:
            await self.bt_client.shutdown()
        if self.ledger_sync:
            await self.ledger_sync.stop()
        if hasattr(self, '_escrow_cleanup_task') and self._escrow_cleanup_task:
            self._escrow_cleanup_task.cancel()
            try:
                await self._escrow_cleanup_task
            except asyncio.CancelledError:
                pass

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

        # Terminate IPFS daemon if we started it
        if hasattr(self, '_ipfs_daemon_proc') and self._ipfs_daemon_proc is not None:
            try:
                self._ipfs_daemon_proc.terminate()
                self._ipfs_daemon_proc = None
                logger.info("IPFS daemon terminated")
            except Exception as e:
                logger.warning(f"Failed to terminate IPFS daemon: {e}")

        self._started = False
        logger.info("PRSM node stopped")

    async def _ensure_ipfs_available(self) -> bool:
        """
        Check IPFS daemon availability. Start it automatically if ipfs is on PATH
        and the daemon is not already running.

        Returns True if IPFS is (or becomes) available.
        Returns False if IPFS is unavailable — node continues without it.
        """
        import shutil
        import subprocess

        ipfs_binary = shutil.which("ipfs")

        # 1. Is the daemon already running? (fast check via HTTP)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://127.0.0.1:5001/api/v0/id",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        logger.info("IPFS daemon already running")
                        return True
        except Exception:
            pass  # Daemon not reachable — try to start it

        # 2. ipfs not installed → warn and continue
        if not ipfs_binary:
            logger.warning(
                "IPFS daemon not running and 'ipfs' not found on PATH. "
                "Data storage features will be limited. "
                "Install IPFS: https://docs.ipfs.tech/install/command-line/"
            )
            return False

        # 3. ipfs is installed → start the daemon as a background subprocess
        try:
            logger.info("Starting IPFS daemon automatically...")
            daemon = subprocess.Popen(
                [ipfs_binary, "daemon", "--init"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            # Small delay to let daemon start
            await asyncio.sleep(2)

            # Verify it's running by re-checking the HTTP endpoint
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://127.0.0.1:5001/api/v0/id",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            logger.info("IPFS daemon started successfully (PID %d)", daemon.pid)
                            return True
            except Exception as e:
                logger.debug("IPFS daemon start verification failed: %s", e)
                logger.info("IPFS daemon may still be starting (PID %d)", daemon.pid)
                return True  # Assume it will come up
        except Exception as e:
            logger.warning("Failed to start IPFS daemon: %s", e)
            return False

    async def _seed_welcome_grant(self) -> None:
        """Reconcile wallet_balances from dag_transactions on every startup."""
        try:
            # ── Step 1: Rebuild wallet_balances cache from dag_transactions ──
            if hasattr(self.ledger, "_db"):
                await self.ledger._db.execute(
                    """INSERT OR REPLACE INTO wallet_balances (wallet_id, balance, version, last_updated)
                       SELECT to_wallet, SUM(amount), 1, MAX(timestamp)
                       FROM dag_transactions
                       WHERE to_wallet IS NOT NULL
                       GROUP BY to_wallet"""
                )
                await self.ledger._db.commit()
                # Reset the in-memory version cache so it matches the DB
                if hasattr(self.ledger, "_balance_version_cache"):
                    self.ledger._balance_version_cache.clear()
                    cursor = await self.ledger._db.execute(
                        "SELECT wallet_id, version FROM wallet_balances"
                    )
                    async for row in cursor:
                        self.ledger._balance_version_cache[row[0]] = row[1]

            # ── Step 2: Check balance and grant if needed ──
            balance = await self.ledger.get_balance(self.identity.node_id)
            if balance <= 0:
                await self.ledger.credit(
                    wallet_id=self.identity.node_id,
                    amount=100.0,
                    tx_type=TransactionType.WELCOME_GRANT,
                    description="Welcome grant for new node",
                )
                # Also prime wallet_balances for the fresh grant
                if hasattr(self.ledger, "_db"):
                    await self.ledger._db.execute(
                        """INSERT INTO wallet_balances (wallet_id, balance, version, last_updated)
                           VALUES (?, ?, 1, ?)
                           ON CONFLICT(wallet_id) DO UPDATE SET balance = excluded.balance""",
                        (self.identity.node_id, 100.0, time.time()),
                    )
                    await self.ledger._db.commit()
                    if hasattr(self.ledger, "_balance_version_cache"):
                        self.ledger._balance_version_cache[self.identity.node_id] = 1
                logger.info(f"Seeded welcome grant: 100 FTNS to {self.identity.node_id[:12]}...")
            else:
                logger.debug(f"Node already has balance: {balance:.6f}")
        except Exception as e:
            logger.warning(f"Welcome-grant reconciliation failed: {e}")

    # ── On-Chain FTNS Transfer Handler ────────────────────────
    async def _on_chain_ftns_transfer(self, transaction) -> None:
        """Broadcast a transaction to the real FTNS contract on Base.

        Called by PaymentEscrow when it locks, releases, or refunds FTNS.
        If the on-chain ledger is not connected, only logs a warning.
        """
        if not self.ftns_ledger or not self.ftns_ledger._is_initialized:
            logger.debug(
                "FTNS on-chain ledger not available — "
                "skipping blockchain broadcast for escrow tx"
            )
            return

        if not hasattr(transaction, "from_wallet") or not hasattr(transaction, "to_wallet"):
            return

        # Skip transfers to/from internal escrow wallets
        to_addr = transaction.to_wallet
        from_addr = transaction.from_wallet

        # Map PRSM wallet IDs to actual Ethereum addresses
        # The wallet ID IS the address for wallets created from key pairs,
        # or it's a named wallet that needs mapping
        if to_addr.startswith("0x") and len(to_addr) >= 40:
            target_address = to_addr
        else:
            # Named wallet — use the default contract address
            # or skip this transfer
            logger.debug(
                f"Skipping on-chain FTNS transfer for named wallet: {to_addr[:20]}…"
            )
            return

        amount = float(transaction.amount) if hasattr(transaction, "amount") else 0
        if amount <= 0:
            return

        try:
            tx_record = await self.ftns_ledger.transfer(
                job_id=transaction.tx_id if hasattr(transaction, "tx_id") else "",
                to_address=target_address,
                amount_ftns=amount,
            )
            if tx_record and tx_record.status == "confirmed":
                logger.info(
                    f"FTNS on-chain: {amount:.6f} confirmed "
                    f"(tx: {tx_record.tx_hash[:16]}…)"
                )
        except Exception as e:
            logger.error(f"FTNS on-chain transfer failed: {e}")

        except Exception as e:
            logger.error(f"Failed to start IPFS daemon: {e}")
            return False

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
            "dag_stats": {
                "note": "DAG ledger in async mode",
                "mode": "dag" if hasattr(self.ledger, '_dag') else "sql"
            },
            "compute": self.compute_provider.get_stats() if self.compute_provider else None,
            "compute_requester": self.compute_requester.get_stats() if self.compute_requester else None,
            "storage": self.storage_provider.get_stats() if self.storage_provider else None,
            "content": self.content_uploader.get_stats() if self.content_uploader else None,
            "content_index": self.content_index.get_stats() if self.content_index else None,
            "content_provider": self.content_provider.get_stats() if self.content_provider else None,
            "ledger_sync": self.ledger_sync.get_stats() if self.ledger_sync else None,
            "escrow": self._payment_escrow.get_stats() if hasattr(self, '_payment_escrow') and self._payment_escrow else None,
            "consensus": self._result_consensus.get_stats() if hasattr(self, '_result_consensus') and self._result_consensus else None,
            "ftns_onchain": (
                self.ftns_ledger.get_summary()
                if self.ftns_ledger and self.ftns_ledger._is_initialized
                else None
            ),
            "agents": self.agent_registry.get_stats() if self.agent_registry else None,
            "collaboration": self.agent_collaboration.get_stats() if self.agent_collaboration else None,
            "bittorrent": {
                "available": self.bt_client.available if self.bt_client else False,
                "provider": self.bt_provider.get_stats() if self.bt_provider else None,
                "requester": self.bt_requester.get_stats() if self.bt_requester else None,
            },
        }
        return status

    def _save_teacher_registry(self) -> None:
        """Persist teacher metadata (not model weights) across restarts."""
        registry_path = Path(self.config.data_dir) / "teachers.json"
        data = {}
        for tid, t in self.teacher_registry.items():
            # Get domain from teacher_model or fall back to specialization
            domain = getattr(t.teacher_model, "domain", None)
            if domain is None:
                domain = t.teacher_model.specialization
            
            data[tid] = {
                "name": t.teacher_model.name,
                "specialization": t.teacher_model.specialization,
                "domain": domain,
                "model_type": t.teacher_model.model_type.value,
                "created_at": getattr(t, "_created_at", time.time()),
            }
        registry_path.write_text(json.dumps(data, indent=2))

    def _load_teacher_registry_meta(self) -> Dict[str, Any]:
        """Load teacher metadata for display (instances are recreated on demand)."""
        registry_path = Path(self.config.data_dir) / "teachers.json"
        if registry_path.exists():
            return json.loads(registry_path.read_text())
        return {}

    def _save_training_runs(self) -> None:
        """Persist completed/failed run metadata for display after restart."""
        path = Path(self.config.data_dir) / "training_runs.json"
        # Only persist terminal states — pending/running don't survive restart
        terminal = {
            run_id: job.to_dict()
            for run_id, job in self.training_jobs.items()
            if job.status in (TrainingJobStatus.COMPLETED,
                              TrainingJobStatus.FAILED,
                              TrainingJobStatus.CANCELLED)
        }
        path.write_text(json.dumps(terminal, indent=2))

    def _load_training_runs(self) -> None:
        """Restore terminal training run records from disk on startup."""
        path = Path(self.config.data_dir) / "training_runs.json"
        if not path.exists():
            return
        data = json.loads(path.read_text())
        for run_id, d in data.items():
            job = TrainingJob(
                run_id=run_id,
                teacher_id=d["teacher_id"],
                status=TrainingJobStatus(d["status"]),
                started_at=d["started_at"],
                completed_at=d.get("completed_at"),
                total_epochs=d.get("total_epochs"),
                error=d.get("error"),
            )
            if "result" in d:
                # Re-attach result dict as a plain dict (no need to reconstruct dataclass)
                job.result = d["result"]
            self.training_jobs[run_id] = job

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

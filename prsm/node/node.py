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

from contextlib import suppress
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
from prsm.node.content_economy import ContentEconomy, RoyaltyModel

# BitTorrent integration
from prsm.core.bittorrent_client import BitTorrentClient, BitTorrentConfig
from prsm.core.bittorrent_manifest import TorrentManifestStore
from prsm.node.bittorrent_provider import BitTorrentProvider, BitTorrentProviderConfig
from prsm.node.bittorrent_requester import BitTorrentRequester, BitTorrentRequesterConfig

logger = logging.getLogger(__name__)


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
        # Content economy manager (Phase 4)
        self.content_economy: Optional[ContentEconomy] = None
        
        # BitTorrent components
        self.bt_client: Optional[BitTorrentClient] = None
        self.bt_manifest_store: Optional[TorrentManifestStore] = None
        self.bt_provider: Optional[BitTorrentProvider] = None
        self.bt_requester: Optional[BitTorrentRequester] = None

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
            self.ledger = DAGLedger(
                str(self.config.ledger_path),
                verify_signatures=False,
            )
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

        # ── Transport / Gossip / Discovery ───────────────────────
        # Derive local capabilities from node roles (used by both backends)
        local_capabilities: list[str] = []
        for role in self.config.roles:
            if role in (NodeRole.FULL, NodeRole.COMPUTE):
                if "compute" not in local_capabilities:
                    local_capabilities.append("compute")
            if role in (NodeRole.FULL, NodeRole.STORAGE):
                if "storage" not in local_capabilities:
                    local_capabilities.append("storage")
        self._local_capabilities = local_capabilities

        if self.config.transport_backend == "libp2p":
            from prsm.node.libp2p_transport import Libp2pTransport
            from prsm.node.libp2p_gossip import Libp2pGossip
            from prsm.node.libp2p_discovery import Libp2pDiscovery

            self.transport = Libp2pTransport(
                identity=self.identity,
                host=self.config.listen_host,
                port=self.config.p2p_port,
                library_path=self.config.libp2p_library_path,
            )
            self.gossip = Libp2pGossip(transport=self.transport)
            self.discovery = Libp2pDiscovery(
                transport=self.transport,
                bootstrap_nodes=self.config.bootstrap_nodes,
                gossip=self.gossip,
            )
            logger.info("Using libp2p transport backend")
        else:
            # ── WebSocket transport (fallback) ────────────────────
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

            self.gossip = GossipProtocol(
                transport=self.transport,
                fanout=self.config.gossip_fanout,
                default_ttl=self.config.gossip_ttl,
                heartbeat_interval=self.config.heartbeat_interval,
            )

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
            logger.info("Using WebSocket transport backend (fallback)")

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
            discovery=self.discovery,
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
                transport=self.transport,
                discovery=self.discovery,
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

        # ── Mobile Agent Dispatch (Ring 2) ────────────────────────────
        try:
            from prsm.compute.agents.dispatcher import AgentDispatcher
            from prsm.compute.agents.executor import AgentExecutor

            self.agent_dispatcher = AgentDispatcher(
                identity=self.identity,
                gossip=self.gossip,
                transport=self.transport,
                escrow=self._payment_escrow,
            )

            self.agent_executor = AgentExecutor(
                identity=self.identity,
                gossip=self.gossip,
            )
            logger.info("Mobile agent dispatch (Ring 2) initialized")
        except ImportError:
            self.agent_dispatcher = None
            self.agent_executor = None
            logger.debug("Mobile agent dispatch not available")

        # ── Swarm Compute (Ring 3) ────────────────────────────────────
        try:
            from prsm.compute.swarm.coordinator import SwarmCoordinator

            self.swarm_coordinator = SwarmCoordinator(
                dispatcher=self.agent_dispatcher,
                result_consensus=getattr(self, '_result_consensus', None),
            )
            logger.info("Swarm compute (Ring 3) initialized")
        except (ImportError, AttributeError):
            self.swarm_coordinator = None
            logger.debug("Swarm compute not available")

        # ── Economy Engine (Ring 4) ───────────────────────────────────
        try:
            from prsm.economy.pricing.engine import PricingEngine
            from prsm.economy.prosumer import ProsumerManager

            self.pricing_engine = PricingEngine()
            self.prosumer_manager = ProsumerManager(
                node_id=self.identity.node_id,
                ledger=self.ledger,
            )

            from prsm.economy.pricing.revenue_split import RevenueSplitEngine
            from prsm.economy.pricing.data_listing import DataListingManager
            from prsm.economy.pricing.spot_arbitrage import SpotArbitrage

            self.revenue_split = RevenueSplitEngine()
            self.data_listing_manager = DataListingManager()
            self.spot_arbitrage = SpotArbitrage(pricing_engine=self.pricing_engine)

            logger.info("Economy engine (Ring 4) initialized")
        except ImportError:
            self.pricing_engine = None
            self.prosumer_manager = None
            self.revenue_split = None
            self.data_listing_manager = None
            self.spot_arbitrage = None
            logger.debug("Economy engine not available")

        # ── On-Chain FTNS Ledger (Base mainnet) ────────────────────
        self.ftns_ledger = OnChainFTNSLedger(
            node_id=self.identity.node_id,
        )
        
        # ── Content Economy (Phase 4) ──────────────────────────────────────
        # Determine royalty model from config
        royalty_model = RoyaltyModel.PHASE4
        if getattr(self.config, 'royalty_model', 'phase4') == 'legacy':
            royalty_model = RoyaltyModel.LEGACY
        
        # Initialize vector store backend (optional)
        _vector_store = None
        vector_backend = getattr(self.config, 'vector_backend', 'memory')
        if vector_backend and vector_backend != 'disabled':
            try:
                from prsm.node.vector_store_backend import create_vector_store
                _vector_store = create_vector_store(
                    backend=vector_backend,
                    postgres_host=getattr(self.config, 'postgres_host', 'localhost'),
                    postgres_port=getattr(self.config, 'postgres_port', 5432),
                    postgres_database=getattr(self.config, 'postgres_database', 'prsm'),
                    postgres_user=getattr(self.config, 'postgres_user', 'prsm'),
                    postgres_password=getattr(self.config, 'postgres_password', ''),
                )
                await _vector_store.initialize()
                logger.info(f"Vector store initialized: {vector_backend}")
            except Exception as e:
                logger.warning(f"Vector store initialization failed: {e}")
                _vector_store = None
        
        self.content_economy = ContentEconomy(
            identity=self.identity,
            ledger=self.ledger,
            gossip=self.gossip,
            content_index=self.content_index,
            ftns_ledger=self.ftns_ledger,
            royalty_model=royalty_model,
            min_replicas=getattr(self.config, 'min_replicas', 3),
            vector_store=_vector_store,
            embedding_fn=_embedding_fn if '_embedding_fn' in dir() else None,
        )
        
        # Register content economy with API routes
        try:
            from prsm.api.content_economy_routes import set_content_economy
            set_content_economy(self.content_economy)
        except ImportError:
            pass  # API routes not available
        
        # Wire ContentEconomy to StorageProvider for replication tracking (Phase 4)
        if self.storage_provider:
            self.storage_provider.content_economy = self.content_economy
        
        # Initialize multi-party escrow for batch settlements (Phase 4)
        from prsm.node.multi_party_escrow import MultiPartyEscrow, EscrowConfig
        self._mp_escrow = MultiPartyEscrow(
            ftns_ledger=self.ftns_ledger,
            config=EscrowConfig(
                min_batch_size=getattr(self.config, 'escrow_min_batch_size', 5),
                min_batch_value=getattr(self.config, 'escrow_min_batch_value', 0.1),
                settlement_interval=getattr(self.config, 'escrow_settlement_interval', 300.0),
            ),
        )
        if self.content_economy:
            self.content_economy.set_escrow(self._mp_escrow)
        
        self.db_initialized = False
        self._broadcast_sent = set()

        # ── Batch Settlement (gas-efficient on-chain broadcasting) ──
        from prsm.economy.batch_settlement import BatchSettlementManager, SettlementMode
        self._batch_settlement = BatchSettlementManager(
            ftns_ledger=self.ftns_ledger,
            node_id=self.identity.node_id,
            connected_address=(
                self.ftns_ledger._connected_address
                if hasattr(self.ftns_ledger, '_connected_address')
                else None
            ),
            mode=SettlementMode.PERIODIC,
            flush_interval=600.0,     # 10 minutes
            flush_threshold=1.0,      # or when pending ≥ 1.0 FTNS
        )
        
        # ── Settler Registry (Phase 6: L2-style staking for batch security) ──
        from prsm.node.settler_registry import SettlerRegistry
        self._settler_registry = SettlerRegistry(
            min_settler_bond=10_000.0,    # 10K FTNS to become a settler
            settlement_threshold=3,        # 3-of-N multi-sig for batch approval
            max_settlers=10,
            ftns_service=_StakingFTNSAdapter(self.ledger, self.identity.node_id),
            staking_manager=self.staking_manager,
        )
        
        # Wire: When batch gets multi-sig approval, trigger settlement
        async def _on_batch_approved(batch):
            """Callback when batch reaches multi-sig threshold."""
            logger.info(
                "Batch approved via multi-sig, executing settlement",
                batch_id=batch.batch_id,
                signatures=len(batch.signatures),
            )
            # The batch settlement manager handles the actual on-chain tx
            result = await self._batch_settlement.flush()
            batch.settled = True
            batch.settlement_tx = result.tx_hashes[0] if result.tx_hashes else None
        
        self._settler_registry.on_settlement_ready(_on_batch_approved)

        # Agent Forge (Ring 5) removed in v1.6.0 — legacy NWTN AGI framework
        self.agent_forge = None

        # ── Confidential Compute (Ring 7) ─────────────────────────────
        try:
            from prsm.compute.tee.confidential_executor import ConfidentialExecutor
            from prsm.compute.tee.models import PrivacyLevel

            self.confidential_executor = ConfidentialExecutor(
                privacy_level=PrivacyLevel.STANDARD,
            )
            logger.info("Confidential compute (Ring 7) initialized")
        except ImportError:
            self.confidential_executor = None
            logger.debug("Confidential compute not available")

        # ── Model Sharding (Ring 8) ───────────────────────────────────
        try:
            from prsm.compute.model_sharding.executor import TensorParallelExecutor
            from prsm.compute.model_sharding.models import PipelineConfig

            self.tensor_executor = TensorParallelExecutor(
                confidential_executor=self.confidential_executor,
                pipeline_config=PipelineConfig(),
            )
            logger.info("Model sharding (Ring 8) initialized")
        except ImportError:
            self.tensor_executor = None
            logger.debug("Model sharding not available")

        # ── NWTN Model Service (Ring 9) ───────────────────────────────
        try:
            from prsm.compute.nwtn.training.model_service import NWTNModelService

            self.nwtn_model_service = NWTNModelService(
                tensor_executor=self.tensor_executor,
            )
            logger.info("NWTN model service (Ring 9) initialized")
        except ImportError:
            self.nwtn_model_service = None
            logger.debug("NWTN model service not available")

        # ── Security Hardening (Ring 10) ──────────────────────────────
        try:
            from prsm.security import IntegrityVerifier, PrivacyBudgetTracker, PipelineAuditLog

            self.integrity_verifier = IntegrityVerifier()
            self.privacy_budget = PrivacyBudgetTracker(max_epsilon=100.0)
            self.pipeline_audit_log = PipelineAuditLog()
            logger.info("Security hardening (Ring 10) initialized")
        except ImportError:
            self.integrity_verifier = None
            self.privacy_budget = None
            self.pipeline_audit_log = None
            logger.debug("Security hardening not available")

        # ── Local Discovery (mDNS fallback) ───────────────────────────
        try:
            from prsm.node.mdns_discovery import MDNSDiscovery

            self.mdns_discovery = MDNSDiscovery(
                node_id=self.identity.node_id,
                p2p_port=self.config.p2p_port,
                display_name=self.config.display_name,
            )
            logger.info("mDNS local discovery available")
        except ImportError:
            self.mdns_discovery = None

        # Wire ledger_sync and agent_registry into subsystems
        self.content_uploader.ledger_sync = self.ledger_sync
        # Wire content_economy into content_uploader for replication tracking
        if self.content_economy:
            self.content_uploader.content_economy = self.content_economy
        if self.compute_provider:
            self.compute_provider.ledger_sync = self.ledger_sync
            # Wire escrow and consensus into compute provider
            self.compute_provider.escrow = self._payment_escrow
            self.compute_provider.consensus = self._result_consensus
            # Wire batch settlement as the on-chain broadcast handler
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

        # NWTN orchestrator removed in v1.6.0 — legacy AGI framework replaced
        # by third-party LLMs invoked via MCP

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

        # Initialize native content storage
        try:
            from prsm.storage import init_content_store
            init_content_store()
            logger.info("ContentStore initialized")
        except Exception as exc:
            logger.warning("ContentStore initialization failed: %s", exc)

        # Initialize on-chain FTNS ledger (best-effort)
        if self.ftns_ledger:
            ft_initialized = await self.ftns_ledger.initialize()
            if ft_initialized:
                logger.info("FTNS on-chain ledger connected to Base mainnet")
            else:
                logger.info("FTNS on-chain ledger unavailable — running local mode only")
                self.ftns_ledger = None

        # Initialize SQLAlchemy database for NWTN features (best-effort)
        if not self.db_initialized:
            try:
                from prsm.core.database import init_database
                await init_database()
                self.db_initialized = True
                logger.info("SQLAlchemy database tables initialized")
            except Exception as e:
                logger.warning(f"SQLAlchemy DB init failed: {e} — NWTN features unavailable")
                self.db_initialized = False

        # Seed welcome grant if the node has no balance
        await self._seed_welcome_grant()

        if self.compute_provider:
            await self.compute_provider.start()
        await self.compute_requester.start()

        if self.storage_provider:
            await self.storage_provider.start()

        # ── Capability Announcement ──────────────────────────────────
        if hasattr(self.discovery, 'set_local_capabilities'):
            cap_list = list(self._local_capabilities)
            backends_list = []
            gpu_available = False
            if self.compute_provider:
                if self.compute_provider.resources.gpu_available:
                    gpu_available = True
                    if "gpu" not in cap_list:
                        cap_list.append("gpu")
                # NWTN backends subsystem removed in v1.6.0 — third-party LLMs
                # are now dispatched directly by MCP clients; no local backend
                # advertisement is required.
            self.discovery.set_local_capabilities(
                capabilities=cap_list,
                backends=backends_list,
                gpu_available=gpu_available,
            )
            await self.discovery.announce_capabilities()

            async def _periodic_capability_announce():
                while self._started:
                    await asyncio.sleep(300)
                    try:
                        await self.discovery.announce_capabilities()
                    except Exception as exc:
                        logger.debug("Capability re-announcement failed: %s", exc)

            self._capability_announce_task = asyncio.create_task(
                _periodic_capability_announce()
            )

        if self.content_index:
            self.content_index.start()
        if self.content_uploader:
            self.content_uploader.start()
        if self.content_provider:
            self.content_provider.start()
        # Wire content_economy into content_provider for payment processing (Phase 4)
        if self.content_economy and self.content_provider:
            self.content_provider.content_economy = self.content_economy
        # Start content economy (Phase 4)
        if self.content_economy:
            await self.content_economy.start()
        # Start multi-party escrow (Phase 4)
        if hasattr(self, '_mp_escrow') and self._mp_escrow:
            self._mp_escrow.start()
        if self.ledger_sync:
            self.ledger_sync.start()
        if self._payment_escrow:
            self._escrow_cleanup_task = asyncio.create_task(self._payment_escrow.periodic_cleanup())
        if hasattr(self, '_batch_settlement') and self._batch_settlement:
            # Update connected_address now that ftns_ledger may have initialized
            if self.ftns_ledger and hasattr(self.ftns_ledger, '_connected_address'):
                self._batch_settlement._connected_address = self.ftns_ledger._connected_address
            self._batch_settlement.start()
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
            # Try local mDNS discovery as fallback
            if hasattr(self, 'mdns_discovery') and self.mdns_discovery:
                self.mdns_discovery.start()
                logger.info("Started mDNS local discovery as bootstrap fallback")
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
        # Stop content economy (Phase 4)
        if self.content_economy:
            await self.content_economy.stop()
        # Stop multi-party escrow (Phase 4)
        if hasattr(self, '_mp_escrow') and self._mp_escrow:
            self._mp_escrow.stop()
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
        if hasattr(self, '_capability_announce_task'):
            self._capability_announce_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._capability_announce_task

        if self.discovery:
            await self.discovery.stop()
        if self.gossip:
            await self.gossip.stop()
        if self.transport:
            await self.transport.stop()
        if self.ledger:
            await self.ledger.close()

        # Close content storage
        try:
            from prsm.storage import close_content_store
            close_content_store()
            logger.info("ContentStore closed")
        except Exception as e:
            logger.warning(f"ContentStore close failed: {e}")

        self._started = False
        logger.info("PRSM node stopped")

    async def _ensure_ipfs_available(self) -> bool:
        """
        Deprecated: IPFS is replaced by native ContentStore.
        Returns True if ContentStore is available.
        """
        try:
            from prsm.storage import get_content_store
            return get_content_store() is not None
        except Exception:
            return False

    async def _seed_welcome_grant(self) -> None:
        """Rebuild wallet_balances cache from dag_transactions on startup.

        wallet_balances is just a performance cache — the source of truth
        is dag_transactions. We rebuild it from scratch once per startup
        to avoid any stale version counters from previous buggy runs.
        """
        try:
            # ── Step 1: Nuke and rebuild wallet_balances from truth ─────
            if hasattr(self.ledger, "_db"):
                # Delete all rows first (safe — cache only)
                await self.ledger._db.execute("DELETE FROM wallet_balances")
                # Rebuild from dag_transactions
                await self.ledger._db.execute(
                    """INSERT INTO wallet_balances (wallet_id, balance, version, last_updated)
                       SELECT w.wallet_id,
                              COALESCE((SELECT SUM(amount) FROM dag_transactions WHERE to_wallet = w.wallet_id), 0) -
                              COALESCE((SELECT SUM(amount) FROM dag_transactions WHERE from_wallet = w.wallet_id), 0),
                              1,
                              COALESCE((SELECT MAX(timestamp) FROM dag_transactions WHERE to_wallet = w.wallet_id OR from_wallet = w.wallet_id), 0)
                       FROM wallets w"""
                )
                await self.ledger._db.commit()
                # Reset the in-memory version cache
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
                    # Ensure wallet exists first
                    if not await self.ledger.wallet_exists(self.identity.node_id):
                        await self.ledger.create_wallet(self.identity.node_id, "node")
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
        """Queue a transaction for batch settlement on Base mainnet.

        KEY SAFETY: This must only be called AFTER the local ledger
        transaction has successfully committed. Never broadcast before
        local commit — otherwise we burn gas on transactions that get
        rolled back by TOCTOU/ConcurrentModification failures.

        Transactions are queued in BatchSettlementManager and flushed
        periodically (default: every 10 min or when pending >= 1.0 FTNS).
        This saves gas by netting opposing transfers and batching commits.
        """
        # Route through batch settlement if available
        if hasattr(self, '_batch_settlement') and self._batch_settlement:
            await self._batch_settlement.enqueue(transaction)
            return

        # Legacy fallback: direct on-chain transfer (no batch settlement)
        await self._direct_on_chain_transfer(transaction)

    async def _direct_on_chain_transfer(self, transaction) -> None:
        """Direct on-chain transfer (legacy fallback, no batching).

        Used when batch settlement is not initialized.
        """
        if not self.ftns_ledger or not self.ftns_ledger._is_initialized:
            return
        if not hasattr(transaction, "from_wallet") or not hasattr(transaction, "to_wallet"):
            return

        # Dedup: skip if we've already broadcast this transaction
        tx_key = (transaction.tx_id if hasattr(transaction, "tx_id") else "",
                  transaction.to_wallet if hasattr(transaction, "to_wallet") else "")
        if not hasattr(self, "_broadcast_sent"):
            self._broadcast_sent = set()
        if tx_key in self._broadcast_sent:
            logger.debug(
                f"Skipping duplicate FTNS broadcast for {tx_key[0][:12]}…"
            )
            return
        self._broadcast_sent.add(tx_key)

        to_addr = transaction.to_wallet
        target_address = None
        if to_addr.startswith("0x") and len(to_addr) >= 40:
            target_address = to_addr
        elif to_addr == self.identity.node_id and self.ftns_ledger._connected_address:
            target_address = self.ftns_ledger._connected_address
            logger.info(
                f"Bridging local payment to on-chain: "
                f"{self.identity.node_id[:12]}... -> {target_address}"
            )
        else:
            logger.debug(
                f"Skipping on-chain FTNS transfer for named wallet: {to_addr[:20]}…"
            )
            return

        amount = float(transaction.amount) if hasattr(transaction, "amount") else 0
        if amount <= 0:
            return

        try:
            tx_record = await self.ftns_ledger.transfer(
                job_id=tx_key[0],
                to_address=target_address,
                amount_ftns=amount,
            )
            if tx_record and tx_record.status == "confirmed":
                logger.info(
                    f"FTNS on-chain: {amount:.6f} confirmed "
                    f"(tx: {tx_record.tx_hash[:16]}…)"
                )
            elif tx_record and tx_record.status == "rejected":
                logger.warning(
                    f"FTNS on-chain transfer rejected: "
                    f"tx={tx_record.tx_hash[:16] if tx_record.tx_hash else 'N/A'}..."
                )
        except Exception as e:
            logger.error(f"FTNS on-chain transfer failed: {e}")

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
            "batch_settlement": self._batch_settlement.get_stats() if hasattr(self, '_batch_settlement') and self._batch_settlement else None,
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
            if self.config.tls_enabled and self.config.tls_cert_path:
                config.ssl_certfile = self.config.tls_cert_path
                config.ssl_keyfile = self.config.tls_key_path
            server = uvicorn.Server(config)
            await server.serve()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"API server error: {e}")

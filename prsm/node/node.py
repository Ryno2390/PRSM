"""
PRSM Node Runtime
=================

Main orchestrator that wires together identity, ledger, transport,
discovery, gossip, compute, storage, and the management API into
a single running node.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional
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
from prsm.node.dag_ledger import DAGLedger
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
from prsm.economy.tokenomics.staking_manager import StakingManager, StakingConfig
from prsm.config.networks import resolve_endpoints as _resolve_endpoints
from prsm.economy.ftns_onchain import OnChainFTNSLedger
from prsm.node.content_economy import ContentEconomy, RoyaltyModel

# BitTorrent integration
from prsm.core.bittorrent_client import BitTorrentClient, BitTorrentConfig
from prsm.core.bittorrent_manifest import TorrentManifestStore
from prsm.node.bittorrent_provider import BitTorrentProvider, BitTorrentProviderConfig
from prsm.node.bittorrent_requester import BitTorrentRequester, BitTorrentRequesterConfig

logger = logging.getLogger(__name__)


def build_persistent_privacy_budget(data_dir, identity, max_epsilon: float = 100.0):
    """Construct a PersistentPrivacyBudgetTracker rooted at <data_dir>/privacy_budget/.

    Phase 3.x.4 wiring factory. Imported by Node.__init__ AND by the
    integration test ``tests/integration/test_node_privacy_budget_persistence.py``
    so any drift between production wiring and the test harness is a
    compile-time error rather than a silent integration gap.

    Raises ``JournalCorruptionError`` (from prsm.security.privacy_budget_persistence)
    if the existing journal at the configured path fails verify_chain
    on construction. Caller MUST let this propagate — silently falling
    back to in-memory loses the audit trail.
    """
    from pathlib import Path

    from prsm.security.privacy_budget_persistence import (
        FilesystemPrivacyBudgetStore,
        PersistentPrivacyBudgetTracker,
    )

    budget_dir = Path(data_dir) / "privacy_budget"
    budget_dir.mkdir(parents=True, exist_ok=True)
    store = FilesystemPrivacyBudgetStore(budget_dir, identity.public_key_b64)
    return PersistentPrivacyBudgetTracker(
        max_epsilon=max_epsilon, store=store, identity=identity
    )


def _is_valid_eth_address(addr: Optional[str]) -> bool:
    """Cheap format check for 0x-prefixed 20-byte Ethereum address."""
    if not isinstance(addr, str):
        return False
    if not addr.startswith("0x") or len(addr) != 42:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in addr[2:])


def _derive_creator_address(ftns_ledger: Optional[Any]) -> Optional[str]:
    """Resolve the on-chain creator 0x address for this node.

    Priority:
      1. ftns_ledger._connected_address — canonical on-chain identity
         derived from FTNS_WALLET_PRIVATE_KEY in OnChainFTNSLedger.__init__.
      2. PRSM_CREATOR_ADDRESS env var — for nodes without the full
         on-chain stack that still want to register content on-chain.
      3. None — backward compat; on-chain routing silently skips and
         local fallback handles payments.

    Invalid addresses (bad format, empty string) log a warning and
    fall through rather than poisoning the on-chain registry with a
    hash bound to a garbage address.

    Phase 1.3 Task 3a.
    """
    if ftns_ledger is not None:
        addr = getattr(ftns_ledger, "_connected_address", None)
        if addr:
            if _is_valid_eth_address(addr):
                return addr
            logger.warning(
                f"ftns_ledger._connected_address has invalid format: "
                f"{addr!r}; falling through to env var."
            )

    env_addr = os.environ.get("PRSM_CREATOR_ADDRESS")
    if env_addr:
        if _is_valid_eth_address(env_addr):
            return env_addr
        logger.warning(
            f"PRSM_CREATOR_ADDRESS env var has invalid format: "
            f"{env_addr!r}; on-chain routing disabled for this node. "
            f"Local royalty fallback will be used."
        )

    return None


def _build_provenance_client_or_none():
    """T6 (2026-05-05): construct an on-chain ProvenanceRegistryClient
    if all required env vars are set. Returns None on any miss — the
    caller treats None as "skip on-chain registration."

    Required env vars:
      PRSM_ONCHAIN_PROVENANCE=1
      PRSM_PROVENANCE_REGISTRY_ADDRESS=<0x...>
      FTNS_WALLET_PRIVATE_KEY=<0x...>
      PRSM_BASE_RPC_URL=<https://...>  (optional; defaults to Base mainnet)

    Mirrors content_economy.py's _get_provenance_client() pattern but
    constructed eagerly at node-startup so the resulting client is
    available to ContentUploader at upload-time without a lazy-init
    race.
    """
    if os.getenv("PRSM_ONCHAIN_PROVENANCE", "").lower() not in ("1", "true", "yes"):
        return None
    addr = os.getenv("PRSM_PROVENANCE_REGISTRY_ADDRESS", "").strip()
    pk = os.getenv("FTNS_WALLET_PRIVATE_KEY", "").strip()
    if not addr or not pk:
        if not addr:
            logger.info(
                "PRSM_ONCHAIN_PROVENANCE=1 but PRSM_PROVENANCE_REGISTRY_ADDRESS "
                "not set — uploads will not register on-chain."
            )
        if not pk:
            logger.info(
                "PRSM_ONCHAIN_PROVENANCE=1 but FTNS_WALLET_PRIVATE_KEY not "
                "set — cannot sign on-chain registerContent calls."
            )
        return None
    try:
        from prsm.economy.web3.provenance_registry import ProvenanceRegistryClient
        rpc_url = _resolve_endpoints().rpc_url
        client = ProvenanceRegistryClient(
            rpc_url=rpc_url,
            contract_address=addr,
            private_key=pk,
        )
        logger.info(
            f"on-chain ProvenanceRegistry wired: {addr} via {rpc_url}"
        )
        return client
    except Exception as exc:
        logger.warning(
            f"failed to construct ProvenanceRegistryClient: "
            f"{type(exc).__name__}: {exc} — uploads will not register on-chain."
        )
        return None


# ──────────────────────────────────────────────────────────────────────
# PRSM-PROV-1 Item 6 — three-band dedup component builders.
# All three return None on any failure; the upload path falls back
# to legacy 2-band auto-attribute behavior when any component is None.
# ──────────────────────────────────────────────────────────────────────


def _build_threshold_resolver_or_none():
    """Construct the canonical ``ThresholdResolver`` from the
    project's ``prsm/data/dedup_thresholds.yaml``. Returns None on
    any IO/parse failure — uploads fall back to ``_SemanticIndex``
    class-constant thresholds.
    """
    try:
        from prsm.data.dedup.thresholds import ThresholdResolver
        return ThresholdResolver.from_default_path()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failed to load ThresholdResolver: %s — uploads will use "
            "legacy 2-band class-constant thresholds",
            exc,
        )
        return None


def _build_arbitration_queue_or_none():
    """Construct a ``FilesystemArbitrationQueue`` rooted at
    ``~/.prsm/arbitration_queue/``. Returns None on any IO failure
    — uploads then run without disputed-band recording (legacy 2-band).
    """
    try:
        from prsm.data.dedup.arbitration import FilesystemArbitrationQueue
        queue_dir = Path.home() / ".prsm" / "arbitration_queue"
        return FilesystemArbitrationQueue(queue_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failed to construct FilesystemArbitrationQueue: %s — "
            "uploads will not record disputed-band records",
            exc,
        )
        return None


def _build_arbitration_proposal_sink_or_none():
    """Construct a ``TokenWeightedVotingProposalSink`` if the operator
    has configured a system proposer address. Returns None otherwise.

    Required env var:
      ``PRSM_ARBITRATION_PROPOSER_ID`` — system-level proposer (the
      Foundation Safe address or a delegate). Must hold sufficient
      FTNS to cover proposal submission fees. Without this, the
      arbitration queue still runs (records persist + are retrievable
      via ``list_pending``), but no governance proposals are auto-
      created — councils may author proposals by hand from queue
      entries.

    Disable explicitly with ``PRSM_ARBITRATION_PROPOSER_ID=""``.
    """
    proposer_id = os.getenv("PRSM_ARBITRATION_PROPOSER_ID", "").strip()
    if not proposer_id:
        return None
    try:
        from prsm.economy.governance.arbitration_sink import (
            TokenWeightedVotingProposalSink,
        )
        from prsm.economy.governance.voting import TokenWeightedVoting
        voting = TokenWeightedVoting()
        sink = TokenWeightedVotingProposalSink(
            voting=voting,
            proposer_id=proposer_id,
        )
        logger.info(
            "TokenWeightedVotingProposalSink wired with proposer_id=%s "
            "(disputed-band records will surface as ARBITRATION_DISPUTE "
            "proposals)",
            proposer_id,
        )
        return sink
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failed to construct TokenWeightedVotingProposalSink "
            "(proposer_id=%s): %s — disputed-band records still queued, "
            "but no governance proposals auto-created",
            proposer_id,
            exc,
        )
        return None


class _FailClosedAnchor:
    """Pre-T3c stub: every lookup returns None, so cross-node manifest
    verification refuses to trust. Used when no production
    PublisherKeyAnchor address is configured. The fail-closed default
    is intentional: better to refuse than to accept unverified bytes.

    Hoisted to module scope (was previously nested in
    ``_build_dht_components_or_none``) so test code can detect the
    pre-anchor-wired path via isinstance.
    """

    def lookup(self, node_id):  # noqa: ARG002
        return None


def _fail_closed_creator_pubkey_for(content_hash):  # noqa: ARG001
    """Fallback creator-pubkey resolver. Returns None for every input.

    Used when no LocalEmbeddingIndex is configured (no T3d resolver
    can be built) or when the T3d wiring fails. Every embedding-DHT
    signature verification then returns None → SignatureVerification
    Error → reject. Cold-start correctness preserved at the cost of
    cross-node embedding fetch.
    """
    return None


def _make_creator_pubkey_for(embedding_index, anchor):
    """T3d (option (a)) — content_hash → creator_node_id → pubkey resolver.

    The verifier hands ``creator_pubkey_for`` a content_hash. We:
      1. look up the local LocalEmbeddingIndex for any record with that
         content_hash → creator_node_id (populated at upload time on
         the publisher node, or after a verified cross-node fetch on a
         relay node)
      2. anchor.lookup(creator_node_id) → base64 pubkey on-chain
      3. base64-decode → bytes for ed25519 verify

    Limitation (intentional): only resolves for content this node has
    previously seen. Cold-start cross-node embedding fetch — content
    B has never seen — still returns None → reject. This is the
    correct fail-closed behavior pre-(b)-extension. Once any node in
    the swarm has cached the (content_hash, creator_id) record, that
    node serves as a relay for verification.

    Returns the same fail-closed stub if either dependency is missing
    (defensive — the caller is responsible for not constructing the
    resolver in that case, but we don't trust the caller).
    """
    if embedding_index is None or anchor is None:
        return _fail_closed_creator_pubkey_for

    def _resolve(content_hash):
        try:
            creator_id = embedding_index.lookup_creator_by_content_hash(
                content_hash,
            )
        except Exception:  # noqa: BLE001
            return None
        if not creator_id:
            return None
        try:
            pubkey_b64 = anchor.lookup(creator_id)
        except Exception:  # noqa: BLE001
            return None
        if not pubkey_b64 or not isinstance(pubkey_b64, str):
            return None
        try:
            import base64
            return base64.b64decode(pubkey_b64, validate=True)
        except Exception:  # noqa: BLE001
            return None

    return _resolve


def _build_publisher_key_anchor_client_or_none():
    """T3c — construct a PublisherKeyAnchorClient from env vars.

    Mirrors ``_build_provenance_client_or_none``: env-driven, fail-soft.
    Returns None when any required piece is missing OR when the web3
    construction itself fails — the caller falls back to the
    fail-closed anchor in that case.

    Required env vars:
      PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS=<0x…>
      PRSM_BASE_RPC_URL=<https://…>  (optional; defaults to Base mainnet
                                       so the same env var that drives
                                       the provenance client also drives
                                       the anchor client)

    The anchor is read-only on the verifier side — no private_key is
    passed. Read-only mode supports lookup() but rejects register_self()
    (the publisher side has its own anchor instance for that with
    PRSM_FTNS_WALLET_PRIVATE_KEY).
    """
    addr = os.getenv("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "").strip()
    if not addr:
        logger.debug(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS not set — DHT manifest "
            "verification will use fail-closed anchor."
        )
        return None
    rpc_url = os.getenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
    try:
        from prsm.security.publisher_key_anchor.client import (
            PublisherKeyAnchorClient,
        )
        client = PublisherKeyAnchorClient(
            contract_address=addr,
            rpc_url=rpc_url,
        )
        logger.info(
            f"PublisherKeyAnchorClient wired: {addr} via {rpc_url}"
        )
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"PublisherKeyAnchorClient construction failed: "
            f"{type(exc).__name__}: {exc} — falling back to "
            f"fail-closed anchor."
        )
        return None


def _build_dht_components_or_none(
    *, identity, listen_host, dht_listen_port,
    manifest_index, embedding_index,
    local_fingerprint_index=None,
):
    """PRSM-DHT-TRANSPORT T3b/T3c — opt-in construction of
    :class:`DHTNodeComponents`.

    Returns ``None`` when DHT is disabled or when the prerequisites
    can't be satisfied. The node continues to function without the
    DHT (FilesystemModelRegistry falls back to local-only lookup,
    ContentUploader skips cross-node embedding gossip) — same
    fail-soft behavior as ``_build_provenance_client_or_none`` above.

    Enabled when EITHER:
      - ``NodeConfig.dht_enabled == True`` (caller passes already
        through the indexes / identity), OR
      - ``PRSM_DHT_ENABLED=1`` env var is set (operator-side override).

    Trust inputs:
      - ``anchor`` for ManifestDHT verification — T3c wires
        :class:`PublisherKeyAnchorClient` from
        ``PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS`` + ``PRSM_BASE_RPC_URL``.
        When the address env var is unset, falls back to
        :class:`_FailClosedAnchor` so the node still boots, but
        cross-node manifest verification refuses every signature.
      - ``creator_pubkey_for`` for EmbeddingDHT — still
        :func:`_fail_closed_creator_pubkey_for`. Production wiring
        needs a content-hash → creator-node-id mapping that isn't
        ratified yet (the on-chain ProvenanceRegistry stores
        creator-as-EVM-address; the corresponding PRSM node_id mapping
        is the gap). Tracked as a follow-on; T3c lights up the
        manifest path without it.
      - ``verify_signature`` is real Ed25519 from cryptography.hazmat.
    """
    if manifest_index is None and embedding_index is None:
        return None
    try:
        from prsm.network.dht_components import DHTNodeComponents
        from prsm.node.transport_adapter import DirectAdapter
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            f"DHT components unavailable (import failed): "
            f"{type(exc).__name__}: {exc}",
        )
        return None

    def _real_verify_signature(pubkey_bytes, message, signature) -> bool:
        if not pubkey_bytes:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(pubkey_bytes).verify(
                signature, message,
            )
        except InvalidSignature:
            return False
        except Exception:  # noqa: BLE001
            return False
        return True

    # T3c+T3d: build a single anchor instance shared between both
    # DHT paths. The anchor itself is constructed once — production
    # PublisherKeyAnchorClient when env vars are set, _FailClosedAnchor
    # otherwise. Both downstream paths inherit the same trust posture.
    shared_anchor = (
        _build_publisher_key_anchor_client_or_none()
        or _FailClosedAnchor()
    )

    anchor_for_manifest = shared_anchor if manifest_index is not None else None

    # T3d (option (a)): when only the embedding DHT is enabled, the
    # local-index resolver still needs an anchor for the on-chain
    # creator-pubkey lookup. Reuse the shared anchor.
    if embedding_index is not None:
        creator_pubkey_for = _make_creator_pubkey_for(
            embedding_index=embedding_index,
            anchor=shared_anchor,
        )
    else:
        creator_pubkey_for = None

    try:
        components = DHTNodeComponents.build(
            my_node_id=identity.node_id,
            my_host=listen_host or "127.0.0.1",
            dht_listen_port=dht_listen_port,
            transport_adapter=DirectAdapter(),
            listen_host=listen_host or "0.0.0.0",
            local_manifest_index=manifest_index,
            local_embedding_index=embedding_index,
            local_fingerprint_index=local_fingerprint_index,
            anchor=anchor_for_manifest,
            creator_pubkey_for=creator_pubkey_for,
            verify_signature=(
                _real_verify_signature if embedding_index is not None else None
            ),
        )
        # Stash the verifier inputs on the instance so start() can
        # forward them — keeps build() pure of trust state, while
        # avoiding a second hop through env. (Field name kept as
        # _t3b_* so the existing test suite continues to introspect
        # via stable attribute names; the underlying anchor is now
        # the production PublisherKeyAnchorClient when configured.)
        components._t3b_anchor = anchor_for_manifest  # noqa: SLF001
        components._t3b_creator_pubkey_for = creator_pubkey_for  # noqa: SLF001
        components._t3b_verify_signature = (  # noqa: SLF001
            _real_verify_signature if embedding_index is not None else None
        )
        return components
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"DHTNodeComponents.build failed: "
            f"{type(exc).__name__}: {exc} — node will run without DHT.",
        )
        return None


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
    - Storage provider (ContentStore pins)
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

        # Native-storage migration PR 2c: ContentPublisher/Retriever
        # composed from the BT layer above. Set in initialize() once
        # the BT layer is live; None when libtorrent is unavailable.
        self.content_publisher: Optional[Any] = None
        self.content_retriever: Optional[Any] = None

        # PRSM-DHT-TRANSPORT T3b: opt-in DHT stack (Manifest + Embedding)
        # Constructed in initialize() iff dht_enabled or PRSM_DHT_ENABLED=1.
        self.dht_components: Optional[Any] = None

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
        _embedding_model_id: Optional[str] = None
        if _HAS_EMBEDDING_API:
            try:
                _embed_api = RealEmbeddingAPI()
                _embedding_fn = _embed_api.generate_embedding
                # T3.6 (PRSM-PROV-1): the model_id used to key the
                # cross-node EmbeddingDHT. Sourced from the same
                # RealEmbeddingAPI that produces vectors here so the
                # (vector, model_id) tuple is internally consistent.
                # Falls back to the env-configured local model name
                # when no remote provider is wired (Item 1 fallback
                # path).
                _embedding_model_id = getattr(
                    _embed_api, "_st_model_name", None,
                )
            except Exception as _e:
                logger.debug(f"Embedding API unavailable, semantic dedup disabled: {_e}")

        _semantic_index_path = Path.home() / ".prsm" / "semantic_index.json"
        # T4.9.next3: persist the per-kind binary fingerprint index next
        # to the semantic index. Same lifecycle: persists across node
        # restarts so warm-cache dedup survives a process bounce.
        _fingerprint_index_path = Path.home() / ".prsm" / "fingerprint_index.json"

        # PRSM-PROV-1 Item 6 — three-band dedup wiring. All three
        # components degrade to None on failure; uploads still work
        # without arbitration (legacy 2-band behavior).
        _threshold_resolver = _build_threshold_resolver_or_none()
        _arbitration_queue = _build_arbitration_queue_or_none()
        _arbitration_proposal_sink = _build_arbitration_proposal_sink_or_none()

        # T3.6 (PRSM-PROV-1): LocalEmbeddingIndex backs the
        # EmbeddingDHT — every successful upload + embedding gets a
        # creator-signed record persisted here, so peers querying us
        # via the DHT can verify and serve them. Stored under
        # ~/.prsm/embedding_index/. None disables the registration
        # path (existing behavior preserved when the embedding_dht
        # subpackage is unavailable).
        _embedding_index = None
        try:
            from prsm.network.embedding_dht.local_index import (
                LocalEmbeddingIndex,
            )
            _embedding_index_path = Path.home() / ".prsm" / "embedding_index"
            _embedding_index_path.mkdir(parents=True, exist_ok=True)
            _embedding_index = LocalEmbeddingIndex(_embedding_index_path)
        except Exception as _e:  # noqa: BLE001
            logger.debug(
                f"EmbeddingDHT local index unavailable, cross-node "
                f"dedup-serve disabled: {_e}"
            )

        # T4.9.next5: parallel server-side store for the binary
        # fingerprint lane. Same lifecycle + failure-mode posture as
        # _embedding_index above. Without this, peers can ASK us for
        # fingerprints but we'd have nothing to serve.
        _local_fingerprint_index = None
        try:
            from prsm.network.embedding_dht.local_fingerprint_index import (
                LocalFingerprintIndex,
            )
            _local_fp_index_path = (
                Path.home() / ".prsm" / "local_fingerprint_index"
            )
            _local_fp_index_path.mkdir(parents=True, exist_ok=True)
            _local_fingerprint_index = LocalFingerprintIndex(
                _local_fp_index_path,
            )
        except Exception as _e:  # noqa: BLE001
            logger.debug(
                f"FingerprintDHT local index unavailable, cross-node "
                f"fingerprint-serve disabled: {_e}"
            )

        # PRSM-DHT-TRANSPORT T3b — construct DHTNodeComponents if the
        # operator opted in. Off-by-default; enable via NodeConfig.dht_enabled
        # or PRSM_DHT_ENABLED=1. The components run their own asyncio loop
        # in a daemon thread (DHTLoopRunner) so the Node's existing event
        # loop is unaffected.
        _dht_enabled = (
            self.config.dht_enabled
            or os.getenv("PRSM_DHT_ENABLED", "").lower() in ("1", "true", "yes")
        )
        if _dht_enabled:
            _manifest_index = None
            try:
                from prsm.network.manifest_dht.local_index import (
                    LocalManifestIndex,
                )
                _manifest_index_path = Path.home() / ".prsm" / "manifest_index"
                _manifest_index_path.mkdir(parents=True, exist_ok=True)
                _manifest_index = LocalManifestIndex(_manifest_index_path)
            except Exception as _e:  # noqa: BLE001
                logger.debug(
                    f"ManifestDHT local index unavailable, cross-node "
                    f"manifest-serve disabled: {_e}"
                )
            self.dht_components = _build_dht_components_or_none(
                identity=self.identity,
                listen_host=self.config.listen_host,
                dht_listen_port=self.config.dht_listen_port,
                manifest_index=_manifest_index,
                embedding_index=_embedding_index,
                local_fingerprint_index=_local_fingerprint_index,
            )
            if self.dht_components is not None:
                logger.info(
                    "DHT components constructed "
                    f"(manifest={_manifest_index is not None}, "
                    f"embedding={_embedding_index is not None})"
                )

        # ── Content Provider (Cross-Node Retrieval) ───────────────────────
        # Phase 1.3: ContentProvider is constructed BEFORE ContentUploader so
        # the uploader can hold a reference and populate provider._local_content
        # on every successful upload / DB hydration. Previously the provider was
        # built after the uploader, leaving register_local_content with zero
        # production callers — the serve path returned not_found for every CID
        # and the on-chain royalty payment never fired end-to-end.
        _bandwidth_limiter = None
        if self.storage_provider:
            _bandwidth_limiter = self.storage_provider.bandwidth_limiter

        self.content_provider = ContentProvider(
            identity=self.identity,
            transport=self.transport,
            gossip=self.gossip,
            content_index=self.content_index,
            bandwidth_limiter=_bandwidth_limiter,
        )

        # Phase 1.3 Task 3e: wire content_provider back into
        # storage_provider so its _on_direct_content_request can
        # defer to the canonical serve path when the provider also
        # has the CID. Without this, both MSG_DIRECT handlers race
        # and the legacy-shape response from storage_provider wins
        # (arrives first because it skips payment), which the
        # canonical ContentResponseMessage.from_payload() parser on
        # the requester side downgrades to ERROR.
        if self.storage_provider is not None:
            self.storage_provider._content_provider = self.content_provider

        # ── On-Chain FTNS Ledger (Base mainnet) ────────────────────
        # Phase 1.3 Task 3a: instantiated BEFORE ContentUploader so the
        # uploader bootstrap can derive creator_address from the ledger's
        # _connected_address. Previously this was constructed ~200 lines
        # later, which left creator_address=None at upload-time and
        # silently bypassed provenance_hash computation / on-chain
        # royalty routing for every production upload.
        self.ftns_ledger = OnChainFTNSLedger(
            node_id=self.identity.node_id,
        )

        # T6 (2026-05-05): on-chain ProvenanceRegistry client. Lazy-init at
        # node construction so ContentUploader can register content on-chain
        # at upload-time. Same env-var contract as content_economy.py:
        # PRSM_ONCHAIN_PROVENANCE=1 + PRSM_PROVENANCE_REGISTRY_ADDRESS +
        # FTNS_WALLET_PRIVATE_KEY. Returns None gracefully when any required
        # piece is missing — the upload still succeeds locally.
        provenance_client = _build_provenance_client_or_none()

        # Native-storage migration PR 2c: content_publisher / content_retriever
        # are attached AFTER the BT layer is initialised below (~line 950).
        # The uploader's internal _publish_content / _fetch_content helpers
        # log + return None until that attachment runs.
        self.content_uploader = ContentUploader(
            identity=self.identity,
            gossip=self.gossip,
            ledger=self.ledger,
            transport=self.transport,
            content_index=self.content_index,
            embedding_fn=_embedding_fn,
            semantic_index_path=_semantic_index_path,
            content_provider=self.content_provider,
            creator_address=_derive_creator_address(self.ftns_ledger),
            provenance_client=provenance_client,
            # T3.6 (PRSM-PROV-1): cross-node embedding gossip wiring.
            # embedding_dht_client stays None until the Phase 6 P2P
            # transport is wired into a Kademlia routing table at the
            # node level — at which point peer-side fetch will engage
            # without further uploader changes (T3.5's escalation
            # path is gated on a real client). Local-side store is
            # active now: every signed embedding lands in
            # _embedding_index so future peer fetches succeed.
            embedding_model_id=_embedding_model_id,
            embedding_dht_client=None,
            embedding_index=_embedding_index,
            # T4.9.next3: persist FingerprintIndex to disk + share the
            # embedding-lane DHT wiring. ``embedding_dht_client`` above
            # is still None (Phase 6 P2P transport hasn't lit it yet),
            # so fingerprint escalation is also dormant until the same
            # client switch is flipped — at which point both lanes
            # engage simultaneously without further uploader changes.
            fingerprint_index_path=_fingerprint_index_path,
            # T4.9.next5: serve-side fingerprint storage. Same instance
            # is also passed to DHTNodeComponents above, so the
            # uploader's _register_local_fingerprint and the
            # EmbeddingDHTServer's fetch handler share a single store.
            local_fingerprint_index=_local_fingerprint_index,
            # PRSM-PROV-1 Item 6 (T6.3 + T6.5 + T6.5.gov.next):
            # disputed-band three-tier wiring. All three may be None
            # — when so, uploads fall back to legacy 2-band auto-
            # attribute behavior.
            threshold_resolver=_threshold_resolver,
            arbitration_queue=_arbitration_queue,
            arbitration_proposal_sink=_arbitration_proposal_sink,
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

            # Native-storage migration PR 2c (2026-05-07): wire
            # ContentPublisher / ContentRetriever onto the already-
            # constructed ContentUploader. The uploader was initialised
            # earlier with content_publisher=None; production uploads
            # logged "content_publisher is None — cannot publish" until
            # this attachment runs. With these set, prsm_upload_dataset
            # actually distributes content via the proprietary
            # BitTorrent layer instead of returning a placeholder.
            from prsm.node.content_publisher import (
                ContentPublisher,
                ContentRetriever,
            )

            staging_dir = self.config.data_dir / "content_publish_staging"
            cache_dir = self.config.data_dir / "content_fetch_cache"

            self.content_publisher = ContentPublisher(
                bt_provider=self.bt_provider,
                staging_dir=staging_dir,
            )
            self.content_retriever = ContentRetriever(
                bt_requester=self.bt_requester,
                cache_dir=cache_dir,
            )
            self.content_uploader.content_publisher = self.content_publisher
            self.content_uploader.content_retriever = self.content_retriever
            logger.info(
                "ContentUploader wired through ContentPublisher (Tier A) — "
                "uploads now distribute via the BitTorrent layer."
            )
        else:
            logger.info("BitTorrent not available - libtorrent may not be installed")
            self.content_publisher = None
            self.content_retriever = None

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

        # ── Remote Shard Dispatcher (Phase 2) ─────────────────────
        # Plugs into TensorParallelExecutor's remote_dispatcher slot
        # (wired below). Tier A verification (receipt-only) in Phase 2;
        # Tiers B/C plug in at Phase 7 via the same VerificationStrategy
        # protocol without changing the dispatch protocol.
        from prsm.compute.remote_dispatcher import RemoteShardDispatcher
        from prsm.compute.shard_receipt import ReceiptOnlyVerification

        self.remote_shard_dispatcher = RemoteShardDispatcher(
            identity=self.identity,
            transport=self.transport,
            payment_escrow=self._payment_escrow,
            verification_strategy=ReceiptOnlyVerification(),
            default_timeout=30.0,
            max_retries=1,
            max_shard_bytes=10 * 1024 * 1024,
            local_fallback=None,
        )

        async def _tensor_remote_dispatch(shard, input_data, assignment):
            """Adapter from TensorParallelExecutor's (shard, bytes,
            assignment) contract to RemoteShardDispatcher.dispatch's
            (shard, np.ndarray, node_id, job_id, stake_tier, amount)
            contract. Wraps output in the dict shape _execute_local
            returns.
            """
            import numpy as _np

            from prsm.compute.model_sharding.models import PipelineStakeTier

            node_id = assignment.get("node_id", "")
            job_id = assignment.get("job_id", "")
            tier_label = assignment.get("stake_tier", "standard")
            tier_map = {t.label: t for t in PipelineStakeTier}
            stake_tier = tier_map.get(tier_label, PipelineStakeTier.STANDARD)
            escrow_amount = float(assignment.get("escrow_amount_ftns", 1.0))

            input_arr = _np.frombuffer(input_data, dtype=_np.float64)
            if input_arr.size == 0:
                input_arr = _np.ones(
                    shard.tensor_shape[-1] if len(shard.tensor_shape) > 1
                    else shard.tensor_shape[0]
                )

            output = await self.remote_shard_dispatcher.dispatch(
                shard=shard,
                input_tensor=input_arr,
                node_id=node_id,
                job_id=job_id,
                stake_tier=stake_tier,
                escrow_amount_ftns=escrow_amount,
            )

            return {
                "shard_index": shard.shard_index,
                "node_id": node_id,
                "output_array": output.tolist(),
                "execution_mode": "remote",
            }

        self._tensor_remote_dispatch = _tensor_remote_dispatch

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

        # Agent Forge (Ring 5) removed in v1.6.0 — legacy NWTN AGI framework.
        #
        # Replacement = QueryOrchestrator (data-query path per Vision §4).
        # Default-disabled. Operators opt in via PRSM_QUERY_ORCHESTRATOR_ENABLED=1
        # AFTER verifying their deployment delivers the canonical workflow
        # end-to-end. Disabled-by-default keeps `agent_forge = None` so the
        # MCP `BROKEN_TOOLS_HIDDEN` gate stays effective until B8 lands.
        #
        # See:
        #   docs/2026-05-08-query-orchestrator-wiring-readiness.md
        #   docs/2026-05-07-aggregator-selector-threat-model.md
        self.agent_forge = self._build_query_orchestrator_or_none()

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
                remote_dispatcher=self._tensor_remote_dispatch,
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

            # Persistent privacy-budget tracker (Phase 3.x.4): wraps
            # the in-memory PrivacyBudgetTracker in a signed, append-only
            # journal at <data_dir>/privacy_budget/. Restart-survival +
            # tamper-evidence; verify_chain on construction refuses to
            # silently reconstitute possibly-wrong cumulative ε state.
            #
            # JournalCorruptionError propagates out of the try block —
            # it indicates an existing journal is broken, which an
            # operator must investigate (vs. silently falling back to
            # in-memory and losing the audit trail).
            self.privacy_budget = build_persistent_privacy_budget(
                self.config.data_dir, self.identity
            )

            self.pipeline_audit_log = PipelineAuditLog()
            logger.info(
                "Security hardening (Ring 10) initialized; privacy-budget "
                "journal at %s/privacy_budget",
                self.config.data_dir,
            )
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

    def _start_dht_components_if_present(self) -> None:
        """T3b — start the DHT components stack on its own loop thread.

        Idempotent. Logs the bound port at INFO so operators can verify
        the listener is reachable. A failed start is logged at WARNING
        but does NOT fail the Node.start() — the rest of the node
        continues to run with the DHT in degraded "library-only" mode.
        """
        if self.dht_components is None:
            return
        try:
            port = self.dht_components.start(
                anchor=getattr(self.dht_components, "_t3b_anchor", None),
                creator_pubkey_for=getattr(
                    self.dht_components, "_t3b_creator_pubkey_for", None,
                ),
                verify_signature=getattr(
                    self.dht_components, "_t3b_verify_signature", None,
                ),
            )
            logger.info(
                f"DHT listener bound on "
                f"{self.config.listen_host}:{port}"
            )
            # T4.9.next4: late-bind the EmbeddingDHTClient that
            # DHTNodeComponents.start() just constructed into the
            # ContentUploader so cross-node dedup engages on both
            # lanes (text-vector + binary fingerprint). Skipped when
            # the embedding lane wasn't enabled at build() time
            # (manifest-only DHT setups).
            embedding_client = getattr(
                self.dht_components, "embedding_client", None,
            )
            if embedding_client is not None and self.content_uploader is not None:
                self.content_uploader.set_embedding_dht_client(embedding_client)
                logger.info(
                    "EmbeddingDHTClient late-bound into ContentUploader — "
                    "cross-node dedup is live (semantic + fingerprint lanes)"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"DHT components start failed: "
                f"{type(exc).__name__}: {exc} — "
                f"continuing without inbound DHT listener.",
            )
            self.dht_components = None

    async def start(self) -> None:
        """Start all subsystems concurrently."""
        if self._started:
            return

        await self.transport.start()
        await self.gossip.start()
        await self.discovery.start()
        # T3b: bring up the DHT listener + clients on their own
        # asyncio loop thread. Idempotent + non-fatal on failure.
        self._start_dht_components_if_present()

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

    def _build_query_orchestrator_or_none(self):
        """Construct QueryOrchestrator from this node's primitives, or
        return None if the operator hasn't opted in via
        `PRSM_QUERY_ORCHESTRATOR_ENABLED=1` OR if any required adapter
        cannot be constructed against current node state.

        Default-disabled. Behavior identical to v1.6.0
        (`agent_forge = None`) until the operator explicitly enables.

        On any wiring failure with the env var set, logs the reason
        and falls back to None — the operator gets a clear signal
        that their deployment is missing something + the canonical
        workflow stays gated rather than half-broken.

        See `prsm/compute/query_orchestrator/node_wiring.py` for the
        factory contract + `docs/2026-05-08-query-orchestrator-wiring-readiness.md`
        for the wiring program.
        """
        from prsm.compute.query_orchestrator.node_wiring import (
            is_query_orchestrator_enabled,
        )
        if not is_query_orchestrator_enabled():
            return None

        try:
            from prsm.compute.query_orchestrator import (
                FoundationBeaconProvider,
                MarketplaceCandidatePoolProvider,
                SemanticIndexAdapter,
                SentenceTransformerEmbedder,
                SwarmDispatcherAdapter,
            )
            from prsm.compute.query_orchestrator.node_wiring import (
                build_query_orchestrator_for_node,
            )
            from prsm.marketplace.directory import MarketplaceDirectory
            from prsm.marketplace.reputation import ReputationTracker
        except ImportError as exc:
            logger.warning(
                "QueryOrchestrator wiring unavailable: %s — falling back "
                "to agent_forge=None",
                exc,
            )
            return None

        # All 5 adapter dependencies required. Each construction step
        # raises clearly if a node-side primitive is missing.
        try:
            if self.content_uploader is None:
                raise RuntimeError(
                    "content_uploader not initialized — cannot wire "
                    "SemanticIndexAdapter"
                )
            if self.agent_dispatcher is None:
                raise RuntimeError(
                    "agent_dispatcher not initialized — cannot wire "
                    "SwarmDispatcherAdapter"
                )
            if self.gossip is None:
                raise RuntimeError(
                    "gossip not initialized — cannot wire "
                    "MarketplaceDirectory"
                )

            # Marketplace + reputation primitives are constructed here
            # because node.py doesn't currently own them. Once the
            # marketplace orchestrator becomes a top-level node
            # subsystem (separate sprint), pull these from self.* instead.
            marketplace_directory = MarketplaceDirectory(self.gossip)
            reputation_tracker = ReputationTracker()

            semantic_index = SemanticIndexAdapter(
                embedder=SentenceTransformerEmbedder(),
                index=self.content_uploader._semantic_index,
            )
            dispatcher = SwarmDispatcherAdapter(
                agent_dispatcher=self.agent_dispatcher,
                per_shard_budget_ftns=100,  # placeholder; orch retry-loop owns
            )
            # AggregatorClient + beacon need a Foundation Safe address
            # that this deployment trusts. Default to mainnet Safe;
            # operators on other networks override via constructor
            # extension (separate ratification + tooling sprint).
            from prsm.compute.query_orchestrator import (
                AggregatorClientAdapter,
                ChainedEndpointResolver,
                HttpAggregateTransport,
                StaticMapEndpointResolver,
                TransportPeerEndpointResolver,
            )
            from prsm.compute.query_orchestrator.foundation_safe_resolver import (
                resolve_foundation_safe_address,
            )
            beacon_provider = FoundationBeaconProvider(
                foundation_safe_address=resolve_foundation_safe_address(),
            )
            # Endpoint resolver: ordered list of backends. Operators
            # supply a static map via PRSM_AGGREGATOR_ENDPOINT_MAP
            # (JSON of {node_id: base_url}) for known aggregators;
            # unknown node_ids fall back to the WS transport peer
            # registry (host:port from the live connection).
            import json as _json_for_endpoints
            import os as _os_for_endpoints
            _static_map_raw = _os_for_endpoints.environ.get(
                "PRSM_AGGREGATOR_ENDPOINT_MAP", "",
            ).strip()
            _static_map = {}
            if _static_map_raw:
                try:
                    _static_map = _json_for_endpoints.loads(_static_map_raw)
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "PRSM_AGGREGATOR_ENDPOINT_MAP malformed JSON: %s — "
                        "ignoring static map, using transport-peer fallback only",
                        exc,
                    )
            _endpoint_resolver = ChainedEndpointResolver([
                StaticMapEndpointResolver(_static_map),
                TransportPeerEndpointResolver(self.transport),
            ])
            aggregator_client = AggregatorClientAdapter(
                prompter_pubkey=self.identity.public_key_bytes,
                prompter_node_id=self.identity.node_id,
                prompter_signer=self.identity.sign,
                prompter_privkey=self.identity.private_key_bytes,
                beacon_provider=beacon_provider,
                transport=HttpAggregateTransport(
                    endpoint_resolver=_endpoint_resolver,
                ),
            )
            candidate_pool_provider = MarketplaceCandidatePoolProvider(
                directory=marketplace_directory,
                reputation=reputation_tracker,
            )

            orchestrator = build_query_orchestrator_for_node(
                semantic_index=semantic_index,
                dispatcher=dispatcher,
                aggregator_client=aggregator_client,
                candidate_pool_provider=candidate_pool_provider,
                beacon_provider=beacon_provider,
            )
            logger.info(
                "QueryOrchestrator wired (env-enabled). agent_forge live."
            )
            return orchestrator
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "QueryOrchestrator construction failed: %s — falling "
                "back to agent_forge=None. (Operator must wire missing "
                "primitive before re-enabling.)",
                exc,
            )
            return None

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
        # T3b: stop the DHT components before transport so any in-flight
        # outbound DHT RPC has the underlying transport adapter still
        # available during teardown. Idempotent.
        if self.dht_components is not None:
            try:
                self.dht_components.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"DHT components stop failed: "
                    f"{type(exc).__name__}: {exc}"
                )
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

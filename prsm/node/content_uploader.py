"""
Content Uploader & Retrieval
============================

Upload content to the PRSM proprietary BitTorrent layer with
provenance tracking for royalties, and retrieve content from the
network by torrent infohash.

Creates a verifiable provenance chain so the original creator
earns FTNS when other nodes access or use their content.

Sprint 4 Phase 4: Added request_content() for cross-node downloads
with provider discovery via ContentIndex and content hash verification.
Phase 1.3 Task 3b: the client-side request_content/server-side
handler pair has been retired here; ContentProvider is the canonical
serve path and ContentProvider.request_content is the client path.
"""

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from prsm.node.content_provider import ContentProvider

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from prsm.node.gossip import (
    GOSSIP_CONTENT_ACCESS,
    GOSSIP_CONTENT_ADVERTISE,
    GOSSIP_PROVENANCE_REGISTER,
    GOSSIP_STORAGE_REQUEST,
    GossipProtocol,
)
from prsm.node.transport import WebSocketTransport
from prsm.node.identity import NodeIdentity
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.storage.erasure import ErasureError
from prsm.storage.models import ShardManifest

logger = logging.getLogger(__name__)

# Royalty rate bounds (FTNS per access)
MIN_ROYALTY_RATE = 0.001
MAX_ROYALTY_RATE = 0.1
DEFAULT_ROYALTY_RATE = 0.01

# Multi-level provenance splits
DERIVATIVE_CREATOR_SHARE = 0.70   # 70% to the derivative creator
SOURCE_CREATOR_SHARE = 0.25       # 25% to each source creator (split evenly)
NETWORK_FEE_SHARE = 0.05          # 5% network fee

# Default sharding threshold: 10MB - files larger than this will be sharded
DEFAULT_SHARDING_THRESHOLD = 10 * 1024 * 1024  # 10MB

# Item 5 (2026-05-06) — chunked embedding parameters.
# Long documents (papers, books, large datasets) get split into
# overlapping chunks before embedding, then mean-pooled into a single
# document-level vector. Tuned for OpenAI text-embedding-ada-002's
# 8191-token limit (~32k chars) with margin: 16k-char chunks fit
# comfortably and overlap covers chunk-boundary phrasing.
_MIN_EMBEDDING_CHARS = 50          # below this, no embedding generated
_CHUNK_SIZE_CHARS = 16_000          # max chars per chunk
_CHUNK_OVERLAP_CHARS = 2_000        # overlap between consecutive chunks
_MAX_CHUNKS = 50                    # cap so a 100MB plaintext upload
                                    # doesn't fan out to 6000 API calls


def _split_for_embedding(text: str) -> List[str]:
    """Split text into overlapping chunks for chunked embedding.

    Returns a list of substrings of length ≤ _CHUNK_SIZE_CHARS. Each
    consecutive pair of chunks shares _CHUNK_OVERLAP_CHARS chars to
    preserve cross-boundary semantic continuity. Hard-capped at
    _MAX_CHUNKS to bound API cost on adversarial uploads.

    For text ≤ _CHUNK_SIZE_CHARS this returns [text] unchanged so the
    common case is a single embedding call (no pooling cost).
    """
    if len(text) <= _CHUNK_SIZE_CHARS:
        return [text]
    chunks: List[str] = []
    stride = _CHUNK_SIZE_CHARS - _CHUNK_OVERLAP_CHARS
    if stride <= 0:
        # Defensive: if mis-configured so overlap >= chunk size, fall
        # back to non-overlapping chunking.
        stride = _CHUNK_SIZE_CHARS
    pos = 0
    while pos < len(text) and len(chunks) < _MAX_CHUNKS:
        chunks.append(text[pos:pos + _CHUNK_SIZE_CHARS])
        pos += stride
    return chunks


class _SemanticIndex:
    """In-memory semantic similarity index for near-duplicate detection on upload.

    Stores L2-normalised embeddings keyed by CID, persisted as JSON so the
    index survives node restarts.  Cosine similarity is computed via dot
    product (O(n) scan — acceptable for early-network scale; swap to a FAISS
    index once the corpus grows into the tens of thousands).

    Similarity thresholds
    ---------------------
    DERIVATIVE_THRESHOLD (0.92): content this similar is auto-registered as a
        derivative work; parent_cids is updated to include the matching CID and
        the 70/25/5 royalty split kicks in automatically.
    DUPLICATE_THRESHOLD (0.99): content this similar is an effective re-upload
        of the same material and is logged as such (upload still proceeds so the
        new creator gets their own provenance record at the derivative rate).
    """

    DERIVATIVE_THRESHOLD: float = 0.92
    DUPLICATE_THRESHOLD: float = 0.99

    # Default upper bound on remote embedding pulls per find_nearest()
    # call. Each pull is a network round-trip; this caps the worst-case
    # blocking time on a sub-threshold local search. The chosen value
    # is a soft heuristic and may need tuning once we have testnet
    # corpus data (Item 6 calibration territory).
    DEFAULT_MAX_REMOTE_PULLS_PER_QUERY: int = 32

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        *,
        model_id: Optional[str] = None,
        dht_client: Optional[Any] = None,
        peer_candidates_fn: Optional[
            Callable[[], Iterable[Tuple[str, str]]]
        ] = None,
        max_remote_pulls_per_query: Optional[int] = None,
    ) -> None:
        """Construct a semantic index, optionally wired to the
        EmbeddingDHT for cross-node escalation (PRSM-PROV-1 Item 3 T3.5).

        Args:
            persist_path: JSON path the in-memory index persists to.
            model_id: Local node's embedding model identifier (e.g.
                ``"sentence-transformers/all-MiniLM-L6-v2"``). Required
                for DHT escalation — the EmbeddingDHT keys by
                ``(content_hash, model_id)`` to enforce cross-model
                isolation. ``None`` disables DHT escalation entirely.
            dht_client: An ``EmbeddingDHTClient`` (duck-typed: any object
                exposing ``find_providers(content_hash, model_id) -> list``
                and ``fetch_embedding(provider, content_hash, model_id)
                -> EmbeddingResponse``). ``None`` disables escalation.
            peer_candidates_fn: A callable returning an iterable of
                ``(content_id, content_hash)`` pairs known via gossip
                whose embeddings could be pulled from the DHT.
                ``content_id`` is the local cache key (CID).
                ``content_hash`` is the on-chain identifier used as
                the DHT lookup key. ``None`` disables escalation.
            max_remote_pulls_per_query: Cap on remote embedding pulls
                per ``find_nearest()`` invocation. Defaults to
                ``DEFAULT_MAX_REMOTE_PULLS_PER_QUERY``.

        Trust model: this code path inherits the
        ``EmbeddingDHTClient`` invariant — it requires a verifier
        (creator-pubkey lookup + Ed25519 verifier). Fetched embeddings
        that fail signature verification are dropped silently with a
        debug log; we do NOT cache poisoned vectors. Failures of any
        kind degrade to local-only behavior — they never break the
        upload path.
        """
        # content_id → (normalised_embedding, creator_id)
        self._index: Dict[str, Tuple] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

        self._model_id = model_id
        self._dht_client = dht_client
        self._peer_candidates_fn = peer_candidates_fn
        self._max_remote_pulls_per_query = (
            max_remote_pulls_per_query
            if max_remote_pulls_per_query is not None
            else self.DEFAULT_MAX_REMOTE_PULLS_PER_QUERY
        )

    def __len__(self) -> int:
        return len(self._index)

    @property
    def dht_enabled(self) -> bool:
        """True iff DHT escalation is fully wired."""
        return (
            self._dht_client is not None
            and self._peer_candidates_fn is not None
            and self._model_id is not None
        )

    def store(self, content_id: str, embedding: "np.ndarray", creator_id: str) -> None:
        """Normalise and store an embedding keyed by content ID."""
        if not _HAS_NUMPY:
            return
        norm = float(np.linalg.norm(embedding))
        normalised = embedding / norm if norm > 0 else embedding
        self._index[content_id] = (normalised, creator_id)
        if self._persist_path:
            self._save()

    def find_nearest(self, embedding: "np.ndarray") -> Optional[Tuple[str, float, str]]:
        """Return (content_id, cosine_similarity, creator_id) for the
        closest stored embedding, or None if the index is empty.

        T3.5: If DHT escalation is wired (model_id + dht_client +
        peer_candidates_fn all set) and the local best similarity is
        below ``DERIVATIVE_THRESHOLD``, pull peer embeddings into the
        local cache and re-scan. Bounded by
        ``max_remote_pulls_per_query`` per call.
        """
        if not _HAS_NUMPY:
            return None
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            return None
        query = embedding / norm

        local_best = self._scan_index(query)

        # DHT escalation: only when local result is weak (no match or
        # sub-derivative). A strong local hit is already a sufficient
        # signal — no point burning network round-trips.
        if (
            self.dht_enabled
            and (local_best is None or local_best[1] < self.DERIVATIVE_THRESHOLD)
        ):
            pulled = self._pull_remote_embeddings()
            if pulled > 0:
                # Re-scan over the augmented index. A peer's embedding
                # may now be the new closest match.
                local_best = self._scan_index(query)

        return local_best

    def _scan_index(
        self, normalised_query: "np.ndarray"
    ) -> Optional[Tuple[str, float, str]]:
        """O(n) cosine-similarity scan against the local index.

        Caller is responsible for L2-normalising the query vector. The
        stored vectors are normalised at ``store()`` time so the scan
        is a pure dot product.
        """
        if not self._index:
            return None
        best_content_id, best_sim, best_creator = None, -1.0, ""
        for content_id, (stored, creator) in self._index.items():
            sim = float(np.dot(normalised_query, stored))
            if sim > best_sim:
                best_sim, best_content_id, best_creator = sim, content_id, creator
        return (best_content_id, best_sim, best_creator)

    def _pull_remote_embeddings(self) -> int:
        """Walk peer-candidates and pull missing embeddings from the
        DHT into the local index.

        Returns the count of embeddings actually inserted. Bounded by
        ``self._max_remote_pulls_per_query``. Idempotent — peers whose
        embedding is already cached locally are skipped.

        Failure modes (all logged at debug, all silently skipped — we
        NEVER raise out of this path because find_nearest() runs on the
        upload-critical path):
            - peer_candidates_fn raises
            - find_providers raises
            - no providers available
            - fetch_embedding raises (transport / parse error)
            - signature verification fails (poisoned response)
            - server returns NOT_FOUND under our model_id
            - vector decoding fails (malformed base64 / wrong dimension)
        """
        if not self.dht_enabled:
            return 0
        try:
            candidates = list(self._peer_candidates_fn())
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"peer_candidates_fn raised; skipping DHT escalation: {exc}"
            )
            return 0

        inserted = 0
        for content_id, content_hash in candidates:
            if inserted >= self._max_remote_pulls_per_query:
                break
            if not isinstance(content_id, str) or not content_id:
                continue
            if not isinstance(content_hash, str) or not content_hash:
                continue
            if content_id in self._index:
                continue

            record = self._fetch_one_embedding(content_hash)
            if record is None:
                continue

            try:
                vector_bytes = base64.b64decode(
                    record.vector_b64, validate=True,
                )
                if len(vector_bytes) != record.dimension * 4:
                    raise ValueError(
                        f"vector_b64 decodes to {len(vector_bytes)} bytes; "
                        f"expected dimension*4={record.dimension * 4}"
                    )
                vector = np.frombuffer(
                    vector_bytes, dtype=np.float32,
                ).copy()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    f"DHT vector decode failed for "
                    f"content_hash={content_hash[:18]}: {exc}"
                )
                continue

            self.store(content_id, vector, record.creator_id)
            inserted += 1

        if inserted:
            logger.info(
                f"EmbeddingDHT: pulled {inserted} peer embeddings "
                f"into local semantic index"
            )
        return inserted

    def _fetch_one_embedding(self, content_hash: str) -> Optional[Any]:
        """Find providers + try them in order until one returns a
        signature-verified embedding for ``(content_hash, model_id)``.

        Returns the verified ``EmbeddingResponse`` or None on any
        failure path (no providers, all providers failed, NOT_FOUND,
        signature verification failed for every candidate).
        """
        try:
            providers = self._dht_client.find_providers(
                content_hash, self._model_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"find_providers raised for content_hash="
                f"{content_hash[:18]}: {exc}"
            )
            return None

        for provider in providers:
            try:
                return self._dht_client.fetch_embedding(
                    provider, content_hash, self._model_id,
                )
            except Exception as exc:  # noqa: BLE001
                # EmbeddingNotFoundError, TransportFailureError,
                # SignatureVerificationError all land here. Try the
                # next provider — the DHT client raises on each
                # poisoning vector independently, so a malicious
                # peer cannot block fetching from honest peers.
                logger.debug(
                    f"fetch_embedding from {getattr(provider, 'node_id', '?')}"
                    f" failed: {exc}"
                )
                continue
        return None

    def _save(self) -> None:
        try:
            data = {
                content_id: {"embedding": emb.tolist(), "creator_id": creator}
                for content_id, (emb, creator) in self._index.items()
            }
            with open(self._persist_path, "w") as fh:
                json.dump(data, fh)
        except Exception as exc:
            logger.warning(f"Semantic index persist failed: {exc}")

    def _load(self) -> None:
        try:
            with open(self._persist_path) as fh:
                data = json.load(fh)
            self._index = {
                content_id: (np.array(entry["embedding"], dtype=np.float32), entry["creator_id"])
                for content_id, entry in data.items()
            }
            logger.info(f"Loaded semantic index: {len(self._index)} entries")
        except Exception as exc:
            logger.warning(f"Semantic index load failed: {exc}")


@dataclass
class UploadedContent:
    """Tracks content uploaded by this node."""
    content_id: str
    filename: str
    size_bytes: int
    content_hash: str       # SHA-256 of raw content
    creator_id: str
    created_at: float = field(default_factory=time.time)
    provenance_signature: str = ""
    royalty_rate: float = DEFAULT_ROYALTY_RATE
    parent_cids: List[str] = field(default_factory=list)
    access_count: int = 0
    total_royalties: float = 0.0
    # Sharding metadata
    is_sharded: bool = False
    manifest_content_id: Optional[str] = None  # content ID of the shard manifest if sharded
    total_shards: int = 0
    # Semantic fingerprint metadata
    embedding_id: Optional[str] = None          # "emb:<content_id>" when an embedding was generated
    near_duplicate_of: Optional[str] = None     # content ID of most-similar existing content
    near_duplicate_similarity: Optional[float] = None  # Cosine similarity to that content ID
    # Phase 1.2: canonical on-chain provenance hash
    # (keccak256(creator_address || sha3_256(file_bytes))).
    # None when the uploader was constructed without a creator_address.
    provenance_hash: Optional[str] = None  # 0x-prefixed hex
    # T6 (2026-05-05): tx hash from on-chain ProvenanceRegistry.registerContent
    # call. Set when the content was registered on chain via the wired
    # ProvenanceRegistryClient. None when on-chain registration was skipped
    # (no client wired) or failed (logged separately; doesn't block upload).
    provenance_tx_hash: Optional[str] = None  # 0x-prefixed hex


class ContentUploader:
    """Upload content to the PRSM network with provenance registration for royalties.

    Flow:
    1. Publish content via the proprietary BitTorrent layer
       (``ContentPublisher.publish``)
    2. Create provenance record (hash, creator, timestamp, signature)
    3. Gossip provenance registration to network
    4. Request storage replication from network peers
    5. Earn royalties when content is accessed
    """

    def __init__(
        self,
        identity: NodeIdentity,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        transport: Optional[WebSocketTransport] = None,
        content_index: Optional[Any] = None,
        ledger_sync: Optional[Any] = None,
        content_economy: Optional[Any] = None,
        sharding_threshold: int = DEFAULT_SHARDING_THRESHOLD,
        embedding_fn: Optional[Callable] = None,
        semantic_index_path: Optional[Path] = None,
        creator_address: Optional[str] = None,
        content_provider: Optional["ContentProvider"] = None,
        provenance_client: Optional[Any] = None,
        embedding_model_id: Optional[str] = None,
        embedding_dht_client: Optional[Any] = None,
        embedding_index: Optional[Any] = None,
        content_publisher: Optional[Any] = None,
        content_retriever: Optional[Any] = None,
        fingerprint_index_path: Optional[Path] = None,
    ):
        self.identity = identity
        self.gossip = gossip
        self.ledger = ledger
        self.transport = transport
        self.content_index = content_index  # For looking up parent content creators
        self.ledger_sync = ledger_sync      # For broadcasting transactions
        self.content_economy = content_economy  # For replication tracking (Phase 4)

        # Publish/fetch via the PRSM proprietary BitTorrent layer.
        self.content_publisher = content_publisher
        self.content_retriever = content_retriever

        # Phase 1.2: 0x address used to compute the canonical provenance_hash
        # for the on-chain registry. None disables on-chain provenance — the
        # local royalty path still works.
        self.creator_address = creator_address

        # Phase 1.3: optional back-reference to the node's ContentProvider so
        # the uploader can populate provider._local_content as soon as an
        # upload succeeds. Without this, register_local_content has no
        # production callers and the serve path returns not_found for every
        # uploaded CID. None disables the wiring (used by legacy unit tests).
        self._content_provider = content_provider

        # T6 (2026-05-05): on-chain ProvenanceRegistry client for
        # creator-bound content registration. None disables on-chain
        # registration — uploads still succeed locally but no on-chain
        # provenance record is created (so downstream royalty distribution
        # via RoyaltyDistributor cannot resolve creator from contentHash).
        self._provenance_client = provenance_client

        # Sharding configuration
        self.sharding_threshold = sharding_threshold

        # Semantic deduplication
        # embedding_fn: async (text: str) -> np.ndarray  — optional, skipped if None
        self._embedding_fn: Optional[Callable] = embedding_fn

        # T3.5 (PRSM-PROV-1): cross-node embedding gossip via the
        # EmbeddingDHT. The DHT keys by (content_hash, model_id), so
        # all three of (model_id, dht_client, content_index) must be
        # wired for cross-node escalation to engage; otherwise the
        # local-only path is used unchanged. ``embedding_dht_client``
        # is duck-typed so legacy unit tests can stub it; production
        # wires the real ``EmbeddingDHTClient``.
        self._embedding_model_id = embedding_model_id
        self._embedding_dht_client = embedding_dht_client
        # T3.6 (PRSM-PROV-1): the LocalEmbeddingIndex this node serves
        # to peers via the EmbeddingDHT. Populated by
        # _register_local_embedding() on every successful upload that
        # produced an embedding AND has an on-chain provenance_hash
        # (no anchor → no peer can verify our signature → MUST NOT
        # offer the embedding for cross-node dedup). None disables
        # this side of the wire — the local _SemanticIndex still works,
        # but peers won't find embeddings on this node via the DHT.
        self._embedding_index = embedding_index
        peer_candidates_fn = (
            self._make_peer_candidates_fn(content_index)
            if (embedding_dht_client is not None and content_index is not None)
            else None
        )
        self._semantic_index = _SemanticIndex(
            persist_path=semantic_index_path,
            model_id=embedding_model_id,
            dht_client=embedding_dht_client,
            peer_candidates_fn=peer_candidates_fn,
        )

        # PRSM-PROV-1 Item 4 T4.6 — per-kind binary fingerprint index.
        # Owns the four non-text-vector lanes (image-pHash,
        # audio-Chromaprint, video-multihash, structural). The
        # text-vector lane stays in ``_semantic_index`` above.
        # Backends are loaded lazily from prsm.data.fingerprints — any
        # missing optional dep yields a ``None`` backend slot which the
        # FingerprintIndex transparently treats as "no backend for this
        # kind", so a host without imagehash/pyacoustid/PyAV/h5py still
        # constructs a valid (text-vector-only) uploader.
        #
        # T4.9.next3: the same EmbeddingDHTClient + peer_candidates_fn
        # that drive ``_semantic_index`` escalation also drive fingerprint
        # escalation — the wire protocol partitions by message type, so
        # one client services both lanes. Reuse the ``peer_candidates_fn``
        # built above; it yields ``(cid, content_hash)`` tuples that are
        # equally valid keys for the fingerprint DHT.
        self._fingerprint_index = self._build_fingerprint_index(
            fingerprint_index_path,
            dht_client=embedding_dht_client,
            peer_candidates_fn=peer_candidates_fn,
        )

        self.uploaded_content: Dict[str, UploadedContent] = {}

        # Manifest tracking: manifest_content_id -> manifest data
        self.shard_manifests: Dict[str, Any] = {}

    async def close(self) -> None:
        pass

    def _register_local_embedding(
        self,
        embedding: "np.ndarray",
        provenance_hash_hex: Optional[str],
    ) -> None:
        """T3.6: register a successfully-generated embedding with the
        local EmbeddingDHT-backed store so peers can fetch it.

        Builds the canonical signing payload (matching the DHT wire
        format), signs it with the local NodeIdentity (the creator's
        Ed25519 private key), wraps it in a LocalEmbeddingRecord, and
        registers it with the LocalEmbeddingIndex.

        Skip conditions (returns silently — never breaks the upload):
          - no embedding_index wired at construction
          - no embedding_model_id wired (cross-model partition would
            be ambiguous)
          - no provenance_hash_hex (no on-chain anchor → no peer can
            verify our signature — see trust model in dht_client.py)
          - numpy not available
          - any registration error (logged at warning, not raised)

        Trust model: the signature here is by the *creator* of the
        content, which is this node's NodeIdentity. A peer fetching
        this embedding via the DHT verifies the signature against the
        on-chain creator pubkey (the PublisherKeyAnchor). A non-creator
        node serving someone else's record relays the bytes verbatim
        — it cannot forge the signature.
        """
        if self._embedding_index is None:
            return
        if not self._embedding_model_id:
            return
        if not provenance_hash_hex:
            return
        if not _HAS_NUMPY:
            return
        try:
            from prsm.network.embedding_dht.local_index import (
                LocalEmbeddingRecord,
            )
            from prsm.network.embedding_dht.protocol import (
                canonical_signing_payload,
            )
        except ImportError as exc:
            logger.debug(
                f"embedding_dht not importable; skipping local "
                f"embedding registration: {exc}"
            )
            return

        try:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.ndim != 1 or vec.shape[0] <= 0:
                logger.debug(
                    f"embedding has unexpected shape {vec.shape}; "
                    f"skipping DHT registration"
                )
                return
            vector_bytes = vec.tobytes(order="C")
            dimension = vec.shape[0]
            created_at = time.time()

            payload = canonical_signing_payload(
                content_hash=provenance_hash_hex,
                model_id=self._embedding_model_id,
                dimension=dimension,
                dtype="float32",
                vector_bytes=vector_bytes,
                created_at=created_at,
            )
            signature_b64 = self.identity.sign(payload)

            record = LocalEmbeddingRecord(
                content_hash=provenance_hash_hex,
                model_id=self._embedding_model_id,
                dimension=dimension,
                dtype="float32",
                vector_b64=base64.b64encode(vector_bytes).decode("ascii"),
                creator_id=self.identity.node_id,
                created_at=created_at,
                signature_b64=signature_b64,
            )
            self._embedding_index.register(record)
            logger.debug(
                f"EmbeddingDHT: registered local embedding for "
                f"content_hash={provenance_hash_hex[:18]} under "
                f"model_id={self._embedding_model_id!r}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"local embedding registration failed (will not block "
                f"upload): {type(exc).__name__}: {exc}"
            )

    def set_embedding_dht_client(self, client: Any) -> None:
        """T4.9.next4: late-bind the EmbeddingDHTClient after node startup.

        ``DHTNodeComponents`` only constructs the client at ``start()``
        time, but ``ContentUploader`` is built earlier (so the upload
        path is reachable before the network listener is up). This
        method threads the running client back into both lanes so
        cross-node dedup goes live the moment ``start()`` returns.

        Idempotent: callers can pass ``None`` to disable, and re-passing
        the same client is a no-op other than a fresh peer-candidates
        rebuild.

        Lockstep guarantee: same client + same ``peer_candidates_fn``
        feed both ``_semantic_index`` (text-vector lane) AND
        ``_fingerprint_index`` (binary lane). Either both lanes flip
        on, or both stay off — no hemispheric DHT.

        Trust model: the client itself enforces the verifier invariant
        at its own constructor (refuses to build without a real
        creator_pubkey_for + verify_signature). Anything reaching this
        method is already trust-enabled.
        """
        self._embedding_dht_client = client

        # Build the peer-candidates supplier lazily here. At ctor time
        # this was skipped because ``embedding_dht_client`` was None;
        # the supplier itself is still safe to build whenever a
        # content_index is available, so make it now.
        peer_candidates_fn = (
            self._make_peer_candidates_fn(self.content_index)
            if (client is not None and self.content_index is not None)
            else None
        )

        # Update both lanes' DHT plumbing in lockstep. Reaching into
        # the private fields is intentional — both indexes are owned
        # by this uploader and the field names are stable test surface.
        self._semantic_index._dht_client = client  # noqa: SLF001
        self._semantic_index._peer_candidates_fn = peer_candidates_fn  # noqa: SLF001
        self._fingerprint_index._dht_client = client  # noqa: SLF001
        self._fingerprint_index._peer_candidates_fn = peer_candidates_fn  # noqa: SLF001

    @staticmethod
    def _build_fingerprint_index(
        persist_path: Optional[Path],
        *,
        dht_client: Optional[Any] = None,
        peer_candidates_fn: Optional[Callable] = None,
    ) -> Any:
        """Construct a FingerprintIndex from whichever optional backend
        deps are installed.

        Each backend is imported lazily; missing deps simply omit that
        kind from the index. A node without any binary-fingerprint deps
        ends up with an empty FingerprintIndex (zero backends) — the
        text-vector path keeps working unchanged and binary uploads
        fall through to BYTE_HASH.

        T4.9.next3: ``dht_client`` + ``peer_candidates_fn`` enable DHT
        escalation in ``FingerprintIndex.find_nearest`` — both must be
        wired or escalation stays disabled and behavior is unchanged.
        The same ``EmbeddingDHTClient`` instance services both the
        embedding (text-vector) and fingerprint (binary) lanes; the
        wire protocol partitions by message type.
        """
        from prsm.data.fingerprints import (
            FingerprintIndex,
            FingerprintKind,
            ImageFingerprint,
            AudioFingerprint,
            VideoFingerprint,
            StructuralFingerprint,
        )
        backends: Dict[Any, Any] = {}
        if ImageFingerprint is not None:
            backends[FingerprintKind.IMAGE_PHASH] = ImageFingerprint()
        if AudioFingerprint is not None:
            backends[FingerprintKind.AUDIO_CHROMAPRINT] = AudioFingerprint()
        if VideoFingerprint is not None:
            backends[FingerprintKind.VIDEO_MULTIHASH] = VideoFingerprint()
        if StructuralFingerprint is not None:
            backends[FingerprintKind.STRUCTURAL] = StructuralFingerprint()
        return FingerprintIndex(
            backends=backends,
            persist_path=persist_path,
            dht_client=dht_client,
            peer_candidates_fn=peer_candidates_fn,
        )

    @staticmethod
    def _make_peer_candidates_fn(
        content_index: Any,
    ) -> Callable[[], Iterable[Tuple[str, str]]]:
        """Build a peer-candidates supplier for ``_SemanticIndex``.

        Walks the supplied ``ContentIndex._records`` map and yields
        ``(cid, provenance_hash)`` for records that:
          - have a non-empty ``provenance_hash`` (the on-chain
            identifier the EmbeddingDHT keys by),
          - declare an ``embedding_id`` (so we know the creator
            actually generated and gossiped a vector for this CID).

        The function is invoked lazily inside
        ``_SemanticIndex._pull_remote_embeddings`` so it picks up new
        peer advertisements between successive uploads without any
        wiring changes here.

        Records without ``provenance_hash`` are silently skipped: a
        peer that uploaded without ``creator_address`` produces no
        on-chain anchor, and the DHT has no way to verify that
        peer's signature, so we MUST NOT trust their embedding.
        """
        def _supplier() -> Iterable[Tuple[str, str]]:
            records = getattr(content_index, "_records", None)
            if records is None:
                return []
            out: List[Tuple[str, str]] = []
            for cid, record in records.items():
                provenance_hash = getattr(record, "provenance_hash", None)
                embedding_id = getattr(record, "embedding_id", None)
                if not provenance_hash or not embedding_id:
                    continue
                if not isinstance(cid, str) or not cid:
                    continue
                out.append((cid, provenance_hash))
            return out

        return _supplier

    def _register_with_provider(self, uploaded: "UploadedContent") -> None:
        """Forward a successfully uploaded record into ContentProvider._local_content.

        No-op when no ContentProvider was injected (backward compat with
        legacy unit tests that construct ContentUploader without a provider).

        The metadata dict mirrors what _handle_content_request dispatches to
        ContentEconomy.process_content_access. Note: the parent CID list is
        stored under ``parent_cids`` (not ``parent_cids``) so the
        existing reader at content_provider.py:582 finds it.
        """
        if self._content_provider is None:
            return
        self._content_provider.register_local_content(
            cid=uploaded.content_id,
            size_bytes=uploaded.size_bytes,
            content_hash=uploaded.content_hash,
            filename=uploaded.filename,
            metadata={
                "creator_id": uploaded.creator_id,
                "royalty_rate": uploaded.royalty_rate,
                "parent_cids": uploaded.parent_cids,
                "provenance_hash": uploaded.provenance_hash,
            },
        )

    def _register_on_chain(
        self,
        provenance_hash_hex: Optional[str],
        royalty_rate: float,
        cid: str,
    ) -> Optional[str]:
        """T6 — register content on the on-chain ProvenanceRegistry.

        Returns the tx hash on success, None on skip or failure.

        Skip conditions (returns None silently):
          - no provenance_client wired
          - no provenance_hash_hex computed (no creator_address)
          - content already registered on-chain (idempotent)

        Failure conditions (returns None + logs warning):
          - rpc / signing errors (BroadcastFailedError, etc.)
          - on-chain revert
          - any other exception during the call

        On-chain registration is best-effort: the upload itself does NOT
        fail if the on-chain call fails. The local provenance record + the
        ContentStore copy still exist; on-chain registration can be retried
        later via a backfill script.
        """
        if self._provenance_client is None:
            return None
        if not provenance_hash_hex:
            return None
        try:
            content_hash_bytes = bytes.fromhex(provenance_hash_hex.removeprefix("0x"))
            if len(content_hash_bytes) != 32:
                logger.warning(
                    f"provenance_hash_hex has invalid length "
                    f"({len(content_hash_bytes)} bytes, expected 32); "
                    f"skipping on-chain registration"
                )
                return None

            # Idempotent: skip if already registered. Saves gas + avoids
            # the inevitable revert on duplicate registerContent.
            if self._provenance_client.is_registered(content_hash_bytes):
                logger.info(
                    f"content_hash {provenance_hash_hex[:18]}… already "
                    f"registered on-chain; skipping"
                )
                return None

            # Convert float royalty_rate (e.g. 0.05 = 5%) to bps (500).
            # ProvenanceRegistry caps at MAX_ROYALTY_RATE_BPS = 9800.
            royalty_rate_bps = round(max(0.0, min(0.98, royalty_rate)) * 10000)
            metadata_uri = f"prsm-bt://{cid}"

            tx_hash, status = self._provenance_client.register_content(
                content_hash=content_hash_bytes,
                royalty_rate_bps=royalty_rate_bps,
                metadata_uri=metadata_uri,
            )
            logger.info(
                f"on-chain provenance registered: tx={tx_hash} "
                f"status={status} hash={provenance_hash_hex[:18]}…"
            )
            return tx_hash
        except Exception as exc:
            # Any failure (broadcast, revert, RPC error) — log but don't
            # propagate. The upload itself stays valid.
            logger.warning(
                f"on-chain provenance registration failed for cid {cid}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    async def _get_embedding(self, content: bytes) -> "Optional[np.ndarray]":
        """Attempt to generate a semantic embedding for content.

        Decodes content as UTF-8 text and calls the configured embedding_fn.
        Returns None if:
        - no embedding_fn was provided at construction
        - content is not valid text (binary blobs)
        - the text is too short to be meaningful (<50 chars)
        - the embedding call fails for any reason (non-fatal)

        Item 5 (2026-05-06) — chunked embedding for long docs:
        Previously this method truncated to 32k chars and embedded the
        prefix only, causing two versions of the same paper differing
        only in (e.g.) the abstract to register as separate originals
        despite 99% body overlap. Now: text > _CHUNK_SIZE_CHARS is
        split into overlapping chunks, each is embedded independently,
        and the chunk vectors are mean-pooled + L2-renormalized into a
        single document-level vector. This preserves the
        Optional[np.ndarray] return contract that _SemanticIndex
        expects (1D vector, normalised) while making long-document
        dedup correct.
        """
        if self._embedding_fn is None or not _HAS_NUMPY:
            return None
        try:
            text = content.decode("utf-8", errors="ignore").strip()
            if len(text) < _MIN_EMBEDDING_CHARS:
                return None

            chunks = _split_for_embedding(text)
            if len(chunks) == 1:
                # Common case: text fits in one chunk. Single API call.
                return await self._embedding_fn(chunks[0])

            # Long doc: embed each chunk + mean-pool. The aggregate is
            # the document-level fingerprint used for dedup.
            chunk_vectors = []
            for chunk in chunks:
                vec = await self._embedding_fn(chunk)
                if vec is None:
                    continue
                chunk_vectors.append(np.asarray(vec, dtype=np.float32))
            if not chunk_vectors:
                return None

            stacked = np.stack(chunk_vectors, axis=0)
            pooled = stacked.mean(axis=0)
            # Renormalize so _SemanticIndex's cosine-similarity scan
            # stays numerically stable. L2 norm of mean-pooled chunks
            # is < 1 in general, so this is necessary.
            norm = float(np.linalg.norm(pooled))
            if norm > 0:
                pooled = pooled / norm
            logger.debug(
                f"Chunked embedding: {len(chunks)} chunks pooled into "
                f"single {pooled.shape[0]}-dim vector"
            )
            return pooled
        except Exception as exc:
            logger.debug(f"Embedding generation skipped: {exc}")
            return None

    def _maybe_compute_binary_fingerprint(
        self,
        content: bytes,
        filename: str,
    ) -> Any:
        """Detect the content kind and run the matching binary backend.

        Returns a ``FingerprintRecord`` on success, or ``None`` when:
          - content type is text (caller already handled via embedding)
          - content type falls through to BYTE_HASH (no backend)
          - the kind has no registered backend (optional dep missing)
          - the backend can't fingerprint this specific input (truncated
            file, wrong codec, etc.)

        Lives outside ``_get_embedding`` so the dispatch logic stays
        observable in ``upload()`` without crossing into the embedding
        path's chunked/text-specific handling.
        """
        try:
            from prsm.data.fingerprints import (
                FingerprintKind,
                detect_content_kind,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"binary fingerprint dispatch skipped for {filename}: {exc}"
            )
            return None

        kind = detect_content_kind(content, filename=filename)
        # Text content is the embedding lane's job; BYTE_HASH has no
        # backend by definition. Both are no-ops for binary dedup.
        if kind in (FingerprintKind.TEXT_VECTOR, FingerprintKind.BYTE_HASH):
            return None

        return self._fingerprint_index.compute(content, kind, filename=filename)

    async def upload(
        self,
        content: bytes,
        filename: str = "untitled",
        metadata: Optional[Dict[str, Any]] = None,
        replicas: int = 3,
        royalty_rate: Optional[float] = None,
        parent_cids: Optional[List[str]] = None,
        force_shard: bool = False,
    ) -> Optional[UploadedContent]:
        """Upload content to storage and register provenance.

        Args:
            content: Raw bytes to upload
            filename: Display name for the content
            metadata: Optional metadata dict
            replicas: Number of storage replicas to request
            royalty_rate: FTNS earned per access (clamped to 0.001–0.1, default 0.01)
            parent_cids: content IDs of source material this content derives from
            force_shard: If True, force sharding regardless of file size

        Returns:
            UploadedContent with content ID and provenance info, or None on failure
        """
        # Clamp royalty rate to bounds
        rate = royalty_rate if royalty_rate is not None else DEFAULT_ROYALTY_RATE
        rate = max(MIN_ROYALTY_RATE, min(MAX_ROYALTY_RATE, rate))

        content_hash = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        parents = parent_cids or []

        # Phase 1.2: compute the canonical creator-bound provenance hash
        # so the on-chain registry and content_economy can find each other.
        provenance_hash_hex: Optional[str] = None
        if self.creator_address:
            try:
                from prsm.economy.web3.provenance_registry import compute_content_hash
                ph = compute_content_hash(self.creator_address, content)
                provenance_hash_hex = "0x" + ph.hex()
            except Exception as exc:
                logger.warning(
                    f"failed to compute provenance_hash for {filename}: {exc}"
                )

        # ── Semantic deduplication ────────────────────────────────────────────
        # Generate an embedding and check for near-duplicates *before* committing
        # to the content store so we can auto-register derivative relationships early.
        embedding = await self._get_embedding(content)
        near_dup_cid: Optional[str] = None
        near_dup_sim: Optional[float] = None

        if embedding is not None:
            match = self._semantic_index.find_nearest(embedding)
            if match is not None:
                match_cid, match_sim, _match_creator = match
                if match_sim >= _SemanticIndex.DERIVATIVE_THRESHOLD:
                    if match_sim >= _SemanticIndex.DUPLICATE_THRESHOLD:
                        logger.warning(
                            f"Near-exact duplicate detected for '{filename}': "
                            f"CID {match_cid[:16]}... (similarity={match_sim:.4f}). "
                            f"Registering as derivative work."
                        )
                    else:
                        logger.info(
                            f"Semantic near-duplicate found for '{filename}': "
                            f"CID {match_cid[:16]}... (similarity={match_sim:.4f}). "
                            f"Registering as derivative work."
                        )
                    near_dup_cid = match_cid
                    near_dup_sim = match_sim
                    # Auto-prepend matching CID as a parent so royalty splits apply
                    if match_cid not in parents:
                        parents = [match_cid] + parents
        # ─────────────────────────────────────────────────────────────────────

        # ── Binary-fingerprint deduplication (PRSM-PROV-1 Item 4 T4.7) ──
        # When the embedding path returned None — i.e. the content isn't
        # text — try the binary fingerprint backends. Image/audio/video/
        # structural are the lanes; for unrecognised content types
        # ``detect_content_kind`` falls through to BYTE_HASH which has no
        # backend, and dedup is skipped (caller still gets a fresh
        # provenance record at the original-creator rate).
        binary_fingerprint_record = None
        if embedding is None and near_dup_cid is None:
            binary_fingerprint_record = self._maybe_compute_binary_fingerprint(
                content, filename,
            )
            if binary_fingerprint_record is not None:
                bin_match = self._fingerprint_index.find_nearest(
                    binary_fingerprint_record,
                )
                if bin_match is not None:
                    dup_threshold = self._fingerprint_index.duplicate_threshold(
                        bin_match.kind,
                    )
                    deriv_threshold = self._fingerprint_index.derivative_threshold(
                        bin_match.kind,
                    )
                    if bin_match.similarity >= deriv_threshold:
                        if bin_match.similarity >= dup_threshold:
                            logger.warning(
                                f"Binary near-duplicate (kind={bin_match.kind.value}) "
                                f"for '{filename}': CID {bin_match.content_id[:16]}... "
                                f"(similarity={bin_match.similarity:.4f}). "
                                f"Registering as derivative work."
                            )
                        else:
                            logger.info(
                                f"Binary near-duplicate (kind={bin_match.kind.value}) "
                                f"for '{filename}': CID {bin_match.content_id[:16]}... "
                                f"(similarity={bin_match.similarity:.4f}). "
                                f"Registering as derivative work."
                            )
                        near_dup_cid = bin_match.content_id
                        near_dup_sim = bin_match.similarity
                        if bin_match.content_id not in parents:
                            parents = [bin_match.content_id] + parents
        # ─────────────────────────────────────────────────────────────────────

        # Check if content should be sharded
        should_shard = force_shard or size_bytes > self.sharding_threshold

        if should_shard:
            # Use sharding for large files
            logger.info(
                f"File {filename} ({size_bytes} bytes) exceeds threshold "
                f"({self.sharding_threshold} bytes), using sharding"
            )
            return await self._upload_with_sharding(
                content=content,
                filename=filename,
                metadata=metadata,
                replicas=replicas,
                royalty_rate=rate,
                parent_cids=parents,
                content_hash=content_hash,
                embedding=embedding,
                near_dup_cid=near_dup_cid,
                near_dup_sim=near_dup_sim,
                provenance_hash_hex=provenance_hash_hex,
                binary_fingerprint_record=binary_fingerprint_record,
            )

        # Standard monolithic upload for small files. The cid is now the
        # BitTorrent infohash returned by ContentPublisher (Tier A).
        cid = await self._publish_content(content, filename, provenance_hash_hex)
        if not cid:
            logger.error(f"Failed to publish {filename}")
            return None

        embedding_id = f"emb:{cid}" if embedding is not None else None

        # T6: on-chain provenance registration. Best-effort; doesn't block.
        provenance_tx_hash = self._register_on_chain(
            provenance_hash_hex=provenance_hash_hex,
            royalty_rate=rate,
            cid=cid,
        )

        # Create provenance record
        # NOTE (Phase 1.3 Task 3d): wire-format keys must match reader
        # expectations — `cid` / `parent_cids`, not `content_id` /
        # `parent_cids`. See content_index._on_content_advertise,
        # _on_provenance_register, local_ledger.upsert_provenance, and
        # storage_provider._on_storage_request.
        provenance_data = {
            "cid": cid,
            "content_hash": content_hash,
            "creator_id": self.identity.node_id,
            "creator_public_key": self.identity.public_key_b64,
            "filename": filename,
            "size_bytes": size_bytes,
            "created_at": time.time(),
            "metadata": metadata or {},
            "royalty_rate": rate,
            "parent_cids": parents,
            "is_sharded": False,
            "embedding_id": embedding_id,
            "near_duplicate_of": near_dup_cid,
            "provenance_hash": provenance_hash_hex,
        }
        provenance_bytes = json.dumps(provenance_data, sort_keys=True).encode()
        provenance_signature = self.identity.sign(provenance_bytes)

        uploaded = UploadedContent(
            content_id=cid,
            filename=filename,
            size_bytes=size_bytes,
            content_hash=content_hash,
            creator_id=self.identity.node_id,
            provenance_signature=provenance_signature,
            royalty_rate=rate,
            parent_cids=parents,
            is_sharded=False,
            embedding_id=embedding_id,
            near_duplicate_of=near_dup_cid,
            near_duplicate_similarity=near_dup_sim,
            provenance_hash=provenance_hash_hex,
            provenance_tx_hash=provenance_tx_hash,
        )
        self.uploaded_content[cid] = uploaded
        self._register_with_provider(uploaded)  # Phase 1.3: populate provider._local_content
        await self._persist_provenance(uploaded)  # Persist to DB

        # Register embedding so future uploads can be checked against this one
        if embedding is not None:
            self._semantic_index.store(cid, embedding, self.identity.node_id)
            # T3.6: also register with the cross-node EmbeddingDHT so
            # peers can fetch this embedding for their own dedup check.
            self._register_local_embedding(embedding, provenance_hash_hex)

        # T4.7: Register the binary fingerprint (if any) so future
        # uploads of similar binary content land on this CID.
        if binary_fingerprint_record is not None:
            self._fingerprint_index.store(
                cid, binary_fingerprint_record, self.identity.node_id,
            )

        # Gossip provenance registration
        await self.gossip.publish(GOSSIP_PROVENANCE_REGISTER, {
            **provenance_data,
            "signature": provenance_signature,
        })

        # Advertise content availability to the network
        await self.gossip.publish(GOSSIP_CONTENT_ADVERTISE, {
            "cid": cid,
            "filename": filename,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "creator_id": self.identity.node_id,
            "provider_id": self.identity.node_id,
            "created_at": provenance_data["created_at"],
            "metadata": metadata or {},
            "royalty_rate": rate,
            "parent_cids": parents,
            "embedding_id": embedding_id,
            "provenance_hash": provenance_hash_hex,
        })

        # Request storage replication
        if replicas > 0:
            await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                "cid": cid,
                "size_bytes": size_bytes,
                "requester_id": self.identity.node_id,
                "replicas_needed": replicas,
            })

        # Track replication status via ContentEconomy (Phase 4)
        if self.content_economy:
            await self.content_economy.track_content_upload(
                content_id=cid,
                size_bytes=size_bytes,
                replicas_requested=replicas,
            )

        dup_msg = f", near_dup={near_dup_cid[:12]}... ({near_dup_sim:.3f})" if near_dup_cid else ""
        logger.info(
            f"Uploaded {filename} ({size_bytes} bytes) -> {cid}, "
            f"royalty={rate} FTNS/access, parents={len(parents)}, replicas={replicas}{dup_msg}"
        )
        return uploaded

    async def _upload_with_sharding(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]],
        replicas: int,
        royalty_rate: float,
        parent_cids: List[str],
        content_hash: str,
        embedding: "Optional[np.ndarray]" = None,
        near_dup_cid: Optional[str] = None,
        near_dup_sim: Optional[float] = None,
        provenance_hash_hex: Optional[str] = None,
        binary_fingerprint_record: Optional[Any] = None,
    ) -> Optional[UploadedContent]:
        """Upload large content using sharding.

        This method delegates to ContentSharder for large files, handling
        the manifest and provenance registration.

        Args:
            content: Raw bytes to upload
            filename: Display name for the content
            metadata: Optional metadata dict
            replicas: Number of storage replicas to request
            royalty_rate: FTNS earned per access
            parent_cids: CIDs of source material this content derives from
            content_hash: Pre-calculated SHA-256 hash of content

        Returns:
            UploadedContent with manifest CID and provenance info, or None on failure
        """
        try:
            # Get the content sharder
            sharder = await self._get_content_sharder()

            # Shard and upload the content
            manifest, manifest_cid = await sharder.shard_content(
                content=content,
                original_cid=None,  # We don't have the original CID yet
                metadata={
                    "filename": filename,
                    "uploader_metadata": metadata or {},
                    "parent_cids": parent_cids,
                },
            )

            # Store the manifest for later retrieval
            self.shard_manifests[manifest_cid] = manifest

            size_bytes = len(content)

            # T6: on-chain provenance registration (sharded variant).
            # Same best-effort semantics as the monolithic path.
            sharded_provenance_tx_hash = self._register_on_chain(
                provenance_hash_hex=provenance_hash_hex,
                royalty_rate=royalty_rate,
                cid=manifest_cid,
            )

            # Create provenance record for the sharded content
            # The manifest content ID serves as the primary identifier for sharded content
            embedding_id = f"emb:{manifest_cid}" if embedding is not None else None
            # Phase 1.3 Task 3d: canonical wire-format keys (`cid`,
            # `parent_cids`) so ContentIndex / local_ledger actually upsert
            # the record instead of silently dropping it on empty-string
            # defaults.
            provenance_data = {
                "cid": manifest_cid,  # Use manifest content ID as the primary identifier
                "content_hash": content_hash,
                "creator_id": self.identity.node_id,
                "creator_public_key": self.identity.public_key_b64,
                "filename": filename,
                "size_bytes": size_bytes,
                "created_at": time.time(),
                "metadata": metadata or {},
                "royalty_rate": royalty_rate,
                "parent_cids": parent_cids,
                "is_sharded": True,
                "total_shards": manifest.total_shards,
                "shard_size": manifest.shard_size,
                "embedding_id": embedding_id,
                "near_duplicate_of": near_dup_cid,
                "provenance_hash": provenance_hash_hex,
            }
            provenance_bytes = json.dumps(provenance_data, sort_keys=True).encode()
            provenance_signature = self.identity.sign(provenance_bytes)

            uploaded = UploadedContent(
                content_id=manifest_cid,  # Use manifest content ID as primary identifier
                filename=filename,
                size_bytes=size_bytes,
                content_hash=content_hash,
                creator_id=self.identity.node_id,
                provenance_signature=provenance_signature,
                royalty_rate=royalty_rate,
                parent_cids=parent_cids,
                is_sharded=True,
                manifest_content_id=manifest_cid,
                total_shards=manifest.total_shards,
                embedding_id=embedding_id,
                near_duplicate_of=near_dup_cid,
                near_duplicate_similarity=near_dup_sim,
                provenance_hash=provenance_hash_hex,
                provenance_tx_hash=sharded_provenance_tx_hash,
            )
            self.uploaded_content[manifest_cid] = uploaded
            self._register_with_provider(uploaded)  # Phase 1.3: populate provider._local_content
            await self._persist_provenance(uploaded)  # Persist to DB

            # Register embedding for future deduplication checks
            if embedding is not None:
                self._semantic_index.store(manifest_cid, embedding, self.identity.node_id)
                # T3.6: also register with the cross-node EmbeddingDHT.
                self._register_local_embedding(embedding, provenance_hash_hex)

            # T4.7: Register the binary fingerprint (if any) for the
            # sharded path too, so a re-upload of the same image/audio/
            # video/structural blob lands on this manifest_cid.
            if binary_fingerprint_record is not None:
                self._fingerprint_index.store(
                    manifest_cid,
                    binary_fingerprint_record,
                    self.identity.node_id,
                )

            # Gossip provenance registration
            await self.gossip.publish(GOSSIP_PROVENANCE_REGISTER, {
                **provenance_data,
                "signature": provenance_signature,
            })

            # Advertise content availability to the network
            await self.gossip.publish(GOSSIP_CONTENT_ADVERTISE, {
                "cid": manifest_cid,
                "filename": filename,
                "size_bytes": size_bytes,
                "content_hash": content_hash,
                "creator_id": self.identity.node_id,
                "provider_id": self.identity.node_id,
                "created_at": provenance_data["created_at"],
                "metadata": metadata or {},
                "royalty_rate": royalty_rate,
                "parent_cids": parent_cids,
                "is_sharded": True,
                "total_shards": manifest.total_shards,
                "embedding_id": embedding_id,
                "provenance_hash": provenance_hash_hex,
            })

            # Request storage replication for each shard
            if replicas > 0:
                # Request replication for the manifest
                await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                    "cid": manifest_cid,
                    "size_bytes": len(manifest.to_json()),
                    "requester_id": self.identity.node_id,
                    "replicas_needed": replicas,
                })

                # Request replication for each shard
                for shard_info in manifest.shards:
                    await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                        "cid": shard_info.content_id,
                        "size_bytes": shard_info.size,
                        "requester_id": self.identity.node_id,
                        "replicas_needed": replicas,
                    })

            logger.info(
                f"Uploaded {filename} ({size_bytes} bytes) with sharding -> "
                f"manifest={manifest_cid}, shards={manifest.total_shards}, "
                f"royalty={royalty_rate} FTNS/access, parents={len(parent_cids)}"
            )
            return uploaded

        except ErasureError as e:
            # The Phase 7-storage refactor renamed ShardingError → ErasureError
            # (see prsm/storage/erasure.py:65). Old name was removed; this
            # except was dead code that would have NameError'd if the try
            # block raised. Caught at the ErasureError base so any of the
            # four concrete subclasses (InsufficientShardsError,
            # DuplicateShardError, CorruptShardError, PayloadChecksumError)
            # trigger the monolithic fallback.
            logger.error(f"Sharding failed for {filename}: {e}")
            # Fall back to monolithic upload via ContentPublisher.
            logger.info(f"Falling back to monolithic upload for {filename}")
            cid = await self._publish_content(
                content, filename, provenance_hash_hex
            )
            if not cid:
                logger.error(f"Failed to publish {filename} (fallback)")
                return None

            size_bytes = len(content)
            # Phase 1.3 Task 3d: canonical wire-format keys (`cid`,
            # `parent_cids`) — reader side expects these.
            provenance_data = {
                "cid": cid,
                "content_hash": content_hash,
                "creator_id": self.identity.node_id,
                "creator_public_key": self.identity.public_key_b64,
                "filename": filename,
                "size_bytes": size_bytes,
                "created_at": time.time(),
                "metadata": metadata or {},
                "royalty_rate": royalty_rate,
                "parent_cids": parent_cids,
                "is_sharded": False,
                "sharding_failed": True,
                "provenance_hash": provenance_hash_hex,
            }
            provenance_bytes = json.dumps(provenance_data, sort_keys=True).encode()
            provenance_signature = self.identity.sign(provenance_bytes)

            uploaded = UploadedContent(
                content_id=cid,
                filename=filename,
                size_bytes=size_bytes,
                content_hash=content_hash,
                creator_id=self.identity.node_id,
                provenance_signature=provenance_signature,
                royalty_rate=royalty_rate,
                parent_cids=parent_cids,
                is_sharded=False,
                provenance_hash=provenance_hash_hex,
            )
            self.uploaded_content[cid] = uploaded
            self._register_with_provider(uploaded)  # Phase 1.3: populate provider._local_content
            await self._persist_provenance(uploaded)  # Persist to DB

            # Still advertise and request replication for fallback
            await self.gossip.publish(GOSSIP_PROVENANCE_REGISTER, {
                **provenance_data,
                "signature": provenance_signature,
            })
            await self.gossip.publish(GOSSIP_CONTENT_ADVERTISE, {
                "cid": cid,
                "filename": filename,
                "size_bytes": size_bytes,
                "content_hash": content_hash,
                "creator_id": self.identity.node_id,
                "provider_id": self.identity.node_id,
                "created_at": provenance_data["created_at"],
                "metadata": metadata or {},
                "royalty_rate": royalty_rate,
                "parent_cids": parent_cids,
                "provenance_hash": provenance_hash_hex,
            })
            if replicas > 0:
                await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                    "cid": cid,
                    "size_bytes": size_bytes,
                    "requester_id": self.identity.node_id,
                    "replicas_needed": replicas,
                })

            return uploaded

        except Exception as e:
            logger.error(f"Unexpected error during sharding of {filename}: {e}")
            return None

    async def upload_json(
        self,
        data: Any,
        filename: str = "data.json",
        metadata: Optional[Dict[str, Any]] = None,
        replicas: int = 3,
        royalty_rate: Optional[float] = None,
        parent_cids: Optional[List[str]] = None,
    ) -> Optional[UploadedContent]:
        """Upload JSON-serializable data to content storage."""
        content = json.dumps(data, indent=2).encode()
        return await self.upload(content, filename, metadata, replicas, royalty_rate, parent_cids)

    async def upload_text(
        self,
        text: str,
        filename: str = "document.txt",
        metadata: Optional[Dict[str, Any]] = None,
        replicas: int = 3,
        royalty_rate: Optional[float] = None,
        parent_cids: Optional[List[str]] = None,
    ) -> Optional[UploadedContent]:
        """Upload text content to content storage."""
        return await self.upload(text.encode("utf-8"), filename, metadata, replicas, royalty_rate, parent_cids)

    async def record_access(self, cid: str, accessor_id: str) -> None:
        """Record that content was accessed, distributing royalties.

        If the content has parent CIDs (derivative work), royalties are split:
        - 70% to the derivative creator (this node)
        - 25% split among source material creators
        - 5% network fee

        If no parents, the full royalty goes to the creator.
        """
        content = self.uploaded_content.get(cid)
        if not content:
            return
        if accessor_id == self.identity.node_id:
            return  # No self-royalties

        content.access_count += 1
        total_royalty = content.royalty_rate

        if content.parent_cids and self.content_index:
            # Multi-level provenance: split royalties
            await self._distribute_multilevel_royalty(content, total_royalty, accessor_id)
        else:
            # Single creator: full royalty
            try:
                tx = await self.ledger.credit(
                    wallet_id=self.identity.node_id,
                    amount=total_royalty,
                    tx_type=TransactionType.CONTENT_ROYALTY,
                    description=f"Royalty for {cid[:12]}... access by {accessor_id[:8]}",
                )
                content.total_royalties += total_royalty
                await self._maybe_broadcast(tx)
            except Exception as e:
                logger.error(f"Royalty credit failed: {e}")

            # Wire to platform FTNS (non-blocking, own error handling)
            await self._platform_royalty_transfer(
                from_user=accessor_id,
                to_user=self.identity.node_id,
                amount=total_royalty,
                content_id=cid,
            )

        # Persist updated access stats to DB (non-blocking)
        await self._update_provenance_access(cid, content.royalty_rate)

    async def _distribute_multilevel_royalty(
        self, content: UploadedContent, total_royalty: float, accessor_id: str
    ) -> None:
        """Split royalty among derivative creator, source creators, and network."""
        cid = content.content_id
        derivative_share = total_royalty * DERIVATIVE_CREATOR_SHARE
        source_pool = total_royalty * SOURCE_CREATOR_SHARE
        network_fee = total_royalty * NETWORK_FEE_SHARE

        # Credit derivative creator (this node)
        try:
            tx = await self.ledger.credit(
                wallet_id=self.identity.node_id,
                amount=derivative_share,
                tx_type=TransactionType.CONTENT_ROYALTY,
                description=f"Derivative royalty for {cid[:12]}... (70%)",
            )
            content.total_royalties += derivative_share
            await self._maybe_broadcast(tx)
        except Exception as e:
            logger.error(f"Derivative royalty credit failed: {e}")

        # Wire derivative share to platform FTNS
        await self._platform_royalty_transfer(
            from_user=accessor_id,
            to_user=self.identity.node_id,
            amount=derivative_share,
            content_id=cid,
            description=f"Derivative royalty 70%: {cid[:12]}...",
        )

        # Credit source material creators (split evenly among parents)
        parent_creators = self._resolve_parent_creators(content.parent_cids)
        if parent_creators:
            per_parent = source_pool / len(parent_creators)
            for parent_creator_id in parent_creators:
                if parent_creator_id == self.identity.node_id:
                    # Source creator is also on this node — credit locally
                    try:
                        await self.ledger.credit(
                            wallet_id=self.identity.node_id,
                            amount=per_parent,
                            tx_type=TransactionType.CONTENT_ROYALTY,
                            description=f"Source royalty for {cid[:12]}... (25%/{len(parent_creators)})",
                        )
                        content.total_royalties += per_parent
                    except Exception as e:
                        logger.error(f"Source royalty credit failed: {e}")

                    # Wire source share to platform FTNS
                    await self._platform_royalty_transfer(
                        from_user=accessor_id,
                        to_user=self.identity.node_id,
                        amount=per_parent,
                        content_id=cid,
                        description=f"Source royalty 25%/{len(parent_creators)}: {cid[:12]}...",
                    )
                # Remote source creators get credited when they receive the
                # GOSSIP_CONTENT_ACCESS message on their own node
        else:
            # No resolvable parents — derivative creator gets the source pool too
            try:
                await self.ledger.credit(
                    wallet_id=self.identity.node_id,
                    amount=source_pool,
                    tx_type=TransactionType.CONTENT_ROYALTY,
                    description=f"Unclaimed source royalty for {cid[:12]}...",
                )
                content.total_royalties += source_pool
            except Exception as e:
                logger.error(f"Unclaimed source royalty credit failed: {e}")

            # Wire unclaimed source pool to platform FTNS
            await self._platform_royalty_transfer(
                from_user=accessor_id,
                to_user=self.identity.node_id,
                amount=source_pool,
                content_id=cid,
                description=f"Unclaimed source royalty 25%: {cid[:12]}...",
            )

        # Network fee
        try:
            await self.ledger.credit(
                wallet_id="system",
                amount=network_fee,
                tx_type=TransactionType.CONTENT_ROYALTY,
                description=f"Network fee for {cid[:12]}... access",
            )
        except Exception as e:
            logger.error(f"Network fee credit failed: {e}")

    def _resolve_parent_creators(self, parent_cids: List[str]) -> List[str]:
        """Look up the creator node IDs for parent content IDs via the content index."""
        creators = []
        if not self.content_index:
            return creators
        for pcid in parent_cids:
            record = self.content_index.lookup(pcid)
            if record and record.creator_id:
                creators.append(record.creator_id)
        return creators

    # ── Network content serving ─────────────────────────────────

    def start(self) -> None:
        """Register gossip subscriptions.

        Phase 1.3 Task 3b: the uploader is no longer a direct-message
        server or client — ContentProvider owns the canonical serve
        path and ContentProvider.request_content / bt_requester are
        the production client paths. The uploader now only subscribes
        to GOSSIP_CONTENT_ACCESS so it can credit source-creator
        royalty shares when derivative content is accessed elsewhere.
        """
        self.gossip.subscribe(GOSSIP_CONTENT_ACCESS, self._on_content_access)
        logger.info("Content uploader started — listening for gossip access events")

    # Phase 1.3 Task 3b: _handle_content_request retired.
    # ContentProvider._handle_content_request at content_provider.py
    # is now the canonical serve path. The uploader was a vestigial
    # duplicate server that used a legacy response shape (found=bool)
    # and ran a duplicate local-ledger royalty distribution via
    # record_access/_distribute_multilevel_royalty. Phase 1.3 Task 1
    # activated the duplicate by populating ContentProvider._local_content
    # for the first time; Task 3b removes the dead copy.
    #
    # Phase 1.3 Task 3b follow-up: the client-side direct-message
    # surface (request_content, _request_from_provider,
    # _handle_content_response, _send_direct, _on_direct_message,
    # _pending_requests) is also retired. It had zero production
    # callers and its response parser used the legacy
    # `response.get("found", False)` shape that only the retired
    # server produced — so it was both dead and broken after Task 3b
    # moved the server side to ContentProvider. The real client path
    # is ContentProvider.request_content.

    async def _publish_content(
        self,
        content: bytes,
        filename: str,
        provenance_hash_hex: Optional[str],
    ) -> Optional[str]:
        """Publish *content* via the proprietary BitTorrent layer.

        Returns the torrent infohash on success or None on failure.
        The returned infohash is used as the content identifier
        (``cid`` in provenance records, ``contentHash`` in on-chain
        RoyaltyDistributor calls).
        """
        if self.content_publisher is None:
            logger.error(
                "ContentUploader.content_publisher is None — cannot publish "
                "%s. Wire content_publisher in the constructor.",
                filename,
            )
            return None
        try:
            result = await self.content_publisher.publish(
                content,
                provenance_id=provenance_hash_hex or "",
                name=filename,
            )
            return result.torrent_infohash
        except Exception as e:
            logger.error(f"Content publish failed for {filename}: {e}")
            return None

    async def _fetch_content(self, cid: str) -> Optional[bytes]:
        """Fetch content bytes by torrent infohash via the BitTorrent layer.

        Returns the content bytes on success or None on failure.
        """
        if self.content_retriever is None:
            logger.error(
                "ContentUploader.content_retriever is None — cannot fetch "
                "%s. Wire content_retriever in the constructor.",
                cid,
            )
            return None
        try:
            return await self.content_retriever.fetch(cid)
        except Exception as e:
            logger.error(f"Content fetch failed for {cid}: {e}")
            return None

    async def _on_content_access(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Credit royalty when we are the original creator or a source creator.

        This handles two cases:
        1. We are the direct creator of the accessed content — call record_access()
           if we have a local record (i.e., the access was served by another node).
        2. We are a *source* creator — our content was used as a parent for a
           derivative work. We earn the source creator share (25% / num_parents).
        """
        if origin == self.identity.node_id:
            return  # Already processed locally

        content_id = data.get("content_id", "")
        accessor_id = data.get("accessor_id", "")
        creator_id = data.get("creator_id", "")
        royalty_rate = data.get("royalty_rate", DEFAULT_ROYALTY_RATE)
        parent_cids = data.get("parent_cids", [])

        if not content_id or not accessor_id:
            return

        # Case 1: We are the direct creator and have a local record
        if creator_id == self.identity.node_id and content_id in self.uploaded_content:
            await self.record_access(content_id, accessor_id)
            return

        # Case 2: We are a source creator for a derivative work
        if parent_cids:
            my_parent_cids = [
                pcid for pcid in parent_cids
                if pcid in self.uploaded_content
            ]
            if my_parent_cids:
                source_pool = royalty_rate * SOURCE_CREATOR_SHARE
                # Count total parents to split evenly
                per_parent = source_pool / len(parent_cids)
                source_royalty = per_parent * len(my_parent_cids)
                try:
                    await self.ledger.credit(
                        wallet_id=self.identity.node_id,
                        amount=source_royalty,
                        tx_type=TransactionType.CONTENT_ROYALTY,
                        description=f"Source royalty for derivative {content_id[:12]}... ({len(my_parent_cids)} parent(s))",
                    )
                    for pcid in my_parent_cids:
                        self.uploaded_content[pcid].total_royalties += per_parent
                    logger.info(f"Source royalty earned: {source_royalty:.4f} FTNS for derivative {content_id[:12]}...")
                except Exception as e:
                    logger.error(f"Source royalty credit failed: {e}")

                # Wire gossip source royalty to platform FTNS
                await self._platform_royalty_transfer(
                    from_user=accessor_id,
                    to_user=self.identity.node_id,
                    amount=source_royalty,
                    content_id=content_id,
                    description=f"Source royalty (gossip) for derivative {content_id[:12]}...",
                )

    async def _maybe_broadcast(self, tx) -> None:
        """Broadcast a transaction via ledger_sync if available."""
        if self.ledger_sync:
            try:
                await self.ledger_sync.broadcast_transaction(tx)
            except Exception as e:
                logger.debug(f"Transaction broadcast failed: {e}")


    async def _persist_provenance(self, uploaded: "UploadedContent") -> None:
        """
        Persist a provenance record to the platform database.

        Non-blocking: if the DB write fails (no connection, schema missing,
        etc.) the in-memory uploaded_content dict remains authoritative and
        the upload is still considered successful.
        """
        try:
            from prsm.core.database import ProvenanceQueries
            record = {
                "content_id": uploaded.content_id,
                "filename": uploaded.filename,
                "size_bytes": uploaded.size_bytes,
                "content_hash": uploaded.content_hash,
                "creator_id": uploaded.creator_id,
                "provenance_signature": uploaded.provenance_signature,
                "royalty_rate": uploaded.royalty_rate,
                "parent_cids": uploaded.parent_cids,
                "access_count": uploaded.access_count,
                "total_royalties": uploaded.total_royalties,
                "is_sharded": uploaded.is_sharded,
                "manifest_content_id": uploaded.manifest_content_id,
                "total_shards": uploaded.total_shards,
                "embedding_id": uploaded.embedding_id,
                "near_duplicate_of": uploaded.near_duplicate_of,
                "near_duplicate_similarity": uploaded.near_duplicate_similarity,
                # Phase 1.3 Task 2: canonical on-chain provenance hash —
                # must survive node restarts so royalty routing stays
                # on-chain across process lifetimes.
                "provenance_hash": uploaded.provenance_hash,
                "created_at": uploaded.created_at,
            }
            success = await ProvenanceQueries.upsert_provenance(record)
            if success:
                logger.debug(f"Provenance persisted to DB: {uploaded.content_id[:12]}...")
        except Exception as e:
            logger.warning(
                f"Provenance DB persist failed for {uploaded.content_id[:12]}... "
                f"(in-memory record intact): {e}"
            )

    async def _update_provenance_access(self, cid: str, royalty_earned: float) -> None:
        """
        Atomically update access_count (+1) and total_royalties in the DB.

        Called after every successful royalty credit in record_access().
        Non-blocking: failure is logged but does not affect the in-memory
        access_count / total_royalties that were already updated.
        """
        try:
            from prsm.core.database import ProvenanceQueries
            await ProvenanceQueries.update_access_stats(
                cid=cid,
                access_count_delta=1,
                royalty_delta=royalty_earned,
            )
        except Exception as e:
            logger.warning(
                f"Provenance access-stats update failed for {cid[:12]}...: {e}"
            )

    async def _platform_royalty_transfer(
        self,
        from_user: str,
        to_user: str,
        amount: float,
        content_id: str,
        description: str = "",
    ) -> None:
        """
        Transfer royalty payment via platform FTNS alongside the local ledger credit.

        The local ledger credit is authoritative for the node's own accounting and
        has already been applied before this method is called. This method keeps the
        platform FTNS balances (visible via the API) in sync.

        Both from_user and to_user are node IDs, used directly as platform FTNS
        account identifiers. AtomicFTNSService.transfer_tokens_atomic() auto-creates
        accounts for first-time nodes via ensure_account_exists().

        Non-blocking: any exception — DB unavailable, insufficient balance,
        same-account transfer — is logged but does not affect the already-applied
        local ledger credit or the served content request.

        The 5% network fee is NOT wired here; it requires a pre-configured system
        account and is deferred to Phase 3.

        Args:
            from_user: Node ID of the content accessor (pays the royalty)
            to_user:   Node ID of the content creator (earns the royalty)
            amount:    FTNS to transfer (positive float, truncated to 6 decimal places)
            content_id: Content identifier — used in description and idempotency key
            description: Human-readable label for the transaction record
        """
        if amount <= 0 or from_user == to_user:
            return  # Guards already checked by caller, but defence-in-depth

        try:
            from decimal import Decimal
            from prsm.economy.tokenomics.atomic_ftns_service import AtomicFTNSService
            import uuid as _uuid

            service = AtomicFTNSService()
            idempotency_key = f"royalty:{content_id}:{from_user}:{_uuid.uuid4().hex[:16]}"

            result = await service.transfer_tokens_atomic(
                from_user_id=from_user,
                to_user_id=to_user,
                amount=Decimal(str(round(amount, 6))),
                idempotency_key=idempotency_key,
                description=description or f"Content royalty: {content_id[:12]}...",
            )

            if result.success:
                logger.debug(
                    f"Platform royalty: {from_user[:8]}→{to_user[:8]} "
                    f"{amount:.4f} FTNS for {content_id[:12]}..."
                )
            else:
                # Most common cause: accessor has zero FTNS balance.
                # Local ledger credit is already applied — content was served.
                logger.info(
                    f"Platform royalty deferred for {content_id[:12]}...: "
                    f"{result.error_message} (local ledger credit intact)"
                )
        except Exception as e:
            logger.warning(
                f"Platform royalty transfer unavailable for {content_id[:12]}...: {e}",
            )

    async def _hydrate_from_db(self) -> int:
        """
        Load provenance records from the platform database into uploaded_content.

        Called during node initialization to restore state after a restart.
        Only loads records where creator_id matches this node's identity,
        and skips CIDs already present in uploaded_content (in-memory wins).

        Returns:
            Number of records loaded from DB.
        """
        try:
            from prsm.core.database import ProvenanceQueries
            records = await ProvenanceQueries.load_all_for_node(self.identity.node_id)
            loaded = 0
            for rec in records:
                if rec["content_id"] in self.uploaded_content:
                    continue  # In-memory record takes precedence
                uploaded = UploadedContent(
                    content_id=rec["content_id"],
                    filename=rec["filename"],
                    size_bytes=rec["size_bytes"],
                    content_hash=rec["content_hash"],
                    creator_id=rec["creator_id"],
                    created_at=rec["created_at"],
                    provenance_signature=rec["provenance_signature"],
                    royalty_rate=rec["royalty_rate"],
                    parent_cids=rec["parent_cids"],
                    access_count=rec["access_count"],
                    total_royalties=rec["total_royalties"],
                    is_sharded=rec["is_sharded"],
                    manifest_content_id=rec["manifest_content_id"],
                    total_shards=rec["total_shards"],
                    embedding_id=rec["embedding_id"],
                    near_duplicate_of=rec["near_duplicate_of"],
                    near_duplicate_similarity=rec["near_duplicate_similarity"],
                    # Phase 1.3 Task 2: restore canonical on-chain hash.
                    # Use .get() so rows written before Phase 1.3 (when the
                    # column did not exist and load_all_for_node omitted the
                    # key) don't KeyError — legacy rows hydrate as None and
                    # fall back to local royalties, matching the pre-1.3
                    # behaviour operators already expect.
                    provenance_hash=rec.get("provenance_hash"),
                )
                self.uploaded_content[rec["content_id"]] = uploaded
                # Phase 1.3: restart must also re-populate provider._local_content
                # so previously-uploaded content stays servable after the restart.
                # provenance_hash is now populated from the DB above (Task 2),
                # so on-chain routing survives restarts for rows uploaded with
                # a configured creator 0x address.
                self._register_with_provider(uploaded)
                loaded += 1
            if loaded > 0:
                logger.info(
                    f"Hydrated {loaded} provenance record(s) from DB "
                    f"for node {self.identity.node_id[:12]}..."
                )
            return loaded
        except Exception as e:
            logger.warning(f"Provenance DB hydration failed: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Return uploader statistics."""
        total_bytes = sum(c.size_bytes for c in self.uploaded_content.values())
        total_royalties = sum(c.total_royalties for c in self.uploaded_content.values())
        total_accesses = sum(c.access_count for c in self.uploaded_content.values())
        sharded_count = sum(1 for c in self.uploaded_content.values() if c.is_sharded)
        total_shards = sum(c.total_shards for c in self.uploaded_content.values() if c.is_sharded)
        derivative_count = sum(1 for c in self.uploaded_content.values() if c.near_duplicate_of)
        return {
            "uploaded_count": len(self.uploaded_content),
            "total_bytes": total_bytes,
            "total_royalties_ftns": total_royalties,
            "total_accesses": total_accesses,
            "sharded_count": sharded_count,
            "total_shards": total_shards,
            "sharding_threshold": self.sharding_threshold,
            "semantic_index_size": len(self._semantic_index),
            "derivative_works_detected": derivative_count,
            "embedding_fn_active": self._embedding_fn is not None,
        }

    async def retrieve_sharded_content(self, manifest_cid: str) -> Optional[bytes]:
        """Retrieve and reassemble sharded content by manifest CID.

        Args:
            manifest_cid: CID of the shard manifest

        Returns:
            Reassembled content bytes, or None on failure
        """
        try:
            # Check if we have the manifest locally
            manifest = self.shard_manifests.get(manifest_cid)

            if not manifest:
                # Try to fetch the manifest via the BitTorrent layer.
                manifest_json = await self._fetch_content(manifest_cid)
                if manifest_json:
                    manifest = ShardManifest.from_json(manifest_json.decode())
                    self.shard_manifests[manifest_cid] = manifest

            if not manifest:
                logger.error(f"Manifest not found for CID {manifest_cid[:12]}...")
                return None

            # Get the sharder and reassemble
            sharder = await self._get_content_sharder()
            content = await sharder.reassemble_content(manifest, verify=True)

            logger.info(
                f"Retrieved and reassembled sharded content from manifest "
                f"{manifest_cid[:12]}... ({len(content)} bytes, {manifest.total_shards} shards)"
            )
            return content

        except Exception as e:
            logger.error(f"Failed to retrieve sharded content {manifest_cid[:12]}...: {e}")
            return None

    def get_manifest(self, manifest_cid: str) -> Optional[ShardManifest]:
        """Get a stored shard manifest by CID.

        Args:
            manifest_cid: CID of the manifest

        Returns:
            ShardManifest if found, None otherwise
        """
        return self.shard_manifests.get(manifest_cid)

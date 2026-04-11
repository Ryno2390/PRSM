"""
Content Uploader & Retrieval
============================

Upload content to IPFS with provenance tracking for royalties,
and retrieve content from the network by CID.

Creates a verifiable provenance chain so the original creator
earns FTNS when other nodes access or use their content.

Sprint 4 Phase 4: Added request_content() for cross-node downloads
with provider discovery via ContentIndex, content hash verification,
and support for both inline (base64) and IPFS gateway transfer modes.
Phase 1.3 Task 3b: the client-side request_content/server-side
handler pair has been retired here; ContentProvider is the canonical
serve path and ContentProvider.request_content is the client path.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

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

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        # content_id → (normalised_embedding, creator_id)
        self._index: Dict[str, Tuple] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    def __len__(self) -> int:
        return len(self._index)

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
        """Return (content_id, cosine_similarity, creator_id) for the closest stored
        embedding, or None if the index is empty."""
        if not _HAS_NUMPY or not self._index:
            return None
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            return None
        query = embedding / norm
        best_content_id, best_sim, best_creator = None, -1.0, ""
        for content_id, (stored, creator) in self._index.items():
            sim = float(np.dot(query, stored))
            if sim > best_sim:
                best_sim, best_content_id, best_creator = sim, content_id, creator
        return (best_content_id, best_sim, best_creator)

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
    parent_content_ids: List[str] = field(default_factory=list)
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


class ContentUploader:
    """Upload content to IPFS with provenance registration for royalties.

    Flow:
    1. Upload content to local IPFS node
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
        ipfs_api_url: str = "http://127.0.0.1:5001",
        creator_address: Optional[str] = None,
        content_provider: Optional["ContentProvider"] = None,
    ):
        self.identity = identity
        self.gossip = gossip
        self.ledger = ledger
        self.transport = transport
        self.content_index = content_index  # For looking up parent content creators
        self.ledger_sync = ledger_sync      # For broadcasting transactions
        self.content_economy = content_economy  # For replication tracking (Phase 4)
        # Legacy IPFS HTTP endpoint for _ipfs_add / _ipfs_cat helpers. Kept
        # for back-compat with call sites that still use the HTTP IPFS
        # daemon; new code should go through ContentStore.
        self.ipfs_api_url = ipfs_api_url

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

        # Sharding configuration
        self.sharding_threshold = sharding_threshold

        # Semantic deduplication
        # embedding_fn: async (text: str) -> np.ndarray  — optional, skipped if None
        self._embedding_fn: Optional[Callable] = embedding_fn
        self._semantic_index = _SemanticIndex(persist_path=semantic_index_path)

        self.uploaded_content: Dict[str, UploadedContent] = {}

        # Manifest tracking: manifest_content_id -> manifest data
        self.shard_manifests: Dict[str, Any] = {}

    async def close(self) -> None:
        pass

    def _register_with_provider(self, uploaded: "UploadedContent") -> None:
        """Forward a successfully uploaded record into ContentProvider._local_content.

        No-op when no ContentProvider was injected (backward compat with
        legacy unit tests that construct ContentUploader without a provider).

        The metadata dict mirrors what _handle_content_request dispatches to
        ContentEconomy.process_content_access. Note: the parent CID list is
        stored under ``parent_cids`` (not ``parent_content_ids``) so the
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
                "parent_cids": uploaded.parent_content_ids,
                "provenance_hash": uploaded.provenance_hash,
            },
        )

    async def _get_embedding(self, content: bytes) -> "Optional[np.ndarray]":
        """Attempt to generate a semantic embedding for content.

        Decodes content as UTF-8 text and calls the configured embedding_fn.
        Returns None if:
        - no embedding_fn was provided at construction
        - content is not valid text (binary blobs)
        - the text is too short to be meaningful (<50 chars)
        - the embedding call fails for any reason (non-fatal)
        """
        if self._embedding_fn is None or not _HAS_NUMPY:
            return None
        try:
            text = content.decode("utf-8", errors="ignore").strip()
            if len(text) < 50:
                return None
            # Truncate to ~32 k chars to stay within typical embedding token limits
            return await self._embedding_fn(text[:32_000])
        except Exception as exc:
            logger.debug(f"Embedding generation skipped: {exc}")
            return None

    async def upload(
        self,
        content: bytes,
        filename: str = "untitled",
        metadata: Optional[Dict[str, Any]] = None,
        replicas: int = 3,
        royalty_rate: Optional[float] = None,
        parent_content_ids: Optional[List[str]] = None,
        force_shard: bool = False,
    ) -> Optional[UploadedContent]:
        """Upload content to storage and register provenance.

        Args:
            content: Raw bytes to upload
            filename: Display name for the content
            metadata: Optional metadata dict
            replicas: Number of storage replicas to request
            royalty_rate: FTNS earned per access (clamped to 0.001–0.1, default 0.01)
            parent_content_ids: content IDs of source material this content derives from
            force_shard: If True, force sharding regardless of file size

        Returns:
            UploadedContent with content ID and provenance info, or None on failure
        """
        # Clamp royalty rate to bounds
        rate = royalty_rate if royalty_rate is not None else DEFAULT_ROYALTY_RATE
        rate = max(MIN_ROYALTY_RATE, min(MAX_ROYALTY_RATE, rate))

        content_hash = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        parents = parent_content_ids or []

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
        # to IPFS so we can auto-register derivative relationships early.
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
                parent_content_ids=parents,
                content_hash=content_hash,
                embedding=embedding,
                near_dup_cid=near_dup_cid,
                near_dup_sim=near_dup_sim,
                provenance_hash_hex=provenance_hash_hex,
            )

        # Standard monolithic upload for small files
        cid = await self._ipfs_add(content, filename)
        if not cid:
            logger.error(f"Failed to upload {filename} to IPFS")
            return None

        embedding_id = f"emb:{cid}" if embedding is not None else None

        # Create provenance record
        provenance_data = {
            "content_id": cid,
            "content_hash": content_hash,
            "creator_id": self.identity.node_id,
            "creator_public_key": self.identity.public_key_b64,
            "filename": filename,
            "size_bytes": size_bytes,
            "created_at": time.time(),
            "metadata": metadata or {},
            "royalty_rate": rate,
            "parent_content_ids": parents,
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
            parent_content_ids=parents,
            is_sharded=False,
            embedding_id=embedding_id,
            near_duplicate_of=near_dup_cid,
            near_duplicate_similarity=near_dup_sim,
            provenance_hash=provenance_hash_hex,
        )
        self.uploaded_content[cid] = uploaded
        self._register_with_provider(uploaded)  # Phase 1.3: populate provider._local_content
        await self._persist_provenance(uploaded)  # Persist to DB

        # Register embedding so future uploads can be checked against this one
        if embedding is not None:
            self._semantic_index.store(cid, embedding, self.identity.node_id)

        # Gossip provenance registration
        await self.gossip.publish(GOSSIP_PROVENANCE_REGISTER, {
            **provenance_data,
            "signature": provenance_signature,
        })

        # Advertise content availability to the network
        await self.gossip.publish(GOSSIP_CONTENT_ADVERTISE, {
            "content_id": cid,
            "filename": filename,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "creator_id": self.identity.node_id,
            "provider_id": self.identity.node_id,
            "created_at": provenance_data["created_at"],
            "metadata": metadata or {},
            "royalty_rate": rate,
            "parent_content_ids": parents,
            "embedding_id": embedding_id,
            "provenance_hash": provenance_hash_hex,
        })

        # Request storage replication
        if replicas > 0:
            await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                "content_id": cid,
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
        parent_content_ids: List[str],
        content_hash: str,
        embedding: "Optional[np.ndarray]" = None,
        near_dup_cid: Optional[str] = None,
        near_dup_sim: Optional[float] = None,
        provenance_hash_hex: Optional[str] = None,
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
                    "parent_content_ids": parent_content_ids,
                },
            )

            # Store the manifest for later retrieval
            self.shard_manifests[manifest_cid] = manifest

            size_bytes = len(content)

            # Create provenance record for the sharded content
            # The manifest content ID serves as the primary identifier for sharded content
            embedding_id = f"emb:{manifest_cid}" if embedding is not None else None
            provenance_data = {
                "content_id": manifest_cid,  # Use manifest content ID as the primary identifier
                "content_hash": content_hash,
                "creator_id": self.identity.node_id,
                "creator_public_key": self.identity.public_key_b64,
                "filename": filename,
                "size_bytes": size_bytes,
                "created_at": time.time(),
                "metadata": metadata or {},
                "royalty_rate": royalty_rate,
                "parent_content_ids": parent_content_ids,
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
                parent_content_ids=parent_content_ids,
                is_sharded=True,
                manifest_content_id=manifest_cid,
                total_shards=manifest.total_shards,
                embedding_id=embedding_id,
                near_duplicate_of=near_dup_cid,
                near_duplicate_similarity=near_dup_sim,
                provenance_hash=provenance_hash_hex,
            )
            self.uploaded_content[manifest_cid] = uploaded
            self._register_with_provider(uploaded)  # Phase 1.3: populate provider._local_content
            await self._persist_provenance(uploaded)  # Persist to DB

            # Register embedding for future deduplication checks
            if embedding is not None:
                self._semantic_index.store(manifest_cid, embedding, self.identity.node_id)

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
                "parent_content_ids": parent_content_ids,
                "is_sharded": True,
                "total_shards": manifest.total_shards,
                "embedding_id": embedding_id,
                "provenance_hash": provenance_hash_hex,
            })

            # Request storage replication for each shard
            if replicas > 0:
                # Request replication for the manifest
                await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                    "content_id": manifest_cid,
                    "size_bytes": len(manifest.to_json()),
                    "requester_id": self.identity.node_id,
                    "replicas_needed": replicas,
                })

                # Request replication for each shard
                for shard_info in manifest.shards:
                    await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                        "content_id": shard_info.content_id,
                        "size_bytes": shard_info.size,
                        "requester_id": self.identity.node_id,
                        "replicas_needed": replicas,
                    })

            logger.info(
                f"Uploaded {filename} ({size_bytes} bytes) with sharding -> "
                f"manifest={manifest_cid}, shards={manifest.total_shards}, "
                f"royalty={royalty_rate} FTNS/access, parents={len(parent_content_ids)}"
            )
            return uploaded

        except ShardingError as e:
            logger.error(f"Sharding failed for {filename}: {e}")
            # Fall back to monolithic upload
            logger.info(f"Falling back to monolithic upload for {filename}")
            cid = await self._ipfs_add(content, filename)
            if not cid:
                logger.error(f"Failed to upload {filename} to IPFS (fallback)")
                return None

            size_bytes = len(content)
            provenance_data = {
                "content_id": cid,
                "content_hash": content_hash,
                "creator_id": self.identity.node_id,
                "creator_public_key": self.identity.public_key_b64,
                "filename": filename,
                "size_bytes": size_bytes,
                "created_at": time.time(),
                "metadata": metadata or {},
                "royalty_rate": royalty_rate,
                "parent_content_ids": parent_content_ids,
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
                parent_content_ids=parent_content_ids,
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
                "content_id": cid,
                "filename": filename,
                "size_bytes": size_bytes,
                "content_hash": content_hash,
                "creator_id": self.identity.node_id,
                "provider_id": self.identity.node_id,
                "created_at": provenance_data["created_at"],
                "metadata": metadata or {},
                "royalty_rate": royalty_rate,
                "parent_content_ids": parent_content_ids,
                "provenance_hash": provenance_hash_hex,
            })
            if replicas > 0:
                await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                    "content_id": cid,
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
        parent_content_ids: Optional[List[str]] = None,
    ) -> Optional[UploadedContent]:
        """Upload JSON-serializable data to content storage."""
        content = json.dumps(data, indent=2).encode()
        return await self.upload(content, filename, metadata, replicas, royalty_rate, parent_content_ids)

    async def upload_text(
        self,
        text: str,
        filename: str = "document.txt",
        metadata: Optional[Dict[str, Any]] = None,
        replicas: int = 3,
        royalty_rate: Optional[float] = None,
        parent_content_ids: Optional[List[str]] = None,
    ) -> Optional[UploadedContent]:
        """Upload text content to content storage."""
        return await self.upload(text.encode("utf-8"), filename, metadata, replicas, royalty_rate, parent_content_ids)

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

        if content.parent_content_ids and self.content_index:
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
        parent_creators = self._resolve_parent_creators(content.parent_content_ids)
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

    def _resolve_parent_creators(self, parent_content_ids: List[str]) -> List[str]:
        """Look up the creator node IDs for parent content IDs via the content index."""
        creators = []
        if not self.content_index:
            return creators
        for pcid in parent_content_ids:
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

    async def _fetch_from_gateway(self, gateway_url: str) -> Optional[bytes]:
        """Fetch content bytes from an IPFS gateway URL."""
        try:
            import aiohttp
            session = await self._get_ipfs_session()
            async with session.get(
                gateway_url,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    logger.debug(f"Gateway returned status {resp.status} for {gateway_url}")
        except Exception as e:
            logger.error(f"Gateway fetch failed for {gateway_url}: {e}")
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
        parent_content_ids = data.get("parent_content_ids", [])

        if not content_id or not accessor_id:
            return

        # Case 1: We are the direct creator and have a local record
        if creator_id == self.identity.node_id and content_id in self.uploaded_content:
            await self.record_access(content_id, accessor_id)
            return

        # Case 2: We are a source creator for a derivative work
        if parent_content_ids:
            my_parent_content_ids = [
                pcid for pcid in parent_content_ids
                if pcid in self.uploaded_content
            ]
            if my_parent_content_ids:
                source_pool = royalty_rate * SOURCE_CREATOR_SHARE
                # Count total parents to split evenly
                per_parent = source_pool / len(parent_content_ids)
                source_royalty = per_parent * len(my_parent_content_ids)
                try:
                    await self.ledger.credit(
                        wallet_id=self.identity.node_id,
                        amount=source_royalty,
                        tx_type=TransactionType.CONTENT_ROYALTY,
                        description=f"Source royalty for derivative {content_id[:12]}... ({len(my_parent_content_ids)} parent(s))",
                    )
                    for pcid in my_parent_content_ids:
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

    # ── IPFS operations ──────────────────────────────────────────

    async def _ipfs_add(self, content: bytes, filename: str) -> Optional[str]:
        """Add content to IPFS, return CID or None."""
        try:
            import aiohttp
            session = await self._get_ipfs_session()
            data = aiohttp.FormData()
            data.add_field("file", content, filename=filename)
            async with session.post(
                f"{self.ipfs_api_url}/api/v0/add",
                data=data,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("Hash", "")
        except Exception as e:
            logger.error(f"IPFS add failed: {e}")
        return None

    async def _ipfs_cat(self, cid: str) -> Optional[bytes]:
        """Fetch content bytes from IPFS."""
        try:
            import aiohttp
            session = await self._get_ipfs_session()
            async with session.post(
                f"{self.ipfs_api_url}/api/v0/cat",
                params={"arg": cid},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
        except Exception as e:
            logger.error(f"IPFS cat failed for {cid}: {e}")
        return None

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
                "parent_content_ids": uploaded.parent_content_ids,
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
                    parent_content_ids=rec["parent_content_ids"],
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
                # Try to fetch the manifest from IPFS
                manifest_json = await self._ipfs_cat(manifest_cid)
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


class _IPFSClientWrapper:
    """Wrapper to make ContentUploader compatible with ContentSharder's IPFS client interface.

    ContentSharder expects an IPFS client with upload_content() and download_content()
    methods that return objects with specific attributes. This wrapper adapts
    ContentUploader's methods to that interface.
    """

    def __init__(self, uploader: ContentUploader):
        self._uploader = uploader

    async def upload_content(
        self,
        content: bytes,
        filename: str = "content.bin",
        pin: bool = True,
    ) -> "_UploadResult":
        """Upload content to IPFS.

        Args:
            content: Raw bytes to upload
            filename: Name for the content
            pin: Whether to pin the content (ignored, always pinned)

        Returns:
            _UploadResult with success status and CID
        """
        cid = await self._uploader._ipfs_add(content, filename)
        if cid:
            return _UploadResult(success=True, cid=cid, error=None)
        else:
            return _UploadResult(success=False, cid=None, error="IPFS add failed")

    async def download_content(self, cid: str) -> "_DownloadResult":
        """Download content from IPFS.

        Args:
            cid: Content identifier to download

        Returns:
            _DownloadResult with success status and content
        """
        content = await self._uploader._ipfs_cat(cid)
        if content is not None:
            return _DownloadResult(
                success=True,
                metadata={"content": content},
                error=None,
            )
        else:
            return _DownloadResult(success=False, metadata=None, error="IPFS cat failed")


@dataclass
class _UploadResult:
    """Result object for IPFS uploads, compatible with ContentSharder interface."""
    success: bool
    cid: Optional[str]
    error: Optional[str]


@dataclass
class _DownloadResult:
    """Result object for IPFS downloads, compatible with ContentSharder interface."""
    success: bool
    metadata: Optional[Dict[str, Any]]
    error: Optional[str]

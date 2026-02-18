"""
Content Uploader
================

Upload content to IPFS with provenance tracking for royalties.
Creates a verifiable provenance chain so the original creator
earns FTNS when other nodes access or use their content.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prsm.node.gossip import (
    GOSSIP_CONTENT_ACCESS,
    GOSSIP_CONTENT_ADVERTISE,
    GOSSIP_PROVENANCE_REGISTER,
    GOSSIP_STORAGE_REQUEST,
    GossipProtocol,
)
from prsm.node.transport import MSG_DIRECT, P2PMessage, PeerConnection, WebSocketTransport
from prsm.node.identity import NodeIdentity
from prsm.node.local_ledger import LocalLedger, TransactionType

logger = logging.getLogger(__name__)

# Royalty rate bounds (FTNS per access)
MIN_ROYALTY_RATE = 0.001
MAX_ROYALTY_RATE = 0.1
DEFAULT_ROYALTY_RATE = 0.01

# Multi-level provenance splits
DERIVATIVE_CREATOR_SHARE = 0.70   # 70% to the derivative creator
SOURCE_CREATOR_SHARE = 0.25       # 25% to each source creator (split evenly)
NETWORK_FEE_SHARE = 0.05          # 5% network fee


@dataclass
class UploadedContent:
    """Tracks content uploaded by this node."""
    cid: str
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
        ipfs_api_url: str = "http://127.0.0.1:5001",
        transport: Optional[WebSocketTransport] = None,
        content_index: Optional[Any] = None,
        ledger_sync: Optional[Any] = None,
    ):
        self.identity = identity
        self.gossip = gossip
        self.ledger = ledger
        self.ipfs_api_url = ipfs_api_url
        self.transport = transport
        self.content_index = content_index  # For looking up parent content creators
        self.ledger_sync = ledger_sync      # For broadcasting transactions

        self.uploaded_content: Dict[str, UploadedContent] = {}
        self._ipfs_session = None

    async def _get_ipfs_session(self):
        if self._ipfs_session is None or self._ipfs_session.closed:
            import aiohttp
            self._ipfs_session = aiohttp.ClientSession()
        return self._ipfs_session

    async def close(self) -> None:
        if self._ipfs_session:
            await self._ipfs_session.close()
            self._ipfs_session = None

    async def upload(
        self,
        content: bytes,
        filename: str = "untitled",
        metadata: Optional[Dict[str, Any]] = None,
        replicas: int = 3,
        royalty_rate: Optional[float] = None,
        parent_cids: Optional[List[str]] = None,
    ) -> Optional[UploadedContent]:
        """Upload content to IPFS and register provenance.

        Args:
            content: Raw bytes to upload
            filename: Display name for the content
            metadata: Optional metadata dict
            replicas: Number of storage replicas to request
            royalty_rate: FTNS earned per access (clamped to 0.001–0.1, default 0.01)
            parent_cids: CIDs of source material this content derives from

        Returns:
            UploadedContent with CID and provenance info, or None on failure
        """
        # Clamp royalty rate to bounds
        rate = royalty_rate if royalty_rate is not None else DEFAULT_ROYALTY_RATE
        rate = max(MIN_ROYALTY_RATE, min(MAX_ROYALTY_RATE, rate))

        content_hash = hashlib.sha256(content).hexdigest()

        # Upload to IPFS
        cid = await self._ipfs_add(content, filename)
        if not cid:
            logger.error(f"Failed to upload {filename} to IPFS")
            return None

        size_bytes = len(content)
        parents = parent_cids or []

        # Create provenance record
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
        }
        provenance_bytes = json.dumps(provenance_data, sort_keys=True).encode()
        provenance_signature = self.identity.sign(provenance_bytes)

        uploaded = UploadedContent(
            cid=cid,
            filename=filename,
            size_bytes=size_bytes,
            content_hash=content_hash,
            creator_id=self.identity.node_id,
            provenance_signature=provenance_signature,
            royalty_rate=rate,
            parent_cids=parents,
        )
        self.uploaded_content[cid] = uploaded

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
        })

        # Request storage replication
        if replicas > 0:
            await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                "cid": cid,
                "size_bytes": size_bytes,
                "requester_id": self.identity.node_id,
                "replicas_needed": replicas,
            })

        logger.info(
            f"Uploaded {filename} ({size_bytes} bytes) -> {cid}, "
            f"royalty={rate} FTNS/access, parents={len(parents)}, replicas={replicas}"
        )
        return uploaded

    async def upload_json(
        self,
        data: Any,
        filename: str = "data.json",
        metadata: Optional[Dict[str, Any]] = None,
        replicas: int = 3,
        royalty_rate: Optional[float] = None,
        parent_cids: Optional[List[str]] = None,
    ) -> Optional[UploadedContent]:
        """Upload JSON-serializable data to IPFS."""
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
        """Upload text content to IPFS."""
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

    async def _distribute_multilevel_royalty(
        self, content: UploadedContent, total_royalty: float, accessor_id: str
    ) -> None:
        """Split royalty among derivative creator, source creators, and network."""
        cid = content.cid
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
        """Look up the creator node IDs for parent CIDs via the content index."""
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
        """Register direct-message handler and gossip subscriptions."""
        if self.transport:
            self.transport.on_message(MSG_DIRECT, self._on_direct_message)
        self.gossip.subscribe(GOSSIP_CONTENT_ACCESS, self._on_content_access)
        logger.info("Content uploader started — listening for content requests")

    async def _on_direct_message(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Route incoming direct messages to the appropriate handler."""
        subtype = msg.payload.get("subtype", "")
        if subtype == "content_request":
            await self._handle_content_request(msg, peer)

    async def _handle_content_request(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Serve content in response to a direct content_request."""
        cid = msg.payload.get("cid", "")
        request_id = msg.payload.get("request_id", "")

        content_info = self.uploaded_content.get(cid)
        if not content_info:
            # We don't have this content
            await self._send_direct(peer.peer_id, {
                "subtype": "content_response",
                "request_id": request_id,
                "cid": cid,
                "found": False,
            })
            return

        # Small files (<=1MB): send inline as base64
        if content_info.size_bytes <= 1_048_576:
            raw = await self._ipfs_cat(cid)
            if raw is not None:
                import base64
                await self._send_direct(peer.peer_id, {
                    "subtype": "content_response",
                    "request_id": request_id,
                    "cid": cid,
                    "found": True,
                    "transfer_mode": "inline",
                    "data_b64": base64.b64encode(raw).decode(),
                    "filename": content_info.filename,
                    "size_bytes": content_info.size_bytes,
                    "content_hash": content_info.content_hash,
                })
            else:
                await self._send_direct(peer.peer_id, {
                    "subtype": "content_response",
                    "request_id": request_id,
                    "cid": cid,
                    "found": False,
                })
                return
        else:
            # Large files: provide IPFS gateway URL
            gateway_url = f"http://127.0.0.1:8080/ipfs/{cid}"
            await self._send_direct(peer.peer_id, {
                "subtype": "content_response",
                "request_id": request_id,
                "cid": cid,
                "found": True,
                "transfer_mode": "gateway",
                "gateway_url": gateway_url,
                "filename": content_info.filename,
                "size_bytes": content_info.size_bytes,
            })

        # Record the access and gossip it
        await self.record_access(cid, msg.sender_id)
        await self.gossip.publish(GOSSIP_CONTENT_ACCESS, {
            "cid": cid,
            "accessor_id": msg.sender_id,
            "creator_id": content_info.creator_id,
            "royalty_rate": content_info.royalty_rate,
            "parent_cids": content_info.parent_cids,
            "timestamp": time.time(),
        })

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

        cid = data.get("cid", "")
        accessor_id = data.get("accessor_id", "")
        creator_id = data.get("creator_id", "")
        royalty_rate = data.get("royalty_rate", DEFAULT_ROYALTY_RATE)
        parent_cids = data.get("parent_cids", [])

        if not cid or not accessor_id:
            return

        # Case 1: We are the direct creator and have a local record
        if creator_id == self.identity.node_id and cid in self.uploaded_content:
            await self.record_access(cid, accessor_id)
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
                        description=f"Source royalty for derivative {cid[:12]}... ({len(my_parent_cids)} parent(s))",
                    )
                    for pcid in my_parent_cids:
                        self.uploaded_content[pcid].total_royalties += per_parent
                    logger.info(f"Source royalty earned: {source_royalty:.4f} FTNS for derivative {cid[:12]}...")
                except Exception as e:
                    logger.error(f"Source royalty credit failed: {e}")

    async def _send_direct(self, peer_id: str, payload: Dict[str, Any]) -> None:
        """Send a direct P2P message to a specific peer."""
        if not self.transport:
            return
        msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload=payload,
        )
        await self.transport.send_to_peer(peer_id, msg)

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

    def get_stats(self) -> Dict[str, Any]:
        """Return uploader statistics."""
        total_bytes = sum(c.size_bytes for c in self.uploaded_content.values())
        total_royalties = sum(c.total_royalties for c in self.uploaded_content.values())
        total_accesses = sum(c.access_count for c in self.uploaded_content.values())
        return {
            "uploaded_count": len(self.uploaded_content),
            "total_bytes": total_bytes,
            "total_royalties_ftns": total_royalties,
            "total_accesses": total_accesses,
        }

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

from prsm.node.gossip import GOSSIP_PROVENANCE_REGISTER, GOSSIP_STORAGE_REQUEST, GossipProtocol
from prsm.node.identity import NodeIdentity
from prsm.node.local_ledger import LocalLedger, TransactionType

logger = logging.getLogger(__name__)


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
    ):
        self.identity = identity
        self.gossip = gossip
        self.ledger = ledger
        self.ipfs_api_url = ipfs_api_url

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
    ) -> Optional[UploadedContent]:
        """Upload content to IPFS and register provenance.

        Args:
            content: Raw bytes to upload
            filename: Display name for the content
            metadata: Optional metadata dict
            replicas: Number of storage replicas to request

        Returns:
            UploadedContent with CID and provenance info, or None on failure
        """
        content_hash = hashlib.sha256(content).hexdigest()

        # Upload to IPFS
        cid = await self._ipfs_add(content, filename)
        if not cid:
            logger.error(f"Failed to upload {filename} to IPFS")
            return None

        size_bytes = len(content)

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
        )
        self.uploaded_content[cid] = uploaded

        # Gossip provenance registration
        await self.gossip.publish(GOSSIP_PROVENANCE_REGISTER, {
            **provenance_data,
            "signature": provenance_signature,
        })

        # Request storage replication
        if replicas > 0:
            await self.gossip.publish(GOSSIP_STORAGE_REQUEST, {
                "cid": cid,
                "size_bytes": size_bytes,
                "requester_id": self.identity.node_id,
                "replicas_needed": replicas,
            })

        logger.info(f"Uploaded {filename} ({size_bytes} bytes) -> {cid}, requesting {replicas} replicas")
        return uploaded

    async def upload_json(
        self,
        data: Any,
        filename: str = "data.json",
        metadata: Optional[Dict[str, Any]] = None,
        replicas: int = 3,
    ) -> Optional[UploadedContent]:
        """Upload JSON-serializable data to IPFS."""
        content = json.dumps(data, indent=2).encode()
        return await self.upload(content, filename, metadata, replicas)

    async def upload_text(
        self,
        text: str,
        filename: str = "document.txt",
        metadata: Optional[Dict[str, Any]] = None,
        replicas: int = 3,
    ) -> Optional[UploadedContent]:
        """Upload text content to IPFS."""
        return await self.upload(text.encode("utf-8"), filename, metadata, replicas)

    async def record_access(self, cid: str, accessor_id: str) -> None:
        """Record that content was accessed, earning a royalty."""
        content = self.uploaded_content.get(cid)
        if not content:
            return
        if accessor_id == self.identity.node_id:
            return  # No self-royalties

        content.access_count += 1
        royalty = 0.01  # FTNS per access

        try:
            await self.ledger.credit(
                wallet_id=self.identity.node_id,
                amount=royalty,
                tx_type=TransactionType.CONTENT_ROYALTY,
                description=f"Royalty for {cid[:12]}... access by {accessor_id[:8]}",
            )
            content.total_royalties += royalty
        except Exception as e:
            logger.error(f"Royalty credit failed: {e}")

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

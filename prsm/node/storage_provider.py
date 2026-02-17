"""
Storage Provider
================

IPFS pin space contribution for the PRSM network.
Auto-detects local IPFS daemon, pledges configurable storage,
accepts pin requests, and earns FTNS rewards for storage proofs.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from prsm.node.gossip import (
    GOSSIP_PROOF_OF_STORAGE,
    GOSSIP_STORAGE_CONFIRM,
    GOSSIP_STORAGE_REQUEST,
    GossipProtocol,
)
from prsm.node.identity import NodeIdentity
from prsm.node.local_ledger import LocalLedger, TransactionType

logger = logging.getLogger(__name__)

# Reward rate: FTNS per GB per epoch (1 hour)
STORAGE_REWARD_RATE = 0.1


@dataclass
class PinnedContent:
    """Tracks content pinned by this node."""
    cid: str
    size_bytes: int
    pinned_at: float = field(default_factory=time.time)
    requester_id: str = ""
    last_verified: float = field(default_factory=time.time)


class StorageProvider:
    """Contributes IPFS pin space to the network and earns FTNS rewards.

    Requires a running IPFS daemon (Kubo) at the configured API URL.
    Gracefully degrades if IPFS is not available — storage features
    are disabled but the node continues to operate.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        ipfs_api_url: str = "http://127.0.0.1:5001",
        pledged_gb: float = 10.0,
        reward_interval: float = 3600.0,  # 1 hour
    ):
        self.identity = identity
        self.gossip = gossip
        self.ledger = ledger
        self.ipfs_api_url = ipfs_api_url
        self.pledged_gb = pledged_gb
        self.reward_interval = reward_interval

        self.ipfs_available = False
        self.pinned_content: Dict[str, PinnedContent] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._ipfs_session = None

    @property
    def used_bytes(self) -> int:
        return sum(p.size_bytes for p in self.pinned_content.values())

    @property
    def used_gb(self) -> float:
        return round(self.used_bytes / (1024**3), 4)

    @property
    def available_gb(self) -> float:
        return max(0.0, self.pledged_gb - self.used_gb)

    async def start(self) -> None:
        """Detect IPFS, register handlers, start reward loop."""
        self._running = True

        # Check IPFS availability
        self.ipfs_available = await self._check_ipfs()
        if self.ipfs_available:
            logger.info(f"IPFS detected at {self.ipfs_api_url}, storage provider active ({self.pledged_gb}GB pledged)")
            self.gossip.subscribe(GOSSIP_STORAGE_REQUEST, self._on_storage_request)
            self._tasks.append(asyncio.create_task(self._reward_loop()))
        else:
            logger.warning(
                f"IPFS not available at {self.ipfs_api_url} — "
                "storage features disabled. Install Kubo (https://docs.ipfs.tech/install/) "
                "and run 'ipfs daemon' to enable."
            )

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        if self._ipfs_session:
            await self._ipfs_session.close()
            self._ipfs_session = None

    async def _check_ipfs(self) -> bool:
        """Check if IPFS daemon is running."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ipfs_api_url}/api/v0/id",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _get_ipfs_session(self):
        """Get or create aiohttp session for IPFS API."""
        if self._ipfs_session is None or self._ipfs_session.closed:
            import aiohttp
            self._ipfs_session = aiohttp.ClientSession()
        return self._ipfs_session

    async def pin_content(self, cid: str) -> bool:
        """Pin content on IPFS."""
        if not self.ipfs_available:
            return False
        try:
            session = await self._get_ipfs_session()
            import aiohttp
            async with session.post(
                f"{self.ipfs_api_url}/api/v0/pin/add",
                params={"arg": cid},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    # Get content size
                    size = await self._get_content_size(cid)
                    self.pinned_content[cid] = PinnedContent(
                        cid=cid,
                        size_bytes=size,
                    )
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to pin {cid}: {e}")
            return False

    async def _get_content_size(self, cid: str) -> int:
        """Get the size of pinned content."""
        try:
            session = await self._get_ipfs_session()
            import aiohttp
            async with session.post(
                f"{self.ipfs_api_url}/api/v0/object/stat",
                params={"arg": cid},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("CumulativeSize", 0)
        except Exception:
            pass
        return 0

    async def verify_pin(self, cid: str) -> bool:
        """Verify that content is still pinned."""
        if not self.ipfs_available:
            return False
        try:
            session = await self._get_ipfs_session()
            import aiohttp
            async with session.post(
                f"{self.ipfs_api_url}/api/v0/pin/ls",
                params={"arg": cid, "type": "all"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return cid in data.get("Keys", {})
        except Exception:
            pass
        return False

    # ── Gossip handlers ──────────────────────────────────────────

    async def _on_storage_request(self, subtype: str, data: Dict[str, Any], origin: str) -> None:
        """Handle a storage request from the network."""
        if not self._running or not self.ipfs_available:
            return

        cid = data.get("cid", "")
        size_bytes = data.get("size_bytes", 0)
        requester_id = data.get("requester_id", origin)

        if not cid:
            return

        # Don't re-pin content we already have
        if cid in self.pinned_content:
            return

        # Check if we have space
        size_gb = size_bytes / (1024**3) if size_bytes > 0 else 0.001
        if size_gb > self.available_gb:
            return

        # Pin the content
        success = await self.pin_content(cid)
        if success:
            self.pinned_content[cid].requester_id = requester_id

            # Confirm to network
            await self.gossip.publish(GOSSIP_STORAGE_CONFIRM, {
                "cid": cid,
                "provider_id": self.identity.node_id,
                "size_bytes": self.pinned_content[cid].size_bytes,
            })
            logger.info(f"Pinned content {cid[:12]}... for {requester_id[:8]}")

    # ── Reward loop ──────────────────────────────────────────────

    async def _reward_loop(self) -> None:
        """Periodically prove storage and claim rewards."""
        while self._running:
            await asyncio.sleep(self.reward_interval)
            if not self.pinned_content:
                continue

            try:
                # Verify a sample of pins
                verified_count = 0
                for cid, content in list(self.pinned_content.items()):
                    if await self.verify_pin(cid):
                        content.last_verified = time.time()
                        verified_count += 1
                    else:
                        # Content no longer pinned — remove from tracking
                        del self.pinned_content[cid]

                if verified_count == 0:
                    continue

                # Calculate reward: FTNS per GB stored
                reward = round(self.used_gb * STORAGE_REWARD_RATE, 6)
                if reward > 0:
                    await self.ledger.credit(
                        wallet_id=self.identity.node_id,
                        amount=reward,
                        tx_type=TransactionType.STORAGE_REWARD,
                        description=f"Storage reward: {self.used_gb:.4f}GB, {verified_count} CIDs verified",
                    )

                    # Announce proof to network
                    await self.gossip.publish(GOSSIP_PROOF_OF_STORAGE, {
                        "provider_id": self.identity.node_id,
                        "verified_cids": verified_count,
                        "total_gb": self.used_gb,
                        "reward_ftns": reward,
                    })

                    logger.info(f"Storage reward: {reward} FTNS for {self.used_gb:.4f}GB ({verified_count} CIDs)")

            except Exception as e:
                logger.error(f"Reward loop error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return storage provider statistics."""
        return {
            "ipfs_available": self.ipfs_available,
            "pledged_gb": self.pledged_gb,
            "used_gb": self.used_gb,
            "available_gb": self.available_gb,
            "pinned_cids": len(self.pinned_content),
            "reward_rate": STORAGE_REWARD_RATE,
        }

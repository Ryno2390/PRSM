"""
Storage Provider
================

IPFS pin space contribution for the PRSM network.
Auto-detects local IPFS daemon, pledges configurable storage,
accepts pin requests, and earns FTNS rewards for storage proofs.

Includes challenge-response storage proof verification to ensure
providers actually store the content they claim to pin.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from datetime import datetime, timezone

from prsm.core.bandwidth_limiter import BandwidthLimiter

if TYPE_CHECKING:
    from prsm.node.config import NodeConfig

from prsm.node.config import is_active_now
from prsm.node.gossip import (
    GOSSIP_CONTENT_ADVERTISE,
    GOSSIP_PROOF_OF_STORAGE,
    GOSSIP_STORAGE_CONFIRM,
    GOSSIP_STORAGE_REQUEST,
    GossipProtocol,
)
from prsm.node.transport import MSG_DIRECT, P2PMessage, PeerConnection, WebSocketTransport
from prsm.node.identity import NodeIdentity
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.storage_proofs import (
    StorageProofVerifier,
    StorageProver,
    StorageChallenge,
    StorageProof,
    ProofType,
    ChallengeStatus,
    ChallengeRecord,
    DEFAULT_CHALLENGE_DIFFICULTY,
    DEFAULT_CHALLENGE_TIMEOUT_MINUTES,
)

logger = logging.getLogger(__name__)

# Reward rate: FTNS per GB per epoch (1 hour)
STORAGE_REWARD_RATE = 0.1

# Default challenge configuration
DEFAULT_CHALLENGE_INTERVAL = 300.0  # 5 minutes between challenges
DEFAULT_CHALLENGE_DIFFICULTY_BYTES = 4096  # 4KB chunks to prove
MAX_CONCURRENT_CHALLENGES = 10  # Max pending challenges at once


@dataclass
class PinnedContent:
    """Tracks content pinned by this node."""
    cid: str
    size_bytes: int
    pinned_at: float = field(default_factory=time.time)
    requester_id: str = ""
    last_verified: float = field(default_factory=time.time)
    last_challenge_time: float = 0.0  # Last time this CID was challenged
    successful_challenges: int = 0  # Count of successful proof verifications
    failed_challenges: int = 0  # Count of failed proof verifications


@dataclass
class ChallengeConfig:
    """Configuration for storage proof challenges."""
    challenge_interval: float = DEFAULT_CHALLENGE_INTERVAL  # Seconds between challenges
    challenge_difficulty: int = DEFAULT_CHALLENGE_DIFFICULTY_BYTES  # Bytes to prove
    challenge_timeout_minutes: int = DEFAULT_CHALLENGE_TIMEOUT_MINUTES
    max_concurrent_challenges: int = MAX_CONCURRENT_CHALLENGES
    enable_challenges: bool = True  # Master switch for challenge system
    min_challenge_interval_per_cid: float = 60.0  # Min seconds between challenges for same CID


class StorageProvider:
    """Contributes IPFS pin space to the network and earns FTNS rewards.

    Requires a running IPFS daemon (Kubo) at the configured API URL.
    Gracefully degrades if IPFS is not available — storage features
    are disabled but the node continues to operate.

    Includes challenge-response storage proof verification to ensure
    this provider (and remote providers) actually store the content
    they claim to pin.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        gossip: GossipProtocol,
        ledger: LocalLedger,
        ipfs_api_url: str = "http://127.0.0.1:5001",
        pledged_gb: float = 10.0,
        reward_interval: float = 3600.0,  # 1 hour
        challenge_config: Optional[ChallengeConfig] = None,
        config: Optional["NodeConfig"] = None,
    ):
        self.identity = identity
        self.gossip = gossip
        self.ledger = ledger
        self.ipfs_api_url = ipfs_api_url
        self.pledged_gb = pledged_gb
        self.reward_interval = reward_interval
        self.config = config  # NodeConfig for scheduling checks
        
        # Bandwidth limits (0 = unlimited)
        self.upload_mbps_limit: float = 0.0
        self.download_mbps_limit: float = 0.0
        
        # Bandwidth limiter for throttling content serving
        self.bandwidth_limiter = BandwidthLimiter(
            upload_mbps=self.upload_mbps_limit,
            download_mbps=self.download_mbps_limit,
        )

        # Challenge configuration
        self.challenge_config = challenge_config or ChallengeConfig()

        self.ipfs_available = False
        self.pinned_content: Dict[str, PinnedContent] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._ipfs_session = None
        self.ledger_sync = None  # Set by node.py after construction

        # Storage proof system - verifier for challenging other providers
        # and prover for answering challenges to this provider
        self._proof_verifier = StorageProofVerifier(
            ipfs_client=None,  # Will be set later if available
            challenge_timeout_minutes=self.challenge_config.challenge_timeout_minutes,
            max_pending_challenges=self.challenge_config.max_concurrent_challenges,
        )
        self._storage_prover: Optional[StorageProver] = None  # Initialized after IPFS check

        # Track pending challenges issued by this provider
        self._pending_challenges: Dict[str, ChallengeRecord] = {}

        # Track challenges received by this provider (to answer)
        self._received_challenges: Dict[str, StorageChallenge] = {}

        # Provider reputation tracking (for remote providers we challenge)
        self._provider_reputation: Dict[str, float] = {}  # provider_id -> reputation score (0.0-1.0)

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
        """Detect IPFS, register handlers, start reward and challenge loops."""
        self._running = True

        # Check IPFS availability
        self.ipfs_available = await self._check_ipfs()
        if self.ipfs_available:
            logger.info(f"IPFS detected at {self.ipfs_api_url}, storage provider active ({self.pledged_gb}GB pledged)")
            
            # Initialize the storage prover for answering challenges
            self._storage_prover = StorageProver(
                identity=self.identity,
                ipfs_client=None,
                ipfs_api_url=self.ipfs_api_url,
            )
            
            # Register gossip handlers
            self.gossip.subscribe(GOSSIP_STORAGE_REQUEST, self._on_storage_request)
            
            # Subscribe to challenge-related gossip messages
            self.gossip.subscribe("storage_challenge", self._on_storage_challenge)
            self.gossip.subscribe("storage_proof_response", self._on_storage_proof_response)
            
            # Start background tasks
            self._tasks.append(asyncio.create_task(self._reward_loop()))
            
            # Start challenge loop if enabled
            if self.challenge_config.enable_challenges:
                self._tasks.append(asyncio.create_task(self._challenge_loop()))
                self._tasks.append(asyncio.create_task(self._challenge_cleanup_loop()))
                logger.info(
                    f"Storage proof challenges enabled: interval={self.challenge_config.challenge_interval}s, "
                    f"difficulty={self.challenge_config.challenge_difficulty} bytes"
                )
        else:
            logger.warning(
                f"IPFS not available at {self.ipfs_api_url} — "
                "storage features disabled. Install Kubo (https://docs.ipfs.tech/install/) "
                "and run 'ipfs daemon' to enable."
            )

    async def stop(self) -> None:
        """Stop all background tasks and cleanup resources."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        
        if self._ipfs_session:
            await self._ipfs_session.close()
            self._ipfs_session = None
            
        # Close the storage prover's IPFS session
        if self._storage_prover:
            await self._storage_prover.close()
            self._storage_prover = None

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

    async def _get_ipfs_session(self) -> Any:
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

        # Check if we're within active hours
        if self.config and not is_active_now(self.config):
            logger.debug("Node is outside active hours, declining storage request")
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

            # Advertise that we can now serve this content
            await self.gossip.publish(GOSSIP_CONTENT_ADVERTISE, {
                "cid": cid,
                "filename": "",
                "size_bytes": self.pinned_content[cid].size_bytes,
                "content_hash": "",
                "creator_id": requester_id,
                "provider_id": self.identity.node_id,
                "created_at": self.pinned_content[cid].pinned_at,
                "metadata": {},
            })

            logger.info(f"Pinned content {cid[:12]}... for {requester_id[:8]}")

    # ── Content serving ────────────────────────────────────────────

    def register_content_handler(self, transport: WebSocketTransport) -> None:
        """Register to handle direct content_request messages for pinned CIDs."""
        self._transport = transport
        transport.on_message(MSG_DIRECT, self._on_direct_content_request)
        logger.info("Storage provider registered for content serving")

    async def _on_direct_content_request(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Serve pinned content via IPFS gateway URL."""
        subtype = msg.payload.get("subtype", "")
        if subtype != "content_request":
            return

        cid = msg.payload.get("cid", "")
        request_id = msg.payload.get("request_id", "")

        if cid not in self.pinned_content:
            return  # Let the content uploader handle unknown CIDs

        gateway_url = f"http://127.0.0.1:8080/ipfs/{cid}"
        response = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload={
                "subtype": "content_response",
                "request_id": request_id,
                "cid": cid,
                "found": True,
                "transfer_mode": "gateway",
                "gateway_url": gateway_url,
                "filename": "",
                "size_bytes": self.pinned_content[cid].size_bytes,
            },
        )
        await self._transport.send_to_peer(peer.peer_id, response)

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
                    tx = await self.ledger.credit(
                        wallet_id=self.identity.node_id,
                        amount=reward,
                        tx_type=TransactionType.STORAGE_REWARD,
                        description=f"Storage reward: {self.used_gb:.4f}GB, {verified_count} CIDs verified",
                    )

                    # Broadcast earning via ledger sync
                    if self.ledger_sync:
                        try:
                            await self.ledger_sync.broadcast_transaction(tx)
                        except Exception:
                            pass

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

    # ── Storage Proof Challenge System ─────────────────────────────────────

    async def _challenge_loop(self) -> None:
        """Periodically issue storage challenges to verify pinned content.
        
        This loop challenges this provider's own pinned content to ensure
        we can prove storage, and can also challenge remote providers.
        """
        while self._running:
            await asyncio.sleep(self.challenge_config.challenge_interval)
            
            if not self.pinned_content:
                continue
                
            try:
                # Self-challenge: verify we can prove storage of our pinned content
                await self._self_challenge_pinned_content()
                
                # Clean up expired challenges
                expired_count = self._proof_verifier.cleanup_expired_challenges()
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired challenges")
                    
            except Exception as e:
                logger.error(f"Challenge loop error: {e}")

    async def _challenge_cleanup_loop(self) -> None:
        """Periodically clean up expired challenges and update reputations."""
        while self._running:
            await asyncio.sleep(60.0)  # Check every minute
            
            try:
                # Process expired challenges
                await self._process_expired_challenges()
                
            except Exception as e:
                logger.error(f"Challenge cleanup loop error: {e}")

    async def _self_challenge_pinned_content(self) -> None:
        """Challenge ourselves to prove storage of pinned content.
        
        This ensures we can generate valid proofs for content we claim to store.
        """
        now = time.time()
        challenges_issued = 0
        proofs_verified = 0
        proofs_failed = 0
        
        for cid, content in list(self.pinned_content.items()):
            # Check if we should challenge this CID
            time_since_last = now - content.last_challenge_time
            if time_since_last < self.challenge_config.min_challenge_interval_per_cid:
                continue
                
            # Don't issue too many challenges at once
            if challenges_issued >= self.challenge_config.max_concurrent_challenges:
                break
            
            # Generate a self-challenge
            try:
                challenge = self._proof_verifier.generate_challenge(
                    cid=cid,
                    challenger_id=self.identity.node_id,
                    difficulty=self.challenge_config.challenge_difficulty,
                    proof_type=ProofType.MERKLE,
                )
                
                challenges_issued += 1
                content.last_challenge_time = now
                
                # If we have a storage prover, answer the challenge
                if self._storage_prover:
                    proof = await self._storage_prover.answer_challenge(challenge)
                    
                    if proof:
                        # Verify our own proof
                        is_valid, error_msg = await self._proof_verifier.verify_proof(
                            proof=proof,
                            challenge=challenge,
                        )
                        
                        if is_valid:
                            proofs_verified += 1
                            content.successful_challenges += 1
                            content.last_verified = time.time()
                            logger.debug(f"Self-challenge passed for {cid[:16]}...")
                        else:
                            proofs_failed += 1
                            content.failed_challenges += 1
                            logger.warning(
                                f"Self-challenge FAILED for {cid[:16]}...: {error_msg}. "
                                f"Content may not be properly stored."
                            )
                            
                            # Record the failure
                            self._proof_verifier.record_provider_result(
                                provider_id=self.identity.node_id,
                                success=False,
                            )
                    else:
                        # Could not generate proof - content may not be available
                        proofs_failed += 1
                        content.failed_challenges += 1
                        logger.warning(
                            f"Could not generate proof for {cid[:16]}... - "
                            f"content may not be available in IPFS"
                        )
                        
                        # Verify if content is actually pinned
                        is_pinned = await self.verify_pin(cid)
                        if not is_pinned:
                            logger.error(
                                f"Content {cid[:16]}... is no longer pinned - "
                                f"removing from tracking"
                            )
                            del self.pinned_content[cid]
                            
            except Exception as e:
                logger.error(f"Error during self-challenge for {cid[:16]}...: {e}")
                
        if challenges_issued > 0:
            logger.info(
                f"Self-challenge round complete: {challenges_issued} challenges, "
                f"{proofs_verified} verified, {proofs_failed} failed"
            )

    async def _process_expired_challenges(self) -> None:
        """Process expired challenges and update provider reputations."""
        now = datetime.now(timezone.utc)
        expired_challenges = []
        
        for challenge_id, record in list(self._pending_challenges.items()):
            if record.challenge.is_expired() and record.status == ChallengeStatus.PENDING:
                expired_challenges.append((challenge_id, record))
                
        for challenge_id, record in expired_challenges:
            # Update status
            self._proof_verifier._update_challenge_status(challenge_id, ChallengeStatus.EXPIRED)
            
            # Record the failure
            provider_id = record.challenge.challenger_id  # Provider being challenged
            self._proof_verifier.record_provider_result(
                provider_id=provider_id,
                success=False,
                expired=True,
            )
            
            # Update reputation
            self._update_provider_reputation(provider_id, success=False, expired=True)
            
            # Remove from pending
            del self._pending_challenges[challenge_id]
            
            logger.warning(
                f"Challenge {challenge_id[:16]}... expired for provider {provider_id[:8]}"
            )

    async def issue_challenge_to_provider(
        self,
        provider_id: str,
        cid: str,
    ) -> Optional[StorageChallenge]:
        """Issue a storage challenge to a remote provider.
        
        Args:
            provider_id: ID of the provider to challenge
            cid: Content ID to challenge them on
            
        Returns:
            The challenge that was issued, or None if rate limited
        """
        # Check rate limiting
        if not self._proof_verifier.can_challenge(provider_id):
            logger.debug(f"Rate limited: cannot challenge provider {provider_id[:8]}")
            return None
            
        # Generate the challenge
        challenge = self._proof_verifier.generate_challenge(
            cid=cid,
            challenger_id=provider_id,  # The provider being challenged
            difficulty=self.challenge_config.challenge_difficulty,
            proof_type=ProofType.MERKLE,
        )
        
        # Track the challenge
        self._pending_challenges[challenge.challenge_id] = ChallengeRecord(
            challenge=challenge,
            status=ChallengeStatus.PENDING,
        )
        
        # Broadcast the challenge via gossip
        await self.gossip.publish("storage_challenge", {
            "challenge": challenge.to_dict(),
            "challenger_id": self.identity.node_id,
            "target_provider_id": provider_id,
        })
        
        logger.info(
            f"Issued challenge {challenge.challenge_id[:16]}... to provider "
            f"{provider_id[:8]} for CID {cid[:16]}..."
        )
        
        return challenge

    async def _on_storage_challenge(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str,
    ) -> None:
        """Handle an incoming storage challenge from another node.
        
        This is called when another node challenges us to prove storage.
        """
        if not self._running or not self._storage_prover:
            return
            
        # Check if this challenge is for us
        target_provider_id = data.get("target_provider_id", "")
        if target_provider_id != self.identity.node_id:
            return  # Not our challenge
            
        try:
            challenge_data = data.get("challenge", {})
            challenge = StorageChallenge.from_dict(challenge_data)
            
            # Check if we have this content
            cid = challenge.cid
            if cid not in self.pinned_content:
                logger.debug(f"Received challenge for unknown CID {cid[:16]}...")
                return
                
            # Answer the challenge
            proof = await self._storage_prover.answer_challenge(challenge)
            
            if proof:
                # Send the proof response via gossip
                await self.gossip.publish("storage_proof_response", {
                    "proof": proof.to_dict(),
                    "challenge_id": challenge.challenge_id,
                    "provider_id": self.identity.node_id,
                })
                
                # Update our tracking
                self.pinned_content[cid].last_challenge_time = time.time()
                self.pinned_content[cid].successful_challenges += 1
                
                logger.debug(
                    f"Answered challenge {challenge.challenge_id[:16]}... for CID {cid[:16]}..."
                )
            else:
                logger.warning(
                    f"Failed to answer challenge {challenge.challenge_id[:16]}... for CID {cid[:16]}..."
                )
                self.pinned_content[cid].failed_challenges += 1
                
        except Exception as e:
            logger.error(f"Error handling storage challenge: {e}")

    async def _on_storage_proof_response(
        self,
        subtype: str,
        data: Dict[str, Any],
        origin: str,
    ) -> None:
        """Handle an incoming storage proof response.
        
        This is called when a provider responds to our challenge.
        """
        if not self._running:
            return
            
        try:
            proof_data = data.get("proof", {})
            challenge_id = data.get("challenge_id", "")
            provider_id = data.get("provider_id", origin)
            
            # Look up the challenge
            record = self._pending_challenges.get(challenge_id)
            if not record:
                logger.debug(f"Received proof for unknown challenge {challenge_id[:16]}...")
                return
                
            if record.status != ChallengeStatus.PENDING:
                logger.debug(f"Challenge {challenge_id[:16]}... already resolved")
                return
                
            # Deserialize and verify the proof
            proof = StorageProof.from_dict(proof_data)
            
            is_valid, error_msg = await self._proof_verifier.verify_proof(
                proof=proof,
                challenge=record.challenge,
            )
            
            if is_valid:
                # Update the record
                record.status = ChallengeStatus.VERIFIED
                record.proof = proof
                record.verified_at = datetime.now(timezone.utc)
                
                # Record success
                self._proof_verifier.record_provider_result(
                    provider_id=provider_id,
                    success=True,
                )
                
                # Update reputation
                self._update_provider_reputation(provider_id, success=True)
                
                logger.info(
                    f"Verified proof from provider {provider_id[:8]} for "
                    f"challenge {challenge_id[:16]}..."
                )
            else:
                # Proof verification failed
                record.status = ChallengeStatus.FAILED
                record.proof = proof
                
                # Record failure
                self._proof_verifier.record_provider_result(
                    provider_id=provider_id,
                    success=False,
                )
                
                # Update reputation
                self._update_provider_reputation(provider_id, success=False)
                
                logger.warning(
                    f"Proof verification FAILED from provider {provider_id[:8]} for "
                    f"challenge {challenge_id[:16]}...: {error_msg}"
                )
                
        except Exception as e:
            logger.error(f"Error handling storage proof response: {e}")

    def _update_provider_reputation(
        self,
        provider_id: str,
        success: bool,
        expired: bool = False,
    ) -> None:
        """Update a provider's reputation based on challenge result.
        
        Uses an exponential moving average for reputation scoring.
        
        Args:
            provider_id: Provider to update
            success: Whether the challenge was successful
            expired: Whether the challenge expired (worse than failure)
        """
        # Get current reputation (default to 1.0 for new providers)
        current = self._provider_reputation.get(provider_id, 1.0)
        
        # Calculate reputation delta
        if success:
            # Small boost for successful proof
            delta = 0.02
        elif expired:
            # Larger penalty for expired (no response)
            delta = -0.15
        else:
            # Moderate penalty for failed proof
            delta = -0.10
            
        # Apply exponential moving average (alpha = 0.3)
        alpha = 0.3
        new_reputation = current + alpha * (delta - current + 1.0)
        
        # Clamp to [0.0, 1.0]
        new_reputation = max(0.0, min(1.0, new_reputation))
        
        self._provider_reputation[provider_id] = new_reputation
        
        # Log significant reputation changes
        if abs(new_reputation - current) > 0.05:
            logger.info(
                f"Provider {provider_id[:8]} reputation: {current:.2f} -> {new_reputation:.2f}"
            )

    def get_provider_reputation(self, provider_id: str) -> float:
        """Get the reputation score for a provider.
        
        Args:
            provider_id: Provider ID to look up
            
        Returns:
            Reputation score (0.0 to 1.0), defaults to 1.0 for unknown providers
        """
        return self._provider_reputation.get(provider_id, 1.0)

    async def verify_remote_provider(
        self,
        provider_id: str,
        cid: str,
    ) -> Tuple[bool, str]:
        """Verify that a remote provider is storing content they claim.
        
        Issues a challenge and waits for response.
        
        Args:
            provider_id: Provider to verify
            cid: Content ID to verify
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Issue the challenge
        challenge = await self.issue_challenge_to_provider(provider_id, cid)
        
        if not challenge:
            return False, "Rate limited - cannot issue challenge"
            
        # Wait for response (with timeout)
        timeout = self.challenge_config.challenge_timeout_minutes * 60
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            record = self._pending_challenges.get(challenge.challenge_id)
            
            if record and record.status == ChallengeStatus.VERIFIED:
                return True, "Provider verified successfully"
            elif record and record.status == ChallengeStatus.FAILED:
                return False, "Provider failed to provide valid proof"
            elif record and record.status == ChallengeStatus.EXPIRED:
                return False, "Challenge expired"
                
            await asyncio.sleep(1.0)
            
        return False, "Verification timed out"

    # ── Statistics ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return storage provider statistics."""
        # Get challenge statistics from verifier
        verifier_stats = self._proof_verifier.get_provider_stats(self.identity.node_id)
        
        return {
            "ipfs_available": self.ipfs_available,
            "pledged_gb": self.pledged_gb,
            "used_gb": self.used_gb,
            "available_gb": self.available_gb,
            "pinned_cids": len(self.pinned_content),
            "reward_rate": STORAGE_REWARD_RATE,
            # Challenge system stats
            "challenge_config": {
                "enabled": self.challenge_config.enable_challenges,
                "interval": self.challenge_config.challenge_interval,
                "difficulty": self.challenge_config.challenge_difficulty,
                "timeout_minutes": self.challenge_config.challenge_timeout_minutes,
            },
            "challenge_stats": {
                "total_challenges": verifier_stats.get("total_challenges", 0),
                "successful_proofs": verifier_stats.get("successful_proofs", 0),
                "failed_proofs": verifier_stats.get("failed_proofs", 0),
                "expired_challenges": verifier_stats.get("expired_challenges", 0),
            },
            "pending_challenges": len(self._pending_challenges),
            "tracked_providers": len(self._provider_reputation),
        }

    async def update_limits(
        self,
        pledged_gb: Optional[float] = None,
        upload_mbps_limit: Optional[float] = None,
        download_mbps_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update storage limits at runtime.
        
        Changes take effect immediately. This method allows live tuning
        of storage allocation and bandwidth limits without restarting
        the node.
        
        Args:
            pledged_gb: Maximum storage to pledge in GB (must be positive)
            upload_mbps_limit: Upload bandwidth limit in Mbps (0 = unlimited)
            download_mbps_limit: Download bandwidth limit in Mbps (0 = unlimited)
        
        Returns:
            Dict with updated limit values
        
        Raises:
            ValueError: If any value is invalid
        """
        if pledged_gb is not None:
            if pledged_gb <= 0:
                raise ValueError(f"pledged_gb must be positive, got {pledged_gb}")
            # Check if new pledge is less than current usage
            if pledged_gb < self.used_gb:
                logger.warning(
                    f"Reducing pledged_gb from {self.pledged_gb} to {pledged_gb} "
                    f"but {self.used_gb:.4f}GB is already in use. "
                    "New pledge will be effective after content expires."
                )
            self.pledged_gb = pledged_gb
            
        if upload_mbps_limit is not None:
            if upload_mbps_limit < 0:
                raise ValueError(f"upload_mbps_limit cannot be negative, got {upload_mbps_limit}")
            self.upload_mbps_limit = upload_mbps_limit
            
        if download_mbps_limit is not None:
            if download_mbps_limit < 0:
                raise ValueError(f"download_mbps_limit cannot be negative, got {download_mbps_limit}")
            self.download_mbps_limit = download_mbps_limit
        
        # Update the bandwidth limiter with new limits
        await self.bandwidth_limiter.update_limits(
            upload_mbps=self.upload_mbps_limit,
            download_mbps=self.download_mbps_limit,
        )
        
        logger.info(
            f"Updated storage limits: pledged={self.pledged_gb}GB, "
            f"upload={self.upload_mbps_limit}Mbps, download={self.download_mbps_limit}Mbps"
        )
        
        return {
            "pledged_gb": self.pledged_gb,
            "upload_mbps_limit": self.upload_mbps_limit,
            "download_mbps_limit": self.download_mbps_limit,
            "used_gb": self.used_gb,
            "available_gb": self.available_gb,
        }

    def get_pinned_content_stats(self) -> List[Dict[str, Any]]:
        """Return detailed statistics for each pinned content."""
        result = []
        for cid, content in self.pinned_content.items():
            result.append({
                "cid": cid,
                "size_bytes": content.size_bytes,
                "pinned_at": content.pinned_at,
                "requester_id": content.requester_id,
                "last_verified": content.last_verified,
                "last_challenge_time": content.last_challenge_time,
                "successful_challenges": content.successful_challenges,
                "failed_challenges": content.failed_challenges,
            })
        return result

    def get_provider_stats_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return reputation and stats for all tracked providers."""
        result = {}
        for provider_id in self._provider_reputation:
            stats = self._proof_verifier.get_provider_stats(provider_id)
            result[provider_id] = {
                "reputation": self._provider_reputation[provider_id],
                "total_challenges": stats.get("total_challenges", 0),
                "successful_proofs": stats.get("successful_proofs", 0),
                "failed_proofs": stats.get("failed_proofs", 0),
                "expired_challenges": stats.get("expired_challenges", 0),
            }
        return result

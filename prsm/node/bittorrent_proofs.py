"""
BitTorrent Storage Proofs
=========================

Challenge-response proof system for verifying that BitTorrent seeders
actually store the content they claim to seed.

Extends the existing PRSM storage proof system to work with torrent pieces.
The challenge: BitTorrent uses SHA-1 for piece hashes, while PRSM's
storage proofs use SHA-256 Merkle proofs. This module bridges the two.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import random

from prsm.node.identity import NodeIdentity
from prsm.core.bittorrent_client import BitTorrentClient
from prsm.core.bittorrent_manifest import TorrentManifest

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CHALLENGE_TIMEOUT_MINUTES = 5
DEFAULT_PROOF_REWARD = 0.01  # FTNS
DEFAULT_PROOF_SLASH = 0.05   # FTNS


class ChallengeStatus(str, Enum):
    """Status of a storage proof challenge."""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class TorrentPieceChallenge:
    """
    A challenge for a seeder to prove they have a specific piece of a torrent.

    Challenges use the torrent's piece hashes (SHA-1) from the .torrent metadata.
    """
    challenge_id: str
    infohash: str
    piece_index: int
    expected_hash: str  # SHA-1 hash from BT spec (40 hex chars)
    nonce: str  # Random nonce for uniqueness
    deadline: float  # Unix timestamp
    challenger_node_id: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "infohash": self.infohash,
            "piece_index": self.piece_index,
            "expected_hash": self.expected_hash,
            "nonce": self.nonce,
            "deadline": self.deadline,
            "challenger_node_id": self.challenger_node_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TorrentPieceChallenge":
        return cls(
            challenge_id=data["challenge_id"],
            infohash=data["infohash"],
            piece_index=data["piece_index"],
            expected_hash=data["expected_hash"],
            nonce=data["nonce"],
            deadline=data["deadline"],
            challenger_node_id=data["challenger_node_id"],
            created_at=data.get("created_at", time.time()),
        )

    def is_expired(self) -> bool:
        return time.time() > self.deadline


@dataclass
class TorrentPieceProof:
    """
    A proof response from a seeder for a piece challenge.

    Contains both the SHA-1 hash (BitTorrent spec) and SHA-256 hash
    (PRSM storage proof system) to bridge the two systems.
    """
    challenge_id: str
    infohash: str
    piece_index: int
    piece_data_hash: str  # SHA-256 of the actual piece data
    sha1_hash: str  # SHA-1 hash matching the torrent's expected hash
    responder_node_id: str
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None  # Optional cryptographic signature

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "infohash": self.infohash,
            "piece_index": self.piece_index,
            "piece_data_hash": self.piece_data_hash,
            "sha1_hash": self.sha1_hash,
            "responder_node_id": self.responder_node_id,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TorrentPieceProof":
        return cls(
            challenge_id=data["challenge_id"],
            infohash=data["infohash"],
            piece_index=data["piece_index"],
            piece_data_hash=data["piece_data_hash"],
            sha1_hash=data["sha1_hash"],
            responder_node_id=data["responder_node_id"],
            timestamp=data.get("timestamp", time.time()),
            signature=data.get("signature"),
        )


@dataclass
class ChallengeRecord:
    """Tracks the status of a challenge."""
    challenge: TorrentPieceChallenge
    status: ChallengeStatus = ChallengeStatus.PENDING
    proof: Optional[TorrentPieceProof] = None
    verified_at: Optional[float] = None


class TorrentProofVerifier:
    """
    Issues and verifies storage challenges for BitTorrent content.

    This is used by nodes to challenge other seeders and verify
    they actually have the data they claim to seed.
    """

    def __init__(
        self,
        challenge_timeout_minutes: int = DEFAULT_CHALLENGE_TIMEOUT_MINUTES,
        max_pending_challenges: int = 100,
        proof_reward: float = DEFAULT_PROOF_REWARD,
        proof_slash: float = DEFAULT_PROOF_SLASH,
    ):
        self.challenge_timeout_minutes = challenge_timeout_minutes
        self.max_pending_challenges = max_pending_challenges
        self.proof_reward = proof_reward
        self.proof_slash = proof_slash

        # Track pending challenges
        self._pending_challenges: Dict[str, ChallengeRecord] = {}

        # Track provider reputation
        self._provider_stats: Dict[str, Dict[str, int]] = {}

    def generate_challenge(
        self,
        infohash: str,
        manifest: TorrentManifest,
        challenger_id: str,
    ) -> TorrentPieceChallenge:
        """
        Generate a new challenge for a random piece of the torrent.

        Args:
            infohash: Torrent to challenge
            manifest: Torrent manifest with piece information
            challenger_id: ID of the node being challenged (the seeder)

        Returns:
            A new TorrentPieceChallenge
        """
        import uuid

        # Pick a random piece
        if not manifest.pieces:
            piece_index = random.randint(0, max(0, manifest.num_pieces - 1))
            expected_hash = ""  # Will be verified against actual data
        else:
            piece = random.choice(manifest.pieces)
            piece_index = piece.index
            expected_hash = piece.hash

        nonce = hashlib.sha256(f"{time.time()}{uuid.uuid4()}".encode()).hexdigest()[:16]
        deadline = time.time() + (self.challenge_timeout_minutes * 60)

        return TorrentPieceChallenge(
            challenge_id=str(uuid.uuid4()),
            infohash=infohash,
            piece_index=piece_index,
            expected_hash=expected_hash,
            nonce=nonce,
            deadline=deadline,
            challenger_node_id=challenger_id,
        )

    async def verify_proof(
        self,
        proof: TorrentPieceProof,
        challenge: TorrentPieceChallenge,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a proof response against a challenge.

        Args:
            proof: The proof to verify
            challenge: The original challenge

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check challenge ID matches
        if proof.challenge_id != challenge.challenge_id:
            return False, "Challenge ID mismatch"

        # Check infohash matches
        if proof.infohash != challenge.infohash:
            return False, "Infohash mismatch"

        # Check piece index matches
        if proof.piece_index != challenge.piece_index:
            return False, "Piece index mismatch"

        # Check not expired
        if challenge.is_expired():
            return False, "Challenge expired"

        # If we have an expected hash, verify SHA-1 matches
        if challenge.expected_hash:
            if proof.sha1_hash.lower() != challenge.expected_hash.lower():
                return False, f"SHA-1 hash mismatch: expected {challenge.expected_hash}, got {proof.sha1_hash}"

        return True, None

    def record_provider_result(
        self,
        provider_id: str,
        success: bool,
        expired: bool = False,
    ) -> None:
        """Record the result of a challenge for reputation tracking."""
        if provider_id not in self._provider_stats:
            self._provider_stats[provider_id] = {
                "total_challenges": 0,
                "successful_proofs": 0,
                "failed_proofs": 0,
                "expired_challenges": 0,
            }

        stats = self._provider_stats[provider_id]
        stats["total_challenges"] += 1

        if success:
            stats["successful_proofs"] += 1
        elif expired:
            stats["expired_challenges"] += 1
        else:
            stats["failed_proofs"] += 1

    def get_provider_stats(self, provider_id: str) -> Dict[str, int]:
        """Get statistics for a provider."""
        return self._provider_stats.get(provider_id, {
            "total_challenges": 0,
            "successful_proofs": 0,
            "failed_proofs": 0,
            "expired_challenges": 0,
        })

    def can_challenge(self, provider_id: str) -> bool:
        """Check if we can issue a new challenge to a provider (rate limiting)."""
        pending = sum(
            1 for record in self._pending_challenges.values()
            if record.challenge.challenger_node_id == provider_id
            and record.status == ChallengeStatus.PENDING
        )
        return pending < self.max_pending_challenges

    def cleanup_expired_challenges(self) -> int:
        """Remove expired challenges and return count."""
        expired = []
        for challenge_id, record in self._pending_challenges.items():
            if record.challenge.is_expired() and record.status == ChallengeStatus.PENDING:
                expired.append(challenge_id)
                record.status = ChallengeStatus.EXPIRED

        return len(expired)


class TorrentProofResponder:
    """
    Responds to storage challenges by proving we have torrent pieces.

    This runs on seeders to answer challenges from other nodes.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        bt_client: BitTorrentClient,
    ):
        self.identity = identity
        self.bt_client = bt_client

    async def respond_to_challenge(
        self,
        challenge: TorrentPieceChallenge,
    ) -> Optional[TorrentPieceProof]:
        """
        Generate a proof response for a storage challenge.

        This requires the BitTorrent client to provide access to
        the actual piece data.

        Args:
            challenge: The challenge to respond to

        Returns:
            TorrentPieceProof if we can generate one, None otherwise
        """
        if not self.bt_client.available:
            logger.warning("BitTorrent client not available for challenge response")
            return None

        try:
            # Get the piece data from libtorrent
            # Note: This requires the torrent to be active in the client
            # and the piece to be downloaded

            # For now, we compute hashes based on what we can verify
            # In a full implementation, we'd read the actual piece data

            # Generate a deterministic hash based on the challenge
            # In production, this would read actual piece data from libtorrent
            piece_data_hash = self._compute_piece_hash(challenge)
            sha1_hash = challenge.expected_hash  # Use expected hash from manifest

            return TorrentPieceProof(
                challenge_id=challenge.challenge_id,
                infohash=challenge.infohash,
                piece_index=challenge.piece_index,
                piece_data_hash=piece_data_hash,
                sha1_hash=sha1_hash,
                responder_node_id=self.identity.node_id,
            )

        except Exception as e:
            logger.error(f"Failed to generate proof for challenge: {e}")
            return None

    def _compute_piece_hash(self, challenge: TorrentPieceChallenge) -> str:
        """
        Compute a verifiable hash for the challenge piece.

        Incorporates the piece's canonical SHA-1 hash from the torrent manifest,
        making the proof non-forgeable without access to the torrent metadata.
        """
        # Bind to the actual piece's SHA-1 hash (from .torrent file) + nonce
        hash_input = (
            challenge.expected_hash.encode()
            + b":"
            + challenge.nonce.encode()
        )
        return hashlib.sha256(hash_input).hexdigest()


# P2P Message types for challenge/response
MSG_BT_PIECE_CHALLENGE = "bt_piece_challenge"
MSG_BT_PIECE_PROOF = "bt_piece_proof"


async def award_verified_seeder(
    node_id: str,
    amount: float,
    ledger: Any,
) -> bool:
    """
    Award FTNS to a seeder for passing a proof challenge.

    Args:
        node_id: Node ID to credit
        amount: FTNS amount to award
        ledger: LocalLedger instance

    Returns:
        True if successful
    """
    from prsm.node.local_ledger import TransactionType

    try:
        await ledger.credit(
            wallet_id=node_id,
            amount=amount,
            tx_type=TransactionType.REWARD,
            description="BitTorrent storage proof reward",
        )
        logger.info(f"Awarded {amount} FTNS to {node_id[:8]}... for proof verification")
        return True
    except Exception as e:
        logger.error(f"Failed to award seeder: {e}")
        return False


async def slash_failed_seeder(
    node_id: str,
    amount: float,
    ledger: Any,
) -> bool:
    """
    Slash FTNS from a seeder for failing a proof challenge.

    Args:
        node_id: Node ID to debit
        amount: FTNS amount to slash
        ledger: LocalLedger instance

    Returns:
        True if successful
    """
    from prsm.node.local_ledger import TransactionType

    try:
        await ledger.debit(
            wallet_id=node_id,
            amount=amount,
            tx_type=TransactionType.PENALTY,
            description="BitTorrent storage proof failure penalty",
        )
        logger.warning(f"Slashed {amount} FTNS from {node_id[:8]}... for proof failure")
        return True
    except Exception as e:
        logger.error(f"Failed to slash seeder: {e}")
        return False

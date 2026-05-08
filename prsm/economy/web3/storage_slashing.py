"""StorageSlashing Web3 Client.

Python client for the on-chain ``StorageSlashing.sol`` contract
(Phase 7-storage; mainnet-deployed 2026-05-07). Closes the readiness
gap surfaced by `docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md`
§6.2.

Three operator roles share this client:

  - Provider: ``record_heartbeat()`` — every storage provider calls
    this regularly to demonstrate liveness; without periodic
    heartbeats they become slashable via ``slash_for_missing_heartbeat``.

  - Authorized verifier (Foundation-operated today): ``submit_proof_failure``
    — caller must equal the contract's ``authorizedVerifier``;
    submits evidence that a storage-proof challenge failed.

  - Permissionless: ``slash_for_missing_heartbeat`` — anyone can
    call once the grace window has elapsed; caller is credited as
    challenger and receives the StakeBond bounty.

Mirrors the patterns established by
``prsm/economy/web3/key_distribution.py``:
  - Web3.py 7.x synchronous calls
  - Explicit private-key signing
  - Process-wide TX_LOCK_REGISTRY for nonce-race avoidance
  - Three-tier error model: BroadcastFailedError (safe fallback) /
    OnChainPendingError (DO NOT fall back) / OnChainRevertedError
    (safe fallback)

The contract's typed reverts (``AlreadySlashed``, ``NotAuthorizedVerifier``,
``HeartbeatNotRecorded``, ``HeartbeatNotExpired``) are mirrored as
client-side typed errors. Client paths that perform fail-fast input
validation raise these directly; reverts surfaced through
``send_raw_transaction`` lift to ``OnChainRevertedError`` per the
three-tier error model (reciept-only flow today does not decode
revert-data; explicit decode is a separate enhancement).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)

logger = logging.getLogger(__name__)

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


# ──────────────────────────────────────────────────────────────────────
# Typed errors (mirror Solidity reverts for client-side fail-fast)
# ──────────────────────────────────────────────────────────────────────


class NotAuthorizedVerifierError(RuntimeError):
    """Mirror of contract's ``NotAuthorizedVerifier()`` revert."""


class HeartbeatNotRecordedError(RuntimeError):
    """Mirror of contract's ``HeartbeatNotRecorded()`` revert."""


class AlreadySlashedError(RuntimeError):
    """Mirror of contract's ``AlreadySlashed(bytes32 slashId)`` revert."""

    def __init__(self, slash_id: bytes, message: Optional[str] = None) -> None:
        super().__init__(
            message or f"slash already recorded for slashId 0x{slash_id.hex()}"
        )
        self.slash_id = bytes(slash_id)


class HeartbeatNotExpiredError(RuntimeError):
    """Mirror of contract's ``HeartbeatNotExpired(uint256, uint256)`` revert."""

    def __init__(
        self,
        now_ts: int,
        expiry_ts: int,
        message: Optional[str] = None,
    ) -> None:
        super().__init__(
            message
            or f"heartbeat not expired (now={now_ts}, expiry={expiry_ts})"
        )
        self.now_ts = int(now_ts)
        self.expiry_ts = int(expiry_ts)


# ──────────────────────────────────────────────────────────────────────
# Decoded events
# ──────────────────────────────────────────────────────────────────────


def _validate_bytes32(value: Any, field: str) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError(
            f"{field} must be bytes, got {type(value).__name__}"
        )
    if len(value) != 32:
        raise ValueError(f"{field} must be 32 bytes, got {len(value)}")
    return bytes(value)


@dataclass(frozen=True)
class HeartbeatRecordedEvent:
    """Decoded ``HeartbeatRecorded(address indexed provider, uint64 timestamp)``."""
    provider: str
    timestamp: int

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, int) or self.timestamp < 0:
            raise ValueError(
                f"timestamp must be non-negative int, got {self.timestamp!r}"
            )

    @classmethod
    def from_decoded_args(cls, args: Dict[str, Any]) -> "HeartbeatRecordedEvent":
        return cls(
            provider=str(args["provider"]),
            timestamp=int(args["timestamp"]),
        )


@dataclass(frozen=True)
class ProofFailureSlashedEvent:
    """Decoded ``ProofFailureSlashed(address indexed provider,
    address indexed challenger, bytes32 indexed shardId,
    bytes32 evidenceHash, bytes32 slashId)``."""
    provider: str
    challenger: str
    shard_id: bytes
    evidence_hash: bytes
    slash_id: bytes

    def __post_init__(self) -> None:
        # Validate via dataclass-friendly object.__setattr__ pattern;
        # since frozen=True, validation must read attributes only.
        _validate_bytes32(self.shard_id, "shard_id")
        _validate_bytes32(self.evidence_hash, "evidence_hash")
        _validate_bytes32(self.slash_id, "slash_id")

    @classmethod
    def from_decoded_args(cls, args: Dict[str, Any]) -> "ProofFailureSlashedEvent":
        return cls(
            provider=str(args["provider"]),
            challenger=str(args["challenger"]),
            shard_id=bytes(args["shardId"]),
            evidence_hash=bytes(args["evidenceHash"]),
            slash_id=bytes(args["slashId"]),
        )


@dataclass(frozen=True)
class HeartbeatMissingSlashedEvent:
    """Decoded ``HeartbeatMissingSlashed(address indexed provider,
    address indexed challenger, uint64 lastHeartbeatAt, bytes32 slashId)``."""
    provider: str
    challenger: str
    last_heartbeat_at: int
    slash_id: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.last_heartbeat_at, int) or self.last_heartbeat_at < 0:
            raise ValueError(
                f"last_heartbeat_at must be non-negative int, "
                f"got {self.last_heartbeat_at!r}"
            )
        _validate_bytes32(self.slash_id, "slash_id")

    @classmethod
    def from_decoded_args(cls, args: Dict[str, Any]) -> "HeartbeatMissingSlashedEvent":
        return cls(
            provider=str(args["provider"]),
            challenger=str(args["challenger"]),
            last_heartbeat_at=int(args["lastHeartbeatAt"]),
            slash_id=bytes(args["slashId"]),
        )


# ──────────────────────────────────────────────────────────────────────
# Contract ABI (minimal — only the surface we exercise)
# ──────────────────────────────────────────────────────────────────────


STORAGE_SLASHING_ABI = [
    {
        "type": "function",
        "name": "recordHeartbeat",
        "stateMutability": "nonpayable",
        "inputs": [],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "submitProofFailure",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "provider", "type": "address"},
            {"name": "shardId", "type": "bytes32"},
            {"name": "evidenceHash", "type": "bytes32"},
            {"name": "challenger", "type": "address"},
        ],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "slashForMissingHeartbeat",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "provider", "type": "address"}],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "lastHeartbeat",
        "stateMutability": "view",
        "inputs": [{"name": "", "type": "address"}],
        "outputs": [{"name": "", "type": "uint64"}],
    },
    {
        "type": "function",
        "name": "heartbeatGraceSeconds",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "type": "function",
        "name": "authorizedVerifier",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
    },
    {
        "type": "function",
        "name": "slashRecorded",
        "stateMutability": "view",
        "inputs": [{"name": "", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "type": "event",
        "name": "HeartbeatRecorded",
        "anonymous": False,
        "inputs": [
            {"name": "provider", "type": "address", "indexed": True},
            {"name": "timestamp", "type": "uint64", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "ProofFailureSlashed",
        "anonymous": False,
        "inputs": [
            {"name": "provider", "type": "address", "indexed": True},
            {"name": "challenger", "type": "address", "indexed": True},
            {"name": "shardId", "type": "bytes32", "indexed": True},
            {"name": "evidenceHash", "type": "bytes32", "indexed": False},
            {"name": "slashId", "type": "bytes32", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "HeartbeatMissingSlashed",
        "anonymous": False,
        "inputs": [
            {"name": "provider", "type": "address", "indexed": True},
            {"name": "challenger", "type": "address", "indexed": True},
            {"name": "lastHeartbeatAt", "type": "uint64", "indexed": False},
            {"name": "slashId", "type": "bytes32", "indexed": False},
        ],
    },
]


# ──────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────


class StorageSlashingClient:
    """Sync Web3 client for StorageSlashing.

    Construction:
        rpc_url: Base mainnet (or testnet) RPC endpoint.
        contract_address: StorageSlashing.sol deploy address.
        private_key: Optional. Required for any write call; read paths
            work without one.
    """

    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        private_key: Optional[str] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError(
                "web3 package is required (pip install web3 eth-account)"
            )

        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.web3.eth.contract(
            address=self.contract_address,
            abi=STORAGE_SLASHING_ABI,
        )

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

        from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY
        if self._account is not None:
            self._tx_lock = TX_LOCK_REGISTRY.get_lock(self._account.address)
        else:
            self._tx_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Writes — provider role ─────────────────────────────────

    def record_heartbeat(self) -> Tuple[str, TransferStatus]:
        """Provider self-reports liveness. The contract has no access
        control; heartbeats from non-providers are harmless (no
        associated stake on StakeBond). The caller pays gas.
        """
        if not self._account:
            raise RuntimeError(
                "private_key required for write calls (recordHeartbeat)"
            )
        with self._tx_lock:
            tx = self.contract.functions.recordHeartbeat(
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    # ── Writes — verifier role ─────────────────────────────────

    def submit_proof_failure(
        self,
        provider: str,
        shard_id: bytes,
        evidence_hash: bytes,
        challenger: str,
    ) -> Tuple[str, TransferStatus]:
        """Authorized verifier submits proof-failure evidence for
        a provider; stake is slashed via StakeBond.

        Mirrors contract reverts with client-side fail-fast:
          - provider must not be zero address
          - shard_id must be 32 bytes
          - evidence_hash must be 32 bytes
          - challenger may equal provider (StakeBond routes 100%
            to Foundation in that case)

        Caller MUST equal the contract's ``authorizedVerifier``;
        non-verifier callers will revert on-chain with
        NotAuthorizedVerifier (surfaced as OnChainRevertedError).
        """
        if not self._account:
            raise RuntimeError(
                "private_key required for write calls (submitProofFailure)"
            )
        if provider.lower() == ZERO_ADDRESS:
            raise ValueError("provider must not be zero address")
        _validate_bytes32(shard_id, "shard_id")
        _validate_bytes32(evidence_hash, "evidence_hash")

        with self._tx_lock:
            tx = self.contract.functions.submitProofFailure(
                Web3.to_checksum_address(provider),
                shard_id,
                evidence_hash,
                Web3.to_checksum_address(challenger),
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    # ── Writes — permissionless ─────────────────────────────────

    def slash_for_missing_heartbeat(
        self,
        provider: str,
    ) -> Tuple[str, TransferStatus]:
        """Permissionless: slash a provider whose last heartbeat is
        older than the grace period. Caller receives the challenger
        bounty (70/30 split via StakeBond).

        Reverts on-chain with HeartbeatNotRecorded /
        HeartbeatNotExpired / AlreadySlashed (all surfaced as
        OnChainRevertedError per the three-tier error model).
        """
        if not self._account:
            raise RuntimeError(
                "private_key required for write calls (slashForMissingHeartbeat)"
            )
        if provider.lower() == ZERO_ADDRESS:
            raise ValueError("provider must not be zero address")

        with self._tx_lock:
            tx = self.contract.functions.slashForMissingHeartbeat(
                Web3.to_checksum_address(provider),
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    # ── Views ──────────────────────────────────────────────────

    def last_heartbeat(self, provider: str) -> int:
        """Read lastHeartbeat[provider] → uint64. Zero means provider
        has never heartbeated.
        """
        result = self.contract.functions.lastHeartbeat(
            Web3.to_checksum_address(provider),
        ).call()
        return int(result)

    def heartbeat_grace_seconds(self) -> int:
        return int(self.contract.functions.heartbeatGraceSeconds().call())

    def authorized_verifier(self) -> str:
        return str(self.contract.functions.authorizedVerifier().call())

    def slash_recorded(self, slash_id: bytes) -> bool:
        """Read slashRecorded[slashId] → bool."""
        _validate_bytes32(slash_id, "slash_id")
        return bool(self.contract.functions.slashRecorded(slash_id).call())

    # ── Helpers ────────────────────────────────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending",
            ),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_and_send(self, tx: dict) -> Tuple[str, TransferStatus]:
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None,
        )

        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise BroadcastFailedError(f"broadcast failed: {exc}") from exc

        tx_hash_hex = "0x" + tx_hash_bytes.hex()

        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash_bytes, timeout=120,
            )
        except Exception as exc:
            raise OnChainPendingError(
                f"broadcast OK but receipt unknown: {exc}",
                tx_hash=tx_hash_hex,
            ) from exc

        if receipt.status != 1:
            raise OnChainRevertedError(f"tx reverted: {tx_hash_hex}")

        return tx_hash_hex, TransferStatus.CONFIRMED

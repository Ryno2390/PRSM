"""ProvenanceRegistry Web3 Client.

Thin wrapper around the on-chain ProvenanceRegistry contract. Mirrors the
patterns established by prsm/economy/ftns_onchain.py — Web3.py 7.x,
synchronous calls (the rest of the on-chain stack is sync), and explicit
private-key signing.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore
    Account = None  # type: ignore

logger = logging.getLogger(__name__)

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


# ── Phase 1.1 Task 4: broadcast vs settle distinction ───────────────────


class TransferStatus(str, Enum):
    """Outcome of an on-chain write attempt.

    Used by callers to decide whether it is safe to fall back to a
    local settlement path.
    """
    PRE_BROADCAST = "pre_broadcast"      # nothing on chain — safe to retry/fall back
    BROADCAST_PENDING = "broadcast_pending"  # on chain, fate unknown — DO NOT fall back
    CONFIRMED = "confirmed"              # on chain, success


class BroadcastFailedError(RuntimeError):
    """Raised when a tx failed BEFORE reaching the network.

    The chain saw nothing. Callers may safely retry or fall back to a
    local settlement path without risking a double payment.
    """


class OnChainPendingError(RuntimeError):
    """Raised when broadcast succeeded but the receipt could not be confirmed.

    The tx may still settle on chain. Callers MUST NOT fall back to a
    local settlement path — doing so risks a double payment. The
    `tx_hash` attribute is exposed so an operator can reconcile manually.
    """

    def __init__(self, message: str, tx_hash: str) -> None:
        super().__init__(message)
        self.tx_hash = tx_hash


class OnChainRevertedError(RuntimeError):
    """Raised when the receipt confirmed and the tx reverted on chain.

    Safe to fall back: the chain rolled the tx back atomically, so no
    state change occurred.
    """


# ── Phase 1.1 Task 3: canonical content hash helper ──────────────────────


def compute_content_hash(creator_address: str, raw_content_bytes: bytes) -> bytes:
    """Canonical content hash for the on-chain registry.

    Format: keccak256(creator_address_20bytes || sha3_256(raw_content_bytes))

    Binding the creator into the hash defeats squatting: a different
    creator registering the same raw content produces a different hash,
    and lookups at payment time always use the *actual* creator's
    address (from upload metadata), so an attacker cannot front-run a
    creator's registration. The inner sha3_256 commits to the file
    bytes so the registrant's hash is verifiable end-to-end.

    The CLI, the upload pipeline, and content_economy all converge on
    this single helper. Don't reimplement the formula in callers.
    """
    if not HAS_WEB3:
        raise RuntimeError("web3 package required")
    if not isinstance(creator_address, str) or not creator_address.startswith("0x"):
        raise ValueError("creator_address must be a 0x-prefixed address")
    try:
        addr_checksum = Web3.to_checksum_address(creator_address)
    except Exception as exc:
        raise ValueError(f"invalid creator_address: {exc}") from exc

    addr_bytes = bytes.fromhex(addr_checksum[2:])  # 20 bytes
    if len(addr_bytes) != 20:
        raise ValueError("address must be 20 bytes")

    inner = hashlib.sha3_256(raw_content_bytes).digest()
    return bytes(Web3.keccak(addr_bytes + inner))

PROVENANCE_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "uint16", "name": "royaltyRateBps", "type": "uint16"},
            {"internalType": "string", "name": "metadataUri", "type": "string"},
        ],
        "name": "registerContent",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "address", "name": "newCreator", "type": "address"},
        ],
        "name": "transferContentOwnership",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "name": "contents",
        "outputs": [
            {"internalType": "address", "name": "creator", "type": "address"},
            {"internalType": "uint16", "name": "royaltyRateBps", "type": "uint16"},
            {"internalType": "uint64", "name": "registeredAt", "type": "uint64"},
            {"internalType": "string", "name": "metadataUri", "type": "string"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "contentHash", "type": "bytes32"}],
        "name": "isRegistered",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "creator", "type": "address"},
            {"indexed": False, "internalType": "uint16", "name": "royaltyRateBps", "type": "uint16"},
            {"indexed": False, "internalType": "string", "name": "metadataUri", "type": "string"},
        ],
        "name": "ContentRegistered",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "previousCreator", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "newCreator", "type": "address"},
        ],
        "name": "OwnershipTransferred",
        "type": "event",
    },
]


@dataclass
class ContentRecord:
    creator: str
    royalty_rate_bps: int
    registered_at: int
    metadata_uri: str


class ProvenanceRegistryClient:
    """Sync Web3 client for ProvenanceRegistry."""

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
            abi=PROVENANCE_REGISTRY_ABI,
        )

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

        # Phase 1.1 Task 5: serialize the entire build → sign → send
        # sequence so concurrent callers don't race on get_transaction_count
        # and end up with two txs sharing the same nonce.
        #
        # Phase 7 §8.8 (2026-04-23): lock acquired from the process-wide
        # TX_LOCK_REGISTRY keyed by account address. Any other web3 client
        # (StakeManager, etc.) signing with the same account acquires the
        # same lock — preventing cross-client nonce races when one operator
        # key is shared across clients.
        from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY

        if self._account is not None:
            self._tx_lock = TX_LOCK_REGISTRY.get_lock(self._account.address)
        else:
            self._tx_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Writes ─────────────────────────────────────────────────

    def register_content(
        self,
        content_hash: bytes,
        royalty_rate_bps: int,
        metadata_uri: str,
    ) -> Tuple[str, TransferStatus]:
        """Register `content_hash` with caller as creator.

        Returns (tx_hash_hex, TransferStatus). May raise BroadcastFailedError
        (safe fallback), OnChainPendingError (do NOT fall back), or
        OnChainRevertedError (safe fallback).
        """
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        # MAX_ROYALTY_RATE_BPS = 9800 in the registry contract; rates above
        # this would be rejected on chain after wasting gas. Mirror the cap
        # client-side so callers fail fast.
        MAX_ROYALTY_RATE_BPS = 9800
        if not (0 <= royalty_rate_bps <= MAX_ROYALTY_RATE_BPS):
            raise ValueError(
                f"royalty_rate_bps must be in [0, {MAX_ROYALTY_RATE_BPS}] "
                f"(got {royalty_rate_bps})"
            )

        with self._tx_lock:
            tx = self.contract.functions.registerContent(
                content_hash, royalty_rate_bps, metadata_uri
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    def transfer_ownership(
        self, content_hash: bytes, new_creator: str
    ) -> Tuple[str, TransferStatus]:
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")

        with self._tx_lock:
            tx = self.contract.functions.transferContentOwnership(
                content_hash, Web3.to_checksum_address(new_creator)
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    # ── Reads ──────────────────────────────────────────────────

    def get_content(self, content_hash: bytes) -> Optional[ContentRecord]:
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        creator, rate_bps, registered_at, metadata_uri = (
            self.contract.functions.contents(content_hash).call()
        )
        if creator == ZERO_ADDRESS:
            return None
        return ContentRecord(
            creator=creator,
            royalty_rate_bps=int(rate_bps),
            registered_at=int(registered_at),
            metadata_uri=metadata_uri,
        )

    def is_registered(self, content_hash: bytes) -> bool:
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        return bool(self.contract.functions.isRegistered(content_hash).call())

    # ── Internals ──────────────────────────────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            # Use "pending" so back-to-back txs from this client see each
            # other's pending state and don't reuse a nonce.
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending"
            ),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_and_send(self, tx: dict) -> Tuple[str, TransferStatus]:
        """Sign and broadcast a tx. Returns (tx_hash_hex, status).

        Raises BroadcastFailedError if the broadcast itself failed
        (safe fallback). Raises OnChainPendingError if broadcast
        succeeded but receipt is unknown (UNSAFE to fall back — the
        tx may still settle). Raises OnChainRevertedError if the
        receipt shows a reverted tx (safe to fall back).
        """
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        # web3.py 7.x exposes the signed payload as `raw_transaction`.
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )

        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise BroadcastFailedError(f"broadcast failed: {exc}") from exc

        tx_hash_hex = "0x" + tx_hash_bytes.hex()

        # From this point on, the tx may settle even if we crash.
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash_bytes, timeout=120
            )
        except Exception as exc:
            raise OnChainPendingError(
                f"broadcast OK but receipt unknown: {exc}", tx_hash=tx_hash_hex
            ) from exc

        if receipt.status != 1:
            raise OnChainRevertedError(f"tx reverted: {tx_hash_hex}")

        return tx_hash_hex, TransferStatus.CONFIRMED

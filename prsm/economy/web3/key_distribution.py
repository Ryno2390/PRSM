"""KeyDistribution Web3 Client.

Python client for the on-chain ``KeyDistribution.sol`` contract
(mainnet-deployed 2026-05-07 alongside StorageSlashing as part of
the Phase 7-storage bundle). Closes the §13 item #2 honest-scope
gap: the contract has been live for over a day, but uploads can't
yet coordinate Tier C key deposit + release-on-payment because the
Python client to drive ``depositKey`` / ``release`` / ``deauthorize``
+ watch ``KeyReleased`` events didn't exist.

Mirrors the patterns established by
``prsm/economy/web3/provenance_registry.py``:
  - Web3.py 7.x synchronous calls
  - Explicit private-key signing
  - Process-wide TX_LOCK_REGISTRY for nonce-race avoidance
  - Three-tier error model: BroadcastFailedError (safe fallback) /
    OnChainPendingError (DO NOT fall back) / OnChainRevertedError
    (safe fallback)
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
# Typed errors
# ──────────────────────────────────────────────────────────────────────


class KeyAlreadyDepositedError(RuntimeError):
    """Mirror of contract's ``KeyAlreadyDeposited(bytes32)`` revert.

    A deposit for the same content_hash already exists. Per contract,
    a second deposit reverts; the publisher must choose a distinct
    content_hash derivative for subsequent versions (no replace-in-
    place semantics).
    """


class KeyNotFoundError(RuntimeError):
    """Mirror of contract's ``KeyNotFound(bytes32)`` revert."""


class PaymentNotVerifiedError(RuntimeError):
    """Mirror of contract's ``PaymentNotVerified(address, bytes32)``
    revert. The royalty distributor did not confirm the recipient
    paid the release fee against this content_hash."""


# ──────────────────────────────────────────────────────────────────────
# Decoded event
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KeyReleasedEvent:
    """Decoded ``KeyReleased(bytes32 indexed contentHash, address
    indexed recipient, bytes encryptedKey)`` event.

    Consumers listen for this event filtered by their address; on
    receipt, decrypt with their X25519 private key (or whatever
    asymmetric scheme the publisher used to wrap the symmetric
    content-decryption key).
    """
    content_hash: bytes
    recipient: str
    encrypted_key: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.content_hash, (bytes, bytearray)) or len(self.content_hash) != 32:
            raise ValueError(
                f"content_hash must be 32 bytes, got "
                f"{len(self.content_hash) if isinstance(self.content_hash, (bytes, bytearray)) else type(self.content_hash).__name__}"
            )
        if not isinstance(self.encrypted_key, (bytes, bytearray)) or len(self.encrypted_key) == 0:
            raise ValueError("encrypted_key must be non-empty bytes")

    @classmethod
    def from_decoded_args(cls, args: Dict[str, Any]) -> "KeyReleasedEvent":
        """Build from a Web3.py decoded-event-args dict.

        ``args`` is the dict produced by
        ``contract.events.KeyReleased().process_log(log)`` where
        keys are the Solidity argument names in camelCase.
        """
        return cls(
            content_hash=bytes(args["contentHash"]),
            recipient=str(args["recipient"]),
            encrypted_key=bytes(args["encryptedKey"]),
        )


# ──────────────────────────────────────────────────────────────────────
# Contract ABI (minimal — only the surface we exercise)
# ──────────────────────────────────────────────────────────────────────


KEY_DISTRIBUTION_ABI = [
    {
        "type": "function",
        "name": "depositKey",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "contentHash", "type": "bytes32"},
            {"name": "encryptedKey", "type": "bytes"},
            {"name": "royalty", "type": "address"},
            {"name": "releaseFeeFtnsWei", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "release",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "contentHash", "type": "bytes32"},
            {"name": "recipient", "type": "address"},
        ],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "deauthorize",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "contentHash", "type": "bytes32"}],
        "outputs": [],
    },
    {
        "type": "event",
        "name": "KeyDeposited",
        "anonymous": False,
        "inputs": [
            {"name": "contentHash", "type": "bytes32", "indexed": True},
            {"name": "publisher", "type": "address", "indexed": True},
            {"name": "royalty", "type": "address", "indexed": True},
            {"name": "releaseFeeFtnsWei", "type": "uint256", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "KeyReleased",
        "anonymous": False,
        "inputs": [
            {"name": "contentHash", "type": "bytes32", "indexed": True},
            {"name": "recipient", "type": "address", "indexed": True},
            {"name": "encryptedKey", "type": "bytes", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "KeyDeauthorized",
        "anonymous": False,
        "inputs": [
            {"name": "contentHash", "type": "bytes32", "indexed": True},
            {"name": "publisher", "type": "address", "indexed": True},
        ],
    },
]


# ──────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────


class KeyDistributionClient:
    """Sync Web3 client for KeyDistribution.

    Construction:
        rpc_url: Base mainnet RPC endpoint.
        contract_address: KeyDistribution.sol deploy address.
        private_key: Optional. Required for write calls
            (depositKey / release / deauthorize); read/event paths
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
            abi=KEY_DISTRIBUTION_ABI,
        )

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

        # Mirror provenance_registry.py: process-wide TX_LOCK_REGISTRY
        # serializes nonce acquisition across web3 clients sharing
        # the same private key.
        from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY
        if self._account is not None:
            self._tx_lock = TX_LOCK_REGISTRY.get_lock(self._account.address)
        else:
            self._tx_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Writes ─────────────────────────────────────────────────

    def deposit_key(
        self,
        content_hash: bytes,
        encrypted_key: bytes,
        royalty_address: str,
        release_fee_ftns_wei: int,
    ) -> Tuple[str, TransferStatus]:
        """Publisher deposits the encrypted content-decryption key.

        Mirrors contract reverts with client-side fail-fast:
          - content_hash must be 32 bytes
          - encrypted_key must be non-empty
          - release_fee must be > 0 (ZeroFee revert)
          - royalty must not be zero address (InvalidAddress revert)

        Returns (tx_hash_hex, TransferStatus). May raise
        BroadcastFailedError (safe fallback), OnChainPendingError
        (do NOT fall back), or OnChainRevertedError (safe fallback).
        """
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        self._validate_content_hash(content_hash)
        if not encrypted_key:
            raise ValueError("encrypted_key must be non-empty bytes")
        if release_fee_ftns_wei <= 0:
            raise ValueError(
                f"release_fee_ftns_wei must be > 0 (got "
                f"{release_fee_ftns_wei})"
            )
        if royalty_address.lower() == ZERO_ADDRESS:
            raise ValueError("royalty_address must not be zero address")

        with self._tx_lock:
            tx = self.contract.functions.depositKey(
                content_hash,
                bytes(encrypted_key),
                Web3.to_checksum_address(royalty_address),
                int(release_fee_ftns_wei),
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    def release(
        self,
        content_hash: bytes,
        recipient: str,
    ) -> Tuple[str, TransferStatus]:
        """Trigger key release to ``recipient``. Gated on the
        royalty distributor confirming the recipient paid the
        release fee. The actual encrypted key bytes surface in the
        ``KeyReleased`` event log; the recipient (or any peer
        watching) reads + decrypts.
        """
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        self._validate_content_hash(content_hash)
        if recipient.lower() == ZERO_ADDRESS:
            raise ValueError("recipient must not be zero address")

        with self._tx_lock:
            tx = self.contract.functions.release(
                content_hash,
                Web3.to_checksum_address(recipient),
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    def deauthorize(
        self,
        content_hash: bytes,
    ) -> Tuple[str, TransferStatus]:
        """Publisher-only: deactivate future releases for this content.

        Past releases CANNOT be revoked — consumers who already
        received the key via ``KeyReleased`` events keep it (per
        contract design plan §8.5).
        """
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        self._validate_content_hash(content_hash)

        with self._tx_lock:
            tx = self.contract.functions.deauthorize(
                content_hash,
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _validate_content_hash(content_hash: bytes) -> None:
        if not isinstance(content_hash, (bytes, bytearray)):
            raise ValueError(
                f"content_hash must be bytes, got {type(content_hash).__name__}"
            )
        if len(content_hash) != 32:
            raise ValueError(
                f"content_hash must be 32 bytes, got {len(content_hash)}"
            )

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

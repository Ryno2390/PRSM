"""ProvenanceRegistryV2 Web3 Client — PRSM-PROV-1 Item 7 T7.4 + T7.6.

Adds embedding-commitment registration + dispute-arbitration helpers on
top of the v1 ProvenanceRegistry surface.

The v2 contract is a SEPARATE deployment from v1 (v1 is non-upgradeable
and lives at Base mainnet 0xdF47...9915). New uploads register here;
old uploads stay in v1. Off-chain code that needs to read provenance
should consult v2 first, then fall back to v1 when v2 returns
``isRegistered = false``.

Mainnet deployment of THIS contract is gated behind L4 audit firm
review per the PRSM-PROV-1 plan §4.5 + PRSM-POL-1 §5. Sepolia
deployments are unrestricted.
"""
from __future__ import annotations

import logging
import struct
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from prsm.economy.web3.provenance_registry import (
    HAS_WEB3,
    ZERO_ADDRESS,
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)

if HAS_WEB3:
    from web3 import Web3
    from eth_account import Account
else:  # pragma: no cover
    Web3 = None  # type: ignore
    Account = None  # type: ignore

logger = logging.getLogger(__name__)


# ---- Canonical commitment formula --------------------------------------


def compute_embedding_commitment(
    model_id: str, dim: int, vector_bytes: bytes,
) -> bytes:
    """Canonical on-chain commitment for a content embedding.

    Format: ``keccak256(model_id_utf8 || uint32_be(dim) || vector_bytes)``

    Binding ``model_id`` and ``dim`` into the hash defeats two attacks:
      1. Cross-model substitution (claim a 384-dim sentence-transformers
         vector matches an ada-002 1536-dim record).
      2. Truncation/extension (claim a vector of different length is the
         "same" content).

    Returns the 32-byte keccak256 digest. Off-chain callers pass it as
    ``embeddingCommitment`` to ``ProvenanceRegistryV2.registerContent``;
    dispute callers pass it as ``claimed`` to
    ``verifyEmbeddingCommitment``.
    """
    if not HAS_WEB3:
        raise RuntimeError("web3 package required")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(dim, int) or not (0 < dim < 2**31):
        raise ValueError("dim must be a positive int < 2**31")
    if not isinstance(vector_bytes, (bytes, bytearray)):
        raise ValueError("vector_bytes must be bytes-like")
    if len(vector_bytes) == 0:
        raise ValueError("vector_bytes must be non-empty")

    payload = model_id.encode("utf-8") + struct.pack(">I", dim) + bytes(vector_bytes)
    return bytes(Web3.keccak(payload))


def compute_kind_tag(kind_label: str) -> bytes:
    """``keccak256(kind_label_utf8)`` — what we store in
    ``fingerprintKind`` so on-chain comparisons stay constant-cost.

    Use the YAML keys exactly: ``"text-vector"``, ``"image-phash"``,
    ``"audio-chromaprint"``, ``"video-multihash"``, ``"structural"``,
    ``"byte-hash"``.
    """
    if not HAS_WEB3:
        raise RuntimeError("web3 package required")
    if not isinstance(kind_label, str) or not kind_label:
        raise ValueError("kind_label must be a non-empty string")
    return bytes(Web3.keccak(kind_label.encode("utf-8")))


ZERO_BYTES32 = b"\x00" * 32


# ---- ABI ----------------------------------------------------------------

PROVENANCE_REGISTRY_V2_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "uint16", "name": "royaltyRateBps", "type": "uint16"},
            {"internalType": "string", "name": "metadataUri", "type": "string"},
            {"internalType": "bytes32", "name": "embeddingCommitment", "type": "bytes32"},
            {"internalType": "bytes32", "name": "fingerprintKind", "type": "bytes32"},
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
            {"internalType": "bytes32", "name": "embeddingCommitment", "type": "bytes32"},
            {"internalType": "bytes32", "name": "fingerprintKind", "type": "bytes32"},
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
        "inputs": [{"internalType": "bytes32", "name": "contentHash", "type": "bytes32"}],
        "name": "getCreatorAndRate",
        "outputs": [
            {"internalType": "address", "name": "creator", "type": "address"},
            {"internalType": "uint16", "name": "royaltyRateBps", "type": "uint16"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "contentHash", "type": "bytes32"}],
        "name": "getEmbeddingCommitment",
        "outputs": [
            {"internalType": "bytes32", "name": "embeddingCommitment", "type": "bytes32"},
            {"internalType": "bytes32", "name": "fingerprintKind", "type": "bytes32"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "claimed", "type": "bytes32"},
        ],
        "name": "verifyEmbeddingCommitment",
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
            {"indexed": False, "internalType": "bytes32", "name": "embeddingCommitment", "type": "bytes32"},
            {"indexed": False, "internalType": "bytes32", "name": "fingerprintKind", "type": "bytes32"},
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
class ContentRecordV2:
    creator: str
    royalty_rate_bps: int
    registered_at: int
    embedding_commitment: bytes
    fingerprint_kind: bytes
    metadata_uri: str


# ---- Client -------------------------------------------------------------


MAX_ROYALTY_RATE_BPS = 9800


class ProvenanceRegistryV2Client:
    """Sync Web3 client for the v2 contract.

    Mirrors the v1 client's architecture: per-account TX lock for nonce
    safety, Web3.py 7.x raw_transaction signing, and the same
    BroadcastFailedError / OnChainPendingError / OnChainRevertedError
    error contract.
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
            abi=PROVENANCE_REGISTRY_V2_ABI,
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

    # ---- Writes ----

    def register_content_v2(
        self,
        content_hash: bytes,
        royalty_rate_bps: int,
        metadata_uri: str,
        embedding_commitment: bytes = ZERO_BYTES32,
        fingerprint_kind: bytes = ZERO_BYTES32,
    ) -> Tuple[str, TransferStatus]:
        """Register `content_hash` on the v2 registry.

        Pass ``ZERO_BYTES32`` for both new args to register byte-hash-only
        content (the dispute path falls back to raw-hash matching).
        """
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        if len(embedding_commitment) != 32:
            raise ValueError("embedding_commitment must be 32 bytes")
        if len(fingerprint_kind) != 32:
            raise ValueError("fingerprint_kind must be 32 bytes")
        if not (0 <= royalty_rate_bps <= MAX_ROYALTY_RATE_BPS):
            raise ValueError(
                f"royalty_rate_bps must be in [0, {MAX_ROYALTY_RATE_BPS}] "
                f"(got {royalty_rate_bps})"
            )

        with self._tx_lock:
            tx = self.contract.functions.registerContent(
                content_hash,
                royalty_rate_bps,
                metadata_uri,
                embedding_commitment,
                fingerprint_kind,
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    def transfer_ownership(
        self, content_hash: bytes, new_creator: str,
    ) -> Tuple[str, TransferStatus]:
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")

        with self._tx_lock:
            tx = self.contract.functions.transferContentOwnership(
                content_hash, Web3.to_checksum_address(new_creator),
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    # ---- Reads ----

    def get_content(self, content_hash: bytes) -> Optional[ContentRecordV2]:
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        (
            creator, rate_bps, registered_at,
            embedding_commitment, fingerprint_kind, metadata_uri,
        ) = self.contract.functions.contents(content_hash).call()
        if creator == ZERO_ADDRESS:
            return None
        return ContentRecordV2(
            creator=creator,
            royalty_rate_bps=int(rate_bps),
            registered_at=int(registered_at),
            embedding_commitment=bytes(embedding_commitment),
            fingerprint_kind=bytes(fingerprint_kind),
            metadata_uri=metadata_uri,
        )

    def is_registered(self, content_hash: bytes) -> bool:
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        return bool(self.contract.functions.isRegistered(content_hash).call())

    def get_embedding_commitment(
        self, content_hash: bytes,
    ) -> Tuple[bytes, bytes]:
        """Returns ``(embedding_commitment, fingerprint_kind)`` for a
        registered piece of content. Both zero when no embedding was
        committed at registration time.
        """
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        commitment, kind = (
            self.contract.functions.getEmbeddingCommitment(content_hash).call()
        )
        return bytes(commitment), bytes(kind)

    def verify_embedding_commitment(
        self, content_hash: bytes, claimed: bytes,
    ) -> bool:
        """Pure on-chain check: does ``claimed`` match the registered
        commitment for ``content_hash``? Zero commitment never matches
        — even a zero claim against a zero record returns False.
        """
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        if len(claimed) != 32:
            raise ValueError("claimed must be 32 bytes")
        return bool(
            self.contract.functions.verifyEmbeddingCommitment(
                content_hash, claimed,
            ).call()
        )

    # ---- Dispute helper (T7.6) ----

    def dispute_provenance(
        self,
        content_hash: bytes,
        model_id: str,
        dim: int,
        vector_bytes: bytes,
    ) -> bool:
        """Verify caller's vector matches the on-chain commitment.

        Computes the canonical commitment from
        ``(model_id, dim, vector_bytes)`` and asks the contract whether
        it matches the registered ``embeddingCommitment``. A True
        return is on-chain proof that this exact vector is what the
        creator committed to at registration time; False means either
        (a) the vector is wrong, (b) no embedding was committed
        (legacy / byte-hash-only), or (c) the content isn't registered
        at all.
        """
        claimed = compute_embedding_commitment(model_id, dim, vector_bytes)
        return self.verify_embedding_commitment(content_hash, claimed)

    # ---- Internals (mirrors v1 client) ----

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
                f"broadcast OK but receipt unknown: {exc}", tx_hash=tx_hash_hex,
            ) from exc

        if receipt.status != 1:
            raise OnChainRevertedError(f"tx reverted: {tx_hash_hex}")

        return tx_hash_hex, TransferStatus.CONFIRMED

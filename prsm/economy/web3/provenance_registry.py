"""ProvenanceRegistry Web3 Client.

Thin wrapper around the on-chain ProvenanceRegistry contract. Mirrors the
patterns established by prsm/economy/ftns_onchain.py — Web3.py 7.x,
synchronous calls (the rest of the on-chain stack is sync), and explicit
private-key signing.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

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

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Writes ─────────────────────────────────────────────────

    def register_content(
        self,
        content_hash: bytes,
        royalty_rate_bps: int,
        metadata_uri: str,
    ) -> str:
        """Register `content_hash` with caller as creator. Returns tx hash hex."""
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        if not (0 <= royalty_rate_bps <= 10000):
            raise ValueError("royalty_rate_bps must be in [0, 10000]")

        tx = self.contract.functions.registerContent(
            content_hash, royalty_rate_bps, metadata_uri
        ).build_transaction(self._tx_overrides())
        return self._sign_and_send(tx)

    def transfer_ownership(self, content_hash: bytes, new_creator: str) -> str:
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")

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
            "nonce": self.web3.eth.get_transaction_count(self._account.address),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_and_send(self, tx: dict) -> str:
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        # web3.py 7.x exposes the signed payload as `raw_transaction`.
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )
        tx_hash = self.web3.eth.send_raw_transaction(raw)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise RuntimeError(f"Transaction failed: 0x{tx_hash.hex()}")
        return "0x" + tx_hash.hex()

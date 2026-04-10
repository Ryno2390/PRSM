"""RoyaltyDistributor Web3 Client.

Wraps the RoyaltyDistributor contract and the FTNS ERC-20 approval flow:
when a payer wants to distribute `gross` FTNS for content X, this client
first checks the existing allowance and only sends an `approve` tx when
needed, then calls `distributeRoyalty`.
"""
from __future__ import annotations

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


ROYALTY_DISTRIBUTOR_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "address", "name": "servingNode", "type": "address"},
            {"internalType": "uint256", "name": "gross", "type": "uint256"},
        ],
        "name": "distributeRoyalty",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "gross", "type": "uint256"},
        ],
        "name": "preview",
        "outputs": [
            {"internalType": "uint256", "name": "creatorAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "networkAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "servingNodeAmount", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "payer", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "creator", "type": "address"},
            {"indexed": False, "internalType": "address", "name": "servingNode", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "creatorAmount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "networkAmount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "servingNodeAmount", "type": "uint256"},
        ],
        "name": "RoyaltyPaid",
        "type": "event",
    },
]

# Minimal ERC-20 ABI subset we need (allowance + approve)
_ERC20_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "address", "name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


@dataclass
class SplitPreview:
    creator_amount: int
    network_amount: int
    serving_node_amount: int


class RoyaltyDistributorClient:
    """Sync Web3 client for RoyaltyDistributor + FTNS approval flow."""

    def __init__(
        self,
        rpc_url: str,
        distributor_address: str,
        ftns_token_address: str,
        private_key: Optional[str] = None,
    ) -> None:
        if not HAS_WEB3:
            raise RuntimeError("web3 package required")

        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.distributor_address = Web3.to_checksum_address(distributor_address)
        self.ftns_address = Web3.to_checksum_address(ftns_token_address)

        self.distributor = self.web3.eth.contract(
            address=self.distributor_address, abi=ROYALTY_DISTRIBUTOR_ABI
        )
        self.token = self.web3.eth.contract(
            address=self.ftns_address, abi=_ERC20_ABI
        )

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Reads ──────────────────────────────────────────────────

    def preview_split(self, content_hash: bytes, gross: int) -> SplitPreview:
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        creator_amt, network_amt, node_amt = (
            self.distributor.functions.preview(content_hash, gross).call()
        )
        return SplitPreview(
            creator_amount=int(creator_amt),
            network_amount=int(network_amt),
            serving_node_amount=int(node_amt),
        )

    def allowance(self) -> int:
        if not self._account:
            raise RuntimeError("private_key required")
        return int(
            self.token.functions.allowance(
                self._account.address, self.distributor_address
            ).call()
        )

    # ── Writes ─────────────────────────────────────────────────

    def distribute_royalty(
        self, content_hash: bytes, serving_node: str, gross: int
    ) -> str:
        if not self._account:
            raise RuntimeError("private_key required")
        if len(content_hash) != 32:
            raise ValueError("content_hash must be 32 bytes")
        if gross <= 0:
            raise ValueError("gross must be positive")

        # Approve only if needed
        current_allowance = int(
            self.token.functions.allowance(
                self._account.address, self.distributor_address
            ).call()
        )
        if current_allowance < gross:
            approve_tx = self.token.functions.approve(
                self.distributor_address, gross
            ).build_transaction(self._tx_overrides())
            self._sign_and_send(approve_tx)

        tx = self.distributor.functions.distributeRoyalty(
            content_hash, Web3.to_checksum_address(serving_node), gross
        ).build_transaction(self._tx_overrides())
        return self._sign_and_send(tx)

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
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )
        tx_hash = self.web3.eth.send_raw_transaction(raw)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise RuntimeError(f"Transaction failed: 0x{tx_hash.hex()}")
        return "0x" + tx_hash.hex()

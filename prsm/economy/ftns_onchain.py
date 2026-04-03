"""
On-Chain FTNS Ledger Adapter
============================

Mirrors the LocalLedger to the real FTNS contract on Base mainnet.
When the node executes compute jobs and escrow transactions, this
adapter broadcasts real FTNS transfers on-chain.

Usage in PRSMNode:
    self.ftns_ledger = OnChainFTNSLedger(
        node_id=self.identity.node_id,
        wallet_private_key=os.getenv("FTNS_WALLET_PRIVATE_KEY"),
        contract_address="0x5276...",
        rpc_url=Base_RPC,
    )
    await self.ftns_ledger.initialize()

This creates a single Web3 connection that:
- Reads real on-chain FTNS balances
- Transfers FTNS on escrow lock/release/refund
- Tracks tx hashes for auditability
"""

import asyncio
import logging
import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from web3 import Web3
    from web3.exceptions import TimeExhausted
    from eth_account import Account
    from eth_utils import to_checksum_address
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    Web3 = None
    Account = None
    to_checksum_address = None
    TimeExhausted = Exception

logger = logging.getLogger(__name__)

# ── Minimal ERC20 ABI ──────────────────────────────────────────────
_ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
]

# ── Base mainnet config ───────────────────────────────────────────────────
# Your FTNS contract address
FTNS_CONTRACT_ADDRESS = os.getenv(
    "FTNS_CONTRACT_ADDRESS",
    "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",  # Your deployed FTNS on Base
)
BASE_RPC_URL = os.getenv(
    "BASE_RPC_URL",
    "https://mainnet.base.org",
)
BASE_CHAIN_ID = 8453


@dataclass
class FTNSTransaction:
    """Record of an on-chain FTNS transfer."""
    job_id: str
    from_addr: str
    to_addr: str
    amount_ftns: float
    tx_hash: str = ""
    status: str = "pending"  # pending | confirmed | rejected
    created_at: float = field(default_factory=lambda: __import__("time").time())
    block_number: Optional[int] = None


class OnChainFTNSLedger:
    """Manages FTNS token transfers on Base mainnet.

    Wraps a Web3 connection to the FTNS contract and provides:
    - get_balance(address) → real on-chain FTNS balance
    - create_escrow(job_id, amount) → approve + lock tokens
    - release_escrow(job_id, to_addr) → transfer tokens
    - record all tx hashes for audit

    The LocalLedger keeps its own internal state for speed;
    OnChainFTNSLedger mirrors the economic events to blockchain.
    """

    def __init__(
        self,
        node_id: str,
        wallet_private_key: Optional[str] = None,
        contract_address: str = FTNS_CONTRACT_ADDRESS,
        rpc_url: str = BASE_RPC_URL,
        chain_id: int = BASE_CHAIN_ID,
    ):
        self.node_id = node_id
        self.wallet_private_key = wallet_private_key or os.getenv(
            "FTNS_WALLET_PRIVATE_KEY"
        )
        self.contract_address = contract_address
        self.rpc_url = rpc_url
        self.chain_id = chain_id

        self.w3: Optional[Web3] = None
        self._account = None
        self._connected_address: Optional[str] = None
        self._token = None
        self._decimals = 18
        self._transactions: List[FTNSTransaction] = []
        self._is_initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Connect to Base mainnet and load the FTNS contract."""
        if self._is_initialized:
            return True

        if not HAS_WEB3:
            logger.warning("web3 not installed — FTNS on-chain mode disabled")
            return False

        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 15}))
            if not self.w3.is_connected():
                logger.error(f"Cannot connect to Base RPC: {self.rpc_url}")
                return False

            # Connect wallet
            if self.wallet_private_key:
                key = self.wallet_private_key if self.wallet_private_key.startswith("0x") else "0x" + self.wallet_private_key
                self._account = Account.from_key(key)
                self._connected_address = self._account.address
                logger.info(f"FTNS wallet connected: {self._connected_address}")
            else:
                # Public read-only mode — can check balances but not send
                logger.info("FTNS ledger in read-only mode (no private key)")

            # Load the FTNS contract
            self._token = self.w3.eth.contract(
                address=to_checksum_address(self.contract_address),
                abi=_ERC20_ABI,
            )

            # Get decimals
            if self.w3.is_connected():
                try:
                    self._decimals = self._token.functions.decimals().call()
                    name = self._token.functions.name().call()
                    symbol = self._token.functions.symbol().call()
                    logger.info(f"FTNS token loaded: {name} ({symbol})")
                except Exception as e:
                    logger.warning(f"Could not read token metadata: {e}")

            self._is_initialized = True
            return True

        except Exception as e:
            logger.error(f"FTNS ledger init failed: {e}")
            return False

    async def get_balance(self, address: Optional[str] = None) -> float:
        """Get FTNS balance for an address from the blockchain."""
        if not self._is_initialized:
            ok = await self.initialize()
            if not ok:
                return 0.0

        addr = address or self._connected_address
        if not addr or not self._token:
            return 0.0

        try:
            bal_wei = self._token.functions.balanceOf(to_checksum_address(addr)).call()
            return bal_wei / (10 ** self._decimals)
        except Exception as e:
            logger.warning(f"Failed to read FTNS balance for {addr[:12]}…: {e}")
            return 0.0

    async def transfer(
        self,
        job_id: str,
        to_address: str,
        amount_ftns: float,
    ) -> Optional[FTNSTransaction]:
        """Send FTNS tokens on-chain.

        Only works if a wallet private key is configured.
        """
        if not self._is_initialized:
            await self.initialize()
        if not self._account or not self._token:
            logger.error("Cannot transfer FTNS — no wallet configured")
            return None
        if amount_ftns <= 0:
            return None

        tx_record = FTNSTransaction(
            job_id=job_id,
            from_addr=self._connected_address,
            to_addr=to_address,
            amount_ftns=amount_ftns,
        )

        async with self._lock:
            try:
                amount_wei = int(amount_ftns * (10 ** self._decimals))

                # All web3 calls are synchronous and block the event loop.
                # Run them in a thread to keep the API responsive.
                import asyncio
                loop = asyncio.get_running_loop()

                nonce = await loop.run_in_executor(
                    None, self.w3.eth.get_transaction_count, self._connected_address
                )

                # Build tx
                tx = {
                    "chainId": self.chain_id,
                    "nonce": nonce,
                    "gasPrice": self.w3.to_wei("5", "gwei"),  # 5 gwei for Base
                    "gas": 100000,
                    "to": self.contract_address,
                    "value": 0,
                    "data": self._token.functions.transfer(
                        to_checksum_address(to_address), amount_wei
                    )._encode_transaction_data(),
                }

                signed = self.w3.eth.account.sign_transaction(tx, self._account.key)
                tx_hash = await loop.run_in_executor(
                    None, self.w3.eth.send_raw_transaction, signed.raw_transaction
                )

                tx_record.tx_hash = tx_hash.hex()
                tx_record.status = "pending"
                self._transactions.append(tx_record)

                logger.info(
                    f"FTNS transfer sent: {amount_ftns:.6f} -> {to_address[:12]}… "
                    f"(tx: {tx_record.tx_hash[:16]}…)"
                )

                # Wait for confirmation in a thread (can take 2-30s on Base)
                receipt = await loop.run_in_executor(
                    None,
                    lambda: self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60),
                )
                if receipt["status"] == 1:
                    tx_record.status = "confirmed"
                    tx_record.block_number = receipt["blockNumber"]
                    logger.info(
                        f"FTNS transfer confirmed in block {tx_record.block_number}"
                    )
                else:
                    tx_record.status = "rejected"
                    logger.warning(f"FTNS transfer reverted: {tx_hash.hex()}")

                return tx_record

            except Exception as e:
                tx_record.status = "rejected"
                logger.error(f"FTNS transfer failed: {e}")
                self._transactions.append(tx_record)
                return None

    def token_info_sync(self) -> Dict[str, Any]:
        """Get on-chain FTNS token info (name, symbol, total supply)."""
        if not self._token:
            return {}

        try:
            name = self._token.functions.name().call()
            symbol = self._token.functions.symbol().call()
            decimals = self._token.functions.decimals().call()
            total_supply = self._token.functions.totalSupply().call()
            return {
                "name": name,
                "symbol": symbol,
                "decimals": decimals,
                "total_supply": total_supply / (10 ** decimals),
                "contract": self.contract_address,
            }
        except Exception:
            return {}

    def get_summary(self) -> Dict[str, Any]:
        """Summary of all on-chain FTNS transactions."""
        statuses = {}
        for tx in self._transactions:
            statuses[tx.status] = statuses.get(tx.status, 0) + 1
        total_sent = sum(tx.amount_ftns for tx in self._transactions if tx.status == "confirmed")
        return {
            "total_transactions": len(self._transactions),
            "by_status": statuses,
            "total_confirmed_ftns": round(total_sent, 6),
            "wallet": self._connected_address,
        }

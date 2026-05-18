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
from typing import Any, Dict, List, Optional
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

# ── Dynamic Gas Pricing (Ring 6) ──────────────────────────────────
DEFAULT_GAS_GWEI = 5
MAX_GAS_GWEI = 50


class RPCFailover:
    """Rotates through RPC endpoints on failure."""

    def __init__(self, urls: list):
        self._urls = urls if urls else ["https://mainnet.base.org"]
        self._index = 0

    @property
    def current_url(self) -> str:
        return self._urls[self._index % len(self._urls)]

    def mark_failed(self) -> None:
        self._index = (self._index + 1) % len(self._urls)

    def mark_success(self) -> None:
        pass


def estimate_gas_price(w3, multiplier: float = 1.2, max_gwei: int = MAX_GAS_GWEI) -> int:
    """Get dynamic gas price from network, with fallback and cap. Returns wei."""
    try:
        network_gas = w3.eth.gas_price
        adjusted = int(network_gas * multiplier)
        cap = max_gwei * 1_000_000_000
        return min(adjusted, cap)
    except Exception:
        return DEFAULT_GAS_GWEI * 1_000_000_000


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

# ── Network config (resolved at module load via PRSM_NETWORK) ─────────────
# T6 (post-2026-05-07): historically these were hardcoded mainnet
# defaults, which meant `prsm join-testnet`'s env-var bundle silently
# bypassed the constructor defaults (testnet env sets
# BASE_SEPOLIA_RPC_URL + FTNS_TOKEN_ADDRESS, not BASE_RPC_URL +
# FTNS_CONTRACT_ADDRESS). Now resolved through `prsm.config.networks`
# which honors PRSM_NETWORK + per-field overrides.
from prsm.config.networks import resolve_endpoints

_resolved = resolve_endpoints()
FTNS_CONTRACT_ADDRESS = _resolved.ftns_token or "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
BASE_RPC_URL = _resolved.rpc_url
BASE_CHAIN_ID = _resolved.chain_id


def _gas_status_for_eth(eth: float) -> str:
    """Pure helper — shared between startup log, endpoint,
    /health/detailed, and the periodic monitor. One source
    of truth for threshold boundaries."""
    if eth < 0.0001:
        return "critical"
    if eth < 0.0005:
        return "low"
    return "ok"


class GasStatusMonitor:
    """Periodic background sampler that logs ONLY on
    status transitions (ok ↔ low ↔ critical) so operators
    get continuous signal without log spam.
    """

    SEVERITY = {"ok": 0, "low": 1, "critical": 2}

    def __init__(self, ledger: "OnChainFTNSLedger",
                 interval_seconds: float = 60.0,
                 webhook_deliverer: Optional[Any] = None,
                 webhook_url: Optional[str] = None,
                 webhook_secret: Optional[str] = None):
        self.ledger = ledger
        self.interval_seconds = interval_seconds
        self._last_status: Optional[str] = None
        self._webhook_deliverer = webhook_deliverer
        self._webhook_url = webhook_url
        self._webhook_secret = webhook_secret

    async def _fire_webhook(
        self, prev: str, new: str, eth: float,
    ) -> None:
        if (
            self._webhook_deliverer is None
            or not self._webhook_url
        ):
            return
        import time as _time
        payload = {
            "event": "gas.transition",
            "node_id": getattr(
                self.ledger, "node_id", "unknown",
            ),
            "address": self.ledger._connected_address,
            "previous_status": prev,
            "new_status": new,
            "eth_balance": eth,
            "timestamp": _time.time(),
        }
        try:
            await self._webhook_deliverer.deliver(
                url=self._webhook_url,
                event="gas.transition",
                payload=payload,
                secret=self._webhook_secret,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "gas-monitor webhook delivery raised "
                "(non-fatal): %s", exc,
            )

    async def _tick_async(self) -> None:
        """Async tick variant that supports webhook firing.

        Used by run_forever() (which already runs in an event
        loop). The sync _tick_sync() remains for unit tests
        that need synchronous semantics.
        """
        prev_status = self._last_status
        self._tick_sync()
        new_status = self._last_status
        if (
            prev_status is not None
            and new_status is not None
            and prev_status != new_status
        ):
            w3 = getattr(self.ledger, "w3", None)
            addr = getattr(
                self.ledger, "_connected_address", None,
            )
            if w3 is not None and addr is not None:
                try:
                    wei = w3.eth.get_balance(addr)
                    await self._fire_webhook(
                        prev_status, new_status, wei / 1e18,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "gas-monitor webhook prep failed "
                        "(non-fatal): %s", exc,
                    )

    def _tick_sync(self) -> None:
        """Single tick. Synchronous helper for unit tests
        + reused inside the async run_forever loop."""
        w3 = getattr(self.ledger, "w3", None)
        addr = getattr(self.ledger, "_connected_address", None)
        if w3 is None or addr is None:
            return
        try:
            wei = w3.eth.get_balance(addr)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "gas-monitor tick failed (non-fatal): %s",
                exc,
            )
            return
        eth = wei / 1e18
        new_status = _gas_status_for_eth(eth)
        if self._last_status is None:
            # First tick is the baseline; don't log it
            # since sprint-504's startup log already did.
            self._last_status = new_status
            return
        if new_status == self._last_status:
            return
        prev = self._last_status
        addr_short = addr[:10] + "…"
        if new_status == "critical":
            logger.error(
                "Operator gas transition %s → critical: "
                "%.10f ETH on %s. Top up NOW — broadcasts "
                "will start failing.",
                prev, eth, addr_short,
            )
        elif new_status == "low" and (
            self.SEVERITY[new_status]
            > self.SEVERITY[prev]
        ):
            logger.warning(
                "Operator gas transition %s → low: %.10f "
                "ETH on %s. Plan to top up soon.",
                prev, eth, addr_short,
            )
        elif (
            self.SEVERITY[new_status]
            < self.SEVERITY[prev]
        ):
            logger.info(
                "Operator gas recovered %s → %s: %.10f "
                "ETH on %s.",
                prev, new_status, eth, addr_short,
            )
        else:
            logger.info(
                "Operator gas transition %s → %s: %.10f "
                "ETH on %s.",
                prev, new_status, eth, addr_short,
            )
        self._last_status = new_status

    async def run_forever(self) -> None:
        """Async loop suitable for asyncio.create_task()."""
        import asyncio as _aio
        while True:
            try:
                await self._tick_async()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "gas-monitor outer exception "
                    "(non-fatal): %s", exc,
                )
            try:
                await _aio.sleep(self.interval_seconds)
            except _aio.CancelledError:
                raise


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
        db_path: Optional[str] = None,
    ):
        self.node_id = node_id
        self.wallet_private_key = wallet_private_key or os.getenv(
            "FTNS_WALLET_PRIVATE_KEY"
        )
        self.contract_address = contract_address
        self.rpc_url = rpc_url
        self.chain_id = chain_id
        self.db_path = db_path

        self.w3: Optional[Web3] = None
        self._account = None
        self._connected_address: Optional[str] = None
        self._token = None
        self._decimals = 18
        self._transactions: List[FTNSTransaction] = []
        self._is_initialized = False
        self._lock = asyncio.Lock()
        self._db = None  # aiosqlite.Connection if persistent

        # Phase 1.3 Task 3a: populate _connected_address synchronously
        # from the wallet private key. Account.from_key(key).address is
        # a pure-local elliptic curve derivation — no network call — so
        # it's safe to do in __init__. This makes the address available
        # at PRSMNode.initialize() time (before the async initialize()
        # runs in start()), which is when ContentUploader is constructed
        # and _derive_creator_address reads this field. Without the sync
        # population, ContentUploader.creator_address would always be
        # None in production and every upload would silently skip
        # provenance_hash computation and on-chain royalty routing.
        #
        # The async initialize() still runs later for RPC/balance/
        # contract-state setup; it idempotently re-derives the same
        # _account and _connected_address (no-op if already set).
        if self.wallet_private_key and Account is not None:
            try:
                key = (
                    self.wallet_private_key
                    if self.wallet_private_key.startswith("0x")
                    else "0x" + self.wallet_private_key
                )
                self._account = Account.from_key(key)
                self._connected_address = self._account.address
            except Exception as exc:
                # Bad key format or missing eth_account — log and leave
                # _connected_address=None so the upload path falls back
                # to PRSM_CREATOR_ADDRESS env var or local royalties.
                logger.warning(
                    f"Could not derive connected_address from "
                    f"FTNS_WALLET_PRIVATE_KEY: {exc}. On-chain routing "
                    f"will require PRSM_CREATOR_ADDRESS env var or fall "
                    f"back to local royalties."
                )

    @property
    def is_persistent(self) -> bool:
        return self.db_path is not None

    async def _init_persistence(self) -> None:
        if not self.is_persistent:
            return
        import aiosqlite
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS onchain_transactions (
                job_id        TEXT PRIMARY KEY,
                tx_hash       TEXT,
                from_addr     TEXT,
                to_addr       TEXT,
                amount_ftns   REAL NOT NULL,
                status        TEXT NOT NULL,
                block_number  INTEGER,
                created_at    REAL NOT NULL
            )
        """)
        # Sprint 510 F39 fix: ensure chain_id column exists.
        # ALTER TABLE adds it with NULL default for pre-sprint-510
        # rows. Wrapped in try/except since the column may already
        # exist on databases initialized after this sprint.
        try:
            await self._db.execute(
                "ALTER TABLE onchain_transactions "
                "ADD COLUMN chain_id INTEGER"
            )
        except Exception:
            pass  # column already exists
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_octx_created "
            "ON onchain_transactions(created_at)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_octx_tx_hash "
            "ON onchain_transactions(tx_hash)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_octx_chain_id "
            "ON onchain_transactions(chain_id)"
        )
        await self._db.commit()
        # Replay rows — sprint 510 F39 fix: filter by this
        # ledger's chain_id so a daemon on Sepolia doesn't
        # see mainnet TX (and vice versa). NULL-chain_id
        # rows (legacy pre-sprint-510) are excluded — they
        # have ambiguous provenance and shouldn't be
        # trusted as belonging to any specific network.
        cur = await self._db.execute(
            "SELECT job_id, tx_hash, from_addr, to_addr, "
            "amount_ftns, status, block_number, created_at "
            "FROM onchain_transactions "
            "WHERE chain_id = ? "
            "ORDER BY created_at ASC",
            (self.chain_id,),
        )
        rows = await cur.fetchall()
        for r in rows:
            self._transactions.append(FTNSTransaction(
                job_id=r[0],
                from_addr=r[2] or "",
                to_addr=r[3] or "",
                amount_ftns=r[4],
                tx_hash=r[1] or "",
                status=r[5],
                block_number=r[6],
                created_at=r[7],
            ))

    def _emit_startup_gas_log(self) -> None:
        """Push-signal counterpart to /wallet/gas-status.

        Called at end of initialize(). Logs WARNING for low gas
        and ERROR for critical so operators see it in startup
        output even without polling /health/detailed.
        """
        if self.w3 is None or self._connected_address is None:
            return
        try:
            wei = self.w3.eth.get_balance(self._connected_address)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "startup gas check failed (non-fatal): %s", exc,
            )
            return
        eth = wei / 1e18
        addr_short = self._connected_address[:10] + "…"
        if eth < 0.0001:
            logger.error(
                "Operator gas CRITICAL: %.10f ETH on %s — "
                "broadcasts will start failing soon. Top up now.",
                eth, addr_short,
            )
        elif eth < 0.0005:
            logger.warning(
                "Operator gas low: %.10f ETH on %s — plan to "
                "top up to avoid mid-job failures.",
                eth, addr_short,
            )
        else:
            logger.info(
                "Operator gas ok: %.10f ETH on %s",
                eth, addr_short,
            )

    async def _close_persistence(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _record_tx(self, tx: FTNSTransaction) -> None:
        if self._db is None:
            return
        await self._db.execute(
            "INSERT OR REPLACE INTO onchain_transactions "
            "(job_id, tx_hash, from_addr, to_addr, amount_ftns, "
            "status, block_number, created_at, chain_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tx.job_id, tx.tx_hash, tx.from_addr, tx.to_addr,
                tx.amount_ftns, tx.status, tx.block_number,
                tx.created_at, self.chain_id,
            ),
        )
        await self._db.commit()

    async def _update_tx_status(self, tx: FTNSTransaction) -> None:
        if self._db is None:
            return
        await self._db.execute(
            "UPDATE onchain_transactions "
            "SET status = ?, block_number = ?, tx_hash = ? "
            "WHERE job_id = ?",
            (tx.status, tx.block_number, tx.tx_hash, tx.job_id),
        )
        await self._db.commit()

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

            await self._init_persistence()

            self._is_initialized = True
            self._emit_startup_gas_log()
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

                # Use "pending" block so concurrent txs under the shared
                # FTNS_WALLET_PRIVATE_KEY (this ledger + RoyaltyDistributorClient,
                # which locks independently) don't collide on nonce. Matches the
                # pattern in royalty_distributor.py:215.
                nonce = await loop.run_in_executor(
                    None,
                    lambda: self.w3.eth.get_transaction_count(
                        self._connected_address, "pending"
                    ),
                )

                # Build tx
                tx = {
                    "chainId": self.chain_id,
                    "nonce": nonce,
                    "gasPrice": estimate_gas_price(self.w3),
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
                await self._record_tx(tx_record)

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
                await self._update_tx_status(tx_record)

                return tx_record

            except Exception as e:
                tx_record.status = "rejected"
                logger.error(f"FTNS transfer failed: {e}")
                self._transactions.append(tx_record)
                await self._record_tx(tx_record)
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

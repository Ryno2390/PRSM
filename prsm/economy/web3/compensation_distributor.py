"""CompensationDistributor Web3 Client.

Python client for the on-chain ``CompensationDistributor.sol`` contract
(Phase 8; mainnet-deployed 2026-05-07). Closes the readiness gap
surfaced by `docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md`
§6.2: contract has been live but no operator-side Python client
shipped, so `pullAndDistribute` invocation + `Distributed` event
monitoring required Foundation Safe direct calls.

Operator-facing surface only — admin functions (updateWeights,
setPoolAddresses) intentionally NOT exposed; those go through
Foundation Safe direct calls. Operator client provides:

  - pull_and_distribute()                permissionless write
  - current_weights()                    view → PoolWeights
  - last_distribution_timestamp()        view → uint64
  - creator_pool() / operator_pool() / grant_pool()  views → address
  - has_scheduled_weights() / scheduled_at()         views
  - DistributedEvent.from_decoded_args(args)         event decode

Mirrors the patterns established by
``prsm/economy/web3/key_distribution.py``:
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


# ──────────────────────────────────────────────────────────────────────
# Pool weights (mirrors Solidity struct PoolWeights)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoolWeights:
    """Mirror of Solidity ``struct PoolWeights``.

    Each field is a uint16 basis-points value. The contract's
    ``_validateWeights`` enforces sum == 10000; this dataclass mirrors
    that constraint so client-side construction fails fast with a
    clear ValueError rather than discovering the mismatch on-chain.
    """
    creator_bps: int
    operator_bps: int
    grant_bps: int

    def __post_init__(self) -> None:
        for field, value in (
            ("creator_bps", self.creator_bps),
            ("operator_bps", self.operator_bps),
            ("grant_bps", self.grant_bps),
        ):
            if not isinstance(value, int):
                raise ValueError(
                    f"{field} must be int, got {type(value).__name__}"
                )
            if value < 0 or value > 0xFFFF:
                raise ValueError(
                    f"{field} must fit uint16 [0, 65535], got {value}"
                )
        total = self.creator_bps + self.operator_bps + self.grant_bps
        if total != 10_000:
            raise ValueError(
                f"weights must sum to 10000 bps, got {total}"
            )


# ──────────────────────────────────────────────────────────────────────
# Decoded events
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DistributedEvent:
    """Decoded ``Distributed(uint256 toCreator, uint256 toOperator,
    uint256 toGrant)`` event.

    Each amount is FTNS wei (18 decimals). Sum equals the FTNS
    balance of the distributor at the time of distribution; per
    contract integer-bps math, dust accrues to ``to_grant``.

    Sprint 549: ``tx_hash`` + ``log_index`` carry the on-chain
    identity of the event so the watcher can dedup across restart
    (mirrors sprint 544's persistent dedup for InboundMonitor).
    Both are Optional — pre-sprint-549 producers without identity
    info still build valid events; the watcher dedup just no-ops
    when either is missing.
    """
    to_creator: int
    to_operator: int
    to_grant: int
    tx_hash: Optional[str] = None
    log_index: Optional[int] = None

    def __post_init__(self) -> None:
        for field, value in (
            ("to_creator", self.to_creator),
            ("to_operator", self.to_operator),
            ("to_grant", self.to_grant),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{field} must be non-negative int, got {value!r}"
                )

    @classmethod
    def from_decoded_args(
        cls,
        args: Dict[str, Any],
        *,
        tx_hash: Optional[str] = None,
        log_index: Optional[int] = None,
    ) -> "DistributedEvent":
        """Build from a Web3.py decoded-event-args dict.

        ``args`` is the dict produced by
        ``contract.events.Distributed().process_log(log)`` where keys
        are the Solidity argument names in camelCase. ``tx_hash`` +
        ``log_index`` (sprint 549) are populated by the client's
        ``get_distributed_events`` from the raw log envelope.
        """
        return cls(
            to_creator=int(args["toCreator"]),
            to_operator=int(args["toOperator"]),
            to_grant=int(args["toGrant"]),
            tx_hash=tx_hash,
            log_index=log_index,
        )


# ──────────────────────────────────────────────────────────────────────
# Contract ABI (minimal — only the surface we exercise)
# ──────────────────────────────────────────────────────────────────────


COMPENSATION_DISTRIBUTOR_ABI = [
    {
        "type": "function",
        "name": "pullAndDistribute",
        "stateMutability": "nonpayable",
        "inputs": [],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "currentWeights",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [
            {"name": "creatorPoolBps", "type": "uint16"},
            {"name": "operatorPoolBps", "type": "uint16"},
            {"name": "grantPoolBps", "type": "uint16"},
        ],
    },
    {
        "type": "function",
        "name": "lastDistributionTimestamp",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint64"}],
    },
    {
        "type": "function",
        "name": "creatorPool",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
    },
    {
        "type": "function",
        "name": "operatorPool",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
    },
    {
        "type": "function",
        "name": "grantPool",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
    },
    {
        "type": "function",
        "name": "hasScheduledWeights",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "type": "function",
        "name": "scheduledAt",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint64"}],
    },
    {
        "type": "event",
        "name": "Distributed",
        "anonymous": False,
        "inputs": [
            {"name": "toCreator", "type": "uint256", "indexed": False},
            {"name": "toOperator", "type": "uint256", "indexed": False},
            {"name": "toGrant", "type": "uint256", "indexed": False},
        ],
    },
]


# ──────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────


class CompensationDistributorClient:
    """Sync Web3 client for CompensationDistributor.

    Construction:
        rpc_url: Base mainnet (or testnet) RPC endpoint.
        contract_address: CompensationDistributor.sol deploy address.
        private_key: Optional. Required for write call
            (pull_and_distribute); read paths work without one.

    Note: ``pull_and_distribute`` is permissionless on-chain — anyone
    can call. The private key is only required to fund the gas of the
    call.
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
            abi=COMPENSATION_DISTRIBUTOR_ABI,
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

    # ── Writes ─────────────────────────────────────────────────

    def pull_and_distribute(self) -> Tuple[str, TransferStatus]:
        """Trigger the permissionless pull-and-distribute flow.

        On-chain effect: pulls the currently-allowed FTNS emission
        from EmissionController, then distributes the contract's full
        FTNS balance across (creator, operator, grant) pools per
        currentWeights. Activates scheduledWeights if their
        scheduledAt has been reached.

        Returns (tx_hash_hex, TransferStatus). May raise
        BroadcastFailedError (safe fallback), OnChainPendingError
        (do NOT fall back), or OnChainRevertedError (safe fallback —
        e.g. if FTNS transfer to a pool fails).
        """
        if not self._account:
            raise RuntimeError(
                "private_key required for write calls (pullAndDistribute)"
            )

        with self._tx_lock:
            tx = self.contract.functions.pullAndDistribute(
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    # ── Views ──────────────────────────────────────────────────

    def current_weights(self) -> PoolWeights:
        """Read currentWeights → PoolWeights dataclass.

        The contract returns a 3-tuple (creatorBps, operatorBps,
        grantBps); we wrap into the dataclass which validates sum
        == 10000 (a contract invariant).
        """
        creator, operator, grant = self.contract.functions.currentWeights().call()
        return PoolWeights(
            creator_bps=int(creator),
            operator_bps=int(operator),
            grant_bps=int(grant),
        )

    def last_distribution_timestamp(self) -> int:
        """Read lastDistributionTimestamp → uint64.

        Zero before any distribution has occurred.
        """
        return int(self.contract.functions.lastDistributionTimestamp().call())

    def creator_pool(self) -> str:
        return str(self.contract.functions.creatorPool().call())

    def operator_pool(self) -> str:
        return str(self.contract.functions.operatorPool().call())

    def grant_pool(self) -> str:
        return str(self.contract.functions.grantPool().call())

    def has_scheduled_weights(self) -> bool:
        return bool(self.contract.functions.hasScheduledWeights().call())

    def scheduled_at(self) -> int:
        return int(self.contract.functions.scheduledAt().call())

    # ── Event stream ──────────────────────────────────────────

    def latest_block(self) -> int:
        return int(self.web3.eth.block_number)

    def get_distributed_events(
        self, from_block: int, to_block: int,
    ) -> "list[DistributedEvent]":
        if from_block > to_block:
            return []
        logs = self.contract.events.Distributed().get_logs(
            from_block=from_block, to_block=to_block,
        )
        out = []
        for log in logs:
            # Sprint 549: thread tx_hash + log_index into the
            # DistributedEvent so the watcher can dedup. Raw web3.py
            # log objects expose transactionHash (bytes) + logIndex
            # (int); coerce to a canonical "0x..." hex string for
            # tx_hash so it matches the EventDedupStore key shape.
            tx_bytes = log.get("transactionHash") if isinstance(
                log, dict,
            ) else getattr(log, "transactionHash", None)
            if isinstance(tx_bytes, (bytes, bytearray)):
                tx_hex: Optional[str] = "0x" + tx_bytes.hex()
            elif isinstance(tx_bytes, str):
                tx_hex = tx_bytes if tx_bytes.startswith("0x") else (
                    "0x" + tx_bytes
                )
            else:
                tx_hex = None
            log_idx = log.get("logIndex") if isinstance(
                log, dict,
            ) else getattr(log, "logIndex", None)
            out.append(DistributedEvent.from_decoded_args(
                log["args"],
                tx_hash=tx_hex,
                log_index=(
                    int(log_idx) if isinstance(log_idx, int) else None
                ),
            ))
        return out

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

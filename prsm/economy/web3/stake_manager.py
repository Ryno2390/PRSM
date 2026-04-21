"""StakeBond Web3 Client — Phase 7 Task 4.

Thin Python wrapper around the on-chain StakeBond contract. Mirrors the
patterns established by provenance_registry.py — Web3.py 7.x,
synchronous calls, explicit private-key signing, per-client tx lock to
prevent nonce races.

StakeBond is the provider-stake registry that backs Tier C slashing.
Providers bond FTNS to claim a tier; successful challenges against them
invalidate the receipt AND burn/redirect their stake (70% to challenger,
30% to Foundation reserve; 100% to Foundation on self-slash).

Writes exposed: bond, request_unbond, withdraw, claim_bounty.
Reads exposed: stake_of, effective_tier, slashed_bounty_payable,
foundation_reserve_balance, unbond_delay_seconds.

Governance entries (setSlasher/setUnbondDelay/setFoundationReserveWallet)
are intentionally NOT wrapped — they run from the owner multi-sig via
the hardhat CLI, not from production Python code.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

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

# Tier thresholds in FTNS wei — must mirror StakeBond.sol:effectiveTier.
# If the contract thresholds change, these MUST be updated in lock-step.
TIER_STANDARD_MIN_WEI = 5_000 * 10**18
TIER_PREMIUM_MIN_WEI = 25_000 * 10**18
TIER_CRITICAL_MIN_WEI = 50_000 * 10**18

# Tier → recommended slash rate (bps). Callers pass one of these to bond()
# unless they are intentionally bonding at a non-standard rate.
SLASH_RATE_STANDARD_BPS = 5_000   # 50%
SLASH_RATE_PREMIUM_BPS = 10_000   # 100%
SLASH_RATE_CRITICAL_BPS = 10_000  # 100%


class StakeStatus(IntEnum):
    """Mirrors StakeBond.sol:StakeStatus."""
    NONE = 0
    BONDED = 1
    UNBONDING = 2
    WITHDRAWN = 3


# Hardcoded ABI subset — only the surface this wrapper exposes. Matches
# contracts/artifacts/contracts/StakeBond.sol/StakeBond.json.
STAKE_BOND_ABI = [
    {
        "inputs": [
            {"internalType": "uint128", "name": "amount", "type": "uint128"},
            {"internalType": "uint16", "name": "tierSlashRateBps", "type": "uint16"},
        ],
        "name": "bond",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "requestUnbond",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "withdraw",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "claimBounty",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "provider", "type": "address"}],
        "name": "stakeOf",
        "outputs": [
            {
                "components": [
                    {"internalType": "uint128", "name": "amount", "type": "uint128"},
                    {"internalType": "uint64", "name": "bonded_at_unix", "type": "uint64"},
                    {"internalType": "uint64", "name": "unbond_eligible_at", "type": "uint64"},
                    {"internalType": "uint8", "name": "status", "type": "uint8"},
                    {"internalType": "uint16", "name": "tier_slash_rate_bps", "type": "uint16"},
                ],
                "internalType": "struct StakeBond.Stake",
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "provider", "type": "address"}],
        "name": "effectiveTier",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "challenger", "type": "address"}],
        "name": "slashedBountyPayable",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "foundationReserveBalance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "unbondDelaySeconds",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "provider", "type": "address"},
            {"indexed": False, "internalType": "uint128", "name": "amount", "type": "uint128"},
            {"indexed": False, "internalType": "uint16", "name": "tierSlashRateBps", "type": "uint16"},
            {"indexed": False, "internalType": "uint64", "name": "bondedAt", "type": "uint64"},
        ],
        "name": "Bonded",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "provider", "type": "address"},
            {"indexed": False, "internalType": "uint64", "name": "eligibleAt", "type": "uint64"},
        ],
        "name": "UnbondRequested",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "provider", "type": "address"},
            {"indexed": False, "internalType": "uint128", "name": "amount", "type": "uint128"},
        ],
        "name": "Withdrawn",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "provider", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "challenger", "type": "address"},
            {"indexed": True, "internalType": "bytes32", "name": "reasonId", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "slashAmount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "challengerBounty", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "foundationShare", "type": "uint256"},
        ],
        "name": "Slashed",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "challenger", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "BountyClaimed",
        "type": "event",
    },
]


@dataclass
class StakeRecord:
    """View of a provider's on-chain stake."""
    amount_wei: int
    bonded_at_unix: int
    unbond_eligible_at: int
    status: StakeStatus
    tier_slash_rate_bps: int

    @property
    def is_bonded(self) -> bool:
        return self.status == StakeStatus.BONDED

    @property
    def is_unbonding(self) -> bool:
        return self.status == StakeStatus.UNBONDING


class StakeManagerClient:
    """Sync Web3 client for StakeBond.

    One client per provider keypair. Governance addresses (owner /
    slasher) are NOT routed through this wrapper — those operate
    from the owner multi-sig via hardhat.
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
            abi=STAKE_BOND_ABI,
        )

        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

        self._tx_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # ── Writes ─────────────────────────────────────────────────

    def bond(
        self, amount_wei: int, tier_slash_rate_bps: int
    ) -> Tuple[str, TransferStatus]:
        """Post FTNS stake and snapshot the slash rate.

        Caller must have already approved the StakeBond address for
        `amount_wei` on the FTNS token contract — this wrapper does NOT
        issue the approval (keeping token and stake concerns separate).

        Returns (tx_hash_hex, TransferStatus). May raise
        BroadcastFailedError, OnChainPendingError, or OnChainRevertedError.
        """
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        if amount_wei <= 0:
            raise ValueError(f"amount_wei must be positive (got {amount_wei})")
        if amount_wei >= 2**128:
            raise ValueError(
                f"amount_wei must fit in uint128 (got {amount_wei})"
            )
        if not (0 <= tier_slash_rate_bps <= 10_000):
            raise ValueError(
                f"tier_slash_rate_bps must be in [0, 10000] "
                f"(got {tier_slash_rate_bps})"
            )

        with self._tx_lock:
            tx = self.contract.functions.bond(
                amount_wei, tier_slash_rate_bps
            ).build_transaction(self._tx_overrides())
            return self._sign_and_send(tx)

    def request_unbond(self) -> Tuple[str, TransferStatus]:
        """Transition BONDED → UNBONDING. Slashing remains active during
        the unbond delay — a provider caught mid-unbond still forfeits."""
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        with self._tx_lock:
            tx = self.contract.functions.requestUnbond().build_transaction(
                self._tx_overrides()
            )
            return self._sign_and_send(tx)

    def withdraw(self) -> Tuple[str, TransferStatus]:
        """Finalize unbond: UNBONDING → WITHDRAWN and return FTNS.
        Reverts on-chain if the unbond delay hasn't elapsed."""
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        with self._tx_lock:
            tx = self.contract.functions.withdraw().build_transaction(
                self._tx_overrides()
            )
            return self._sign_and_send(tx)

    def claim_bounty(self) -> Tuple[str, TransferStatus]:
        """Claim any slashed-bounty FTNS owed to caller as challenger.
        Reverts on-chain if the caller has zero bounty payable."""
        if not self._account:
            raise RuntimeError("private_key required for write calls")
        with self._tx_lock:
            tx = self.contract.functions.claimBounty().build_transaction(
                self._tx_overrides()
            )
            return self._sign_and_send(tx)

    # ── Reads ──────────────────────────────────────────────────

    def stake_of(self, provider: str) -> StakeRecord:
        """Return the full stake record for `provider`. If the provider
        has never bonded, the record's status is NONE and amount is 0."""
        provider_cs = Web3.to_checksum_address(provider)
        raw = self.contract.functions.stakeOf(provider_cs).call()
        # raw is a tuple (amount, bonded_at, unbond_eligible_at, status, rate_bps)
        amount, bonded_at, unbond_eligible_at, status, rate_bps = raw
        return StakeRecord(
            amount_wei=int(amount),
            bonded_at_unix=int(bonded_at),
            unbond_eligible_at=int(unbond_eligible_at),
            status=StakeStatus(int(status)),
            tier_slash_rate_bps=int(rate_bps),
        )

    def effective_tier(self, provider: str) -> str:
        """Return the current effective tier ("open"/"standard"/
        "premium"/"critical"). Drops to "open" during UNBONDING."""
        provider_cs = Web3.to_checksum_address(provider)
        return self.contract.functions.effectiveTier(provider_cs).call()

    def slashed_bounty_payable(self, challenger: str) -> int:
        """Return FTNS wei claimable by `challenger` from past slashings."""
        challenger_cs = Web3.to_checksum_address(challenger)
        return int(
            self.contract.functions.slashedBountyPayable(challenger_cs).call()
        )

    def foundation_reserve_balance(self) -> int:
        return int(self.contract.functions.foundationReserveBalance().call())

    def unbond_delay_seconds(self) -> int:
        return int(self.contract.functions.unbondDelaySeconds().call())

    # ── Internals ──────────────────────────────────────────────

    def _tx_overrides(self) -> dict:
        return {
            "from": self._account.address,
            "nonce": self.web3.eth.get_transaction_count(
                self._account.address, "pending"
            ),
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.web3.eth.chain_id,
        }

    def _sign_and_send(self, tx: dict) -> Tuple[str, TransferStatus]:
        signed = self.web3.eth.account.sign_transaction(tx, self._account.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )

        try:
            tx_hash_bytes = self.web3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise BroadcastFailedError(f"broadcast failed: {exc}") from exc

        tx_hash_hex = "0x" + tx_hash_bytes.hex()

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


# Re-export the shared exception types so callers importing from this
# module don't need to reach into provenance_registry.
__all__ = [
    "StakeManagerClient",
    "StakeRecord",
    "StakeStatus",
    "TransferStatus",
    "BroadcastFailedError",
    "OnChainPendingError",
    "OnChainRevertedError",
    "TIER_STANDARD_MIN_WEI",
    "TIER_PREMIUM_MIN_WEI",
    "TIER_CRITICAL_MIN_WEI",
    "SLASH_RATE_STANDARD_BPS",
    "SLASH_RATE_PREMIUM_BPS",
    "SLASH_RATE_CRITICAL_BPS",
]

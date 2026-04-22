"""Read-only Python wrapper around Phase 8 EmissionController.

Per docs/2026-04-22-phase8-design-plan.md §4.3 + §6 Task 4.

The wrapper is deliberately read-only — the controller's write surface
(`mintAuthorized`, `setAuthorizedDistributor`, `pause`/`resume`) is consumed
by the on-chain CompensationDistributor and by Foundation-multi-sig Hardhat
CLI flows, not by production Python. Giving operator dashboards a writable
client would be a foot-gun.

Usage:

    client = EmissionClient(w3, controller_address="0xabc...")
    snap = client.snapshot()
    print(f"Epoch {snap.current_epoch}, rate {snap.current_epoch_rate_per_sec} wei/s")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

try:
    from web3 import Web3
    HAS_WEB3 = True
except ImportError:  # pragma: no cover
    HAS_WEB3 = False
    Web3 = None  # type: ignore


logger = logging.getLogger(__name__)


# Hardcoded ABI subset covering only the reads the watcher + dashboard need.
# Matches contracts/contracts/EmissionController.sol. Kept in lock-step with
# the Solidity source; CI runs a regression check against the compiled
# artifact.
EMISSION_CONTROLLER_ABI: list[dict[str, Any]] = [
    {
        "inputs": [],
        "name": "currentEpoch",
        "outputs": [{"internalType": "uint32", "name": "", "type": "uint32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "currentEpochRate",
        "outputs": [{"internalType": "uint256", "name": "ratePerSecond", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "timeUntilNextHalving",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "mintedToDate",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "mintCap",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "paused",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "lastMintTimestamp",
        "outputs": [{"internalType": "uint64", "name": "", "type": "uint64"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "recipient", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
            {"indexed": False, "internalType": "uint32", "name": "epoch", "type": "uint32"},
            {"indexed": False, "internalType": "uint256", "name": "epochRate", "type": "uint256"},
        ],
        "name": "Minted",
        "type": "event",
    },
]


@dataclass(frozen=True)
class EmissionState:
    """Point-in-time snapshot of the on-chain emission curve."""

    current_epoch: int
    current_epoch_rate_per_sec: int
    time_until_next_halving_sec: int
    minted_to_date_wei: int
    mint_cap_wei: int
    is_paused: bool
    last_mint_timestamp: int


@dataclass(frozen=True)
class MintEvent:
    """One observed `Minted` event."""

    recipient: str
    amount_wei: int
    epoch: int
    epoch_rate_wei_per_sec: int
    block_number: int
    tx_hash: str


class EmissionClient:
    """Sync web3 read wrapper around EmissionController.

    Matches the structural pattern of prsm.economy.web3.stake_manager —
    explicit Web3 instance injection, no hidden state. Callers own provider
    + network configuration.
    """

    def __init__(self, w3: "Web3", controller_address: str) -> None:
        if not HAS_WEB3:  # pragma: no cover
            raise RuntimeError("web3 package required")
        self._w3 = w3
        self._controller_address = Web3.to_checksum_address(controller_address)
        self._contract = w3.eth.contract(
            address=self._controller_address,
            abi=EMISSION_CONTROLLER_ABI,
        )

    # ---- scalar reads ------------------------------------------------------

    def current_epoch(self) -> int:
        return int(self._contract.functions.currentEpoch().call())

    def current_epoch_rate_per_sec(self) -> int:
        return int(self._contract.functions.currentEpochRate().call())

    def time_until_next_halving_sec(self) -> int:
        return int(self._contract.functions.timeUntilNextHalving().call())

    def minted_to_date_wei(self) -> int:
        return int(self._contract.functions.mintedToDate().call())

    def mint_cap_wei(self) -> int:
        return int(self._contract.functions.mintCap().call())

    def is_paused(self) -> bool:
        return bool(self._contract.functions.paused().call())

    def last_mint_timestamp(self) -> int:
        return int(self._contract.functions.lastMintTimestamp().call())

    # ---- bundled snapshot --------------------------------------------------

    def snapshot(self) -> EmissionState:
        """All scalar reads in one returned struct.

        Note: individual calls are NOT atomic — they're separate RPCs and a
        block boundary can fall between them. For most dashboard uses the
        cross-field drift is acceptable (sub-second at typical poll
        cadences). Monitoring alerts that need atomicity should read at a
        pinned block number.
        """
        return EmissionState(
            current_epoch=self.current_epoch(),
            current_epoch_rate_per_sec=self.current_epoch_rate_per_sec(),
            time_until_next_halving_sec=self.time_until_next_halving_sec(),
            minted_to_date_wei=self.minted_to_date_wei(),
            mint_cap_wei=self.mint_cap_wei(),
            is_paused=self.is_paused(),
            last_mint_timestamp=self.last_mint_timestamp(),
        )

    # ---- event stream ------------------------------------------------------

    def latest_block(self) -> int:
        return int(self._w3.eth.block_number)

    def get_minted_events(self, from_block: int, to_block: int) -> List[MintEvent]:
        """Fetch Minted events in [from_block, to_block] (inclusive).

        Ordering is chain-order (ascending block, then log index). Callers
        passing a huge range should chunk — RPCs typically cap get_logs at
        ~10k blocks.
        """
        if from_block > to_block:
            return []
        event = self._contract.events.Minted()
        logs = event.get_logs(from_block=from_block, to_block=to_block)
        result: list[MintEvent] = []
        for log in logs:
            args = log["args"]
            result.append(
                MintEvent(
                    recipient=args["recipient"],
                    amount_wei=int(args["amount"]),
                    epoch=int(args["epoch"]),
                    epoch_rate_wei_per_sec=int(args["epochRate"]),
                    block_number=int(log["blockNumber"]),
                    tx_hash=log["transactionHash"].hex()
                    if hasattr(log["transactionHash"], "hex")
                    else str(log["transactionHash"]),
                )
            )
        return result

    @property
    def controller_address(self) -> str:
        return self._controller_address

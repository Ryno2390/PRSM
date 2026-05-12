"""Sprint 298 — EmergencyPauseClient + Foundation Safe
pause-tx composer.

Vision §14 "Smart-contract exploit risk" names emergency
pause as the only mitigation that limits blast radius
DURING an active exploit. The Foundation Safe (2-of-3
hardware multisig at `0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791`)
holds pause authority on every Phase 1.3 + audit-bundle +
Phase 7/8 contract (sole-owned via acceptOwnership
ceremonies 2026-05-04/06/07).

PRSM never executes pause directly. This client COMPOSES the
pause transaction that operators upload to the Safe UI for
multi-sig signing. The composer surface itself doesn't grant
authority — it only encodes the well-known OpenZeppelin
Pausable selectors against the canonical contract addresses
from `prsm.config.networks`.

Architecture:
- `PauseEligibleContract` — registry entry; name + kind +
  human-readable description
- `PAUSE_ELIGIBLE_CONTRACTS` — static registry, pinned to
  the contracts that are owned by Foundation Safe + carry
  OZ Pausable
- `EmergencyPauseClient` — composer + status reader
  - `is_paused(contract_name)` — calls `paused()` view
  - `status_all()` — bulk status query, returns
    Dict[name, ContractPauseStatus]
  - `compose_pause_tx(contract_name)` → Safe-uploadable
    dict (to + data + value + warning + explorer_url)
  - `compose_unpause_tx(contract_name)` → same for unpause
- `from_env()` factory reads `PRSM_NETWORK` (defaults
  mainnet) + uses `prsm.config.networks` canonical addresses
- Backend protocol abstracts the chain RPC for test
  injection; production wires real `web3.eth.call`.

Status query is read-only (anyone can call `paused()` on a
public contract). Compose is composer-only — even a
compromised operator cannot pause without Foundation Safe
multi-sig approval. The composer surface exists purely to
SAVE OPERATORS TIME constructing pause calldata by hand
during an active incident; speed matters when an exploit is
ongoing.

Function selectors (well-known OZ Pausable standard, pinned
in case the spec evolves):
  pause()    0x8456cb59
  unpause()  0x3f4ba83a
  paused()   0x5c975abb
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol

logger = logging.getLogger(__name__)


# ── OZ Pausable function selectors ───────────────────────

PAUSE_SELECTOR = "0x8456cb59"
UNPAUSE_SELECTOR = "0x3f4ba83a"
PAUSED_SELECTOR = "0x5c975abb"


@dataclass
class PauseEligibleContract:
    """Registry entry for a Foundation-Safe-owned + OZ-
    Pausable contract."""

    name: str
    address: Optional[str] = None  # populated at runtime from networks.py
    kind: str = "OZPausable"  # OZPausable / ERC20Pausable / Custom
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "address": self.address,
            "kind": self.kind,
            "description": self.description,
        }


# Registry: contracts that are pause-eligible. The address
# field is None here — populated at runtime via the
# from_env() factory using networks.py canonical addresses.
PAUSE_ELIGIBLE_CONTRACTS = [
    PauseEligibleContract(
        name="ftns_token",
        kind="ERC20Pausable",
        description=(
            "FTNS ERC-20 token transfers. PAUSER_ROLE held "
            "by Foundation Safe. Pausing halts all transfers "
            "including royalty distributions and operator "
            "compensation payouts."
        ),
    ),
    PauseEligibleContract(
        name="royalty_distributor",
        description=(
            "RoyaltyDistributor v2 (A-08 ceremony "
            "2026-05-09). Halts royalty claim() execution "
            "for content creators; existing claimable "
            "balances preserved."
        ),
    ),
    PauseEligibleContract(
        name="escrow_pool",
        description=(
            "EscrowPool. Halts new escrow creation + release; "
            "existing escrows preserved for post-pause "
            "settlement."
        ),
    ),
    PauseEligibleContract(
        name="stake_bond",
        description=(
            "StakeBond. Halts new stakes + slash execution; "
            "existing bonds preserved."
        ),
    ),
    PauseEligibleContract(
        name="settlement_registry",
        description=(
            "BatchSettlementRegistry. Halts new settlement "
            "batch submission."
        ),
    ),
    PauseEligibleContract(
        name="signature_verifier",
        description=(
            "Ed25519Verifier. Halts on-chain signature "
            "verification — used as a defense-in-depth "
            "shutdown when downstream contracts cannot pause."
        ),
    ),
    PauseEligibleContract(
        name="emission_controller",
        description=(
            "EmissionController. Halts FTNS emission "
            "execution; the halving curve continues "
            "logically but no tokens mint during the pause."
        ),
    ),
    PauseEligibleContract(
        name="compensation_distributor",
        description=(
            "CompensationDistributor. Halts operator + "
            "creator payouts."
        ),
    ),
    PauseEligibleContract(
        name="storage_slashing",
        description=(
            "StorageSlashing. Halts slash execution on "
            "failed storage proofs; failed challenges queue "
            "for post-pause review."
        ),
    ),
    PauseEligibleContract(
        name="key_distribution",
        description=(
            "KeyDistribution. Halts Tier-C decryption-key "
            "release; sensitive content stays encrypted "
            "during the pause."
        ),
    ),
]


@dataclass
class ContractPauseStatus:
    """Per-contract status snapshot."""

    name: str
    address: Optional[str]
    paused: Optional[bool]  # None when unknown (uncommissioned or RPC error)
    commissioned: bool  # True iff address populated
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "address": self.address,
            "paused": self.paused,
            "commissioned": self.commissioned,
            "error": self.error,
        }


class _ChainBackend(Protocol):
    """Dependency-injected chain RPC. Production wraps a
    real `web3.eth.call`; tests use a fake."""

    def call(self, to_address: str, data: str) -> bytes: ...


_DEFAULT_WARNING = (
    "DESTRUCTIVE: pausing this contract halts user-facing "
    "operations. Requires Foundation Safe 2-of-3 hardware "
    "multisig approval. Upload the encoded calldata to the "
    "Safe UI; signers (Ledger/Trezor/OneKey) verify the "
    "target address + selector before signing."
)


class EmergencyPauseClient:
    def __init__(
        self,
        contract_addresses: Mapping[str, Optional[str]],
        rpc_url: Optional[str],
        *,
        chain_id: Optional[int] = None,
        backend: Optional[_ChainBackend] = None,
    ) -> None:
        self._addresses = dict(contract_addresses)
        self._rpc_url = rpc_url
        self._chain_id = chain_id
        self._backend = backend

    @classmethod
    def from_env(
        cls, *, backend: Optional[_ChainBackend] = None,
    ) -> "EmergencyPauseClient":
        from prsm.config.networks import get_network_config

        network_name = os.environ.get("PRSM_NETWORK", "mainnet")
        cfg = get_network_config(network_name)
        addresses = {
            "ftns_token": cfg.ftns_token,
            "royalty_distributor": cfg.royalty_distributor,
            "escrow_pool": cfg.escrow_pool,
            "stake_bond": cfg.stake_bond,
            "settlement_registry": cfg.settlement_registry,
            "signature_verifier": cfg.signature_verifier,
            "emission_controller": cfg.emission_controller,
            "compensation_distributor": (
                cfg.compensation_distributor
            ),
            "storage_slashing": cfg.storage_slashing,
            "key_distribution": cfg.key_distribution,
        }
        rpc_url = (
            os.environ.get("BASE_RPC_URL")
            or cfg.rpc_url_default
        )
        return cls(
            contract_addresses=addresses,
            rpc_url=rpc_url,
            chain_id=cfg.chain_id,
            backend=backend,
        )

    # ── Registry access ──────────────────────────────────

    def get_contract(
        self, contract_name: str,
    ) -> PauseEligibleContract:
        for c in PAUSE_ELIGIBLE_CONTRACTS:
            if c.name == contract_name:
                # Populate address from runtime config
                return PauseEligibleContract(
                    name=c.name,
                    address=self._addresses.get(c.name),
                    kind=c.kind,
                    description=c.description,
                )
        raise ValueError(
            f"{contract_name!r} not in registry. "
            f"Valid: "
            f"{[c.name for c in PAUSE_ELIGIBLE_CONTRACTS]}"
        )

    # ── Read paths ───────────────────────────────────────

    def is_paused(
        self, contract_name: str,
    ) -> Optional[bool]:
        """True if paused, False if not paused, None if
        unknown (uncommissioned address or RPC error).
        Fail-soft — never raises out of this method so
        operator monitoring loops don't crash on transient
        RPC issues."""
        c = self.get_contract(contract_name)  # raises if unknown
        if c.address is None or self._backend is None:
            return None
        try:
            result = self._backend.call(
                c.address, PAUSED_SELECTOR,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EmergencyPauseClient: paused() call raised "
                "for %s @ %s: %s",
                contract_name, c.address, exc,
            )
            return None
        # OZ Pausable.paused() returns ABI-encoded bool — 32
        # bytes big-endian; last byte=1 if paused.
        if not result or len(result) < 32:
            return None
        return result[-1] == 1

    def status_all(
        self,
    ) -> Dict[str, ContractPauseStatus]:
        out: Dict[str, ContractPauseStatus] = {}
        for c_template in PAUSE_ELIGIBLE_CONTRACTS:
            addr = self._addresses.get(c_template.name)
            commissioned = bool(addr)
            paused: Optional[bool] = None
            error: Optional[str] = None
            if commissioned and self._backend is not None:
                try:
                    result = self._backend.call(
                        addr, PAUSED_SELECTOR,
                    )
                    if result and len(result) >= 32:
                        paused = result[-1] == 1
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
            out[c_template.name] = ContractPauseStatus(
                name=c_template.name,
                address=addr,
                paused=paused,
                commissioned=commissioned,
                error=error,
            )
        return out

    # ── Compose paths ────────────────────────────────────

    def compose_pause_tx(
        self, contract_name: str,
    ) -> Dict[str, Any]:
        return self._compose_tx(
            contract_name,
            selector=PAUSE_SELECTOR,
            action="pause",
        )

    def compose_unpause_tx(
        self, contract_name: str,
    ) -> Dict[str, Any]:
        return self._compose_tx(
            contract_name,
            selector=UNPAUSE_SELECTOR,
            action="unpause",
        )

    def _compose_tx(
        self,
        contract_name: str,
        *,
        selector: str,
        action: str,
    ) -> Dict[str, Any]:
        c = self.get_contract(contract_name)
        if c.address is None:
            raise ValueError(
                f"{contract_name!r} address not configured "
                f"on this network (set via PRSM_NETWORK env "
                f"or networks.py registry)"
            )
        explorer = self._explorer_url_for_address(c.address)
        return {
            "action": action,
            "to": c.address,
            "data": selector,
            "value": "0",
            "contract_name": contract_name,
            "description": (
                f"{action.upper()} the {contract_name} "
                f"contract. {c.description}"
            ),
            "warning": _DEFAULT_WARNING,
            "explorer_url": explorer,
            "chain_id": self._chain_id,
            "instructions": (
                "1) Open the Foundation Safe UI; "
                "2) Create a new transaction with the "
                "`to`, `data`, and `value` fields above; "
                "3) 2-of-3 hardware signers verify the "
                "target address matches `explorer_url` "
                "and the selector matches `data` before "
                "signing; "
                f"4) Execute. The contract will be "
                f"{action}d immediately on confirmation."
            ),
        }

    def _explorer_url_for_address(
        self, address: str,
    ) -> Optional[str]:
        if self._chain_id == 8453:
            return f"https://basescan.org/address/{address}"
        if self._chain_id == 84532:
            return (
                f"https://sepolia.basescan.org/address/"
                f"{address}"
            )
        return None

"""Sprint 299 — InsuranceFundTracker + recovery transfer
composer (Vision §14 mitigation item 2).

Vision §14: "Foundation reserves at least 5% of treasury
value as a dedicated insurance fund earmarked for exploit
recovery. Public, on-chain verification. Sized to ensure
that even successful exploits do not result in unrecoverable
user losses."

This module ships the operator-side surface to that promise:

  InsuranceFundTracker
    Reads on-chain FTNS balance of a designated insurance-
    fund address; compares against the Foundation Safe
    treasury total to compute reserve ratio in basis points.
    Default target 500 bps (5%). Tunable via env.

  compose_recovery_transfer_tx
    Safe-uploadable ERC-20 transfer payload that moves
    insurance funds to a recovery wallet during post-exploit
    response. COMPOSER-ONLY: PRSM never executes the
    transfer; Foundation Safe 2-of-3 multisig is the gate.

  encode_erc20_transfer_calldata
    Pure helper that ABI-encodes the standard ERC-20
    transfer(address,uint256) call. Used by the composer +
    available as a module-level utility for callers that
    want to construct custom recovery flows.

Mirrors sprint 298's safety scaffolding: every composed tx
carries DESTRUCTIVE warning + explorer URL + numbered Safe-
UI instructions + audit-trail reason field.

ERC-20 transfer selector: 0xa9059cbb (transfer(address,uint256))
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


ERC20_TRANSFER_SELECTOR = "0xa9059cbb"

_HEX_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def encode_erc20_transfer_calldata(
    recipient: str, amount_wei: int,
) -> str:
    """ABI-encode ERC-20 transfer(address,uint256). Returns
    0x-prefixed hex string.

    Layout:
      bytes 0-3:   selector (0xa9059cbb)
      bytes 4-35:  recipient address, right-aligned in
                   32-byte slot (12 zero bytes + 20-byte addr)
      bytes 36-67: amount, big-endian uint256
    """
    if not recipient or not isinstance(recipient, str):
        raise ValueError("recipient must be a non-empty hex address")
    if not _HEX_ADDR_RE.match(recipient):
        raise ValueError(
            f"recipient must be a 0x-prefixed 40-hex-char "
            f"Ethereum address, got {recipient!r}"
        )
    if not isinstance(amount_wei, int) or amount_wei <= 0:
        raise ValueError(
            f"amount_wei must be a positive integer, "
            f"got {amount_wei!r}"
        )
    # Strip 0x, left-pad to 64 hex chars (32 bytes)
    addr_padded = recipient[2:].lower().rjust(64, "0")
    amount_hex = format(amount_wei, "064x")
    return ERC20_TRANSFER_SELECTOR + addr_padded + amount_hex


@dataclass
class InsuranceFundStatus:
    """Snapshot of insurance fund + treasury balances."""

    fund_address: Optional[str]
    treasury_address: Optional[str]
    fund_balance_wei: Optional[int]
    treasury_balance_wei: Optional[int]
    reserve_ratio_bps: Optional[int]
    target_bps: int
    target_met: bool
    commissioned: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fund_address": self.fund_address,
            "treasury_address": self.treasury_address,
            "fund_balance_wei": self.fund_balance_wei,
            "treasury_balance_wei": self.treasury_balance_wei,
            "reserve_ratio_bps": self.reserve_ratio_bps,
            "target_bps": self.target_bps,
            "target_met": self.target_met,
            "commissioned": self.commissioned,
            "error": self.error,
        }


class _BalanceBackend(Protocol):
    """Dependency-injected balance reader. Production wraps
    real `ERC20.balanceOf` via web3; tests use a fake."""

    def balance_of(self, address: str) -> int: ...


_DEFAULT_TARGET_BPS = 500  # 5%


_RECOVERY_WARNING = (
    "DESTRUCTIVE: this transfer permanently moves FTNS from "
    "the insurance fund. Requires Foundation Safe 2-of-3 "
    "hardware multisig approval. Upload the encoded calldata "
    "to the Safe UI; signers (Ledger/Trezor/OneKey) verify "
    "the target token address + transfer selector + "
    "recipient + amount before signing. Recovery rationale "
    "MUST be documented out-of-band before signing."
)


class InsuranceFundTracker:
    def __init__(
        self,
        *,
        fund_address: Optional[str],
        treasury_address: Optional[str],
        ftns_token_address: Optional[str] = None,
        target_bps: int = _DEFAULT_TARGET_BPS,
        chain_id: Optional[int] = None,
        backend: Optional[_BalanceBackend] = None,
    ) -> None:
        self.fund_address = fund_address
        self.treasury_address = treasury_address
        self.ftns_token_address = ftns_token_address
        self.target_bps = target_bps
        self.chain_id = chain_id
        self._backend = backend

    @classmethod
    def from_env(
        cls, *, backend: Optional[_BalanceBackend] = None,
    ) -> "InsuranceFundTracker":
        from prsm.config.networks import get_network_config

        network_name = os.environ.get("PRSM_NETWORK", "mainnet")
        cfg = get_network_config(network_name)

        fund_address = (
            os.environ.get("PRSM_INSURANCE_FUND_ADDRESS")
            or None
        )
        # Treasury defaults to Foundation Safe per Vision §14
        treasury_address = (
            os.environ.get("PRSM_TREASURY_ADDRESS")
            or cfg.foundation_safe
        )
        target_raw = (
            os.environ.get(
                "PRSM_INSURANCE_FUND_TARGET_BPS",
            ) or ""
        ).strip()
        target_bps = _DEFAULT_TARGET_BPS
        if target_raw:
            try:
                parsed = int(target_raw)
                if 0 < parsed <= 10_000:
                    target_bps = parsed
                else:
                    logger.warning(
                        "PRSM_INSURANCE_FUND_TARGET_BPS=%r "
                        "out of range (0, 10000]; using "
                        "default %d",
                        target_raw, _DEFAULT_TARGET_BPS,
                    )
            except (ValueError, TypeError):
                logger.warning(
                    "PRSM_INSURANCE_FUND_TARGET_BPS=%r "
                    "is not an integer; using default %d",
                    target_raw, _DEFAULT_TARGET_BPS,
                )

        return cls(
            fund_address=fund_address,
            treasury_address=treasury_address,
            ftns_token_address=cfg.ftns_token,
            target_bps=target_bps,
            chain_id=cfg.chain_id,
            backend=backend,
        )

    # ── Read path ────────────────────────────────────────

    def status(self) -> InsuranceFundStatus:
        """Snapshot of insurance fund + treasury + ratio.

        Fail-soft: backend exceptions surface as None
        balances + an error string, never raise out.
        """
        commissioned = bool(
            self.fund_address and self.treasury_address,
        )
        if not commissioned:
            return InsuranceFundStatus(
                fund_address=self.fund_address,
                treasury_address=self.treasury_address,
                fund_balance_wei=None,
                treasury_balance_wei=None,
                reserve_ratio_bps=None,
                target_bps=self.target_bps,
                target_met=False,
                commissioned=False,
                error=None,
            )

        fund_balance: Optional[int] = None
        treasury_balance: Optional[int] = None
        error_msgs = []
        if self._backend is not None:
            try:
                fund_balance = int(
                    self._backend.balance_of(self.fund_address),
                )
            except Exception as exc:  # noqa: BLE001
                error_msgs.append(
                    f"fund balance: {exc}"
                )
                fund_balance = None
            try:
                treasury_balance = int(
                    self._backend.balance_of(
                        self.treasury_address,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                error_msgs.append(
                    f"treasury balance: {exc}"
                )
                treasury_balance = None

        # Compute ratio + target_met. Defensive against
        # None inputs + divide-by-zero.
        reserve_ratio: Optional[int] = None
        target_met = False
        if (
            fund_balance is not None
            and treasury_balance is not None
        ):
            if treasury_balance == 0:
                reserve_ratio = 0
                target_met = False
            else:
                reserve_ratio = (
                    fund_balance * 10_000
                    // treasury_balance
                )
                target_met = (
                    reserve_ratio >= self.target_bps
                )

        return InsuranceFundStatus(
            fund_address=self.fund_address,
            treasury_address=self.treasury_address,
            fund_balance_wei=fund_balance,
            treasury_balance_wei=treasury_balance,
            reserve_ratio_bps=reserve_ratio,
            target_bps=self.target_bps,
            target_met=target_met,
            commissioned=commissioned,
            error=(
                "; ".join(error_msgs) if error_msgs else None
            ),
        )

    # ── Compose path ─────────────────────────────────────

    def compose_recovery_transfer_tx(
        self,
        recipient: str,
        amount_wei: int,
        reason: str,
    ) -> Dict[str, Any]:
        if not self.fund_address:
            raise ValueError(
                "insurance fund address not configured "
                "(set PRSM_INSURANCE_FUND_ADDRESS)"
            )
        if not self.ftns_token_address:
            raise ValueError(
                "ftns_token address not configured; cannot "
                "compose recovery transfer"
            )
        if not reason or not reason.strip():
            raise ValueError(
                "reason is required for audit trail "
                "(short statement of recovery rationale)"
            )
        # encode_erc20_transfer_calldata validates recipient
        # + amount; let its errors bubble.
        data = encode_erc20_transfer_calldata(
            recipient, amount_wei,
        )
        return {
            "action": "recovery_transfer",
            "to": self.ftns_token_address,
            "data": data,
            "value": "0",
            "from_fund": self.fund_address,
            "recipient": recipient,
            "amount_wei": str(amount_wei),
            "reason": reason,
            "chain_id": self.chain_id,
            "warning": _RECOVERY_WARNING,
            "explorer_url": (
                self._explorer_url_for_address(
                    self.ftns_token_address,
                )
            ),
            "instructions": (
                "1) Open the Foundation Safe UI; "
                "2) Create a new transaction with the `to` "
                "(FTNS token contract), `data` (ABI-encoded "
                "transfer call), and `value=0` fields above; "
                "3) 2-of-3 hardware signers verify the "
                "target token, recipient address embedded "
                "in the calldata, and amount before "
                "signing; "
                "4) Execute. FTNS moves from the insurance "
                "fund to the recovery recipient on "
                "confirmation. Document the rationale in "
                "the council ratification record."
            ),
        }

    def _explorer_url_for_address(
        self, address: str,
    ) -> Optional[str]:
        if self.chain_id == 8453:
            return f"https://basescan.org/address/{address}"
        if self.chain_id == 84532:
            return (
                f"https://sepolia.basescan.org/address/"
                f"{address}"
            )
        return None

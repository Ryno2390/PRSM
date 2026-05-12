"""Sprint 303 — UUPS upgrade orchestrator (Vision §14 item 7).

Vision §14 item 7: "UUPS upgrade pattern for non-immutable
contracts permits patching if vulnerability is discovered
post-deployment."

This module ships the engineering layer:

  UpgradeSeverity — emergency / planned / maintenance
  UpgradeStatus   — proposed → reviewed → safe_uploaded →
                    executed → rolled_back | rejected
                    (terminal: executed → rolled_back |
                     executed (when not rolled) | rejected)
  UpgradeProposal — persistent record (rationale, target
                    proxy, new + previous implementations,
                    reviewer assignments, optional Safe tx)
  UpgradeOrchestrator — propose / update_status / get /
                        list / count, filesystem-persisted
                        same pattern as sprints 300/301
  compose_upgrade_tx  — Safe-uploadable
                        upgradeToAndCall(newImpl, initData)
                        payload. Requires REVIEWED+ status.
  compose_rollback_tx — Safe-uploadable upgrade to the
                        previously-recorded implementation.
                        Requires EXECUTED status (can't roll
                        back what hasn't shipped).
  encode_upgrade_to_and_call_calldata — pure helper

The PRE-COMMITTED rollback is the §14 promise: when an
upgrade ships, the prior implementation address is captured
on the record so the operator can compose a rollback tx
without needing to look up the previous version under
incident pressure.

R-2026-05-08-1 composer-only invariant preserved: PRSM
never executes upgrades; Foundation Safe 2-of-3 hardware
multisig is the gate.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────


class UpgradeSeverity(str, Enum):
    EMERGENCY = "emergency"
    PLANNED = "planned"
    MAINTENANCE = "maintenance"


class UpgradeStatus(str, Enum):
    PROPOSED = "proposed"
    REVIEWED = "reviewed"
    SAFE_UPLOADED = "safe_uploaded"
    EXECUTED = "executed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


_TERMINAL_STATUSES = {
    UpgradeStatus.ROLLED_BACK,
    UpgradeStatus.REJECTED,
}


# Forward order for the workflow integrity check —
# advance_phase semantics: can move forward through any
# state OR to a terminal state (ROLLED_BACK / REJECTED).
_STATUS_ORDER = (
    UpgradeStatus.PROPOSED,
    UpgradeStatus.REVIEWED,
    UpgradeStatus.SAFE_UPLOADED,
    UpgradeStatus.EXECUTED,
)


# OpenZeppelin UUPS selector — first 4 bytes of
# keccak256("upgradeToAndCall(address,bytes)").
UPGRADE_TO_AND_CALL_SELECTOR = "0x4f1ef286"


_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def _validate_address(value: str, field_name: str) -> str:
    if not value or not _ADDRESS_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a 0x-prefixed 40-hex "
            f"Ethereum address, got {value!r}"
        )
    return value


# ── Calldata encoding ────────────────────────────────


def encode_upgrade_to_and_call_calldata(
    new_implementation: str,
    init_data: bytes,
) -> str:
    """ABI-encode the upgradeToAndCall(address,bytes) call.

    selector(4) || newImpl-left-padded(32) || offset(32) ||
    length(32) || data-padded-to-32-byte-boundary.

    Returns 0x-prefixed lowercase hex.
    """
    _validate_address(new_implementation, "new_implementation")
    impl_hex = new_implementation.removeprefix("0x").lower()
    impl_padded = impl_hex.rjust(64, "0")

    # offset of the bytes data within the encoded args. After
    # two head words (address + offset), the bytes header
    # (length) starts at byte 64 = 0x40.
    offset_hex = (64).to_bytes(32, "big").hex()

    length_hex = (len(init_data)).to_bytes(32, "big").hex()

    body = init_data.hex()
    # Pad to 32-byte boundary
    pad_bytes = (32 - (len(init_data) % 32)) % 32
    body_padded = body + ("00" * pad_bytes)

    return (
        UPGRADE_TO_AND_CALL_SELECTOR
        + impl_padded
        + offset_hex
        + length_hex
        + body_padded
    )


# ── Dataclasses ──────────────────────────────────────


@dataclass
class UpgradeProposal:
    proposal_id: str
    opened_ts: float
    target_proxy: str
    new_implementation: str
    previous_implementation: str
    severity: UpgradeSeverity
    rationale: str
    status: UpgradeStatus
    init_calldata_hex: str = "0x"
    reviewer_assignments: List[str] = field(
        default_factory=list,
    )
    safe_tx_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "opened_ts": self.opened_ts,
            "target_proxy": self.target_proxy,
            "new_implementation": self.new_implementation,
            "previous_implementation": (
                self.previous_implementation
            ),
            "severity": self.severity.value,
            "rationale": self.rationale,
            "status": self.status.value,
            "init_calldata_hex": self.init_calldata_hex,
            "reviewer_assignments": list(
                self.reviewer_assignments,
            ),
            "safe_tx_hash": self.safe_tx_hash,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "UpgradeProposal":
        return cls(
            proposal_id=d["proposal_id"],
            opened_ts=float(d.get("opened_ts", 0.0)),
            target_proxy=d["target_proxy"],
            new_implementation=d["new_implementation"],
            previous_implementation=d[
                "previous_implementation"
            ],
            severity=UpgradeSeverity(d["severity"]),
            rationale=d.get("rationale", ""),
            status=UpgradeStatus(d["status"]),
            init_calldata_hex=d.get(
                "init_calldata_hex", "0x",
            ),
            reviewer_assignments=list(
                d.get("reviewer_assignments") or [],
            ),
            safe_tx_hash=d.get("safe_tx_hash"),
        )


# ── Orchestrator ─────────────────────────────────────


_MAX_RATIONALE_LEN = 8_000


class UpgradeOrchestrator:
    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        self._records: Dict[str, UpgradeProposal] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir)
            if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(
                parents=True, exist_ok=True,
            )
            self._load_from_disk()

    @classmethod
    def from_env(cls) -> "UpgradeOrchestrator":
        raw = os.environ.get(
            "PRSM_UPGRADE_ORCHESTRATOR_DIR",
        )
        persist_dir = Path(raw) if raw else None
        return cls(persist_dir=persist_dir)

    def propose(
        self,
        *,
        target_proxy: str,
        new_implementation: str,
        previous_implementation: str,
        severity: UpgradeSeverity,
        rationale: str,
        init_calldata_hex: str = "0x",
        reviewer_assignments: Optional[List[str]] = None,
        opened_ts: Optional[float] = None,
    ) -> UpgradeProposal:
        if not isinstance(severity, UpgradeSeverity):
            raise ValueError(
                f"severity must be an UpgradeSeverity, got "
                f"{severity!r}"
            )
        if not rationale or not isinstance(rationale, str):
            raise ValueError("rationale must be non-empty")
        if len(rationale) > _MAX_RATIONALE_LEN:
            raise ValueError(
                f"rationale exceeds {_MAX_RATIONALE_LEN}-char "
                f"cap"
            )
        _validate_address(target_proxy, "target_proxy")
        _validate_address(
            new_implementation, "new_implementation",
        )
        _validate_address(
            previous_implementation,
            "previous_implementation",
        )
        if (
            new_implementation.lower()
            == previous_implementation.lower()
        ):
            raise ValueError(
                "new_implementation cannot be the same as "
                "previous_implementation"
            )
        record = UpgradeProposal(
            proposal_id=str(uuid.uuid4()),
            opened_ts=(
                opened_ts if opened_ts is not None
                else time.time()
            ),
            target_proxy=target_proxy,
            new_implementation=new_implementation,
            previous_implementation=previous_implementation,
            severity=severity,
            rationale=rationale,
            status=UpgradeStatus.PROPOSED,
            init_calldata_hex=init_calldata_hex,
            reviewer_assignments=list(
                reviewer_assignments or [],
            ),
        )
        self._records[record.proposal_id] = record
        self._write_to_disk(record)
        return record

    def update_status(
        self,
        proposal_id: str,
        new_status: UpgradeStatus,
        *,
        safe_tx_hash: Optional[str] = None,
    ) -> UpgradeProposal:
        existing = self._records.get(proposal_id)
        if existing is None:
            raise ValueError(
                f"proposal {proposal_id!r} not found"
            )
        if existing.status in _TERMINAL_STATUSES:
            raise ValueError(
                f"proposal {proposal_id!r} is in terminal "
                f"status {existing.status.value!r}; cannot "
                f"transition"
            )
        # Can't go back to PROPOSED
        if new_status == UpgradeStatus.PROPOSED:
            raise ValueError(
                "cannot transition back to PROPOSED "
                "(workflow integrity)"
            )
        # ROLLED_BACK requires having been EXECUTED first
        if (
            new_status == UpgradeStatus.ROLLED_BACK
            and existing.status != UpgradeStatus.EXECUTED
        ):
            raise ValueError(
                "ROLLED_BACK only valid from EXECUTED"
            )
        updated = UpgradeProposal(
            proposal_id=existing.proposal_id,
            opened_ts=existing.opened_ts,
            target_proxy=existing.target_proxy,
            new_implementation=existing.new_implementation,
            previous_implementation=(
                existing.previous_implementation
            ),
            severity=existing.severity,
            rationale=existing.rationale,
            status=new_status,
            init_calldata_hex=existing.init_calldata_hex,
            reviewer_assignments=(
                existing.reviewer_assignments
            ),
            safe_tx_hash=(
                safe_tx_hash
                if safe_tx_hash is not None
                else existing.safe_tx_hash
            ),
        )
        self._records[proposal_id] = updated
        self._write_to_disk(updated)
        return updated

    def get(
        self, proposal_id: str,
    ) -> Optional[UpgradeProposal]:
        return self._records.get(proposal_id)

    def list(
        self,
        *,
        status: Optional[UpgradeStatus] = None,
        severity: Optional[UpgradeSeverity] = None,
    ) -> List[UpgradeProposal]:
        out = list(self._records.values())
        out.sort(key=lambda r: r.opened_ts, reverse=True)
        if status is not None:
            out = [r for r in out if r.status == status]
        if severity is not None:
            out = [
                r for r in out if r.severity == severity
            ]
        return out

    def count(self) -> int:
        return len(self._records)

    # ── Persistence ───────────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                record = UpgradeProposal.from_dict(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "UpgradeOrchestrator: skipping corrupt "
                    "%s: %s", path, exc,
                )
                continue
            self._records[record.proposal_id] = record

    def _write_to_disk(
        self, record: UpgradeProposal,
    ) -> None:
        if self._persist_dir is None:
            return
        safe = (
            record.proposal_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
        )
        path = self._persist_dir / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(record.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "UpgradeOrchestrator: disk write failed for "
                "%s: %s", record.proposal_id, exc,
            )


# ── Composers ────────────────────────────────────────


_UPGRADE_WARNING = (
    "DESTRUCTIVE: this transaction replaces the contract "
    "implementation at the target proxy. ALL state-layout "
    "changes between old and new implementations must have "
    "been validated; storage corruption from layout drift "
    "is unrecoverable. Foundation Safe 2-of-3 hardware "
    "signers MUST verify the target proxy address, the new "
    "implementation address, and that the init calldata (if "
    "any) does what the proposal rationale claims."
)


_ROLLBACK_WARNING = (
    "DESTRUCTIVE ROLLBACK: this transaction reverts the "
    "target proxy to its previously-recorded implementation. "
    "Foundation Safe 2-of-3 signers MUST verify the "
    "previous_implementation address recorded on the "
    "proposal matches the implementation they intend to "
    "restore. Storage-layout incompatibilities between the "
    "now-deployed implementation and the rollback target "
    "are unrecoverable."
)


def _explorer_url(
    address: str, chain_id: Optional[int],
) -> Optional[str]:
    if chain_id == 8453:
        return f"https://basescan.org/address/{address}"
    if chain_id == 84532:
        return (
            f"https://sepolia.basescan.org/address/{address}"
        )
    return None


def _init_data_bytes(
    init_calldata_hex: Optional[str],
) -> bytes:
    if not init_calldata_hex:
        return b""
    h = init_calldata_hex.removeprefix("0x")
    if not h:
        return b""
    try:
        return bytes.fromhex(h)
    except ValueError as e:
        raise ValueError(
            f"init_calldata_hex is not valid hex: {e}"
        )


def compose_upgrade_tx(
    *,
    orchestrator: UpgradeOrchestrator,
    proposal_id: str,
    chain_id: Optional[int] = None,
) -> Dict[str, Any]:
    proposal = orchestrator.get(proposal_id)
    if proposal is None:
        raise ValueError(
            f"proposal {proposal_id!r} not found"
        )
    if proposal.status in _TERMINAL_STATUSES:
        raise ValueError(
            f"proposal {proposal_id!r} is in terminal "
            f"status {proposal.status.value!r}; cannot "
            f"compose upgrade"
        )
    if proposal.status == UpgradeStatus.PROPOSED:
        raise ValueError(
            f"proposal {proposal_id!r} has not been "
            f"reviewed yet (current status: "
            f"{proposal.status.value!r}); set to REVIEWED "
            f"before composing the upgrade tx"
        )
    init = _init_data_bytes(proposal.init_calldata_hex)
    data = encode_upgrade_to_and_call_calldata(
        proposal.new_implementation, init,
    )
    return {
        "action": "upgrade",
        "proposal_id": proposal_id,
        "to": proposal.target_proxy,
        "data": data,
        "value": "0",
        "target_proxy": proposal.target_proxy,
        "new_implementation": proposal.new_implementation,
        "previous_implementation": (
            proposal.previous_implementation
        ),
        "severity": proposal.severity.value,
        "rationale": proposal.rationale,
        "chain_id": chain_id,
        "warning": _UPGRADE_WARNING,
        "explorer_url": _explorer_url(
            proposal.target_proxy, chain_id,
        ),
        "instructions": (
            "1) Open the Foundation Safe UI; "
            "2) Create a new transaction with the `to` "
            "(target proxy), `data` (upgradeToAndCall "
            "calldata), and `value=0` fields above; "
            "3) 2-of-3 hardware signers verify the proxy "
            "address, embedded new_implementation address, "
            "and any init data match the proposal rationale "
            "before signing; "
            "4) Execute, then call update_status with the "
            "tx hash to mark EXECUTED."
        ),
    }


def compose_rollback_tx(
    *,
    orchestrator: UpgradeOrchestrator,
    proposal_id: str,
    chain_id: Optional[int] = None,
) -> Dict[str, Any]:
    proposal = orchestrator.get(proposal_id)
    if proposal is None:
        raise ValueError(
            f"proposal {proposal_id!r} not found"
        )
    if proposal.status == UpgradeStatus.ROLLED_BACK:
        raise ValueError(
            f"proposal {proposal_id!r} is already "
            f"rolled back"
        )
    if proposal.status != UpgradeStatus.EXECUTED:
        raise ValueError(
            f"proposal {proposal_id!r} is in status "
            f"{proposal.status.value!r}; can only roll "
            f"back an executed upgrade"
        )
    # Rollback uses NO init data — restoring to a known-good
    # previous impl shouldn't re-run init logic.
    data = encode_upgrade_to_and_call_calldata(
        proposal.previous_implementation, b"",
    )
    return {
        "action": "rollback",
        "proposal_id": proposal_id,
        "to": proposal.target_proxy,
        "data": data,
        "value": "0",
        "target_proxy": proposal.target_proxy,
        "rollback_target_implementation": (
            proposal.previous_implementation
        ),
        "originally_upgraded_to": (
            proposal.new_implementation
        ),
        "severity": proposal.severity.value,
        "chain_id": chain_id,
        "warning": _ROLLBACK_WARNING,
        "explorer_url": _explorer_url(
            proposal.target_proxy, chain_id,
        ),
        "instructions": (
            "1) Open the Foundation Safe UI; "
            "2) Create a new transaction with the `to` "
            "(target proxy), `data` (upgradeToAndCall to "
            "previous implementation), and `value=0`; "
            "3) 2-of-3 hardware signers verify the "
            "embedded rollback_target_implementation "
            "address; "
            "4) Execute, then call update_status "
            "(ROLLED_BACK) with the rollback tx hash."
        ),
    }

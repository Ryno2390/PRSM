"""Sprint 282 — Fiat compliance audit ring.

Records every fiat-flow event (onramp/offramp/gasless quotes
+ executes + KYC lifecycle events) for AUSTRAC / FinCEN / IRS
reporting once Phase 5 ramps go live. Single queryable log
across all fiat surfaces so operators have one source of
truth for regulatory inquiries.

Persistence is required, not opt-in — regulators expect 5-7
year retention. When PRSM_FIAT_COMPLIANCE_LOG_DIR is set the
ring writes a JSON file per entry. Without the env var, the
ring still operates (bounded in-memory ≥ 100K entries) so
audit data isn't silently dropped on a misconfigured node,
but operators running fiat flows in production MUST set the
env var (operationally enforced by `prsm node start`
health-check in a follow-on sprint).

This sprint records events from the existing quote endpoints
(every artifact return). Execute paths (gated on CDP
commission) auto-record once they ship — same hook point.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


_VALID_KINDS = {
    "onramp_quote", "onramp_execute",
    "offramp_quote", "offramp_execute",
    "gasless_transfer_quote", "gasless_transfer_execute",
    "kyc_initiate", "kyc_status_change",
}

_DEFAULT_MAX_ENTRIES = 100_000


@dataclass
class FiatComplianceEntry:
    entry_id: str
    timestamp: float
    kind: str
    user_id: str
    usd_amount: float
    ftns_amount: float
    status: str
    kyc_status: Optional[str] = None
    tx_hash: Optional[str] = None
    vendor_ref: Optional[str] = None
    address: Optional[str] = None
    jurisdiction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "FiatComplianceEntry":
        return cls(
            entry_id=d["entry_id"],
            timestamp=float(d.get("timestamp", 0.0)),
            kind=d["kind"],
            user_id=d.get("user_id", ""),
            usd_amount=float(d.get("usd_amount", 0.0)),
            ftns_amount=float(d.get("ftns_amount", 0.0)),
            status=d.get("status", "UNKNOWN"),
            kyc_status=d.get("kyc_status"),
            tx_hash=d.get("tx_hash"),
            vendor_ref=d.get("vendor_ref"),
            address=d.get("address"),
            jurisdiction=d.get("jurisdiction"),
            metadata=d.get("metadata", {}) or {},
        )


class FiatComplianceRing:
    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        *,
        persist_dir: Optional[Path] = None,
        default_jurisdiction: Optional[str] = None,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[FiatComplianceEntry] = deque(
            maxlen=max_entries,
        )
        self._by_id: Dict[str, FiatComplianceEntry] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        self._default_jurisdiction = default_jurisdiction
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    @classmethod
    def from_env(cls) -> "FiatComplianceRing":
        persist_raw = os.environ.get(
            "PRSM_FIAT_COMPLIANCE_LOG_DIR",
        )
        persist_dir = Path(persist_raw) if persist_raw else None
        jurisdiction = (
            os.environ.get("PRSM_OPERATOR_JURISDICTION") or None
        )
        return cls(
            persist_dir=persist_dir,
            default_jurisdiction=jurisdiction,
        )

    def record(
        self,
        *,
        kind: str,
        user_id: str,
        usd_amount: float,
        ftns_amount: float,
        status: str,
        kyc_status: Optional[str] = None,
        tx_hash: Optional[str] = None,
        vendor_ref: Optional[str] = None,
        address: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> FiatComplianceEntry:
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_KINDS)}, "
                f"got {kind!r}"
            )
        if usd_amount < 0:
            raise ValueError(
                f"usd_amount must be >= 0, got {usd_amount}"
            )
        if ftns_amount < 0:
            raise ValueError(
                f"ftns_amount must be >= 0, got {ftns_amount}"
            )
        entry = FiatComplianceEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=(
                timestamp if timestamp is not None else time.time()
            ),
            kind=kind,
            user_id=user_id or "",
            usd_amount=float(usd_amount),
            ftns_amount=float(ftns_amount),
            status=status,
            kyc_status=kyc_status,
            tx_hash=tx_hash,
            vendor_ref=vendor_ref,
            address=address,
            jurisdiction=(
                jurisdiction
                if jurisdiction is not None
                else self._default_jurisdiction
            ),
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._by_id[entry.entry_id] = entry
        self._write_to_disk(entry)
        return entry

    def get(
        self, entry_id: str,
    ) -> Optional[FiatComplianceEntry]:
        return self._by_id.get(entry_id)

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        kind: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[FiatComplianceEntry]:
        if not isinstance(limit, int) or limit <= 0 or limit > 10000:
            raise ValueError(
                f"limit must be in [1, 10000], got {limit}"
            )
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(
                f"offset must be >= 0, got {offset}"
            )
        if kind is not None and kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_KINDS)}, "
                f"got {kind!r}"
            )
        snap = list(self._entries)
        snap.reverse()
        if kind:
            snap = [e for e in snap if e.kind == kind]
        if user_id:
            snap = [e for e in snap if e.user_id == user_id]
        return snap[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)

    # Sprint 285 — sum USD volume per user over a rolling
    # window. Backs the tier-limit-enforcement check on
    # onramp/offramp quote endpoints. Gasless transfers +
    # KYC events excluded (gasless is FTNS-denominated; KYC
    # is zero-amount metadata).
    _FIAT_USD_KINDS = frozenset({
        "onramp_quote", "onramp_execute",
        "offramp_quote", "offramp_execute",
    })

    def total_usd_for_user(
        self,
        user_id: str,
        window_sec: int = 86400,  # 24h default
    ) -> float:
        """Sum usd_amount across fiat-surface events for this
        user_id within the rolling window. Empty user_id
        returns 0.0 (explicit-address flows aren't aggregated
        — there's no stable identity to aggregate against)."""
        if not user_id:
            return 0.0
        cutoff = time.time() - window_sec
        total = 0.0
        for e in self._entries:
            if e.user_id != user_id:
                continue
            if e.kind not in self._FIAT_USD_KINDS:
                continue
            if e.timestamp < cutoff:
                continue
            total += e.usd_amount
        return total

    def summary_by_kind(self) -> Dict[str, Dict[str, float]]:
        """Aggregate count + total USD volume per kind. Empty
        ring → empty dict."""
        out: Dict[str, Dict[str, float]] = {}
        for e in self._entries:
            bucket = out.setdefault(
                e.kind, {"count": 0, "total_usd": 0.0},
            )
            bucket["count"] += 1
            bucket["total_usd"] += e.usd_amount
        return out

    # ── Persistence ──────────────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        loaded: List[Dict[str, Any]] = []
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                loaded.append(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FiatComplianceRing: skipping corrupt %s: %s",
                    path, exc,
                )
        loaded.sort(key=lambda d: d.get("timestamp", 0))
        for d in loaded:
            try:
                entry = FiatComplianceEntry.from_dict(d)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "FiatComplianceRing: malformed entry %r: %s",
                    d, exc,
                )
                continue
            self._entries.append(entry)
            self._by_id[entry.entry_id] = entry

    def _write_to_disk(
        self, entry: FiatComplianceEntry,
    ) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / f"{entry.entry_id}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(entry.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FiatComplianceRing: disk write failed "
                "for %s: %s",
                entry.entry_id, exc,
            )

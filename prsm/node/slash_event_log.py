"""In-memory ring buffer for on-chain slash events.

Production-debugging feature: each on-chain
ProofFailureSlashed / HeartbeatMissingSlashed event observed by
the StorageSlashingWatcher appends to a bounded ring buffer.
Operators verify slashing impact via GET /admin/slash-history.

v1 entry shape mirrors what watcher callbacks deliver — the
on-chain decoded events do not carry block_number or tx_hash;
operators correlating to chain explorers grep by slash_id_hex.
A future enrichment pass on the watcher can add block/tx fields
without breaking this surface (additive only).

Bounded by deque(maxlen=...). v1 is in-process; v2 (sprint 91)
adds opt-in filesystem persistence via persist_dir constructor
arg. Operators set PRSM_SLASH_EVENT_LOG_DIR to enable.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


logger = logging.getLogger(__name__)


_VALID_KINDS = frozenset({
    "proof_failure_slashed",
    "heartbeat_missing_slashed",
})


@dataclass(frozen=True)
class SlashEventEntry:
    timestamp: float
    kind: str
    provider: str
    challenger: str
    slash_id_hex: str
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "provider": self.provider,
            "challenger": self.challenger,
            "slash_id": self.slash_id_hex,
            "extras": dict(self.extras),
        }


_DEFAULT_MAX_ENTRIES = 256


class SlashEventRing:
    """Bounded in-memory ring buffer of SlashEventEntry records."""

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, "
                f"got {max_entries!r}"
            )
        self._max_entries = max_entries
        self._entries: Deque[SlashEventEntry] = deque(maxlen=max_entries)
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Scan persist_dir + populate ring in timestamp order.
        Corrupt files logged + skipped (fail-soft)."""
        assert self._persist_dir is not None
        loaded: list = []
        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                # Reconstruct kind validation by routing through
                # the normal append path's gate.
                if data.get("kind") not in _VALID_KINDS:
                    raise ValueError(
                        f"invalid kind: {data.get('kind')!r}"
                    )
                loaded.append(data)
            except Exception as exc:
                logger.warning(
                    "SlashEventRing: skipping corrupt file %s: %s",
                    path, exc,
                )
        loaded.sort(key=lambda d: d.get("timestamp", 0))
        for d in loaded:
            self._entries.append(SlashEventEntry(
                timestamp=d["timestamp"],
                kind=d["kind"],
                provider=d["provider"],
                challenger=d["challenger"],
                slash_id_hex=d["slash_id"],
                extras=d.get("extras", {}),
            ))

    def _write_to_disk(self, entry: SlashEventEntry) -> None:
        if self._persist_dir is None:
            return
        # Filename: {timestamp_int}_{slash_id_first10}.json. Bounded
        # length, alphabetic-sortable, collision-resistant.
        slash_id_short = entry.slash_id_hex[:10]
        ts_int = int(entry.timestamp)
        path = (
            self._persist_dir / f"{ts_int}_{slash_id_short}.json"
        )
        try:
            path.write_text(json.dumps(entry.to_dict()))
        except Exception as exc:
            logger.warning(
                "SlashEventRing: disk write failed (fail-soft): %s",
                exc,
            )

    def append(
        self,
        *,
        kind: str,
        provider: str,
        challenger: str,
        slash_id: bytes,
        extras: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_KINDS)}, "
                f"got {kind!r}"
            )
        entry = SlashEventEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            kind=kind,
            provider=provider,
            challenger=challenger,
            slash_id_hex="0x" + slash_id.hex(),
            extras=extras or {},
        )
        self._entries.append(entry)
        self._write_to_disk(entry)

    def recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        provider: Optional[str] = None,
    ) -> List[SlashEventEntry]:
        if limit <= 0 or limit > 1000:
            raise ValueError(f"limit must be in [1, 1000], got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        snap = list(self._entries)
        snap.reverse()
        if provider:
            p_lower = provider.lower()
            snap = [e for e in snap if e.provider.lower() == p_lower]
        return snap[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)
